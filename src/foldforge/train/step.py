"""One eager training micro step of a FoldForge AlphaFold3-family model (Protenix v1): single-stage (full) finetuning -- every parameter
trains, the confidence head jointly with the rest.

Order (MiniWorld recipe): trunk ``passes`` times -- every pass but the last under ``no_grad`` (the previous recycles are detached) -- then
``num_augment`` EDM noise draws of the ground truth (random rotation / translation, one sigma per sample), the diffusion head on the noisy
coordinates, the diffusion (+ bond) loss, the distogram loss on the trunk pair, and the backward of that sum (diffusion head + distogram head +
trunk). The confidence head then reads a structure from a no-grad mini-rollout (``rollout_steps`` steps, one sample) with the trunk single /
pair / target features DETACHED (Protenix ``ConfidenceHead(stop_gradient=True)``, confidence.py:178-181; MiniWorld
``benchmarks/protenix_finetune/step.py:219``): the confidence loss trains the confidence head only, the trunk and the diffusion head learn from
the diffusion and distogram losses, so the two backwards are independent and their gradients simply add up in ``param.grad``.

``cfg.rollout`` places the mini-rollout: ``after`` the trunk backward (default), ``before`` the diffusion forward, or on a second CUDA stream
(``side``, MiniWorld ``--side_conf``) overlapped with the diffusion-train forward / backward; the confidence head runs after the join. Activations run in bf16 under ``activation_autocast`` (fp32 master
parameters); the losses are fp32.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from foldforge.modules.engine.configure import activation_autocast
from foldforge.train.loss import atom_weights, bond_loss, diffusion_mse, distogram_loss
from foldforge.train.loss.confidence import confidence_loss, confidence_targets
from foldforge.train.loss.diffusion import random_rotation_translation, sample_sigma
from foldforge.train.profile import PhaseTimer

if TYPE_CHECKING:
    from foldforge.train.config import TrainConfig
    from foldforge.train.data import TrainBatch


def run_trunk_train(
    model, batch_data, passes: int, timer: PhaseTimer | None = None
) -> dict[str, torch.Tensor]:
    """``AlphaFold3._run_trunk`` with gradient on the last pass only (earlier recycles are detached) and no MC dropout."""
    timer = timer or PhaseTimer(False)
    target_feat, structure_feat = model.create_target_feat_embedding(batch_data)
    embeddings = {
        "pair": model._ssm_state(batch_data.num_res, target_feat.device),
        "single": torch.zeros(
            [batch_data.num_res, model.evoformer_seq_channel],
            dtype=torch.float32,
            device=target_feat.device,
        ),
        "target_feat": target_feat,
        "structure_target_feat": structure_feat,
    }
    presample = (
        getattr(model.evoformer, "foldforge_msa_presampled", False)
        and model.evoformer.msa_stack_num_layer
    )
    for index in range(passes):
        last = index == passes - 1
        pass_batch = (
            replace(batch_data, msa=model.evoformer.sample_msa(batch_data.msa))
            if presample
            else batch_data
        )
        with timer.phase(f"trunk_fwd_r{index}"), nullcontext() if last else torch.no_grad():
            embeddings = model.evoformer(
                batch=pass_batch,
                prev=embeddings,
                target_feat=target_feat,
                first_pass=index == 0,
                last_pass=last,
                recycle_dropout=0.0,
            )
            if model.reference_precision:
                embeddings["pair"] = embeddings["pair"].float()
                embeddings["single"] = embeddings["single"].float()
    return embeddings


def mini_rollout(
    model, features, batch_data, detached: dict[str, torch.Tensor], steps: int
) -> torch.Tensor:
    """No-grad structure of the (current) diffusion head from the detached trunk outputs: ``[res, atoms, 3]``."""
    head = model.diffusion_head
    saved = (model.num_samples, model.diffusion_steps)
    head.eval()
    try:
        with torch.no_grad():
            rollout = model._prepare_diffusion_inputs(features, batch_data, detached)
            model.num_samples, model.diffusion_steps = 1, steps
            return model._sample_diffusion(rollout.batch, rollout.embeddings)[
                "atom_positions"
            ][0]
    finally:
        model.num_samples, model.diffusion_steps = saved
        head._fold = None  # the rollout's fold-scoped conditioning
        head.train()


def confidence_step(
    model,
    features: dict[str, torch.Tensor],
    batch_data,
    detached: dict[str, torch.Tensor],
    x_mini: torch.Tensor,
    x0: torch.Tensor,
    atom_mask: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Confidence head (trunk inputs detached) on the rollout structure ``x_mini`` -> pLDDT / PDE / PAE cross entropies against the targets of
    ``x_mini`` and the ground truth ``x0`` ``[res, atoms, 3]``."""
    tf = batch_data.token_features
    head_batch = model._head_batch(batch_data)
    gather = head_batch.pseudo_beta_info.token_atoms_to_pseudo_beta
    outputs = model.confidence_head(
        dense_atom_positions=x_mini.detach(),
        embeddings=detached,
        seq_mask=tf.mask,
        token_atoms_to_pseudo_beta=gather,
        asym_id=tf.asym_id,
        batch=head_batch,
    )
    targets = confidence_targets(
        x_mini,
        x0,
        atom_mask,
        gather,
        tf.mask,
        tf.aatype,
        tf.is_dna.bool() | tf.is_rna.bool(),
        tf.is_ligand,
        model.confidence_head,
    )
    w = cfg.loss
    return confidence_loss(
        outputs,
        targets,
        plddt_loss=w.plddt_loss,
        pde_loss=w.pde_loss,
        pae_loss=w.pae_loss,
    )


def micro_step(
    model,
    example: TrainBatch,
    cfg: TrainConfig,
    passes: int,
    *,
    scale: float = 1.0,
    timer: PhaseTimer | None = None,
) -> dict[str, float]:
    """Forward + loss + backward of one example; gradients accumulate into ``param.grad`` (scaled by ``scale``). Returns the loss parts.

    The heads read LEAF copies of the trunk outputs, so the backward runs in independent pieces whose gradients add up: the diffusion head
    (``augment_chunk`` samples at a time -- a smaller chunk holds fewer diffusion activations), the distogram head, then ONE backward through the
    trunk with the accumulated gradient of the leaves, then the confidence path. ``cfg.rollout`` places the mini-rollout: ``before`` (right after
    the trunk forward: its transient memory stacks on the saved trunk activations), ``side`` (on a second CUDA stream, overlapped with the
    diffusion-train forward / backward) or ``after`` (after the trunk backward, when the activations are freed)."""
    timer = timer or PhaseTimer(False)
    device = example.atom_positions.device
    sigma_data = cfg.noise.sigma_data
    n = cfg.num_augment
    chunk = cfg.augment_chunk if 0 < cfg.augment_chunk < n else n
    only = cfg.only  # "" = the whole step; "trunk" | "diffusion" | "rollout" | "confidence" = one part alone (memory / time measurements)
    mode = cfg.rollout if cfg.confidence and not only else "none"
    side_stream = (
        torch.cuda.Stream(device) if mode == "side" and device.type == "cuda" else None
    )
    if mode == "side" and side_stream is None:
        mode = "after"
    box: dict[str, torch.Tensor] = {}
    out: dict[str, float] = {"loss": 0.0, "diffusion": 0.0, "bond": 0.0, "distogram": 0.0}

    with activation_autocast(model):
        features = {
            k: (v.float() if v.dtype == torch.float64 else v)
            for k, v in example.features.items()
        }
        batch_data = model._prepare_batch(features)
        with torch.no_grad() if only in ("diffusion", "rollout", "confidence") else nullcontext():
            embeddings = run_trunk_train(model, batch_data, passes, timer)
        detached = {
            k: (v.detach() if torch.is_tensor(v) else v) for k, v in embeddings.items()
        }
        leaves = {
            k: v.detach().requires_grad_(True)
            for k, v in embeddings.items()
            if torch.is_tensor(v) and v.requires_grad
        }
        head_inputs = {**embeddings, **leaves}
        if only == "trunk":  # trunk forward + backward alone: a dummy loss on its outputs stands for the heads' gradient
            dummy = sum(v.float().pow(2).mean() for v in embeddings.values() if torch.is_tensor(v) and v.requires_grad)
            with timer.phase("trunk_bwd"):
                (dummy * 1e-6 * scale).backward()
            out["loss"] = float(dummy.detach())
            out["passes"] = passes
            return out
        if only in ("rollout", "confidence"):
            with timer.phase("mini_rollout"):
                box["x_mini"] = mini_rollout(model, features, batch_data, detached, cfg.rollout_steps)
            if only == "rollout":
                out["passes"] = passes
                return out
            x0 = example.atom_positions.float()
            amask = (
                example.atom_mask
                if example.atom_mask is not None
                else batch_data.predicted_structure_info.atom_mask
            ).bool()
            with timer.phase("confidence_fwd"):
                l_conf, parts = confidence_step(model, features, batch_data, detached, box["x_mini"], x0, amask, cfg)
            with timer.phase("confidence_bwd"):
                (l_conf * scale).backward()
            out["loss"] = out["confidence"] = float(l_conf.detach())
            out.update({k: float(v.detach()) for k, v in parts.items()})
            out["passes"] = passes
            return out
        diffusion = model._prepare_diffusion_inputs(features, batch_data, head_inputs)
        if mode == "before":
            with timer.phase("mini_rollout"):
                box["x_mini"] = mini_rollout(
                    model, features, batch_data, detached, cfg.rollout_steps
                )
        elif mode == "side":
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream), timer.phase("mini_rollout"):
                box["x_mini"] = mini_rollout(
                    model, features, batch_data, detached, cfg.rollout_steps
                )

        head = model.diffusion_head
        atom_mask = (
            example.atom_mask
            if example.atom_mask is not None
            else batch_data.predicted_structure_info.atom_mask
        ).bool()
        x0 = example.atom_positions.float()
        tf = batch_data.token_features
        weights = atom_weights(
            tf.is_dna,
            tf.is_rna,
            tf.is_ligand,
            atom_mask,
            alpha_dna=cfg.loss.alpha_dna,
            alpha_rna=cfg.loss.alpha_rna,
            alpha_ligand=cfg.loss.alpha_ligand,
        )
        w = cfg.loss
        # diffusion head (+ bond): ``chunk`` augmentations at a time, one backward per chunk (the leaves collect the trunk-side gradient)
        for first in range(0, n, chunk):
            m = min(chunk, n - first)
            share = m / n
            with timer.phase("diffusion_train_fwd"):
                sigma = sample_sigma(cfg.noise, m, device)
                x0_aug = random_rotation_translation(
                    x0[None].expand(m, *x0.shape).contiguous(),
                    cfg.noise.translation_noise,
                )
                noise = torch.randn_like(x0_aug) * sigma.view(m, 1, 1, 1)
                x_noisy = (x0_aug + noise) * atom_mask[None, :, :, None]
                table = head.conditioning_table(
                    diffusion.batch, diffusion.embeddings, sigma
                )
                x_pred = head(
                    x_noisy,
                    sigma,
                    diffusion.batch,
                    diffusion.embeddings,
                    True,
                    single_cond=table,
                )
                head._fold = None  # the fold's conditioning holds the autograd graph of this chunk
                l_diffusion = diffusion_mse(
                    x_pred, x0_aug, sigma, atom_mask, weights, sigma_data
                )
                l_bond = bond_loss(
                    x_pred, x0_aug, sigma, example.bond_pairs, sigma_data
                )
                chunk_loss = (
                    w.diffusion_loss * l_diffusion + w.bond_loss * l_bond
                ) * share
            with timer.phase("diffusion_train_bwd"):
                (chunk_loss * scale).backward()
            out["diffusion"] += float(l_diffusion.detach()) * share
            out["bond"] += float(l_bond.detach()) * share
            out["loss"] += float(chunk_loss.detach())
            del x_pred, x_noisy, x0_aug, noise, table, chunk_loss, l_diffusion, l_bond
        with timer.phase("distogram_fwd_bwd"):
            logits = model.distogram_head(model._head_batch(batch_data), head_inputs)[
                "logits"
            ]
            l_disto = distogram_loss(
                logits,
                model.distogram_head.breaks,
                x0,
                atom_mask,
                batch_data.pseudo_beta_info.token_atoms_to_pseudo_beta,
                tf.mask,
            )
            (w.distogram_loss * l_disto * scale).backward()
        out["distogram"] = float(l_disto.detach())
        out["loss"] += w.distogram_loss * out["distogram"]

    # ONE backward through the trunk with the gradient the heads left on the leaves
    with timer.phase("trunk_bwd"):
        # A leaf no head sent a gradient to (``grad is None``) contributes nothing to the trunk backward. ``target_feat`` is expected to be one: the
        # diffusion head reads ``structure_target_feat`` where the family has one (diffusion_head.py: ``embeddings.get("structure_target_feat",
        # embeddings["target_feat"])``; Protenix v1 does), so no head consumes ``target_feat``. Any other missing leaf is named in the log.
        expected = {"target_feat"} if "structure_target_feat" in embeddings else set()
        missing = sorted(k for k in leaves if leaves[k].grad is None and k not in expected)
        if missing:
            logging.getLogger("foldforge.train").warning(
                "no gradient reached the trunk output(s): %s", ", ".join(missing)
            )
        used = [k for k in leaves if leaves[k].grad is not None]
        if used:
            torch.autograd.backward(
                [embeddings[k] for k in used], [leaves[k].grad for k in used]
            )
    del leaves, head_inputs, diffusion, embeddings

    if cfg.confidence and not only:
        with activation_autocast(model):
            if mode == "side":
                torch.cuda.current_stream(device).wait_stream(side_stream)
                box["x_mini"].record_stream(torch.cuda.current_stream(device))
            elif mode == "after":
                with timer.phase("mini_rollout"):
                    box["x_mini"] = mini_rollout(
                        model, features, batch_data, detached, cfg.rollout_steps
                    )
            with timer.phase("confidence_fwd"):
                l_conf, parts = confidence_step(
                    model,
                    features,
                    batch_data,
                    detached,
                    box["x_mini"],
                    x0,
                    atom_mask,
                    cfg,
                )
        with timer.phase("confidence_bwd"):
            (l_conf * scale).backward()
        out["loss"] += float(l_conf.detach())
        out["confidence"] = float(l_conf.detach())
        out.update({k: float(v.detach()) for k, v in parts.items()})
    out["passes"] = passes
    return out
