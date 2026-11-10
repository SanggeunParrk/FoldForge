"""One eager training micro step of a FoldForge AlphaFold3-family model (Protenix v1).

Order (MiniWorld recipe): trunk ``passes`` times -- every pass but the last under ``no_grad`` (the previous recycles are detached) -- then
``num_augment`` EDM noise draws of the ground truth (random rotation / translation, one sigma per sample), the diffusion head on the noisy
coordinates, the diffusion (+ bond) loss, and the distogram loss on the trunk pair. Activations run in bf16 under ``activation_autocast``
(fp32 master parameters); the losses are fp32.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from foldforge.modules.engine.configure import activation_autocast
from foldforge.train.loss import atom_weights, bond_loss, diffusion_mse, distogram_loss
from foldforge.train.loss.diffusion import random_rotation_translation, sample_sigma

if TYPE_CHECKING:
    from foldforge.train.config import TrainConfig
    from foldforge.train.data import TrainBatch


def run_trunk_train(model, batch_data, passes: int) -> dict[str, torch.Tensor]:
    """``AlphaFold3._run_trunk`` with gradient on the last pass only (earlier recycles are detached) and no MC dropout."""
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
        with nullcontext() if last else torch.no_grad():
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


def micro_step(
    model, example: TrainBatch, cfg: TrainConfig, passes: int, *, scale: float = 1.0
) -> dict[str, float]:
    """Forward + loss + backward of one example; gradients accumulate into ``param.grad`` (scaled by ``scale``). Returns the loss parts."""
    device = example.atom_positions.device
    sigma_data = cfg.noise.sigma_data
    with activation_autocast(model):
        features = {
            k: (v.float() if v.dtype == torch.float64 else v)
            for k, v in example.features.items()
        }
        batch_data = model._prepare_batch(features)
        embeddings = run_trunk_train(model, batch_data, passes)
        diffusion = model._prepare_diffusion_inputs(features, batch_data, embeddings)

        head = model.diffusion_head
        atom_mask = (
            example.atom_mask
            if example.atom_mask is not None
            else batch_data.predicted_structure_info.atom_mask
        )
        atom_mask = atom_mask.bool()
        x0 = example.atom_positions.float()
        n = cfg.num_augment
        sigma = sample_sigma(cfg.noise, n, device)
        x0_aug = random_rotation_translation(
            x0[None].expand(n, *x0.shape).contiguous(), cfg.noise.translation_noise
        )
        noise = torch.randn_like(x0_aug) * sigma.view(n, 1, 1, 1)
        x_noisy = (x0_aug + noise) * atom_mask[None, :, :, None]
        table = head.conditioning_table(
            diffusion.batch, diffusion.embeddings, sigma
        )  # [n, res, c]: one noise level per sample
        x_pred = head(
            x_noisy,
            sigma,
            diffusion.batch,
            diffusion.embeddings,
            True,
            single_cond=table,
        )
        head._fold = (
            None  # the fold's conditioning holds the autograd graph of this step
        )

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
        l_diffusion = diffusion_mse(
            x_pred, x0_aug, sigma, atom_mask, weights, sigma_data
        )
        l_bond = bond_loss(x_pred, x0_aug, sigma, example.bond_pairs, sigma_data)
        logits = model.distogram_head(model._head_batch(batch_data), embeddings)[
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
        w = cfg.loss
        total = (
            w.diffusion_loss * l_diffusion
            + w.bond_loss * l_bond
            + w.distogram_loss * l_disto
        )
    (total * scale).backward()
    return {
        "loss": float(total.detach()),
        "diffusion": float(l_diffusion.detach()),
        "bond": float(l_bond.detach()),
        "distogram": float(l_disto.detach()),
        "passes": passes,
    }
