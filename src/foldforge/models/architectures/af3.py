from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from foldforge.data.features import dense as features
from foldforge.data.features import dense_batch as feat_batch
from foldforge.diffusion.augmentation import (
    masked_dense_rigid_motion,
)
from foldforge.diffusion.edm.sampling import EulerSampler
from foldforge.diffusion.edm.schedules import PowerLawSchedule
from foldforge.models import sampler_graph
from foldforge.models.bucketing import MSA_SHAPES, ceiling, pad_axis
from foldforge.modules.dense import atom_cross_attention, diffusion_head, featurization
from foldforge.modules.dense.fused_template import FusedTemplateEmbedding
from foldforge.modules.dense.head import ConfidenceHead, DistogramHead
from foldforge.modules.dense.pair_init import (
    ChaiTokenEmbedder,
    ContactConditioning,
    SummedInputEmbedder,
    token_bond_types,
)
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
from foldforge.modules.dense.structural_tokens import StructuralTokenExpander
from foldforge.modules.dense.template import TemplateEmbedding
from foldforge.modules.engine import graph as engine_graph
from foldforge.utils.rigid_align import kabsch_rotation

# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


def weighted_rigid_align(
    mobile: torch.Tensor, reference: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Kabsch-align ``mobile`` onto ``reference`` over the dense atom layout.

    The last three axes are the layout -- token, atom-in-token, xyz -- and the
    weight covers exactly those. Any axis in front of them is a separate
    structure that gets its own rotation, so only the layout is flattened.
    Flattening everything would fold the samples into the atom list and solve
    one Kabsch problem across all of them, which is a different question and,
    for more than one sample, not even a well-shaped one.
    """
    w = weight[..., None].to(mobile.dtype)
    total = w.sum((-3, -2), keepdim=True)
    centre_m = (mobile * w).sum((-3, -2), keepdim=True) / total
    centre_r = (reference * w).sum((-3, -2), keepdim=True) / total
    lead = mobile.shape[:-3]
    a = (mobile - centre_m).reshape(*lead, -1, 3)
    b = (reference - centre_r).reshape(*lead, -1, 3)
    covariance = torch.einsum("...nc,...nd->...cd", a * w.reshape(-1, 1), b).float()
    # The optimal PROPER rotation (a reflection is never returned), by Horn's quaternion
    # method:
    # no SVD, whose host-side status check cannot be captured in a CUDA graph.
    rotation = kabsch_rotation(covariance).to(mobile.dtype)
    return (a @ rotation).reshape(mobile.shape) + centre_r


def _to_residue_layout(
    samples: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_res: int,
) -> dict[str, torch.Tensor]:
    """Bring a structural-token diffusion result back to the residue layout.

    Exact, not a choice: every residue atom sits in exactly one structural slot,
    and `residue_atom_gather` names it. Everything downstream -- the confidence
    head, the mmCIF writer, the whole output path -- reads residues, so this is
    where the structural token set ends.
    """
    gather = batch["structbook/residue_atom_gather"].to(torch.int64)[:num_res]
    valid = gather >= 0
    index = torch.where(valid, gather, torch.zeros_like(gather)).reshape(-1)
    out = {}
    for key, value in samples.items():
        flat = value.reshape(value.shape[0], -1, *value.shape[3:])
        taken = flat[:, index].reshape(value.shape[0], *gather.shape, *value.shape[3:])
        mask = valid.reshape(1, *gather.shape, *((1,) * (value.ndim - 3)))
        out[key] = torch.where(mask, taken, torch.zeros_like(taken))
    return out


@dataclass(frozen=True)
class DiffusionInputs:
    """Fold-local diffusion inputs and the optional return to residue coordinates.

    The residue trunk embeddings remain with the caller for the heads; only
    these inputs move onto structural tokens. No model parameters live here.
    """

    batch: feat_batch.Batch
    embeddings: dict[str, torch.Tensor]
    residue_layout: dict[str, torch.Tensor] | None = None
    num_res: int = 0

    def restore(self, samples: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Restore structural samples, or retain the existing residue tensors."""
        if self.residue_layout is None:
            return samples
        return _to_residue_layout(samples, self.residue_layout, self.num_res)


@dataclass(frozen=True)
class TrunkOutput:
    """Residue head inputs and prepared diffusion inputs for repeated sampling."""

    batch: feat_batch.Batch
    embeddings: dict[str, torch.Tensor]
    diffusion: DiffusionInputs


class Evoformer(nn.Module):
    """Represent evoformer."""

    def __init__(self, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()

        self.spec = spec
        self.msa_channel = spec.msa_channel
        self.msa_stack_num_layer = spec.msa_layers
        self.pairformer_num_layer = spec.trunk_layers
        self.num_msa = 1024

        self.seq_channel = spec.seq_channel
        self.pair_channel = spec.pair_channel
        self.c_target_feat = spec.target_feat_channel

        # AF3 initialises the pair from the raw target features; the families
        # that follow Algorithm 1 literally initialise it from the single
        # projection of those features, so the two read different widths.
        c_pair_init = (
            spec.seq_channel if spec.pair_init_from_single else self.c_target_feat
        )
        self.left_single = nn.Linear(c_pair_init, self.pair_channel, bias=False)
        self.right_single = nn.Linear(c_pair_init, self.pair_channel, bias=False)

        self.prev_embedding_layer_norm = nn.LayerNorm(self.pair_channel)
        self.prev_embedding = nn.Linear(
            self.pair_channel, self.pair_channel, bias=False
        )

        self.c_rel_feat = spec.relpos_channel
        self.position_activations = nn.Linear(
            self.c_rel_feat, self.pair_channel, bias=spec.relpos_bias
        )

        if not spec.no_bond_embedding:
            self.bond_embedding = nn.Linear(1, self.pair_channel, bias=False)
        if spec.bond_type_and_contact_init:
            self.token_bonds_type_embed = nn.Linear(7, self.pair_channel, bias=False)
            self.contact_conditioning = ContactConditioning(self.pair_channel)

        # A family with no template stack must not BUILD one: its checkpoint
        # carries no weights for it, and a template embedder left at random
        # init would add noise to the pair on every input that has a template.
        self.template_embedding = None
        if spec.template_layers:
            self.template_embedding = (
                TemplateEmbedding(spec)
                if spec.template == "af3"
                else FusedTemplateEmbedding(spec)
            )

        # A family that folds from a language model alone has no MSA stack, and
        # skipping the CALL rather than only the blocks is what keeps these two
        # projections out of a parameter tree that carries no weights for them.
        self.msa_stack = None
        if self.msa_stack_num_layer:
            self.msa_activations = nn.Linear(
                spec.msa_feat_channel, self.msa_channel, bias=spec.msa_activations_bias
            )
            self.extra_msa_target_feat = nn.Linear(
                self.c_target_feat, self.msa_channel, bias=False
            )
            self.msa_stack = nn.ModuleList(
                [
                    self._msa_block(spec)
                    for _ in range(self.msa_stack_num_layer)
                ]
            )

        self.single_activations = nn.Linear(
            self.c_target_feat, self.seq_channel, bias=False
        )

        self.prev_single_embedding_layer_norm = nn.LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False
        )

        self.trunk_pairformer = nn.ModuleList(
            [self._trunk_block(spec) for _ in range(self.pairformer_num_layer)]
        )

        # ESMFold2 recycles through a discretised diagonal SSM rather than an
        # addition. Same two modules as AF3's recycling, applied to the OTHER
        # operand, plus this per-channel decay.
        if spec.ssm_recycle:
            self.recycle_decay = nn.Parameter(torch.ones(self.pair_channel))

        # A readout projection and a short "coda" of pair blocks AFTER the
        # recycle loop. AF3 has no post-trunk pair stage, so nothing in the
        # parameter tree asks for these -- a tree diff compares what the graph
        # WANTS against what the checkpoint supplies, and a stage the graph
        # never builds is invisible to it. The reference found it by folding.
        self.parcae_readout = None
        if spec.coda_layers:
            self.parcae_readout = nn.Linear(
                self.pair_channel, self.pair_channel, bias=False
            )
            self.trunk_coda = nn.ModuleList(
                [self._pair_only_block(spec) for _ in range(spec.coda_layers)]
            )

        # Four pair-only blocks refining the language model's pair before it
        # joins the injection.
        self.lm_encoder = None
        if spec.lm_encoder_layers:
            self.lm_encoder = nn.ModuleList(
                [self._pair_only_block(spec) for _ in range(spec.lm_encoder_layers)]
            )

        # The family that folds on structural tokens expands the trunk's output
        # onto them and refines it with its own stack of the ordinary block.
        self.structural_token_expander = None
        if spec.structural_tokens:
            self.structural_token_expander = StructuralTokenExpander(
                c_single=self.seq_channel,
                c_pair=self.pair_channel,
                c_target_feat=self.c_target_feat,
            )
            self.structural_token_refiner = nn.ModuleList(
                [self._refiner_block(spec) for _ in range(spec.structural_refiner_layers)]
            )

    def _refiner_block(self, spec: DenseSpec) -> nn.Module:
        """One block of OpenDDE's structural-token refiner (the engine class; its single attention takes the expander's
        bias through ``extra_pair_bias``)."""
        return engine_graph.make_pairformer_block(
            spec,
            d_pair=self.pair_channel,
            d_single=self.seq_channel,
            qkv_dim=spec.pair_qkv_dim,
            transition_n=spec.structural_refiner_transition_factor,
            single_transition_n=spec.pairformer_transition_factor,
            n_head_single=spec.structural_refiner_heads,
        )

    def _trunk_block(self, spec: DenseSpec) -> nn.Module:
        """One trunk pairformer block: the engine class."""
        return engine_graph.make_pairformer_block(
            spec,
            d_pair=self.pair_channel,
            d_single=self.seq_channel if spec.trunk_single_track else 0,
            qkv_dim=spec.pair_qkv_dim,
            pair_attention=spec.pair_attention,
        )

    def _msa_block(self, spec: DenseSpec) -> nn.Module:
        """One MSA-stack block: engine parts."""
        return engine_graph.make_evoformer_block(spec, self.msa_channel, self.pair_channel, spec.pair_heads)

    def _pair_only_block(self, spec: DenseSpec) -> nn.Module:
        """Build one trunk block with neither a single track nor pair attentions."""
        return engine_graph.make_pairformer_block(
            spec, d_pair=self.pair_channel, qkv_dim=spec.pair_qkv_dim, pair_attention=spec.pair_attention
        )

    def _embed_lm_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add the language model's pair representation to the injection.

        The shim that turns ESM-C's hidden states into this tensor is a separate
        graph and hands its output in on the batch. What is left is the encoder,
        four pair-only blocks, and the dropout -- which this family keeps at
        INFERENCE and resamples every recycle pass, so it cannot move outside
        the loop.
        """
        lm = batch.lm_pair
        if lm is None:
            # Folding this family without its language model still produces a
            # structure, and a plausible-looking one -- which is why the gap
            # has to be an error rather than a quiet fallback.
            message = (
                "This family folds from a language model; none reached the "
                "trunk, and folding without it is a different model"
            )
            raise ValueError(message)
        lm = lm.to(dtype=pair_activations.dtype)
        rate = self.spec.lm_pair_dropout
        if rate:
            keep = torch.rand_like(lm) >= rate
            lm = lm * keep.to(lm.dtype) / (1.0 - rate)
        if self.lm_encoder is not None:
            for block in self.lm_encoder:
                lm = engine_graph.run_block(block, lm, pair_mask)
        return pair_activations + lm

    def expand_structural(
        self,
        embeddings: dict[str, Any],
        book: dict[str, torch.Tensor],
        batch: feat_batch.Batch,
        asym_id: torch.Tensor,
    ) -> dict[str, Any]:
        """Carry the residue trunk's output onto OpenDDE's structural tokens.

        The expander splits each residue's single and pair into its subtokens and
        states the relationships the pair cannot -- which subtokens share a
        parent, which are backbone twins, which back onto the next residue in the
        chain -- as a pair term and an attention bias. Its own refiner stack then
        lets those relationships propagate before the diffusion reads them.

        ``batch`` is the STRUCTURAL batch, so its mask counts structural tokens.
        """
        if self.structural_token_expander is None:
            message = "This family does not fold on structural tokens"
            raise RuntimeError(message)
        target_feat, single, pair, bias = self.structural_token_expander(
            embeddings["target_feat"],
            embeddings["single"],
            embeddings["pair"],
            book["parent_residue_idx"],
            book["subtoken_role_id"],
            asym_id,
            book["prev_parent_residue_idx"],
            book["next_parent_residue_idx"],
        )
        seq_mask = batch.token_features.mask
        pair_mask = (seq_mask[:, None] * seq_mask[None, :]).to(pair.dtype)
        for block in self.structural_token_refiner:
            pair, single = engine_graph.run_block(
                block, pair, pair_mask, single, seq_mask, extra_pair_bias=bias
            )
        return {
            **embeddings,
            "single": single,
            "pair": pair,
            "target_feat": target_feat,
            "structure_target_feat": target_feat,
            "structural_pair_attn_bias": bias,
        }

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        if self.spec.relpos == "chai1":
            rel_feat = featurization.chai_relative_encoding(
                batch.token_features, pair_activations.dtype
            )
        else:
            rel_feat = featurization.create_relative_encoding(
                batch.token_features,
                max_relative_idx,
                max_relative_chain,
                chain_bucket_on_same_chain=self.spec.chain_bucket_on_same_chain,
            ).to(dtype=pair_activations.dtype)

        return pair_activations + self.position_activations(rel_feat)

    def _seq_pair_embedding(
        self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Pair embedding from sequence."""
        left_single = self.left_single(target_feat)[:, None]
        right_single = self.right_single(target_feat)[None]
        pair_activations = left_single + right_single

        mask = token_features.mask

        pair_mask = (mask[:, None] * mask[None, :]).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _embed_bonds(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations."""
        # Construct contact matrix.
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens),
            dtype=pair_activations.dtype,
            device=pair_activations.device,
        )

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_polymer_ligand.dtype)[:, None]
        # If valid mask then it will be all 1's, so idxs should be unchanged.
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )

        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )

        gather_idxs = torch.concatenate(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
        )
        # A device scalar: a Python 1.0 becomes a CPU tensor whose host-to-device copy
        # cannot be captured in a CUDA graph.
        one = torch.ones((), dtype=contact_matrix.dtype, device=contact_matrix.device)
        contact_matrix[gather_idxs[:, 0], gather_idxs[:, 1]] = one
        if self.spec.symmetric_bonds:
            contact_matrix[gather_idxs[:, 1], gather_idxs[:, 0]] = one

        # Because all the padded index's are 0's.
        contact_matrix[0, 0].zero_()

        pair_activations = pair_activations + self.bond_embedding(
            contact_matrix[:, :, None]
        )
        if self.spec.bond_type_and_contact_init:
            # Both terms contribute on EVERY pair: bond order 0 and the unspecified
            # contact class are learned vectors, not zeros.
            bond_types = token_bond_types(batch, symmetric=self.spec.symmetric_bonds)
            pair_activations = pair_activations + self.token_bonds_type_embed(
                nn.functional.one_hot(bond_types, 7).to(pair_activations.dtype)
            )
            pair_activations = pair_activations + self.contact_conditioning(
                num_tokens, pair_activations
            )
        return pair_activations

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        if self.template_embedding is None:
            return pair_activations
        templates = batch.templates
        asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask,
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self,
        msa_batch: features.MSA,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
        token_features: features.TokenFeatures,
    ) -> torch.Tensor:
        """Process MSA and returns updated pair activations."""
        if self.msa_stack is None:
            # A family with no MSA stack never reaches here; the callers check
            # first, and this is the invariant rather than a fallback.
            message = "This family folds without an MSA stack"
            raise RuntimeError(message)
        dtype = pair_activations.dtype

        policy = self.spec.msa_subsample
        if policy == "keep_query":
            msa_batch = featurization.subsample_msa_keep_query(msa_batch, self.num_msa)
        else:
            if policy == "shuffle":
                msa_batch = featurization.shuffle_msa(msa_batch)
            elif policy != "ordered":
                message = f"unknown MSA subsampling policy {policy!r}"
                raise ValueError(message)
            msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)
        if getattr(self, "foldforge_msa_bucketing", False):
            extent = ceiling(msa_batch.rows.shape[0], MSA_SHAPES, "msa")
            msa_batch = replace(
                msa_batch,
                rows=pad_axis(msa_batch.rows, 0, extent),
                mask=pad_axis(msa_batch.mask, 0, extent),
                deletion_matrix=pad_axis(msa_batch.deletion_matrix, 0, extent),
            )

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(
            msa_batch,
            layout=self.spec.msa_feat_layout,
            is_ligand=token_features.is_ligand,
            asym_id=token_features.asym_id,
        ).to(dtype=dtype)
        if self.spec.msa_query_paired is not None:
            paired = torch.zeros_like(msa_feat[..., :1])
            paired[0] = self.spec.msa_query_paired
            msa_feat = torch.cat([msa_feat, paired], dim=-1)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]

        # Evoformer MSA stack.
        pair_input = pair_activations
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )

        if self.spec.msa_double_add:
            # The vendor's MSA module returns the updated pair and its caller adds
            # that to the pair again; the weights were fitted with both copies.
            pair_activations = pair_activations + pair_input
        return pair_activations

    def _pair_only_trunk(
        self,
        *,
        prev: dict[str, Any],
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        single_init: torch.Tensor,
        target_feat: torch.Tensor,
        first_pass: bool,
        last_pass: bool,
    ) -> dict[str, Any]:
        """Run the trunk of a family with no single track, given its injection.

        The single is still projected from the target features and still
        recycled -- the checkpoint carries both projections -- it is simply
        never updated, because no block here has a single track to update it.
        """
        # The recycle carries the PRE-CODA tensor: feeding the coda's output
        # back would recycle a differently scaled representation through a
        # projection on every pass.
        recycled = prev.get("pair_pre_coda", prev["pair"]).to(
            dtype=pair_activations.dtype
        )
        if (
            self.spec.recycle_from_initial
            and first_pass
            and not self.spec.ssm_random_init
        ):
            recycled = pair_activations
        # With `ssm_random_init` the first pass reads the recurrent state `forward` drew
        # (see `_ssm_state`).
        projected = self.prev_embedding(
            self.prev_embedding_layer_norm(pair_activations)
        )
        pair_activations = (
            self.recycle_decay.to(projected.dtype) * recycled + projected
            if self.spec.ssm_recycle
            else recycled + projected
        )

        single_activations = single_init + self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev["single"].to(single_init.dtype))
        )
        for block in self.trunk_pairformer:
            pair_activations = engine_graph.run_block(block, pair_activations, pair_mask)

        pair_pre_coda = pair_activations
        pair_activations = self._finalize_pair_only(
            pair_activations, pair_mask, last_pass=last_pass
        )

        return {
            "single": single_activations,
            "pair": pair_activations,
            "pair_pre_coda": pair_pre_coda,
            "target_feat": target_feat,
            "structure_target_feat": prev["structure_target_feat"],
        }

    def _finalize_pair_only(
        self, pair: torch.Tensor, mask: torch.Tensor, *, last_pass: bool
    ) -> torch.Tensor:
        """Apply readout/coda only after the final pair-only recurrence."""
        if self.parcae_readout is not None and last_pass:
            pair = self.parcae_readout(pair)
            for block in self.trunk_coda:
                pair = engine_graph.run_block(block, pair, mask)
        return pair

    def _pair_only_injection(
        self,
        batch: feat_batch.Batch,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Build the full LM/MSA injection before the pair-only recurrence reads it."""
        pair = self._relative_encoding(batch, pair)
        if not self.spec.no_bond_embedding:
            pair = self._embed_bonds(batch=batch, pair_activations=pair)
        if self.msa_stack_num_layer:
            pair = self._embed_process_msa(
                msa_batch=batch.msa,
                pair_activations=pair,
                pair_mask=pair_mask,
                token_features=batch.token_features,
                target_feat=target_feat,
            )
        return self._embed_lm_pair(batch, pair, pair_mask)

    def _recycle_pair(self, recycled: torch.Tensor, dropout: float) -> torch.Tensor:
        """Project the recycled pair, under the fold's MC dropout if it has one."""
        term = self.prev_embedding(self.prev_embedding_layer_norm(recycled))
        if dropout:
            term = nn.functional.dropout(term, p=dropout, training=True)
        return term

    def forward(
        self,
        batch: feat_batch.Batch,
        prev: dict[str, Any],
        target_feat: torch.Tensor,
        *,
        first_pass: bool = False,
        last_pass: bool = True,
        recycle_dropout: float = 0.0,
    ) -> dict[str, Any]:
        """Compute the module output."""
        # The single projection is needed first where the pair is built from it.
        single_init = self.single_activations(target_feat)
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features,
            single_init if self.spec.pair_init_from_single else target_feat,
        )

        pair_init = None
        if self.spec.diffusion_pair_init_cond:
            # Every term here is an addition, so applying the relative encoding
            # first and capturing before the recycle add gives the same
            # activation and a clean z_init to condition the diffusion on. The
            # order is reversed only for the families that need it: summation is
            # not associative in floating point, and the rest stay bit-identical.
            pair_activations = self._relative_encoding(batch, pair_activations)
            pair_init = pair_activations

        if not self.spec.trunk_single_track:
            # ESMFold2 assembles the ENTIRE injection -- relative encoding,
            # bonds, the MSA encoder and the language model's pair -- BEFORE the
            # recurrence reads it. For an addition that would be the same
            # function; for the SSM it is not, because the injection is what
            # gets normalised and projected. Running the MSA encoder afterwards
            # would feed it the recycled term instead of z_init, and that term
            # would then be re-injected every pass rather than decaying.
            pair_activations = self._pair_only_injection(
                batch, pair_activations, pair_mask, target_feat
            )
            return self._pair_only_trunk(
                prev=prev,
                pair_activations=pair_activations,
                pair_mask=pair_mask,
                single_init=single_init,
                target_feat=target_feat,
                first_pass=first_pass,
                last_pass=last_pass,
            )

        return self._single_pair_trunk(
            batch,
            prev,
            target_feat,
            single_init,
            pair_activations,
            pair_mask,
            pair_init,
            first_pass=first_pass,
            recycle_dropout=recycle_dropout,
        )

    def _single_pair_trunk(
        self,
        batch: feat_batch.Batch,
        prev: dict[str, Any],
        target_feat: torch.Tensor,
        single_init: torch.Tensor,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        pair_init: torch.Tensor | None,
        *,
        first_pass: bool,
        recycle_dropout: float,
    ) -> dict[str, Any]:
        """Run the single/pair schedule, retaining its recycle/addition order."""
        recycled = prev["pair"]
        if self.spec.recycle_from_initial and first_pass:
            # The carry starts at the INITIAL representations rather than zeros,
            # so pass one already adds recycle_proj(norm(z_init)).
            recycled = pair_init if pair_init is not None else pair_activations
        pair_activations = pair_activations + self._recycle_pair(
            recycled.to(pair_activations.dtype), recycle_dropout
        )

        if pair_init is None:
            pair_activations = self._relative_encoding(batch, pair_activations)

        if not self.spec.no_bond_embedding:
            # A family whose token-pair stream carries no bond feature has no
            # weight here; running it would add a random-init term on every
            # input that has an intra-ligand bond.
            pair_activations = self._embed_bonds(
                batch=batch, pair_activations=pair_activations
            )

        pair_activations = self._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
        )

        single_activations = single_init
        recycled_single = prev["single"]
        if self.spec.recycle_from_initial and first_pass:
            recycled_single = single_activations
        single_activations = single_activations + self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(
                recycled_single.to(single_activations.dtype)
            )
        )

        pair_activations = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            token_features=batch.token_features,
            # The recycled single is the one AFTER its recycle add -- the same
            # tensor the pairformer starts from -- not the raw carry.
            target_feat=(
                single_activations.to(target_feat.dtype)
                if self.spec.msa_single_from_recycle
                else target_feat
            ),
        )

        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = engine_graph.run_block(
                pairformer_b,
                pair_activations,
                pair_mask,
                single_activations,
                batch.token_features.mask,
            )

        return {
            "single": single_activations,
            "pair": pair_activations,
            **({"pair_init": pair_init} if pair_init is not None else {}),
            "target_feat": target_feat,
            "structure_target_feat": prev["structure_target_feat"],
        }


class AlphaFold3(nn.Module):
    """Represent alpha fold3."""

    def __init__(
        self,
        num_recycles: int = 10,
        num_samples: int = 5,
        diffusion_steps: int = 200,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:
        super().__init__()

        if num_recycles < 1:
            message = "num_recycles must be positive (total trunk passes)"
            raise ValueError(message)
        self.spec = spec

        self.reference_precision = False
        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps

        self.evoformer_pair_channel = spec.pair_channel
        self.evoformer_seq_channel = spec.seq_channel

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder(
            spec=spec
        )

        self.evoformer = Evoformer(spec)

        self.diffusion_head = diffusion_head.DiffusionHead(spec)

        if spec.input_embedder == "summed":
            self.input_embedder = SummedInputEmbedder(spec.seq_channel)
        elif spec.input_embedder == "chai1":
            self.input_embedder = ChaiTokenEmbedder(spec.seq_channel)

        self.distogram_head = DistogramHead(
            c_pair=spec.pair_channel,
            num_bins=spec.distogram_bins,
            bias=spec.distogram_bias,
            hidden=spec.distogram_hidden,
            mean_symmetrised=spec.distogram_mean_symmetrised,
        )
        self.confidence_head = ConfidenceHead(
            c_single=spec.seq_channel,
            c_pair=spec.pair_channel,
            c_target_feat=spec.target_feat_channel,
            n_pairformer_layers=spec.confidence_layers,
            spec=spec,
        )

    @property
    def activation_dtype(self) -> torch.dtype:
        """The dtype the trunk's activations start in: bf16 for a ``master`` model (fp32 parameters, activations under
        :func:`~foldforge.modules.engine.configure.activation_autocast`), fp32 otherwise. Set by ``configure_model``; parameters are
        always fp32."""
        return getattr(self, "foldforge_activation_dtype", torch.float32)

    def create_target_feat_embedding(
        self, batch: feat_batch.Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the trunk and the structure target features.

        They are the same tensor unless the family trained a second projection
        for the diffusion module.
        """
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )

        dtype = self.activation_dtype
        if self.spec.input_embedder == "summed":
            summed = self.input_embedder(batch, enc.token_act).to(dtype)
            return summed, summed
        if self.spec.input_embedder == "chai1":
            trunk, structure = self.input_embedder(
                batch, enc.token_act, batch.token_features.lm_embeddings
            )
            return trunk.to(dtype), structure.to(dtype)
        both = torch.concatenate([target_feat, enc.token_act], dim=-1).to(dtype)
        return both, both

    def _ssm_state(self, num_res: int, device: torch.device) -> torch.Tensor:
        """Initialize the pair state read by the first trunk pass.

        The SSM recurrence starts from truncated-normal noise (std sqrt(2 / 5c),
        cut at 3 std); its decay carries this state through every pass. Draw it
        from the trunk seed outside the pass: ``trunc_normal_`` has a rejection
        loop that reads the device on the host and cannot be graph-captured.
        """
        shape = [num_res, num_res, self.evoformer_pair_channel]
        if not (self.spec.ssm_random_init and not self.spec.trunk_single_track):
            return torch.zeros(shape, device=device, dtype=torch.float32)
        std = (2.0 / (5.0 * self.evoformer_pair_channel)) ** 0.5
        return torch.nn.init.trunc_normal_(
            torch.empty(shape, device=device, dtype=torch.float32),
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Adapt the dense checkpoint layout to the shared EDM sampler."""
        mask = batch.predicted_structure_info.atom_mask
        sigmas = PowerLawSchedule(
            PowerLawSchedule.Config(
                sigma_max=self.spec.sigma_max,
                sigma_min=self.spec.sigma_min,
                rho=self.spec.rho,
                terminal_zero=False,
                include_minimum_before_zero=False,
            )
        )(self.diffusion_steps, device=mask.device)
        sampler_config = EulerSampler.Config(
            gamma_0=self.spec.gamma_0,
            gamma_min=self.spec.gamma_min,
            noise_scale=self.spec.noise_scale,
            step_scale=self.spec.step_scale,
        )
        if self.spec.churn_total is not None:
            # EDM's own parameterisation: one constant across the window, so it
            # depends on how many steps the trajectory has.
            sampler_config = EulerSampler.Config(
                gamma_0=self.spec.gamma_0,
                gamma_min=self.spec.gamma_min,
                noise_scale=self.spec.noise_scale,
                step_scale=self.spec.step_scale,
                churn_gamma=min(
                    self.spec.churn_total / self.diffusion_steps, 2.0**0.5 - 1.0
                ),
                churn_sigma_min=self.spec.churn_sigma_min,
                churn_sigma_max=self.spec.churn_sigma_max,
                variance_floor=self.spec.sampler_variance_floor,
            )
        sampler = EulerSampler(sampler_config)

        def denoise(
            coords: torch.Tensor,
            sigma: torch.Tensor,
            single_cond: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Compute denoise."""
            return self.diffusion_head(
                positions_noisy=coords,
                noise_level=sigma,
                batch=batch,
                embeddings=embeddings,
                use_conditioning=True,
                single_cond=single_cond,
            )

        def augment(coords: torch.Tensor) -> torch.Tensor:
            """Compute augment."""
            return masked_dense_rigid_motion(coords, mask)

        def realign(noisy: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
            """Rigid-align the noisy coordinates onto the denoised prediction.

            This family augments the state -- rotates and translates it -- every
            step, and its denoiser is NOT equivariant: it reads a fixed
            reference conformer through rotary position embeddings, so the
            prediction comes back in its own frame. Differencing across the two
            frames would make the Euler gradient a rotation as much as a
            gradient.
            """
            return weighted_rigid_align(noisy, denoised, mask)

        scope = getattr(self, "foldforge_sampler_scope", None)
        try:
            if scope is not None:
                positions = sampler_graph.sample(
                    sampler,
                    denoise,
                    (self.num_samples, *mask.shape, 3),
                    sigmas,
                    device=mask.device,
                    generator=None,
                    augment=augment,
                    align=realign if self.spec.realign_sampler else None,
                    coordinate_dims=2,
                    scopes=scope,
                    # the engine's fused rigid-motion glue is miniworld's
                    # (`backends.SERVES["sampler_ops"]`)
                    rigid_mask=mask
                    if getattr(self, "foldforge_sampler_ops", True)
                    else None,
                    conditioning=lambda levels: self.diffusion_head.conditioning_table(
                        batch, embeddings, levels
                    ),
                )
            else:
                positions = sampler.sample(
                    denoise,
                    (self.num_samples, *mask.shape, 3),
                    sigmas,
                    device=mask.device,
                    coordinate_dims=2,
                    chunk_size=None,
                    initialize_all=True,
                    augment=augment,
                    align=realign if self.spec.realign_sampler else None,
                )
        finally:
            # The fold's conditioning is not needed once its steps are done.
            self.diffusion_head._fold = None  # noqa: SLF001 - release fold-scoped conditioning
        if not isinstance(positions, torch.Tensor):
            message = "Sampling without trajectories must return a Tensor"
            raise TypeError(message)
        return {
            "atom_positions": positions,
            "mask": torch.tile(mask[None], (self.num_samples, 1, 1)),
        }

    def _confidence_over_samples(
        self,
        positions: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        batch_data: feat_batch.Batch,
    ) -> dict[str, torch.Tensor]:
        """Confidence outputs of the diffusion samples ``[S, ...]``.

        The samples go through the head in chunks (``ConfidenceHead.sample_chunk``: all
        at once while their pair stream is small, fewer at large token counts, never
        fewer than one); a chunk of one is the sequential per-sample call.
        """
        head = self.confidence_head
        count = positions.shape[0]
        chunker = getattr(head, "sample_chunk", None)
        chunk = 1 if chunker is None else chunker(count, positions.shape[1])
        outputs = []
        for start in range(0, count, chunk):
            part = positions[start : start + chunk]
            result = head(
                dense_atom_positions=part[0] if part.shape[0] == 1 else part,
                embeddings=embeddings,
                seq_mask=batch_data.token_features.mask,
                token_atoms_to_pseudo_beta=batch_data.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch_data.token_features.asym_id,
                batch=batch_data,
            )
            if part.shape[0] == 1:
                result = {key: value[None] for key, value in result.items()}
            outputs.append(result)
        return {
            key: outputs[0][key]
            if len(outputs) == 1
            else torch.cat([o[key] for o in outputs])
            for key in outputs[0]
        }

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> feat_batch.Batch:
        """Apply fold-level input conventions before embedding or RNG draws."""
        batch_data = feat_batch.Batch.from_data_dict(batch)
        if self.spec.empty_msa_without_alignment and not batch_data.msa.has_alignment():
            # Exact mode (Chai-1): its profile would otherwise be a one-hot of
            # the sequence, which moves every token of the initial single.
            batch_data = replace(batch_data, msa=batch_data.msa.without_alignment())
        return batch_data

    def _run_trunk(self, batch_data: feat_batch.Batch) -> dict[str, torch.Tensor]:
        """Initialize and recycle the trunk, drawing its MC decision once per fold."""
        num_res = batch_data.num_res

        target_feat, structure_feat = self.create_target_feat_embedding(batch_data)

        embeddings = {
            "pair": self._ssm_state(num_res, target_feat.device),
            "single": torch.zeros(
                [num_res, self.evoformer_seq_channel],
                dtype=torch.float32,
                device=target_feat.device,
            ),
            "target_feat": target_feat,
            "structure_target_feat": structure_feat,
        }

        # FoldForge counts total trunk passes for every family, including the first.
        passes = self.num_recycles
        # One coin per fold decides whether this fold's recycles run under MC
        # dropout (see DenseSpec.recycle_mc_dropout); the trunk seed owns it.
        recycle_dropout = 0.0
        if self.spec.recycle_mc_dropout is not None:
            apply_rate, rate = self.spec.recycle_mc_dropout
            if float(torch.rand((), device=target_feat.device)) < apply_rate:
                recycle_dropout = rate
        self.recycle_dropout_applied = bool(recycle_dropout)
        for pass_index in range(passes):
            embeddings = self.evoformer(
                batch=batch_data,
                prev=embeddings,
                target_feat=target_feat,
                first_pass=pass_index == 0,
                last_pass=pass_index == passes - 1,
                recycle_dropout=recycle_dropout,
            )
            if self.reference_precision:
                embeddings["pair"] = embeddings["pair"].float()
                embeddings["single"] = embeddings["single"].float()
        return embeddings

    def _prepare_diffusion_inputs(
        self,
        batch: dict[str, torch.Tensor],
        batch_data: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> DiffusionInputs:
        """Select the diffusion layout without replacing the residue head inputs."""
        # OpenDDE folds on STRUCTURAL tokens: the trunk stays on residues and
        # the denoiser runs on each residue's backbone and sidechain subtokens,
        # built from a second feature set the featurisation attached alongside.
        # Its confidence head and distogram read the residue branch, so only the
        # diffusion moves -- and its output comes back to the residue layout,
        # atom for atom, because the two share an atom axis.
        diffusion_batch, diffusion_embeddings = batch_data, embeddings
        if self.spec.structural_tokens:
            diffusion_batch = feat_batch.Batch.from_data_dict(
                {
                    key.removeprefix("struct/"): value
                    for key, value in batch.items()
                    if key.startswith("struct/")
                }
            )
            diffusion_embeddings = self.evoformer.expand_structural(
                embeddings,
                {
                    key.removeprefix("structbook/"): value
                    for key, value in batch.items()
                    if key.startswith("structbook/")
                },
                diffusion_batch,
                batch_data.token_features.asym_id,
            )

        return DiffusionInputs(
            batch=diffusion_batch,
            embeddings=diffusion_embeddings,
            residue_layout=batch if self.spec.structural_tokens else None,
            num_res=batch_data.num_res,
        )

    def _predict_heads(
        self,
        batch_data: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        samples: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Predict heads on the residue trunk and restored residue samples."""
        confidence_output = self._confidence_over_samples(
            samples["atom_positions"], embeddings, batch_data
        )

        distogram = self.distogram_head(batch_data, embeddings)

        return {
            "diffusion_samples": samples,
            "distogram": distogram,
            **confidence_output,
        }

    def run_trunk(self, batch: dict[str, torch.Tensor]) -> TrunkOutput:
        """Prepare residue/structural representations once for repeated sampling."""
        # Featurisation may hand float64 arrays (reference conformer positions of some families); autocast leaves float64 alone, so a
        # float64 input would meet a bf16 weight. Single precision is what every family's graph computes in.
        batch = {key: value.float() if value.dtype == torch.float64 else value for key, value in batch.items()}
        batch_data = self._prepare_batch(batch)
        embeddings = self._run_trunk(batch_data)
        diffusion = self._prepare_diffusion_inputs(batch, batch_data, embeddings)
        return TrunkOutput(batch_data, embeddings, diffusion)

    def sample_diffusion(
        self, trunk: TrunkOutput
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Sample and decode heads from a retained trunk without rerunning it."""
        samples = self._sample_diffusion(
            trunk.diffusion.batch, trunk.diffusion.embeddings
        )
        samples = trunk.diffusion.restore(samples)
        return self._predict_heads(trunk.batch, trunk.embeddings, samples)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Use the same trunk/sampling boundaries as staged inference."""
        return self.sample_diffusion(self.run_trunk(batch))
