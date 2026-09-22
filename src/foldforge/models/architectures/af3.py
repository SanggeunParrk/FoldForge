from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from team_gm.diffusion.augmentation import masked_dense_rigid_motion
from team_gm.diffusion.edm.sampling import EulerSampler
from team_gm.diffusion.edm.schedules import PowerLawSchedule
from team_gm.modules.bucketing import MSA_SHAPES, ceiling, pad_axis

from foldforge.data.features import dense as features
from foldforge.data.features import dense_batch as feat_batch
from foldforge.modules.dense import atom_cross_attention, diffusion_head, featurization
from foldforge.modules.dense.fused_template import FusedTemplateEmbedding
from foldforge.modules.dense.head import ConfidenceHead, DistogramHead
from foldforge.modules.dense.pair_init import (
    ChaiTokenEmbedder,
    ContactConditioning,
    SummedInputEmbedder,
    token_bond_types,
)
from foldforge.modules.dense.pairformer import EvoformerBlock, PairformerBlock
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
from foldforge.modules.dense.structural_tokens import StructuralTokenExpander
from foldforge.modules.dense.template import TemplateEmbedding

# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


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
        self.symmetric_bonds = spec.symmetric_bonds
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

        self.template_embedding = (
            TemplateEmbedding(spec)
            if spec.template == "af3"
            else FusedTemplateEmbedding(spec)
        )

        self.msa_activations = nn.Linear(
            spec.msa_feat_channel, self.msa_channel, bias=spec.msa_activations_bias
        )
        self.extra_msa_target_feat = nn.Linear(
            self.c_target_feat, self.msa_channel, bias=False
        )
        self.msa_stack = nn.ModuleList(
            [
                EvoformerBlock(
                    c_msa=self.msa_channel,
                    c_pair=self.pair_channel,
                    n_heads_pair=spec.pair_heads,
                    spec=spec,
                )
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
            [
                PairformerBlock(
                    c_pair=self.pair_channel,
                    c_single=self.seq_channel,
                    n_heads_pair=spec.pair_heads,
                    num_intermediate_factor=spec.pairformer_transition_factor,
                    with_single=True,
                    spec=spec,
                    pair_qkv_dim=spec.pair_qkv_dim,
                )
                for _ in range(self.pairformer_num_layer)
            ]
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
                [
                    PairformerBlock(
                        c_pair=self.pair_channel,
                        c_single=self.seq_channel,
                        n_heads=spec.structural_refiner_heads,
                        n_heads_pair=spec.pair_heads,
                        num_intermediate_factor=(
                            spec.structural_refiner_transition_factor
                        ),
                        single_intermediate_factor=(spec.pairformer_transition_factor),
                        with_single=True,
                        spec=spec,
                        pair_qkv_dim=spec.pair_qkv_dim,
                    )
                    for _ in range(spec.structural_refiner_layers)
                ]
            )

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
            pair, single = block(
                pair, pair_mask, single, seq_mask, extra_pair_bias=bias
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
        contact_matrix[gather_idxs[:, 0], gather_idxs[:, 1]] = 1.0
        if self.symmetric_bonds:
            contact_matrix[gather_idxs[:, 1], gather_idxs[:, 0]] = 1.0

        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0

        pair_activations = pair_activations + self.bond_embedding(
            contact_matrix[:, :, None]
        )
        if self.spec.bond_type_and_contact_init:
            # Both terms contribute on EVERY pair: bond order 0 and the unspecified
            # contact class are learned vectors, not zeros.
            bond_types = token_bond_types(batch, symmetric=self.symmetric_bonds)
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
        dtype = pair_activations.dtype

        if not self.spec.msa_keep_order:
            msa_batch = featurization.shuffle_msa(msa_batch)
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

    def forward(
        self,
        batch: feat_batch.Batch,
        prev: dict[str, Any],
        target_feat: torch.Tensor,
        *,
        first_pass: bool = False,
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

        recycled = prev["pair"]
        if self.spec.recycle_from_initial and first_pass:
            # The carry starts at the INITIAL representations rather than zeros,
            # so pass one already adds recycle_proj(norm(z_init)).
            recycled = pair_init if pair_init is not None else pair_activations
        pair_activations = pair_activations + self.prev_embedding(
            self.prev_embedding_layer_norm(recycled.to(pair_activations.dtype))
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

        pair_activations = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            token_features=batch.token_features,
            target_feat=(
                prev["single"].to(target_feat.dtype)
                if self.spec.msa_single_from_recycle
                else target_feat
            ),
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

        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
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

        dtype = self.evoformer.left_single.weight.dtype
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

        def denoise(coords: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            """Compute denoise."""
            return self.diffusion_head(
                positions_noisy=coords,
                noise_level=sigma,
                batch=batch,
                embeddings=embeddings,
                use_conditioning=True,
            )

        def augment(coords: torch.Tensor) -> torch.Tensor:
            """Compute augment."""
            return masked_dense_rigid_motion(coords, mask)

        positions = sampler.sample(
            denoise,
            (self.num_samples, *mask.shape, 3),
            sigmas,
            device=mask.device,
            coordinate_dims=2,
            chunk_size=None,
            initialize_all=True,
            augment=augment,
        )
        if not isinstance(positions, torch.Tensor):
            message = "Sampling without trajectories must return a Tensor"
            raise TypeError(message)
        return {
            "atom_positions": positions,
            "mask": torch.tile(mask[None], (self.num_samples, 1, 1)),
        }

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute the module output."""
        batch_data = feat_batch.Batch.from_data_dict(batch)
        num_res = batch_data.num_res

        target_feat, structure_feat = self.create_target_feat_embedding(batch_data)

        embeddings = {
            "pair": torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel],
                device=target_feat.device,
                dtype=torch.float32,
            ),
            "single": torch.zeros(
                [num_res, self.evoformer_seq_channel],
                dtype=torch.float32,
                device=target_feat.device,
            ),
            "target_feat": target_feat,
            "structure_target_feat": structure_feat,
        }

        # AF3 counts ADDITIONAL passes, so ten recycles run the trunk eleven
        # times; a family that counts TOTAL passes runs it ten. Getting this
        # wrong is invisible -- by then the recycle has nearly converged and an
        # extra pass only perturbs the output.
        passes = (
            max(1, self.num_recycles)
            if self.spec.recycles_are_total
            else self.num_recycles + 1
        )
        for pass_index in range(passes):
            embeddings = self.evoformer(
                batch=batch_data,
                prev=embeddings,
                target_feat=target_feat,
                first_pass=pass_index == 0,
            )
            if self.reference_precision:
                embeddings["pair"] = embeddings["pair"].float()
                embeddings["single"] = embeddings["single"].float()

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

        samples = self._sample_diffusion(diffusion_batch, diffusion_embeddings)
        if self.spec.structural_tokens:
            samples = _to_residue_layout(samples, batch, num_res)

        confidence_output_per_sample = []
        confidence_output_per_sample.extend(
            self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                embeddings=embeddings,
                seq_mask=batch_data.token_features.mask,
                token_atoms_to_pseudo_beta=batch_data.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch_data.token_features.asym_id,
                batch=batch_data,
            )
            for sample_dense_atom_position in samples["atom_positions"]
        )

        confidence_output = {}
        for key in confidence_output_per_sample[0]:
            confidence_output[key] = torch.stack(
                [sample[key] for sample in confidence_output_per_sample], dim=0
            )

        distogram = self.distogram_head(batch_data, embeddings)

        return {
            "diffusion_samples": samples,
            "distogram": distogram,
            **confidence_output,
        }
