from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
from team_gm.diffusion.augmentation import masked_dense_rigid_motion
from team_gm.diffusion.edm.sampling import EulerSampler
from team_gm.diffusion.edm.schedules import PowerLawSchedule
from team_gm.modules.bucketing import MSA_SHAPES, ceiling, pad_axis

from foldforge.data.features import dense as features
from foldforge.data.features import dense_batch as feat_batch
from foldforge.modules.dense import atom_cross_attention, diffusion_head, featurization
from foldforge.modules.dense.head import ConfidenceHead, DistogramHead
from foldforge.modules.dense.pairformer import EvoformerBlock, PairformerBlock
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
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


class Evoformer(nn.Module):
    """Represent evoformer."""

    def __init__(self, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()

        self.msa_channel = spec.msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024

        self.seq_channel = spec.seq_channel
        self.pair_channel = spec.pair_channel
        self.symmetric_bonds = spec.symmetric_bonds
        self.c_target_feat = 447

        self.left_single = nn.Linear(self.c_target_feat, self.pair_channel, bias=False)
        self.right_single = nn.Linear(self.c_target_feat, self.pair_channel, bias=False)

        self.prev_embedding_layer_norm = nn.LayerNorm(self.pair_channel)
        self.prev_embedding = nn.Linear(
            self.pair_channel, self.pair_channel, bias=False
        )

        self.c_rel_feat = 139
        self.position_activations = nn.Linear(
            self.c_rel_feat, self.pair_channel, bias=False
        )

        self.bond_embedding = nn.Linear(1, self.pair_channel, bias=False)

        self.template_embedding = TemplateEmbedding(spec)

        self.msa_activations = nn.Linear(34, self.msa_channel, bias=False)
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
                    with_single=True,
                    spec=spec,
                )
                for _ in range(self.pairformer_num_layer)
            ]
        )

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            max_relative_idx,
            max_relative_chain,
        ).to(dtype=pair_activations.dtype)

        pair_activations += self.position_activations(rel_feat)
        return pair_activations

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

        bonds_act = self.bond_embedding(contact_matrix[:, :, None])

        return pair_activations + bonds_act

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
    ) -> torch.Tensor:
        """Process MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

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
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )

        return pair_activations

    def forward(
        self,
        batch: feat_batch.Batch,
        prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the module output."""
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )

        pair_activations += self.prev_embedding(
            self.prev_embedding_layer_norm(prev["pair"].to(pair_activations.dtype))
        )

        pair_activations = self._relative_encoding(batch, pair_activations)

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
            target_feat=target_feat,
        )

        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(
                prev["single"].to(single_activations.dtype)
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
            "target_feat": target_feat,
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

        self.gamma_0 = spec.gamma_0
        self.gamma_min = spec.gamma_min
        self.noise_scale = spec.noise_scale
        self.step_scale = spec.step_scale

        self.evoformer_pair_channel = spec.pair_channel
        self.evoformer_seq_channel = spec.seq_channel

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder(
            spec=spec
        )

        self.evoformer = Evoformer(spec)

        self.diffusion_head = diffusion_head.DiffusionHead(spec)

        self.distogram_head = DistogramHead(c_pair=spec.pair_channel)
        self.confidence_head = ConfidenceHead(
            c_single=spec.seq_channel,
            c_pair=spec.pair_channel,
            spec=spec,
        )

    def create_target_feat_embedding(self, batch: feat_batch.Batch) -> torch.Tensor:
        """Create target feat embedding."""
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

        return torch.concatenate([target_feat, enc.token_act], dim=-1).to(
            self.evoformer.left_single.weight.dtype
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
                terminal_zero=False,
                include_minimum_before_zero=False,
            )
        )(self.diffusion_steps, device=mask.device)
        sampler = EulerSampler(
            EulerSampler.Config(
                gamma_0=self.gamma_0,
                gamma_min=self.gamma_min,
                noise_scale=self.noise_scale,
                step_scale=self.step_scale,
            )
        )

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

        target_feat = self.create_target_feat_embedding(batch_data)

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
        }

        # Recycles are additional trunk passes after the initial pass.
        for _ in range(self.num_recycles + 1):
            embeddings = self.evoformer(
                batch=batch_data, prev=embeddings, target_feat=target_feat
            )
            if self.reference_precision:
                embeddings["pair"] = embeddings["pair"].float()
                embeddings["single"] = embeddings["single"].float()

        samples = self._sample_diffusion(batch_data, embeddings)

        confidence_output_per_sample = []
        confidence_output_per_sample.extend(
            self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                embeddings=embeddings,
                seq_mask=batch_data.token_features.mask,
                token_atoms_to_pseudo_beta=batch_data.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch_data.token_features.asym_id,
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
