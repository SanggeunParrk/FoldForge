# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

from dataclasses import dataclass

import torch
import torch.nn as nn
from team_gm.modules.blocks.composition import template_embedding_mean

from foldforge.data.constants import residue_names
from foldforge.data.features import dense as features
from foldforge.data.features import dense_protein as protein_data_processing
from foldforge.eval import dense_confidence as scoring
from foldforge.modules import ops as fastnn
from foldforge.modules.dense import pairformer
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
from foldforge.utils import dense_geometry as geometry


@dataclass
class DistogramFeaturesConfig:
    # The left edge of the first bin.
    """Represent distogram features config."""

    min_bin: float = 3.25
    # The left edge of the final bin. The final bin catches everything larger than
    # `max_bin`.
    max_bin: float = 50.75
    # The number of bins in the distogram.
    num_bins: int = 39


def dgram_from_positions(positions, config: DistogramFeaturesConfig):
    """Compute distogram from amino acid positions.

    Args:
      positions: (num_res, 3) Position coordinates.
      config: Distogram bin configuration.

    Returns:
      Distogram with the specified number of bins.
    """
    lower_breaks = torch.linspace(
        config.min_bin, config.max_bin, config.num_bins, device=positions.device
    )
    lower_breaks = torch.square(lower_breaks)
    upper_breaks_last = torch.ones(1, device=lower_breaks.device) * 1e8
    upper_breaks = torch.concatenate([lower_breaks[1:], upper_breaks_last], dim=-1)
    dist2 = torch.sum(
        torch.square(
            torch.unsqueeze(positions, dim=-2) - torch.unsqueeze(positions, dim=-3)
        ),
        dim=-1,
        keepdim=True,
    )

    dgram = (dist2 > lower_breaks).to(dtype=torch.float32) * (dist2 < upper_breaks).to(
        dtype=torch.float32
    )
    return dgram


def make_backbone_rigid(
    positions: geometry.Vec3Array,
    mask: torch.Tensor,
    group_indices: torch.Tensor,
) -> tuple[geometry.Rigid3Array, torch.Tensor]:
    """Make backbone Rigid3Array and mask.

    Args:
      positions: (num_res, num_atoms) of atom positions as Vec3Array.
      mask: (num_res, num_atoms) for atom mask.
      group_indices: (num_res, num_group, 3) for atom indices forming groups.

    Returns:
      tuple of backbone Rigid3Array and mask (num_res,).
    """
    backbone_indices = group_indices[:, 0]

    # main backbone frames differ in sidechain frame convention.
    # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
    # Hence using c, b, a, each of shape (num_res,).
    c, b, a = [backbone_indices[..., i] for i in range(3)]
    c, b, a = [x.to(dtype=torch.int64).unsqueeze(1) for x in [c, b, a]]

    # slice_index = jax.vmap(lambda x, i: x[i])
    # rigid_mask = (
    #     slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
    # ).astype(jnp.float32)

    rigid_mask = (
        torch.gather(mask, 1, a).squeeze(1)
        * torch.gather(mask, 1, b).squeeze(1)
        * torch.gather(mask, 1, c).squeeze(1)
    )

    frame_positions = []
    frame_positions.extend(
        geometry.Vec3Array(
            x=torch.gather(positions.x, 1, indices).squeeze(1),
            y=torch.gather(positions.y, 1, indices).squeeze(1),
            z=torch.gather(positions.z, 1, indices).squeeze(1),
        )
        for indices in [a, b, c]
    )

    rotation = geometry.Rot3Array.from_two_vectors(
        frame_positions[2] - frame_positions[1],
        frame_positions[0] - frame_positions[1],
    )
    rigid = geometry.Rigid3Array(rotation, frame_positions[1])

    return rigid, rigid_mask.to(dtype=torch.float32)


class TemplateEmbedding(nn.Module):
    """Embed a set of templates."""

    def __init__(self, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()

        self.pair_channel = spec.pair_channel
        self.num_channels = spec.template_channel

        self.single_template_embedding = SingleTemplateEmbedding(spec)

        self.output_linear = nn.Linear(self.num_channels, self.pair_channel, bias=False)

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the module output."""
        return template_embedding_mean(
            templates.aatype.shape[0],
            embed=lambda index: self.single_template_embedding(
                query_embedding, templates[index], padding_mask_2d, multichain_mask_2d
            ),
            empty=query_embedding.new_zeros(
                *query_embedding.shape[:-1], self.num_channels
            ),
            project=self.output_linear,
        )


class SingleTemplateEmbedding(nn.Module):
    """Embed a single template."""

    def __init__(self, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()

        self.num_channels = spec.template_channel
        pair_channel = spec.pair_channel
        self.template_stack_num_layer = 2

        self.dgram_features_config = DistogramFeaturesConfig()

        self.query_embedding_norm = fastnn.LayerNorm(pair_channel)
        self.template_feature_bias = (
            nn.Parameter(torch.zeros(self.num_channels))
            if spec.template_feature_bias
            else None
        )
        self.template_pair_embedding_0 = nn.Linear(39, self.num_channels, bias=False)
        self.template_pair_embedding_1 = nn.Linear(1, self.num_channels, bias=False)
        self.template_pair_embedding_2 = nn.Linear(31, self.num_channels, bias=False)
        self.template_pair_embedding_3 = nn.Linear(31, self.num_channels, bias=False)
        self.template_pair_embedding_4 = nn.Linear(1, self.num_channels, bias=False)
        self.template_pair_embedding_5 = nn.Linear(1, self.num_channels, bias=False)
        self.template_pair_embedding_6 = nn.Linear(1, self.num_channels, bias=False)
        self.template_pair_embedding_7 = nn.Linear(1, self.num_channels, bias=False)
        self.template_pair_embedding_8 = nn.Linear(
            pair_channel, self.num_channels, bias=False
        )

        self.template_embedding_iteration = nn.ModuleList(
            [
                pairformer.PairformerBlock(
                    c_pair=self.num_channels,
                    n_heads_pair=spec.pair_heads,
                    num_intermediate_factor=2,
                    spec=spec,
                    with_single=False,
                    pair_qkv_dim=spec.template_qkv_dim,
                )
                for _ in range(self.template_stack_num_layer)
            ]
        )

        self.output_layer_norm = fastnn.LayerNorm(self.num_channels)

    def construct_input(
        self, query_embedding, templates: features.Templates, multichain_mask_2d
    ) -> torch.Tensor:
        # Compute distogram feature for the template.

        """Compute construct input."""
        dtype = query_embedding.dtype

        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask

        dense_atom_positions = templates.atom_positions
        dense_atom_positions *= dense_atom_mask[..., None]

        pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )
        pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        pseudo_beta_mask_2d *= multichain_mask_2d
        dgram = dgram_from_positions(pseudo_beta_positions, self.dgram_features_config)
        dgram *= pseudo_beta_mask_2d[..., None]
        dgram = dgram.to(dtype=dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype=dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = torch.nn.functional.one_hot(
            aatype.to(dtype=torch.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        ).to(dtype=dtype)
        to_concat.append((aatype[None, :, :], 1))
        to_concat.append((aatype[:, None, :], 1))

        # Compute a feature representing the normalized vector between each
        # backbone affine - i.e. in each residues local frame, what direction are
        # each of the other residues.

        template_group_indices = torch.take_along_dim(
            protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(
                device=templates.aatype.device
            ),
            templates.aatype.to(dtype=torch.int64)[..., None, None],
            dim=0,
        )

        rigid, backbone_mask = make_backbone_rigid(
            geometry.Vec3Array.from_array(dense_atom_positions),
            dense_atom_mask,
            template_group_indices.to(dtype=torch.int32),
        )

        points = rigid.translation

        rigid.rotation = geometry.Rot3Array(
            rigid.rotation.xx[:, None],
            rigid.rotation.xy[:, None],
            rigid.rotation.xz[:, None],
            rigid.rotation.yx[:, None],
            rigid.rotation.yy[:, None],
            rigid.rotation.yz[:, None],
            rigid.rotation.zx[:, None],
            rigid.rotation.zy[:, None],
            rigid.rotation.zz[:, None],
        )
        rigid.translation = geometry.Vec3Array(
            rigid.translation.x[:, None],
            rigid.translation.y[:, None],
            rigid.translation.z[:, None],
        )

        rigid_vec = rigid.inverse().apply_to_point(points)
        unit_vector = rigid_vec.normalized()
        unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

        unit_vector = [x.to(dtype=dtype) for x in unit_vector]
        backbone_mask = backbone_mask.to(dtype=dtype)

        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        # Note that the backbone_mask takes into account C, CA and N (unlike
        # pseudo beta mask which just needs CB) so we add both masks as features.
        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = self.query_embedding_norm(query_embedding)

        to_concat.append((query_embedding, 1))

        act = 0

        for i, (raw_x, n_input_dims) in enumerate(to_concat):
            x = raw_x
            if n_input_dims == 0:
                x = x[..., None]
            act += self.get_submodule(f"template_pair_embedding_{i}")(x)

        if not isinstance(act, torch.Tensor):
            message = "Template embedding requires at least one feature"
            raise TypeError(message)
        if self.template_feature_bias is not None:
            # A fused feature projection's bias; nine bias-free Linears summed
            # cannot express it, so it is added once here.
            act = act + self.template_feature_bias
        return act

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the module output."""
        act = self.construct_input(query_embedding, templates, multichain_mask_2d)

        for pairformer_block in self.template_embedding_iteration:
            act = pairformer_block(act, pair_mask=padding_mask_2d)

        act = self.output_layer_norm(act)

        if not isinstance(act, torch.Tensor):
            message = "Template embedding requires at least one feature"
            raise TypeError(message)
        if self.template_feature_bias is not None:
            # A fused feature projection's bias; nine bias-free Linears summed
            # cannot express it, so it is added once here.
            act = act + self.template_feature_bias
        return act
