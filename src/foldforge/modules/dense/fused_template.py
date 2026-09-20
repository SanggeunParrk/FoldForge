"""Fused template embedder shared by the Boltz-2, Protenix and RoseTTAFold3 weights.

AF3 embeds each template feature with its own projection, runs a small pairformer
and averages. This form projects one concatenated feature vector, adds the normed
query pair, runs the template pairformer, norms, averages the PRESENT templates and
projects ``relu`` of the mean back to the pair width:

    v = z_proj(z_norm(z)) + a_proj(features);  v = [v +] pairformer(v);  v = v_norm(v)
    u = mean over present templates;            out = u_proj(relu(u))

Families differ in the feature builder and in the two conventions named on the
spec, never in this forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldforge.data.features import dense as features
from foldforge.data.features import dense_protein as protein_data_processing
from foldforge.eval import dense_confidence as scoring
from foldforge.modules import ops as fastnn
from foldforge.modules.dense import pairformer
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec

_EPS = 1e-10


def _backbone(aatype: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor):
    """C, CA, N positions and their joint mask; rigid group 0 lists them in that order."""
    group = protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(aatype.device)
    index = group[aatype.to(torch.int64)][:, 0].to(torch.int64)  # (N, 3) = [C, CA, N]
    picked = torch.gather(positions, 1, index[..., None].expand(-1, -1, 3))
    picked_mask = torch.gather(mask, 1, index)
    return picked[:, 0], picked[:, 1], picked[:, 2], picked_mask.prod(dim=1).float()


def boltz2_template_features(
    aatype: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """109 features: CB distogram 38, CB mask, frame-sign vector 3, frame mask, restype 33 x 2."""
    tokens = aatype.shape[0]
    covered = mask.sum(-1) > 0
    beta, beta_mask = scoring.pseudo_beta_fn(aatype, positions, mask)
    c_atom, ca, n_atom, frame_mask = _backbone(aatype, positions, mask)
    first = c_atom - ca
    first = first / (first.norm(dim=-1, keepdim=True) + _EPS)
    second = n_atom - ca
    second = second - first * (first * second).sum(-1, keepdim=True)
    second = second / (second.norm(dim=-1, keepdim=True) + _EPS)
    rotation = torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-1)
    distance = ((beta[:, None] - beta[None]).square().sum(-1) + _EPS).sqrt()
    edges = torch.linspace(3.25, 50.75, 37, device=distance.device)
    distogram = F.one_hot((distance[..., None] > edges).sum(-1), 38)
    # The vendor normalises a (..., 3, 1) vector over its size-one axis, so the
    # trained feature is the per-component SIGN of R_j^T (ca_i - ca_j), not a unit
    # vector. The weights depend on it.
    local = torch.einsum("jdk,ijd->ijk", rotation, ca[:, None] - ca[None])
    geometry = (
        torch.cat(
            [
                distogram.float(),
                (beta_mask[:, None] * beta_mask[None])[..., None].float(),
                torch.sign(local),
                (frame_mask[:, None] * frame_mask[None])[..., None],
            ],
            dim=-1,
        )
        * visible[..., None].float()
    )
    restype = F.one_hot(torch.where(covered, aatype.to(torch.int64) + 2, 0), 33).float()
    return torch.cat(
        [
            geometry,
            restype[:, None].expand(tokens, tokens, 33),
            restype[None].expand(tokens, tokens, 33),
        ],
        dim=-1,
    )


_FEATURES = {"boltz2": (boltz2_template_features, 109)}


class FusedTemplateEmbedding(nn.Module):
    """Template embedder for weights trained with the fused form."""

    def __init__(self, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()
        self.spec = spec
        self.build, width = _FEATURES[spec.template]
        channels = spec.template_channel
        self.z_norm = fastnn.LayerNorm(spec.pair_channel)
        self.z_proj = nn.Linear(spec.pair_channel, channels, bias=False)
        self.a_proj = nn.Linear(width, channels, bias=False)
        self.tmpl_pairformer = nn.ModuleList(
            [
                pairformer.PairformerBlock(
                    c_pair=channels,
                    n_heads_pair=spec.template_heads,
                    num_intermediate_factor=spec.template_transition_factor,
                    with_single=False,
                    spec=spec,
                    pair_qkv_dim=spec.template_qkv_dim,
                )
                for _ in range(spec.template_layers)
            ]
        )
        self.v_norm = fastnn.LayerNorm(channels)
        self.u_proj = nn.Linear(channels, spec.pair_channel, bias=False)

    def forward(
        self,
        query_embedding: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Return the pair update from every present template."""
        dtype = query_embedding.dtype
        count = templates.aatype.shape[0]
        query = self.z_proj(self.z_norm(query_embedding))
        total = torch.zeros_like(query)
        present = query.new_zeros(())
        for index in range(count):
            template = templates[index]
            mask = template.atom_mask
            if not bool(mask.any()):
                continue
            visible = multichain_mask_2d
            if self.spec.template_visibility_by_coverage:
                # Visibility follows the SOURCE TEMPLATE, not the chain: a template
                # that covers two chains exposes their cross-chain block, and an
                # uncovered chain still sees itself.
                covered = mask.sum(-1) > 0
                visible = (covered[:, None] & covered[None]) | (
                    ~covered[:, None] & ~covered[None] & multichain_mask_2d.bool()
                )
            value = query + self.a_proj(
                self.build(
                    template.aatype,
                    template.atom_positions.float(),
                    mask.float(),
                    visible,
                ).to(dtype)
            )
            stacked = value
            for block in self.tmpl_pairformer:
                stacked = block(stacked, padding_mask_2d)
            if self.spec.template_stack_outer_residual and len(self.tmpl_pairformer):
                stacked = value + stacked
            total = total + self.v_norm(stacked)
            present = present + 1
        mean = total / present.clamp_min(1.0)
        return self.u_proj(torch.relu(mean))
