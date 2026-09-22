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

import math

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


#: AF3 residue index -> the OpenFold3/Protenix 32-class order.
_AF3_TO_OF3 = (
    *range(21),
    31,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    25,
)


def protenix2_template_features(
    aatype: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """108 features: CB distogram 39, CB mask, restype 32 x 2, unit vector 3, frame mask.

    The distogram is masked by the chain visibility as well as by coverage. A
    distogram one-hot is nonzero for EVERY pair, because a distance always lands
    in some bin, so an unmasked cross-chain entry is not a zero but a confident
    fabricated inter-chain distance from a template that carries none.
    """
    tokens = aatype.shape[0]
    beta, beta_mask = scoring.pseudo_beta_fn(aatype, positions, mask)
    c_atom, ca, n_atom, frame_mask = _backbone(aatype, positions, mask)
    eps = 1e-6
    first = c_atom - ca
    first = first / (first.norm(dim=-1, keepdim=True) + eps)
    second = n_atom - ca
    second = second - first * (first * second).sum(-1, keepdim=True)
    second = second / (second.norm(dim=-1, keepdim=True) + eps)
    rotation = torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-1)
    # A real unit vector of (ca_j - ca_i) in residue i's frame.
    unit = torch.einsum("ilk,ijl->ijk", rotation, ca[None, :, :] - ca[:, None, :])
    unit = unit / (unit.norm(dim=-1, keepdim=True) + eps)

    squared = (beta[:, None] - beta[None]).square().sum(-1)[..., None]
    lower = torch.linspace(3.25, 50.75, 39, device=squared.device).square()
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])])
    distogram = ((squared > lower) & (squared < upper)).float()

    covered = (beta_mask[:, None] * beta_mask[None] * visible)[..., None].float()
    framed = (frame_mask[:, None] * frame_mask[None] * visible)[..., None].float()
    remap = torch.tensor(_AF3_TO_OF3, device=aatype.device, dtype=torch.int64)
    restype = F.one_hot(remap[aatype.to(torch.int64)], 32).float()
    return torch.cat(
        [
            distogram * covered,
            covered,
            # The j-varying block comes FIRST, which is the vendor's own order;
            # the projection is converted with no column permutation.
            restype[None, :, :].expand(tokens, tokens, 32),
            restype[:, None, :].expand(tokens, tokens, 32),
            unit * framed,
            framed,
        ],
        dim=-1,
    )


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


def rf3_template_features(
    aatype: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """66 features: CA distogram 64 (1 to 20 A), has-condition, joint noise level.

    An exact template has noise scale zero, so its noise level is the constant
    ``(log(1e-4 / 16) + 1.2) / 1.5``. The condition mask gates every channel.
    """
    _, ca, _, _ = _backbone(aatype, positions, mask)
    group = protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX.to(aatype.device)
    ca_index = group[aatype.to(torch.int64)][:, 0, 1].to(torch.int64)
    ca_mask = torch.gather(mask, 1, ca_index[:, None])[:, 0].float()
    distance = ((ca[:, None] - ca[None]).square().sum(-1) + _EPS).sqrt()
    distance = torch.nan_to_num(distance, nan=1e9)
    edges = torch.cat([torch.arange(1.0, 4.0, 0.1), torch.arange(4.0, 20.5, 0.5)]).to(
        distance.device
    )
    distogram = F.one_hot((distance[..., None] > edges).sum(-1), 64).float()
    condition = (ca_mask[:, None] * ca_mask[None] * visible.float())[..., None]
    noise = torch.full_like(condition, (math.log(1e-4 / 16.0) + 1.2) / 1.5)
    return torch.cat([distogram, condition, noise], dim=-1) * condition


_FEATURES = {
    "boltz2": (boltz2_template_features, 109),
    "rf3": (rf3_template_features, 66),
    "protenix2": (protenix2_template_features, 108),
}


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
                    tri_hidden_dim=spec.template_hidden_dim,
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
        if self.spec.template == "rf3":
            return self._single_pass(
                query, templates, padding_mask_2d, multichain_mask_2d
            )
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

    def _single_pass(
        self,
        query: torch.Tensor,
        templates: features.Templates,
        padding_mask_2d: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """One pass on the MEAN template feature, run even with no template.

        The vendor has no per-template loop and no gating: the normed query pair
        alone drives the stack when nothing is supplied, so the term is live on
        every recycle. No outer residual around the stack.
        """
        feature = query.new_zeros(*query.shape[:2], self.a_proj.in_features)
        present = 0
        count = templates.aatype.shape[0] if self.spec.use_input_templates else 0
        for index in range(count):
            template = templates[index]
            if not bool(template.atom_mask.any()):
                continue
            feature = feature + self.build(
                template.aatype,
                template.atom_positions.float(),
                template.atom_mask.float(),
                multichain_mask_2d,
            ).to(query.dtype)
            present += 1
        value = query + self.a_proj(feature / max(present, 1))
        for block in self.tmpl_pairformer:
            value = block(value, padding_mask_2d)
        return self.u_proj(torch.relu(self.v_norm(value)))
