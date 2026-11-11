"""Expanding residue tokens into structural tokens, for the family that folds on them.

One predictor here does not run its diffusion on residues. Between the trunk and
the diffusion module it splits each residue into a backbone token and a sidechain
token -- glycine, and anything whose split comes out empty, stays a single token --
and runs the diffusion and the confidence heads on that expanded set.

This module is the expansion itself: each structural token copies its parent
residue's single and pair representations, then learned role embeddings, a
per-role-pair projection of the pair, learned pair-init terms for the structural
relationships, and a learned attention bias are added. The refiner that follows
is the ordinary pairformer block, so it is not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from foldforge.modules.engine.configure import engine_activation

#: The vendor's structural-token roles.
ATOM, PROTEIN_BB, PROTEIN_SC, DNA_BB, DNA_BASE, RNA_BB, RNA_BASE = range(7)
N_ROLES = 7
#: Bytes of one fp32 band of the structural pair built at a time (rows x S x c x 4).
_PAIR_BAND_BYTES = 1 << 30
_BACKBONE_ROLES = (PROTEIN_BB, DNA_BB, RNA_BB)
_BASE_ROLES = (DNA_BASE, RNA_BASE)

#: Role-pair classes of the learned pair term, in the vendor's order. Anything
#: the table does not name takes the last class.
_ROLE_PAIR_CLASSES = 8
_ROLE_PAIR_OTHER = 7


def _pair_kinds(role: torch.Tensor) -> dict[str, torch.Tensor]:
    """Which role family each token belongs to."""
    backbone = torch.zeros_like(role, dtype=torch.bool)
    for value in _BACKBONE_ROLES:
        backbone |= role == value
    base = torch.zeros_like(role, dtype=torch.bool)
    for value in _BASE_ROLES:
        base |= role == value
    return {"bb": backbone, "sc": role == PROTEIN_SC, "base": base}


class StructuralTokenExpander(nn.Module):
    """Residue-level single and pair representations, on structural tokens."""

    def __init__(self, c_single: int, c_pair: int, c_target_feat: int) -> None:
        super().__init__()
        self.c_single = c_single
        self.c_pair = c_pair

        self.single_input_role_embedding = nn.Embedding(N_ROLES, c_target_feat)
        self.single_role_embedding = nn.Embedding(N_ROLES, c_single)
        self.single_split_norm = nn.LayerNorm(c_single)
        self.single_split_1 = nn.Linear(c_single, 2 * c_single, bias=False)
        self.single_split_2 = nn.Linear(2 * c_single, c_single, bias=False)

        # One projection per ordered role pair, selected per token pair.
        self.pair_block_proj = nn.Parameter(
            torch.zeros(N_ROLES * N_ROLES, c_pair, c_pair)
        )
        self.same_parent_embedding = nn.Embedding(2, c_pair)
        self.same_residue_twin_embedding = nn.Embedding(2, c_pair)
        self.prev_bb_chain_embedding = nn.Embedding(2, c_pair)
        self.next_bb_chain_embedding = nn.Embedding(2, c_pair)
        self.role_pair_type_embedding = nn.Embedding(_ROLE_PAIR_CLASSES, c_pair)

        self.attn_bias_same_parent = nn.Parameter(torch.zeros(()))
        self.attn_bias_same_residue_twin = nn.Parameter(torch.zeros(()))
        self.attn_bias_prev_bb_chain = nn.Parameter(torch.zeros(()))
        self.attn_bias_next_bb_chain = nn.Parameter(torch.zeros(()))
        self.attn_bias_role_pair_type = nn.Parameter(torch.zeros(_ROLE_PAIR_CLASSES))

    def pair_features(
        self,
        parent: torch.Tensor,
        role: torch.Tensor,
        asym_id: torch.Tensor,
        prev_parent: torch.Tensor,
        next_parent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """The structural relationships both the pair term and the bias read.

        ``asym_id`` is RESIDUE-level, like everything else the trunk produces; a
        subtoken's chain is its parent's, so it is gathered here.
        """
        kind = _pair_kinds(role)
        bb, sc, base = kind["bb"], kind["sc"], kind["base"]
        same_parent = parent[:, None] == parent[None, :]
        chain = asym_id[parent]
        same_chain = chain[:, None] == chain[None, :]
        twin = same_parent & (
            (bb[:, None] & (sc[None, :] | base[None, :]))
            | (bb[None, :] & (sc[:, None] | base[:, None]))
        )
        both_bb = bb[:, None] & bb[None, :]
        prev_bb = both_bb & same_chain & (prev_parent[:, None] == parent[None, :])
        next_bb = both_bb & same_chain & (next_parent[:, None] == parent[None, :])

        role_pair = torch.full_like(parent[:, None] * parent[None, :], _ROLE_PAIR_OTHER)
        for left, right, value in (
            (bb, bb, 0),
            (bb, sc, 1),
            (sc, bb, 2),
            (sc, sc, 3),
            (bb, base, 4),
            (base, bb, 5),
            (base, base, 6),
        ):
            role_pair = torch.where(left[:, None] & right[None, :], value, role_pair)
        return {
            "role": role,
            "same_parent": same_parent,
            "twin": twin,
            "prev_bb": prev_bb,
            "next_bb": next_bb,
            "role_pair_type": role_pair,
        }

    def expand_single(
        self,
        target_feat: torch.Tensor,
        single: torch.Tensor,
        parent: torch.Tensor,
        role: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy each parent's single representations, then split and re-role them."""
        target_struct = target_feat[parent] + self.single_input_role_embedding(role)
        parent_single = single[parent]
        split = self.single_split_2(
            torch.nn.functional.silu(
                self.single_split_1(self.single_split_norm(parent_single))
            )
        )
        return target_struct, parent_single + split + self.single_role_embedding(role)

    def expand_pair(
        self,
        pair: torch.Tensor,
        parent: torch.Tensor,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Copy each parent pair, project it by role pair, add the relation terms.

        The result is ``[S, S, c]`` for S structural tokens, in the dtype the engine's blocks take (an fp32 pair under autocast is stored
        as the autocast dtype: the first refiner block would round it there). It is built a band of rows at a time: the gathered parent
        pair, the per-role projections and the relation terms each used to be a whole ``[S, S, c]`` fp32 tensor alive at once (five of
        them: 12.4 GiB each at 12 chains of 5I28), which is what made this the step that ran out of memory.
        """
        role = features["role"]
        # The 49 projections are indexed by the two tokens' OWN roles, as
        # `role_i * n_roles + role_j` -- NOT by the eight-class role-pair type,
        # which is a different quantity that only the embedding below reads.
        # Using the type here reaches eight of the forty-nine, and reaches them
        # for the wrong pairs.
        block = role[:, None] * N_ROLES + role[None, :]
        present = torch.unique(block).tolist()
        size, channels = parent.shape[0], self.c_pair
        out = pair.new_empty(
            (size, size, channels), dtype=engine_activation(pair.new_empty(0)).dtype
        )
        rows_per_band = max(1, min(size, _PAIR_BAND_BYTES // (size * channels * 4)))
        for start in range(0, size, rows_per_band):
            rows = slice(start, start + rows_per_band)
            parent_pair = pair[parent[rows]][:, parent]
            delta = torch.zeros_like(parent_pair)
            # One role pair at a time: all of them at once keeps every projection of the band live.
            for index in present:
                masked = torch.where((block[rows] == index)[..., None], parent_pair, 0.0)
                delta = delta + masked @ self.pair_block_proj[index]
            band = parent_pair + delta
            for name, table in (
                ("same_parent", self.same_parent_embedding),
                ("twin", self.same_residue_twin_embedding),
                ("prev_bb", self.prev_bb_chain_embedding),
                ("next_bb", self.next_bb_chain_embedding),
            ):
                band = band + table(features[name][rows].to(torch.int64))
            band = band + self.role_pair_type_embedding(features["role_pair_type"][rows])
            out[rows] = band.to(out.dtype)
        return out

    def attention_bias(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """One learned scalar per relationship, plus a per-role-pair table."""
        bias = self.attn_bias_role_pair_type[features["role_pair_type"]]
        for name, scalar in (
            ("same_parent", self.attn_bias_same_parent),
            ("twin", self.attn_bias_same_residue_twin),
            ("prev_bb", self.attn_bias_prev_bb_chain),
            ("next_bb", self.attn_bias_next_bb_chain),
        ):
            bias = bias + scalar * features[name].to(bias.dtype)
        return bias

    def forward(
        self,
        target_feat: torch.Tensor,
        single: torch.Tensor,
        pair: torch.Tensor,
        parent: torch.Tensor,
        role: torch.Tensor,
        asym_id: torch.Tensor,
        prev_parent: torch.Tensor,
        next_parent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the structural target features, single, pair and attention bias."""
        parent = parent.to(torch.int64)
        role = role.to(torch.int64)
        features = self.pair_features(
            parent,
            role,
            asym_id,
            prev_parent.to(torch.int64),
            next_parent.to(torch.int64),
        )
        target_struct, single_struct = self.expand_single(
            target_feat, single, parent, role
        )
        return (
            target_struct,
            single_struct,
            self.expand_pair(pair, parent, features),
            self.attention_bias(features),
        )
