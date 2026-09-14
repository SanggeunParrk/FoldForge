"""Released-checkpoint interfaces around team-gm's shared Pairformer composition.

Only signatures, parameter layouts and upstream raw-delta contracts live here.
The ordering of Pairformer operations is owned by team-gm for all three families.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch
from team_gm.modules.blocks.pairformer import PairformerBlock
from torch import nn

from foldforge.modules.af_family import transition_residual, triangle_residual

_ATTENTION_OPTIONS = ContextVar("released_pairformer_options", default=(None, None))
_UNBATCHED_PAIR_RANK = 3


class _Transition(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return transition_residual(self.source, x)


class _TriangleMultiplication(nn.Module):
    def __init__(self, source: nn.Module, *, unbatched: bool) -> None:
        super().__init__()
        self.source, self.unbatched = source, unbatched

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.unbatched:
            return triangle_residual(
                self.source, x[0], None if mask is None else mask[0]
            )[None]
        return triangle_residual(self.source, x, mask)


class _TriangleAttention(nn.Module):
    def __init__(self, source: nn.Module, *, af3: bool, ending: bool) -> None:
        super().__init__()
        self.source, self.af3, self.ending = source, af3, ending

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.af3:
            return x + self.source(x[0], mask=None if mask is None else mask[0])[None]
        if self.ending:
            x = x.transpose(-2, -3).contiguous()
            mask = None if mask is None else mask.transpose(-1, -2)
        result = x + self.source(
            x,
            mask=None if mask is None else mask.float(),
            triangle_attention="torch",
            inplace_safe=True,
            chunk_size=_ATTENTION_OPTIONS.get()[0],
        )
        return result.transpose(-2, -3).contiguous() if self.ending else result


class _SingleAttention(nn.Module):
    def __init__(self, source: nn.Module, *, af3: bool) -> None:
        super().__init__()
        self.af3 = af3
        if af3:
            self.norm = source.single_pair_logits_norm
            self.projection = source.single_pair_logits_projection
            self.attention = source.single_attention_
        else:
            self.attention = source.attention_pair_bias

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.af3:
            logits = self.projection(self.norm(pair[0])).permute(2, 0, 1)
            delta = self.attention(
                single[0], None if mask is None else mask[0], pair_logits=logits
            )
            return single + delta[None]
        extra = _ATTENTION_OPTIONS.get()[1]
        options = {"extra_attn_bias": extra} if extra is not None else {}
        return single + self.attention(a=single, s=None, z=pair, **options)


class ReleasedPairformer(nn.Module):
    """Preserve the model call signature while using a team-gm block internally."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.af3 = source.__class__.__module__.startswith("foldforge.models.af3.")
        with_single = source.with_single if self.af3 else source.c_s > 0
        mul_out, mul_in, att_start, att_end = (
            (
                source.triangle_multiplication_outgoing,
                source.triangle_multiplication_incoming,
                source.pair_attention1,
                source.pair_attention2,
            )
            if self.af3
            else (
                source.tri_mul_out,
                source.tri_mul_in,
                source.tri_att_start,
                source.tri_att_end,
            )
        )
        self.block = PairformerBlock.from_components(
            triangle_outgoing=_TriangleMultiplication(mul_out, unbatched=self.af3),
            triangle_incoming=_TriangleMultiplication(mul_in, unbatched=self.af3),
            attention_starting=_TriangleAttention(
                att_start, af3=self.af3, ending=False
            ),
            attention_ending=_TriangleAttention(att_end, af3=self.af3, ending=True),
            transition_pair=_Transition(source.pair_transition),
            attention_single=_SingleAttention(source, af3=self.af3)
            if with_single
            else None,
            transition_single=_Transition(source.single_transition)
            if with_single
            else None,
        )
        self.eval()


class AF3Pairformer(ReleasedPairformer):
    """AF3 unbatched pair-first signature."""

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
    ) -> Any:
        """Run the released AF3 signature through the common block."""
        if self.training:
            message = "Released Pairformer adapter is qualified for inference only"
            raise RuntimeError(message)
        pair, single = self.block(
            pair[None],
            None if single is None else single[None],
            None if seq_mask is None else seq_mask[None],
            pair_mask=None if pair_mask is None else pair_mask[None].bool(),
        )
        return (pair[0], single[0]) if single is not None else pair[0]


class ProtenixPairformer(ReleasedPairformer):
    """Protenix/OpenDDE single-first signature, with scoped attention options."""

    def forward(
        self,
        s: torch.Tensor | None,
        z: torch.Tensor,
        pair_mask: torch.Tensor | None,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,  # noqa: FBT001, FBT002 - upstream call signature
        chunk_size: int | None = None,
        extra_attn_bias: torch.Tensor | None = None,
    ) -> tuple:
        """Run the released single-first signature through the common block."""
        if self.training:
            message = "Released Pairformer adapter is qualified for inference only"
            raise RuntimeError(message)
        if triangle_multiplicative != "torch" or triangle_attention != "torch":
            message = (
                "Select the backend through the model loader, not upstream kernel flags"
            )
            raise ValueError(message)
        del (
            inplace_safe
        )  # Adapter uses safe residual results and preserves source inputs.
        unbatched = z.ndim == _UNBATCHED_PAIR_RANK
        token = _ATTENTION_OPTIONS.set((chunk_size, extra_attn_bias))
        try:
            pair, single = self.block(
                z[None] if unbatched else z,
                s[None] if unbatched and s is not None else s,
                pair_mask=None
                if pair_mask is None
                else (pair_mask[None] if unbatched else pair_mask).bool(),
            )
        finally:
            _ATTENTION_OPTIONS.reset(token)
        return (
            single[0] if unbatched and single is not None else single,
            pair[0] if unbatched else pair,
        )

    # OpenDDE's local stack calls this entry point explicitly even without Fold-CP.
    # Both names route through the same team-gm composition.
    forward_source = forward


def install_pairformers(model: nn.Module) -> nn.Module:
    """Replace released Pairformer compositions after strict checkpoint loading."""
    converted = 0

    def visit(parent: nn.Module) -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            if type(child).__name__ == "PairformerBlock" and type(
                child
            ).__module__.startswith("foldforge.models."):
                adapter = (
                    AF3Pairformer
                    if type(child).__module__.startswith("foldforge.models.af3.")
                    else ProtenixPairformer
                )
                setattr(parent, name, adapter(child))
                converted += 1
            else:
                visit(child)

    visit(model)
    model.foldforge_load_report["team_gm_pairformer_blocks"] = converted
    return model
