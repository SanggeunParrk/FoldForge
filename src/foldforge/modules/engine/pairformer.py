"""Released-checkpoint interface around the engine's shared Pairformer composition.

Only the signature and the upstream raw-delta contracts live here: the ordering of
Pairformer operations is owned by the engine, the numerics by the wrapped dense modules.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch
from miniworld_engine.architecture.blocks.pairformer import PairformerBlock
from torch import nn

from foldforge.modules.engine.modules import engine_of
from foldforge.modules.engine.routing import (
    enabled,
    transition_residual,
    triangle_residual,
)

#: A bias over every head that the stack supplying it builds (a statement about token
#: pairs, not about heads); scoped to one block call.
_EXTRA_PAIR_BIAS = ContextVar("released_pairformer_extra_pair_bias", default=None)


class _Transition(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return transition_residual(self.source, x)


class _TriangleMultiplication(nn.Module):
    """The engine block is batched; the released module is not."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return triangle_residual(self.source, x[0], None if mask is None else mask[0])[
            None
        ]


class _TriangleAttention(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.source.residual(x[0], None if mask is None else mask[0])[None]


class _SingleAttention(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.norm = source.single_pair_logits_norm
        self.projection = source.single_pair_logits_projection
        self.attention = source.single_attention_

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        bias = _EXTRA_PAIR_BIAS.get()
        built = engine_of(self)
        if built is not None and bias is None and enabled(self.attention, single):
            # The engine module owns the residual and reaches the hand-CUDA legs.
            return built(single, pair, None if mask is None else mask > 0)
        logits = self.projection(self.norm(pair[0])).permute(2, 0, 1)
        if bias is not None:
            logits = logits + bias[None].to(logits.dtype)
        delta = self.attention(
            single[0], None if mask is None else mask[0], pair_logits=logits
        )
        return single + delta[None]


class AF3Pairformer(nn.Module):
    """AF3 unbatched pair-first signature around the engine's ``PairformerBlock``."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        with_single = source.with_single
        attention_start = getattr(source, "pair_attention1", None)
        attention_end = getattr(source, "pair_attention2", None)
        self.block = PairformerBlock.from_components(
            triangle_outgoing=_TriangleMultiplication(
                source.triangle_multiplication_outgoing
            ),
            triangle_incoming=_TriangleMultiplication(
                source.triangle_multiplication_incoming
            ),
            attention_starting=None
            if attention_start is None
            else _TriangleAttention(attention_start),
            attention_ending=None
            if attention_end is None
            else _TriangleAttention(attention_end),
            transition_pair=_Transition(source.pair_transition),
            attention_single=_SingleAttention(source) if with_single else None,
            transition_single=_Transition(source.single_transition)
            if with_single
            else None,
        )
        self.eval()

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        extra_pair_bias: torch.Tensor | None = None,
    ) -> Any:
        """Run the released AF3 signature through the common block.

        ``extra_pair_bias`` reaches the single attention through a scoped variable,
        which is what lets a fused block take a bias the released signature has no
        argument for.
        """
        if self.training:
            message = "Released Pairformer adapter is qualified for inference only"
            raise RuntimeError(message)
        token = _EXTRA_PAIR_BIAS.set(extra_pair_bias)
        try:
            pair, single = self.block(
                pair[None],
                None if single is None else single[None],
                None if seq_mask is None else seq_mask[None],
                pair_mask=None if pair_mask is None else pair_mask[None].bool(),
            )
        finally:
            _EXTRA_PAIR_BIAS.reset(token)
        return (pair[0], single[0]) if single is not None else pair[0]


def install_pairformers(model: nn.Module) -> nn.Module:
    """Replace released Pairformer compositions after strict checkpoint loading."""
    converted = 0

    def visit(parent: nn.Module) -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            # The shared block threads each update through the running
            # activation. A block that declares another schedule -- every update
            # reading the representation ENTERING it -- keeps its own forward:
            # swapping it would silently run the weights under the wrong
            # composition.
            own_schedule = getattr(child, "parallel", False)
            if (
                type(child).__name__ == "PairformerBlock"
                and hasattr(child, "triangle_multiplication_outgoing")
                and not own_schedule
            ):
                setattr(parent, name, AF3Pairformer(child))
                converted += 1
            else:
                visit(child)

    visit(model)
    model.foldforge_load_report["engine_pairformer_blocks"] = converted
    return model
