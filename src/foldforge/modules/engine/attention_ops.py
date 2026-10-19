"""The engine's projected-attention routes, off on the cards that have no Triton.

``miniworld_engine.ops`` reaches Triton kernels only (the hand-CUDA legs live in the engine's
modules, see ``modules.py``). H100 and B200 run hand-written CUDA or the PyTorch reference and
never an untuned Triton kernel, so on those cards each route here is the dense reference
computation; elsewhere it is the engine op.
"""

from __future__ import annotations

import torch
from miniworld_engine.architecture.checkpoints import backend_attention as _engine
from miniworld_engine.architecture.checkpoints.backend_attention import (
    engine_dense_attention,
    engine_triangle_attention,
)
from miniworld_engine.modules.dispatch import restricted_card

__all__ = [
    "engine_attention_supported",
    "engine_dense_attention",
    "engine_triangle_attention",
    "gated_projection",
    "layernorm_projection",
    "pair_bias_projection",
    "triton_allowed",
]


def triton_allowed(x: torch.Tensor) -> bool:
    """False on H100 / B200: a call there never reaches a Triton op."""
    return not (x.is_cuda and restricted_card(x.device))


def engine_attention_supported(
    module: torch.nn.Module, x: torch.Tensor, head_dim: int
) -> bool:
    """Whether the engine's pair-biased attention op serves this call."""
    return triton_allowed(x) and _engine.engine_attention_supported(module, x, head_dim)


def gated_projection(
    module: torch.nn.Module,
    linear: torch.nn.Module,
    gate: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """``linear(sigmoid(gate) * value)``; the engine's fused op where it is allowed."""
    if triton_allowed(value):
        return _engine.gated_projection(module, linear, gate, value)
    return linear(torch.sigmoid(gate) * value)


def layernorm_projection(
    norm: torch.nn.Module, linear: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """``linear(norm(x))``; the engine's fused LN + narrow projection where it is allowed."""
    if triton_allowed(x):
        return _engine.layernorm_projection(norm, linear, x)
    return linear(norm(x))


pair_bias_projection = layernorm_projection
