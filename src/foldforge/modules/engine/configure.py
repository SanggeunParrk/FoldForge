"""Precision and backend configuration of a loaded model.

Norm parameters stay FP32 (the released values), every other parameter takes the native
dtype, and every module learns which backend it was asked for.
"""

from __future__ import annotations

from typing import Any

import torch
from miniworld_engine import ops
from miniworld_engine.architecture.exceptions import ImplementationType
from miniworld_engine.modules.dispatch import restricted_card
from torch import nn
from torch.nn import functional as F

from foldforge.modules.engine.native_linear import NativeLinear
from foldforge.modules.engine.routing import enabled

_UNBATCHED_TOKEN_RANK = 2


class NativeLayerNorm(nn.Module):
    """FP32 parameters and reduction, original checkpoint affine names."""

    def __init__(self, source: nn.Module, backend: str) -> None:
        super().__init__()
        self.normalized_shape = tuple(
            getattr(source, "normalized_shape", getattr(source, "c_in", ()))
        )
        self.eps = source.eps
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer(
            "unit_scale", torch.ones(self.normalized_shape), persistent=False
        )
        self.register_buffer(
            "zero_bias", torch.zeros(self.normalized_shape), persistent=False
        )
        self.foldforge_implementation = (
            ImplementationType.MINIWORLD_ENGINE
            if backend == "miniworld"
            else ImplementationType(backend)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and return the original activation dtype."""
        if enabled(self, x) and len(self.normalized_shape) == 1:
            if restricted_card(x.device):
                # H100 / B200 have no Triton: the engine's general CUDA LayerNorm, or the
                # reference below for a width it does not take.
                from miniworld_engine.kernels.layernorm.cuda.general import (
                    layernorm_cuda,
                    supported,
                )

                if supported(x):
                    return layernorm_cuda(x, self.weight, self.bias, self.eps)
                return F.layer_norm(
                    x.float(), self.normalized_shape, self.weight, self.bias, self.eps
                ).to(x.dtype)
            shape = x.shape
            value = (
                x.reshape(1, 1, shape[-1])
                if x.ndim == 1
                else x.unsqueeze(0)
                if x.ndim == _UNBATCHED_TOKEN_RANK
                else x
            )
            return ops.layer_norm(
                value,
                self.weight if self.weight is not None else self.unit_scale,
                self.bias if self.bias is not None else self.zero_bias,
                self.eps,
            ).reshape(shape)
        return F.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        ).to(x.dtype)


def configure_model(
    model: nn.Module,
    backend: str = "miniworld",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Set native parameter precision, preserving the exact norm values."""
    if backend not in {"miniworld", "pytorch", "cuequivariance"}:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)
    if backend == "cuequivariance" and torch.device(device).type == "cuda":
        from cuequivariance_ops_torch import init_triton_cache

        init_triton_cache()
    norms = []

    def replace(parent: Any) -> None:
        for name, child in list(parent.named_children()):
            if (
                "LayerNorm" in type(child).__name__
                and hasattr(child, "weight")
                and hasattr(child, "eps")
            ):
                replacement = NativeLayerNorm(child, backend)
                setattr(parent, name, replacement)
                norms.append(replacement)
            elif isinstance(child, nn.Linear):
                setattr(parent, name, NativeLinear.from_linear(child))
            else:
                replace(child)

    replace(model)
    saved = [
        (
            n,
            None if n.weight is None else n.weight.detach().float().clone(),
            None if n.bias is None else n.bias.detach().float().clone(),
        )
        for n in norms
    ]
    # Fixed geometry/Fourier/confidence tables are not learned parameters.
    fixed_buffers = [
        (m, name, value.detach().clone())
        for m in model.modules()
        for name, value in m.named_buffers(recurse=False)
        if value is not None and value.dtype == torch.float32
    ]
    model.to(device=device, dtype=dtype)
    for m, name, value in fixed_buffers:
        setattr(m, name, value.to(device))
    for n, w, b in saved:
        n.unit_scale = n.unit_scale.float()
        n.zero_bias = n.zero_bias.float()
        if w is not None:
            n.weight.data = w.to(device)
        if b is not None:
            n.bias.data = b.to(device)
    for module in model.modules():
        module.foldforge_implementation = (
            ImplementationType.MINIWORLD_ENGINE
            if backend == "miniworld"
            else ImplementationType(backend)
        )
    model.eval().requires_grad_(requires_grad=False)
    norm_ids = {id(p) for n in norms for p in n.parameters(recurse=False)}
    for name, param in model.named_parameters():
        expected = torch.float32 if id(param) in norm_ids else dtype
        if param.dtype != expected:
            msg = f"{name}: {param.dtype}, expected {expected}"
            raise TypeError(msg)
    return model
