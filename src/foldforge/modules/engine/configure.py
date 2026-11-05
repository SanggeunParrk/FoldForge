"""Precision and backend configuration of a loaded model.

Norm parameters stay FP32 (the released values), every other parameter takes the native
dtype, and every module learns which backend it was asked for.
"""

from __future__ import annotations

from typing import Any

import torch
from miniworld_engine.modules.exceptions import ImplementationType as EngineImpl
from miniworld_engine.modules.primitives import LayerNorm as EngineLayerNorm
from torch import nn
from torch.nn import functional as F

from foldforge.layers.implementation import ImplementationType
from foldforge.modules.engine.backends import BACKENDS
from foldforge.modules.engine.native_linear import NativeLinear
from foldforge.modules.engine.routing import enabled

#: The engine ``LayerNorm`` a ``NativeLayerNorm`` hands its BF16 calls to (out of band: not a child).
_ENGINE_NORM = "_engine_norm_module"


def implementation_of(backend: str) -> object:
    """The ``foldforge_implementation`` a module carries for a backend name.

    The engine's enum has no member for ``anthropic`` (its kernels are reached through adapters, never the
    engine modules that ``enabled`` guards), so that backend's name stays a string.
    """
    if backend == "miniworld":
        return ImplementationType.MINIWORLD_ENGINE
    if backend == "anthropic":
        return backend
    return ImplementationType(backend)


class NativeLayerNorm(nn.Module):
    """FP32 parameters and reduction, original checkpoint affine names.

    On the ``miniworld`` backend a BF16 call is served by the engine's ``LayerNorm`` module
    (``miniworld_engine.modules.primitives.LayerNorm``: the general CUDA LayerNorm on H100 / B200, its
    portable kernels elsewhere), which shares this module's parameters and is attached out of band
    (no state_dict key). Every other call runs the FP32 PyTorch statements.
    """

    def __init__(self, source: nn.Module, backend: str) -> None:
        super().__init__()
        self.normalized_shape = tuple(
            getattr(source, "normalized_shape", getattr(source, "c_in", ()))
        )
        self.eps = source.eps
        self.weight = source.weight
        self.bias = source.bias
        self.foldforge_implementation = implementation_of(backend)
        # Built for the miniworld backend only (the engine resolves its CUDA / portable leg for the card here).
        object.__setattr__(
            self, _ENGINE_NORM, self._engine_norm() if backend == "miniworld" else None
        )

    def _engine_norm(self) -> nn.Module | None:
        """The engine ``LayerNorm`` over this module's own parameters, or None for a form it does not hold."""
        if len(self.normalized_shape) != 1 or (self.weight is None and self.bias is not None):
            return None
        with torch.device("meta"):
            norm = EngineLayerNorm(
                self.normalized_shape,
                eps=self.eps,
                elementwise_affine=self.weight is not None,
                bias=self.bias is not None,
                implementation=EngineImpl.MINIWORLD,
            )
        # Shared parameters: ``configure_model`` / ``.to`` replace their data in place, the engine module follows.
        norm.weight = self.weight
        norm.bias = self.bias
        return norm.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize in FP32 and return the original activation dtype."""
        engine = self.__dict__.get(_ENGINE_NORM)
        if engine is not None and enabled(self, x):
            return engine(x)
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
    if backend not in BACKENDS:
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
        if w is not None:
            n.weight.data = w.to(device)
        if b is not None:
            n.bias.data = b.to(device)
    for module in model.modules():
        module.foldforge_implementation = implementation_of(backend)
    model.eval().requires_grad_(requires_grad=False)
    norm_ids = {id(p) for n in norms for p in n.parameters(recurse=False)}
    for name, param in model.named_parameters():
        expected = torch.float32 if id(param) in norm_ids else dtype
        if param.dtype != expected:
            msg = f"{name}: {param.dtype}, expected {expected}"
            raise TypeError(msg)
    return model
