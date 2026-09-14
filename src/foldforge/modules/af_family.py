"""Shared AF-family engine adapters with explicit residual ownership.

Model-specific modules retain released checkpoint names. Callers use the residual
entry points below; no subtraction is used to undo a fused BF16 residual.
"""

from __future__ import annotations

from typing import Any

import torch
from miniworld_engine import ops
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.precision import NativeLinear
from torch import nn
from torch.nn import functional as F

_UNBATCHED_TOKEN_RANK = 2
_UNBATCHED_PAIR_RANK = 3


def enabled(module: nn.Module, x: torch.Tensor) -> bool:
    """Whether this tensor is eligible for the requested engine backend."""
    return (
        getattr(module, "foldforge_implementation", ImplementationType.PYTORCH)
        == ImplementationType.MINIWORLD_ENGINE
        and x.is_cuda
        and x.dtype == torch.bfloat16
    )


def transition_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the complete transition including its single residual."""
    if not enabled(module, x):
        return x + module(x)
    if hasattr(module, "input_layer_norm"):
        norm = module.input_layer_norm
        wa, wb = module.transition1.weight.chunk(2, dim=0)
        ws = module.transition2.weight
        n = module.num_intermediate_factor
    else:
        norm = module.layernorm1
        wa, wb = module.linear_no_bias_a.weight, module.linear_no_bias_b.weight
        ws = module.linear_no_bias.weight
        n = module.n
    # A semantic batch axis for upstream unbatched token activations.
    unbatched = x.ndim == _UNBATCHED_TOKEN_RANK
    value = x.unsqueeze(0) if unbatched else x
    out = ops.transition(
        value,
        ln_in_weight=norm.weight,
        ln_in_bias=norm.bias,
        expand_a_weight=wa,
        expand_b_weight=wb,
        squeeze_weight=ws,
        n=n,
        eps=norm.eps,
    )
    return out.squeeze(0) if unbatched else out


def triangle_residual(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Map released projection layouts and return the single residual."""
    if not enabled(module, x) or (
        hasattr(module, "c_hidden") and module.c_hidden != x.shape[-1]
    ):
        # Current engine TriMul requires equal pair and contraction widths.
        # Protenix templates use c_pair=64, c_hidden=128: retain their equation.
        return x + module(x, mask=mask)
    if hasattr(module, "projection"):
        norm, center = module.left_norm_input, module.center_norm
        # Official Haiku projection layout interleaves a,b per hidden channel.
        d = module.c_pair
        wp = module.projection.weight.reshape(d, 2, d).transpose(0, 1).reshape(2 * d, d)
        wg = module.gate.weight.reshape(d, 2, d).transpose(0, 1).reshape(2 * d, d)
        out, gate = module.output_projection, module.gating_linear
        outgoing = module.equation == "cik,cjk->cij"
        if not outgoing:
            # Xfold incoming contracts b[k,i] * a[k,j].
            # Engine incoming contracts a[k,i] * b[k,j].
            wp = torch.cat(wp.chunk(2)[::-1])
            wg = torch.cat(wg.chunk(2)[::-1])
    else:
        norm, center = module.layer_norm_in, module.layer_norm_out
        wp = torch.cat((module.linear_a_p.weight, module.linear_b_p.weight))
        wg = torch.cat((module.linear_a_g.weight, module.linear_b_g.weight))
        out, gate = module.linear_z, module.linear_g
        outgoing = module._outgoing  # noqa: SLF001 - pinned upstream direction flag
    unbatched = x.ndim == _UNBATCHED_PAIR_RANK
    value = x.unsqueeze(0) if unbatched else x
    pair_mask = mask.unsqueeze(0) if unbatched and mask is not None else mask
    result = ops.triangle_multiplicative_update(
        value,
        "outgoing" if outgoing else "incoming",
        mask=pair_mask,
        norm_in_weight=norm.weight,
        norm_in_bias=norm.bias,
        p_in_weight=wp,
        g_in_weight=wg,
        norm_out_weight=center.weight,
        norm_out_bias=center.bias,
        p_out_weight=out.weight,
        g_out_weight=gate.weight,
        eps=norm.eps,
    )
    return result.squeeze(0) if unbatched else result


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


def configure_model(  # noqa: C901 - atomic precision conversion
    model: nn.Module,
    backend: str = "miniworld",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Set native parameter precision, preserving the exact norm values."""
    if backend not in {"miniworld", "pytorch"}:
        msg = f"Unknown backend: {backend}"
        raise ValueError(msg)
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
