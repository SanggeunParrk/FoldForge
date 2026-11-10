"""The dense released-layout graph's bindings to the shared reference equations (converter graph only).

The dense modules (:mod:`foldforge.models.checkpoints.legacy`) hold the released parameter layout; these functions bind that layout to the
reference equations in :mod:`.reference`. Nothing here is imported at inference time.
"""

from __future__ import annotations

import torch
from torch import nn

from foldforge.models.checkpoints.legacy.reference.transition import (
    TransitionConfig,
    transition,
)
from foldforge.models.checkpoints.legacy.reference.triangle import (
    TriangleConfig,
    triangle_update,
)


def transition_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The complete transition including its residual."""
    return x + module(x)


def _inverse_length(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """1 / the real sequence length as a 0-d device tensor, counted without a host read (graph-capture safe)."""
    if mask is None:
        return torch.full((), 1.0 / x.shape[-2], device=x.device)
    count = mask.reshape(-1, mask.shape[-1]).any(0).sum().clamp_min(1)
    return count.float().reciprocal()


def triangle_residual(module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The complete triangle multiplication including its residual (RoseTTAFold3: the contraction divided by the real length)."""
    scale = _inverse_length(x, mask) if getattr(module, "divide_by_length", False) else None
    return x + reference_triangle(module, x, mask, contraction_scale=scale)


# Frozen configurations must be built outside forward: Pydantic validation inside a compiled call breaks its graph.
_DEFAULT_TRANSITION_CONFIG = TransitionConfig()
_TRIANGLE_CONFIGS = {outgoing: TriangleConfig(outgoing=outgoing) for outgoing in (False, True)}


def reference_transition(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Bind the dense transition projection names to the shared equation."""
    return transition(
        x,
        normalize=module.input_layer_norm,
        project_a=lambda value: value.to(module.transition1.weight.dtype) @ module.transition1.weight.T,
        project_b=None,
        project_out=module.transition2,
        config=getattr(module, "transition_config", _DEFAULT_TRANSITION_CONFIG),
        training=module.training,
    )


def reference_triangle(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None, contraction_scale: torch.Tensor | None = None
) -> torch.Tensor:
    """Map the packed checkpoint weights to the shared contraction (a raw delta)."""
    return triangle_update(
        x,
        mask,
        normalize=module.left_norm_input,
        project_a=module.projection,
        gate_a=module.gate,
        center=module.center_norm,
        project_out=module.output_projection,
        gate_out=module.gating_linear,
        config=_TRIANGLE_CONFIGS[module.equation == "cik,cjk->cij"],
        contraction_scale=contraction_scale,
    )
