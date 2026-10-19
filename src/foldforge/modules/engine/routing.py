"""AF-family engine adapters with explicit residual ownership.

The dense modules keep the released checkpoint names; this module binds them to the
engine's whole-op entry points (``miniworld_engine.ops``) and to the shared reference
equations in ``foldforge.modules.reference``. Callers use the ``*_residual`` entry points: the
engine ops own their residual, so no subtraction is used to undo a fused BF16 residual.
"""

from __future__ import annotations

import torch
from miniworld_engine import ops
from miniworld_engine.architecture.exceptions import ImplementationType
from torch import nn

from foldforge.modules.engine.attention_ops import triton_allowed
from foldforge.modules.engine.modules import engine_of, residue_mask
from foldforge.modules.reference.transition import (
    TransitionConfig,
    adaptive_normalization,
    swiglu,
    transition,
)
from foldforge.modules.reference.triangle import TriangleConfig, triangle_update

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
    norm = module.input_layer_norm
    wa, wb = module.transition1.weight.chunk(2, dim=0)
    # A semantic batch axis for upstream unbatched token activations.
    unbatched = x.ndim == _UNBATCHED_TOKEN_RANK
    value = x.unsqueeze(0) if unbatched else x
    out = ops.transition(
        value,
        ln_in_weight=norm.weight,
        ln_in_bias=norm.bias,
        expand_a_weight=wa,
        expand_b_weight=wb,
        squeeze_weight=module.transition2.weight,
        n=module.num_intermediate_factor,
        eps=norm.eps,
    )
    return out.squeeze(0) if unbatched else out


def _length_divided_triangle(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Triangle update whose contraction is divided by the real sequence length.

    A LayerNorm follows, so only its epsilon can see the division:
    ``LN_eps(y / L) == LN_{eps * L**2}(y)``. The fused kernels share one epsilon
    between their two norms, so this convention takes the reference path.
    """
    center = module.center_norm
    length = x.shape[-2]
    if mask is not None:
        length = int(mask.reshape(-1, mask.shape[-1]).any(0).sum())
    epsilon = center.eps
    center.eps = epsilon * float(max(length, 1)) ** 2
    try:
        return reference_triangle(module, x, mask)
    finally:
        center.eps = epsilon


def triangle_residual(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Map the released projection layout to the engine op and return the residual."""
    cueq = (
        getattr(module, "foldforge_implementation", None)
        == ImplementationType.CUEQUIVARIANCE
        and x.is_cuda
        and x.shape[-1] % 32 == 0
    )
    if getattr(module, "divide_by_length", False):
        return x + _length_divided_triangle(module, x, mask)
    built = engine_of(module)
    if built is not None and enabled(module, x) and x.ndim == _UNBATCHED_PAIR_RANK:
        # The engine module owns the residual and reaches the hand-CUDA legs.
        return built(x[None], residue_mask(mask))[0]
    # The engine op needs the contraction width to equal the pair width; a stack
    # that widens only its projections (``tri_hidden_dim``) keeps the equation.
    if (
        not (enabled(module, x) or cueq)
        or module.center_norm.weight.shape[0] != x.shape[-1]
        or not (cueq or triton_allowed(x))
    ):
        return x + reference_triangle(module, x, mask)
    norm, center = module.left_norm_input, module.center_norm
    # Official Haiku projection layout interleaves a,b per hidden channel.
    d = module.c_pair
    wp = module.projection.weight.reshape(d, 2, d).transpose(0, 1).reshape(2 * d, d)
    wg = module.gate.weight.reshape(d, 2, d).transpose(0, 1).reshape(2 * d, d)
    outgoing = module.equation == "cik,cjk->cij"
    if not outgoing:
        # Xfold incoming contracts b[k,i] * a[k,j].
        # Engine incoming contracts a[k,i] * b[k,j].
        wp = torch.cat(wp.chunk(2)[::-1])
        wg = torch.cat(wg.chunk(2)[::-1])
    unbatched = x.ndim == _UNBATCHED_PAIR_RANK
    value = x.unsqueeze(0) if unbatched else x
    pair_mask = mask.unsqueeze(0) if unbatched and mask is not None else mask
    update = ops.triangle_multiplicative_update
    if cueq:
        from cuequivariance_torch import triangle_multiplicative_update

        update = triangle_multiplicative_update
    result = update(
        value,
        "outgoing" if outgoing else "incoming",
        mask=pair_mask,
        norm_in_weight=norm.weight,
        norm_in_bias=norm.bias,
        p_in_weight=wp,
        g_in_weight=wg,
        norm_out_weight=center.weight,
        norm_out_bias=center.bias,
        p_out_weight=module.output_projection.weight,
        g_out_weight=module.gating_linear.weight,
        eps=norm.eps,
    )
    # The vendor returns a delta; the MiniWorld op owns its residual.
    if cueq:
        result = value + result
    return result.squeeze(0) if unbatched else result


# Frozen configurations must be built outside forward: Pydantic validation inside
# a compiled call breaks its graph even when used as getattr's unused default.
_DEFAULT_TRANSITION_CONFIG = TransitionConfig()
_TRIANGLE_CONFIGS = {
    outgoing: TriangleConfig(outgoing=outgoing) for outgoing in (False, True)
}


def reference_transition(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Bind the dense transition's projection names to the shared equation."""
    return transition(
        x,
        normalize=module.input_layer_norm,
        project_a=lambda value: (
            value.to(module.transition1.weight.dtype) @ module.transition1.weight.T
        ),
        project_b=None,
        project_out=module.transition2,
        config=getattr(module, "transition_config", _DEFAULT_TRANSITION_CONFIG),
        training=module.training,
    )


def reference_adaln(
    module: nn.Module, x: torch.Tensor, cond: torch.Tensor | None
) -> torch.Tensor:
    """Bind optional single-conditioning to the shared adaptive normalization."""
    conditioned = module.use_single_cond
    if (cond is not None) != conditioned:
        message = "Condition presence must match the checkpoint configuration"
        raise ValueError(message)
    return adaptive_normalization(
        x,
        cond,
        normalize=module.layer_norm,
        normalize_cond=getattr(module, "single_cond_layer_norm", None)
        if conditioned
        else None,
        scale=module.single_cond_scale if conditioned else None,
        shift=module.single_cond_bias if conditioned else None,
        identity_scale=getattr(module, "identity_scale", False),
        activate_cond=getattr(module, "activate_cond", False),
    )


def reference_conditioned_transition(
    module: nn.Module, x: torch.Tensor, cond: torch.Tensor | None
) -> torch.Tensor:
    """Bind the transition layout; normalization, SwiGLU and gating are shared."""
    normalize = module.adaptive_layernorm
    if (
        not module.use_single_cond
        and enabled(module, x)
        and not module.training
        and triton_allowed(x)
    ):
        normalized = normalize(x, cond)
        wa, wb = module.transition1.weight.chunk(2, dim=0)
        unbatched = normalized.ndim == _UNBATCHED_TOKEN_RANK
        value = normalized.unsqueeze(0) if unbatched else normalized
        out = ops.swiglu_ffn(
            value, wa, wb, module.adaptive_zero_init.transition2.weight
        )
        return out.squeeze(0) if unbatched else out
    hidden = swiglu(
        normalize(x, cond),
        lambda value: (
            value.to(module.transition1.weight.dtype) @ module.transition1.weight.T
        ),
        None,
    )
    return module.adaptive_zero_init(hidden, cond)


def reference_triangle(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None
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
    )
