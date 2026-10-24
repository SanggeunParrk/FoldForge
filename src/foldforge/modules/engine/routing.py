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
_BATCHED_PAIR_RANK = 4
#: The cuequivariance triangle kernels take channel widths that are multiples of 32.
_CUEQ_CHANNELS = 32


def enabled(module: nn.Module, x: torch.Tensor) -> bool:
    """Whether this tensor is eligible for the requested engine backend."""
    return (
        getattr(module, "foldforge_implementation", ImplementationType.PYTORCH)
        == ImplementationType.MINIWORLD_ENGINE
        and x.is_cuda
        and x.dtype == torch.bfloat16
    )


def set_tf32_policy(
    model: nn.Module, *, tf32: bool, diffusion_tf32: bool = False
) -> None:
    """Record whether the engine's FP32 kernels (TF32 tensor cores) may serve this model's modules.

    The engine's FP32 paths multiply on TF32 tensor cores; exact FP32 has no engine kernel. A module
    therefore runs its FP32 engine kernel only when TF32 is allowed (``tf32``: ``execution.tf32``) and
    keeps the dense FP32 statements otherwise. ``diffusion_tf32`` additionally allows it in the diffusion
    head (the ``af3_default`` policy, whose atom blocks have always run on TF32 tensor cores) while the
    trunk stays exact.
    """
    head = getattr(model, "diffusion_head", None)
    diffusion = set() if head is None else {id(m) for m in head.modules()}
    for module in model.modules():
        module.foldforge_tf32 = tf32 or (diffusion_tf32 and id(module) in diffusion)


def enabled_fp32(module: nn.Module, x: torch.Tensor) -> bool:
    """Whether an FP32 activation may take the engine's FP32 (TF32 tensor core) kernels."""
    return (
        getattr(module, "foldforge_implementation", ImplementationType.PYTORCH)
        == ImplementationType.MINIWORLD_ENGINE
        and x.is_cuda
        and x.dtype == torch.float32
        and getattr(module, "foldforge_tf32", False)
    )


def served_by(module: nn.Module, x: torch.Tensor) -> object | None:
    """The engine module attached to ``module`` when ``x`` is eligible for it, else None.

    The one gate of every dense module that has an engine twin: the module owns the call
    (and its residual, where it has one) when this returns it.
    """
    built = engine_of(module)
    if built is None:
        return None
    if getattr(module, "foldforge_implementation", None) == ANTHROPIC:
        return built if not anthropic_refused(module, built, x) else None
    return built if enabled(module, x) else None


#: ``foldforge_implementation`` of a module on the Anthropic backend (the engine's enum has no member for it).
ANTHROPIC = "anthropic"


def anthropic_refused(module: nn.Module, built: object, x: torch.Tensor) -> bool:
    """Whether the Anthropic kernel behind ``built`` cannot take this call (it then runs the PyTorch reference).

    Asks the engine's own non-raising ``refusal`` predicate for the call; ``engine.unsupported`` decides between
    ONE log line per reason (the default) and an error. A batch of pair planes is not a refusal: the pairformer
    adapters send those through one plane at a time.
    """
    from foldforge.modules.engine.backends import call_refusal

    if x.ndim == _BATCHED_PAIR_RANK and x.shape[0] > 1 and hasattr(built, "foldforge_plane_only"):
        return True
    refuses = getattr(built, "foldforge_refuses", None)
    if refuses is None:
        return False
    return call_refusal(
        getattr(module, "foldforge_unsupported", "fallback"), built.foldforge_op, ANTHROPIC, refuses(x)
    )


def cueq_refused(module: nn.Module, op: str, x: torch.Tensor) -> bool:
    """Whether the cuequivariance kernel cannot take this call (it then runs the PyTorch reference, logged once)."""
    from foldforge.modules.engine.backends import call_refusal

    reason = None if x.shape[-1] % _CUEQ_CHANNELS == 0 else f"the channel width {x.shape[-1]} is not a multiple of 32"
    return call_refusal(getattr(module, "foldforge_unsupported", "fallback"), op, "cuequivariance", reason)


def cueq_declined(module: nn.Module, reason: str) -> None:
    """A trimul call the vendor op cannot express runs the PyTorch reference: say so once (or raise)."""
    from foldforge.modules.engine.backends import call_refusal

    call_refusal(getattr(module, "foldforge_unsupported", "fallback"), "trimul", "cuequivariance", reason)


def cueq_single_attention(
    wrapper: nn.Module, single: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor | None:
    """``single + attention_pair_bias(single, pair)`` on cuequivariance's fused op, None where it cannot take the call.

    The Pairformer's single attention is AF3's: LayerNorm, q (with bias) / k / v / gate projections of the
    normalised single, a pair-bias projection of the normalised pair, a gated output projection. The vendor op
    adds no residual and has no gate bias, so a stack with one (Chai-1) keeps the reference.
    """
    from foldforge.modules.engine.backends import call_refusal

    attention = wrapper.attention
    if (
        attention.use_single_cond
        or attention.kq_norm
        or not attention.use_gating_query
        or attention.gate_bias
        or attention.q_projection.bias is None
        or attention.qkv_dim * attention.num_head != attention.c_x
    ):
        why = "the vendor op is AF3's single attention (no single conditioning, k/q norm or gate bias)"
    elif single.dtype != torch.bfloat16 or not single.is_cuda:
        why = f"bf16 on CUDA only, got {single.dtype} on {single.device.type}"
    else:
        why = None
    if call_refusal(getattr(wrapper, "foldforge_unsupported", "fallback"), "single_attention", "cuequivariance", why):
        return None
    from cuequivariance_torch import attention_pair_bias

    layer_norm = attention.adaptive_layernorm.layer_norm
    out = attention_pair_bias(
        single,
        pair,
        None if mask is None else mask > 0,
        attention.num_head,
        layer_norm.weight,
        layer_norm.bias,
        attention.q_projection.weight,
        attention.q_projection.bias,
        attention.k_projection.weight,
        attention.v_projection.weight,
        attention.gating_query.weight,
        attention.adaptive_zero_init.transition2.weight,
        wrapper.projection.weight,
        None,
        None,
        wrapper.norm.weight,
        wrapper.norm.bias,
        eps=layer_norm.eps,
    )
    return single + (out[0] if isinstance(out, tuple) else out)


def anthropic_transition(module: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """``x + transition(x)`` from the release's Transition kernel (word ``v2``), None where it cannot take the call.

    The release's selection raises ``Refusal`` by name for a shape it has no cell for; that is the reason logged.
    """
    from foldforge.modules.engine.backends import call_refusal

    policy = getattr(module, "foldforge_unsupported", "fallback")
    if not x.is_cuda or x.dtype != torch.bfloat16:
        why = f"the release's kernels are bf16 on CUDA, got {x.dtype} on {x.device.type}"
    elif torch.is_grad_enabled():
        why = "the release's calls are inference-only (grad is enabled)"
    else:
        why = None
    if call_refusal(policy, "transition", ANTHROPIC, why):
        return None
    from miniworld_engine.integrations import anthropic

    norm = module.input_layer_norm
    wa, wb = module.transition1.weight.chunk(2, dim=0)
    parts = (wa, wb, module.transition2.weight, norm.weight, norm.bias)
    key = (norm.eps, *((id(t), t._version, t.dtype, tuple(t.shape)) for t in parts))
    cached = module.__dict__.get("_anthropic_transition")
    if cached is None or cached[0] != key:
        packed = anthropic.pack_transition(
            w_a=wa.contiguous(), w_b=wb.contiguous(), w_o=module.transition2.weight,
            ln_w=norm.weight, ln_b=norm.bias, eps=norm.eps,
        )
        cached = (key, packed)
        object.__setattr__(module, "_anthropic_transition", cached)
    unbatched = x.ndim < _BATCHED_PAIR_RANK
    value = x.reshape(1, -1, x.shape[-1]) if unbatched else x
    try:
        out, _selection = anthropic.transition(value, cached[1], row="v2")
    except Exception as error:  # the release refuses by name (opt_core ``Refusal``)
        if type(error).__name__ != "Refusal":
            raise
        call_refusal(policy, "transition", ANTHROPIC, str(error))
        return None
    return out.reshape(x.shape) if unbatched else out


def transition_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the complete transition including its single residual."""
    if getattr(module, "foldforge_implementation", None) == ANTHROPIC:
        served = anthropic_transition(module, x)
        if served is not None:
            return served
    if not (enabled(module, x) or enabled_fp32(module, x)):
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


def _inverse_length(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """1 / the real sequence length as a 0-d device tensor: counted without a host read, so a
    captured graph takes each replay's length."""
    if mask is None:
        return torch.full((), 1.0 / x.shape[-2], device=x.device)
    count = mask.reshape(-1, mask.shape[-1]).any(0).sum().clamp_min(1)
    return count.float().reciprocal()


def _length_divided_triangle(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Triangle update whose contraction is divided by the real sequence length (the reference)."""
    return reference_triangle(
        module, x, mask, contraction_scale=_inverse_length(x, mask)
    )


def _length_divided_engine(
    built: nn.Module, x: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """The engine TriMul on a contraction divided by the real length (RoseTTAFold3).

    The left operand ``sigmoid(gate) * (W_left x)`` is linear in ``W_left``, so the engine module
    runs with ``W_left / length``: the contraction it normalises is the released ``y / L``.
    """
    scale = _inverse_length(x, mask)
    built.to_left.weight.copy_(built.foldforge_left_weight.float() * scale)
    return built(x[None], residue_mask(mask))[0]


def triangle_residual(
    module: nn.Module, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Map the released projection layout to the engine op and return the residual."""
    cueq = (
        getattr(module, "foldforge_implementation", None)
        == ImplementationType.CUEQUIVARIANCE
        and x.is_cuda
        and not cueq_refused(module, "trimul", x)
    )
    built = served_by(module, x)
    if getattr(module, "divide_by_length", False):
        if built is not None and x.ndim == _UNBATCHED_PAIR_RANK:
            return _length_divided_engine(built, x, mask)
        if cueq:
            cueq_declined(module, "the contraction is divided by the real sequence length (RoseTTAFold3); the vendor op has no such form")
        return x + _length_divided_triangle(module, x, mask)
    if built is not None and x.ndim == _UNBATCHED_PAIR_RANK:
        # The engine module owns the residual and reaches the hand-CUDA legs.
        return built(x[None], residue_mask(mask))[0]
    # The engine op needs the contraction width to equal the pair width; a stack
    # that widens only its projections (``tri_hidden_dim``) keeps the equation.
    if cueq and module.center_norm.weight.shape[0] != x.shape[-1]:
        cueq_declined(module, "the contraction is wider than the pair; the vendor op has one width")
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
    module: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    contraction_scale: torch.Tensor | None = None,
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
