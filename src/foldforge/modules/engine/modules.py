"""The engine's own ``nn.Module`` ops, built from the released weights of dense modules.

The engine keeps its hand-CUDA (H100 / B200) legs behind ``miniworld_engine.modules.*``; FoldForge calls no engine
kernel function for model computation. A dense module that can be expressed as an engine module gets one at
install time, fed from the same tensors; any other dense module runs its PyTorch statements:

===============================  ===========================================================
dense module                     engine module
===============================  ===========================================================
``TriangleMultiplication``       ``TriangleMultiplication``   (residual included)
``GridSelfAttention``            ``TriangleAttention``        (residual included)
``MSAAttention``                 ``MSAPairWeightedAveraging`` (residual included)
``OuterProductMean``             ``OuterProductMean``         (raw update)
``Transition``                   ``Transition``               (residual included; pair, single and MSA transitions)
``DiffusionTransformer``         ``DiTBlock`` per block       (residual included)
``DiffusionTransformer``         ``DiTVariantStack``          (Boltz-2 / RoseTTAFold3 / Chai-1: the whole stack)
Pairformer single attention      ``AttentionPairBias``        (residual included)
===============================  ===========================================================

Only a module whose released equation the engine module computes is converted; anything else keeps
the dense reference path (see ``tests/kernels/test_engine_modules.py`` for the parity checks). The
engine modules are attached out of band (not as submodules), so ``state_dict`` and the strict
checkpoint load never see them.

Masks: the engine takes a residue mask ``[B, L]``. Every pair mask in the dense graph is the outer
product of the token mask (``seq_mask[:, None] * seq_mask[None, :]``), so its diagonal is the token
mask; ``residue_mask`` reads it from there without a host sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from miniworld_engine.modules import (
    AttentionPairBias,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.dit.module import DiTBlock
from miniworld_engine.modules.dit_variant import (
    AtomDiTVariantStack,
    DiTVariantStack,
)
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import ESMFOLD2 as _ENGINE_ESMFOLD2
from miniworld_engine.modules.local_dit import LocalDiTBlock

from foldforge.modules.engine.backends import Support, call_refusal
from foldforge.modules.engine.installation import (
    BuildResult,
    Candidate,
    record_installation,
)

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Any

    from torch import nn

    from foldforge.modules.engine.backends import BackendPlan
_ENGINE = ImplementationType.MINIWORLD
_ATTR = "_engine_module"
_TOKEN_DIT_COND = 384
#: The dense classes with an FP32 engine kernel on H100 (TF32 tensor cores): the diffusion transformers.
_FP32_ENGINE_CLASSES = ("DiffusionTransformer", "DiffusionCrossAttTransformer")
#: The dense classes an FP32 model also gets an engine module for: ``Transition`` runs FP32 on TF32 tensor cores (H100
#: generic sm_90a kernels), served only where TF32 is allowed (``routing.transition_support``).
_FP32_TWIN_CLASSES = (*_FP32_ENGINE_CLASSES, "Transition")
_DEFAULT_EPS = 1e-5
_ATOM_WIDTH, _ATOM_PAIR, _ATOM_KEYS = 128, 16, 128
_ATOM_ROPE_PAIRS = 16  # rotary pairs per 32-channel atom head (3 x 2 spatial + 10 uid)

def attach(module: nn.Module, engine_module: object) -> None:
    """Attach out of band: not a registered child, so no state_dict key and no ``.to`` move."""
    object.__setattr__(module, _ATTR, engine_module)
    kind = {
        "DiffusionTransformer": "token",
        "DiffusionCrossAttTransformer": "atom",
    }.get(type(module).__name__)
    if kind is not None:
        from foldforge.modules.engine.transformers import attach_adapter

        attach_adapter(module, engine_module, kind)


def engine_of(module: nn.Module) -> object | None:
    """The engine module attached to ``module``, if any."""
    return module.__dict__.get(_ATTR)


def attachment_support(module: nn.Module) -> Support:
    """Check attachment only; shape and backend eligibility belong to the caller."""
    return Support(refusal="not_installed") if engine_of(module) is None else Support()


def token_mask(pair_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Token masks from outer-product pair masks with any leading batch axes."""
    if pair_mask is None:
        return None
    return (pair_mask.diagonal(dim1=-2, dim2=-1) > 0).contiguous()


def residue_mask(pair_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Token mask ``[1, L]`` (bool) from an outer-product pair mask ``[L, L]``."""
    mask = token_mask(pair_mask)
    return None if mask is None else mask[None]


def _copy(
    engine_module: nn.Module,
    sources: dict[str, torch.Tensor],
    like: torch.Tensor,
    *,
    keep_master: bool = False,
) -> nn.Module:
    """Fill ``engine_module`` from named tensors: norms stay FP32, matrices take ``like.dtype``.

    ``keep_master``: a matrix that is FP32 in the loaded model stays FP32 (the engine's DiT blocks
    take FP32 master weights under BF16 activations), so the diffusion transformers see the released
    values instead of a BF16 rounding of them.
    """
    with torch.no_grad():
        for name, parameter in engine_module.named_parameters():
            source = sources[name].detach()
            norm = "ln_" in name
            dtype = (
                torch.float32
                if norm or (keep_master and source.dtype == torch.float32)
                else like.dtype
            )
            parameter.data = source.to(device=like.device, dtype=dtype).contiguous()
    return engine_module.to(like.device).eval().requires_grad_(requires_grad=False)


def _finish(built: nn.Module, like: torch.Tensor) -> nn.Module:
    """An engine module built by ``from_released``: on the model's device, inference only."""
    return built.to(like.device).eval().requires_grad_(requires_grad=False)


def _build_triangle_multiplication(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> BuildResult:
    """Engine TriMul from a dense ``TriangleMultiplication`` (``from_released``; None when it cannot serve it)."""
    # ``from_released`` sets one eps on the module's LayerNorms.
    if module.left_norm_input.eps != module.center_norm.eps:
        return BuildResult(refusal="norm_eps_mismatch")
    built = TriangleMultiplication.from_released(
        module.state_dict(),
        outgoing=module.equation == "cik,cjk->cij",
        dtype=like.dtype,
        norm_eps=module.left_norm_input.eps,
        implementation=implementation,
        p_drop=0.0,
    )
    _finish(built, like)
    if getattr(module, "divide_by_length", False):
        # RoseTTAFold3 divides the contraction by the real length before the centre norm. The engine's own
        # ``divide_by_length`` refuses every kernel; the left operand is linear in its projection, so the call scales
        # a copy of that weight by 1 / length on the device (``routing.triangle_residual``) and the kernels run unchanged.
        built.foldforge_left_weight = built.to_left.weight.detach().clone()
    return BuildResult(value=built)


def _build_triangle_attention(module: nn.Module, like: torch.Tensor) -> BuildResult:
    """Engine TriangleAttention from a dense ``GridSelfAttention`` (``from_released``).

    The ending node takes its pair bias from the transposed pair (``transposed_bias``) or, as AF3's
    released code does, from the pair as it is. RoseTTAFold3's gate / out-projection biases and Chai-1's
    second out-projection (``dual_output``) follow from the weights; its ending node returned as the
    transposed pair has it is ``untransposed_output``. The heads may span more than the pair width
    (template stacks: pair 64, 4 x 32 heads): ``from_released`` reads the hidden width from the weights.
    """
    ending_as_is = module.transpose and not module.transposed_bias
    built = TriangleAttention.from_released(
        module.state_dict(),
        n_head=module.num_head,
        starting=not module.transpose,
        dtype=like.dtype,
        norm_eps=module.act_norm.eps,
        implementation=_ENGINE,
        transposed_bias=not ending_as_is,
        untransposed_output=module.untransposed_output,
    )
    return BuildResult(value=_finish(built, like))


def _build_pair_weighted_averaging(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> BuildResult:
    """Engine MSAPairWeightedAveraging from a dense ``MSAAttention`` (``from_released``).

    The engine drops the pair LayerNorm offset: it adds one constant to every key's logit of a
    head, which the softmax over keys cancels exactly.
    """
    built = MSAPairWeightedAveraging.from_released(
        module.state_dict(),
        n_head=module.num_head,
        dtype=like.dtype,
        norm_eps=module.act_norm.eps,
        implementation=implementation,
        p_drop=0.0,
    )
    built.ln_pair.eps = module.pair_norm.eps
    return BuildResult(value=_finish(built, like))


def _build_outer_product_mean(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> BuildResult:
    """Engine OuterProductMean from a dense ``OuterProductMean`` (``from_released``).

    The kernels divide by ``clamp(count, 1)`` (``clamped_norm``); the released form divides by ``count + 1e-3``, which the
    engine computes exactly (``count_offset``) on its PyTorch statements only: its kernels refuse it. The two differ by
    0.1% at a single row and not at all beyond bf16 resolution after that; they differ at pairs with no valid row
    (padding), which the graph masks. So the twin takes the clamped divisor and the kernels serve it. ``bias_after_norm``
    is the engine's ``normalize_before_proj``. RoseTTAFold3's projection biases and Chai-1's groups (G outer products
    of K-wide halves, LayerNorm'd, no count division) follow from the weights.
    """
    built = OuterProductMean.from_released(
        module.state_dict(),
        dtype=like.dtype,
        norm_eps=module.layer_norm_input.eps,
        bias_after_norm=module.bias_after_norm,
        clamped_norm=True,
        implementation=implementation,
    )
    if module.groups > 1:
        built.product_norm.eps = module.product_norm.eps
    return BuildResult(value=_finish(built, like))


def _build_transition(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> BuildResult:
    """Engine ``Transition`` from a dense ``Transition`` (``from_released``): LayerNorm -> SwiGLU -> projection, residual included.

    The pair, single and MSA transitions of the trunk, template and confidence stacks. ``n`` follows from the weights; the norm
    keeps the module's eps and FP32 affine. An FP32 model gets an FP32 twin (TF32 kernels, served where TF32 is allowed).
    """
    norm = module.input_layer_norm
    if norm.weight is None or norm.bias is None:
        return BuildResult(refusal="non_affine_norm")
    built = Transition.from_released(
        module.state_dict(),
        dtype=like.dtype,
        norm_eps=norm.eps,
        implementation=implementation,
    )
    return BuildResult(value=_finish(built, like))


def _build_diffusion_blocks(module: nn.Module, like: torch.Tensor) -> BuildResult:
    """One engine ``DiTBlock`` per block of a dense ``DiffusionTransformer`` (sigmoid-form blocks).

    AF3 shares one pair LayerNorm and projects the pair bias once per super block of
    ``super_block_size`` blocks; the OpenFold lineage norms and projects per block. Each engine
    block owns its LayerNorm gamma and head rows, so the shared form is replicated/sliced here and
    the per-block form maps one to one. The pair LayerNorm offset is dropped in both: it adds one
    constant to every key's logit of a head, which the softmax over keys cancels exactly.
    """
    attention = module.self_attention
    first = attention[0]
    per_block = module.per_block_pair
    norms = (
        list(module.pair_input_layer_norm)
        if per_block
        else [module.pair_input_layer_norm] * module.num_blocks
    )
    eps_values = _eps_of(first.adaptive_layernorm) | {
        getattr(norm, "eps", None) for norm in norms
    }
    plain = (
        not module.parallel
        and eps_values == {_DEFAULT_EPS}
        and all(
            not block.kq_norm
            and block.use_gating_query
            and not block.gate_bias
            and not block.adaptive_layernorm.identity_scale
            and not getattr(block.adaptive_layernorm, "rms_norm", False)
            and not getattr(block.adaptive_layernorm, "activate_cond", False)
            and block.q_projection.bias is not None
            for block in attention
        )
        and all(
            type(block).__name__ == "DiffusionTransition"
            and not block.adaptive_layernorm.identity_scale
            for block in module.transition_block
        )
    )
    if not plain:
        return BuildResult(refusal="block_structure_mismatch")
    heads = module.num_head
    blocks = []
    for index, (attn, trans) in enumerate(
        zip(attention, module.transition_block, strict=True)
    ):
        if per_block:
            rows = module.pair_logits_projection[index].weight
        else:
            super_block, offset = divmod(index, module.super_block_size)
            rows = module.pair_logits_projection[super_block].weight[
                offset * heads : (offset + 1) * heads
            ]
        expand_a, expand_b = trans.transition1.weight.chunk(2, dim=0)
        block = DiTBlock(
            module.c_act,
            module.c_single_cond,
            module.c_pair_cond,
            heads,
            expand_a.shape[0] // module.c_act,
            implementation=_ENGINE,
        )
        a_ln, t_ln = attn.adaptive_layernorm, trans.adaptive_layernorm
        _copy(
            block,
            {
                "attention.ada_ln_in.ln_cond.weight": a_ln.single_cond_layer_norm.weight,
                "attention.ada_ln_in.to_scale.weight": a_ln.single_cond_scale.weight,
                "attention.ada_ln_in.to_scale.bias": a_ln.single_cond_scale.bias,
                "attention.ada_ln_in.to_bias.weight": a_ln.single_cond_bias.weight,
                "attention.to_query.weight": attn.q_projection.weight,
                "attention.to_query.bias": attn.q_projection.bias,
                "attention.to_key.weight": attn.k_projection.weight,
                "attention.to_value.weight": attn.v_projection.weight,
                "attention.ln_pair.weight": norms[index].weight,
                "attention.to_bias.weight": rows,
                "attention.to_gate.weight": attn.gating_query.weight,
                "attention.to_out.weight": attn.adaptive_zero_init.transition2.weight,
                "attention.to_scale.weight": attn.adaptive_zero_init.adaptive_zero_cond.weight,
                "attention.to_scale.bias": attn.adaptive_zero_init.adaptive_zero_cond.bias,
                "transition.ada_ln_in.ln_cond.weight": t_ln.single_cond_layer_norm.weight,
                "transition.ada_ln_in.to_scale.weight": t_ln.single_cond_scale.weight,
                "transition.ada_ln_in.to_scale.bias": t_ln.single_cond_scale.bias,
                "transition.ada_ln_in.to_bias.weight": t_ln.single_cond_bias.weight,
                "transition.expand_a.weight": expand_a,
                "transition.expand_b.weight": expand_b,
                "transition.squeeze.weight": trans.adaptive_zero_init.transition2.weight,
                "transition.to_scale.weight": trans.adaptive_zero_init.adaptive_zero_cond.weight,
                "transition.to_scale.bias": trans.adaptive_zero_init.adaptive_zero_cond.bias,
            },
            like,
            keep_master=True,
        )
        blocks.append(block)
    return BuildResult(value=blocks)


def _variant_adaln_plain(norm: nn.Module, identity: bool, eps: float) -> bool:
    """Whether an AdaLN is the layer-norm form the variant stacks compute (not ESMFold2's RMS / SiLU form)."""
    return bool(
        norm.use_single_cond
        and not getattr(norm, "rms_norm", False)
        and not getattr(norm, "activate_cond", False)
        and norm.identity_scale == identity
        and norm.layer_norm.eps == eps
        and (identity or norm.single_cond_layer_norm.eps == _DEFAULT_EPS)
        and (norm.single_cond_scale.bias is None) == identity
    )


def _variant_transition_ok(
    trans: nn.Module, identity: bool, eps: float, gated: bool
) -> bool:
    zero = trans.adaptive_zero_init
    return bool(
        type(trans).__name__
        == ("GatedDiffusionTransition" if gated else "DiffusionTransition")
        and _variant_adaln_plain(trans.adaptive_layernorm, identity, eps)
        and zero.project
        and not zero.raw_gate
        and zero.adaptive_zero_cond.bias is not None
    )


def _build_diffusion_variant_stack(
    module: nn.Module, like: torch.Tensor
) -> BuildResult:
    """The engine's ``DiTVariantStack`` for a dense ``DiffusionTransformer`` the per-block ``DiTBlock`` cannot serve.

    Boltz-2 (sequential, up-gate transition), RoseTTAFold3 (parallel, kq-norm) and Chai-1 (parallel, identity
    AdaLN, no gating query, affine pair LayerNorm): the engine owns the whole stack, residuals included.
    The pair LayerNorm offset is dropped by the engine's bias (a per-head constant on the logits).
    """
    attention = module.self_attention
    first = attention[0]
    first_norm = first.adaptive_layernorm
    identity = first_norm.identity_scale
    eps = first_norm.layer_norm.eps
    per_block = module.per_block_pair
    norms = (
        list(module.pair_input_layer_norm)
        if per_block
        else [module.pair_input_layer_norm] * module.num_blocks
    )
    gated = type(module.transition_block[0]).__name__ == "GatedDiffusionTransition"
    heads = module.num_head
    factor = module.transition_block[0].num_intermediate_factor
    if not (
        len({(norm.eps, norm.bias is None) for norm in norms}) == 1
        and all(
            _variant_adaln_plain(block.adaptive_layernorm, identity, eps)
            and block.kq_norm == first.kq_norm
            and block.use_gating_query == first.use_gating_query
            and not block.gate_bias
            and block.q_projection.bias is not None
            and block.adaptive_zero_init.project
            and not block.adaptive_zero_init.raw_gate
            and block.adaptive_zero_init.adaptive_zero_cond.bias is not None
            and (
                not block.kq_norm
                or block.query_layer_norm.eps
                == block.key_layer_norm.eps
                == _DEFAULT_EPS
            )
            for block in attention
        )
        and all(
            _variant_transition_ok(block, identity, eps, gated)
            and block.num_intermediate_factor == factor
            for block in module.transition_block
        )
    ):
        return BuildResult(refusal="stack_structure_mismatch")
    # The block options (identity AdaLN, gating, kq norm, up-gate, pair norm bias) follow from the weights.
    stack = DiTVariantStack.from_released(
        module.state_dict(),
        n_head=heads,
        parallel=module.parallel,
        super_block_size=module.super_block_size,
        implementation=_ENGINE,
        adaln_eps=eps,
        pair_eps=norms[0].eps,
    )
    return BuildResult(value=_finish(stack, like))


def _build_atom_variant_stack(module: nn.Module, like: torch.Tensor) -> BuildResult:
    """The engine's ``AtomDiTVariantStack`` for a dense ``DiffusionCrossAttTransformer`` the ``LocalDiTBlock`` cannot serve.

    Boltz-2 (shared pair projection, up-gate), RoseTTAFold3 (per-block pair LayerNorm, kq-norm, parallel) and
    Chai-1 (identity AdaLN, no gating query, no output projection, same-conformer mask by ``uid``). The window
    is AF3's slid one or the clipped, OR-masked one (``atom_blocks``).
    """
    if getattr(module, "rope", False):
        return BuildResult(refusal="rotary_not_supported_by_variant")
    attention = module.cross_attention
    first = attention[0]
    first_norm = first.q_adaptive_layernorm
    identity = first_norm.identity_scale
    eps = first_norm.layer_norm.eps
    per_block = module.per_block_pair
    norms = (
        list(module.pair_input_layer_norm)
        if per_block
        else [module.pair_input_layer_norm] * module.num_blocks
    )
    window = {("slide", False): "slide", ("pad", True): "pad"}.get(
        (module.key_window, first.key_masked)
    )
    gated = type(module.transition_block[0]).__name__ == "GatedDiffusionTransition"
    factor = module.transition_block[0].num_intermediate_factor
    if not (
        window is not None
        and not module.chained_key_norm
        and module.window_half is None
        and module.keys_subset == _ATOM_KEYS
        and module.c_query == _ATOM_WIDTH
        and module.c_single_cond == _ATOM_WIDTH
        and module.c_pair_cond == _ATOM_PAIR
        and module.num_head == 4
        and len({(norm.eps, norm.bias is None) for norm in norms}) == 1
        and all(
            block.key_masked == first.key_masked
            and _variant_adaln_plain(block.q_adaptive_layernorm, identity, eps)
            and _variant_adaln_plain(block.k_adaptive_layernorm, identity, eps)
            and block.kq_norm == first.kq_norm
            and block.use_gating_query == first.use_gating_query
            and block.adaptive_zero_init.project == first.adaptive_zero_init.project
            and not block.adaptive_zero_init.raw_gate
            and block.adaptive_zero_init.adaptive_zero_cond.bias is not None
            and block.q_projection.bias is not None
            and (
                not block.kq_norm
                or block.query_layer_norm.eps
                == block.key_layer_norm.eps
                == _DEFAULT_EPS
            )
            for block in attention
        )
        and all(
            _variant_transition_ok(block, identity, eps, gated)
            and block.num_intermediate_factor == factor
            for block in module.transition_block
        )
    ):
        return BuildResult(refusal="stack_structure_mismatch")
    # The block options (identity AdaLN, gating, output projection, kq norm, up-gate, per-block pair norm) follow from the weights.
    stack = AtomDiTVariantStack.from_released(
        module.state_dict(),
        parallel=module.parallel,
        key_window=window,
        implementation=_ENGINE,
        adaln_eps=eps,
        pair_eps=norms[0].eps,
    )
    return BuildResult(value=_finish(stack, like))


def variant_selection(stack: nn.Module, *inputs: torch.Tensor | None) -> str:
    """Which engine implementation runs the variant stack for this call (``stack.serving``): ``cuda_candidate``,
    ``portable_candidate`` or ``pytorch_candidate``.

    The engine owns the support predicates. A call neither the CUDA rows nor the portable path serve is served by the
    engine's PyTorch statements (``kernels/dit_variants/reference.py``: any card, inference and training), so the dense
    FoldForge stack is never the fallback of a built variant stack. A selection is not a kernel trace.
    """
    return f"{stack.serving(*inputs)}_candidate"


def variant_serves(stack: nn.Module, *inputs: torch.Tensor | None) -> bool:
    """Whether the engine serves the call: ``serving()`` is ``cuda``, ``portable`` or ``pytorch`` (the PyTorch statements count)."""
    return stack.serving(*inputs) in ("cuda", "portable", "pytorch")


def _eps_of(adaptive_norm: nn.Module) -> set[float | None]:
    """The epsilons of an AdaLN's norms; a form without them (RMS conditioning) gives ``None``, which is not the default."""
    return {
        getattr(getattr(adaptive_norm, name, None), "eps", None)
        for name in ("layer_norm", "single_cond_layer_norm")
    }


def _rms_adaln(norm: nn.Module) -> bool:
    """ESMFold2's AdaLN: affine-free RMSNorm, SiLU conditioning, (s + 1) scale, no biases."""
    return bool(
        norm.identity_scale
        and getattr(norm, "rms_norm", False)
        and getattr(norm, "activate_cond", False)
        and norm.single_cond_scale.bias is None
    )


def _raw_gate(zero_init: nn.Module) -> bool:
    """ESMFold2's output gate: raw (no sigmoid), no bias, with the output projection."""
    return bool(
        zero_init.raw_gate
        and zero_init.project
        and zero_init.adaptive_zero_cond.bias is None
    )


def _build_esmfold2_atom_blocks(module: nn.Module, like: torch.Tensor) -> BuildResult:
    """ESMFold2's atom stack: one engine ``LocalDiTBlock(**ESMFOLD2)`` per block.

    AF3's block (parallel K/V AdaLN, slid window, atom pair bias, gating query, q bias) with the
    RMS / SiLU AdaLN, raw gates, per-head RMSNorm + rotary q / k and 32 queries x 192 keys under
    the rank-based +/- 64 window. Matches the dense module to fp32 rounding (engine
    ``tests/cpu/compile/test_local_dit_esmfold2_variant.py``).
    """
    attention = module.cross_attention
    heads = module.num_head
    if not (
        module.window_half == _ENGINE_ESMFOLD2["window_half"]
        and module.keys_subset == _ENGINE_ESMFOLD2["keys"]
        and module.key_window == "slide"
        and not module.chained_key_norm
        and not module.parallel
        and not module.per_block_pair
        and module.c_query == _ATOM_WIDTH
        and module.c_single_cond == _ATOM_WIDTH
        and module.c_pair_cond == _ATOM_PAIR
        and module.pair_input_layer_norm.bias is None
        and module.key_dim_per_head == 2 * _ATOM_ROPE_PAIRS
        and all(
            not block.kq_norm
            and not block.key_masked
            and block.use_gating_query
            and block.q_projection.bias is not None
            and _rms_adaln(block.q_adaptive_layernorm)
            and _rms_adaln(block.k_adaptive_layernorm)
            and _raw_gate(block.adaptive_zero_init)
            for block in attention
        )
        and all(
            type(block).__name__ == "DiffusionTransition"
            and _rms_adaln(block.adaptive_layernorm)
            and _raw_gate(block.adaptive_zero_init)
            for block in module.transition_block
        )
    ):
        return BuildResult(refusal="esm_atom_structure_mismatch")
    blocks = []
    for index, (attn, trans) in enumerate(
        zip(attention, module.transition_block, strict=True)
    ):
        rows = module.pair_logits_projection.weight[index * heads : (index + 1) * heads]
        expand_a, expand_b = trans.transition1.weight.chunk(2, dim=0)
        block = LocalDiTBlock(
            _ATOM_WIDTH,
            _ATOM_WIDTH,
            _ATOM_PAIR,
            heads,
            expand_a.shape[0] // _ATOM_WIDTH,
            implementation=_ENGINE,
            **_ENGINE_ESMFOLD2,
        )
        q_ln, k_ln, t_ln = (
            attn.q_adaptive_layernorm,
            attn.k_adaptive_layernorm,
            trans.adaptive_layernorm,
        )
        ones = torch.ones(module.key_dim_per_head)  # the q / k RMSNorm is affine-free
        _copy(
            block,
            {
                "attention.ada_ln_in.to_scale.weight": q_ln.single_cond_scale.weight,
                "attention.ada_ln_in.to_bias.weight": q_ln.single_cond_bias.weight,
                "attention.ada_ln_kv.to_scale.weight": k_ln.single_cond_scale.weight,
                "attention.ada_ln_kv.to_bias.weight": k_ln.single_cond_bias.weight,
                "attention.to_query.weight": attn.q_projection.weight,
                "attention.to_query.bias": attn.q_projection.bias,
                "attention.to_key.weight": attn.k_projection.weight,
                "attention.to_value.weight": attn.v_projection.weight,
                "attention.norm_query.weight": ones,
                "attention.norm_key.weight": ones,
                "attention.ln_pair.weight": module.pair_input_layer_norm.weight,
                "attention.to_bias.weight": rows,
                "attention.to_gate.weight": attn.gating_query.weight,
                "attention.to_out.weight": attn.adaptive_zero_init.transition2.weight,
                "attention.to_scale.weight": attn.adaptive_zero_init.adaptive_zero_cond.weight,
                "transition.ada_ln_in.to_scale.weight": t_ln.single_cond_scale.weight,
                "transition.ada_ln_in.to_bias.weight": t_ln.single_cond_bias.weight,
                "transition.expand_a.weight": expand_a,
                "transition.expand_b.weight": expand_b,
                "transition.squeeze.weight": trans.adaptive_zero_init.transition2.weight,
                "transition.to_scale.weight": trans.adaptive_zero_init.adaptive_zero_cond.weight,
            },
            like,
            keep_master=True,
        )
        blocks.append(block)
    return BuildResult(value=blocks)


def _build_atom_blocks(module: nn.Module, like: torch.Tensor) -> BuildResult:
    """One engine ``LocalDiTBlock`` per block of a dense ``DiffusionCrossAttTransformer``.

    The K/V AdaLN reads the raw atoms (AF3, ``kv_norm="parallel"``) or the first AdaLN's output
    (Protenix, OpenDDE: ``"chained"``); the 128-key window is AF3's slid one or the clipped, OR-masked
    one. Anything else (rotary, kq-norm, parallel schedule, RMS conditioning, other widths, other
    transitions) keeps the reference path.
    """
    if getattr(module, "rope", False):
        return _build_esmfold2_atom_blocks(module, like)
    attention = module.cross_attention
    first = attention[0]
    per_block = module.per_block_pair
    norms = (
        list(module.pair_input_layer_norm)
        if per_block
        else [module.pair_input_layer_norm] * module.num_blocks
    )
    eps_values = (
        _eps_of(first.q_adaptive_layernorm)
        | _eps_of(first.k_adaptive_layernorm)
        | {getattr(norm, "eps", None) for norm in norms}
    )
    # AF3 slides each window inside the real atoms (a padded key never exists, so the AND and OR
    # mask forms agree); the pad window of the OpenFold lineage clips it and masks with the OR form.
    window = {("slide", False): "slide", ("pad", True): "pad"}.get(
        (module.key_window, first.key_masked)
    )
    plain = (
        not module.parallel
        and window is not None
        and module.keys_subset == _ATOM_KEYS
        and module.c_query == _ATOM_WIDTH
        and module.c_single_cond == _ATOM_WIDTH
        and module.c_pair_cond == _ATOM_PAIR
        and eps_values == {_DEFAULT_EPS}
        and all(
            not block.kq_norm
            and block.key_masked == first.key_masked
            and block.use_gating_query
            and block.q_projection.bias is not None
            and not any(
                a.identity_scale
                or getattr(a, "rms_norm", False)
                or getattr(a, "activate_cond", False)
                for a in (block.q_adaptive_layernorm, block.k_adaptive_layernorm)
            )
            for block in attention
        )
        and all(
            type(block).__name__ == "DiffusionTransition"
            and not block.adaptive_layernorm.identity_scale
            for block in module.transition_block
        )
    )
    if not plain:
        return BuildResult(refusal="block_structure_mismatch")
    heads = module.num_head
    kv_norm = "chained" if module.chained_key_norm else "parallel"
    blocks = []
    for index, (attn, trans) in enumerate(
        zip(attention, module.transition_block, strict=True)
    ):
        rows = (
            module.pair_logits_projection[index].weight
            if per_block
            else module.pair_logits_projection.weight[
                index * heads : (index + 1) * heads
            ]
        )
        expand_a, expand_b = trans.transition1.weight.chunk(2, dim=0)
        block = LocalDiTBlock(
            _ATOM_WIDTH,
            _ATOM_WIDTH,
            _ATOM_PAIR,
            heads,
            expand_a.shape[0] // _ATOM_WIDTH,
            cross_attention=True,
            kv_norm=kv_norm,
            key_window=window,
            implementation=_ENGINE,
        )
        q_ln, k_ln, t_ln = (
            attn.q_adaptive_layernorm,
            attn.k_adaptive_layernorm,
            trans.adaptive_layernorm,
        )
        _copy(
            block,
            {
                "attention.ada_ln_in.ln_cond.weight": q_ln.single_cond_layer_norm.weight,
                "attention.ada_ln_in.to_scale.weight": q_ln.single_cond_scale.weight,
                "attention.ada_ln_in.to_scale.bias": q_ln.single_cond_scale.bias,
                "attention.ada_ln_in.to_bias.weight": q_ln.single_cond_bias.weight,
                "attention.ada_ln_kv.ln_cond.weight": k_ln.single_cond_layer_norm.weight,
                "attention.ada_ln_kv.to_scale.weight": k_ln.single_cond_scale.weight,
                "attention.ada_ln_kv.to_scale.bias": k_ln.single_cond_scale.bias,
                "attention.ada_ln_kv.to_bias.weight": k_ln.single_cond_bias.weight,
                "attention.to_query.weight": attn.q_projection.weight,
                "attention.to_query.bias": attn.q_projection.bias,
                "attention.to_key.weight": attn.k_projection.weight,
                "attention.to_value.weight": attn.v_projection.weight,
                "attention.ln_pair.weight": norms[index].weight,
                "attention.to_bias.weight": rows,
                "attention.to_gate.weight": attn.gating_query.weight,
                "attention.to_out.weight": attn.adaptive_zero_init.transition2.weight,
                "attention.to_scale.weight": attn.adaptive_zero_init.adaptive_zero_cond.weight,
                "attention.to_scale.bias": attn.adaptive_zero_init.adaptive_zero_cond.bias,
                "transition.ada_ln_in.ln_cond.weight": t_ln.single_cond_layer_norm.weight,
                "transition.ada_ln_in.to_scale.weight": t_ln.single_cond_scale.weight,
                "transition.ada_ln_in.to_scale.bias": t_ln.single_cond_scale.bias,
                "transition.ada_ln_in.to_bias.weight": t_ln.single_cond_bias.weight,
                "transition.expand_a.weight": expand_a,
                "transition.expand_b.weight": expand_b,
                "transition.squeeze.weight": trans.adaptive_zero_init.transition2.weight,
                "transition.to_scale.weight": trans.adaptive_zero_init.adaptive_zero_cond.weight,
                "transition.to_scale.bias": trans.adaptive_zero_init.adaptive_zero_cond.bias,
            },
            like,
            keep_master=True,
        )
        blocks.append(block)
    return BuildResult(value=blocks)


def _build_single_attention(
    attention: nn.Module, norm: nn.Module, projection: nn.Module, like: torch.Tensor
) -> BuildResult:
    """Engine AttentionPairBias from the Pairformer's single attention (AF3 form).

    The pair LayerNorm offset is dropped: it adds one constant to every key's logit of a head,
    which the softmax over keys cancels exactly.
    """
    if attention.use_single_cond or attention.kq_norm or not attention.use_gating_query:
        return BuildResult(refusal="single_attention_structure_mismatch")
    if (
        attention.q_projection.bias is None
        or attention.qkv_dim * attention.num_head != attention.c_x
    ):
        return BuildResult(refusal="single_attention_structure_mismatch")
    # The released names: the attention at ``single_attention_``, the pair logits' norm / projection beside it.
    state = {f"single_attention_.{k}": v for k, v in attention.state_dict().items()}
    state |= {f"single_pair_logits_norm.{k}": v for k, v in norm.state_dict().items()}
    state |= {f"single_pair_logits_projection.{k}": v for k, v in projection.state_dict().items()}
    layer_norm = attention.adaptive_layernorm.layer_norm
    built = AttentionPairBias.from_released(
        state,
        n_head=attention.num_head,
        prefix="single_attention_",
        dtype=like.dtype,
        norm_eps=layer_norm.eps,
        implementation=_ENGINE,
        # Chai-1: a constant on the gate logits
        **({"gate_bias": attention.gate_bias} if attention.gate_bias else {}),
    )
    built.ln_pair.eps = norm.eps
    return BuildResult(value=_finish(built, like))


#: The ops whose engine module can run the Anthropic payload (``implementation="anthropic"``).
_ANTHROPIC_MODULES = ("trimul", "opm", "msa_pwa")


def gate_anthropic(built: nn.Module, op: str, device: torch.device) -> str | None:
    """Give an engine module built for the Anthropic payload the predicate that says whether a call fits it.

    ``foldforge_refuses(x)`` is the engine's own non-raising ``refusal`` for the call (None when the payload serves it);
    ``routing.served_by`` asks it before every call, so a shape the payload does not take runs the PyTorch reference
    (one log line, or an error under ``engine.unsupported: error``) instead of raising inside the engine module.
    Returns the answer for a small probe call on ``device`` (the machine and the module's widths, not the sequence length):
    why this module can never be served, or None.
    """
    from miniworld_engine.baselines import anthropic_msa, anthropic_trimul

    built.foldforge_op = op
    grad = torch.is_grad_enabled
    if op == "trimul":
        h = built.d_hidden
        built.foldforge_plane_only = True

        def refuses(x: torch.Tensor) -> str | None:
            return anthropic_trimul.refusal(
                x if x.ndim == _PAIR_BATCHED_RANK else x[None],
                x.shape[-1],
                h,
                grad=grad(),
                dropout=False,
            )

    elif op == "opm":
        dims = (built.to_left.weight.shape[0], built.to_out.weight.shape[0])

        def refuses(x: torch.Tensor) -> str | None:
            if not built.normalize_before_proj:
                return "the release's epilogue adds the output bias after the division by the pair count; this stack adds it before"
            return anthropic_msa.opm_refusal(
                x[None], *dims, grad=grad(), interchain=False
            )

    else:
        dims = (
            built.to_value.weight.shape[1],
            built.to_bias.weight.shape[1],
            built.n_head,
            built.to_value.weight.shape[0] // built.n_head,
        )

        def refuses(x: torch.Tensor) -> str | None:
            return anthropic_msa.pwa_refusal(x[None], *dims, grad=grad(), dropout=False)

    built.foldforge_refuses = refuses
    if hasattr(built, "foldforge_left_weight"):
        # RoseTTAFold3's length-divided contraction scales a weight copy on every call; the payload packs its weights
        # once per version, so each call would repack (and, in a captured graph, allocate) them again
        return "the contraction is divided by the real sequence length (RoseTTAFold3), a per-call weight scale the payload cannot take"
    channels = (built.ln_pair if op == "trimul" else built.ln_msa).weight.shape[0]
    shape = (16, 16, channels) if op == "trimul" else (8, 16, channels)
    with torch.no_grad():
        return refuses(torch.empty(shape, dtype=torch.bfloat16, device=device))


_PAIR_BATCHED_RANK = 4


def _installation_candidates() -> dict[str, tuple[str, tuple[Candidate, ...]]]:
    """One ordered candidate chain per dense module shape, independent of family names."""
    return {
        "TriangleMultiplication": (
            "trimul",
            (Candidate("engine_trimul", _build_triangle_multiplication),),
        ),
        "GridSelfAttention": (
            "triangle_attention",
            (
                Candidate(
                    "engine_triangle_attention",
                    lambda m, l, _i: _build_triangle_attention(m, l),
                ),
            ),
        ),
        "MSAAttention": (
            "msa_pwa",
            (Candidate("engine_msa_pwa", _build_pair_weighted_averaging),),
        ),
        "OuterProductMean": (
            "opm",
            (Candidate("engine_opm", _build_outer_product_mean),),
        ),
        "Transition": (
            "transition",
            (Candidate("engine_trunk_transition", _build_transition),),
        ),
        "_SingleAttention": (
            "single_attention",
            (
                Candidate(
                    "engine_single_attention",
                    lambda m, l, _i: _build_single_attention(
                        m.attention, m.norm, m.projection, l
                    ),
                ),
            ),
        ),
        "DiffusionTransformer": (
            "dit_blocks",
            (
                Candidate(
                    "engine_dit_blocks",
                    lambda m, l, _i: _build_diffusion_blocks(m, l),
                    conditioning_width=_TOKEN_DIT_COND,
                ),
                Candidate(
                    "engine_dit_variant_blocks",
                    lambda m, l, _i: _build_diffusion_variant_stack(m, l),
                    fp32=False,
                ),
            ),
        ),
        "DiffusionCrossAttTransformer": (
            "atom_blocks",
            (
                Candidate(
                    "engine_atom_blocks", lambda m, l, _i: _build_atom_blocks(m, l)
                ),
                Candidate(
                    "engine_atom_variant_blocks",
                    lambda m, l, _i: _build_atom_variant_stack(m, l),
                    fp32=False,
                ),
            ),
        ),
    }


def install_engine_modules(  # noqa: PLR0915 - preserve legacy counters alongside explicit install decisions
    model: nn.Module,
    like: torch.Tensor,
    skip: Collection[str] = (),
    plan: BackendPlan | None = None,
) -> dict[str, int]:
    """Attach matching engine candidates; return compatible legacy counters.

    ``installation_report(model)`` records every candidate decision and actual
    attachment. Legacy transformer counters also count constructed candidates
    skipped by ``engine.skip``; use the new report's ``attached_units`` instead.

    ``like`` carries the model's dtype and device; ``skip`` (``engine.skip`` of the config) names the ``engine_*`` count keys
    whose modules stay dense. ``plan`` (``backends.BackendPlan``) says which backend each op runs on: an op whose backend
    is neither ``miniworld`` nor, for the ops with an Anthropic engine module, ``anthropic`` gets no engine module and
    keeps the dense reference path (``None``: every op on ``miniworld``, the default). An FP32 model (``precision: fp32``) gets the diffusion
    transformers only: they are the modules with an FP32 engine kernel (TF32 tensor cores) on H100. The
    pair stack's modules (TriMul, triangle attention, outer product mean, pair weighted averaging, pair
    bias attention) have BF16 kernels only, and the variant stacks serve BF16; attaching them to an FP32
    model would copy weights nothing reads.
    """
    fp32_model = like.dtype == torch.float32
    counts = {
        "engine_trimul": 0,
        "engine_triangle_attention": 0,
        "engine_msa_pwa": 0,
        "engine_opm": 0,
        "engine_trunk_transition": 0,
        "engine_dit_blocks": 0,
        "engine_single_attention": 0,
        "engine_atom_blocks": 0,
        "engine_dit_variant_blocks": 0,
        "engine_atom_variant_blocks": 0,
    }
    skip = set(skip)
    catalog = _installation_candidates()
    decisions: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "enabled": True,
        "coverage": "engine_module_candidates",
        "is_runtime_dispatch": False,
        "modules": decisions,
    }
    record_installation(model, report)

    def implementation(op: str) -> ImplementationType | None:
        """The engine implementation that builds ``op``'s module, None when the op's backend has none."""
        backend = "miniworld" if plan is None else plan.of(op)
        if backend == "miniworld":
            return _ENGINE
        if backend == "anthropic" and op in _ANTHROPIC_MODULES:
            return ImplementationType.ANTHROPIC
        return None

    for path, module in model.named_modules():
        name = type(module).__name__
        if name not in catalog or (
            (name == "OuterProductMean" and not hasattr(module, "left_projection"))
            or (name == "Transition" and not hasattr(module, "transition1"))
        ):
            continue
        op, candidates = catalog[name]
        backend = "miniworld" if plan is None else plan.of(op)
        row: dict[str, Any] = {
            "op": op,
            "backend": backend,
            "requested_backend": "miniworld" if plan is None else plan.requested(op),
            "status": "unsupported",
            "reason": None,
            "attempts": [],
            "attached_units": 0,
            "attached": False,
            "preexisting_twin": engine_of(module) is not None,
        }
        decisions[path or "<root>"] = row
        if fp32_model and name not in _FP32_TWIN_CLASSES:
            row.update(status="precision_ineligible", reason="bf16_twin_only")
            continue
        impl = implementation(op)
        if impl is None:
            row.update(status="backend_without_twin", reason="backend_uses_other_route")
            continue
        for candidate in candidates:
            result = candidate.build(module, like, impl)
            built = result.value
            row["attempts"].append(
                {
                    "candidate": candidate.key,
                    "status": "built" if built is not None else "refused",
                    "reason": result.refusal,
                }
            )
            if built is None:
                row["reason"] = result.refusal
                continue
            key = candidate.key
            transformer = name in _FP32_ENGINE_CLASSES
            units = (
                (
                    len(built)
                    if isinstance(built, list)
                    else cast("int", module.num_blocks)
                )
                if transformer
                else 1
            )
            # Compatibility: old transformer counters include built-but-skipped candidates.
            if transformer:
                counts[key] += units if built else 0
            row["built_candidate"] = key
            if key in skip:
                row.update(status="skipped", reason="engine_skip")
                break
            if transformer and not built:
                row.update(status="empty_candidate", reason="no_blocks")
                break
            if getattr(built, "implementation", None) == ImplementationType.ANTHROPIC:
                never = gate_anthropic(cast("nn.Module", built), op, like.device)
                # the payload can never serve this module (the machine, its widths): the reference, stated now
                if never is not None:
                    row.update(status="backend_refused", reason=never)
                    call_refusal(
                        getattr(plan, "unsupported", "fallback"), op, "anthropic", never
                    )
                    counts["anthropic_refused"] = counts.get("anthropic_refused", 0) + 1
                    break
            attach(module, built)
            if transformer and isinstance(built, list):
                module.foldforge_engine_fp32 = all(
                    p.dtype == torch.float32 for p in built[0].parameters()
                )
            if not transformer:
                counts[key] += 1
            row.update(
                status="attached", reason=None, attached=True, attached_units=units
            )
            break
    for path, module in model.named_modules():
        if (path or "<root>") in decisions:
            decisions[path or "<root>"]["present_after_install"] = (
                engine_of(module) is not None
            )
    report["legacy_counts"] = dict(counts)
    report["attached_modules"] = sum(row["attached"] for row in decisions.values())
    report["attached_units"] = sum(row["attached_units"] for row in decisions.values())
    return counts


def clear_engine_weight_caches() -> None:
    """Drop the engine's eager weight re-layout cache before weights are replaced.

    The H100 Transition kernels keep ``w.t().contiguous()`` keyed by ``(address, version, shape,
    dtype)`` with no liveness check, so a new weight allocated at the address of a freed one with
    the same version is served the old copy (silently wrong output). A process that builds a second
    model, or a test that builds many, clears it first.
    """
    from miniworld_engine.modules import runtime

    runtime.clear_weight_caches()
