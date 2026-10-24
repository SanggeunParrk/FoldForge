"""The engine's own ``nn.Module`` ops, built from the released weights of dense modules.

The engine keeps its hand-CUDA (H100 / B200) legs in ``miniworld_engine.modules.*``; the whole-op
entry points in ``miniworld_engine.ops`` reach only the portable Triton legs. So a dense module that
can be expressed as an engine module gets one at install time, fed from the same tensors:

===============================  ===========================================================
dense module                     engine module
===============================  ===========================================================
``TriangleMultiplication``       ``TriangleMultiplication``   (residual included)
``GridSelfAttention``            ``TriangleAttention``        (residual included)
``MSAAttention``                 ``MSAPairWeightedAveraging`` (residual included)
``OuterProductMean``             ``OuterProductMean``         (raw update)
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

import inspect
from typing import TYPE_CHECKING

import torch
from miniworld_engine.modules import (
    AttentionPairBias,
    MSAPairWeightedAveraging,
    OuterProductMean,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.dit.module import DiTBlock
from miniworld_engine.modules.exceptions import ImplementationType
from miniworld_engine.modules.local_dit import LocalDiTBlock

if TYPE_CHECKING:
    from collections.abc import Collection

    from torch import nn

_ENGINE = ImplementationType.MINIWORLD
#: The engine's ``TriangleAttention`` takes the ending node's bias orientation (``transposed_bias``).
_ENGINE_HAS_BIAS_ORIENTATION = (
    "transposed_bias" in inspect.signature(TriangleAttention.__init__).parameters
)
#: The engine's ``TriangleAttention`` takes RoseTTAFold3's biases and Chai-1's output orientations (2.3.8).
_ENGINE_HAS_TRIATTN_VARIANTS = (
    "dual_output" in inspect.signature(TriangleAttention.__init__).parameters
)
#: The engine's ``OuterProductMean`` takes RoseTTAFold3's projection biases and Chai-1's groups (2.3.8).
_ENGINE_HAS_OPM_VARIANTS = (
    "groups" in inspect.signature(OuterProductMean.__init__).parameters
)
#: The engine's ``AttentionPairBias`` takes Chai-1's constant gate bias (2.3.8).
_ENGINE_HAS_APB_GATE_BIAS = (
    "gate_bias" in inspect.signature(AttentionPairBias.__init__).parameters
)
#: The engine's ``LocalDiTBlock`` takes AlphaFold 3's K/V norm and key window (``kv_norm`` / ``key_window``).
_ENGINE_HAS_ATOM_VARIANT = (
    "kv_norm" in inspect.signature(LocalDiTBlock.__init__).parameters
)
_ATTR = "_engine_module"
_TOKEN_DIT_COND = 384
#: The dense classes with an FP32 engine kernel on H100 (TF32 tensor cores): the diffusion transformers.
_FP32_ENGINE_CLASSES = ("DiffusionTransformer", "DiffusionCrossAttTransformer")
_DEFAULT_EPS = 1e-5
_ATOM_WIDTH, _ATOM_PAIR, _ATOM_KEYS = 128, 16, 128
_ATOM_ROPE_PAIRS = 16  # rotary pairs per 32-channel atom head (3 x 2 spatial + 10 uid)
try:  # the engine's Boltz-2 / RoseTTAFold3 / Chai-1 token and atom stacks (2.3.8)
    from miniworld_engine.modules.dit_variant import (
        AtomDiTVariantStack,
        DiTVariantStack,
    )
except ImportError:
    AtomDiTVariantStack = DiTVariantStack = None
try:  # the engine's ESMFold2 atom block options (RMS / SiLU AdaLN, rotary q / k, 192-key window)
    from miniworld_engine.modules.local_dit import ESMFOLD2 as _ENGINE_ESMFOLD2
except ImportError:
    _ENGINE_ESMFOLD2 = None


def attach(module: nn.Module, engine_module: object) -> None:
    """Attach out of band: not a registered child, so no state_dict key and no ``.to`` move."""
    object.__setattr__(module, _ATTR, engine_module)


def engine_of(module: nn.Module) -> object | None:
    """The engine module attached to ``module``, if any."""
    return module.__dict__.get(_ATTR)


def residue_mask(pair_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Token mask ``[1, L]`` (bool) from an outer-product pair mask ``[L, L]``."""
    if pair_mask is None:
        return None
    return (pair_mask.diagonal() > 0)[None].contiguous()


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


def _set_eps(engine_module: nn.Module, eps: float) -> None:
    for sub in engine_module.modules():
        if hasattr(sub, "normalized_shape") and hasattr(sub, "eps"):
            sub.eps = eps


def triangle_multiplication(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> nn.Module | None:
    """Engine TriMul from a dense ``TriangleMultiplication`` (None when it cannot serve it)."""
    d = module.c_pair
    # The hidden width may differ from the pair width (template stacks: pair 64,
    # hidden 128); the engine takes it as d_hidden.
    h = module.center_norm.weight.shape[0]
    if module.left_norm_input.eps != module.center_norm.eps:
        return None
    outgoing = module.equation == "cik,cjk->cij"
    # The released projection interleaves a,b per hidden channel; the engine takes them as two
    # matrices. Incoming contracts b[k,i] * a[k,j] where the engine contracts a[k,i] * b[k,j].
    projection = module.projection.weight.reshape(h, 2, d).transpose(0, 1)
    gate = module.gate.weight.reshape(h, 2, d).transpose(0, 1)
    left, right = (0, 1) if outgoing else (1, 0)
    built = TriangleMultiplication(
        d, d_hidden=h, outgoing=outgoing, implementation=implementation, p_drop=0.0
    )
    _copy(
        built,
        {
            "ln_pair.weight": module.left_norm_input.weight,
            "ln_pair.bias": module.left_norm_input.bias,
            "to_left.weight": projection[left],
            "to_right.weight": projection[right],
            "to_left_gate.weight": gate[left],
            "to_right_gate.weight": gate[right],
            "ln_out.weight": module.center_norm.weight,
            "ln_out.bias": module.center_norm.bias,
            "to_out.weight": module.output_projection.weight,
            "to_gate.weight": module.gating_linear.weight,
        },
        like,
    )
    _set_eps(built, module.left_norm_input.eps)
    if getattr(module, "divide_by_length", False):
        # RoseTTAFold3 divides the contraction by the real length before the centre norm. The
        # left operand is linear in its projection, so the call scales a copy of that weight by
        # 1 / length on the device (``routing.triangle_residual``) and the kernels run unchanged.
        built.foldforge_left_weight = built.to_left.weight.detach().clone()
    return built


def triangle_attention(module: nn.Module, like: torch.Tensor) -> nn.Module | None:
    """Engine TriangleAttention from a dense ``GridSelfAttention``.

    The ending node takes its pair bias from the transposed pair (``transposed_bias``) or, as AF3's
    released code does, from the pair as it is. An engine without the ``transposed_bias`` option
    knows only the first, so a module on the second is left on the reference path there. The
    released variants map one to one: RoseTTAFold3's gate / out-projection biases (``gate_bias`` /
    ``out_bias``), Chai-1's ending node returned as the transposed pair has it
    (``untransposed_output``) and its confidence head's second out-projection (``dual_output``).
    """
    hidden = module.num_head * module.qkv_dim
    ending_as_is = module.transpose and not module.transposed_bias
    if ending_as_is and not _ENGINE_HAS_BIAS_ORIENTATION:
        return None
    variant = {
        "gate_bias": module.gating_query.bias is not None,
        "out_bias": module.output_projection.bias is not None,
        "untransposed_output": module.untransposed_output,
        "dual_output": module.dual_output,
    }
    if any(variant.values()) and not _ENGINE_HAS_TRIATTN_VARIANTS:
        return None
    orientation = (
        {"transposed_bias": not ending_as_is} if _ENGINE_HAS_BIAS_ORIENTATION else {}
    )
    # The heads may span more than the pair width (template stacks: pair 64,
    # 4 x 32 heads); the engine takes it as d_hidden.
    built = TriangleAttention(
        module.c_pair,
        module.num_head,
        d_hidden=hidden,
        starting=not module.transpose,
        implementation=_ENGINE,
        p_drop=0.0,
        **orientation,
        **({k: v for k, v in variant.items() if v}),
    )
    sources = {
        "ln_pair.weight": module.act_norm.weight,
        "ln_pair.bias": module.act_norm.bias,
        "to_query.weight": module.q_projection.weight,
        "to_key.weight": module.k_projection.weight,
        "to_value.weight": module.v_projection.weight,
        "to_bias.weight": module.pair_bias_projection.weight,
        "to_gate.weight": module.gating_query.weight,
        "to_gate.bias": module.gating_query.bias,
        "to_out.weight": module.output_projection.weight,
        "to_out.bias": module.output_projection.bias,
    }
    if module.dual_output:
        sources["to_out_t.weight"] = module.output_projection_transposed.weight
        sources["to_out_t.bias"] = module.output_projection_transposed.bias
    _copy(built, sources, like)
    _set_eps(built, module.act_norm.eps)
    return built


def pair_weighted_averaging(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> nn.Module:
    """Engine MSAPairWeightedAveraging from a dense ``MSAAttention``.

    The engine drops the pair LayerNorm offset: it adds one constant to every key's logit of a
    head, which the softmax over keys cancels exactly.
    """
    built = MSAPairWeightedAveraging(
        module.c_msa,
        module.c_pair,
        module.num_head,
        module.value_dim,
        implementation=implementation,
        p_drop=0.0,
    )
    _copy(
        built,
        {
            "ln_msa.weight": module.act_norm.weight,
            "ln_msa.bias": module.act_norm.bias,
            "to_value.weight": module.v_projection.weight,
            "ln_pair.weight": module.pair_norm.weight,
            "to_bias.weight": module.pair_logits.weight,
            "to_gate.weight": module.gating_query.weight,
            "to_out.weight": module.output_projection.weight,
        },
        like,
    )
    _set_eps(built, module.act_norm.eps)
    built.ln_pair.eps = module.pair_norm.eps
    return built


def outer_product_mean(
    module: nn.Module, like: torch.Tensor, implementation: ImplementationType = _ENGINE
) -> nn.Module | None:
    """Engine OuterProductMean from a dense ``OuterProductMean``.

    The engine always divides by ``clamp(count, 1)``; the released form divides by ``count + 1e-3``.
    The two differ by 0.1% at a single row and not at all beyond bf16 resolution after that; they
    differ at pairs with no valid row (padding), which the graph masks. ``bias_after_norm`` is the
    engine's ``normalize_before_proj``. RoseTTAFold3's projection biases and Chai-1's groups (G
    outer products of K-wide halves, LayerNorm'd, no count division) map to the engine's
    ``projection_bias`` / ``groups``.
    """
    groups = module.groups
    projection_bias = module.left_projection.bias is not None
    if (groups != 1 or projection_bias) and not _ENGINE_HAS_OPM_VARIANTS:
        return None
    channels = module.num_outer_channel
    variant = (
        {"projection_bias": projection_bias, "groups": groups}
        if _ENGINE_HAS_OPM_VARIANTS
        else {}
    )
    built = OuterProductMean(
        module.c_msa,
        module.num_output_channel,
        groups * channels,
        normalize_before_proj=module.bias_after_norm,
        implementation=implementation,
        **variant,
    )
    sources = {
        "ln_msa.weight": module.layer_norm_input.weight,
        "ln_msa.bias": module.layer_norm_input.bias,
        "to_left.weight": module.left_projection.weight,
        "to_left.bias": module.left_projection.bias,
        "to_right.weight": module.right_projection.weight,
        "to_right.bias": module.right_projection.bias,
        "to_out.weight": module.output_w.reshape(
            groups * channels * channels, module.num_output_channel
        ).T,
        "to_out.bias": module.output_b,
    }
    if groups > 1:
        sources["product_norm.weight"] = module.product_norm.weight
        sources["product_norm.bias"] = module.product_norm.bias
    _copy(built, sources, like)
    _set_eps(built, module.layer_norm_input.eps)
    if groups > 1:
        built.product_norm.eps = module.product_norm.eps
    return built


def diffusion_blocks(module: nn.Module, like: torch.Tensor) -> list[nn.Module] | None:
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
        return None
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
    return blocks


def _variant_adaln(norm: nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    """The engine ``VariantAdaLN`` tensors of a dense ``AdaptiveLayerNorm`` (sigmoid form or identity form)."""
    sources = {
        f"{prefix}.to_scale.weight": norm.single_cond_scale.weight,
        f"{prefix}.to_shift.weight": norm.single_cond_bias.weight,
    }
    if not norm.identity_scale:
        sources[f"{prefix}.cond_norm_weight"] = norm.single_cond_layer_norm.weight
        sources[f"{prefix}.to_scale.bias"] = norm.single_cond_scale.bias
    return sources


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


def _variant_transition_sources(
    trans: nn.Module, prefix: str
) -> dict[str, torch.Tensor]:
    """Engine block tensors of a dense (Gated)DiffusionTransition: ``expand`` rows are [gate | value]."""
    sources = {
        f"{prefix}.expand.weight": trans.transition1.weight,
        f"{prefix}.squeeze.weight": trans.adaptive_zero_init.transition2.weight,
        f"{prefix}.trans_zero.weight": trans.adaptive_zero_init.adaptive_zero_cond.weight,
        f"{prefix}.trans_zero.bias": trans.adaptive_zero_init.adaptive_zero_cond.bias,
        **_variant_adaln(trans.adaptive_layernorm, f"{prefix}.trans_adaln"),
    }
    if hasattr(trans, "a_to_b"):
        sources[f"{prefix}.up.weight"] = trans.a_to_b.weight
    return sources


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


def diffusion_variant_stack(module: nn.Module, like: torch.Tensor) -> nn.Module | None:
    """The engine's ``DiTVariantStack`` for a dense ``DiffusionTransformer`` the per-block ``DiTBlock`` cannot serve.

    Boltz-2 (sequential, up-gate transition), RoseTTAFold3 (parallel, kq-norm) and Chai-1 (parallel, identity
    AdaLN, no gating query, affine pair LayerNorm): the engine owns the whole stack, residuals included.
    The pair LayerNorm offset is dropped by the engine's bias (a per-head constant on the logits).
    """
    if DiTVariantStack is None:
        return None
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
        return None
    options = {
        "parallel": module.parallel,
        "identity_scale": identity,
        "gating": first.use_gating_query,
        "kq_norm": first.kq_norm,
        "up_gate": gated,
        "pair_norm_bias": norms[0].bias is not None,
        "adaln_eps": eps,
        "pair_eps": norms[0].eps,
    }
    with torch.device(like.device):  # the default initialiser is cheap on the GPU
        stack = DiTVariantStack(
            module.num_blocks,
            module.c_act,
            module.c_single_cond,
            module.c_pair_cond,
            heads,
            factor,
            implementation=_ENGINE,
            **options,
        )
    sources: dict[str, torch.Tensor] = {}
    for index, (attn, trans) in enumerate(
        zip(attention, module.transition_block, strict=True)
    ):
        prefix = f"blocks.{index}"
        zero = attn.adaptive_zero_init
        if per_block:
            rows = module.pair_logits_projection[index].weight
        else:
            super_block, offset = divmod(index, module.super_block_size)
            rows = module.pair_logits_projection[super_block].weight[
                offset * heads : (offset + 1) * heads
            ]
        sources |= {
            f"{prefix}.to_q.weight": attn.q_projection.weight,
            f"{prefix}.to_q.bias": attn.q_projection.bias,
            f"{prefix}.to_k.weight": attn.k_projection.weight,
            f"{prefix}.to_v.weight": attn.v_projection.weight,
            f"{prefix}.to_out.weight": zero.transition2.weight,
            f"{prefix}.attn_zero.weight": zero.adaptive_zero_cond.weight,
            f"{prefix}.attn_zero.bias": zero.adaptive_zero_cond.bias,
            f"{prefix}.pair_norm.weight": norms[index].weight,
            f"{prefix}.to_pair_bias.weight": rows,
            **_variant_adaln(attn.adaptive_layernorm, f"{prefix}.attn_adaln"),
            **_variant_transition_sources(trans, prefix),
        }
        if norms[index].bias is not None:
            sources[f"{prefix}.pair_norm.bias"] = norms[index].bias
        if attn.use_gating_query:
            sources[f"{prefix}.to_g.weight"] = attn.gating_query.weight
        if attn.kq_norm:
            sources |= {
                f"{prefix}.q_norm.weight": attn.query_layer_norm.weight,
                f"{prefix}.q_norm.bias": attn.query_layer_norm.bias,
                f"{prefix}.k_norm.weight": attn.key_layer_norm.weight,
                f"{prefix}.k_norm.bias": attn.key_layer_norm.bias,
            }
    return _copy(stack, sources, like, keep_master=True)


def atom_variant_stack(module: nn.Module, like: torch.Tensor) -> nn.Module | None:
    """The engine's ``AtomDiTVariantStack`` for a dense ``DiffusionCrossAttTransformer`` the ``LocalDiTBlock`` cannot serve.

    Boltz-2 (shared pair projection, up-gate), RoseTTAFold3 (per-block pair LayerNorm, kq-norm, parallel) and
    Chai-1 (identity AdaLN, no gating query, no output projection, same-conformer mask by ``uid``). The window
    is AF3's slid one or the clipped, OR-masked one (``atom_blocks``).
    """
    if AtomDiTVariantStack is None or getattr(module, "rope", False):
        return None
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
        return None
    with torch.device(like.device):
        stack = AtomDiTVariantStack(
            module.num_blocks,
            _ATOM_WIDTH,
            _ATOM_WIDTH,
            _ATOM_PAIR,
            factor,
            parallel=module.parallel,
            per_block_pair_norm=per_block,
            pair_norm_bias=norms[0].bias is not None,
            key_window=window,
            implementation=_ENGINE,
            identity_scale=identity,
            gating=first.use_gating_query,
            project_output=first.adaptive_zero_init.project,
            kq_norm=first.kq_norm,
            up_gate=gated,
            adaln_eps=eps,
            pair_eps=norms[0].eps,
        )
    sources: dict[str, torch.Tensor] = {}
    if not per_block:
        sources |= {
            "pair_norm.weight": norms[0].weight,
            "to_pair_bias.weight": module.pair_logits_projection.weight,
        }
        if norms[0].bias is not None:
            sources["pair_norm.bias"] = norms[0].bias
    for index, (attn, trans) in enumerate(
        zip(attention, module.transition_block, strict=True)
    ):
        prefix = f"blocks.{index}"
        zero = attn.adaptive_zero_init
        sources |= {
            f"{prefix}.to_q.weight": attn.q_projection.weight,
            f"{prefix}.to_q.bias": attn.q_projection.bias,
            f"{prefix}.to_k.weight": attn.k_projection.weight,
            f"{prefix}.to_v.weight": attn.v_projection.weight,
            f"{prefix}.attn_zero.weight": zero.adaptive_zero_cond.weight,
            f"{prefix}.attn_zero.bias": zero.adaptive_zero_cond.bias,
            **_variant_adaln(attn.q_adaptive_layernorm, f"{prefix}.q_adaln"),
            **_variant_adaln(attn.k_adaptive_layernorm, f"{prefix}.k_adaln"),
            **_variant_transition_sources(trans, prefix),
        }
        if per_block:
            sources[f"{prefix}.pair_norm.weight"] = norms[index].weight
            sources[f"{prefix}.to_pair_bias.weight"] = module.pair_logits_projection[
                index
            ].weight
            if norms[index].bias is not None:
                sources[f"{prefix}.pair_norm.bias"] = norms[index].bias
        if attn.use_gating_query:
            sources[f"{prefix}.to_g.weight"] = attn.gating_query.weight
        if zero.project:
            sources[f"{prefix}.to_out.weight"] = zero.transition2.weight
        if attn.kq_norm:
            sources |= {
                f"{prefix}.q_norm.weight": attn.query_layer_norm.weight,
                f"{prefix}.q_norm.bias": attn.query_layer_norm.bias,
                f"{prefix}.k_norm.weight": attn.key_layer_norm.weight,
                f"{prefix}.k_norm.bias": attn.key_layer_norm.bias,
            }
    return _copy(stack, sources, like, keep_master=True)


def variant_serves(stack: nn.Module, *inputs: torch.Tensor | None) -> bool:
    """Whether the engine's H100 inference kernels take this call; if not, the caller keeps the dense path."""
    from miniworld_engine.kernels.dit_variants import interface

    serves = (
        interface.atom_serves
        if isinstance(stack, AtomDiTVariantStack)
        else interface.serves
    )
    return bool(serves(stack, *inputs))


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


def _esmfold2_atom_blocks(
    module: nn.Module, like: torch.Tensor
) -> list[nn.Module] | None:
    """ESMFold2's atom stack: one engine ``LocalDiTBlock(**ESMFOLD2)`` per block.

    AF3's block (parallel K/V AdaLN, slid window, atom pair bias, gating query, q bias) with the
    RMS / SiLU AdaLN, raw gates, per-head RMSNorm + rotary q / k and 32 queries x 192 keys under
    the rank-based +/- 64 window. Matches the dense module to fp32 rounding (engine
    ``tests/cpu/compile/test_local_dit_esmfold2_variant.py``).
    """
    if _ENGINE_ESMFOLD2 is None:
        return None
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
        return None
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
    return blocks


def atom_blocks(module: nn.Module, like: torch.Tensor) -> list[nn.Module] | None:
    """One engine ``LocalDiTBlock`` per block of a dense ``DiffusionCrossAttTransformer``.

    The K/V AdaLN reads the raw atoms (AF3, ``kv_norm="parallel"``) or the first AdaLN's output
    (Protenix, OpenDDE: ``"chained"``); the 128-key window is AF3's slid one or the clipped, OR-masked
    one. Anything else (rotary, kq-norm, parallel schedule, RMS conditioning, other widths, other
    transitions) keeps the reference path.
    """
    if getattr(module, "rope", False):
        return _esmfold2_atom_blocks(module, like)
    if not _ENGINE_HAS_ATOM_VARIANT:
        return None
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
        return None
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
    return blocks


def single_attention(
    attention: nn.Module, norm: nn.Module, projection: nn.Module, like: torch.Tensor
) -> nn.Module | None:
    """Engine AttentionPairBias from the Pairformer's single attention (AF3 form).

    The pair LayerNorm offset is dropped: it adds one constant to every key's logit of a head,
    which the softmax over keys cancels exactly.
    """
    if (
        attention.use_single_cond
        or attention.kq_norm
        or not attention.use_gating_query
        or (attention.gate_bias and not _ENGINE_HAS_APB_GATE_BIAS)
        or attention.q_projection.bias is None
        or attention.qkv_dim * attention.num_head != attention.c_x
    ):
        return None
    built = AttentionPairBias(
        attention.c_x,
        norm.weight.shape[0],
        attention.num_head,
        implementation=_ENGINE,
        # Chai-1: a constant on the gate logits
        **({"gate_bias": attention.gate_bias} if attention.gate_bias else {}),
    )
    layer_norm = attention.adaptive_layernorm.layer_norm
    _copy(
        built,
        {
            "ln_single.weight": layer_norm.weight,
            "ln_single.bias": layer_norm.bias,
            "to_query.weight": attention.q_projection.weight,
            "to_query.bias": attention.q_projection.bias,
            "to_key.weight": attention.k_projection.weight,
            "to_value.weight": attention.v_projection.weight,
            "ln_pair.weight": norm.weight,
            "to_bias.weight": projection.weight,
            "to_gate.weight": attention.gating_query.weight,
            "to_out.weight": attention.adaptive_zero_init.transition2.weight,
        },
        like,
    )
    built.ln_single.eps = layer_norm.eps
    built.ln_pair.eps = norm.eps
    return built


#: The ops whose engine module can run the Anthropic payload (``implementation="anthropic"``).
_ANTHROPIC_MODULES = ("trimul", "opm", "msa_pwa")


def gate_anthropic(built: nn.Module, op: str) -> None:
    """Give an engine module built for the Anthropic payload the predicate that says whether a call fits it.

    ``foldforge_refuses(x)`` is the engine's own non-raising ``refusal`` for the call (None when the payload serves it);
    ``routing.served_by`` asks it before every call, so a shape the payload does not take runs the PyTorch reference
    (one log line, or an error under ``engine.unsupported: error``) instead of raising inside the engine module.
    """
    from miniworld_engine.integrations import anthropic_msa, anthropic_trimul

    built.foldforge_op = op
    grad = torch.is_grad_enabled
    if op == "trimul":
        h = built.d_hidden
        built.foldforge_plane_only = True

        def refuses(x: torch.Tensor) -> str | None:
            return anthropic_trimul.refusal(
                x if x.ndim == _PAIR_BATCHED_RANK else x[None], x.shape[-1], h, grad=grad(), dropout=False
            )

    elif op == "opm":
        dims = (built.to_left.weight.shape[0], built.to_out.weight.shape[0])

        def refuses(x: torch.Tensor) -> str | None:
            if not built.normalize_before_proj:
                return "the release's epilogue adds the output bias after the division by the pair count; this stack adds it before"
            return anthropic_msa.opm_refusal(x[None], *dims, grad=grad(), interchain=False)

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


_PAIR_BATCHED_RANK = 4


def install_engine_modules(  # noqa: PLR0915 - one flat table of module classes
    model: nn.Module,
    like: torch.Tensor,
    skip: Collection[str] = (),
    plan: object | None = None,
) -> dict[str, int]:
    """Attach an engine module to every dense module that has an exact one; return the counts.

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
        "engine_dit_blocks": 0,
        "engine_single_attention": 0,
        "engine_atom_blocks": 0,
        "engine_dit_variant_blocks": 0,
        "engine_atom_variant_blocks": 0,
    }
    skip = set(skip)

    def implementation(op: str) -> ImplementationType | None:
        """The engine implementation that builds ``op``'s module, None when the op's backend has none."""
        backend = "miniworld" if plan is None else plan.of(op)
        if backend == "miniworld":
            return _ENGINE
        if backend == "anthropic" and op in _ANTHROPIC_MODULES:
            return ImplementationType.ANTHROPIC
        return None

    for module in model.modules():
        name = type(module).__name__
        built: object | None = None
        key = ""
        if fp32_model and name not in _FP32_ENGINE_CLASSES:
            continue
        if name == "TriangleMultiplication":
            impl = implementation("trimul")
            built = None if impl is None else triangle_multiplication(module, like, impl)
            key = "engine_trimul"
        elif name == "GridSelfAttention":
            miniworld = implementation("triangle_attention") is not None
            built = triangle_attention(module, like) if miniworld else None
            key = "engine_triangle_attention"
        elif name == "MSAAttention":
            impl = implementation("msa_pwa")
            built = None if impl is None else pair_weighted_averaging(module, like, impl)
            key = "engine_msa_pwa"
        elif name == "OuterProductMean" and hasattr(module, "left_projection"):
            impl = implementation("opm")
            built = None if impl is None else outer_product_mean(module, like, impl)
            key = "engine_opm"
        elif name == "_SingleAttention":
            built = (
                single_attention(module.attention, module.norm, module.projection, like)
                if implementation("single_attention") is not None
                else None
            )
            key = "engine_single_attention"
        elif name == "DiffusionCrossAttTransformer":
            if implementation("atom_blocks") is None:
                continue
            built = atom_blocks(module, like)
            key = "engine_atom_blocks"
            if built is None and not fp32_model:
                built = atom_variant_stack(module, like)
                key = "engine_atom_variant_blocks"
                counts[key] += module.num_blocks if built else 0
            else:
                counts[key] += len(built or [])
            if built and key not in skip:
                attach(module, built)
                if isinstance(built, list):
                    module.foldforge_engine_fp32 = all(
                        p.dtype == torch.float32 for p in built[0].parameters()
                    )
            continue
        elif name == "DiffusionTransformer":
            if implementation("dit_blocks") is None:
                continue
            # the per-block engine DiTBlock serves a 384-wide conditioning only (ESMFold2's is 768, Boltz-2's too)
            built = (
                diffusion_blocks(module, like)
                if module.c_single_cond == _TOKEN_DIT_COND
                else None
            )
            key = "engine_dit_blocks"
            if built is None and not fp32_model:
                built = diffusion_variant_stack(module, like)
                key = "engine_dit_variant_blocks"
                counts[key] += module.num_blocks if built else 0
            else:
                counts[key] += len(built or [])
            if built and key not in skip:
                attach(module, built)
                if isinstance(built, list):
                    # read by the dense forward (a compiled region reads an attribute, not the blocks' parameters)
                    module.foldforge_engine_fp32 = all(
                        p.dtype == torch.float32 for p in built[0].parameters()
                    )
            continue
        if built is not None and key not in skip:
            attach(module, built)
            counts[key] += 1
            if getattr(built, "implementation", None) == ImplementationType.ANTHROPIC:
                gate_anthropic(built, {"engine_trimul": "trimul", "engine_opm": "opm"}.get(key, "msa_pwa"))
    return counts


def clear_engine_weight_caches() -> None:
    """Drop the engine's eager weight re-layout cache before weights are replaced.

    The H100 Transition kernels keep ``w.t().contiguous()`` keyed by ``(address, version, shape,
    dtype)`` with no liveness check, so a new weight allocated at the address of a freed one with
    the same version is served the old copy (silently wrong output). A process that builds a second
    model, or a test that builds many, clears it first.
    """
    from miniworld_engine.kernels.transition.cuda import _weights

    _weights._CACHE.clear()
