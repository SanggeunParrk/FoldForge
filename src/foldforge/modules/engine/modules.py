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
    from torch import nn

_ENGINE = ImplementationType.MINIWORLD
#: The engine's ``TriangleAttention`` takes the ending node's bias orientation (``transposed_bias``).
_ENGINE_HAS_BIAS_ORIENTATION = (
    "transposed_bias" in inspect.signature(TriangleAttention.__init__).parameters
)
#: The engine's ``LocalDiTBlock`` takes AlphaFold 3's K/V norm and key window (``kv_norm`` / ``key_window``).
_ENGINE_HAS_ATOM_VARIANT = (
    "kv_norm" in inspect.signature(LocalDiTBlock.__init__).parameters
)
_ATTR = "_engine_module"
_DEFAULT_EPS = 1e-5
_ATOM_WIDTH, _ATOM_PAIR, _ATOM_KEYS = 128, 16, 128


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


def triangle_multiplication(module: nn.Module, like: torch.Tensor) -> nn.Module | None:
    """Engine TriMul from a dense ``TriangleMultiplication`` (None when it cannot serve it)."""
    d = module.c_pair
    # The hidden width may differ from the pair width (template stacks: pair 64,
    # hidden 128); the engine takes it as d_hidden.
    h = module.center_norm.weight.shape[0]
    if getattr(module, "divide_by_length", False) or (
        module.left_norm_input.eps != module.center_norm.eps
    ):
        return None
    outgoing = module.equation == "cik,cjk->cij"
    # The released projection interleaves a,b per hidden channel; the engine takes them as two
    # matrices. Incoming contracts b[k,i] * a[k,j] where the engine contracts a[k,i] * b[k,j].
    projection = module.projection.weight.reshape(h, 2, d).transpose(0, 1)
    gate = module.gate.weight.reshape(h, 2, d).transpose(0, 1)
    left, right = (0, 1) if outgoing else (1, 0)
    built = TriangleMultiplication(
        d, d_hidden=h, outgoing=outgoing, implementation=_ENGINE, p_drop=0.0
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
    return built


def triangle_attention(module: nn.Module, like: torch.Tensor) -> nn.Module | None:
    """Engine TriangleAttention from a dense ``GridSelfAttention``.

    The ending node takes its pair bias from the transposed pair (``transposed_bias``) or, as AF3's
    released code does, from the pair as it is. An engine without the ``transposed_bias`` option
    knows only the first, so a module on the second is left on the reference path there.
    """
    hidden = module.num_head * module.qkv_dim
    ending_as_is = module.transpose and not module.transposed_bias
    if (
        module.dual_output
        or module.untransposed_output
        or module.gating_query.bias is not None
        or module.output_projection.bias is not None
        or (ending_as_is and not _ENGINE_HAS_BIAS_ORIENTATION)
    ):
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
    )
    _copy(
        built,
        {
            "ln_pair.weight": module.act_norm.weight,
            "ln_pair.bias": module.act_norm.bias,
            "to_query.weight": module.q_projection.weight,
            "to_key.weight": module.k_projection.weight,
            "to_value.weight": module.v_projection.weight,
            "to_bias.weight": module.pair_bias_projection.weight,
            "to_gate.weight": module.gating_query.weight,
            "to_out.weight": module.output_projection.weight,
        },
        like,
    )
    _set_eps(built, module.act_norm.eps)
    return built


def pair_weighted_averaging(module: nn.Module, like: torch.Tensor) -> nn.Module:
    """Engine MSAPairWeightedAveraging from a dense ``MSAAttention``.

    The engine drops the pair LayerNorm offset: it adds one constant to every key's logit of a
    head, which the softmax over keys cancels exactly.
    """
    built = MSAPairWeightedAveraging(
        module.c_msa,
        module.c_pair,
        module.num_head,
        module.value_dim,
        implementation=_ENGINE,
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


def outer_product_mean(module: nn.Module, like: torch.Tensor) -> nn.Module | None:
    """Engine OuterProductMean from a dense ``OuterProductMean`` (single group, no projection bias).

    The engine always divides by ``clamp(count, 1)``; the released form divides by ``count + 1e-3``.
    The two differ by 0.1% at a single row and not at all beyond bf16 resolution after that; they
    differ at pairs with no valid row (padding), which the graph masks. ``bias_after_norm`` is the
    engine's ``normalize_before_proj``.
    """
    if module.groups != 1 or module.left_projection.bias is not None:
        return None
    channels = module.num_outer_channel
    built = OuterProductMean(
        module.c_msa,
        module.num_output_channel,
        channels,
        normalize_before_proj=module.bias_after_norm,
        implementation=_ENGINE,
    )
    _copy(
        built,
        {
            "ln_msa.weight": module.layer_norm_input.weight,
            "ln_msa.bias": module.layer_norm_input.bias,
            "to_left.weight": module.left_projection.weight,
            "to_right.weight": module.right_projection.weight,
            "to_out.weight": module.output_w.reshape(
                channels * channels, module.num_output_channel
            ).T,
            "to_out.bias": module.output_b,
        },
        like,
    )
    _set_eps(built, module.layer_norm_input.eps)
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


def _eps_of(adaptive_norm: nn.Module) -> set[float | None]:
    """The epsilons of an AdaLN's norms; a form without them (RMS conditioning) gives ``None``, which is not the default."""
    return {
        getattr(getattr(adaptive_norm, name, None), "eps", None)
        for name in ("layer_norm", "single_cond_layer_norm")
    }


def atom_blocks(module: nn.Module, like: torch.Tensor) -> list[nn.Module] | None:
    """One engine ``LocalDiTBlock`` per block of a dense ``DiffusionCrossAttTransformer``.

    The K/V AdaLN reads the raw atoms (AF3, ``kv_norm="parallel"``) or the first AdaLN's output
    (Protenix, OpenDDE: ``"chained"``); the 128-key window is AF3's slid one or the clipped, OR-masked
    one. Anything else (rotary, kq-norm, parallel schedule, RMS conditioning, other widths, other
    transitions) keeps the reference path.
    """
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
        or attention.gate_bias
        or attention.q_projection.bias is None
        or attention.qkv_dim * attention.num_head != attention.c_x
    ):
        return None
    built = AttentionPairBias(
        attention.c_x,
        norm.weight.shape[0],
        attention.num_head,
        implementation=_ENGINE,
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


def install_engine_modules(model: nn.Module, like: torch.Tensor) -> dict[str, int]:
    """Attach an engine module to every dense module that has an exact one; return the counts."""
    counts = {
        "engine_trimul": 0,
        "engine_triangle_attention": 0,
        "engine_msa_pwa": 0,
        "engine_opm": 0,
        "engine_dit_blocks": 0,
        "engine_single_attention": 0,
        "engine_atom_blocks": 0,
    }
    for module in model.modules():
        name = type(module).__name__
        built: object | None = None
        key = ""
        if name == "TriangleMultiplication":
            built, key = triangle_multiplication(module, like), "engine_trimul"
        elif name == "GridSelfAttention":
            built, key = triangle_attention(module, like), "engine_triangle_attention"
        elif name == "MSAAttention":
            built, key = pair_weighted_averaging(module, like), "engine_msa_pwa"
        elif name == "OuterProductMean" and hasattr(module, "left_projection"):
            built, key = outer_product_mean(module, like), "engine_opm"
        elif name == "_SingleAttention":
            built = single_attention(
                module.attention, module.norm, module.projection, like
            )
            key = "engine_single_attention"
        elif name == "DiffusionCrossAttTransformer":
            built = atom_blocks(module, like)
            key = "engine_atom_blocks"
            counts[key] += len(built) if built else 0
            if built:
                attach(module, built)
            continue
        elif name == "DiffusionTransformer":
            built = diffusion_blocks(module, like)
            key = "engine_dit_blocks"
            counts[key] += len(built) if built else 0
            if built:
                attach(module, built)
            continue
        if built is not None:
            attach(module, built)
            counts[key] += 1
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
