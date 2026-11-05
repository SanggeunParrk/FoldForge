"""Serve token and atom transformers through one adapter contract.

Installation selects a per-block or whole-stack layout adapter once. Dense callers
use ``serve_token`` / ``serve_atom`` and receive an output or None for dense fallback.
The four layout handlers preserve the original views, padding and cache-key
lifetimes; engine weights remain attached out of band. Host observations are
reported separately from compiled execution and CUDA Graph replay counts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldforge.modules.engine.modules import (
    attachment_support,
    engine_of,
    variant_selection,
)
from foldforge.modules.engine.routing import enabled, enabled_fp32
from foldforge.modules.engine.serving import (
    ADAPTER_ATTR,
    Dispatch,
    TransformerAdapter,
    observe,
)

_UNBATCHED_TOKEN_RANK = 2

#: Token widths of the engine's Hopper FP32 token DiT (`integrations.token_dit.LAYOUTS`,
#: 16 heads, 384-wide conditioning).
_FP32_TOKEN_WIDTHS = (768, 1024)
_FP32_TOKEN_TILE = 128


def _fp32_token_blocks(module: nn.Module, built: object, act: torch.Tensor) -> bool:
    """Whether FP32 activations may take the engine's per-block token DiT (TF32 tensor cores, FP32 residual stream).

    The blocks must hold FP32 parameters (the engine's FP32 path reads no other), TF32 must be allowed for
    this module (`routing.set_tf32_policy`), and the call must be the shape the Hopper FP32 kernels
    take: whole 128-token tiles at a 768 / 1024 width. Anything else keeps the dense FP32 statements.
    """
    return (
        isinstance(built, list)
        and enabled_fp32(module, act)
        and act.shape[-1] in _FP32_TOKEN_WIDTHS
        and act.shape[-2] % _FP32_TOKEN_TILE == 0
        and getattr(module, "foldforge_engine_fp32", False)
    )


def _serves_extra_bias(act: torch.Tensor, extra_pair_bias: torch.Tensor) -> bool:
    """Whether the engine's token DiT takes this [L, L] bias (Hopper, BF16, ``act``'s token count)."""
    from miniworld_engine.modules import runtime

    return runtime.token_dit_serves_extra_bias(
        act[None, None] if act.ndim == _UNBATCHED_TOKEN_RANK else act[:, None],
        extra_pair_bias,
    )


def _token_stack(
    built: nn.Module,
    module: nn.Module,
    act: torch.Tensor,
    mask: torch.Tensor,
    single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    key_mask: torch.Tensor | None,
    extra_pair_bias: torch.Tensor | None,
) -> Dispatch:
    """Serve a whole token stack without changing its fold-level bias cache keys."""
    if not enabled(module, act):
        return Dispatch(refusal="backend_or_dtype")
    unbatched = act.ndim == _UNBATCHED_TOKEN_RANK
    if key_mask is None:
        key_mask = (mask > 0)[None]
    if extra_pair_bias is not None:
        return Dispatch(refusal="extra_pair_bias_unsupported")
    x = act[None] if unbatched else act
    selection = variant_selection(built, x, single_cond, pair_cond, key_mask[0])
    out = built(x, single_cond, pair_cond, key_mask[0])
    return Dispatch(out[0] if unbatched else out, selection=selection)


def _token_blocks(
    built: list[nn.Module],
    module: nn.Module,
    act: torch.Tensor,
    mask: torch.Tensor,
    single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    key_mask: torch.Tensor | None,
    extra_pair_bias: torch.Tensor | None,
) -> Dispatch:
    """Serve per-block token DiT with the original view and expansion conventions."""
    if not (
        enabled(module, act)
        or (extra_pair_bias is None and _fp32_token_blocks(module, built, act))
    ):
        return Dispatch(refusal="backend_dtype_or_tf32_layout")
    if extra_pair_bias is not None and not _serves_extra_bias(act, extra_pair_bias):
        return Dispatch(refusal="extra_pair_bias_shape_or_policy")
    unbatched = act.ndim == _UNBATCHED_TOKEN_RANK
    if key_mask is None:
        key_mask = (mask > 0)[None]
    cond = single_cond.expand(*act.shape[:-1], single_cond.shape[-1])
    if unbatched:
        x, cond = act[None, None], cond[None, None]
    else:
        x, cond = act[:, None], cond[:, None]
    pair = pair_cond[None]
    extra = {} if extra_pair_bias is None else {"extra_pair_bias": extra_pair_bias}
    for block in built:
        x = block(x, cond, pair, key_mask, **extra)
    return Dispatch(x[0, 0] if unbatched else x[:, 0])


def _atom_stack(
    built: nn.Module,
    module: nn.Module,
    queries_act: torch.Tensor,
    queries_mask: torch.Tensor,
    queries_single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    pair_mask: torch.Tensor | None,
    rope_q: tuple[torch.Tensor, torch.Tensor] | None,
    rope_k: tuple[torch.Tensor, torch.Tensor] | None,
    engine_inputs: tuple[torch.Tensor | None, ...] | None,
) -> Dispatch:
    """Serve an atom stack, preserving conformer ids and pre-hoisted even windows."""
    subsets, queries, channels = queries_act.shape[-3:]
    samples = queries_act.shape[:-3]
    atoms = subsets * queries
    uid = engine_inputs[2] if engine_inputs and len(engine_inputs) > 2 else None
    if pair_mask is not None and uid is None:
        return Dispatch(refusal="pair_mask_requires_conformer_ids")
    if rope_q is not None or rope_k is not None:
        return Dispatch(refusal="rotary_unsupported_by_variant")
    pair, key_mask, uid, even = (
        tuple(engine_inputs) + (None,) * (4 - len(engine_inputs))
        if engine_inputs
        else (None,) * 4
    )
    if subsets % 2 and even is not None:
        pair, key_mask, uid = even
        pad = queries
    else:
        pad = 0
    pair = pair[0] if pair is not None else pair_cond
    key_mask = (
        key_mask[0] if key_mask is not None else (queries_mask > 0).reshape(atoms)
    )
    x = queries_act.reshape(-1, atoms, channels)
    cond = queries_single_cond.reshape(atoms, channels)
    if pad:
        x, cond = F.pad(x, (0, 0, 0, pad)), F.pad(cond, (0, 0, 0, pad))
    selection = variant_selection(built, x, cond, pair, key_mask, uid)
    out = built(x, cond, pair, key_mask, uid)[:, :atoms]
    return Dispatch(
        out.reshape(*samples, subsets, queries, channels), selection=selection
    )


def _atom_blocks(
    built: list[nn.Module],
    module: nn.Module,
    queries_act: torch.Tensor,
    queries_mask: torch.Tensor,
    queries_single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    pair_mask: torch.Tensor | None,
    rope_q: tuple[torch.Tensor, torch.Tensor] | None,
    rope_k: tuple[torch.Tensor, torch.Tensor] | None,
    engine_inputs: tuple[torch.Tensor | None, ...] | None,
) -> Dispatch:
    """Serve ordinary and ESM rotary blocks through their original engine signatures."""
    subsets, queries, channels = queries_act.shape[-3:]
    samples = queries_act.shape[:-3]
    atoms = subsets * queries
    # ESMFold2's blocks rotate q / k themselves and build its sliding window (the only
    # pair_mask
    # that family has, from ``window_half``); AF3's take neither.
    rotary = bool(getattr(built[0], "rope", False))
    if (
        (rope_q is not None) != rotary
        or (rope_k is not None) != rotary
        or (pair_mask is not None and not (rotary and module.window_half is not None))
    ):
        return Dispatch(refusal="rotary_or_pair_mask_mismatch")
    # FP32 blocks multiply on TF32 tensor cores: only where TF32 is allowed
    # (`routing.set_tf32_policy`)
    if (
        getattr(module, "foldforge_engine_fp32", False)
        and queries_act.dtype == torch.float32
        and not enabled_fp32(module, queries_act)
    ):
        return Dispatch(refusal="fp32_requires_tf32")
    pair, key_mask, *_ = engine_inputs if engine_inputs is not None else (None, None)
    if pair is None:
        pair = pair_cond.reshape(1, subsets, queries, -1, pair_cond.shape[-1])
    if key_mask is None:
        key_mask = (queries_mask > 0).reshape(1, atoms)
    x = queries_act.reshape(-1, 1, atoms, channels)
    cond = queries_single_cond.reshape(1, 1, atoms, channels).expand(
        x.shape[0], 1, -1, -1
    )
    rope = (
        {"rope": tuple(t.reshape(1, atoms, t.shape[-1]) for t in rope_q)}
        if rotary and rope_q is not None
        else {}
    )
    for block in built:
        x = block(x, cond, pair, key_mask, **rope)
    return Dispatch(x.reshape(*samples, subsets, queries, channels))


def attach_adapter(module: nn.Module, built: object, kind: str) -> TransformerAdapter:
    """Normalize the built representation once, outside the model's hot forward."""
    stack = isinstance(built, nn.Module)
    serves = {
        ("token", False): _token_blocks,
        ("token", True): _token_stack,
        ("atom", False): _atom_blocks,
        ("atom", True): _atom_stack,
    }
    adapter = TransformerAdapter(
        built, f"{kind}_{'stack' if stack else 'blocks'}", serves[kind, stack]
    )
    object.__setattr__(module, ADAPTER_ATTR, adapter)
    return adapter


def _serve(module: nn.Module, kind: str, *inputs: object) -> torch.Tensor | None:
    built = engine_of(module)
    support = attachment_support(module)
    if not support.accepted:
        return observe(module, Dispatch(refusal=support.refusal))
    if torch.is_inference_mode_enabled():
        return observe(
            module, Dispatch(refusal="inference_mode_has_no_tensor_versions")
        )
    adapter = module.__dict__.get(ADAPTER_ATTR)
    if adapter is None or adapter.built is not built:
        # Covers explicitly attached twins outside the production installer.
        adapter = attach_adapter(module, built, kind)
    return observe(module, adapter.try_run(module, *inputs))


def serve_token(
    module: nn.Module,
    act: torch.Tensor,
    mask: torch.Tensor,
    single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    key_mask: torch.Tensor | None,
    extra_pair_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Hand one token call through the common adapter, or return None for dense execution."""
    return _serve(
        module, "token", act, mask, single_cond, pair_cond, key_mask, extra_pair_bias
    )


def serve_atom(
    module: nn.Module,
    queries_act: torch.Tensor,
    queries_mask: torch.Tensor,
    queries_single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    pair_mask: torch.Tensor | None,
    rope_q: tuple[torch.Tensor, torch.Tensor] | None,
    rope_k: tuple[torch.Tensor, torch.Tensor] | None,
    engine_inputs: tuple[torch.Tensor | None, ...] | None,
) -> torch.Tensor | None:
    """Hand one atom call through the same adapter contract as the token transformer."""
    return _serve(
        module,
        "atom",
        queries_act,
        queries_mask,
        queries_single_cond,
        pair_cond,
        pair_mask,
        rope_q,
        rope_k,
        engine_inputs,
    )
