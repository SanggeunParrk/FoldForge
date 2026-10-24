"""How a dense diffusion transformer hands one call to the engine blocks attached to it.

``modules.install_engine_modules`` attaches one of three engine objects to a dense
``DiffusionTransformer`` / ``DiffusionCrossAttTransformer``: a list of per-block engine blocks (AF3's
token and atom blocks, ESMFold2's rotary atom blocks) or one whole-stack module (the Boltz-2 /
RoseTTAFold3 / Chai-1 variants). The two serve functions below are the only place that knows those
calling conventions: each returns the transformer's output, or None when the engine does not take
the call, and the dense module then runs its own blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldforge.modules.engine.modules import engine_of, variant_serves
from foldforge.modules.engine.routing import enabled, enabled_fp32

_UNBATCHED_TOKEN_RANK = 2

#: Token widths of the engine's Hopper FP32 token DiT (`integrations.token_dit.LAYOUTS`, 16 heads, 384-wide conditioning).
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
    from miniworld_engine.integrations.token_dit import serves_extra_bias

    return serves_extra_bias(
        act[None, None] if act.ndim == _UNBATCHED_TOKEN_RANK else act[:, None],
        extra_pair_bias,
    )


def serve_token(
    module: nn.Module,
    act: torch.Tensor,
    mask: torch.Tensor,
    single_cond: torch.Tensor,
    pair_cond: torch.Tensor,
    key_mask: torch.Tensor | None,
    extra_pair_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """The token transformer on the engine: attention + conditioned transition, residuals included.

    The engine caches the pair bias across solver steps by tensor version, which only a ``no_grad``
    (not ``inference_mode``) tensor has. ``key_mask`` ([1, N_token] bool, made once per fold by the
    caller) and ``pair_cond`` are keyed by address for the same reason. ``extra_pair_bias`` ([L, L],
    the structural-token family's) is folded into each block's hoisted pair bias, so it must be one
    tensor object for all of a fold's steps; only the per-block engine blocks take it.
    """
    built = engine_of(module)
    if built is None or torch.is_inference_mode_enabled():
        return None
    if not (
        enabled(module, act)
        or (extra_pair_bias is None and _fp32_token_blocks(module, built, act))
    ):
        return None
    unbatched = act.ndim == _UNBATCHED_TOKEN_RANK
    if key_mask is None:
        key_mask = (mask > 0)[None]
    if isinstance(built, nn.Module):
        # The whole variant stack (Boltz-2 / RoseTTAFold3 / Chai-1): it keys its hoisted pair bias
        # on the pair tensor and the mask's address, so the mask is passed as the fold-level row.
        if extra_pair_bias is not None:
            return None
        x = act[None] if unbatched else act
        if not variant_serves(built, x, single_cond, pair_cond, key_mask[0]):
            return None
        out = built(x, single_cond, pair_cond, key_mask[0])
        return out[0] if unbatched else out
    if extra_pair_bias is not None and not _serves_extra_bias(act, extra_pair_bias):
        return None
    cond = single_cond.expand(*act.shape[:-1], single_cond.shape[-1])
    if unbatched:
        x, cond = act[None, None], cond[None, None]
    else:
        x, cond = act[:, None], cond[:, None]
    pair = pair_cond[None]
    extra = {} if extra_pair_bias is None else {"extra_pair_bias": extra_pair_bias}
    for block in built:
        x = block(x, cond, pair, key_mask, **extra)
    return x[0, 0] if unbatched else x[:, 0]


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
    """The atom transformer on the engine: the atoms of the query subsets are one flat run of windows.

    Subset ``w`` holds atoms ``32 w .. 32 w + 31``, its keys are the AF3 slid window (or the clipped
    one of the OpenFold lineage), so the dense ``pair_cond`` is the engine's trunked pair as it is;
    the real atoms come first. ``engine_inputs`` is the engine's ``(pair, key_mask[, uid[, even]])``
    layout made once per fold by the caller: the engine hoists its pair bias per tensor object, and
    a view made here would be a new object at every call.
    """
    built = engine_of(module)
    if built is None or torch.is_inference_mode_enabled():
        return None
    subsets, queries, channels = queries_act.shape[-3:]
    samples = queries_act.shape[:-3]
    atoms = subsets * queries
    if isinstance(built, nn.Module):
        # A variant stack: Chai-1's same-conformer pair mask is carried as conformer ids (``uid``).
        uid = engine_inputs[2] if engine_inputs and len(engine_inputs) > 2 else None
        if (
            (pair_mask is not None and uid is None)
            or rope_q is not None
            or rope_k is not None
        ):
            return None
        pair, key_mask, uid, even = (
            tuple(engine_inputs) + (None,) * (4 - len(engine_inputs))
            if engine_inputs
            else (None,) * 4
        )
        if subsets % 2 and even is not None:
            # The fold's inputs with one empty window appended: the stack's own padding would be
            # a new tensor, and a new pair bias, at every step.
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
        if not variant_serves(built, x, cond, pair, key_mask, uid):
            return None
        out = built(x, cond, pair, key_mask, uid)[:, :atoms]
        return out.reshape(*samples, subsets, queries, channels)
    # ESMFold2's blocks rotate q / k themselves and build its sliding window (the only pair_mask
    # that family has, from ``window_half``); AF3's take neither.
    rotary = bool(getattr(built[0], "rope", False))
    if (
        (rope_q is not None) != rotary
        or (rope_k is not None) != rotary
        or (pair_mask is not None and not (rotary and module.window_half is not None))
    ):
        return None
    # FP32 blocks multiply on TF32 tensor cores: only where TF32 is allowed (`routing.set_tf32_policy`)
    if (
        getattr(module, "foldforge_engine_fp32", False)
        and queries_act.dtype == torch.float32
        and not enabled_fp32(module, queries_act)
    ):
        return None
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
        if rotary
        else {}
    )
    for block in built:
        x = block(x, cond, pair, key_mask, **rope)
    return x.reshape(*samples, subsets, queries, channels)
