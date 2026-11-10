"""The diffusion transformers as FoldForge builds them: engine ``TokenDiT*`` / ``AtomDiT*`` BLOCKS, the stack assembled here.

The engine owns one block per model (``TokenDiT(model=...)``, ``AtomDiT(model=...)``: the model's own class, so a state dict has that
class's keys); the stack is FoldForge's: a ``ModuleList`` ``blocks`` run in order. Each block hoists its own pair bias from the pair
(``attention.ln_pair`` + ``attention.to_bias``), so there is no shared pair norm / projection in the engine layout: the released AF3
stack's shared parameters are broadcast / sliced per block by the offline converter (:mod:`foldforge.models.checkpoints.legacy.released`).

Shapes (engine layout): ``single`` / ``cond`` ``[A, B, L, d]`` (A = diffusion samples), token ``pair`` ``[B, L, L, d_pair]``, atom
``pair`` the trunked ``[B, windows, 32, 128, d_pair]``, ``mask`` the bool key mask ``[B, L]`` (or ``[A, B, L]``), ``uid`` ``[B, N]``
(Chai-1's conformer ids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from miniworld_engine.modules import AtomDiT, TokenDiT
from miniworld_engine.modules.atom_dit import KEY_WINDOWS
from torch import nn

from foldforge.modules.engine.modules import Precision, finish, model_keys

if TYPE_CHECKING:
    from torch import Tensor

_UNBATCHED_RANK = 2


class TokenDiTStack(nn.Module):
    """``blocks``: the engine token DiT blocks of one family, in order (state-dict keys ``blocks.{i}.<engine names>``).

    Called like the dense token transformer: ``act`` ``[S, L, d]`` (or ``[L, d]``), ``mask`` ``[L]``, ``single_cond`` ``[L, d_cond]``
    (broadcast over the samples, or ``[S, L, d_cond]``), ``pair_cond`` ``[L, L, d_pair]``; ``key_mask`` ``[1, L]`` bool (made once per fold
    by the caller; made here when absent); ``extra_pair_bias`` ``[L, L]``: OpenDDE's bias, added to every head of every block."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        act: Tensor,
        mask: Tensor,
        single_cond: Tensor,
        pair_cond: Tensor,
        extra_pair_bias: Tensor | None = None,
        key_mask: Tensor | None = None,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Every block in order (each owns its residuals)."""
        kw: dict[str, Any] = {}
        if compute_dtype is not None:
            kw["compute_dtype"] = compute_dtype
        if extra_pair_bias is not None:
            kw["extra_pair_bias"] = extra_pair_bias
        if key_mask is None:
            key_mask = (mask > 0)[None]
        unbatched = act.ndim == _UNBATCHED_RANK
        cond = single_cond.expand(*act.shape[:-1], single_cond.shape[-1])
        x, cond = (act[None, None], cond[None, None]) if unbatched else (act[:, None], cond[:, None])
        pair = pair_cond[None]
        for block in self.blocks:
            x = block(x, cond, pair, key_mask, **kw)
        return x[0, 0] if unbatched else x[:, 0]


class AtomDiTStack(nn.Module):
    """``blocks``: the engine atom DiT blocks of one family, in order (keys ``blocks.{i}.<engine names>``).

    Called like the dense atom transformer on the windowed layout: ``queries_act`` ``[..., windows, 32, d]``, ``queries_mask``
    ``[windows, 32]``, ``queries_single_cond`` ``[windows, 32, d]``, ``pair_cond`` ``[windows, 32, 128, d_pair]``. The engine block derives the
    128-key window itself (from the key mask: the real atoms come first), so ``queries_to_keys`` / ``keys_mask`` / ``keys_single_cond`` are
    not read; ``engine_inputs = (pair [1, windows, 32, 128, d_pair], key_mask [1, atoms], uid)`` is the per-fold layout the caller
    made once (the engine keys its pair bias on the tensor object); ``uid`` only for Chai-1's blocks."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        queries_act: Tensor,
        queries_mask: Tensor,
        queries_to_keys: object,  # noqa: ARG002
        keys_mask: Tensor,  # noqa: ARG002
        queries_single_cond: Tensor,
        keys_single_cond: Tensor,  # noqa: ARG002
        pair_cond: Tensor,
        pair_mask: Tensor | None = None,
        rope_q: object = None,
        rope_k: object = None,
        engine_inputs: tuple[Tensor | None, ...] | None = None,
    ) -> Tensor:
        """Every block in order."""
        if rope_q is not None or rope_k is not None:
            raise ValueError("a rotary atom transformer is not an engine AtomDiT block with a pair")
        windows, queries, channels = queries_act.shape[-3:]
        atoms = windows * queries
        lead = queries_act.shape[:-3]
        pair, key_mask, uid = ((*engine_inputs, None, None, None) if engine_inputs else (None, None, None))[:3]
        if pair_mask is not None and uid is None:
            raise ValueError("a pair mask (same-conformer attention) needs the conformer ids in engine_inputs")
        if uid is not None and uid.ndim == 1:  # the encoder keeps the flat ids [N]; the engine takes [B, N]
            uid = uid[None]
        if pair is None:
            pair = pair_cond.reshape(1, windows, queries, -1, pair_cond.shape[-1])
        if key_mask is None:
            key_mask = (queries_mask > 0).reshape(1, atoms)
        x = queries_act.reshape(-1, 1, atoms, channels)
        cond = queries_single_cond.reshape(1, 1, atoms, channels).expand(x.shape[0], 1, atoms, channels)
        for block in self.blocks:
            x = block(x, cond, pair, key_mask, uid) if uid is not None else block(x, cond, pair, key_mask)
        return x.reshape(*lead, windows, queries, channels)


def engine_key_window(window: str) -> tuple[str, str | None]:
    """``(engine key_window, note)`` of a FoldForge ``atom_key_window``. The engine has ``pad`` / ``slide`` / ``slide_qblock``; Chai-1's
    exact ``circular`` was removed (it becomes ``pad`` + the conformer-id mask ``uid``; output may differ: ``docs/ENGINE-UNSUPPORTED.md``)."""
    if window in KEY_WINDOWS:
        return window, None
    if window == "circular":
        return "pad", "circular key window is not an engine key window: pad + uid (outputs may differ)"
    raise ValueError(f"unknown atom key window {window!r}")


def token_dit_stack(
    family: str,
    *,
    n_blocks: int,
    d_single: int,
    d_cond: int,
    d_pair: int,
    n_head: int,
    n: int = 2,
    adaln_eps: float | None = None,
    precision: Precision = "master",
) -> TokenDiTStack:
    """``n_blocks`` ``TokenDiT(model=...)`` blocks of ``family``. ``adaln_eps``: Chai-1's AdaLN epsilon (engine default 0.1, the release;
    FoldForge's fast mode 1e-5)."""
    key = model_keys(family).token_dit
    if key is None:
        raise ValueError(f"family {family!r} has no engine token DiT")
    extra = {"adaln_eps": adaln_eps} if adaln_eps is not None else {}
    blocks = [TokenDiT(key, d_single, d_cond, d_pair, n_head, n, **extra) for _ in range(n_blocks)]
    return finish(TokenDiTStack(blocks), precision)  # type: ignore[return-value]


def atom_dit_stack(
    family: str,
    *,
    n_blocks: int,
    d_single: int = 128,
    d_cond: int = 128,
    d_pair: int = 16,
    n_head: int = 4,
    n: int = 2,
    key_window: str | None = None,
    adaln_eps: float | None = None,
    precision: Precision = "master",
) -> AtomDiTStack:
    """``n_blocks`` ``AtomDiT(model=...)`` blocks of ``family``. ``key_window``: the FoldForge fast modes pass ``"slide"``
    (:func:`engine_key_window` for the spec's own word); ``None`` keeps the class's default. ``adaln_eps``: Chai-1's."""
    key = model_keys(family).atom_dit
    if key is None:
        raise ValueError(f"family {family!r} has no engine atom DiT with a pair (see docs/ENGINE-UNSUPPORTED.md)")
    extra: dict[str, Any] = {"key_window": key_window} if key_window is not None else {}
    if adaln_eps is not None:
        extra["adaln_eps"] = adaln_eps
    blocks = [AtomDiT(key, d_single, d_cond, d_pair, n_head, n, **extra) for _ in range(n_blocks)]
    return finish(AtomDiTStack(blocks), precision)  # type: ignore[return-value]

