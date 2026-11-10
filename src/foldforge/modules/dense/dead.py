"""Numerically dead residual modules of a checkpoint: skipped at inference.

Some released checkpoints carry residual modules whose every parameter has decayed to about 1e-37 (Protenix v1: the confidence head's
triangle attentions / multiplications and pair transition, and the trunk's attention-pair-bias / single transitions of 35 / 30 of its 48 blocks).
Such a module's update is exactly zero, so ``x + update`` is ``x``; the Protenix kits skip them.

The engine's residual modules (``Transition``, ``AttentionPairBias``, ``TriangleAttention`` / ``TriangleMultiplication`` and their variants,
``OuterProductMean``, ``MSAPairWeightedAveraging``) declare a plain ``dead`` attribute: flagged, they return their residual input unchanged
when they run without grad outside training. The engine no longer ships the loader that sets it, so :func:`mark_dead` does, once the
weights are final (the same bound as the engine's old ``architecture.checkpoints.dead_modules``: 1e-30). The flag is a plain attribute, so
``torch.compile`` guards on it and CUDA graphs capture the skip. ``engine.skip_dead_modules: false`` leaves every module running (a bit-exact A/B; training needs it).
"""

from __future__ import annotations

import torch
from torch import nn

THRESHOLD = 1e-30


def _skippable(module: nn.Module) -> bool:
    """An engine module whose class declares the ``dead`` flag (the engine decides which of its modules can honour it)."""
    return hasattr(type(module), "dead")


@torch.no_grad()
def is_dead(module: nn.Module, threshold: float = THRESHOLD) -> bool:
    """Every parameter of ``module`` (recursively) below ``threshold`` in absolute value (a module without parameters is not dead)."""
    params = [p for p in module.parameters() if p.numel()]
    return bool(params) and all(
        float(p.detach().abs().max()) < threshold for p in params
    )


def mark_dead(model: nn.Module, threshold: float = THRESHOLD) -> dict[str, int]:
    """Flag the dead instances of the skippable engine modules in ``model`` (and clear the flag on the others); returns {class name: count}."""
    counts: dict[str, int] = {}
    for module in model.modules():
        if _skippable(module):
            module.dead = is_dead(module, threshold)
            if module.dead:
                counts[type(module).__name__] = counts.get(type(module).__name__, 0) + 1
    return counts


def clear_dead(model: nn.Module) -> int:
    """Run every module again (the ``dead`` flag is cleared on every module that declares it); returns how many were flagged."""
    cleared = 0
    for module in model.modules():
        if _skippable(module):
            cleared += int(bool(module.dead))
            module.dead = False
    return cleared
