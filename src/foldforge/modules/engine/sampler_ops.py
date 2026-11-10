"""The engine's fused sampler glue and atom pooling, built in one place (miniworld-engine 2.5.0).

``AugmentNoise`` / ``AlignEuler`` (the rigid-motion, noise and Kabsch steps of one sampler step) and ``PoolAtoms`` (the atom-to-token gather +
relu + masked mean of the atom encoder) are engine classes with a ``cuda`` (H100) and a ``triton`` backend and no autograd. FoldForge builds them
here, with the backend the engine policy picks for the device (``backends.choose_backend``: ``cuda`` on an H100, ``triton`` on another CUDA card,
nothing without CUDA), instead of re-deriving the compute capability at each call site. ``None`` means the torch ops stay in charge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from foldforge.modules.engine.backends import choose_backend

if TYPE_CHECKING:
    import torch

_POOL: dict[str, Any] = {}


def sampler_backend(cls: type, device: torch.device) -> str | None:
    """The kernel backend (``cuda`` or ``triton``) the engine policy gives ``cls`` on ``device``, or None (no CUDA device, or no kernel backend)."""
    backend, _reason = choose_backend("miniworld", cls, device)
    return None if backend == "pytorch" else backend


def augment_align(
    device: torch.device, *, churn: dict[str, Any], step_scale: float
) -> tuple[Any, Any] | None:
    """``(AugmentNoise, AlignEuler)`` for one sampler call on ``device``, or None where the engine glue does not serve the device.

    Built once per sampler call and never inside a graph capture."""
    from miniworld_engine.modules import AlignEuler, AugmentNoise

    backend = sampler_backend(AugmentNoise, device)
    if backend is None or sampler_backend(AlignEuler, device) != backend:
        return None
    return AugmentNoise(**churn, backend=backend), AlignEuler(
        step_scale=step_scale, backend=backend
    )


def pool_atoms(device: torch.device) -> Any | None:
    """The engine ``PoolAtoms`` for ``device`` (built once per backend), or None where it does not serve the device."""
    from miniworld_engine.modules import PoolAtoms

    backend = sampler_backend(PoolAtoms, device)
    if backend is None:
        return None
    if backend not in _POOL:
        _POOL[backend] = PoolAtoms(backend=backend)
    return _POOL[backend]
