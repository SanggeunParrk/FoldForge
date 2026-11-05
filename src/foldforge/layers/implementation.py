"""Backend names shared by the blocks: the historical ``ImplementationType`` (``MINIWORLD_ENGINE``) and its translation to the engine's enum.

Moved from ``miniworld_engine.architecture.exceptions`` / ``_engine_impl`` (removed in miniworld-engine 2.4.2); the engine's own enum is
``miniworld_engine.modules.exceptions.ImplementationType`` (``MINIWORLD``), which :func:`to_engine_impl` maps to.
"""

from enum import Enum

import torch
from miniworld_engine import settings
from miniworld_engine.modules.exceptions import (
    ImplementationType as EngineImplementation,
)


class ImplementationType(str, Enum):
    """Implementation types for certain modules."""

    PYTORCH = "pytorch"
    MINIWORLD_ENGINE = "miniworld_engine"
    CUEQUIVARIANCE = "cuequivariance"
    ANTHROPIC = "anthropic"
    # Source compatibility for old consumers; iteration uses the canonical member.
    MINIWORLD_KERNELS = MINIWORLD_ENGINE

    @classmethod
    def _missing_(cls, value):
        if value == "miniworld_kernels":
            return cls.MINIWORLD_ENGINE
        return None


class InvalidImplementationError(ValueError):
    """Raised when an invalid implementation type is specified."""

    def __init__(self, implementation: str) -> None:
        valid = ", ".join(f"'{impl.value}'" for impl in ImplementationType)
        msg = f"Invalid implementation: '{implementation}'. Valid options: {valid}."
        super().__init__(msg)


def legacy_options_removed(where: str, **options: bool) -> None:
    """Raise when a removed legacy option (magnitude-preserving ``magnitude_preserving`` / ``mp_full`` / ``mp_residual`` or rotation
    modulation ``use_rotation``) is switched on. The options are accepted, and must be false, so configs that spell them out as false keep
    loading; the code behind them is in git history (see the commit that added this function)."""
    on = sorted(name for name, value in options.items() if value)
    if on:
        msg = f"{where}: the legacy option(s) {', '.join(on)} were removed (no engine kernel or current model uses them); restore them from git history if needed."
        raise NotImplementedError(msg)


_B200_CAPABILITY = (10, 0)


def torch_pointwise() -> bool:
    """Whether pointwise and normalization modules run as PyTorch ops here.

    B200 policy: the pieces around the GEMMs and attention are PyTorch ops that
    ``torch.compile`` fuses (inductor), not the engine's hand-written Triton
    kernels. ``miniworld_engine.settings.b200_portable_kernels`` keeps the engine
    Triton kernels (the A/B switch). An engine that predates the switch keeps
    its Triton kernels, so this is a no-op until the engine pin has it.

    Call it when a module is BUILT: it asks the device, which a traced forward
    must not.
    """
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == _B200_CAPABILITY
    ):
        return False
    return not getattr(settings.current(), "b200_portable_kernels", True)


def to_engine_impl_pointwise(
    implementation: ImplementationType,
) -> EngineImplementation:
    """Like :func:`to_engine_impl`, but PyTorch under :func:`torch_pointwise`."""
    if torch_pointwise():
        return EngineImplementation.PYTORCH
    return to_engine_impl(implementation)


def to_engine_impl(implementation: ImplementationType) -> EngineImplementation:
    """Preserve team-gm configuration values while selecting engine backends."""
    implementation = ImplementationType(implementation)
    return EngineImplementation(
        "miniworld"
        if implementation == ImplementationType.MINIWORLD_ENGINE
        else implementation.value,
    )
