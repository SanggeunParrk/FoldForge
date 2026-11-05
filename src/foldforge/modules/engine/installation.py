"""Shared candidate results and tensor-free installation diagnostics.

These records describe construction/attachment, never runtime kernel dispatch.
Existing raw factory entry points and legacy counters remain compatible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BuildResult:
    """One candidate construction, or the reason its equation/API does not fit."""

    value: Any = None
    refusal: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A named factory with the installer's existing precision/width constraints."""

    key: str
    factory: Callable[..., BuildResult]
    fp32: bool = True
    conditioning_width: int | None = None

    def build(
        self, module: Any, like: torch.Tensor, implementation: Any
    ) -> BuildResult:
        """Evaluate exactly once; the first constructed candidate stops the chain."""
        if not self.fp32 and like.dtype == torch.float32:
            return BuildResult(refusal="fp32_variant_disabled")
        if (
            self.conditioning_width is not None
            and module.c_single_cond != self.conditioning_width
        ):
            return BuildResult(refusal="conditioning_width")
        return self.factory(module, like, implementation)


def installation_report(model: Any) -> dict[str, Any]:
    """Read install decisions separately from host execution observations."""
    return model.__dict__.get("_foldforge_installation", {})


def record_installation(model: Any, report: dict[str, Any]) -> None:
    """Keep only metadata on the model, without registering an engine module."""
    object.__setattr__(model, "_foldforge_installation", report)


def phase_decisions(model: Any) -> dict[str, dict[str, Any]]:
    """Read parameter-tree conversion decisions outside the twin candidate table."""
    return model.__dict__.get("_foldforge_installation_phases", {})


def record_phase(model: Any, phase: str, decisions: dict[str, Any]) -> None:
    """Store a fresh set of tensor-free conversion decisions for one phase."""
    phases = model.__dict__.get("_foldforge_installation_phases")
    if phases is None:
        phases = {}
        object.__setattr__(model, "_foldforge_installation_phases", phases)
    phases[phase] = decisions
