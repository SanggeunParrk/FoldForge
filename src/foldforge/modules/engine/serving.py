"""Host-side observations of the engine modules a model holds: which backend each runs and which calls it refused.

The guard of :mod:`foldforge.modules.engine.policy` counts calls (``served`` per backend) and refusals (``refused`` per reason) per module.
The counters observe Python calls, including eager execution and graph capture; compiled regions and CUDA graph replays do not update
them (graph replay counts belong to ``Execution.report``). They are not a kernel trace.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch

_STATS_ATTR = "_foldforge_observations"


@dataclass
class Observations:
    """Calls of one engine module, aggregated by backend and by refusal reason."""

    served: Counter[str] = field(default_factory=Counter)
    refused: Counter[str] = field(default_factory=Counter)


def observations_of(module: torch.nn.Module) -> Observations:
    """The module's counters (created on first use)."""
    stats = module.__dict__.get(_STATS_ATTR)
    if stats is None:
        stats = Observations()
        object.__setattr__(module, _STATS_ATTR, stats)
    return stats


def observe_call(module: torch.nn.Module, backend: str, refusal: str | None) -> None:
    """Record one call outside Dynamo tracing: ``backend`` ran it, or ``refusal`` sent it to the pytorch backend."""
    if torch.compiler.is_compiling():
        return
    stats = observations_of(module)
    if refusal is None:
        stats.served[backend] += 1
    else:
        stats.refused[refusal] += 1
        stats.served["pytorch"] += 1


def reset_observations(model: torch.nn.Module) -> None:
    """Start observations at the same boundary as a new Execution session."""
    for module in model.modules():
        module.__dict__.pop(_STATS_ATTR, None)


def report_observations(model: torch.nn.Module) -> dict[str, Any]:
    """The engine modules of ``model``: their selected backend and what was observed on them."""
    modules = {}
    for name, module in model.named_modules():
        stats = module.__dict__.get(_STATS_ATTR)
        if stats is None and not getattr(module, "_foldforge_guarded", False):
            continue
        modules[name or "<root>"] = {
            "class": type(module).__name__,
            "backend": getattr(module, "backend", None),
            "host_observed": stats is not None,
            "served_host_calls": {} if stats is None else dict(stats.served),
            "refused_host_calls": {} if stats is None else dict(stats.refused),
        }
    return {
        "coverage": "engine_modules_with_a_backend_guard",
        "counting": "host_calls_including_capture; excludes_compiled_regions_and_graph_replays",
        "selections_are_kernel_traces": False,
        "modules": modules,
    }
