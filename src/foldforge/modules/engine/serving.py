"""A shared adapter contract and bounded host-side dispatch/support observations.

These counters observe Python calls, including eager execution and graph capture.
Compiled regions and CUDA graph replays do not update them; graph replay counts
belong to Execution.report. A selected adapter is not a per-kernel trace.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from foldforge.modules.engine.backends import Support

ADAPTER_ATTR = "_foldforge_transformer_adapter"
_STATS_ATTR = "_foldforge_transformer_observations"


@dataclass(frozen=True)
class Dispatch(Support):
    """An executed output or a named refusal to hand back to the dense path."""

    value: Any = None


@dataclass(frozen=True)
class EngineAdapter:
    """A uniform callable around an engine module or Pairformer composition."""

    built: Any
    kind: str
    serve: Callable[..., Dispatch]

    def try_run(self, *args: Any) -> Dispatch:
        """Check support and execute through the adapter's equation-preserving layout."""
        return self.serve(self.built, *args)

    def describe(self) -> dict[str, Any]:
        """Describe attachment without claiming that it executed."""
        blocks = self.built if isinstance(self.built, list) else [self.built]
        return {
            "adapter": self.kind,
            "execution_scope": "composition" if self.kind == "pairformer" else "module",
            "implementations": sorted({type(block).__name__ for block in blocks}),
            "attached_objects": len(blocks),
        }


# Preserve the existing transformer interface while Pairformer shares its contract.
TransformerAdapter = EngineAdapter


@dataclass
class Observations:
    """Host observations, aggregated by a bounded set of reasons and selections."""

    served: int = 0
    refusals: Counter[str] = field(default_factory=Counter)
    selections: Counter[str] = field(default_factory=Counter)
    supported_checks: int = 0
    refused_checks: Counter[str] = field(default_factory=Counter)


def observe_support(module: torch.nn.Module, support: Support) -> Support:
    """Record a candidate check without counting it as an executed operation."""
    if not torch.compiler.is_compiling():
        stats = module.__dict__.get(_STATS_ATTR)
        if stats is None:
            stats = Observations()
            object.__setattr__(module, _STATS_ATTR, stats)
        if support.accepted:
            stats.supported_checks += 1
        else:
            stats.refused_checks[support.refusal or "unsupported"] += 1
    return support


def observe(module: torch.nn.Module, result: Dispatch) -> Any:
    """Record only outside Dynamo tracing; never mutate counters in a compiled graph."""
    if not torch.compiler.is_compiling():
        stats = module.__dict__.get(_STATS_ATTR)
        if stats is None:
            stats = Observations()
            object.__setattr__(module, _STATS_ATTR, stats)
        if result.value is None:
            stats.refusals[result.refusal or "unsupported"] += 1
        else:
            stats.served += 1
            stats.selections[result.selection] += 1
    return result.value


def reset_observations(model: torch.nn.Module) -> None:
    """Start observations at the same boundary as a new Execution session."""
    for module in model.modules():
        module.__dict__.pop(_STATS_ATTR, None)


def report_observations(model: torch.nn.Module) -> dict[str, Any]:
    """Separate attached implementations, observed executions and named refusals."""
    modules = {}
    for name, module in model.named_modules():
        adapter = module.__dict__.get(ADAPTER_ATTR)
        built = module.__dict__.get("_engine_module")
        stats = module.__dict__.get(_STATS_ATTR)
        if adapter is None and built is None and stats is None:
            continue
        modules[name or "<root>"] = {
            "attached": adapter is not None or built is not None,
            "host_observed": stats is not None,
            **(
                adapter.describe()
                if adapter is not None
                else {
                    "adapter": "attached_op" if built is not None else "unattached_op",
                    "execution_scope": "support_checks_only",
                    "implementations": [] if built is None else [type(built).__name__],
                }
            ),
            "served_host_calls": 0 if stats is None else stats.served,
            "refused_host_calls": {} if stats is None else dict(stats.refusals),
            "selections": {} if stats is None else dict(stats.selections),
            "supported_host_checks": 0 if stats is None else stats.supported_checks,
            "refused_host_checks": {} if stats is None else dict(stats.refused_checks),
        }
    return {
        "coverage": "token_atom_and_pairformer; operation_support_checks",
        "counting": "host_calls_including_capture; excludes_compiled_regions_and_graph_replays",
        "selections_are_kernel_traces": False,
        "modules": modules,
    }
