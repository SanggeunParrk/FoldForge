# Optional chemistry/GPU backends load at the selected execution boundary.
"""Model-specific execution boundaries and complete-forward measurements."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import torch
from team_gm.modules.execution import ExecutedCallable
from team_gm.modules.execution import (
    copy_containers as copy_containers,  # noqa: PLC0414 - intentional public re-export
)
from torch._dynamo.utils import counters

from foldforge.models import entry
from foldforge.utils.seed import RNGState

if TYPE_CHECKING:
    from collections.abc import Callable

    from foldforge.models.config import ExecutionConfig

Result = TypeVar("Result")
MAX_BENCHMARK_REPEATS = 20


class Execution:
    """Install actual call wrappers; reports never infer execution from flags."""

    def __init__(
        self, model: torch.nn.Module, family: str, config: ExecutionConfig
    ) -> None:
        torch.set_float32_matmul_precision(
            "high" if getattr(model, "reference_tf32", False) else "highest"
        )
        self.autocast_scopes = getattr(model, "reference_autocast_scopes", [])
        self.config = config
        self.wrappers = []
        self.initial_graphs = counters["stats"]["unique_graphs"]
        # Bucketing by PADDING the denoiser belonged to the flat layout, which
        # no family is on any more: the dense graph buckets its own inputs and
        # the sequence one does not bucket at all.
        if not (config.compile or config.cuda_graph):
            return
        if config.scope == "model":
            owner, method = model, "forward"
        elif entry(family).layout == "sequence_atoms":
            owner, method = (
                model.get_submodule("structure_head.diffusion_module"),
                "denoise",
            )
        else:
            owner, method = model.get_submodule("diffusion_head"), "forward"
        label = f"{family}.{config.scope}"
        wrapped = ExecutedCallable(getattr(owner, method), config, label)
        setattr(owner, method, wrapped)
        self.wrappers.append(wrapped)

    def report(self) -> dict[str, Any]:
        """Report observed compiler graphs and successful manual replays."""
        compiled = int(counters["stats"]["unique_graphs"] - self.initial_graphs)
        if self.config.compile and self.wrappers and compiled == 0:
            message = "Compilation was requested but no compiled graph executed"
            raise RuntimeError(message)
        replays = sum(x.replays for x in self.wrappers)
        return {
            "autocast": bool(self.autocast_scopes),
            "autocast_scopes": self.autocast_scopes,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "compile_requested": self.config.compile,
            "cuda_graph_requested": self.config.cuda_graph,
            "compile": self.config.compile and compiled > 0,
            "cuda_graph": replays > 0,
            "execution_scope": self.config.scope,
            "compiled_graphs": compiled,
            "cuda_graph_captures": sum(x.captures for x in self.wrappers),
            "cuda_graph_evictions": sum(x.evictions for x in self.wrappers),
            "cuda_graph_replays": replays,
            "wrapped_calls": sum(x.calls for x in self.wrappers),
        }


def measured_forward[Result](
    fn: Callable[[], Result], repeats: int
) -> tuple[Result, dict[str, Any]]:
    """Time complete model forwards, preserving identical per-call random state."""
    if not 0 <= repeats <= MAX_BENCHMARK_REPEATS:
        message = f"benchmark_repeats must be in [0,{MAX_BENCHMARK_REPEATS}]"
        raise ValueError(message)
    rng_state = RNGState.capture()
    timings = []
    result = None
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(repeats + 1):
        if iteration:
            # Do not inflate peak memory by retaining the previous full prediction.
            del result
        rng_state.restore()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    # repeats + 1 always executes at least once after the range check.
    return cast("Result", result), {
        "model_seconds_cold": timings[0],
        "model_seconds_warm": timings[1:],
        "model_seconds_warm_median": float(np.median(timings[1:])) if repeats else None,
        "model_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "benchmark_scope": (
            "model_forward; excludes input featurization and output decoding"
        ),
    }
