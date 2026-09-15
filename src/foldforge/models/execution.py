# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Model-specific execution boundaries and complete-forward measurements."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import torch
from team_gm.modules.execution import ExecutedCallable
from team_gm.modules.execution import (
    copy_containers as copy_containers,  # noqa: PLC0414 - intentional public re-export
)
from torch._dynamo.utils import counters

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
        torch.set_float32_matmul_precision("highest")
        self.config = config
        self.wrappers = []
        self.bucket_adapter = None
        self.bucket_model = None
        self.initial_graphs = counters["stats"]["unique_graphs"]
        flat_buckets = config.bucketing and family in {"protenix", "opendde"}
        if flat_buckets:
            if config.scope != "denoiser":
                msg = "Flat-atom bucketing requires scope=denoiser"
                raise ValueError(msg)
            from team_gm.modules.checkpoints.stacks import PairformerStack
            from team_gm.modules.checkpoints.triangle import TriangleAttention

            setattr(model, "inference_bucketing", True)  # noqa: B010 - runtime metadata
            self.bucket_model = model
            for layer in model.modules():
                if isinstance(layer, (PairformerStack, TriangleAttention)):
                    setattr(layer, "inference_bucketing", True)  # noqa: B010 - runtime metadata
        if not (config.compile or config.cuda_graph or flat_buckets):
            return
        if config.scope == "model":
            owner, method = model, "forward"
        elif family == "esmfold2":
            owner, method = (
                model.get_submodule("structure_head.diffusion_module"),
                "denoise",
            )
        elif family == "af3":
            owner, method = model.get_submodule("diffusion_head"), "forward"
        else:
            owner, method = model.get_submodule("diffusion_module"), "forward"
        label = f"{family}.{config.scope}"
        wrapped = ExecutedCallable(getattr(owner, method), config, label)
        if flat_buckets:
            from team_gm.modules.checkpoints.padding import BucketedDenoiser

            self.bucket_adapter = BucketedDenoiser(wrapped)
            setattr(owner, method, self.bucket_adapter)
        else:
            setattr(owner, method, wrapped)
        self.wrappers.append(wrapped)

    def report(self) -> dict[str, Any]:
        """Report observed compiler graphs and successful manual replays."""
        compiled = int(counters["stats"]["unique_graphs"] - self.initial_graphs)
        if self.config.compile and self.wrappers and compiled == 0:
            message = "Compilation was requested but no compiled graph executed"
            raise RuntimeError(message)
        replays = sum(x.replays for x in self.wrappers)
        buckets = {}
        if self.bucket_model is not None:
            buckets = {
                "denoiser_buckets": None
                if self.bucket_adapter is None or self.bucket_adapter.shape is None
                else vars(self.bucket_adapter.shape),
                "pair_buckets": {
                    name: layer.inference_pair_shape
                    for name, layer in self.bucket_model.named_modules()
                    if hasattr(layer, "inference_pair_shape")
                },
                "sampled_msa_buckets": {
                    name: layer.inference_msa_shape
                    for name, layer in self.bucket_model.named_modules()
                    if hasattr(layer, "inference_msa_shape")
                },
            }
        return {
            **buckets,
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
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    timings = []
    result = None
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(repeats + 1):
        if iteration:
            # Do not inflate peak memory by retaining the previous full prediction.
            del result
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
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
