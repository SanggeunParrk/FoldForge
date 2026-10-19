# Optional chemistry/GPU backends load at the selected execution boundary.
"""Execution boundaries: bounded compile + CUDA-graph callables, complete-forward measurements."""

from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import numpy as np
import torch
from torch._dynamo.utils import counters

from foldforge.utils.seed import RNGState

if TYPE_CHECKING:
    from collections.abc import Callable

    from foldforge.models.config import ExecutionConfig


class ExecutionPolicy(Protocol):
    """Configuration supplied by the terminal application."""

    compile: bool
    cuda_graph: bool
    max_graphs: int


def _map(value, tensor_fn):
    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                f.name: _map(getattr(value, f.name), tensor_fn)
                for f in dataclasses.fields(value)
                if f.init
            },
        )
    if isinstance(value, dict):
        return type(value)((k, _map(v, tensor_fn)) for k, v in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_map(v, tensor_fn) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(_map(v, tensor_fn) for v in value)
    if value is None or isinstance(
        value, (str, int, float, bool, torch.dtype, torch.device)
    ):
        return value
    raise TypeError(f"Unsupported graph input/output type: {type(value).__name__}")


def _flatten(value):
    tensors = []

    def add(t):
        if t.device.type == "cpu":
            if t.numel() > 64 or t.dtype not in (torch.int32, torch.int64, torch.bool):
                raise ValueError(
                    "Only small integer CPU shape metadata is supported in CUDA graphs"
                )
            return (
                "cpu_metadata",
                tuple(t.shape),
                str(t.dtype),
                tuple(t.reshape(-1).tolist()),
            )
        tensors.append(t)
        return (
            "tensor",
            len(tensors) - 1,
            tuple(t.shape),
            str(t.dtype),
            str(t.device),
            tuple(t.stride()),
        )

    # Dataclasses cannot always be rebuilt with signature tuples in tensor fields.
    def visit(v):
        if isinstance(v, torch.Tensor):
            return add(v)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return (
                type(v),
                tuple(
                    (f.name, visit(getattr(v, f.name)))
                    for f in dataclasses.fields(v)
                    if f.init
                ),
            )
        if isinstance(v, dict):
            return (type(v), tuple((k, visit(x)) for k, x in v.items()))
        if isinstance(v, (tuple, list)):
            return (type(v), tuple(visit(x) for x in v))
        if v is None or isinstance(
            v, (str, int, float, bool, torch.dtype, torch.device)
        ):
            return v
        raise TypeError(f"Unsupported graph input: {type(v).__name__}")

    return visit(value), tensors


def copy_containers(value):
    """Restore per-call dictionaries without copying immutable input tensors."""
    return _map(value, lambda t: t)


class _Graph:
    def __init__(self, fn, args, kwargs):
        _, inputs = _flatten((args, kwargs))
        if not inputs or any(t.device.type != "cuda" for t in inputs):
            raise ValueError("CUDA graph tensor inputs must be on the GPU")
        self.args, self.kwargs = _map((args, kwargs), lambda t: t.detach().clone())
        _, self.inputs = _flatten((self.args, self.kwargs))
        # Warmup/autotuning must not consume the sampler's random stream.
        # Replay itself advances any generator used by the captured callable.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    fn(*self.args, **self.kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = fn(*self.args, **self.kwargs)
            torch.cuda.synchronize()
        # Freed capture temporaries remain reserved in this private pool. Live
        # tensor allocation deltas miss them and cannot bound graph-cache memory.
        self.retained_bytes = sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot(self.graph.pool())
        ) + sum(value.numel() * value.element_size() for value in self.inputs)

    def __call__(self, args, kwargs):
        _, values = _flatten((args, kwargs))
        for dst, src in zip(self.inputs, values, strict=True):
            dst.copy_(src)
        self.graph.replay()
        # Caller may retain or modify outputs across denoising steps/samples.
        return _map(self.output, lambda t: t.clone())


class ExecutedCallable:
    def __init__(self, fn, config: ExecutionPolicy, label: str):
        self.original = fn
        self.config = config
        self.label = label
        self.calls = 0
        self.replays = 0
        self.captures = 0
        self.evictions = 0
        self.graphs = OrderedDict()
        self.fn = (
            torch.compile(
                fn,
                backend="inductor",
                dynamic=False,
                # Preserve the native-BF16 rounding boundaries used by checkpoints.
                options={
                    "triton.cudagraphs": False,
                    "emulate_precision_casts": True,
                },
            )
            if config.compile
            else fn
        )

    def release_large_graphs(self, max_bytes=None):
        """Free large graph pools at a request boundary so eager stages can run.

        Small graphs remain reusable. Large denoisers still replay throughout
        sampling, then recapture on the next request using the compiled callable.
        """
        if not self.graphs:
            return
        if max_bytes is None:
            max_bytes = (
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory
                // 3
            )
        if sum(graph.retained_bytes for graph in self.graphs.values()) <= max_bytes:
            return
        torch.cuda.synchronize()
        self.evictions += len(self.graphs)
        self.graphs.clear()
        torch.cuda.empty_cache()

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError("Inference execution requires no_grad or inference_mode")
        self.calls += 1
        if not self.config.cuda_graph:
            return self.fn(*args, **kwargs)
        signature, _ = _flatten((args, kwargs))
        if signature not in self.graphs:
            if len(self.graphs) >= self.config.max_graphs:
                raise RuntimeError(
                    f"{self.label}: max_graphs={self.config.max_graphs} exceeded; use a separate runner for this shape"
                )
            try:
                self.graphs[signature] = _Graph(self.fn, args, kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"CUDA graph capture failed at {self.label}: {exc}"
                ) from exc
            self.captures += 1
        self.replays += 1
        return self.graphs[signature](args, kwargs)


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
        # no family is on any more: the dense graph buckets its own inputs.
        if not (config.compile or config.cuda_graph):
            return
        if config.scope == "model":
            owner, method = model, "forward"
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
