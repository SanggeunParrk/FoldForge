# Optional chemistry/GPU backends load at the selected execution boundary.
"""Execution boundaries: bounded compile + CUDA-graph callables, complete-forward measurements."""

from __future__ import annotations

import contextlib
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


def _static_inputs(on: bool):
    """The engine's ``static_weights`` + ``static_inputs`` scopes while a graph is captured; a no-op when off.

    ``static_weights``: the weights stay fixed between replays (inference). ``static_inputs``: the
    callable's pair and conditioning inputs keep their contents for the life of the graph. Together a
    capture serves from the eager entries of the engine's hoisted tables (the token DiT's pair bias,
    the atom blocks' conditioning), so a replay re-runs none of the kernels that make them.
    """
    if not on:
        return contextlib.nullcontext()
    from miniworld_engine.kernels import _capture

    stack = contextlib.ExitStack()
    stack.enter_context(_capture.static_weights())
    stack.enter_context(_capture.static_inputs())
    return stack


def copy_containers(value):
    """Restore per-call dictionaries without copying immutable input tensors."""
    return _map(value, lambda t: t)


class _Graph:
    def __init__(self, fn, args, kwargs, *, static_inputs=False):
        _, inputs = _flatten((args, kwargs))
        if not inputs or any(t.device.type != "cuda" for t in inputs):
            raise ValueError("CUDA graph tensor inputs must be on the GPU")
        self.args, self.kwargs = _map((args, kwargs), lambda t: t.detach().clone())
        _, self.inputs = _flatten((self.args, self.kwargs))
        # Warmup/autotuning must not consume the sampler's random stream.
        # Replay itself advances any generator used by the captured callable.
        # `static_inputs`: the engine hoists tables made from per-item inputs (a pair's bias, an
        # atom block's conditioning) out of the replay. Valid here because the callable's pair and
        # conditioning inputs keep their contents for the life of the graph (see `epoch_key`).
        hoist = _static_inputs(static_inputs)
        with hoist, torch.random.fork_rng(devices=[torch.cuda.current_device()]):
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
        # The tensor object and version last copied into each static input: a caller that passes
        # the very same, unmodified tensor again (a fold's conditioning, the batch features)
        # needs no copy. Identity, not address: a freed tensor's address is reused at once.
        self._copied = [None] * len(self.inputs)

    def __call__(self, args, kwargs):
        _, values = _flatten((args, kwargs))
        for index, (dst, src) in enumerate(zip(self.inputs, values, strict=True)):
            seen = self._copied[index]
            version = getattr(src, "_version", None)
            if seen is not None and seen[0] is src and seen[1] == version:
                continue
            dst.copy_(src)
            self._copied[index] = (src, version) if version is not None else None
        self.graph.replay()
        # Caller may retain or modify outputs across denoising steps/samples.
        return _map(self.output, lambda t: t.clone())


def _pair_tokens(args, kwargs) -> int:
    """Token count of the largest pair-shaped ([N, N, ...]) tensor among the inputs, 0 if none."""
    _, tensors = _flatten((args, kwargs))
    return max(
        (t.shape[0] for t in tensors if t.ndim >= 3 and t.shape[0] == t.shape[1]),
        default=0,
    )


class ExecutedCallable:
    def __init__(
        self,
        fn,
        config: ExecutionPolicy,
        label: str,
        max_graph_tokens=None,
        *,
        epoch_key: str | None = None,
        epoch_fn: Callable[[], object] | None = None,
        static_inputs: bool = False,
    ):
        # `epoch_fn` is the same idea for a callable whose fold is not an argument: it is asked at each
        # call, and a changed answer drops the graphs.
        self.epoch_fn = epoch_fn
        self.max_graph_tokens = max_graph_tokens
        # >0 while an enclosing capture owns this callable: it then runs `fn` directly, so the
        # compiled function is traced into the outer graph instead of starting a graph of its own.
        self.bypass = 0
        # `epoch_key` names a keyword argument that changes when the callable's per-fold
        # inputs were remade: the graphs of an older epoch are dropped, not kept beside the new.
        self.epoch_key = epoch_key
        self.epoch = None
        self.static_inputs = static_inputs
        self.uncaptured = 0
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
        # The epoch keys the graphs only: the wrapped callable (and the compiler) never see it,
        # or every new fold would recompile.
        epoch = kwargs.pop(self.epoch_key, None) if self.epoch_key is not None else None
        if self.bypass or not self.config.cuda_graph:
            return self.fn(*args, **kwargs)
        if (
            self.max_graph_tokens is not None
            and _pair_tokens(args, kwargs) > self.max_graph_tokens
        ):
            self.uncaptured += 1
            return self.fn(*args, **kwargs)
        if self.epoch_fn is not None:
            epoch = self.epoch_fn()
        if (self.epoch_key is not None or self.epoch_fn is not None) and epoch != self.epoch:
            if self.graphs:
                torch.cuda.synchronize()
                self.graphs.clear()
            self.epoch = epoch
        signature, _ = _flatten((args, kwargs))
        if signature not in self.graphs:
            if len(self.graphs) >= self.config.max_graphs:
                raise RuntimeError(
                    f"{self.label}: max_graphs={self.config.max_graphs} exceeded; use a separate runner for this shape"
                )
            try:
                self.graphs[signature] = _Graph(
                    self.fn, args, kwargs, static_inputs=self.static_inputs
                )
            except Exception as exc:
                raise RuntimeError(
                    f"CUDA graph capture failed at {self.label}: {exc}"
                ) from exc
            self.captures += 1
        self.replays += 1
        return self.graphs[signature](args, kwargs)


class SamplerScope:
    """Capture scope of the sampler graph: the denoiser runs inside it, under the engine's static contracts."""

    def __init__(self, denoiser: ExecutedCallable) -> None:
        self.denoiser = denoiser
        self.captures = 0
        self.replays = 0

    @contextlib.contextmanager
    def __call__(self):
        self.denoiser.bypass += 1
        try:
            with _static_inputs(True):
                yield
        finally:
            self.denoiser.bypass -= 1


Result = TypeVar("Result")
MAX_BENCHMARK_REPEATS = 20


class Execution:
    """Install actual call wrappers; reports never infer execution from flags."""

    def __init__(
        self, model: torch.nn.Module, family: str, config: ExecutionConfig
    ) -> None:
        tf32 = bool(config.tf32 or getattr(model, "reference_tf32", False))
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        # The engine's FP32 kernels are TF32 tensor core kernels: they serve where TF32 is allowed, and in the
        # diffusion head of the af3_default policy (``reference_precision``).
        from foldforge.modules.engine.routing import set_tf32_policy

        set_tf32_policy(
            model,
            tf32=tf32,
            diffusion_tf32=bool(getattr(model, "reference_precision", False)),
        )
        self.autocast_scopes = getattr(model, "reference_autocast_scopes", [])
        head = getattr(model, "confidence_head", None)
        if head is not None and hasattr(head, "sample_chunk_override"):
            head.sample_chunk_override = config.confidence_sample_chunk
        self.config = config
        self.wrappers = []
        self.sampler_scope = None
        self.initial_graphs = counters["stats"]["unique_graphs"]
        # Bucketing by PADDING the denoiser belonged to the flat layout, which
        # no family is on any more: the dense graph buckets its own inputs.
        if not (config.compile or config.cuda_graph):
            return
        if config.scope == "model":
            sites = [(model, "forward", "model", config.compile)]
        else:
            sites = [
                (model.get_submodule("diffusion_head"), "_step", "denoiser", config.compile)
            ]
            if config.scope == "trunk" and getattr(model.spec, "trunk_graph", True):
                # One graph per trunk pass (first / middle / last differ in their
                # flags), the confidence head and the distogram head: everything
                # that runs on the GPU between featurisation and decoding.
                # Host-side randomness (the MC-dropout coin, the sampler's noise)
                # stays outside; only deterministic GPU work is captured.
                # These sites are graph-only: compiling the trunk gained no time
                # over the graph (Protenix1 5I28: 0.73 s vs 0.74 s) and changed
                # the result (pLDDT 73.86 vs 70.34 eager).
                sites += [
                    (model.get_submodule(name), "forward", name, False)
                    for name in config.trunk_sites
                ]
        for owner, method, site, compile_site in sites:
            site_config = config.model_copy(update={"compile": compile_site})
            denoiser = site == "denoiser"
            wrapped = ExecutedCallable(
                getattr(owner, method),
                site_config,
                f"{family}.{config.scope}.{site}",
                max_graph_tokens=None if denoiser else config.trunk_graph_max_tokens,
                epoch_key="fold_epoch" if denoiser else None,
                static_inputs=denoiser,
            )
            setattr(owner, method, wrapped)
            self.wrappers.append(wrapped)
        self.sampler_scope = None
        if config.cuda_graph and config.scope == "trunk" and config.sampler_graph:
            # The whole sampler step goes in one graph (see `models.sampler_graph`).
            self.sampler_scope = SamplerScope(self.wrappers[0])
        model.foldforge_sampler_scope = self.sampler_scope

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
            "cuda_graph": replays > 0 or getattr(self.sampler_scope, "replays", 0) > 0,
            "execution_scope": self.config.scope,
            "compiled_graphs": compiled,
            "cuda_graph_captures": sum(x.captures for x in self.wrappers),
            "cuda_graph_evictions": sum(x.evictions for x in self.wrappers),
            "cuda_graph_replays": replays,
            "wrapped_calls": sum(x.calls for x in self.wrappers),
            "sampler_graph_captures": getattr(self.sampler_scope, "captures", 0),
            "sampler_graph_replays": getattr(self.sampler_scope, "replays", 0),
            "uncaptured_calls": sum(x.uncaptured for x in self.wrappers),
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
