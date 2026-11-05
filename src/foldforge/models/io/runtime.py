# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Shared model execution, validation and output loop."""

from __future__ import annotations

import importlib
import json
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._pytree import tree_map

from foldforge.data.ccd import CCDDatabase
from foldforge.models.execution import Execution, measured_forward
from foldforge.models.io.output import Decoded, cpu_tree, json_value
from foldforge.models.msa_policy import RECORD
from foldforge.models.sampling import bind_sampling_seed
from foldforge.result import PredictionResult
from foldforge.utils.seed import RNGState, seed_all, seed_context

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.models.io.request import Request


def to_device(features: Any, device: str | torch.device) -> Any:
    """Move feature tensors without modifying the resolved input containers."""
    return tree_map(
        lambda value: value.to(device) if isinstance(value, torch.Tensor) else value,
        features,
    )


@dataclass
class Case:
    """Represent case."""

    forward: Callable[[], Any]
    decode: Callable[[Any, dict], Decoded]
    features: Any = None
    feature_name: str | None = None
    # ESM's feature decoding uses ordinary tensors after the forward.
    inference_mode: bool = True
    run_trunk: Callable[[], Any] | None = None
    sample_diffusion: Callable[[Any], Any] | None = None


class Runtime:
    """Represent runtime."""

    def __init__(
        self,
        request: Request,
        *,
        model: AlphaFold3 | None = None,
        execution: Execution | None = None,
        language_models: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.request = request
        self.execution = None
        self._model = model
        self._resident_execution = execution
        self.language_models = language_models if language_models is not None else {}
        self.device = device or torch.device(request.device or "cuda")

    def load_model(self) -> AlphaFold3:
        """Use a resident network, or load once for a legacy direct request."""
        if self._model is None:
            from foldforge.models.loading import load_configured

            if self.request.checkpoint is None:
                message = "A checkpoint is required"
                raise ValueError(message)
            self._model = load_configured(
                self.request.model,
                self.request.checkpoint,
                self.request.to_config(),
                samples=self.request.samples,
                device=self.device,
            )
        self._model.num_samples = self.request.samples
        return self._model

    def bind(self, model: torch.nn.Module) -> Execution:
        """Bind ."""
        if self.execution is not None:
            msg = "A runtime binds exactly one checkpoint"
            raise RuntimeError(msg)
        if "distogram" in self.request.output.image_names:
            image_model: Any = model
            image_model.save_distogram = True
            if hasattr(model, "distogram_head"):
                image_model.distogram_head.save_distogram = True
        bind_sampling_seed(model, self.request.diffusion_seed)
        self.execution = self._resident_execution or Execution(
            model, self.request.model, self.request.execution
        )
        self.execution.begin_request()
        return self.execution


def predict(
    request: Request,
    *,
    model: AlphaFold3 | None = None,
    execution: Execution | None = None,
    language_models: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> tuple[PredictionResult, ...]:
    """Execute the shared pipeline and return results; persistence is optional."""
    from foldforge.models import entry

    layout = entry(request.model).layout
    adapter = importlib.import_module(f"{__package__}.{layout}")
    runtime = Runtime(
        request,
        model=model,
        execution=execution,
        language_models=language_models,
        device=device,
    )
    database = CCDDatabase(request.ccd_db)
    output_dir = request.out
    if request.write_outputs and output_dir is None:
        message = "An output run directory is required"
        raise ValueError(message)
    if request.write_outputs and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    device_scope = (
        torch.cuda.device(runtime.device)
        if runtime.device.type == "cuda"
        and (device is not None or request.device is not None)
        else nullcontext()
    )
    with device_scope, seed_context(request.trunk_seed), database.activate():
        cases = adapter.prepare(request, database, runtime)
        try:
            for case in cases:
                if runtime.execution is None:
                    msg = "Adapter did not bind its checkpoint execution"
                    raise RuntimeError(msg)
                if (
                    request.write_outputs
                    and output_dir is not None
                    and case.features is not None
                ):
                    if (
                        case.feature_name is None
                        or case.feature_name != Path(case.feature_name).name
                    ):
                        msg = "Feature artifact must have a filename"
                        raise ValueError(msg)
                    torch.save(cpu_tree(case.features), output_dir / case.feature_name)
                context = torch.inference_mode if case.inference_mode else torch.no_grad
                # Model loading and feature preparation must not shift sampling RNG.
                seed_all(request.trunk_seed)
                with context():
                    output, measurements = measured_forward(
                        case.forward, request.execution.benchmark_repeats
                    )
                torch.cuda.synchronize()
                result = finish_case(
                    case,
                    output,
                    measurements,
                    request,
                    runtime.execution,
                    extra={"device": str(runtime.device)},
                )
                results.append(result)
        finally:
            cases.close()
    return tuple(results)


def finish_case(
    case: Case,
    output: Any,
    measurements: dict[str, Any],
    request: Request,
    execution: Execution,
    *,
    extra: dict[str, Any] | None = None,
) -> PredictionResult:
    """Decode/validate the same result contract for whole and staged inference."""
    decoded = case.decode(output, measurements)
    decoded.report = {
        **decoded.report,
        "trunk_seed": request.trunk_seed,
        "diffusion_seed": request.diffusion_seed,
        "seed_policy": "split-v1",
        "msa_policy": RECORD,
        "model": request.model,
        "backend": request.backend,
        "precision": request.precision,
        "samples": request.samples,
        "steps": request.steps,
        "trunk_passes": request.recycles,
        "recycles": request.recycles,
        "autocast": False,
        **execution.report(),
        **measurements,
        **(extra or {}),
    }
    result = PredictionResult.from_decoded(
        decoded, expected_samples=request.samples, images=request.output.image_names
    )
    if request.write_outputs and request.out is not None:
        result = result.save(request.out)
    return result


@dataclass
class TrunkRecord:
    """Adapter-owned features/decoder plus a retained network trunk and RNG state."""

    request: Request
    case: Case
    value: Any
    rng: RNGState
    report: dict[str, Any]
    elapsed_seconds: float


@contextmanager
def model_schedule(model: Any, request: Request) -> Iterator[None]:
    """Scope per-target pass/sample/step counts without changing the loaded policy."""
    previous = (model.num_recycles, model.num_samples, model.diffusion_steps)
    model.num_recycles, model.num_samples, model.diffusion_steps = (
        request.recycles,
        request.samples,
        request.steps,
    )
    try:
        yield
    finally:
        model.num_recycles, model.num_samples, model.diffusion_steps = previous


def run_trunk(
    request: Request,
    *,
    model: AlphaFold3,
    execution: Execution,
    language_models: dict[str, Any],
    device: torch.device,
) -> TrunkRecord:
    """Featurize and execute the trunk once, closing all global chemistry scopes."""
    from foldforge.models import entry

    start = time.monotonic()
    adapter = importlib.import_module(f"{__package__}.{entry(request.model).layout}")
    runtime = Runtime(
        request,
        model=model,
        execution=execution,
        language_models=language_models,
        device=device,
    )
    database = CCDDatabase(request.ccd_db)
    with (
        seed_context(request.trunk_seed),
        database.activate(),
        model_schedule(model, request),
    ):
        cases = adapter.prepare(request, database, runtime)
        try:
            case = next(cases)
            if case.run_trunk is None or case.sample_diffusion is None:
                message = "The selected adapter does not provide staged inference"
                raise RuntimeError(message)
            seed_all(request.trunk_seed)
            context = torch.inference_mode if case.inference_mode else torch.no_grad
            with context():
                value, measurements = measured_forward(
                    case.run_trunk, request.execution.benchmark_repeats
                )
            torch.cuda.synchronize()
            rng = RNGState.capture()
            elapsed = time.monotonic() - start
            report = {
                "model": request.model,
                "device": str(device),
                "stage": "trunk",
                "trunk_seed": request.trunk_seed,
                "trunk_passes": request.recycles,
                "trunk_elapsed_seconds": elapsed,
                "trunk_measurements": measurements,
                **execution.report(require_compile=False),
            }
            return TrunkRecord(request, case, value, rng, report, elapsed)
        finally:
            cases.close()


def sample_diffusion(
    record: TrunkRecord, request: Request, *, model: AlphaFold3, execution: Execution
) -> PredictionResult:
    """Resample a retained trunk with independent diffusion randomness and heads."""
    start = time.monotonic()
    sample = record.case.sample_diffusion
    if sample is None:
        message = "This trunk has no sampling callback"
        raise RuntimeError(message)
    database = CCDDatabase(request.ccd_db)
    with (
        seed_context(request.trunk_seed),
        database.activate(),
        model_schedule(model, request),
    ):
        record.rng.restore()
        bind_sampling_seed(model, request.diffusion_seed)
        execution.begin_request()
        with torch.no_grad():
            output, measurements = measured_forward(
                lambda: sample(record.value),
                request.execution.benchmark_repeats,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        return finish_case(
            record.case,
            output,
            measurements,
            request,
            execution,
            extra={
                "stage": "diffusion",
                "device": record.report["device"],
                "trunk_reused": True,
                "trunk_execution": record.report,
                "diffusion_elapsed_seconds": elapsed,
                "elapsed_seconds": record.elapsed_seconds + elapsed,
                "trunk_passes": request.recycles,
            },
        )


def run(request: Request) -> int:
    """Compatibility CLI writer around the library execution primitive."""
    for result in predict(request):
        print(  # noqa: T201 - compatibility CLI output
            json.dumps(json_value(result.report), indent=2, allow_nan=False), flush=True
        )
    return 0
