# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Shared model execution, validation and output loop."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._pytree import tree_map

from foldforge.data.ccd import CCDDatabase
from foldforge.models.execution import Execution, measured_forward
from foldforge.models.io.output import Decoded, cpu_tree, json_value, write_output
from foldforge.models.sampling import bind_sampling_seed
from foldforge.utils.seed import seed_all, seed_context

if TYPE_CHECKING:
    from collections.abc import Callable

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


class Runtime:
    """Represent runtime."""

    def __init__(self, request: Request) -> None:
        self.request = request
        self.execution = None

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
        bind_sampling_seed(model, self.request.model, self.request.diffusion_seed)
        self.execution = Execution(model, self.request.model, self.request.execution)
        return self.execution


def run(request: Request) -> int:
    """Run the configured operation."""
    from foldforge.models import entry

    layout = entry(request.model).layout
    adapter = importlib.import_module(f"{__package__}.{layout}")
    runtime = Runtime(request)
    database = CCDDatabase(request.ccd_db)
    if request.out is None:
        message = "An output run directory is required"
        raise ValueError(message)
    request.out.mkdir(parents=True, exist_ok=True)
    with seed_context(request.trunk_seed), database.activate():
        cases = adapter.prepare(request, database, runtime)
        try:
            for case in cases:
                if runtime.execution is None:
                    msg = "Adapter did not bind its checkpoint execution"
                    raise RuntimeError(msg)
                if case.features is not None:
                    if (
                        case.feature_name is None
                        or case.feature_name != Path(case.feature_name).name
                    ):
                        msg = "Feature artifact must have a filename"
                        raise ValueError(msg)
                    torch.save(cpu_tree(case.features), request.out / case.feature_name)
                context = torch.inference_mode if case.inference_mode else torch.no_grad
                # Model loading and feature preparation must not shift sampling RNG.
                seed_all(request.trunk_seed)
                with context():
                    output, measurements = measured_forward(
                        case.forward, request.execution.benchmark_repeats
                    )
                torch.cuda.synchronize()
                decoded = case.decode(output, measurements)
                decoded.report = {
                    **decoded.report,
                    "trunk_seed": request.trunk_seed,
                    "diffusion_seed": request.diffusion_seed,
                    "seed_policy": "split-v1",
                    "model": request.model,
                    "backend": request.backend,
                    "precision": request.precision,
                    "autocast": False,
                    **runtime.execution.report(),
                    **measurements,
                }
                report = write_output(
                    decoded,
                    request.out,
                    expected_samples=request.samples,
                    images=request.output.image_names,
                )
                print(  # noqa: T201 - CLI output contract
                    json.dumps(json_value(report), indent=2, allow_nan=False),
                    flush=True,
                )
        finally:
            cases.close()
    return 0
