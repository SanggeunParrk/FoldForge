"""Typed request passed directly from the public CLI to checkpoint adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from foldforge.data.inputs.validation import validate_inference_seed
from foldforge.models import is_dense, registered_models
from foldforge.models.config import ExecutionConfig, OutputConfig
from foldforge.models.io.paths import run_directory
from foldforge.models.msa_policy import PREPARED_ROWS

if TYPE_CHECKING:
    from pathlib import Path

    from foldforge.data.inputs.build import Input


@dataclass(frozen=True)
class Request:
    """Represent request."""

    model: str
    ccd_db: Path
    out: Path | None = None
    checkpoint: Path | None = None
    resolved_input: Input | None = None
    input: Path | None = None
    backend: str = "miniworld"
    precision: str = "bf16"
    trunk_seed: int = 0
    diffusion_seed: int = 0
    recycles: int | None = None
    steps: int | None = None
    samples: int = 1
    msa_depth: int = PREPARED_ROWS
    guidance: bool | None = None
    templates: bool = False
    no_msa: bool = False
    variant: str = "protenix_base_default_v1.0.0"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    target: str = "1ubq"

    def __post_init__(self) -> None:
        for name in ("trunk_seed", "diffusion_seed"):
            object.__setattr__(self, name, validate_inference_seed(getattr(self, name)))
        object.__setattr__(self, "out", run_directory(self.out, model=self.model))
        if self.model not in registered_models():
            msg = f"Unsupported checkpoint adapter: {self.model}"
            raise ValueError(msg)
        if self.backend not in {"miniworld", "pytorch", "cuequivariance"}:
            msg = f"Unsupported backend: {self.backend}"
            raise ValueError(msg)
        if self.precision not in {"bf16", "fp32", "af3_default", "model_default"}:
            msg = f"Unsupported precision: {self.precision}"
            raise ValueError(msg)
        if self.precision == "af3_default" and not is_dense(self.model):
            msg = "af3_default precision applies only to dense AF3-graph families"
            raise ValueError(msg)
        if any(
            v is not None and v < 1
            for v in (self.samples, self.recycles, self.steps, self.msa_depth)
        ):
            msg = "samples, recycles, steps and msa-depth must be positive"
            raise ValueError(msg)
        if self.input is None or self.checkpoint is None:
            msg = "A request names an input and a checkpoint"
            raise ValueError(msg)
        if self.recycles is None or self.steps is None:
            msg = "A request states its recycles and steps"
            raise ValueError(msg)
        if (
            self.resolved_input is not None
            and self.resolved_input.spec.ccd_db.resolve() != self.ccd_db.resolve()
        ):
            msg = "Spec and selected CCD database disagree"
            raise ValueError(msg)
