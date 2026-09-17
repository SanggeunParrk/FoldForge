"""Typed request passed directly from the public CLI to checkpoint adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from foldforge.data.inputs.validation import validate_inference_seed
from foldforge.models.config import ExecutionConfig, OutputConfig
from foldforge.models.io.paths import run_directory
from foldforge.models.msa_policy import PREPARED_ROWS

if TYPE_CHECKING:
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
    # Language-model inputs and optional structure evaluation.
    target: str = "1ubq"
    data_root: Path = Path("validation/inputs/data")
    input_spec: Path | None = None
    lm_cache: Path | None = None
    lm_checkpoint: Path | None = None
    lm_source: str = "compute"
    compare: Path | None = None
    experimental: Path | None = None
    chain_map: str | None = None

    def __post_init__(self) -> None:
        for name in ("trunk_seed", "diffusion_seed"):
            object.__setattr__(self, name, validate_inference_seed(getattr(self, name)))
        object.__setattr__(self, "out", run_directory(self.out, model=self.model))
        if self.model not in {"af3", "esmfold2", "protenix", "opendde"}:
            msg = f"Unsupported checkpoint adapter: {self.model}"
            raise ValueError(msg)
        if self.backend not in {"miniworld", "pytorch", "cuequivariance"}:
            msg = f"Unsupported backend: {self.backend}"
            raise ValueError(msg)
        if self.precision not in {"bf16", "fp32", "af3_default", "model_default"}:
            msg = f"Unsupported precision: {self.precision}"
            raise ValueError(msg)
        if self.precision == "af3_default" and self.model != "af3":
            msg = "af3_default precision applies only to AF3"
            raise ValueError(msg)
        if any(
            v is not None and v < 1
            for v in (self.samples, self.recycles, self.steps, self.msa_depth)
        ):
            msg = "samples, recycles, steps and msa-depth must be positive"
            raise ValueError(msg)
        if self.model != "esmfold2" and (self.input is None or self.checkpoint is None):
            msg = "This checkpoint layout requires input and checkpoint paths"
            raise ValueError(msg)
        if self.model != "esmfold2" and (self.recycles is None or self.steps is None):
            msg = "This checkpoint layout requires explicit recycles and steps"
            raise ValueError(msg)
        if (
            self.resolved_input is not None
            and self.resolved_input.spec.ccd_db.resolve() != self.ccd_db.resolve()
        ):
            msg = "Spec and selected CCD database disagree"
            raise ValueError(msg)

    @property
    def dtype(self) -> str:
        """Compute dtype."""
        return "float32" if self.precision in {"fp32", "model_default"} else "bfloat16"

    @property
    def implementation(self) -> str:
        """Compute implementation."""
        return "miniworld_engine" if self.backend == "miniworld" else self.backend
