"""Typed request passed directly from the public CLI to checkpoint adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from foldforge.models.config import ExecutionConfig
from foldforge.models.io.paths import run_directory

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
    seed: int = 0
    recycles: int | None = None
    steps: int | None = None
    samples: int = 1
    msa_depth: int = 512
    guidance: bool | None = None
    templates: bool = False
    no_msa: bool = False
    variant: str = "protenix_base_default_v1.0.0"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
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
        object.__setattr__(self, "out", run_directory(self.out, model=self.model))
        if self.model not in {"af3", "esmfold2", "protenix", "opendde"}:
            msg = f"Unsupported checkpoint adapter: {self.model}"
            raise ValueError(msg)
        if self.backend not in {"miniworld", "pytorch", "cuequivariance"}:
            msg = f"Unsupported backend: {self.backend}"
            raise ValueError(msg)
        if self.precision not in {"bf16", "fp32"}:
            msg = f"Unsupported precision: {self.precision}"
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
        return "bfloat16" if self.precision == "bf16" else "float32"

    @property
    def implementation(self) -> str:
        """Compute implementation."""
        return "miniworld_engine" if self.backend == "miniworld" else self.backend
