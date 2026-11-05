"""Typed request passed directly from the public CLI to checkpoint adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from foldforge.data.inputs.validation import validate_inference_seed
from foldforge.models import is_dense, registered_models
from foldforge.models.config import (
    Config,
    DebugConfig,
    EngineConfig,
    ExecutionConfig,
    OutputConfig,
)
from foldforge.models.io.paths import output_directory, run_directory
from foldforge.models.msa_policy import PREPARED_ROWS
from foldforge.modules.engine.backends import BACKENDS

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
    mode: str = "fast"
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    target: str = "1ubq"
    input_data: dict[str, Any] | None = None
    write_outputs: bool = True
    verbose: bool = True
    output_policy: Literal["runs", "path"] = "runs"
    device: str | None = None

    def __post_init__(self) -> None:
        for name in ("trunk_seed", "diffusion_seed"):
            object.__setattr__(self, name, validate_inference_seed(getattr(self, name)))
        destination = self._destination()
        object.__setattr__(self, "out", destination)
        if self.model not in registered_models():
            msg = f"Unsupported checkpoint adapter: {self.model}"
            raise ValueError(msg)

        if self.backend not in BACKENDS:
            msg = f"Unsupported backend: {self.backend}"
            raise ValueError(msg)
        if self.precision not in {"bf16", "fp32", "af3_default", "model_default"}:
            msg = f"Unsupported precision: {self.precision}"
            raise ValueError(msg)
        if self.precision == "af3_default" and not is_dense(self.model):
            msg = "af3_default precision applies only to dense AF3-graph families"
            raise ValueError(msg)
        if any(
            v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1)
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

    def _destination(self) -> Path | None:
        if self.output_policy == "path":
            return output_directory(self.out) if self.out is not None else None
        return (
            run_directory(self.out, model=self.model)
            if self.write_outputs or self.out is not None
            else None
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        model: str,
        ccd_db: Path,
        checkpoint: Path,
        input: Path,  # noqa: A002 - existing Request field and adapter contract
        samples: int,
        out: Path | None = None,
        resolved_input: Input | None = None,
        target: str = "prediction",
        input_data: dict[str, Any] | None = None,
        write_outputs: bool = True,
        verbose: bool = True,
        output_policy: Literal["runs", "path"] = "runs",
        device: str | None = None,
    ) -> Request:
        """Translate resolved configuration into runtime fields in one place."""
        shared = {
            name: getattr(config, name)
            for name in (
                "backend",
                "precision",
                "trunk_seed",
                "diffusion_seed",
                "mode",
                "execution",
                "output",
                "engine",
                "debug",
            )
        }
        return cls(
            model=model,
            ccd_db=ccd_db,
            checkpoint=checkpoint,
            input=input,
            samples=samples,
            out=out,
            resolved_input=resolved_input,
            target=target,
            recycles=config.trunk.recycles if config.trunk.recycles is not None else 10,
            steps=config.diffusion.steps if config.diffusion.steps is not None else 200,
            msa_depth=config.trunk.msa_depth
            if config.trunk.msa_depth is not None
            else PREPARED_ROWS,
            input_data=input_data,
            write_outputs=write_outputs,
            verbose=verbose,
            output_policy=output_policy,
            device=device,
            **shared,
        )

    def to_config(self) -> Config:
        """Recover the validated model policy for legacy direct requests."""
        shared = {
            name: getattr(self, name)
            for name in (
                "backend",
                "precision",
                "trunk_seed",
                "diffusion_seed",
                "mode",
                "execution",
                "output",
                "engine",
                "debug",
            )
        }
        return Config.model_validate(
            {
                **shared,
                "trunk": {"recycles": self.recycles, "msa_depth": self.msa_depth},
                "diffusion": {"steps": self.steps},
            }
        )
