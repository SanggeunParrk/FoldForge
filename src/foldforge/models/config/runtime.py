"""Nested Pydantic runtime configuration, following MiniWorld model configs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrunkConfig(BaseModel):
    """Released trunk recurrence and alignment depth overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    recycles: int | None = Field(default=None, ge=1)
    msa_depth: int | None = Field(default=None, ge=1)


class DiffusionConfig(BaseModel):
    """Released diffusion sampling schedule overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    steps: int | None = Field(default=None, ge=1)


class ExecutionConfig(BaseModel):
    """Select a measured execution boundary; capture failure is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    compile: bool = False
    cuda_graph: bool = False
    bucketing: bool = False
    scope: Literal["denoiser", "model"] = "denoiser"
    max_graphs: int = Field(default=4, ge=1, le=32)
    benchmark_repeats: int = Field(default=0, ge=0, le=20)

    @model_validator(mode="after")
    def graph_boundary(self) -> ExecutionConfig:
        """Keep host-side random sampling outside captured GPU work."""
        if self.cuda_graph and self.scope == "model":
            message = (
                "CUDA graphs require scope=denoiser; "
                "full-model host sampling cannot be captured"
            )
            raise ValueError(message)
        return self


class Config(BaseModel):
    """Execution policy is separate from target data and released weight layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: Literal["miniworld", "pytorch", "cuequivariance"] = "miniworld"
    precision: Literal["bf16", "fp32", "af3_default", "model_default"] = "bf16"
    seed: int = 0
    trunk: TrunkConfig = Field(default_factory=TrunkConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    variant: str | None = None
