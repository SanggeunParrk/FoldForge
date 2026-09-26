"""Nested Pydantic runtime configuration, following MiniWorld model configs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from foldforge.data.inputs.validation import validate_inference_seed


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


IMAGE_KINDS = ("pae", "pde", "distogram", "plddt", "msa", "template")
ImageKind = Literal["pae", "pde", "distogram", "plddt", "msa", "template", "all"]


class OutputConfig(BaseModel):
    """Opt-in diagnostic PNGs, separate from inference measurements."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    images: tuple[ImageKind, ...] = ()

    @property
    def image_names(self) -> tuple[str, ...]:
        """Expand all and remove duplicates in a stable order."""
        return (
            IMAGE_KINDS if "all" in self.images else tuple(dict.fromkeys(self.images))
        )


class Config(BaseModel):
    """Execution policy is separate from target data and released weight layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    backend: Literal["miniworld", "pytorch", "cuequivariance"] = "miniworld"
    precision: Literal["bf16", "fp32", "af3_default", "model_default"] = "bf16"
    trunk_seed: Annotated[int, BeforeValidator(validate_inference_seed)] = 0
    diffusion_seed: Annotated[int, BeforeValidator(validate_inference_seed)] = 0
    trunk: TrunkConfig = Field(default_factory=TrunkConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    variant: str | None = None
    #: "fast" folds with AF3's computation wherever a release's own folds the
    #: same; "exact" keeps every release convention (DenseSpec.EXACT_CONVENTIONS).
    mode: Literal["fast", "exact"] = "fast"

    @model_validator(mode="before")
    @classmethod
    def legacy_seed(cls, values: object) -> object:
        """Translate old seed-only configs without ambiguous mixed policies."""
        if isinstance(values, dict) and "seed" in values:
            if "trunk_seed" in values or "diffusion_seed" in values:
                message = "Use either seed or trunk_seed/diffusion_seed, not both"
                raise ValueError(message)
            values = dict(values)
            seed = values.pop("seed")
            values.update(trunk_seed=seed, diffusion_seed=seed)
        return values
