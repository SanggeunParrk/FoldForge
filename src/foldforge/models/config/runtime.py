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
    # Pad tokens, atoms and MSA rows up a fixed ladder (models.bucketing). On by default: without it the
    # dense atom layout (24 slots per token) makes the atom blocks compute ~3x the real atoms.
    bucketing: bool = True
    # float32 matmuls on TF32 tensor cores (torch "high") instead of full FP32 ("highest").
    tf32: bool = False
    scope: Literal["denoiser", "trunk", "model"] = "denoiser"
    max_graphs: int = Field(default=4, ge=1, le=32)
    # scope=trunk only: above this many tokens the trunk and head sites run uncaptured (same kernels).
    # There the GPU is already full and the graph pools would only take memory from the denoiser.
    trunk_graph_max_tokens: int = Field(default=512, ge=0)
    # scope=trunk only: the sampler step (re-pose, re-noise, denoise, align, update) as one graph.
    sampler_graph: bool = True
    # scope=trunk only: which trunk-side callables get a graph of their own.
    trunk_sites: tuple[Literal["evoformer", "confidence_head", "distogram_head"], ...] = (
        "evoformer",
        "confidence_head",
        "distogram_head",
    )
    benchmark_repeats: int = Field(default=0, ge=0, le=20)

    @model_validator(mode="after")
    def graph_boundary(self) -> ExecutionConfig:
        """Keep host-side random sampling outside captured GPU work."""
        if self.cuda_graph and self.scope == "model":
            message = (
                "CUDA graphs require scope=denoiser or trunk; "
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
    #: "fast" folds with AF3's computation wherever a release's own folds the
    #: same; "exact" keeps every release convention (DenseSpec.EXACT_CONVENTIONS).
    mode: Literal["fast", "exact"] = "fast"

    @model_validator(mode="before")
    @classmethod
    def legacy_seed(cls, values: object) -> object:
        """Translate old seed-only configs without ambiguous mixed policies."""
        if isinstance(values, dict) and "variant" in values:
            message = (
                "`variant` is gone: each release is its own model "
                "(fold protenix1 or protenix2)"
            )
            raise ValueError(message)
        if isinstance(values, dict) and "seed" in values:
            if "trunk_seed" in values or "diffusion_seed" in values:
                message = "Use either seed or trunk_seed/diffusion_seed, not both"
                raise ValueError(message)
            values = dict(values)
            seed = values.pop("seed")
            values.update(trunk_seed=seed, diffusion_seed=seed)
        return values
