"""Nested Pydantic runtime configuration, following MiniWorld model configs."""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from foldforge.data.inputs.validation import validate_inference_seed
from foldforge.modules.engine.backends import BACKENDS, validate_ops

#: Environment variables that used to steer behaviour and are now config keys (or gone). Setting one is an
#: error, not a silent no-op: the run would otherwise differ from what its environment says.
REMOVED_ENVIRONMENT = {
    "FF_ENGINE_SKIP": "set `engine.skip` in the config (docs/guides/CONFIG.md)",
    "FF_TRUNK_BF16": "removed: the experiment is concluded (af3_default keeps the released dtypes)",
    "FOLDFORGE_DENSE_SPEC_OVERRIDE": "set `debug.dense_spec_override` in the config (docs/guides/CONFIG.md)",
    "FOLDFORGE_CONFIDENCE_SAMPLE_CHUNK": "set `execution.confidence_sample_chunk` in the config (docs/guides/CONFIG.md)",
}


def reject_removed_environment() -> None:
    """Raise when an environment variable that used to change behaviour is still set."""
    stale = {name: hint for name, hint in REMOVED_ENVIRONMENT.items() if name in os.environ}
    if stale:
        lines = "; ".join(f"{name}: {hint}" for name, hint in stale.items())
        message = f"environment variables no longer read -- {lines}"
        raise RuntimeError(message)


#: Parts of the engine install that ``engine.skip`` can leave dense (the keys of the load report's counts).
ENGINE_SKIP_KEYS = (
    "pairformers",
    "conditioning",
    "engine_trimul",
    "engine_triangle_attention",
    "engine_msa_pwa",
    "engine_opm",
    "engine_dit_blocks",
    "engine_single_attention",
    "engine_atom_blocks",
    "engine_dit_variant_blocks",
    "engine_atom_variant_blocks",
)


class EngineConfig(BaseModel):
    """What the engine install touches: the debugging ``skip`` and the per-op backend choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    #: Parts left dense (PyTorch) instead of engine-served; each is one of ``ENGINE_SKIP_KEYS``.
    skip: tuple[Literal[ENGINE_SKIP_KEYS], ...] = ()  # type: ignore[valid-type]
    #: Ops that take another backend than ``Config.backend``: ``{op: backend}``, ops from ``backends.OPS``.
    ops: dict[str, str] = Field(default_factory=dict)
    #: An op the backend has no kernel for: ``fallback`` runs the PyTorch reference and logs one line, ``error`` raises.
    unsupported: Literal["fallback", "error"] = "fallback"

    @field_validator("ops")
    @classmethod
    def known_ops(cls, ops: dict[str, str]) -> dict[str, str]:
        """Only known ops and backends."""
        return validate_ops(ops)


class DebugConfig(BaseModel):
    """Porting aids; empty by default."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    #: ``DenseSpec`` fields replaced after the family's spec is chosen, to price one forward
    #: convention at a time. Fields that change a parameter shape make the strict load fail loudly.
    dense_spec_override: dict[str, Any] | None = None


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
    # Samples per confidence-head call; None = the head's own rule (all while their pair stream is small,
    # one per call at large token counts; ConfidenceHead.sample_chunk). 1 is the sequential path.
    confidence_sample_chunk: int | None = Field(default=None, ge=1)

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
    backend: Literal[BACKENDS] = "miniworld"  # type: ignore[valid-type]
    precision: Literal["bf16", "fp32", "af3_default", "model_default"] = "bf16"
    trunk_seed: Annotated[int, BeforeValidator(validate_inference_seed)] = 0
    diffusion_seed: Annotated[int, BeforeValidator(validate_inference_seed)] = 0
    trunk: TrunkConfig = Field(default_factory=TrunkConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    #: "fast" folds with AF3's computation wherever a release's own folds the
    #: same; "exact" keeps every release convention (DenseSpec.EXACT_CONVENTIONS).
    mode: Literal["fast", "exact"] = "fast"

    @model_validator(mode="before")
    @classmethod
    def legacy_seed(cls, values: object) -> object:
        """Translate old seed-only configs without ambiguous mixed policies."""
        reject_removed_environment()
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
