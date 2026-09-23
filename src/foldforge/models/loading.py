# ruff: noqa: PLC0415 - checkpoint format dependencies are imported lazily
"""One construction, checkpoint validation, precision and backend lifecycle.

Architecture classes own network topology. Weight readers own serialization and
key conversion. Neither of them chooses runtime precision or installs backends.
"""

from __future__ import annotations

import dataclasses
import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from foldforge.models import entry
from foldforge.models.checkpoints import DEFAULT_FILES, resolve
from foldforge.models.config import PROTENIX_FAMILIES, VARIANTS
from foldforge.models.msa_policy import RECORD


def clear_native_compute_override(model: Any, dtype: Any) -> None:
    """Under native BF16, drop a projection's FP32 compute override.

    Native BF16 must apply to the GEMM, not only the stored weights: a released
    geometry or conditioning projection may carry an FP32 compute override, and
    left in place it keeps the matmul in FP32 while the weights say otherwise.
    Nothing here reads the layout -- the rule is the same wherever it applies.
    """
    if dtype != torch.bfloat16:
        return
    from team_gm.modules.precision import NativeLinear

    for layer in model.modules():
        if isinstance(layer, NativeLinear):
            layer.compute_dtype = None


def resolve_family(
    name: str, variant: str | None = None, configs: object | None = None
) -> tuple[str | None, str | None]:
    """Resolve a model name (and release) to its dense family and variant.

    One model publishes several releases and the release is the family, so the
    two are resolved together. Callers that need the family BEFORE building the
    model -- the input adapters, which featurise differently for a family that
    folds on structural tokens -- ask here rather than loading the checkpoint.
    """
    if variant is not None and name != "protenix":
        message = "variant applies only to Protenix"
        raise ValueError(message)
    family = entry(name).family
    if name == "protenix":
        variant = (
            variant
            or (getattr(configs, "model_name", None) if configs is not None else None)
            or VARIANTS[0]
        )
        if variant not in VARIANTS:
            message = f"Unsupported Protenix variant {variant!r}; choose {VARIANTS}"
            raise ValueError(message)
        family = PROTENIX_FAMILIES[variant]
    return family, variant


def load_checkpoint(  # noqa: C901, PLR0912, PLR0915 - one explicit checkpoint lifecycle
    name: str,
    checkpoint: str | Path | None = None,
    *,
    backend: str = "miniworld",
    dtype: torch.dtype | None = None,
    device: str | torch.device = "cuda",
    variant: str | None = None,
    configs: Any = None,
    recycles: int = 10,
    samples: int = 5,
    steps: int = 200,
    implementation: str | None = None,
    precision_policy: str | None = None,
) -> torch.nn.Module:
    """Strictly load weights, then apply the same declared inference contract."""
    if implementation is not None:
        backend = getattr(implementation, "value", implementation)
    if backend == "miniworld_engine":
        backend = "miniworld"
    if backend not in {"miniworld", "pytorch", "cuequivariance"}:
        message = f"Unknown backend: {backend}"
        raise ValueError(message)
    if precision_policy == "model_default":
        precision_policy = "af3_default"
        dtype = torch.bfloat16
    dtype = torch.bfloat16 if dtype is None else dtype
    if precision_policy not in {None, "bf16", "fp32", "af3_default", "model_default"}:
        message = f"Unsupported precision policy: {precision_policy}"
        raise ValueError(message)
    if precision_policy == "af3_default" and dtype != torch.bfloat16:
        message = "af3_default requires a BF16 trunk"
        raise ValueError(message)

    spec = entry(name)
    if spec.architecture is None:
        raise NotImplementedError(name)
    family, variant = resolve_family(name, variant, configs)
    if checkpoint is None:
        filename = {
            **DEFAULT_FILES,
            **({"protenix": f"{variant}.bin.zst"} if name == "protenix" else {}),
        }[name]
        checkpoint = resolve(name, filename)
    checkpoint = Path(checkpoint)
    package, _, symbol = spec.architecture.rpartition(".")
    architecture = getattr(import_module(package), symbol)
    report = {"checkpoint": str(checkpoint), "strict": True}

    if spec.layout != "dense_atoms":
        message = f"{name} has no loader: every family is a row of the dense graph"
        raise NotImplementedError(message)
    from foldforge.models.checkpoints.haiku import import_jax_weights_
    from foldforge.modules.dense.spec import SPECS

    dense_spec = SPECS[family or "alphafold3"]
    override = os.environ.get("FOLDFORGE_DENSE_SPEC_OVERRIDE")
    if override:
        # Porting aid: flip forward conventions of a family to price each one.
        # Fields that change a parameter shape make the strict load fail loudly.
        dense_spec = dataclasses.replace(dense_spec, **json.loads(override))
        report["dense_spec_override"] = override
    model = architecture(
        num_recycles=recycles,
        num_samples=samples,
        diffusion_steps=steps,
        spec=dense_spec,
    )
    report["family"] = dense_spec.family
    report.update(
        import_jax_weights_(model, checkpoint, preserve_dtype=True)
        if precision_policy == "af3_default"
        else import_jax_weights_(model, checkpoint)
    )

    setattr(model, "foldforge_load_report", report)  # noqa: B010 - runtime metadata
    from team_gm.modules.checkpoints.af_family import configure_model
    from team_gm.modules.checkpoints.conditioning import install_conditioning
    from team_gm.modules.checkpoints.pairformer import install_pairformers

    source = model.get_submodule("diffusion_head.fourier_embeddings")
    fourier_fp32 = (source.weight.float().clone(), source.bias.float().clone())
    if precision_policy == "af3_default":
        model.reference_precision = True
        # Configure shared wrappers in FP32, preserving every released FP32
        # value. Restore only originally BF16 tensors; their round trip via
        # FP32 is exact. This also preserves FP32 input/output projections.
        released_dtypes = {id(p): p.dtype for p in model.parameters()}
        model = configure_model(model, backend, torch.float32, device)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(released_dtypes[id(parameter)])
    else:
        model = configure_model(model, backend, dtype, device)
    # Noise Fourier features stay FP32 whatever the parameter dtype.
    fourier = model.get_submodule("diffusion_head.fourier_embeddings")
    fourier.weight = fourier_fp32[0].to(device)
    fourier.bias = fourier_fp32[1].to(device)
    model = install_conditioning(install_pairformers(model))
    if dtype == torch.bfloat16 and precision_policy != "af3_default":
        clear_native_compute_override(model, dtype)
    if precision_policy == "model_default":
        from foldforge.models.precision import reference_precision

        model = reference_precision(model, name)
        report.update(
            precision_policy="model_default",
            autocast_scopes=model.reference_autocast_scopes,
        )
    report.update(
        backend=backend,
        parameter_dtype=str(dtype),
        device=str(device),
        msa_policy=RECORD,
    )
    if precision_policy == "af3_default":
        report.update(
            precision_policy="af3_default",
            parameter_dtype="mixed (released checkpoint dtypes)",
            trunk_parameter_dtype="torch.bfloat16",
            diffusion_parameter_dtype="torch.float32",
        )
    setattr(model, "foldforge_load_report", report)  # noqa: B010 - runtime metadata
    return model
