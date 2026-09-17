# ruff: noqa: PLC0415 - checkpoint format dependencies are imported lazily
"""One construction, checkpoint validation, precision and backend lifecycle.

Architecture classes own network topology. Weight readers own serialization and
key conversion. Neither of them chooses runtime precision or installs backends.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from team_gm.modules.checkpoints.layers import skip_random_init

from foldforge.models import entry
from foldforge.models.checkpoints import resolve
from foldforge.models.config import VARIANTS, configuration
from foldforge.models.msa_policy import RECORD, apply_esmfold2, apply_flat


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
        if name == "af3":
            precision_policy = "af3_default"
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    dtype = torch.bfloat16 if dtype is None else dtype
    if precision_policy not in {None, "bf16", "fp32", "af3_default", "model_default"}:
        message = f"Unsupported precision policy: {precision_policy}"
        raise ValueError(message)
    if precision_policy == "af3_default" and (name != "af3" or dtype != torch.bfloat16):
        message = "af3_default requires AF3 with a BF16 trunk"
        raise ValueError(message)

    spec = entry(name)
    if spec.architecture is None:
        raise NotImplementedError(name)
    if variant is not None and name != "protenix":
        message = "variant applies only to Protenix"
        raise ValueError(message)
    if name == "protenix":
        variant = (
            variant
            or (getattr(configs, "model_name", None) if configs is not None else None)
            or VARIANTS[0]
        )
        if variant not in VARIANTS:
            message = f"Unsupported Protenix variant {variant!r}; choose {VARIANTS}"
            raise ValueError(message)
    if checkpoint is None:
        filename = {
            "af3": "af3.bin.zst",
            "opendde": "opendde.pt",
            "protenix": f"{variant}.pt",
            "esmfold2": None,
        }[name]
        checkpoint = resolve(name) if filename is None else resolve(name, filename)
    checkpoint = Path(checkpoint)
    package, _, symbol = spec.architecture.rpartition(".")
    architecture = getattr(import_module(package), symbol)
    report = {"checkpoint": str(checkpoint), "strict": True}

    if spec.layout == "dense_atoms":
        from foldforge.models.checkpoints.haiku import import_jax_weights_

        model = architecture(
            num_recycles=recycles, num_samples=samples, diffusion_steps=steps
        )
        report.update(
            import_jax_weights_(model, checkpoint, preserve_dtype=True)
            if precision_policy == "af3_default"
            else import_jax_weights_(model, checkpoint)
        )
    else:
        if spec.layout == "sequence_atoms":
            from safetensors.torch import load_file
            from team_gm.modules.exceptions import ImplementationType

            from foldforge.models.checkpoints.esmfold2 import convert_model
            from foldforge.models.config.esmfold2 import ESMFold2Config

            configs = (
                ESMFold2Config.from_json(checkpoint) if configs is None else configs
            )
            configs = apply_esmfold2(configs)
            implementation_type = (
                ImplementationType.MINIWORLD_ENGINE
                if backend == "miniworld"
                else ImplementationType(backend)
            )
            model = architecture(configs, implementation_type)
            state = convert_model(load_file(checkpoint / "model.safetensors"), configs)
        else:
            configs = configuration(name, variant) if configs is None else configs
            configs = apply_flat(configs)
            if name == "opendde":
                configs.triangle_multiplicative = "torch"
                configs.triangle_attention = "torch"
            with skip_random_init():
                model = architecture(configs)
            blob = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            state = (
                blob["model"]
                if name == "protenix"
                else blob.get("model", blob.get("state_dict", blob))
            )
            state = {k.removeprefix("module."): v for k, v in state.items()}
            report["variant"] = getattr(configs, "model_name", "opendde_v1")
        model.load_state_dict(state, strict=True)
        report["state_entries"] = len(state)

    setattr(model, "foldforge_load_report", report)  # noqa: B010 - runtime metadata
    if spec.layout == "sequence_atoms":
        from foldforge.models.precision import inference_precision

        model = inference_precision(model, torch.device(device), dtype)
    else:
        from team_gm.modules.checkpoints.af_family import configure_model
        from team_gm.modules.checkpoints.conditioning import install_conditioning
        from team_gm.modules.checkpoints.pairformer import install_pairformers

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
        if spec.layout == "dense_atoms":
            from foldforge.models.config.fourier import _BIAS, _WEIGHT

            fourier = model.get_submodule("diffusion_head.fourier_embeddings")
            fourier.weight = torch.tensor(_WEIGHT, device=device)
            fourier.bias = torch.tensor(_BIAS, device=device)
        model = install_conditioning(install_pairformers(model))
    if dtype == torch.bfloat16 and precision_policy != "af3_default":
        from team_gm.modules.precision import NativeLinear

        # Native BF16 must apply to the GEMM, not only stored weights. Upstream
        # geometry/conditioning projections may carry an FP32 compute override.
        for layer in model.modules():
            if isinstance(layer, NativeLinear):
                layer.compute_dtype = None
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
