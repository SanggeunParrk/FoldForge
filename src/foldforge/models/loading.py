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
) -> torch.nn.Module:
    """Strictly load weights, then apply the same declared inference contract."""
    if implementation is not None:
        backend = getattr(implementation, "value", implementation)
    if backend == "miniworld_engine":
        backend = "miniworld"
    if backend not in {"miniworld", "pytorch", "cuequivariance"}:
        message = f"Unknown backend: {backend}"
        raise ValueError(message)
    dtype = torch.bfloat16 if dtype is None else dtype
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
        report.update(import_jax_weights_(model, checkpoint))
    else:
        if spec.layout == "sequence_atoms":
            from safetensors.torch import load_file
            from team_gm.modules.exceptions import ImplementationType

            from foldforge.models.checkpoints.esmfold2 import convert_model
            from foldforge.models.config.esmfold2 import ESMFold2Config

            configs = (
                ESMFold2Config.from_json(checkpoint) if configs is None else configs
            )
            model = architecture(configs, ImplementationType(backend))
            state = convert_model(load_file(checkpoint / "model.safetensors"), configs)
        else:
            configs = configuration(name, variant) if configs is None else configs
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

        model = configure_model(model, backend, dtype, device)
        if spec.layout == "dense_atoms":
            from foldforge.models.config.fourier import _BIAS, _WEIGHT

            fourier = model.get_submodule("diffusion_head.fourier_embeddings")
            fourier.weight = torch.tensor(_WEIGHT, device=device)
            fourier.bias = torch.tensor(_BIAS, device=device)
        model = install_conditioning(install_pairformers(model))
    report.update(backend=backend, parameter_dtype=str(dtype), device=str(device))
    setattr(model, "foldforge_load_report", report)  # noqa: B010 - runtime metadata
    return model
