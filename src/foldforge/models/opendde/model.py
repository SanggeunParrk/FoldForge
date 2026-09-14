# GPU dependencies are imported at the model boundary so registry/help stay cheap.
# ruff: noqa: PLC0415
"""OpenDDE configuration, released checkpoint and shared engine loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from foldforge.models.af_conditioning import install_conditioning
from foldforge.models.af_pairformer import install_pairformers
from foldforge.modules.af_family import configure_model

from .ported.config.inference import build_inference_config
from .ported.model.opendde import OpenDDE
from .ported.model.triangular.layers import skip_random_init


def configuration(arguments: str = "") -> Any:
    """Build the released model defaults with explicit caller overrides."""
    return build_inference_config(arg_str=arguments, model_name="opendde_v1")


def load(
    checkpoint: str | Path | None = None,
    *,
    configs: Any = None,
    backend: str = "miniworld",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> torch.nn.Module:
    """Strictly load released weights and configure native parameter precision."""
    if checkpoint is None:
        from foldforge.checkpoints import resolve

        checkpoint = resolve("opendde", "opendde.pt")
    configs = configuration() if configs is None else configs
    configs.triangle_multiplicative = "torch"
    configs.triangle_attention = "torch"
    with skip_random_init():
        model = OpenDDE(configs)
    checkpoint = Path(checkpoint)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = blob.get("model", blob.get("state_dict", blob))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.foldforge_load_report = {
        "checkpoint": str(checkpoint),
        "variant": "opendde_v1",
        "state_entries": len(state),
        "strict": True,
    }
    return install_conditioning(
        install_pairformers(configure_model(model, backend, dtype, device))
    )
