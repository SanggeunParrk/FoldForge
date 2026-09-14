# GPU dependencies are imported at the model boundary so registry/help stay cheap.
# ruff: noqa: PLC0415
"""Released Protenix v1/v2 configuration and strict checkpoint loader."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from foldforge.models.af_conditioning import install_conditioning
from foldforge.models.af_pairformer import install_pairformers
from foldforge.modules.af_family import configure_model

from .ported.config.config import parse_configs
from .ported.configs.configs_base import configs as configs_base
from .ported.configs.configs_data import data_configs
from .ported.configs.configs_inference import inference_configs
from .ported.configs.configs_model_type import model_configs
from .ported.model.protenix import Protenix
from .ported.model.triangular.layers import skip_random_init

VARIANTS = (
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix-v2",
)


def configuration(
    variant: str = "protenix_base_default_v1.0.0", arguments: str = ""
) -> Any:
    """Build the released model defaults with explicit caller overrides."""
    if variant not in VARIANTS:
        msg = f"Unsupported Protenix variant {variant!r}; choose {VARIANTS}"
        raise ValueError(msg)
    tree = deepcopy({**configs_base, "data": data_configs, **inference_configs})

    def update(dst: Any, src: Any) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                update(dst[k], v)
            else:
                dst[k] = deepcopy(v)

    update(tree, model_configs[variant])
    tree["model_name"] = variant
    tree["triangle_multiplicative"] = "torch"
    tree["triangle_attention"] = "torch"
    return parse_configs(tree, arg_str=arguments, fill_required_with_null=True)


def load(
    checkpoint: str | Path | None = None,
    *,
    variant: str = "protenix_base_default_v1.0.0",
    configs: Any = None,
    backend: str = "miniworld",
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> torch.nn.Module:
    """Strictly load released weights and configure native parameter precision."""
    if checkpoint is None:
        from foldforge.checkpoints import resolve

        checkpoint = resolve("protenix", variant + ".pt")
    configs = configuration(variant) if configs is None else configs
    with skip_random_init():
        model = Protenix(configs)
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)[
        "model"
    ]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.foldforge_load_report = {
        "checkpoint": str(checkpoint),
        "variant": configs.model_name,
        "state_entries": len(state),
        "strict": True,
    }
    return install_conditioning(
        install_pairformers(configure_model(model, backend, dtype, device))
    )
