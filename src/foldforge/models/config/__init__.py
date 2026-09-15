# ruff: noqa: PLC0415 - load only the requested configuration schema
"""Checkpoint configuration schemas and released defaults."""

from copy import deepcopy
from typing import Any

from foldforge.models.config.runtime import (
    Config,
    DiffusionConfig,
    ExecutionConfig,
    TrunkConfig,
)

__all__ = [
    "Config",
    "DiffusionConfig",
    "ExecutionConfig",
    "TrunkConfig",
    "configuration",
]

VARIANTS = (
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
    "protenix-v2",
)


def configuration(name: str, variant: str | None = None, arguments: str = "") -> Any:
    """Build the released model defaults with explicit caller overrides."""
    if name == "opendde":
        from foldforge.models.config.opendde.config.inference import (
            build_inference_config,
        )

        config = build_inference_config(arg_str=arguments, model_name="opendde_v1")
        config.triangle_multiplicative = "torch"
        config.triangle_attention = "torch"
        return config
    if name != "protenix":
        message = f"{name} has no flat-atom configuration schema"
        raise ValueError(message)
    variant = variant or VARIANTS[0]
    from foldforge.models.config.protenix.config.config import parse_configs
    from foldforge.models.config.protenix.configs.configs_base import (
        configs as configs_base,
    )
    from foldforge.models.config.protenix.configs.configs_data import data_configs
    from foldforge.models.config.protenix.configs.configs_inference import (
        inference_configs,
    )
    from foldforge.models.config.protenix.configs.configs_model_type import (
        model_configs,
    )

    if variant not in VARIANTS:
        msg = f"Unsupported Protenix variant {variant!r}; choose {VARIANTS}"
        raise ValueError(msg)
    tree = deepcopy({**configs_base, "data": data_configs, **inference_configs})

    def update(dst: Any, src: Any) -> None:
        """Update ."""
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                update(dst[k], v)
            else:
                dst[k] = deepcopy(v)

    update(tree, model_configs[variant])
    tree["model_name"] = variant
    tree["triangle_multiplicative"] = "torch"
    tree["triangle_attention"] = "torch"
    return parse_configs(configs=tree, arg_str=arguments, fill_required_with_null=True)
