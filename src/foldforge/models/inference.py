"""CLI-to-library adaptation; prediction orchestration belongs to foldforge.api."""
# ruff: noqa: PLC0415 - selected CLI execution imports optional backends lazily

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from foldforge.api import (
    _validate_target as _validate_input,  # noqa: F401 - legacy helper export
)
from foldforge.api import predict, resolve_config
from foldforge.data.inputs.build import load
from foldforge.models.io.paths import run_directory

if TYPE_CHECKING:
    from foldforge.models.config import Config
    from foldforge.models.io.options import FoldArguments


def _explicit(arguments: FoldArguments, names: tuple[str, ...]) -> dict:
    """Collect explicit overrides without replacing omitted or false values."""
    return {
        name: getattr(arguments, name)
        for name in names
        if getattr(arguments, name) is not None
    }


def resolve_cli_config(arguments: FoldArguments) -> Config:
    """Merge explicit options; omitted booleans never overwrite YAML."""
    data = resolve_config(arguments.config).model_dump()
    data.update(
        _explicit(
            arguments, ("backend", "precision", "mode", "trunk_seed", "diffusion_seed")
        )
    )
    if arguments.seed is not None:
        data.update(trunk_seed=arguments.seed, diffusion_seed=arguments.seed)
    data["execution"].update(
        _explicit(
            arguments,
            (
                "compile",
                "cuda_graph",
                "bucketing",
                "tf32",
                "max_graphs",
                "benchmark_repeats",
            ),
        )
    )
    if arguments.execution_scope is not None:
        data["execution"]["scope"] = arguments.execution_scope
    data["trunk"].update(_explicit(arguments, ("recycles", "msa_depth")))
    if arguments.steps is not None:
        data["diffusion"]["steps"] = arguments.steps
    if arguments.save_images:
        data["output"]["images"] = arguments.save_images
    return resolve_config(data)


def execute(arguments: FoldArguments) -> int:
    """Call the public API used by external Python applications."""
    output = run_directory(arguments.out, model=arguments.model)
    config = resolve_cli_config(arguments)
    inputs = load(arguments.spec) if arguments.spec is not None else arguments.input
    if inputs is None:
        message = "Choose exactly one of --spec or --input"
        raise ValueError(message)
    samples = arguments.samples
    if arguments.input is not None and samples is None:
        samples = 5
    result = predict(
        inputs,
        config,
        model=arguments.model,
        checkpoint=arguments.checkpoint,
        out=output,
        samples=samples,
        ccd_db=arguments.ccd_db,
        engine_cache_dir=arguments.engine_cache_dir,
        verbose=True,
        device=arguments.device,
    )
    print(json.dumps(result.report, indent=2, allow_nan=False), flush=True)  # noqa: T201 - CLI contract
    return 0


def run(model: str, argv: list[str]) -> int:
    """Compatibility entry point through the shared parser."""
    import rich_click as click
    import yaml

    from foldforge.models.io.options import parse

    arguments = parse(model, argv)
    try:
        return execute(arguments)
    except (OSError, ValueError, yaml.YAMLError) as error:
        failure = click.ClickException(str(error))
        failure.show()
        raise SystemExit(failure.exit_code) from error
