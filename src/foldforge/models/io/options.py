"""One lightweight fold parser; input preparation and GPU imports live elsewhere."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rich_click as click

from foldforge.models import known_models
from foldforge.models.config.runtime import IMAGE_KINDS
from foldforge.modules.engine.backends import BACKENDS


@dataclass(frozen=True)
class FoldArguments:
    """Explicit CLI overrides; None means preserve the YAML/default value."""

    model: str
    spec: Path | None = None
    input: Path | None = None
    config: Path | None = None
    checkpoint: Path | None = None
    out: Path | None = None
    ccd_db: Path | None = None
    engine_cache_dir: Path | None = None
    device: str | None = None
    backend: str | None = None
    precision: str | None = None
    mode: str | None = None
    seed: int | None = None
    trunk_seed: int | None = None
    diffusion_seed: int | None = None
    recycles: int | None = None
    steps: int | None = None
    samples: int | None = None
    msa_depth: int | None = None
    compile: bool | None = None
    cuda_graph: bool | None = None
    bucketing: bool | None = None
    tf32: bool | None = None
    execution_scope: str | None = None
    max_graphs: int | None = None
    benchmark_repeats: int | None = None
    save_images: tuple[str, ...] = ()


def normalize_image_arguments(arguments: list[str]) -> list[str]:
    """Accept the old space-separated image list as repeated Click options."""
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            result.extend(arguments[index:])
            break
        if argument == "--save-images":
            end = index + 1
            while end < len(arguments) and not arguments[end].startswith("-"):
                end += 1
            values = arguments[index + 1 : end]
            if values:
                result.extend(f"--save-images={value}" for value in values)
            else:
                result.append(argument)
            index = end
        else:
            result.append(argument)
            index += 1
    return result


class FoldCommand(click.RichCommand):
    """Local compatibility translation before the single fold parser."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Normalize legacy image flags before Click validation."""
        return super().parse_args(ctx, normalize_image_arguments(args))


@click.command(
    "fold", cls=FoldCommand, context_settings={"help_option_names": ["-h", "--help"]}
)
@click.rich_config(
    help_config={
        "option_groups": {
            "*": [
                {
                    "name": "Input and output",
                    "options": [
                        "model",
                        "--spec",
                        "--input",
                        "--config",
                        "--checkpoint",
                        "--ccd-db",
                        "--out",
                    ],
                },
                {
                    "name": "Model and sampling",
                    "options": [
                        "--backend",
                        "--precision",
                        "--mode",
                        "--recycles",
                        "--steps",
                        "--samples",
                        "--msa-depth",
                        "--trunk-seed",
                        "--diffusion-seed",
                        "--seed",
                    ],
                },
                {
                    "name": "Execution",
                    "options": [
                        "--compile",
                        "--cuda-graph",
                        "--bucketing",
                        "--tf32",
                        "--execution-scope",
                        "--max-graphs",
                        "--benchmark-repeats",
                        "--engine-cache-dir",
                        "--device",
                    ],
                },
                {"name": "Diagnostics", "options": ["--save-images", "--help"]},
            ]
        }
    }
)
@click.argument("model", type=click.Choice(known_models()), metavar="MODEL")
@click.option(
    "--spec",
    type=click.Path(path_type=Path, dir_okay=False),
    help="MiniWorld YAML input specification.",
)
@click.option(
    "--input",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Existing adapter JSON input (compatibility).",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, dir_okay=False),
    help="YAML configuration; explicit CLI options override it.",
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    help="Checkpoint override; otherwise use the model registry.",
)
@click.option(
    "--ccd-db",
    type=click.Path(path_type=Path),
    help="CCD database for adapter input; YAML inputs declare their own.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Run name or path within the configured runs directory.",
)
@click.option(
    "--backend",
    type=click.Choice(BACKENDS),
    help="Kernel backend (default: miniworld).",
)
@click.option(
    "--precision",
    type=click.Choice(("bf16", "fp32", "af3_default", "model_default")),
    help="Parameter precision (default: bf16).",
)
@click.option(
    "--mode",
    type=click.Choice(("fast", "exact")),
    help="Model conventions (default: fast).",
)
@click.option(
    "--recycles",
    type=click.IntRange(min=1),
    help="Total trunk passes, including the first (default: 10).",
)
@click.option(
    "--steps", type=click.IntRange(min=1), help="Diffusion steps (default: 200)."
)
@click.option(
    "--samples",
    type=click.IntRange(min=1),
    help="Override samples and batch size in YAML; adapter default: 5.",
)
@click.option(
    "--msa-depth", type=click.IntRange(min=1), help="Maximum prepared MSA depth."
)
@click.option(
    "--trunk-seed",
    type=click.IntRange(0, 2**32 - 1),
    help="Input, MSA, conformer and trunk randomness.",
)
@click.option(
    "--diffusion-seed",
    type=click.IntRange(0, 2**32 - 1),
    help="Diffusion noise and rigid augmentation.",
)
@click.option(
    "--seed",
    type=click.IntRange(0, 2**32 - 1),
    help="Compatibility shorthand for both seeds.",
)
@click.option(
    "--compile/--no-compile",
    default=None,
    help="Compile the configured execution boundary.",
)
@click.option(
    "--cuda-graph/--no-cuda-graph", default=None, help="Capture and replay CUDA graphs."
)
@click.option(
    "--bucketing/--no-bucketing",
    default=None,
    help="Pad inputs to supported buckets (default: enabled).",
)
@click.option(
    "--tf32/--no-tf32",
    default=None,
    help="Allow TF32 tensor-core kernels (default: disabled).",
)
@click.option(
    "--execution-scope",
    type=click.Choice(("denoiser", "trunk", "model")),
    help="Compile/graph scope (default: denoiser).",
)
@click.option(
    "--max-graphs",
    type=click.IntRange(1, 32),
    help="Maximum cached graphs (default: 4).",
)
@click.option(
    "--benchmark-repeats",
    type=click.IntRange(0, 20),
    help="Warm timing repetitions (default: 0).",
)
@click.option("--device", help="CUDA device fixed for this model, e.g. cuda:1.")
@click.option(
    "--engine-cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Use this engine autotune cache for the run.",
)
@click.option(
    "--save-images",
    type=click.Choice((*IMAGE_KINDS, "all")),
    multiple=True,
    help="Save diagnostic images; repeat for multiple kinds.",
)
def fold_command(model: str, **options: Any) -> FoldArguments:
    """Predict a structure with any supported model.

    For example:
      foldforge fold chai1 --spec input.yaml --config config.yaml --out run-1
      foldforge fold af3 --spec input.yaml --recycles 10 --compile --cuda-graph

    Choose exactly one of --spec or --input. Omitted flags preserve configuration.
    """
    arguments = FoldArguments(model=model, **options)
    if (arguments.spec is None) == (arguments.input is None):
        message = "Choose exactly one of --spec or --input."
        raise click.UsageError(message)
    if arguments.input is not None and arguments.checkpoint is None:
        message = "--input requires --checkpoint."
        raise click.UsageError(message)
    if arguments.seed is not None and (
        arguments.trunk_seed is not None or arguments.diffusion_seed is not None
    ):
        message = "Use either --seed or --trunk-seed/--diffusion-seed."
        raise click.UsageError(message)
    return arguments


def parse(model: str, argv: list[str]) -> FoldArguments:
    """Use the same parser for old Python entry points and the public CLI."""
    try:
        result = fold_command.main(
            args=[model, *argv], prog_name="foldforge fold", standalone_mode=False
        )
        if not isinstance(result, FoldArguments):
            raise SystemExit(result if isinstance(result, int) else 0)
    except click.ClickException as error:
        error.show()
        raise SystemExit(error.exit_code) from error
    else:
        return result
