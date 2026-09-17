# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Compatibility flags translated once into the same typed inference request."""

from __future__ import annotations

import argparse
from pathlib import Path

from foldforge.data.ccd import default_path
from foldforge.models.config import ExecutionConfig, OutputConfig
from foldforge.models.io.request import Request


def parse(model: str, argv: list[str] | None = None) -> Request:  # noqa: PLR0915 - translate compatibility flags at one boundary
    """Parse ."""
    parser = argparse.ArgumentParser(prog=f"foldforge fold {model}")
    sequence_layout = model == "esmfold2"
    parser.add_argument("--ccd-db", type=Path, default=default_path())
    parser.add_argument("--checkpoint", type=Path, required=not sequence_layout)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run name or path inside runs/ (default: unique model run)",
    )
    from foldforge.models.config.runtime import IMAGE_KINDS

    parser.add_argument(
        "--save-images",
        nargs="+",
        choices=(*IMAGE_KINDS, "all"),
        help="Save selected diagnostic PNGs (or all); disabled by default",
    )
    parser.add_argument("--recycles", type=int, default=None if sequence_layout else 10)
    parser.add_argument("--steps", type=int, default=None if sequence_layout else 200)
    parser.add_argument("--samples", type=int, default=1 if sequence_layout else 5)
    parser.add_argument("--seed", type=int, help="Legacy shorthand for both seeds")
    parser.add_argument(
        "--trunk-seed", type=int, help="Input/conformer/MSA/trunk randomness"
    )
    parser.add_argument(
        "--diffusion-seed", type=int, help="Diffusion noise and rigid augmentation"
    )
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--bucketing", action="store_true")
    parser.add_argument(
        "--execution-scope", choices=("denoiser", "model"), default="denoiser"
    )
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    if sequence_layout:
        parser.add_argument("--input-spec", type=Path)
        parser.add_argument("--target", default="1ubq")
        parser.add_argument(
            "--data-root", type=Path, default=Path("validation/inputs/data")
        )
        parser.add_argument("--lm-cache", type=Path)
        parser.add_argument("--lm-checkpoint", type=Path)
        parser.add_argument(
            "--lm-source", choices=("compute", "cache"), default="compute"
        )
        parser.add_argument("--msa-depth", type=int, default=512)
        parser.add_argument(
            "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
        )
        parser.add_argument(
            "--implementation",
            choices=("miniworld_engine", "pytorch", "cuequivariance"),
            default="miniworld_engine",
        )
        parser.add_argument("--compare", type=Path)
        parser.add_argument("--experimental", type=Path)
        parser.add_argument("--chain-map")
    else:
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument(
            "--backend",
            choices=("miniworld", "pytorch", "cuequivariance"),
            default="miniworld",
        )
        parser.add_argument(
            "--precision",
            choices=("bf16", "fp32", "af3_default", "model_default")
            if model == "af3"
            else ("bf16", "fp32"),
            default="bf16",
        )
        if model in {"protenix", "opendde"}:
            parser.add_argument("--templates", action="store_true")
            parser.add_argument("--no-msa", action="store_true")
        if model == "protenix":
            parser.add_argument(
                "--variant",
                default="protenix_base_default_v1.0.0",
                choices=(
                    "protenix_base_default_v1.0.0",
                    "protenix_base_20250630_v1.0.0",
                    "protenix-v2",
                ),
            )
    options = vars(parser.parse_args(argv))
    output = OutputConfig(images=tuple(options.pop("save_images") or ()))
    legacy_seed = options.pop("seed")
    if legacy_seed is not None and any(
        options[name] is not None for name in ("trunk_seed", "diffusion_seed")
    ):
        parser.error("Use either --seed or --trunk-seed/--diffusion-seed")
    for name in ("trunk_seed", "diffusion_seed"):
        if options[name] is None:
            options[name] = 0 if legacy_seed is None else legacy_seed
    execution = ExecutionConfig(
        **{
            key: options.pop(key)
            for key in (
                "compile",
                "cuda_graph",
                "bucketing",
                "max_graphs",
                "benchmark_repeats",
            )
        },
        scope=options.pop("execution_scope"),
    )
    if sequence_layout:
        options["precision"] = "bf16" if options.pop("dtype") == "bfloat16" else "fp32"
        backend = options.pop("implementation")
        options["backend"] = "miniworld" if backend == "miniworld_engine" else backend
    try:
        return Request(model=model, execution=execution, output=output, **options)
    except ValueError as error:
        parser.error(str(error))


def run(model: str, argv: list[str] | None = None) -> int:
    """Run the configured operation."""
    request = parse(model, argv)
    from foldforge.models.io.runtime import run as predict

    return predict(request)
