# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""One MiniWorld-format data/config entry point for every released predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from foldforge.data.ccd import CCDDatabase
from foldforge.data.inputs.build import Input, limit_msa, load, write_adapter_input
from foldforge.models.config import Config
from foldforge.models.io.paths import run_directory


def _validate_input(
    model: str,
    target: Input,
    config: Config,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate checkpoint capabilities before creating output artifacts."""
    if model == "esmfold2" and target.spec.template:
        msg = "ESMFold2 checkpoint has no template conditioning path"
        raise ValueError(msg)
    if config.variant is not None and model != "protenix":
        msg = "variant applies only to Protenix"
        raise ValueError(msg)
    if args.lm_cache and model != "esmfold2":
        parser.error("--lm-cache applies only to ESMFold2")
    if target.spec.save_trajectory:
        msg = "Released adapters write final structures; set save_trajectory: false"
        raise ValueError(msg)
    if target.spec.n_trunk_samples != 1:
        msg = "Released adapters currently require n_trunk_samples=1"
        raise ValueError(msg)
    if target.spec.diffusion_batch_size != target.spec.n_diffusion_samples:
        msg = (
            "Set diffusion_batch_size=n_diffusion_samples; these adapters do "
            "not chunk sampling"
        )
        raise ValueError(msg)


def run(model: str, argv: list[str]) -> int:
    """Run the configured operation."""
    parser = argparse.ArgumentParser(prog=f"foldforge fold {model}")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--trunk-seed", type=int, help="Input/conformer/MSA/trunk randomness"
    )
    parser.add_argument(
        "--diffusion-seed", type=int, help="Diffusion noise and rigid augmentation"
    )
    from foldforge.models.config.runtime import IMAGE_KINDS

    parser.add_argument(
        "--save-images",
        nargs="+",
        choices=(*IMAGE_KINDS, "all"),
        help="Save selected diagnostic PNGs (or all); disabled by default",
    )
    parser.add_argument("--out", type=Path, help="Run name or path inside runs/")
    parser.add_argument("--lm-cache", type=Path)
    parser.add_argument("--engine-cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        args.out = run_directory(args.out, model=model)
    except ValueError as error:
        parser.error(str(error))
    if args.engine_cache_dir is not None:
        if not args.engine_cache_dir.is_dir():
            parser.error("--engine-cache-dir must be an existing cache directory")
        from miniworld_engine.autotune import cache

        cache._CACHE_ROOT = args.engine_cache_dir.resolve()  # noqa: SLF001 - shared runtime integration hook
        cache._load_cache.clear()  # noqa: SLF001 - shared runtime integration hook
    config = (
        Config.model_validate(yaml.safe_load(args.config.read_text()) or {})
        if args.config
        else Config()
    )
    overrides = {
        name: getattr(args, name)
        for name in ("trunk_seed", "diffusion_seed")
        if getattr(args, name) is not None
    }
    if args.save_images is not None:
        overrides["output"] = {"images": args.save_images}
    config = Config.model_validate({**config.model_dump(), **overrides})
    target = load(args.spec)
    _validate_input(model, target, config, args, parser)
    db = CCDDatabase(target.spec.ccd_db)
    for chain in target.chains:
        for ccd in set(chain.ccds):
            db.lookup[ccd]  # Fail before loading weights if input chemistry is missing.
    args.out.mkdir(parents=True, exist_ok=True)
    if config.trunk.msa_depth is not None:
        target = limit_msa(target, config.trunk.msa_depth, args.out / "prepared-msa")
    (args.out / "msa-species-aliases.json").write_text(
        json.dumps(target.msa_species, indent=2) + "\n"
    )
    (args.out / "input.resources.json").write_text(
        json.dumps(target.resources(), indent=2) + "\n"
    )
    (args.out / "input.spec.json").write_text(
        target.spec.model_dump_json(indent=2) + "\n"
    )
    (args.out / "model.config.json").write_text(config.model_dump_json(indent=2) + "\n")
    (args.out / "chain-map.json").write_text(
        json.dumps(
            [{"index": c.index, "id": c.id, "letter": c.letter} for c in target.chains],
            indent=2,
        )
        + "\n"
    )
    from foldforge.models.io.request import Request
    from foldforge.models.io.runtime import run as predict

    checkpoint = args.checkpoint
    if checkpoint is None and model != "esmfold2":
        from foldforge.models.checkpoints import resolve

        checkpoint = (
            resolve(model, (config.variant or "protenix_base_default_v1.0.0") + ".pt")
            if model == "protenix"
            else resolve(model, "af3.bin.zst" if model == "af3" else "opendde.pt")
        )
    path = None
    if model != "esmfold2":
        path = args.out / "input.adapter.json"
        write_adapter_input(target, model, path, config.trunk_seed)
    return predict(
        Request(
            model=model,
            ccd_db=db.root,
            out=args.out,
            checkpoint=checkpoint,
            resolved_input=target,
            input=path,
            target=target.spec.name or path.stem
            if path is not None
            else target.spec.name or "prediction",
            backend=config.backend,
            precision=config.precision,
            trunk_seed=config.trunk_seed,
            diffusion_seed=config.diffusion_seed,
            samples=target.spec.n_diffusion_samples,
            recycles=config.trunk.recycles
            if model == "esmfold2"
            else (config.trunk.recycles or 10),
            steps=config.diffusion.steps
            if model == "esmfold2"
            else (config.diffusion.steps or 200),
            msa_depth=config.trunk.msa_depth or 512,
            templates=bool(target.spec.template),
            variant=config.variant or "protenix_base_default_v1.0.0",
            execution=config.execution,
            output=config.output,
            lm_cache=args.lm_cache.resolve() if args.lm_cache else None,
            lm_source="cache" if args.lm_cache else "compute",
        )
    )
