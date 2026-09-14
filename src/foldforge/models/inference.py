"""One MiniWorld-format data/config entry point for every released predictor."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import yaml

from foldforge.data.ccd import CCDDatabase
from foldforge.data.inference.build import limit_msa, load, write_adapter_input

from .config import Config


def run(model: str, argv: list[str]):
    parser = argparse.ArgumentParser(prog=f"foldforge fold {model}")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lm-cache", type=Path)
    args = parser.parse_args(argv)
    config = (
        Config.model_validate(yaml.safe_load(args.config.read_text()) or {})
        if args.config
        else Config()
    )
    target = load(args.spec)
    if model == "esmfold2" and target.spec.template:
        raise ValueError("ESMFold2 checkpoint has no template conditioning path")
    if config.variant is not None and model != "protenix":
        raise ValueError("variant applies only to Protenix")
    if args.lm_cache and model != "esmfold2":
        parser.error("--lm-cache applies only to ESMFold2")
    if target.spec.save_trajectory:
        raise ValueError(
            "Released adapters write final structures; set save_trajectory: false"
        )
    if target.spec.n_trunk_samples != 1:
        raise ValueError("Released adapters currently require n_trunk_samples=1")
    if target.spec.diffusion_batch_size != target.spec.n_diffusion_samples:
        raise ValueError(
            "Set diffusion_batch_size=n_diffusion_samples; these adapters do not chunk sampling"
        )
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
    common = ["--ccd-db", str(db.root), "--out", str(args.out)]
    if config.execution.compile:
        common += ["--compile"]
    if config.execution.cuda_graph:
        common += ["--cuda-graph"]
    common += [
        "--max-graphs",
        str(config.execution.max_graphs),
        "--execution-scope",
        config.execution.scope,
        "--benchmark-repeats",
        str(config.execution.benchmark_repeats),
    ]
    if args.checkpoint:
        common += ["--checkpoint", str(args.checkpoint)]
    elif model != "esmfold2":
        from foldforge.checkpoints import resolve

        common += [
            "--checkpoint",
            str(
                resolve(
                    model, (config.variant or "protenix_base_default_v1.0.0") + ".pt"
                )
                if model == "protenix"
                else resolve(model, "af3.bin.zst" if model == "af3" else "opendde.pt")
            ),
        ]
    if model == "esmfold2":
        common += [
            "--input-spec",
            str(args.spec.resolve()),
            "--target",
            target.spec.name,
            "--msa-depth",
            str(config.trunk.msa_depth or 512),
            "--seed",
            str(config.seed),
            "--dtype",
            "bfloat16" if config.precision == "bf16" else "float32",
            "--implementation",
            "miniworld_engine" if config.backend == "miniworld" else "pytorch",
            "--samples",
            str(target.spec.n_diffusion_samples),
        ]
        if args.lm_cache:
            common += [
                "--lm-cache",
                str(args.lm_cache.resolve()),
                "--lm-source",
                "cache",
            ]
        if config.trunk.recycles:
            common += ["--recycles", str(config.trunk.recycles)]
        if config.diffusion.steps:
            common += ["--steps", str(config.diffusion.steps)]
    else:
        if args.lm_cache:
            parser.error("--lm-cache applies only to ESMFold2")
        path = args.out / "input.adapter.json"
        write_adapter_input(target, model, path, config.seed)
        common += [
            "--input",
            str(path),
            "--backend",
            config.backend,
            "--precision",
            config.precision,
            "--recycles",
            str(config.trunk.recycles or 10),
            "--steps",
            str(config.diffusion.steps or 200),
            "--samples",
            str(target.spec.n_diffusion_samples),
        ]
        if model != "af3":
            common += ["--seed", str(config.seed)]
            if target.spec.template:
                common += ["--templates"]
        if config.variant:
            if model != "protenix":
                raise ValueError("variant applies only to Protenix")
            common += ["--variant", config.variant]
    return importlib.import_module(f"foldforge.models.{model}.inference").main(common)
