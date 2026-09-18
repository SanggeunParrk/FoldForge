"""Sweep diffusion step counts for one target and one diffusion seed.

Loads the checkpoint once, prepares the input once, and runs a complete forward
per requested step count with the released schedule, solver and precision policy
unchanged. Each step count writes its own prediction directory under ``--out`` so
``evaluate_diffusion_sweep.py`` can score every sample against the experimental
structure, against the same-seed 200-step run, and for local geometry.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import torch
import yaml

from foldforge.data.ccd import CCDDatabase
from foldforge.data.inputs.build import limit_msa, load, write_adapter_input
from foldforge.models import entry
from foldforge.models.config import Config
from foldforge.models.io import runtime as runtime_module
from foldforge.models.io.output import write_output
from foldforge.models.io.paths import run_directory
from foldforge.models.io.request import Request
from foldforge.models.msa_policy import PREPARED_ROWS
from foldforge.utils.seed import seed_all, seed_context

STEP_ATTRIBUTE = {"af3": "diffusion_steps"}


def resolve_spec(target: str, out: Path, samples: int) -> Path:
    """Resolve the qualification spec the way the benchmark driver does."""
    source = Path("validation/inputs/qualification") / target / "input.yaml"
    spec = yaml.safe_load(source.read_text())
    for key in ("fasta", "a3m", "msa_db"):
        spec[key] = {
            k: str((source.parent / Path(v)).resolve())
            for k, v in spec.get(key, {}).items()
        }
    for key in ("ccd_db", "template_db", "cif_db"):
        if spec.get(key):
            spec[key] = str((source.parent / Path(spec[key])).resolve())
    sibling = source.parent / "template.lmdb"
    if spec.get("template_db") and not Path(spec["template_db"]).exists():
        if not sibling.exists():
            message = f"No template LMDB for {target}"
            raise FileNotFoundError(message)
        spec["template_db"] = str(sibling.resolve())
    spec.update(n_diffusion_samples=samples, diffusion_batch_size=samples)
    path = out / "input.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="af3", choices=sorted(STEP_ATTRIBUTE))
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[200, 100, 68, 50, 30, 20]
    )
    parser.add_argument("--trunk-seed", type=int, default=0)
    parser.add_argument("--diffusion-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--precision", default="af3_default")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        message = "The sweep needs an allocated GPU"
        raise RuntimeError(message)

    out = run_directory(args.out, model=args.model)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = resolve_spec(args.target, out, args.samples)
    config = Config.model_validate(
        {
            "backend": args.backend,
            "precision": args.precision,
            "trunk_seed": args.trunk_seed,
            "diffusion_seed": args.diffusion_seed,
            "trunk": {"recycles": 10, "msa_depth": PREPARED_ROWS},
            "diffusion": {"steps": max(args.steps)},
            "execution": {
                "compile": args.compile,
                "cuda_graph": False,
                "bucketing": True,
                "scope": "denoiser",
            },
        }
    )
    (out / "model.config.json").write_text(config.model_dump_json(indent=2) + "\n")
    target = limit_msa(load(spec_path), PREPARED_ROWS, out / "prepared-msa")
    database = CCDDatabase(target.spec.ccd_db)
    adapter_input = out / "input.adapter.json"
    write_adapter_input(target, args.model, adapter_input, config.trunk_seed)
    request = Request(
        model=args.model,
        ccd_db=database.root,
        out=out,
        resolved_input=target,
        input=adapter_input,
        target=target.spec.name or adapter_input.stem,
        backend=config.backend,
        precision=config.precision,
        trunk_seed=config.trunk_seed,
        diffusion_seed=config.diffusion_seed,
        samples=target.spec.n_diffusion_samples,
        recycles=config.trunk.recycles or 10,
        steps=config.diffusion.steps or 200,
        msa_depth=config.trunk.msa_depth or PREPARED_ROWS,
        templates=bool(target.spec.template),
        execution=config.execution,
    )

    captured: dict[str, torch.nn.Module] = {}
    original_bind = runtime_module.Runtime.bind

    def bind(self: runtime_module.Runtime, model: torch.nn.Module) -> object:
        execution = original_bind(self, model)
        captured["model"] = model
        return execution

    runtime_module.Runtime.bind = bind  # type: ignore[method-assign]
    layout = entry(args.model).layout
    adapter = importlib.import_module(f"foldforge.models.io.{layout}")
    runtime = runtime_module.Runtime(request)
    records = []
    with seed_context(config.trunk_seed), database.activate():
        for case in adapter.prepare(request, database, runtime):
            model = captured["model"]
            attribute = STEP_ATTRIBUTE[args.model]
            for steps in args.steps:
                setattr(model, attribute, steps)
                seed_all(config.trunk_seed)
                context = torch.inference_mode if case.inference_mode else torch.no_grad
                torch.cuda.synchronize()
                start = time.perf_counter()
                with context():
                    output = case.forward()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                decoded = case.decode(output, {"model_seconds": seconds})
                decoded.report = {
                    **decoded.report,
                    "steps": steps,
                    "trunk_seed": config.trunk_seed,
                    "diffusion_seed": config.diffusion_seed,
                    "model_seconds": seconds,
                    "solver": "euler",
                    "schedule": "released",
                    **runtime.execution.report(),
                }
                report = write_output(
                    decoded, out / f"steps-{steps}", expected_samples=request.samples
                )
                records.append(
                    {
                        "target": args.target,
                        "steps": steps,
                        "diffusion_seed": config.diffusion_seed,
                        "model_seconds": seconds,
                        "prediction_cifs": report["prediction_cifs"],
                        "denoiser_calls": report.get("wrapped_calls"),
                    }
                )
                print(  # noqa: T201 - CLI output contract
                    f"{args.target} steps={steps} seed={config.diffusion_seed} "
                    f"forward={seconds:.1f}s",
                    flush=True,
                )
                del output, decoded
                torch.cuda.empty_cache()
    (out / "sweep.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
