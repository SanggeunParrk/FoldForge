"""Measure complete inference subprocesses on a Slurm-allocated GPU.

E2E wall time includes imports, inputs, weights, compile/capture, forward and
output serialization. It is not a warmed model-only throughput number.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from foldforge.models.io.paths import run_directory


def main() -> int:  # noqa: PLR0912 - complete benchmark scenario matrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["af3", "esmfold2", "protenix", "opendde"]
    )
    parser.add_argument("--root", type=Path, default=Path("benchmark/e2e"))
    parser.add_argument("--backends", nargs="+", default=["pytorch", "miniworld"])
    parser.add_argument("--targets", nargs="+", default=["3ptb", "4yx2"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--no-bucketing", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--recycles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--msa-depth", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        msg = "Run inside an allocated Slurm compute job"
        raise RuntimeError(msg)
    args.root = run_directory(args.root, model=args.model)
    args.root.mkdir(parents=True, exist_ok=True)
    result_path = args.root / f"e2e-{args.model}.json"
    rows = (
        json.loads(result_path.read_text())
        if args.resume and result_path.exists()
        else []
    )
    for target in args.targets:
        source = (
            Path("validation/inputs/qualification")
            / target
            / ("esm.yaml" if args.model == "esmfold2" else "input.yaml")
        )
        spec = yaml.safe_load(source.read_text())
        for key in ("fasta", "a3m", "msa_db"):
            spec[key] = {
                k: str((source.parent / Path(v)).resolve())
                for k, v in spec.get(key, {}).items()
            }
        for key in ("ccd_db", "template_db", "cif_db"):
            if spec.get(key):
                spec[key] = str((source.parent / Path(spec[key])).resolve())
        spec.update(n_diffusion_samples=args.samples, diffusion_batch_size=args.samples)
        path = args.root / "inputs" / args.model / f"{target}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(spec))
        for backend in args.backends:
            label = f"{args.model}-{target}-{backend}"
            previous = next(
                (
                    row
                    for row in rows
                    if row["target"] == target and row["backend"] == backend
                ),
                None,
            )
            output = args.root / "e2e" / label
            output.mkdir(parents=True, exist_ok=True)
            config = {
                "backend": backend,
                "precision": "bf16",
                "seed": 0,
                "trunk": {"recycles": args.recycles, "msa_depth": args.msa_depth},
                "diffusion": {"steps": args.steps},
                "execution": {
                    "compile": True,
                    "cuda_graph": True,
                    "bucketing": not args.no_bucketing,
                    "scope": "denoiser",
                    "max_graphs": 4,
                    "benchmark_repeats": args.benchmark_repeats,
                },
            }
            if previous is not None and previous["status"] == 0:
                if previous["config"] != config:
                    msg = f"Resume config changed for {label}; use another output root"
                    raise ValueError(msg)
                print("SKIP completed", label, flush=True)  # noqa: T201 - CLI output contract
                continue
            config_path = output / "benchmark-config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            command = [
                sys.executable,
                "-c",
                (
                    "import sys; from foldforge.models.inference import run; raise "
                    "SystemExit(run(sys.argv[1],sys.argv[2:]))"
                ),
                args.model,
                "--spec",
                str(path),
                "--config",
                str(config_path),
                "--out",
                str(output),
            ]
            if args.model == "esmfold2":
                command += [
                    "--lm-cache",
                    str(
                        Path("validation/inputs/data")
                        / target
                        / "cache/esmc_hidden_states.pt"
                    ),
                ]
            print("START", label, flush=True)  # noqa: T201 - CLI output contract
            start = time.perf_counter()
            error = None
            if previous is not None:
                archive = output / f"attempt-{time.time_ns()}"
                archive.mkdir()
                (archive / "result.json").write_text(
                    json.dumps(previous, indent=2) + "\n"
                )
                if (output / "process.log").exists():
                    (output / "process.log").rename(archive / "process.log")
            with (output / "process.log").open("w") as log:
                process = subprocess.Popen(  # noqa: S603 - argument vector; no shell expansion
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    status = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    status = 124
                    error = "timeout; process group terminated"
            elapsed = time.perf_counter() - start
            row = {
                "model": args.model,
                "target": target,
                "backend": backend,
                "process_seconds": elapsed,
                "status": status,
                "error": error,
                "config": config,
                "gpu": torch.cuda.get_device_name(),
                "slurm_job": os.environ["SLURM_JOB_ID"],
                "scope": (
                    f"complete CLI process with {args.benchmark_repeats + 1} "
                    "model forwards, including weights/compile/capture/input/output; "
                    "cached ESMC embeddings"
                ),
                "output": str(output),
            }
            reports = list(output.glob(f"{target}*.json"))
            if status == 0 and len(reports) == 1:
                report = json.loads(reports[0].read_text())
                assert report["compile"], report
                assert report["cuda_graph"], report
                assert report["precision"] == "bf16"
                assert report["autocast"] is False
                assert len(report["prediction_cifs"]) == args.samples
                row["report"] = report
                from Bio.PDB.MMCIF2Dict import MMCIF2Dict

                cif = MMCIF2Dict(report["prediction_cif"])
                row["output_atoms"] = len(cif["_atom_site.Cartn_x"])
                row["output_chains"] = len(set(cif["_atom_site.label_asym_id"]))
                row["resources"] = json.loads(
                    (output / "input.resources.json").read_text()
                )
            elif status == 0:
                row["status"] = 1
                row["error"] = "missing prediction report"
            if previous is not None:
                rows.remove(previous)
            rows.append(row)
            (args.root / f"e2e-{args.model}.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            print("FINISH", label, row["status"], round(elapsed, 3), flush=True)  # noqa: T201 - CLI output contract
    return int(any(row["status"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
