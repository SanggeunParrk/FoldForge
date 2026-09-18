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

MODES = {
    "pytorch_eager_fp32": ("pytorch", False, False, "fp32"),
    "pytorch_compile_fp32": ("pytorch", True, False, "fp32"),
    "pytorch_compile_bf16": ("pytorch", True, False, "bf16"),
    "cuequiv_compile": ("cuequivariance", True, False, "bf16"),
    "miniworld_graph": ("miniworld", True, True, "bf16"),
}


REFERENCE_MODES = {
    "pytorch_eager_reference": ("pytorch", False, False, "model_default"),
    "pytorch_compile_reference": ("pytorch", True, False, "model_default"),
}

AF3_REFERENCE_MODES = {
    "pytorch_eager_reference": ("pytorch", False, False, "af3_default"),
    "pytorch_compile_reference": ("pytorch", True, False, "af3_default"),
}

# Historical BF16 names remain explicit-only for reproducing recorded runs.
LEGACY_MODES = {
    "pytorch_eager": ("pytorch", False, False, "bf16"),
    "pytorch_compile": ("pytorch", True, False, "bf16"),
}


def main() -> int:  # noqa: PLR0912 - complete benchmark scenario matrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["af3", "esmfold2", "protenix", "opendde"]
    )
    parser.add_argument("--root", type=Path, default=Path("benchmark/e2e"))
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--backends",
        nargs="+",
        choices=["pytorch", "miniworld", "cuequivariance"],
        help="Legacy comparison: compile and CUDA graph enabled for every backend",
    )
    parser.add_argument("--targets", nargs="+", default=["3ptb", "4yx2"])
    parser.add_argument("--resume", action="store_true")
    execution.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES) + list(AF3_REFERENCE_MODES) + list(LEGACY_MODES),
        help="Explicit execution modes; defaults to all five modes",
    )
    parser.add_argument("--variant", choices=["protenix-v2"])
    parser.add_argument("--no-templates", action="store_true")
    parser.add_argument("--lm-cache", type=Path)
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--trunk-seed", type=int, default=0)
    parser.add_argument("--diffusion-seed", type=int, default=0)
    parser.add_argument("--no-bucketing", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--recycles", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--msa-depth",
        type=int,
        default=16384,
        help="Input MSA row cap; the default is AF3's msa_crop_size",
    )
    parser.add_argument(
        "--template-n",
        type=int,
        default=4,
        choices=range(5),
        help="Templates per chain; the default is AF3's max_templates",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        msg = "Run inside an allocated Slurm compute job"
        raise RuntimeError(msg)
    if not torch.cuda.is_available():
        msg = "Allocated job has no usable CUDA device"
        raise RuntimeError(msg)
    if args.variant and args.model != "protenix":
        parser.error("--variant applies only to Protenix")
    default_modes = [
        *REFERENCE_MODES,
        "pytorch_compile_bf16",
        "cuequiv_compile",
        "miniworld_graph",
    ]
    selected_modes = args.modes or default_modes
    reference_modes = AF3_REFERENCE_MODES if args.model == "af3" else REFERENCE_MODES
    scenarios = (
        [(backend, backend, True, True, "bf16") for backend in args.backends]
        if args.backends
        else [
            (mode, *{**MODES, **reference_modes, **LEGACY_MODES}[mode])
            for mode in selected_modes
        ]
    )
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
        sibling_template_db = source.parent / "template.lmdb"
        if (
            spec.get("template_db")
            and not Path(spec["template_db"]).exists()
            and sibling_template_db.exists()
        ):
            spec["template_db"] = str(sibling_template_db.resolve())
        # ESMFold2 has no template conditioning path; its input keeps the same
        # MSA cap and is recorded as template-free rather than failing.
        templates_dropped = args.model == "esmfold2" and bool(spec.get("template"))
        if args.no_templates or templates_dropped:
            spec["template"] = {}
        spec["template_n"] = args.template_n
        spec.update(n_diffusion_samples=args.samples, diffusion_batch_size=args.samples)
        path = args.root / "inputs" / args.model / f"{target}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(spec))
        for mode, backend, compile_enabled, graph_enabled, precision in scenarios:
            label = f"{args.model}-{target}-{mode}"
            previous = next(
                (
                    row
                    for row in rows
                    if row["target"] == target
                    and row.get("mode", row["backend"]) == mode
                ),
                None,
            )
            output = args.root / "e2e" / label
            output.mkdir(parents=True, exist_ok=True)
            config = {
                "backend": backend,
                "precision": precision,
                "trunk_seed": args.trunk_seed,
                "diffusion_seed": args.diffusion_seed,
                "trunk": {"recycles": args.recycles, "msa_depth": args.msa_depth},
                "diffusion": {"steps": args.steps},
                "execution": {
                    "compile": compile_enabled,
                    "cuda_graph": graph_enabled,
                    "bucketing": not args.no_bucketing,
                    "scope": "denoiser",
                    "max_graphs": 4,
                    "benchmark_repeats": args.benchmark_repeats,
                },
            }
            if args.variant:
                config["variant"] = args.variant
            if previous is not None and previous["status"] == 0:
                if previous["config"] != config or previous.get("input_spec") != spec:
                    msg = f"Resume inputs/config changed for {label}; use another root"
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
                        args.lm_cache
                        or (
                            Path("validation/inputs/data")
                            / target
                            / "cache/esmc_hidden_states.pt"
                        )
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
            if status and error is None:
                log_text = (output / "process.log").read_text(errors="replace")
                if "OutOfMemoryError" in log_text or "out of memory" in log_text:
                    error = "cuda_oom"
            row = {
                "model": args.model,
                "mode": mode,
                "target": target,
                "backend": backend,
                "process_seconds": elapsed,
                "status": status,
                "error": error,
                "config": config,
                "gpu": torch.cuda.get_device_name(),
                "slurm_job": os.environ["SLURM_JOB_ID"],
                "host": os.uname().nodename,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "input_spec": spec,
                "templates_dropped": (
                    "esmfold2 checkpoint has no template conditioning path"
                    if templates_dropped
                    else None
                ),
                "input_resources": (
                    json.loads((output / "input.resources.json").read_text())
                    if (output / "input.resources.json").exists()
                    else None
                ),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
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
                assert report["compile"] is compile_enabled, report
                assert report["cuda_graph"] is graph_enabled, report
                assert report["compile_requested"] is compile_enabled
                assert report["cuda_graph_requested"] is graph_enabled
                assert report["backend"] == backend
                assert report["trunk_seed"] == args.trunk_seed
                assert report["diffusion_seed"] == args.diffusion_seed
                assert report["seed_policy"] == "split-v1"
                if not compile_enabled:
                    assert report["compiled_graphs"] == 0, report
                assert len(report["model_seconds_warm"]) == args.benchmark_repeats
                assert report["precision"] == precision
                assert report["autocast"] is (
                    precision == "model_default"
                    and args.model in {"esmfold2", "protenix"}
                )
                if args.model == "af3":
                    assert report["samples_per_denoiser_call"] == args.samples
                    if compile_enabled or graph_enabled:
                        assert report["wrapped_calls"] == (
                            args.steps * (args.benchmark_repeats + 1)
                        )
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
