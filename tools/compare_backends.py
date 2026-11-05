"""Compare each backend with PyTorch for one model and input.

Run one model on one input under each backend and compare them with the PyTorch
backend.

One command, one config: the same ``foldforge fold`` (the spec and the runtime
config you give) runs once per backend in its own process, so the Anthropic payload
locations are present only for the Anthropic run and the engine's JIT state is not
shared. Each row records the backend, the backend of every op, the warm median model
time, the cold time, the mean pLDDT and the difference of the predicted coordinates
from the PyTorch run: max and rms of the per-atom displacement as they come out, and
after the best rigid superposition (the part of the difference that is a different
pose rather than a different structure). A run that refuses, is out of memory or
crashes is a row too.

python tools/compare_backends.py af3 --spec input.yaml --config runtime.yaml --out
compare/af3-n1 python tools/compare_backends.py af3 --spec input.yaml --config
runtime.yaml --out compare/af3-n1 --backends pytorch miniworld

Run it inside a Slurm GPU allocation. ``--config`` is any runtime config
(docs/guides/CONFIG.md); its ``backend`` is replaced by each backend in turn and its
``engine.ops`` / ``engine.unsupported`` are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from foldforge.modules.engine.backends import BACKENDS

REFERENCE = "pytorch"
#: Locations the Anthropic backend reads; only its own run sees them (the engine's
#:miniworld modules use a named payload).
PAYLOAD_VARIABLES = (
    "MINIWORLD_ANTHROPIC_ROOT",
    "OPT_CORE_DIR",
    "TRIMUL_NATIVE_BUILD_DIR",
)


def superpose(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Kabsch-align moving coordinates onto fixed coordinates.

    ``moving`` rigidly superposed on ``fixed`` (Kabsch, equal weights), both
    ``[atoms, 3]``.
    """
    a = moving - moving.mean(0)
    b = fixed - fixed.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    return a @ rotation + fixed.mean(0)


def coordinate_difference(
    coordinates: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Run coordinate difference.

    Max / rms per-atom displacement over all samples, raw and after superposition,
    in the arrays' unit (Angstrom).
    """
    if coordinates.shape != reference.shape:
        return {"shape_mismatch": float("nan")}
    raw, fitted = [], []
    for sample, other in zip(coordinates, reference, strict=True):
        keep = np.isfinite(sample).all(-1) & np.isfinite(other).all(-1)
        raw.append(np.linalg.norm(sample[keep] - other[keep], axis=-1))
        fitted.append(
            np.linalg.norm(superpose(sample[keep], other[keep]) - other[keep], axis=-1)
        )
    raw, fitted = np.concatenate(raw), np.concatenate(fitted)
    return {
        "coord_max": float(raw.max()),
        "coord_rms": float(np.sqrt((raw**2).mean())),
        "coord_max_superposed": float(fitted.max()),
        "coord_rms_superposed": float(np.sqrt((fitted**2).mean())),
    }


def run_backend(args: argparse.Namespace, backend: str, base: dict) -> dict:
    """One ``foldforge fold`` process on one backend; the row of its result."""
    root = args.out / backend
    root.mkdir(parents=True, exist_ok=True)
    config = {**base, "backend": backend}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    environment = dict(os.environ)
    if backend != "anthropic":
        for name in PAYLOAD_VARIABLES:
            environment.pop(name, None)
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from foldforge.models.inference import run; "
            "raise SystemExit(run(sys.argv[1], sys.argv[2:]))"
        ),
        args.model,
        "--spec",
        str(args.spec),
        "--config",
        str(root / "config.yaml"),
        "--out",
        str(root / "run"),
    ]
    row = {
        "model": args.model,
        "backend": backend,
        "config": config,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    start = time.perf_counter()
    with (root / "process.log").open("w") as log:
        status = subprocess.run(  # noqa: S603
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
            timeout=args.timeout,
        ).returncode
    row.update(exit=status, wall_s=round(time.perf_counter() - start, 1))
    text = (root / "process.log").read_text(errors="replace")
    reports = sorted((root / "run").glob("*-seed*.json"))
    if status == 0 and reports:
        report = json.loads(reports[0].read_text())
        loaded = report["checkpoint"].get("backends", {})
        row.update(
            status="ok",
            cold=round(report["model_seconds_cold"], 3),
            warm_median=(
                round(report["model_seconds_warm_median"], 3)
                if report["model_seconds_warm_median"] is not None
                else float("nan")
            ),
            warm=[round(x, 3) for x in report["model_seconds_warm"]],
            plddt=round(report["mean_plddt"], 3),
            ops=loaded.get("ops"),
            fallbacks=loaded.get("fallbacks"),
            coordinates=str(reports[0].with_suffix(".npz")),
        )
    else:
        tail = [line for line in text.splitlines() if line.strip()][-1:] or [""]
        if "OutOfMemoryError" in text or "out of memory" in text:
            row.update(status="oom")
        elif "BackendUnsupportedError" in text:
            row.update(status="refused")
        else:
            row.update(status="error")
        row["error"] = tail[0][:300]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=[REFERENCE, "miniworld", "cuequivariance", "anthropic"],
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="Append the rows to this jsonl (default: <out>/rows.jsonl)",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        message = "Run inside an allocated Slurm compute job"
        raise RuntimeError(message)
    args.out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.config.read_text()) or {}
    order = [REFERENCE, *[b for b in args.backends if b != REFERENCE]]
    rows = [run_backend(args, backend, base) for backend in order]
    reference = next(
        (r for r in rows if r["backend"] == REFERENCE and r["status"] == "ok"), None
    )
    for row in rows:
        if row["status"] != "ok":
            continue
        if reference is None:
            row["vs_pytorch"] = "no pytorch run"
        elif row is not reference:
            current = np.load(row["coordinates"])["coordinates"]
            row.update(
                coordinate_difference(
                    current, np.load(reference["coordinates"])["coordinates"]
                )
            )
            row["plddt_delta"] = round(row["plddt"] - reference["plddt"], 3)
    target = args.rows or args.out / "rows.jsonl"
    with target.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(  # noqa: T201
        f"{'backend':16}{'status':9}{'warm median s':>14}{'pLDDT':>8}"
        f"{'rms A':>8}{'max A':>8}{'rms A fit':>10}"
    )
    for row in rows:
        print(  # noqa: T201
            f"{row['backend']:16}{row['status']:9}"
            f"{row.get('warm_median', float('nan')):14.3f}"
            f"{row.get('plddt', float('nan')):8.2f}"
            f"{row.get('coord_rms', float('nan')):8.3f}"
            f"{row.get('coord_max', float('nan')):8.3f}"
            f"{row.get('coord_rms_superposed', float('nan')):10.3f}"
        )
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
