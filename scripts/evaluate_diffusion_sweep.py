"""Score a diffusion sweep: experimental agreement, drift from 200 steps, geometry.

Reads every ``<root>/<target>/seed-<s>/sweep.json`` written by
``sweep_diffusion_steps.py`` and scores each sample with the same functions as
the benchmark reports: CA lDDT and RMSD against the experimental structure on
observed CA atoms, CA RMSD against the same-seed reference step count, peptide
and clash counts, and the forward time. Writes ``<root>/sweep_results.json`` and
``<root>/sweep_results.md`` with one row per step count aggregated over targets,
seeds and samples, followed by a per-target table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from compare_af3_precision import atoms, compare, geometry

EXPERIMENTAL = {
    "1ubq": "validation/inputs/data/1ubq/1ubq.cif",
    "3ptb": "validation/inputs/data/3ptb/3ptb.cif",
    "1a1k": "validation/inputs/data/1a1k/1a1k.cif",
    "4yx2": "validation/inputs/data/4yx2/4yx2.cif",
    "5i28x1": "validation/inputs/data/5i28/5i28.cif",
}


def score_sample(sample: dict, reference: dict, experiment: dict | None) -> dict:
    record = {"geometry": geometry(sample), "vs_reference": compare(sample, reference)}
    if experiment is not None:
        shared = {k: v for k, v in sample.items() if k in experiment}
        record["vs_experiment"] = compare(
            shared, {k: v for k, v in experiment.items() if k in shared}
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-steps", type=int, default=200)
    args = parser.parse_args()
    results: list[dict] = []
    experiments: dict[str, dict | None] = {}
    for sweep in sorted(args.root.glob("*/seed-*/sweep.json")):
        records = json.loads(sweep.read_text())
        by_steps = {r["steps"]: r for r in records}
        if args.reference_steps not in by_steps:
            continue
        target = records[0]["target"]
        if target not in experiments:
            path = EXPERIMENTAL.get(target)
            experiments[target] = atoms(Path(path)) if path else None
        reference = [
            atoms(Path(p)) for p in by_steps[args.reference_steps]["prediction_cifs"]
        ]
        for record in records:
            samples = [atoms(Path(p)) for p in record["prediction_cifs"]]
            for index, (sample, ref) in enumerate(zip(samples, reference, strict=True)):
                scored = score_sample(sample, ref, experiments[target])
                results.append(
                    {
                        "target": target,
                        "diffusion_seed": record["diffusion_seed"],
                        "steps": record["steps"],
                        "sample": index,
                        "model_seconds": record["model_seconds"],
                        **scored,
                    }
                )
    (args.root / "sweep_results.json").write_text(json.dumps(results, indent=2) + "\n")

    def summarize(rows: list[dict]) -> dict:
        def mean(values: list[float]) -> float | None:
            return statistics.fmean(values) if values else None

        return {
            "n": len(rows),
            "exp_ca_lddt": mean(
                [r["vs_experiment"]["ca_lddt"] for r in rows if "vs_experiment" in r]
            ),
            "exp_ca_rmsd": mean(
                [r["vs_experiment"]["ca_rmsd_A"] for r in rows if "vs_experiment" in r]
            ),
            "ref_ca_rmsd": mean([r["vs_reference"]["ca_rmsd_A"] for r in rows]),
            "ca_plddt": mean([r["geometry"]["ca_mean_plddt"] for r in rows]),
            "peptide_outliers": sum(
                r["geometry"]["peptide_outside_1_0_to_1_7_A"] for r in rows
            ),
            "clashes": sum(r["geometry"]["heavy_atom_pairs_below_1_A"] for r in rows),
            "seconds": mean([r["model_seconds"] for r in rows]),
        }

    def fmt(value: float | None, digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Diffusion step sweep",
        "",
        "Released schedule and Euler solver; only the step count changes. Experimental",
        "columns use observed CA atoms; drift is CA RMSD against the same-seed",
        f"{args.reference_steps}-step samples; counts sum over all samples.",
        "",
        "| Steps | Samples | Exp CA lDDT | Exp CA RMSD (A) | Drift vs ref (A) | "
        "CA pLDDT | Peptide outliers | Clashes | Forward (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_steps: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_steps[r["steps"]].append(r)
    for steps in sorted(by_steps, reverse=True):
        s = summarize(by_steps[steps])
        lines.append(
            f"| {steps} | {s['n']} | {fmt(s['exp_ca_lddt'])} | "
            f"{fmt(s['exp_ca_rmsd'])} | {fmt(s['ref_ca_rmsd'])} | "
            f"{fmt(s['ca_plddt'], 2)} | {s['peptide_outliers']} | "
            f"{s['clashes']} | {fmt(s['seconds'], 1)} |"
        )
    lines += ["", "## Per target", ""]
    by_target: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_target[r["target"]][r["steps"]].append(r)
    for target, per_steps in by_target.items():
        lines += [
            f"### {target}",
            "",
            "| Steps | Exp CA lDDT | Exp CA RMSD (A) | Drift vs ref (A) | "
            "CA pLDDT | Peptide outliers | Clashes | Forward (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for steps in sorted(per_steps, reverse=True):
            s = summarize(per_steps[steps])
            lines.append(
                f"| {steps} | {fmt(s['exp_ca_lddt'])} | {fmt(s['exp_ca_rmsd'])} | "
                f"{fmt(s['ref_ca_rmsd'])} | {fmt(s['ca_plddt'], 2)} | "
                f"{s['peptide_outliers']} | "
                f"{s['clashes']} | {fmt(s['seconds'], 1)} |"
            )
        lines.append("")
    (args.root / "sweep_results.md").write_text("\n".join(lines))
    print(args.root / "sweep_results.md")  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
