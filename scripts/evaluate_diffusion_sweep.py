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

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from compare_af3_precision import aligned_rmsd, atoms, compare, geometry

from foldforge.eval.structure import tm_score

EXPERIMENTAL = {
    "1ubq": "validation/inputs/data/1ubq/1ubq.cif",
    "3ptb": "validation/inputs/data/3ptb/3ptb.cif",
    "1a1k": "validation/inputs/data/1a1k/1a1k.cif",
    "4yx2": "validation/inputs/data/4yx2/4yx2.cif",
    "5i28x1": "validation/inputs/data/5i28/5i28.cif",
}


def ca_tm(sample: dict, target: dict) -> float | None:
    """Zhang-Skolnick TM-score on matched CA atoms, normalised by the target."""
    keys = sorted(k for k in sample.keys() & target.keys() if k[-1] == "CA")
    if not keys:
        return None
    mobile = torch.tensor(np.array([sample[k][0] for k in keys]), dtype=torch.float64)
    reference = torch.tensor(
        np.array([target[k][0] for k in keys]), dtype=torch.float64
    )
    return tm_score(mobile, reference)


def all_atom_rmsd(sample: dict, target: dict) -> float | None:
    """Kabsch RMSD over every matched heavy atom, not only CA."""
    keys = sorted(sample.keys() & target.keys())
    if len(keys) < 3:  # noqa: PLR2004 - a rigid fit needs three points
        return None
    x = np.array([sample[k][0] for k in keys])
    y = np.array([target[k][0] for k in keys])
    return aligned_rmsd(x, y)


def score_sample(sample: dict, reference: dict, experiment: dict | None) -> dict:
    record = {"geometry": geometry(sample), "vs_reference": compare(sample, reference)}
    record["vs_reference"]["tm_score"] = ca_tm(sample, reference)
    record["vs_reference"]["all_atom_rmsd_A"] = all_atom_rmsd(sample, reference)
    if experiment is not None:
        shared = {k: v for k, v in sample.items() if k in experiment}
        record["vs_experiment"] = compare(
            shared, {k: v for k, v in experiment.items() if k in shared}
        )
        record["vs_experiment"]["tm_score"] = ca_tm(shared, experiment)
        record["vs_experiment"]["all_atom_rmsd_A"] = all_atom_rmsd(shared, experiment)
    return record


def plot_similarity(results: list[dict], root: Path, reference_steps: int) -> None:
    """TM-score and CA RMSD against the same-seed reference, per target."""
    targets = sorted({r["target"] for r in results})
    steps = sorted({r["steps"] for r in results})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for target in targets:
        rows = [r for r in results if r["target"] == target]
        for ax, key, label in (
            (axes[0], "tm_score", "TM-score vs reference"),
            (axes[1], "ca_rmsd_A", "CA RMSD vs reference (A)"),
            (axes[2], "all_atom_rmsd_A", "All-atom RMSD vs reference (A)"),
        ):
            xs, means, lows, highs = [], [], [], []
            for step in steps:
                values = [
                    r["vs_reference"][key]
                    for r in rows
                    if r["steps"] == step and r["vs_reference"].get(key) is not None
                ]
                if not values:
                    continue
                xs.append(step)
                means.append(float(np.mean(values)))
                lows.append(float(np.min(values)))
                highs.append(float(np.max(values)))
            if not xs:
                continue
            line = ax.plot(xs, means, marker="o", label=target)[0]
            ax.fill_between(xs, lows, highs, color=line.get_color(), alpha=0.12)
            ax.set_ylabel(label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("diffusion steps")
        ax.grid(visible=True, alpha=0.2, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(
        f"Similarity to the same-seed {reference_steps}-step samples "
        "(mean, min-max over seeds and samples)",
        fontsize=10,
    )
    for extension in ("svg", "png"):
        fig.savefig(root / f"sweep_similarity.{extension}", dpi=180)
    plt.close(fig)


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
    plot_similarity(results, args.root, args.reference_steps)

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
            "ref_aa_rmsd": mean(
                [
                    r["vs_reference"]["all_atom_rmsd_A"]
                    for r in rows
                    if r["vs_reference"].get("all_atom_rmsd_A") is not None
                ]
            ),
            "exp_aa_rmsd": mean(
                [
                    r["vs_experiment"]["all_atom_rmsd_A"]
                    for r in rows
                    if "vs_experiment" in r
                    and r["vs_experiment"].get("all_atom_rmsd_A") is not None
                ]
            ),
            "ref_tm": mean(
                [
                    r["vs_reference"]["tm_score"]
                    for r in rows
                    if r["vs_reference"].get("tm_score") is not None
                ]
            ),
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
        "| Steps | Samples | Exp CA lDDT | Exp CA RMSD (A) | Exp all-atom RMSD (A) | "
        "TM vs ref | CA RMSD vs ref (A) | All-atom RMSD vs ref (A) | CA pLDDT | "
        "Peptide outliers | Clashes | Forward (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_steps: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_steps[r["steps"]].append(r)
    for steps in sorted(by_steps, reverse=True):
        s = summarize(by_steps[steps])
        lines.append(
            f"| {steps} | {s['n']} | {fmt(s['exp_ca_lddt'])} | "
            f"{fmt(s['exp_ca_rmsd'])} | {fmt(s['exp_aa_rmsd'])} | {fmt(s['ref_tm'])} | "
            f"{fmt(s['ref_ca_rmsd'])} | {fmt(s['ref_aa_rmsd'])} | "
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
            "| Steps | Exp CA lDDT | Exp CA RMSD (A) | Exp all-atom RMSD (A) | "
            "TM vs ref | CA RMSD vs ref (A) | All-atom RMSD vs ref (A) | CA pLDDT | "
            "Peptide outliers | Clashes | Forward (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for steps in sorted(per_steps, reverse=True):
            s = summarize(per_steps[steps])
            lines.append(
                f"| {steps} | {fmt(s['exp_ca_lddt'])} | {fmt(s['exp_ca_rmsd'])} | "
                f"{fmt(s['exp_aa_rmsd'])} | {fmt(s['ref_tm'])} | "
                f"{fmt(s['ref_ca_rmsd'])} | "
                f"{fmt(s['ref_aa_rmsd'])} | "
                f"{fmt(s['ca_plddt'], 2)} | "
                f"{s['peptide_outliers']} | "
                f"{s['clashes']} | {fmt(s['seconds'], 1)} |"
            )
        lines.append("")
    (args.root / "sweep_results.md").write_text("\n".join(lines))
    print(args.root / "sweep_results.md")  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
