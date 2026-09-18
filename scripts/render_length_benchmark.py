"""Render the replicated-5I28 benchmark: a five-mode matrix plus length scaling.

Each ``--set LABEL MATRIX SCALING`` names one GPU class: ``MATRIX`` holds the
four-model, five-mode rows at one length and ``SCALING`` holds the reference and
MiniWorld rows across lengths, including rows that failed with ``cuda_oom``.
Targets are ``5i28x<N>``: the 128-residue azurin chain repeated N times, so the
token count is 128*N. Writes ``docs/benchmark_results.md`` and the assets.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from render_benchmark_results import MODELS, MODES, validate
from render_benchmark_results import MSA_POLICY as POLICY

SCALING_MODES = {
    "pytorch_eager_reference": MODES["pytorch_eager_reference"],
    "pytorch_compile_reference": MODES["pytorch_compile_reference"],
    "miniworld_graph": MODES["miniworld_graph"],
}
CHAIN_RESIDUES = 128
COLORS = {
    "pytorch_eager_reference": "#94a3b8",
    "pytorch_compile_reference": "#64748b",
    "pytorch_compile_bf16": "#4b82c4",
    "cuequiv_compile": "#c98a37",
    "miniworld_graph": "#258572",
}


def tokens_of(target: str) -> int:
    match = re.fullmatch(r"5i28x(\d+)", target)
    if not match:
        message = f"Unexpected target {target!r}; expected 5i28x<N>"
        raise ValueError(message)
    return CHAIN_RESIDUES * int(match.group(1))


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("e2e-*.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        message = f"No e2e-*.json under {root}"
        raise ValueError(message)
    return rows


def gpu_of(rows: list[dict]) -> str:
    gpus = sorted({row["gpu"] for row in rows})
    if len(gpus) != 1:
        message = f"Mixed GPUs in one set: {gpus}"
        raise ValueError(message)
    return gpus[0]


def matrix_section(label: str, rows: list[dict], assets: Path) -> list[str]:
    targets = {row["target"] for row in rows}
    if len(targets) != 1:
        message = f"Matrix rows must share one target, got {sorted(targets)}"
        raise ValueError(message)
    target = targets.pop()
    indexed = validate(rows, target=target)
    gpu = gpu_of(rows)
    tokens = tokens_of(target)
    (assets / f"benchmark_matrix_{label}.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    positions = np.arange(len(MODELS))
    width = 0.16
    for i, (mode, name) in enumerate(MODES.items()):
        timings = [indexed[m, mode]["report"]["model_seconds_warm"] for m in MODELS]
        values = [statistics.median(t) for t in timings]
        error = np.array(
            [
                [v - min(t) for v, t in zip(values, timings, strict=True)],
                [max(t) - v for v, t in zip(values, timings, strict=True)],
            ]
        )
        bars = ax.bar(
            positions + (i - 2) * width,
            values,
            width,
            color=COLORS[mode],
            label=name,
            yerr=error,
            capsize=2,
        )
        ax.bar_label(bars, labels=[f"{x:.1f}" for x in values], fontsize=8, padding=4)
    ax.set_xticks(positions, list(MODELS.values()))
    ax.set_ylabel("Warm complete-model forward (seconds; lower is faster)")
    ax.set_title(
        f"{target} · {tokens} tokens · {gpu}\nfive execution modes · five samples"
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.legend(ncols=2, loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(visible=True, alpha=0.2)
    ax.set_axisbelow(True)
    for extension in ("svg", "png"):
        fig.savefig(assets / f"benchmark_latency_{label}.{extension}", dpi=180)
    plt.close(fig)

    lines = [
        f"## {gpu}: five modes at {tokens} tokens",
        "",
        f"![Latency at {tokens} tokens](assets/benchmark_latency_{label}.svg)",
        "",
        "| Model | Reference eager | Reference compile | Native BF16 compile | "
        "cuEq BF16 compile | MiniWorld BF16 compile + graph | "
        "MiniWorld vs reference compile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, name in MODELS.items():
        medians = [
            indexed[model, mode]["report"]["model_seconds_warm_median"]
            for mode in MODES
        ]
        ratio = medians[1] / medians[4]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{x:.3f}" for x in medians)
            + f" | {ratio:.2f}x |"
        )
    lines += [
        "",
        "| Model | Mode | Peak allocated GiB | Warm min-max (s) | Initial forward (s) "
        "| Compiled graphs / manual replays | Job |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, name in MODELS.items():
        for mode, mode_name in MODES.items():
            report = indexed[model, mode]["report"]
            warm = report["model_seconds_warm"]
            lines.append(
                f"| {name} | {mode_name} | "
                f"{report['model_peak_allocated_bytes'] / 2**30:.2f} | "
                f"{min(warm):.3f}-{max(warm):.3f} | "
                f"{report['model_seconds_cold']:.3f} | "
                f"{report['compiled_graphs']} / {report['cuda_graph_replays']} | "
                f"{indexed[model, mode]['slurm_job']} |"
            )
    lines += ["", f"[Raw rows](assets/benchmark_matrix_{label}.json).", ""]
    return lines


def compact_row(row: dict) -> dict:
    """Shrink a published row: no per-token confidence, one entry for N equal chains."""
    row = {k: v for k, v in row.items() if k != "resources"}
    report = row.get("report")
    if report and "confidence" in report:
        row["report"] = {k: v for k, v in report.items() if k != "confidence"}
    chains = row.get("input_resources")
    if isinstance(chains, list) and chains:
        first = {k: v for k, v in chains[0].items() if k != "chain_index"}
        same = all(
            {k: v for k, v in chain.items() if k != "chain_index"} == first
            for chain in chains
        )
        if same:
            row["input_resources"] = {"identical_chains": len(chains), "chain": first}
    return row


def model_table(model: str, targets: list[str], index: dict, cell) -> list[str]:  # noqa: ANN001 - local formatter
    lines: list[str] = []
    lines += [
        f"### {MODELS[model]}",
        "",
        "| Tokens | Chains | "
        + " | ".join(f"{name} s | GiB" for name in SCALING_MODES.values())
        + " |",
        "|---:|---:|" + "---:|---:|" * len(SCALING_MODES),
    ]
    for target in targets:
        found = [index.get((model, mode, target)) for mode in SCALING_MODES]
        if all(row is None for row in found):
            continue
        cells = []
        for row in found:
            cells += cell(row)
        lines.append(
            f"| {tokens_of(target)} | {tokens_of(target) // CHAIN_RESIDUES} | "
            + " | ".join(cells)
            + " |"
        )
        if all(row is None or row["status"] for row in found):
            lines += ["", "Every longer length fails in every mode."]
            break
    lines.append("")
    return lines


def scaling_section(label: str, rows: list[dict], assets: Path) -> list[str]:
    gpu = gpu_of(rows)
    rows = [row for row in rows if row["mode"] in SCALING_MODES]
    index = {(row["model"], row["mode"], row["target"]): row for row in rows}
    targets = sorted({row["target"] for row in rows}, key=tokens_of)
    models = [m for m in MODELS if any(k[0] == m for k in index)]
    (assets / f"benchmark_scaling_{label}.json").write_text(
        json.dumps([compact_row(row) for row in rows], indent=2) + "\n"
    )

    def cell(row: dict | None) -> tuple[str, str]:
        if row is None:
            return ("-", "-")
        if row["status"]:
            reason = row.get("error") or f"exit {row['status']}"
            return (f"**{'OOM' if reason == 'cuda_oom' else reason}**", "-")
        report = row["report"]
        single = len(report["model_seconds_warm"]) == 1
        return (
            f"{report['model_seconds_warm_median']:.2f}{'†' if single else ''}",
            f"{report['model_peak_allocated_bytes'] / 2**30:.1f}",
        )

    lines = [
        f"## {gpu}: length scaling",
        "",
        f"![Latency versus length](assets/scaling_latency_{label}.svg)",
        "",
        f"![Peak memory versus length](assets/scaling_memory_{label}.svg)",
        "",
        "Warm median seconds and peak allocated GiB per length. **OOM** marks a "
        "CUDA out-of-memory failure recorded in that process log; other failures "
        "show their reason. Every length is its own process. † marks one warm "
        "forward instead of the median of five, used where one forward takes "
        "tens of minutes.",
        "",
    ]
    for model in models:
        lines += model_table(model, targets, index, cell)

    for metric, ylabel, name in (
        ("model_seconds_warm_median", "Warm forward (s)", "latency"),
        ("model_peak_allocated_bytes", "Peak allocated (GiB)", "memory"),
    ):
        fig, axes = plt.subplots(
            1, len(models), figsize=(4.2 * len(models), 4.4), layout="constrained"
        )
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models, strict=True):
            shown: set[int] = set()
            for mode, mode_name in SCALING_MODES.items():
                xs, ys = [], []
                for target in targets:
                    row = index.get((model, mode, target))
                    if row is None or row["status"]:
                        continue
                    value = row["report"][metric]
                    xs.append(tokens_of(target))
                    ys.append(value / 2**30 if name == "memory" else value)
                if xs:
                    ax.plot(xs, ys, marker="o", color=COLORS[mode], label=mode_name)
                    shown.update(xs)
                failed = [
                    tokens_of(t)
                    for t in targets
                    if (row := index.get((model, mode, t))) is not None
                    and row["status"]
                ]
                if failed and ys:
                    shown.add(min(failed))
                    ax.plot(
                        [min(failed)],
                        [ys[-1]],
                        marker="x",
                        color=COLORS[mode],
                        linestyle="none",
                        markersize=9,
                    )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ticks = sorted(shown)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=8, rotation=45)
            ax.minorticks_off()
            ax.set_title(MODELS[model])
            ax.set_xlabel("tokens")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(visible=True, alpha=0.2, which="both")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{gpu} · x marks the first length that failed", fontsize=10)
        for extension in ("svg", "png"):
            fig.savefig(assets / f"scaling_{name}_{label}.{extension}", dpi=180)
        plt.close(fig)
    lines += [f"[Raw rows](assets/benchmark_scaling_{label}.json).", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        nargs=3,
        action="append",
        metavar=("LABEL", "MATRIX", "SCALING"),
        required=True,
        help="One GPU class: label, matrix results dir, scaling results dir "
        "('-' skips one)",
    )
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--notes", type=Path)
    args = parser.parse_args()
    assets = args.docs / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    updated = datetime.now(tz=UTC).date().isoformat()
    lines = [
        "# Inference benchmark results",
        "",
        f"Updated {updated}. **5I28 azurin, 128 residues per chain, replicated N",
        "times as N identical chains.** Every chain carries the same MSA and",
        "templates, so length scales while the per-residue conditioning does not.",
        "",
        "Reference means each model's released precision policy in FoldForge's PyTorch",
        "backend; native BF16 stores learned parameters in BF16 except FP32 norms and",
        "uses no autocast. Latency is the median of five warm complete-model forwards",
        "and excludes featurization, checkpoint loading, initial compilation/capture,",
        "precomputed ESMC embeddings and CIF output. Error bars are warm min/max.",
        "",
        "Inputs follow one MSA policy for every model: up to "
        f"{POLICY['prepared_rows']} prepared alignment rows, "
        f"{POLICY['sampled_rows_per_recycle']} rows sampled per recycle, and "
        f"{POLICY['templates_per_chain']} templates per chain",
        "(ESMFold2 has no template path). Compile and manual CUDA graphs cover the",
        "denoiser; 200 requested diffusion steps (ESMFold2 keeps 134 after its sigma",
        "cutoff), 10 recycles, five samples, trunk_seed=0 and diffusion_seed=0.",
        "",
    ]
    for label, matrix, scaling in args.set:
        if matrix != "-":
            lines += matrix_section(label, load_rows(Path(matrix)), assets)
        if scaling != "-":
            lines += scaling_section(label, load_rows(Path(scaling)), assets)
    lines += [
        "## Reproduction",
        "",
        "```bash",
        "python scripts/benchmark_end_to_end.py --model af3 --targets 5i28x4 \\",
        "  --root matrix --benchmark-repeats 5 --steps 200 --recycles 10 --samples 5",
        "python scripts/benchmark_end_to_end.py --model af3 \\",
        "  --targets 5i28x4 5i28x8 5i28x12 5i28x16 5i28x20 5i28x24 5i28x32 \\",
        "  --modes pytorch_eager_reference pytorch_compile_reference \\",
        "    miniworld_graph \\",
        "  --root scaling --benchmark-repeats 5 --steps 200 --recycles 10 --samples 5",
        "python scripts/render_length_benchmark.py --set a100 matrix scaling",
        "```",
        "",
        "Use `--model opendde`, `--model esmfold2` or `--model protenix --variant "
        "protenix-v2`; inputs live under `validation/inputs/qualification/5i28x<N>/`.",
        "",
    ]
    if args.notes:
        lines += ["## Notes", "", args.notes.read_text().strip(), ""]
    (args.docs / "benchmark_results.md").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
