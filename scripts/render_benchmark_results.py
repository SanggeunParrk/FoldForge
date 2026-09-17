"""Render the complete four-model, five-mode 4YX2 benchmark matrix."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = {
    "pytorch_eager_reference": "PyTorch eager · default",
    "pytorch_compile_reference": "PyTorch compile · default",
    "pytorch_compile_bf16": "PyTorch compile · native BF16",
    "cuequiv_compile": "cuEq + compile · native BF16",
    "miniworld_graph": "MiniWorld + compile + graph · native BF16",
}
MODELS = {
    "af3": "AF3",
    "opendde": "OpenDDE",
    "esmfold2": "ESMFold2",
    "protenix": "Protenix v2",
}


REPEATS = 5
AF3_TRUNK_PASSES = 11
# Input contract follows the official AF3 pipeline: msa_crop_size and max_templates.
MSA_DEPTH = 16384
TEMPLATE_N = 4


def validate(rows: list[dict]) -> dict:
    policies = {r["report"].get("seed_policy", "legacy") for r in rows}
    if len(policies) != 1:
        message = "Do not mix legacy and split-seed benchmark measurements"
        raise ValueError(message)
    indexed = {(r["model"], r["mode"]): r for r in rows}
    if len(rows) != len(MODES) * len(MODELS) or set(indexed) != {
        (m, s) for m in MODELS for s in MODES
    }:
        message = "Expected exactly four models by five modes"
        raise ValueError(message)
    for (model, mode), row in indexed.items():
        report, config = row["report"], row["config"]
        reference = mode.endswith("reference")
        expected_precision = (
            ("af3_default" if model == "af3" else "model_default")
            if reference
            else "bf16"
        )
        expected_backend = (
            "cuequivariance"
            if mode == "cuequiv_compile"
            else "miniworld"
            if mode == "miniworld_graph"
            else "pytorch"
        )
        timings = report["model_seconds_warm"]
        expected_autocast = reference and model in {"esmfold2", "protenix"}
        valid = (
            row["status"] == 0
            and row["target"] == "4yx2"
            and row["backend"] == config["backend"] == expected_backend
            and (
                config.get("seed") == 0
                if "seed" in config
                else config.get("trunk_seed") == config.get("diffusion_seed") == 0
            )
            and config["execution"]["scope"] == "denoiser"
            and config["execution"]["bucketing"]
            and report["execution_scope"] == "denoiser"
            and (report["compiled_graphs"] > 0) == (mode != "pytorch_eager_reference")
            and (report["cuda_graph_replays"] > 0) == (mode == "miniworld_graph")
            and len(timings) == REPEATS
            and all(np.isfinite(t) and t > 0 for t in timings)
            and report["model_seconds_warm_median"] == statistics.median(timings)
            and config["precision"] == expected_precision
            and report["precision"] == expected_precision
            and report["autocast"] == expected_autocast
            and config["trunk"] == {"recycles": 10, "msa_depth": MSA_DEPTH}
            and config["diffusion"] == {"steps": 200}
            and row["input_spec"]["n_diffusion_samples"] == REPEATS
            and row["input_spec"].get("template_n") == TEMPLATE_N
            and (
                not row["input_spec"].get("template")
                if model == "esmfold2"
                else bool(row["input_spec"].get("template"))
            )
            and report["compile"] == (mode != "pytorch_eager_reference")
            and report["cuda_graph"] == (mode == "miniworld_graph")
            and len(report["prediction_cifs"]) == REPEATS
        )
        if not valid:
            message = f"Invalid measurement contract: {model}/{mode}"
            raise ValueError(message)
        if row["input_spec"] != indexed[model, "pytorch_eager_reference"]["input_spec"]:
            message = f"Different model inputs: {model}/{mode}"
            raise ValueError(message)
        if model == "protenix" and config.get("variant") != "protenix-v2":
            message = "Expected Protenix v2 checkpoint"
            raise ValueError(message)
        if model == "af3":
            if report["trunk_passes"] != AF3_TRUNK_PASSES:
                message = "AF3 requires 11 trunk passes"
                raise ValueError(message)
            if report.get("samples_per_denoiser_call") != REPEATS:
                message = "AF3 must batch all five diffusion samples"
                raise ValueError(message)
            if mode != "pytorch_eager_reference" and report["wrapped_calls"] != (
                200 * (REPEATS + 1)
            ):
                message = "AF3 requires 200 denoiser calls per complete forward"
                raise ValueError(message)
    return indexed


def render(rows: list[dict], docs: Path) -> None:
    indexed = validate(rows)
    seed_description = (
        "trunk_seed=0 and diffusion_seed=0 (split-v1)"
        if rows[0]["report"].get("seed_policy") == "split-v1"
        else "the historical coupled seed=0 policy"
    )
    assets = docs / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "benchmark_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    positions = np.arange(4)
    width = 0.16
    colors = ["#94a3b8", "#64748b", "#4b82c4", "#c98a37", "#258572"]
    for i, (mode, label) in enumerate(MODES.items()):
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
            color=colors[i],
            label=label,
            yerr=error,
            capsize=2,
        )
        ax.bar_label(bars, labels=[f"{x:.1f}" for x in values], fontsize=8, padding=4)
    ax.set_xticks(positions, list(MODELS.values()))
    ax.set_ylabel("Warm complete-model forward (seconds; lower is faster)")
    ax.set_title(
        "4YX2 · 594 residues · A100 80GB PCIe\nReleased reference "
        "precision versus native BF16 · five samples"
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.legend(ncols=2, loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(visible=True, alpha=0.2)
    ax.set_axisbelow(True)
    fig.supxlabel(
        "Denoiser compile/graph; 200 requested steps (ESMFold2: 134 after "
        "sigma clipping).",
        fontsize=9,
    )
    for extension in ("svg", "png"):
        fig.savefig(assets / f"benchmark_latency.{extension}", dpi=180)
    plt.close(fig)
    lines = [
        "# Inference benchmark results",
        "",
        "Updated 2026-09-16. **4YX2: 594 residues, three chains (163 + 218 + 213).**",
        "All four models now have five measured configurations. Reference means each",
        "model's released **precision policy** in FoldForge's PyTorch "
        "backend; it is not",
        "the original application's complete runtime. Native BF16 stores "
        "learned parameters",
        "in BF16 except FP32 norms and uses no autocast.",
        "",
        "![Inference latency](assets/benchmark_latency.svg)",
        "",
        "Latency is the median of **five warm complete-model forwards**, in seconds.",
        "It includes trunk, diffusion and confidence. It excludes "
        "featurization, checkpoint",
        "loading, initial compilation/capture, precomputed ESMC "
        "embeddings and CIF output.",
        "The initial forward is reported separately. Error bars are warm min/max.",
        "",
        "| Model | Reference eager | Reference compile | Native BF16 "
        "compile | cuEq BF16 compile | MiniWorld BF16 compile + graph | "
        "MiniWorld vs reference compile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in MODELS.items():
        values = [
            indexed[model, mode]["report"]["model_seconds_warm_median"]
            for mode in MODES
        ]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{v:.3f}" for v in values)
            + f" | {values[1] / values[-1]:.2f}x |"
        )
    lines += [
        "",
        "**Quality caveat:** Protenix v2 and OpenDDE native BF16 runs show more",
        "local geometry outliers than their default-precision references. The speed",
        "comparison does **not** establish structure-quality equivalence; see the",
        "per-sample checks below.",
        "",
        "## Precision and execution conditions",
        "",
        "| Model | Reference parameter storage and execution | TF32 in reference |",
        "|---|---|---|",
        "| AF3 | Released mixed parameters: BF16 trunk/confidence "
        "Pairformers; FP32 input atom encoder, diffusion, norms and final "
        "heads. No autocast. | disabled |",
        "| OpenDDE | FP32 parameters and execution; no autocast. | enabled |",
        "| ESMFold2 | FP32 parameters; BF16 autocast in input/trunk, "
        "confidence folding trunk and diffusion pair transitions. "
        "Remaining diffusion/head projections FP32. Atom FlashAttention "
        "uses BF16 Q/K/V. | enabled |",
        "| Protenix v2 | FP32 parameters; BF16 autocast with diffusion "
        "explicitly in FP32. Confidence stays in the outer BF16 autocast "
        "scope. | enabled |",
        "",
        "Reference AMP scopes follow [Protenix "
        "inference](https://github.com/bytedance/Protenix/blob/main/runner"
        "/inference.py),",
        "[ESMFold2 pinned implementation](https://github.com/Biohub/transf"
        "ormers/blob/b435f1f92dd5b4a653be57157d4f4f5ddba4f145/src/transfor"
        "mers/models/esmfold2/modeling_esmfold2.py),",
        "and [OpenDDE defaults](https://github.com/aurekaresearch/OpenDDE/"
        "blob/main/opendde/config/model_base.py).",
        "These compare backend configurations with common FoldForge "
        "adapters, not exact",
        "upstream end-to-end applications. PyTorch mode retains the "
        "architecture's permitted",
        "FlashAttention path and uses PyTorch for the other replaceable operations.",
        "",
        f"All cases use {seed_description}, an input MSA cap of {MSA_DEPTH} rows "
        f"(AF3's msa_crop_size) and up to {TEMPLATE_N} templates per chain",
        "(AF3's max_templates) for AF3, Protenix v2 and OpenDDE. ESMFold2 has no",
        "template conditioning path and runs the same MSA cap without templates.",
        "Every case requests 200 steps",
        "and five samples. Token buckets use multiples of 128: 594 -> "
        "640. OpenDDE expands",
        "internally to 1140 structural tokens -> 1152. Atom buckets are "
        "8192. Inputs and",
        "ESMC cached embeddings are identical across modes within each model.",
        "",
        "Compile and manual CUDA graphs apply to the **denoiser**; trunk time remains",
        "included. Inductor automatic graphs are disabled. AF3's 10 "
        "additional recycles",
        "give 11 trunk passes; the other adapters use 10 loops/cycles. All four models",
        "batch five diffusion samples. All four MiniWorld token-attention paths",
        "use `num_aug=5` with pair bias shared across samples. ESMFold2 atom SWA",
        "builds RoPE once per structure and expands with `num_aug=5`; "
        "its FlashAttention",
        "interface uses flattened `[augmentation * batch, atoms, heads, dim]` rows.",
        "ESMFold2 retains 134 actual steps after its sigma=256 cutoff; "
        "other models use 200.",
        "",
        "Hardware is A100 80GB PCIe on cssb3/gpu02 and gpu03. Slurm job IDs",
        "are retained in the raw results.",
        "Each mode owns one GPU, requests 16 CPUs and 96 GiB RAM, and "
        "uses OMP_NUM_THREADS=8.",
        "PyTorch 2.10.0+cu128. Fresh processes may reuse disk caches. "
        "Missing MiniWorld",
        "tuned entries can trigger initial heuristic autotuning; these "
        "are warm latencies,",
        "not a claim of fully tuned performance. Per-case logs are "
        "preserved under runs/.",
        "",
        "## Memory and execution evidence",
        "",
        "| Model | Mode | Peak allocated GiB | Warm min-max (s) | Initial "
        "forward (s) | Compiled graphs / manual replays | Job |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, label in MODELS.items():
        for mode, mode_label in MODES.items():
            row = indexed[model, mode]
            r = row["report"]
            t = r["model_seconds_warm"]
            lines.append(
                f"| {label} | {mode_label} | "
                f"{r['model_peak_allocated_bytes'] / 2**30:.2f} | "
                f"{min(t):.3f}-{max(t):.3f} | {r['model_seconds_cold']:.3f} | "
                f"{r['compiled_graphs']} / {r['cuda_graph_replays']} | "
                f"{row['slurm_job']} |"
            )
    lines += [
        "",
        "Peak allocation includes initial and repeated forwards. "
        "Execution counters come",
        "from observed compiler graphs and manual replays, not requested flags.",
        "",
        "[Raw configs, inputs, timings and execution "
        "counters](assets/benchmark_results.json).",
        "",
        "## Reproduction",
        "",
        "Run inside an allocated GPU job:",
        "",
        "```bash",
        "python scripts/benchmark_end_to_end.py --model esmfold2 --targets 4yx2 \\",
        "  --root benchmark-default --benchmark-repeats 5 --steps 200 \\",
        "  --recycles 10 --samples 5 \\",
        f"  --msa-depth {MSA_DEPTH} --template-n {TEMPLATE_N}",
        "```",
        "",
        "Use `--model opendde`, or `--model protenix --variant protenix-v2`.",
        "The first two modes select `model_default` (`af3_default` for AF3). The three",
        "comparison modes select native BF16. Whole-model FP32 remains an explicit",
        "diagnostic mode; it is not substituted for the model-default reference.",
        "",
    ]
    path = docs / "benchmark_results.md"
    previous = path.read_text() if path.exists() else ""
    marker = "## AF3: official default precision versus native BF16"
    if marker in previous:
        detail = previous[previous.index(marker) :].split(
            "## Historical four-model comparison"
        )[0]
        lines += [detail.rstrip(), ""]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    sample_path = args.results / "sample_axes_audits.json"
    if not sample_path.exists():
        message = "Model sample-axis audits are required for current measurements"
        raise ValueError(message)
    sample_audits = json.loads(sample_path.read_text())
    if set(sample_audits) != {"esmfold2", "protenix", "opendde"} or not all(
        audit.get("passed")
        and audit.get("samples") == REPEATS
        and audit.get("token_attention")
        and all(
            shape["q"][0] == REPEATS and shape["bias"][0] == 1
            for shape in audit["token_attention"]
        )
        for audit in sample_audits.values()
    ):
        message = "Invalid augmentation dispatch evidence"
        raise ValueError(message)
    rows = []
    for path in sorted(args.results.glob("e2e-*.json")):
        rows.extend(json.loads(path.read_text()))
    render(rows, args.docs)
    audit_path = args.results / "precision_audits.json"
    quality_path = args.results / "structure_checks.json"
    if audit_path.exists() and quality_path.exists():
        audits = json.loads(audit_path.read_text())
        quality = json.loads(quality_path.read_text())
        if not all(
            all(record["checks"].values())
            for model in audits.values()
            for record in model.values()
        ):
            message = "Cannot publish failed precision audits"
            raise ValueError(message)
        for source in (audit_path, quality_path):
            (args.docs / "assets" / source.name).write_text(source.read_text())

        def max_peptide(model: str, mode: str) -> float:
            return max(x["geometry"]["peptide_max_A"] for x in quality[model][mode])

        section = [
            "## Other-model precision and structural checks",
            "",
            "All six reference/native BF16 dtype audits passed. Native BF16 explicitly",
            "clears inherited FP32 Linear compute overrides, including geometry and",
            "diffusion-conditioning projections. Norm parameters remain FP32; no",
            "autocast is used in native modes. Sampler coordinates, geometry buffers",
            "and numerical reductions can still use FP32; this is not a claim that",
            "every floating-point tensor is BF16.",
            "",
            "For MiniWorld native BF16, maximum peptide C-N lengths in Protenix v2",
            f"and OpenDDE are {max_peptide('protenix', 'miniworld_graph'):.3f} A and "
            f"{max_peptide('opendde', 'miniworld_graph'):.3f} A, respectively.",
            "Their compiled default-precision references reach "
            f"{max_peptide('protenix', 'pytorch_compile_reference'):.3f} A and "
            f"{max_peptide('opendde', 'pytorch_compile_reference'):.3f} A.",
            "Inspect the outlier counts below; finite coordinates alone "
            "are not a quality pass.",
            "",
            "Superseded Protenix/OpenDDE runs that retained FP32 projection overrides",
            "are excluded from the table. Their raw artifacts remain under `.bench/`.",
            "",
            "Below: same-index samples compared with each model's compiled reference.",
            "CA RMSD uses all residues after rigid alignment; it includes numerical",
            "trajectory divergence and is not an experimental accuracy score. Geometry",
            "counts sum five samples; these do not establish accuracy equivalence.",
            "",
            "| Model | Mode | CA RMSD range (A) | Finite samples | Peptide C-N "
            "outside 1.0-1.7 A | Heavy-atom pairs <1 A |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for model, cases in quality.items():
            for mode, mode_label in MODES.items():
                samples = cases[mode]
                rmsd = [x["vs_reference"]["ca_rmsd_A"] for x in samples]
                finite = sum(x["geometry"]["finite"] for x in samples)
                bonds = sum(
                    x["geometry"]["peptide_outside_1_0_to_1_7_A"] for x in samples
                )
                clashes = sum(
                    x["geometry"]["heavy_atom_pairs_below_1_A"] for x in samples
                )
                section.append(
                    f"| {MODELS[model]} | {mode_label} | "
                    f"{min(rmsd):.3f}-{max(rmsd):.3f} | "
                    f"{finite}/5 | {bonds} | {clashes} |"
                )
        section += [
            "",
            "[Precision audits](assets/precision_audits.json) · "
            "[Per-sample structural checks](assets/structure_checks.json).",
            "",
            "",
        ]
        path = args.docs / "benchmark_results.md"
        marker = "## AF3: official default precision versus native BF16"
        text = path.read_text()
        detail = "\n".join(section)
        path.write_text(
            text.replace(marker, detail + marker) if marker in text else text + detail
        )
    sample_path = args.results / "sample_axes_audits.json"
    if sample_path.exists():
        sample_audits = json.loads(sample_path.read_text())
        if set(sample_audits) != {"esmfold2", "protenix", "opendde"} or not all(
            audit.get("passed") for audit in sample_audits.values()
        ):
            message = "Missing or failed model sample-axis audits"
            raise ValueError(message)
        (args.docs / "assets" / sample_path.name).write_text(sample_path.read_text())
        section = [
            "## Shared sample-axis verification",
            "",
            "32 GPU tests passed, including two structures with three samples and",
            "different atom masks, compared with independent sample execution.",
            "Real-checkpoint eager audits observe the kernel boundary independently",
            "of compiled timings. Trunk triangle rows are excluded from these records.",
            "",
            "| Model | Token Q shape | Core layout | Shared pair bias shape |",
            "|---|---|---|---|",
        ]
        for model, audit in sample_audits.items():
            shape = audit["token_attention"][0]
            section.append(
                f"| {MODELS[model]} | `{shape['q']}` | `{shape['layout']}` | "
                f"`{shape['bias']}` |"
            )
        section += [
            "",
            "A is augmentation, B structure batch, H heads, L tokens and D head width.",
            "ESMFold2 already used augmentation-aware token attention; its atom path",
            "now builds static conditioning/RoPE per structure and expands through",
            "the SWA API with num_aug=5. Protenix/OpenDDE square token attention now",
            "preserves the sample axis instead of flattening it into structure batch.",
            "Dispatch also checks projected Q/K/V dtypes: FP32 residuals previously",
            "bypassed the engine even when native projections produced BF16 Q/K/V.",
            "Regression tests cover FP32 residuals with BF16 projection parameters.",
            "Their rectangular atom windows retain batched shared PyTorch attention:",
            "the engine pair-biased augmentation core supports square attention only.",
            "",
            "All five ESMFold2 modes and the MiniWorld modes for Protenix/OpenDDE were",
            "remeasured. Unaffected measurements were retained. Graph replay counts",
            "are 804 for ESMFold2 and 1200 for Protenix/OpenDDE: six complete forwards",
            "times 134 or 200 denoising steps, each handling five samples together.",
            "",
            "[Observed kernel shapes](assets/sample_axes_audits.json) · "
            "[Measured source hashes](assets/sample_axes_source_manifest.json).",
            "",
            "",
        ]
        path = args.docs / "benchmark_results.md"
        text = path.read_text()
        marker = "## AF3: official default precision versus native BF16"
        detail = "\n".join(section)
        path.write_text(
            text.replace(marker, detail + marker) if marker in text else text + detail
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
