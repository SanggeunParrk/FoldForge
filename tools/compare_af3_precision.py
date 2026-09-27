"""Compare all AF3 precision samples by atom identity and experimental CA sites.

This evaluates numerical drift, local geometry and agreement with one experimental
complex. It does not establish dataset-level BF16 accuracy equivalence.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.spatial.distance import cdist

NUM_SAMPLES = 5
LDDT_CUTOFF = 15
CONTACT_CUTOFF = 8
CA_MIN, CA_MAX = 3.4, 4.2
PEPTIDE_MAX = 1.7


def atoms(path: Path):
    data = MMCIF2Dict(str(path))
    result = {}
    count = len(data["_atom_site.label_atom_id"])
    for i in range(count):
        if data.get("_atom_site.pdbx_PDB_model_num", ["1"] * count)[i] != "1":
            continue
        seq = data["_atom_site.label_seq_id"][i]
        if seq in (".", "?") or data["_atom_site.type_symbol"][i] in ("H", "D"):
            continue
        key = (
            data["_atom_site.label_asym_id"][i],
            int(seq),
            data["_atom_site.label_comp_id"][i],
            data["_atom_site.label_atom_id"][i],
        )
        occupancy = float(data.get("_atom_site.occupancy", ["1"] * count)[i])
        value = (
            np.array([float(data[f"_atom_site.Cartn_{axis}"][i]) for axis in "xyz"]),
            float(data.get("_atom_site.B_iso_or_equiv", ["0"] * count)[i]),
            occupancy,
        )
        if key not in result or occupancy > result[key][2]:
            result[key] = value
    return result


def aligned_rmsd(x: np.ndarray, y: np.ndarray):
    xx, yy = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xx.T @ yy)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return float(np.sqrt(np.mean(np.sum((xx @ rotation - yy) ** 2, axis=-1))))


def compare(a: dict, b: dict):
    keys = sorted(k for k in a.keys() & b.keys() if k[-1] == "CA")
    if not keys:
        msg = "No matched CA atoms"
        raise ValueError(msg)
    x, y = np.array([a[k][0] for k in keys]), np.array([b[k][0] for k in keys])
    dx, dy = cdist(x, x), cdist(y, y)
    neighbors = (dy < LDDT_CUTOFF) & (dy > 0)
    diff = np.abs(dx - dy)
    score = sum((diff < t).astype(float) for t in (0.5, 1.0, 2.0, 4.0)) / 4
    per_atom = np.sum(score * neighbors, axis=1) / np.maximum(1, neighbors.sum(1))
    chains = np.array([k[0] for k in keys])
    inter = chains[:, None] != chains[None, :]
    target_contacts = inter & (dy < CONTACT_CUTOFF)
    pred_contacts = inter & (dx < CONTACT_CUTOFF)
    tp = np.count_nonzero(target_contacts & pred_contacts)
    per_chain = {}
    for chain in sorted(set(chains)):
        mask = chains == chain
        per_chain[chain] = {
            "ca_count": int(mask.sum()),
            "rmsd_A": aligned_rmsd(x[mask], y[mask]),
        }
    return {
        "ca_count": len(keys),
        "ca_rmsd_A": aligned_rmsd(x, y),
        "ca_lddt": float(per_atom.mean()),
        "per_chain": per_chain,
        "interface_contact_precision": tp / max(1, int(pred_contacts.sum())),
        "interface_contact_recall": tp / max(1, int(target_contacts.sum())),
    }


def geometry(a: dict):
    ca = []
    peptide = []
    by_res = {(k[0], k[1], k[3]): v[0] for k, v in a.items()}
    for (chain, seq, atom), value in by_res.items():
        if atom == "CA" and (chain, seq + 1, "CA") in by_res:
            ca.append(np.linalg.norm(value - by_res[chain, seq + 1, "CA"]))
        if atom == "C" and (chain, seq + 1, "N") in by_res:
            peptide.append(np.linalg.norm(value - by_res[chain, seq + 1, "N"]))
    keys = sorted(a)
    xyz = np.array([a[k][0] for k in keys])
    finite = bool(np.isfinite(xyz).all())
    distances = cdist(xyz, xyz)
    clash = np.triu(distances < 1.0, 1)
    ca = np.array(ca)
    peptide = np.array(peptide)
    return {
        "atoms": len(a),
        "finite": finite,
        "ca_mean_plddt": float(np.mean([v[1] for k, v in a.items() if k[-1] == "CA"])),
        "adjacent_ca_count": len(ca),
        "adjacent_ca_outside_3_4_to_4_2_A": int(((ca < CA_MIN) | (ca > CA_MAX)).sum()),
        "adjacent_ca_max_A": float(ca.max()),
        "peptide_bonds": len(peptide),
        "peptide_outside_1_0_to_1_7_A": int(
            ((peptide < 1.0) | (peptide > PEPTIDE_MAX)).sum()
        ),
        "peptide_max_A": float(peptide.max()),
        "heavy_atom_pairs_below_1_A": int(clash.sum()),
    }


def write_summary(result: dict, report_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "pytorch_eager_reference": "PyTorch eager AF3 default",
        "pytorch_compile_reference": "PyTorch compile AF3 default",
        "pytorch_compile_bf16": "PyTorch compile native BF16",
        "cuequiv_compile": "cuEq compile native BF16",
        "miniworld_graph": "MiniWorld compile + graph native BF16",
        "pytorch_compile_bf16_previous": "Previous BF16 weights / FP32 DiT residual",
    }
    modes = [
        m
        for m in labels
        if m in result["models"] and m != "pytorch_compile_bf16_previous"
    ]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    values = [result["models"][m]["latency_s"] for m in modes]
    bars = ax.barh(
        [labels[m] for m in modes],
        values,
        xerr=np.array(
            [
                [
                    result["models"][m]["latency_s"]
                    - min(result["models"][m]["warm_seconds"])
                    for m in modes
                ],
                [
                    max(result["models"][m]["warm_seconds"])
                    - result["models"][m]["latency_s"]
                    for m in modes
                ],
            ]
        ),
        capsize=3,
        color=["#94a3b8", "#64748b", "#4b82c4", "#c98a37", "#258572", "#baa5c4"][
            : len(modes)
        ],
    )
    ax.bar_label(bars, labels=[f"{x:.2f} s" for x in values], padding=4)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.17)
    ax.set_xlabel("Median warm complete-model forward (s; lower is faster)")
    ax.set_title(
        "AF3 / 4YX2 / 594 residues / A100 80GB\n5 samples / 200 steps / 10 "
        "recycles; denoiser compile/graph"
    )
    ax.spines[["top", "right"]].set_visible(False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(report_path.with_suffix("." + extension), dpi=180)
    plt.close(fig)
    rows = [
        "# AF3 native BF16 precision validation",
        "",
        "![Latency](" + report_path.with_suffix(".svg").name + ")",
        "",
        "Same 4YX2 input, checkpoint, seed 0, 10 recycles (11 trunk passes), "
        "200 diffusion "
        "steps and five batched samples per forward (200 denoiser calls). "
        "Median of five "
        "warm complete-model forwards; first forward excluded. Compile and "
        "manual CUDA graphs cover the denoiser. FP32 operations use highest "
        "matmul "
        "precision (TF32 disabled). No autocast in any mode.",
        "",
        "| Mode | Latency (s) | Peak GiB | Mean CA pLDDT | Experimental CA "
        "lDDT (%) | Experimental CA RMSD (A) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        r = result["models"][mode]
        samples = r["samples"]
        plddt = np.mean([x["geometry"]["ca_mean_plddt"] for x in samples])
        lddt = 100 * np.mean([x["vs_experimental"]["ca_lddt"] for x in samples])
        rmsd = np.mean([x["vs_experimental"]["ca_rmsd_A"] for x in samples])
        rows.append(
            f"| {labels[mode]} | {r['latency_s']:.3f} | {r['peak_gib']:.2f} | "
            f"{plddt:.2f} | {lddt:.2f} | {rmsd:.3f} |"
        )
    rows += [
        "",
        "Quality columns average all five samples; experimental comparison "
        "uses 528 observed CA atoms. Numerical comparisons below match the "
        "same sample index against PyTorch compile AF3 default "
        "using all 594 "
        "predicted CA atoms.",
        "",
        "| Mode | Reference CA RMSD, all 594 (A) | Observed 528 (A) | "
        "Peptide C-N outliers "
        "(sum / 5 samples) | Heavy-atom pairs <1 A (sum / 5) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in modes:
        samples = result["models"][mode]["samples"]
        rms = [x["vs_reference"]["ca_rmsd_A"] for x in samples]
        observed = [x["vs_reference_observed"]["ca_rmsd_A"] for x in samples]
        bonds = sum(x["geometry"]["peptide_outside_1_0_to_1_7_A"] for x in samples)
        clashes = sum(x["geometry"]["heavy_atom_pairs_below_1_A"] for x in samples)
        rows.append(
            f"| {labels[mode]} | {min(rms):.3f}-{max(rms):.3f} | "
            f"{min(observed):.3f}-{max(observed):.3f} | {bonds} | {clashes} |"
        )
    rows += [
        "",
        "Peptide outliers use C-N outside 1.0-1.7 A. The <1 A heavy-atom "
        "count is a gross-overlap diagnostic, not a full stereochemical "
        "clashscore. lDDT here uses CA distances within 15 A and "
        "thresholds 0.5/1/2/4 A. Missing experimental residues are "
        "excluded. Per-chain RMSD, interface contacts and within-mode "
        "sample diversity are included in the JSON.",
        "",
        "This single-complex experiment does not establish dataset-level "
        "accuracy equivalence. Same-seed diffusion trajectories can "
        "diverge after precision changes.",
        "",
    ]
    report_path.write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--experimental", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = []
    for p in sorted(args.results.rglob("e2e-af3.json")):
        rows.extend(json.loads(p.read_text()))
    if not rows or any(r["status"] for r in rows):
        msg = "Missing or failed benchmark"
        raise ValueError(msg)
    indexed = {r["mode"]: r for r in rows}
    if len(indexed) != len(rows):
        msg = "Duplicate modes"
        raise ValueError(msg)
    reference = indexed["pytorch_compile_reference"]
    predictions = {
        m: [atoms(Path(p)) for p in r["report"]["prediction_cifs"]]
        for m, r in indexed.items()
    }
    experiment = atoms(args.experimental)
    result = {
        "reference_mode": "pytorch_compile_reference",
        "experimental": str(args.experimental),
        "models": {},
    }
    for mode, row in indexed.items():
        if row["input_spec"] != reference["input_spec"]:
            msg = "Different input specification"
            raise ValueError(msg)
        if row["report"].get("samples_per_denoiser_call") != NUM_SAMPLES:
            msg = "Expected five batched diffusion samples"
            raise ValueError(msg)
        samples = predictions[mode]
        refs = predictions["pytorch_compile_reference"]
        if len(samples) != len(refs):
            msg = "Different sample count"
            raise ValueError(msg)
        metrics = []
        for i, (sample, ref) in enumerate(zip(samples, refs, strict=True)):
            if set(sample) != set(ref):
                msg = "Different atom identities"
                raise ValueError(msg)
            metrics.append(
                {
                    "sample": i,
                    "geometry": geometry(sample),
                    "vs_reference": compare(sample, ref),
                    "vs_reference_observed": compare(
                        {k: v for k, v in sample.items() if k in experiment},
                        {k: v for k, v in ref.items() if k in experiment},
                    ),
                    "vs_experimental": compare(sample, experiment),
                }
            )
        result["models"][mode] = {
            "precision": row["config"]["precision"],
            "backend": row["backend"],
            "latency_s": row["report"]["model_seconds_warm_median"],
            "warm_seconds": row["report"]["model_seconds_warm"],
            "peak_gib": row["report"]["model_peak_allocated_bytes"] / 2**30,
            "compiled_graphs": row["report"]["compiled_graphs"],
            "samples_per_denoiser_call": row["report"]["samples_per_denoiser_call"],
            "samples": metrics,
            "within_mode_ca_rmsd_A": [
                compare(samples[a], samples[b])["ca_rmsd_A"]
                for a, b in itertools.combinations(range(len(samples)), 2)
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.report is not None:
        write_summary(result, args.report)


if __name__ == "__main__":
    main()
