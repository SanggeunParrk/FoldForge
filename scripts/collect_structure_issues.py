"""Gather benchmark samples with geometry or drift problems beside their references.

Reads a benchmark results directory (``e2e-<model>.json`` rows whose reports list
``prediction_cifs``), checks every sample of every mode with the same peptide,
clash and CA-drift definitions as ``compare_af3_precision.py``, and copies only
the flagged samples into ``<output>/<model>/<mode>/sample-<i>.cif`` with the
same-index compiled reference at ``<output>/<model>/reference/sample-<i>.cif``.
``<output>/README.md`` lists what is wrong in each flagged sample, with residues.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from compare_af3_precision import PEPTIDE_MAX, atoms, compare, geometry
from scipy.spatial.distance import cdist

REFERENCE = "pytorch_compile_reference"
MAX_LISTED = 6


def locate(path: str, results: Path) -> Path:
    """Resolve a report CIF path, falling back to a copy under ``results/e2e``."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    local = results / "e2e" / candidate.parent.name / candidate.name
    if local.exists():
        return local
    message = f"Prediction not found here or under {results / 'e2e'}: {path}"
    raise FileNotFoundError(message)


def peptide_breaks(a: dict) -> list[dict]:
    by_res = {(k[0], k[1], k[3]): v[0] for k, v in a.items()}
    breaks = []
    for (chain, seq, atom), value in by_res.items():
        if atom == "C" and (chain, seq + 1, "N") in by_res:
            distance = float(np.linalg.norm(value - by_res[chain, seq + 1, "N"]))
            if not 1.0 <= distance <= PEPTIDE_MAX:
                breaks.append(
                    {"chain": chain, "residues": [seq, seq + 1], "c_n_A": distance}
                )
    return sorted(breaks, key=lambda b: (b["chain"], b["residues"][0]))


def clashes(a: dict) -> list[dict]:
    keys = sorted(a)
    xyz = np.array([a[k][0] for k in keys])
    rows, cols = np.nonzero(np.triu(cdist(xyz, xyz) < 1.0, 1))
    return [
        {
            "a": f"{keys[i][0]}:{keys[i][2]}{keys[i][1]}:{keys[i][3]}",
            "b": f"{keys[j][0]}:{keys[j][2]}{keys[j][1]}:{keys[j][3]}",
            "distance_A": float(np.linalg.norm(xyz[i] - xyz[j])),
        }
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True)
    ]


def describe(sample: dict, reference: dict | None) -> dict:
    record = {
        "geometry": geometry(sample),
        "peptide_breaks": peptide_breaks(sample),
        "clashes": clashes(sample),
    }
    if reference is not None:
        record["vs_reference"] = compare(sample, reference)
    return record


def flagged(record: dict, rmsd_threshold: float) -> list[str]:
    reasons = []
    if record["peptide_breaks"]:
        reasons.append(f"{len(record['peptide_breaks'])} peptide C-N outside 1.0-1.7 A")
    if record["clashes"]:
        reasons.append(f"{len(record['clashes'])} heavy-atom pairs < 1 A")
    drift = record.get("vs_reference", {}).get("ca_rmsd_A")
    if drift is not None and drift > rmsd_threshold:
        reasons.append(f"CA RMSD {drift:.2f} A from the reference sample")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rmsd-threshold", type=float, default=1.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = ["# Benchmark samples with geometry or drift problems", ""]
    index.append(
        f"Source: `{args.results}`. Flags: any peptide C-N outside 1.0-1.7 A, any "
        f"heavy-atom pair below 1 A, or CA RMSD above {args.rmsd_threshold} A from the "
        "same-index compiled default-precision reference. Reference samples are "
        "listed too when they carry a defect of their own."
    )
    issues: dict[str, dict] = {}
    for path in sorted(args.results.glob("e2e-*.json")):
        model = path.stem.removeprefix("e2e-")
        rows = {r["mode"]: r for r in json.loads(path.read_text()) if r["status"] == 0}
        if REFERENCE not in rows:
            continue
        ref_paths = [
            locate(p, args.results)
            for p in rows[REFERENCE]["report"]["prediction_cifs"]
        ]
        refs = [atoms(p) for p in ref_paths]
        issues[model] = {}
        index += ["", f"## {model}", ""]
        ref_records = [describe(r, None) for r in refs]
        ref_dir = args.output / model / "reference"
        for i, (record, source) in enumerate(zip(ref_records, ref_paths, strict=True)):
            reasons = flagged(record, args.rmsd_threshold)
            if reasons:
                ref_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, ref_dir / f"sample-{i}.cif")
                issues[model].setdefault(REFERENCE, {})[str(i)] = record
                index.append(
                    f"- reference sample {i} itself: {'; '.join(reasons)} -> "
                    f"`{model}/reference/sample-{i}.cif`"
                )
        index += [
            "",
            "| mode | sample | problem | where | files |",
            "|---|---:|---|---|---|",
        ]
        for mode, row in rows.items():
            if mode == REFERENCE:
                continue
            sources = [
                locate(p, args.results) for p in row["report"]["prediction_cifs"]
            ]
            for i, source in enumerate(sources):
                record = describe(atoms(source), refs[i])
                reasons = flagged(record, args.rmsd_threshold)
                if not reasons:
                    continue
                issues[model].setdefault(mode, {})[str(i)] = record
                mode_dir = args.output / model / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                ref_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, mode_dir / f"sample-{i}.cif")
                if not (ref_dir / f"sample-{i}.cif").exists():
                    shutil.copy2(ref_paths[i], ref_dir / f"sample-{i}.cif")
                where = []
                where += [
                    f"{b['chain']}{b['residues'][0]}-{b['residues'][1]} "
                    f"C-N {b['c_n_A']:.2f} A"
                    for b in record["peptide_breaks"][:MAX_LISTED]
                ]
                where += [
                    f"{c['a']} / {c['b']} {c['distance_A']:.2f} A"
                    for c in record["clashes"][:MAX_LISTED]
                ]
                extra = (
                    len(record["peptide_breaks"]) + len(record["clashes"]) - len(where)
                )
                if extra > 0:
                    where.append(f"and {extra} more in issues.json")
                files = (
                    f"`{model}/{mode}/sample-{i}.cif` vs "
                    f"`{model}/reference/sample-{i}.cif`"
                )
                index.append(
                    f"| {mode} | {i} | {'; '.join(reasons)} | "
                    f"{'<br>'.join(where) or '-'} | {files} |"
                )
    (args.output / "issues.json").write_text(json.dumps(issues, indent=2) + "\n")
    (args.output / "README.md").write_text("\n".join(index) + "\n")
    print(args.output / "README.md")  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
