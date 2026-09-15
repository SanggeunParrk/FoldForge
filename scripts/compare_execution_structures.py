"""Compare matched single-target, single-sample graph/compile predictions.

Limits: 0.25 A aligned protein CA RMSD and one token-pLDDT point (0-100).
Confidence is read from the common Prediction payload, not optional CIF B factors.
This measures numerical regression, not biological accuracy or bitwise parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

MAX_RMSD_ANGSTROM = 0.25


def ca_atoms(path: Path) -> dict:
    data = MMCIF2Dict(str(path))
    out = {}
    for i, atom in enumerate(data["_atom_site.label_atom_id"]):
        if atom != "CA" or data["_atom_site.type_symbol"][i] != "C":
            continue
        key = (
            data.get("_atom_site.auth_asym_id", data["_atom_site.label_asym_id"])[i],
            data.get("_atom_site.auth_seq_id", data["_atom_site.label_seq_id"])[i],
            data["_atom_site.label_comp_id"][i],
        )
        out[key] = [float(data[f"_atom_site.Cartn_{axis}"][i]) for axis in "xyz"]
    return out


def one_file(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        message = f"Expected one {pattern} in {root}, found {len(paths)}"
        raise ValueError(message)
    return paths[0]


def mean_plddt(root: Path) -> float:
    payload = torch.load(
        one_file(root, "*.prediction.pt"), map_location="cpu", weights_only=True
    )
    scores = payload["plddt"]
    if scores is None or scores.ndim != 2 or scores.shape[0] != 1:  # noqa: PLR2004 - tensor rank or format cardinality
        message = "Comparison requires one sample with token pLDDT"
        raise ValueError(message)
    if not torch.isfinite(scores).all():
        message = "Prediction contains non-finite pLDDT"
        raise ValueError(message)
    return float(scores.float().mean()) * 100.0


def compare(root: Path) -> list[dict]:
    rows = []
    for model in sorted(root.iterdir()):
        if not model.is_dir():
            continue
        graph, compiled = model / "graph", model / "compile"
        if not graph.exists() and not compiled.exists():
            continue
        a, b = ca_atoms(one_file(graph, "*.cif")), ca_atoms(one_file(compiled, "*.cif"))
        if not a or set(a) != set(b):
            message = f"Protein CA identities differ or are absent for {model.name}"
            raise ValueError(message)
        keys = sorted(a)
        x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
        xx, yy = x - x.mean(0), y - y.mean(0)
        u, _, vt = np.linalg.svd(xx.T @ yy)
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        rmsd = float(np.sqrt(np.mean(np.sum((xx @ rotation - yy) ** 2, axis=-1))))
        pa, pb = mean_plddt(graph), mean_plddt(compiled)
        rows.append(
            {
                "model": model.name,
                "ca_count": len(keys),
                "ca_aligned_rmsd_A": rmsd,
                "token_plddt_graph": pa,
                "token_plddt_compile": pb,
                "token_plddt_delta": abs(pa - pb),
                "pass": bool(rmsd <= MAX_RMSD_ANGSTROM and abs(pa - pb) <= 1.0),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Directory containing MODEL/graph and MODEL/compile"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows, indent=2) + "\n"
    args.output.write_text(text)
    print(text, end="")  # noqa: T201 - diagnostic report
    return 0 if rows and all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
