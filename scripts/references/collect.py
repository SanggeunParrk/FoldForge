"""Normalise one release run into outputs/<target>/<family>/seed<k>/.

Usage: collect.py FAMILY TARGET SEED RAW_DIR

Every release writes its own layout; this finds the five sample CIFs, writes
them as sample<i>.cif.gz in sample order, and records each sample's mean pLDDT
(from the CIF B-factors, or the confidence JSON where a release's B-factor is
not its pLDDT) in the seed's manifest.json.
"""

from __future__ import annotations

import gzip
import json
import re
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {
    "boltz2": "**/predictions/*/*_model_*.cif",
    "protenix1": "**/predictions/*_sample_*.cif",
    "protenix2": "**/predictions/*_sample_*.cif",
    "openfold3": "**/*_model.cif",
    "openfold3-preview2": "**/*_model.cif",
    "rosettafold3": "**/seed-*_sample-*/*_model.cif",
    "intellifold2": "**/predictions/*/*_sample-*.cif",
    "chai1": "pred.model_idx_*.cif",
    "opendde": "**/predictions/*_sample_*.cif",
    "esmfold2": "sample_*.cif",
}


def sample_index(path: Path) -> int:
    match = re.search(r"(?:sample[-_]|model_|idx_)(\d+)", path.name)
    return int(match.group(1)) if match else -1


def bfactor_plddt(path: Path) -> float:
    cols, values = [], []
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
        elif cols and line.startswith(("ATOM", "HETATM")):
            row = shlex.split(line)
            index = cols.index("B_iso_or_equiv")
            if len(row) > index:  # Chai-lab writes short rows for ions
                values.append(float(row[index]))
    b = np.asarray(values)
    return float(b.mean() * (100 if b.max() <= 1 else 1))


def json_plddt(path: Path) -> float | None:
    """IntelliFold's B-factors are not its pLDDT; its confidence JSON is."""
    found = sorted(path.parent.glob(path.stem + "_confidences.json"))
    if not found:
        return None
    return float(np.mean(json.loads(found[0].read_text())["atom_plddts"]) * 100)


def main() -> None:
    family, target = sys.argv[1], sys.argv[2]
    seed, raw = int(sys.argv[3]), Path(sys.argv[4])
    cifs = sorted(raw.glob(PATTERNS[family]), key=sample_index)
    if family.startswith("openfold3"):
        cifs = [p for p in cifs if "sample" in p.name]
    if len(cifs) != 5:  # noqa: PLR2004 - every reference run draws five samples
        message = f"{family} {target} seed {seed}: {len(cifs)} samples in {raw}"
        raise SystemExit(message)
    dest = ROOT / "outputs" / target / family / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, cif in enumerate(cifs):
        with gzip.open(dest / f"sample{i}.cif.gz", "wb", compresslevel=9) as fh:
            fh.write(cif.read_bytes())
        plddt = json_plddt(cif) if family == "intellifold2" else None
        if plddt is None:
            plddt = bfactor_plddt(cif)
        samples.append(
            {
                "file": f"sample{i}.cif.gz",
                "source": cif.name,
                "mean_plddt": round(plddt, 3),
            }
        )
    manifest = {"family": family, "target": target, "seed": seed, "samples": samples}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(  # noqa: T201 - the job log records each seed's pLDDTs
        f"{family} {target} seed {seed}: {[s['mean_plddt'] for s in samples]}"
    )


if __name__ == "__main__":
    main()
