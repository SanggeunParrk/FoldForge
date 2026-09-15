"""Compare inference I/O artifacts, including confidence heads and CIF atom order.

Run on a compute node with the FoldForge environment activated. Both directories
contain one subdirectory per model, produced with identical input and settings.
Timing/report metadata is deliberately excluded from the numerical comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def compare(reference: Path, actual: Path) -> dict:
    stats = []

    def check(
        left: dict[str, np.ndarray], right: dict[str, np.ndarray], label: str
    ) -> None:
        if isinstance(left, (torch.Tensor, np.ndarray)):
            if not isinstance(right, (torch.Tensor, np.ndarray)):
                msg = f"Changed tensor type: {label}"
                raise TypeError(msg)
            a, b = torch.as_tensor(left), torch.as_tensor(right)
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True, msg=label)
            stats.append({"path": label, "shape": list(a.shape), "exact": True})
        elif isinstance(left, dict):
            assert isinstance(right, dict), label
            assert left.keys() == right.keys(), label
            for key, value in left.items():
                check(value, right[key], f"{label}.{key}")
        elif isinstance(left, (tuple, list)):
            assert isinstance(right, (tuple, list)), label
            assert len(left) == len(right), label
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                check(a, b, f"{label}[{index}]")
        else:
            assert left == right, f"Changed scalar or absent head: {label}"

    predictions = sorted(reference.glob("*.prediction.pt"))
    assert predictions, f"No reference predictions in {reference}"
    for pattern in ("*.prediction.pt", "features-*.pt", "*.npz", "*.cif"):
        old_files = {p.name for p in reference.glob(pattern)}
        new_files = {p.name for p in actual.glob(pattern)}
        assert old_files == new_files, (pattern, old_files, new_files)
    for path in predictions + sorted(reference.glob("features-*.pt")):
        check(
            torch.load(path, map_location="cpu", weights_only=False),
            torch.load(actual / path.name, map_location="cpu", weights_only=False),
            path.name,
        )
    for path in sorted(reference.glob("*.npz")):
        with np.load(path) as left, np.load(actual / path.name) as right:
            assert set(left.files) == set(right.files), path.name
            for key in left.files:
                check(left[key], right[key], f"{path.name}.{key}")
    cifs = []
    for path in sorted(reference.glob("*.cif")):
        left, right = MMCIF2Dict(str(path)), MMCIF2Dict(str(actual / path.name))
        atom_keys = {key for key in left if key.startswith("_atom_site.")}
        assert atom_keys, path.name
        assert atom_keys == {key for key in right if key.startswith("_atom_site.")}, (
            path.name
        )
        for key in atom_keys:
            assert left[key] == right[key], (path.name, key)
        cifs.append(
            {"file": path.name, "atom_site_fields": len(atom_keys), "exact": True}
        )
    assert cifs, f"No reference structures in {reference}"
    return {
        "reference": str(reference),
        "actual": str(actual),
        "tensor_count": len(stats),
        "tensors": stats,
        "cifs": cifs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["af3", "esmfold2", "protenix", "opendde"]
    )
    args = parser.parse_args()
    results = {
        model: compare(args.reference / model, args.actual / model)
        for model in args.models
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n")
    print(  # noqa: T201 - CLI output contract
        json.dumps(
            {
                model: {
                    "tensors": result["tensor_count"],
                    "cifs": len(result["cifs"]),
                    "exact": True,
                }
                for model, result in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
