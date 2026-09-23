"""Is every input stream a model reads actually connected?

A folding model with a DEAD input still folds. ESMFold2 ran for months with no
language model: a selector compared a molecule-type class against a boolean and
kept exactly the tokens it meant to drop, so the tower saw an empty sequence and
its shim turned the resulting zeros into one vector repeated at every pair
position. The fold looked ordinary -- clean bonds, no clashes, plausible pLDDT --
because a second input, the MSA encoder, carried the structure by itself.

Two checks catch that class of defect, and neither needs a reference:

1. **Variation.** Count the DISTINCT rows of the tensor. One distinct row means
   the stream carries no information at all. Padding should differ from real
   tokens; if the padded region has the same statistics as the real one, the
   stream is not reading the input.

2. **Information.** For a pair stream, ask whether a linear read-out of it ranks
   true long-range contacts above chance. A stream that varies but ranks at the
   baseline is connected to something other than the structure.

Magnitude proves nothing: the dead constant measured 45% of the injection by
RMS. Ablation is the other half of the answer and belongs in a fold, not here --
an input whose removal changes nothing is not connected, whatever it measures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def distinct_rows(tensor: np.ndarray) -> tuple[int, int]:
    """Return (distinct, total) over the last axis' rows."""
    flat = tensor.reshape(-1, tensor.shape[-1])
    return len(np.unique(flat, axis=0)), int(flat.shape[0])


def ca_positions(path: Path) -> dict[int, list[float]]:
    """CA coordinates by residue number, from an mmCIF the writer produced."""
    rows: dict[int, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        field = line.split()
        if field[3] != "CA":
            continue
        try:
            number = int(field[8])
        except ValueError:
            continue
        rows.setdefault(
            number, [float(field[10]), float(field[11]), float(field[12])]
        )
    return rows


#: A CA-CA pair this close is a contact; the usual structural-biology cut.
CONTACT_ANGSTROM = 8.0


def contact_readout(pair: np.ndarray, reference: Path, separation: int = 6) -> str:
    """How many of the 100 best-ranked long-range pairs are real contacts."""
    truth = ca_positions(reference)
    order = sorted(truth)
    coords = np.array([truth[i] for i in order])
    length = len(order)
    pair = pair[:length, :length]
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    index = np.arange(length)
    far = np.abs(index[:, None] - index[None]) >= separation
    label = ((distance < CONTACT_ANGSTROM) & far)[far].astype(float)
    design = np.c_[pair[far], np.ones(int(far.sum()))]
    weights, *_ = np.linalg.lstsq(design, label, rcond=None)
    ranked = label[np.argsort(-(design @ weights))]
    return (
        f"top-100 holds {int(ranked[:100].sum()):3d} true contacts "
        f"(baseline {100 * label.mean():.1f}%)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensors", type=Path, nargs="+", help=".npy dumps to check")
    parser.add_argument(
        "--reference", type=Path, help="deposited mmCIF, to score a pair stream"
    )
    parser.add_argument("--real-tokens", type=int, help="trim padding to this many")
    args = parser.parse_args()
    for path in args.tensors:
        tensor = np.load(path)
        distinct, total = distinct_rows(tensor)
        verdict = "DEAD -- one value everywhere" if distinct == 1 else "varies"
        shape = f"{tensor.shape}"
        rows = f"{distinct}/{total}"

        print(f"{path.name:28} {shape:22} {rows:>14} rows  {verdict}")  # noqa: T201
        if args.reference and tensor.ndim == 3 and tensor.shape[0] == tensor.shape[1]:  # noqa: PLR2004 - a square pair stream
            print(f"{'':28} {contact_readout(tensor, args.reference)}")  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    sys.exit(main())
