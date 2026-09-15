"""Score a prediction against a deposited structure: CA RMSD and TM-score.

pLDDT is the model's opinion of itself. This is the measurement that can
contradict it, so it is the one that decides whether a port is correct.

Two things here are easy to get wrong and are handled explicitly:

* **Matching.** Residues are paired on ``(chain, residue number)`` after an
  explicit chain map, and the residue *names* are checked. A crystal structure
  is missing every disordered residue, so pairing by order or by count silently
  produces a plausible number from a misaligned pair list. A name mismatch is
  reported, never averaged over.
* **Parsing.** An mmCIF has many loops, and a hand-rolled reader that keys on
  ``label_atom_id == "CA"`` will happily read ``chem_comp_atom`` rows with the
  ``atom_site`` column map. Biopython's parser knows where the loops end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

if TYPE_CHECKING:
    from pathlib import Path

#: TM-score's normalisation constant floor, from Zhang & Skolnick (2004).
_D0_FLOOR = 0.5
_MIN_TM_LENGTH = 15
#: Smallest residue set a Kabsch superposition is determined by.
_MIN_FIT = 4


def deposit_ca(path: Path) -> dict[tuple[str, int], tuple[str, tuple[float, ...]]]:
    """``(chain, residue number) -> (residue name, xyz)`` for every CA in a deposit.

    First model, first altloc only: later models and alternate conformations are
    duplicates for this purpose, and averaging them would move atoms to places
    the crystallographer never saw.
    """
    # Model-produced CIFs can omit occupancy/B factors. Read the atom-site
    # table with Biopython's CIF tokenizer instead of requiring a full Structure.
    data = MMCIF2Dict(str(path))
    atoms = data["_atom_site.label_atom_id"]
    count = len(atoms)
    chains = data.get("_atom_site.auth_asym_id", data.get("_atom_site.label_asym_id"))
    if chains is None:
        message = "The atom_site category is missing chain identifiers"
        raise ValueError(message)
    labels = data.get("_atom_site.label_seq_id", ["?"] * count)
    numbers = data.get("_atom_site.auth_seq_id", labels)
    names = data["_atom_site.label_comp_id"]
    elements = data.get("_atom_site.type_symbol", ["C"] * count)
    models = data.get("_atom_site.pdbx_PDB_model_num", ["1"] * count)
    out: dict[tuple[str, int], tuple[str, tuple[float, ...]]] = {}
    for index, atom in enumerate(atoms):
        if atom != "CA" or elements[index].upper() != "C" or models[index] != models[0]:
            continue
        number = numbers[index] if numbers[index] not in {".", "?"} else labels[index]
        if number in {".", "?"}:
            continue
        xyz = tuple(float(data[f"_atom_site.Cartn_{axis}"][index]) for axis in "xyz")
        out.setdefault((chains[index], int(number)), (names[index].strip(), xyz))

    return out


def kabsch_rmsd(mobile: torch.Tensor, target: torch.Tensor) -> float:
    """RMSD after optimal superposition."""
    rotation, offset = _kabsch(mobile, target)
    aligned = mobile @ rotation.T + offset
    return float(((aligned - target) ** 2).sum(-1).mean().sqrt())


def _kabsch(mobile: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Rotation and translation putting ``mobile`` onto ``target``."""
    mobile_centre = mobile.mean(0, keepdim=True)
    target_centre = target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd((target - target_centre).T @ (mobile - mobile_centre))
    sign = torch.sign(torch.linalg.det(u @ vh))
    flip = torch.diag(torch.tensor([1.0, 1.0, sign], dtype=mobile.dtype))
    rotation = u @ flip @ vh
    return rotation, target_centre - mobile_centre @ rotation.T


def tm_score(mobile: torch.Tensor, target: torch.Tensor) -> float | None:
    """TM-score of ``mobile`` against ``target``, normalised by the target length.

    RMSD is dominated by the worst-placed residues, so a prediction with a right
    core and one flung loop scores badly and reads as wrong. TM-score weights
    each pair by ``1 / (1 + (d/d0)^2)``, which saturates for far pairs — above
    0.5 the two structures share a fold, and that is the question being asked.

    This is the Zhang-Skolnick search, not a single Kabsch fit: the score is a
    maximum over superpositions, so it is seeded from fragments of decreasing
    length and each seed is refined by re-fitting on the residues that currently
    fall inside a distance cutoff. Reporting a one-shot all-atom fit instead
    would systematically *understate* the score.
    """
    length = target.shape[0]
    if length < _MIN_TM_LENGTH:
        # None, not an exception: too few matched residues is a fact about the
        # pairing that the caller needs to report alongside the counts, not a
        # crash that throws away the rest of the run.
        return None
    d0 = max(1.24 * (length - 15) ** (1 / 3) - 1.8, _D0_FLOOR)

    def distances(rotation: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Compute distances."""
        return ((mobile @ rotation.T + offset) - target).norm(dim=-1)

    best = 0.0
    seed_length = length
    while seed_length >= _MIN_FIT:
        for start in range(length - seed_length + 1):
            index = torch.arange(start, start + seed_length)
            cutoff = d0
            for _ in range(20):
                rotation, offset = _kabsch(mobile[index], target[index])
                distance = distances(rotation, offset)
                best = max(best, float((1 / (1 + (distance / d0) ** 2)).mean()))
                # Re-select on the current fit and refit; widen the cutoff if too
                # few survive, or the next Kabsch is underdetermined.
                selected = torch.nonzero(distance < cutoff, as_tuple=False).flatten()
                while selected.numel() < _MIN_FIT and cutoff < distance.max():
                    cutoff += 0.5
                    selected = torch.nonzero(
                        distance < cutoff, as_tuple=False
                    ).flatten()
                if selected.numel() < _MIN_FIT or (
                    selected.numel() == index.numel()
                    and bool((selected == index).all())
                ):
                    break
                index = selected
        seed_length //= 2
    return best


def _best_offset(
    numbers: dict[int, str],
    deposited: dict[tuple[str, int], tuple[str, tuple[float, ...]]],
    target_chain: str,
) -> tuple[int, int]:
    """Residue-number offset for one chain, and how many names agree under it.

    Numbering cannot be assumed: an input builder may emit 0-based indices while
    a deposit starts at 1 — or at 17, or with gaps where residues are disordered.
    Rather than hardcode a convention that silently breaks on the next target,
    pick the offset that makes the most *residue names* agree. That is
    self-checking: a wrong offset agrees on a handful of positions by chance, and
    the count reported next to the score says which happened.
    """
    deposit_numbers = [n for c, n in deposited if c == target_chain]
    if not numbers or not deposit_numbers:
        return 0, 0
    low = min(deposit_numbers) - max(numbers)
    high = max(deposit_numbers) - min(numbers)
    best = (0, -1)
    for offset in range(low, high + 1):
        agree = sum(
            1
            for number, name in numbers.items()
            if deposited.get((target_chain, number + offset), (None,))[0] == name
        )
        if agree > best[1]:
            best = (offset, agree)
    return best


def score_against_deposit(
    predicted: dict[tuple[str, int], tuple[str, tuple[float, ...]]],
    deposited: dict[tuple[str, int], tuple[str, tuple[float, ...]]],
    chain_map: dict[str, str],
) -> dict:
    """Pair the two on ``(chain, residue number)`` and score what matched."""
    offsets: dict[str, dict[str, int]] = {}
    for chain in sorted({c for c, _ in predicted}):
        target_chain = chain_map.get(chain)
        if target_chain is None:
            continue
        names = {n: v[0] for (c, n), v in predicted.items() if c == chain}
        offset, agree = _best_offset(names, deposited, target_chain)
        offsets[chain] = {
            "residue_number_offset": offset,
            "names_agreeing": agree,
            "residues_in_chain": len(names),
        }

    pairs: list[tuple[str, tuple, tuple]] = []
    mismatched: list[str] = []
    for (chain, number), (name, xyz) in sorted(predicted.items()):
        target_chain = chain_map.get(chain)
        if target_chain is None:
            continue
        offset = offsets.get(chain, {}).get("residue_number_offset", 0)
        found = deposited.get((target_chain, number + offset))
        if found is None:
            continue
        if found[0] != name:
            mismatched.append(f"{chain}{number}: predicted {name}, deposit {found[0]}")
            continue
        pairs.append((chain, xyz, found[1]))

    report = {
        # Per chain, so a target where one chain aligned and another did not is
        # legible rather than averaged into a single misleading number.
        "alignment": offsets,
        "matched_residues": len(pairs),
        "predicted_residues": len(predicted),
        "deposited_residues": len(deposited),
        # Non-empty means the chain map or the numbering is wrong. The RMSD below
        # is then computed on whatever happened to line up, which is worse than
        # no number at all — so it is reported next to it, not swallowed.
        "residue_name_mismatches": mismatched[:10],
        "n_residue_name_mismatches": len(mismatched),
    }
    if not pairs:
        report["error"] = "no residues matched — check the chain map"
        return report

    mobile = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    target = torch.tensor([p[2] for p in pairs], dtype=torch.float64)
    report["rmsd"] = round(kabsch_rmsd(mobile, target), 3)
    tm = tm_score(mobile, target)
    report["tm_score"] = round(tm, 4) if tm is not None else None
    report["per_chain"] = {}
    for chain in sorted({p[0] for p in pairs}):
        subset = [p for p in pairs if p[0] == chain]
        # Each chain superposed on its own: a global number on a complex is
        # dominated by whichever domain is placed worst and hides that the rest
        # may be right.
        chain_mobile = torch.tensor([p[1] for p in subset], dtype=torch.float64)
        chain_target = torch.tensor([p[2] for p in subset], dtype=torch.float64)
        chain_tm = tm_score(chain_mobile, chain_target)
        report["per_chain"][chain] = {
            "matched": len(subset),
            "rmsd": round(kabsch_rmsd(chain_mobile, chain_target), 3),
            "tm_score": round(chain_tm, 4) if chain_tm is not None else None,
        }
    return report
