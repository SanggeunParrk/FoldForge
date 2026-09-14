"""The scorer's two failure modes: a wrong pairing, and a crash instead of a report."""

import math

import torch

from foldforge.eval.structure import kabsch_rmsd, score_against_deposit, tm_score

# MQIFV... — ubiquitin's first residues, so a shifted pairing is visibly wrong.
NAMES = [
    "MET",
    "GLN",
    "ILE",
    "PHE",
    "VAL",
    "LYS",
    "THR",
    "LEU",
    "THR",
    "GLY",
    "LYS",
    "THR",
    "ILE",
    "THR",
    "LEU",
    "GLU",
    "VAL",
    "GLU",
    "PRO",
    "SER",
]


def _chain(numbers, coords) -> dict:
    return {
        ("A", n): (NAMES[i], tuple(coords[i].tolist())) for i, n in enumerate(numbers)
    }


def _helix(n) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64)
    return torch.stack([2.3 * torch.cos(t), 2.3 * torch.sin(t), 1.5 * t], dim=-1)


def test_zero_based_prediction_aligns_to_one_based_deposit():
    # The real case: the input builder emits 0..N-1, the deposit starts at 1.
    coords = _helix(len(NAMES))
    predicted = _chain(range(len(NAMES)), coords)
    deposited = _chain(range(1, len(NAMES) + 1), coords)
    report = score_against_deposit(predicted, deposited, {"A": "A"})
    assert report["alignment"]["A"]["residue_number_offset"] == 1
    assert report["matched_residues"] == len(NAMES)
    assert report["rmsd"] < 1e-6
    assert math.isclose(report["tm_score"], 1.0, abs_tol=1e-3)


def test_a_shifted_pairing_is_rejected_not_scored():
    # Same numbering, but the deposit's sequence is rotated: no offset makes the
    # names agree, so the scorer must report that rather than produce a number
    # from whatever lined up.
    coords = _helix(len(NAMES))
    predicted = _chain(range(len(NAMES)), coords)
    rotated = {
        ("A", n + 1): (NAMES[(i + 7) % len(NAMES)], tuple(coords[i].tolist()))
        for i, n in enumerate(range(len(NAMES)))
    }
    report = score_against_deposit(predicted, rotated, {"A": "A"})
    assert report["alignment"]["A"]["names_agreeing"] < len(NAMES)
    assert report["matched_residues"] < len(NAMES)


def test_too_few_matches_reports_instead_of_raising():
    coords = _helix(4)
    predicted = {("A", i): (NAMES[i], tuple(coords[i].tolist())) for i in range(4)}
    report = score_against_deposit(predicted, predicted, {"A": "A"})
    # RMSD is still meaningful on 4 points; TM-score is not, and says so.
    assert report["tm_score"] is None
    assert report["rmsd"] is not None


def test_tm_score_beats_a_single_kabsch_fit_on_a_displaced_tail():
    # Why the Zhang-Skolnick search matters: one flung tail drags a single fit,
    # and a one-shot TM would understate a structure whose core is right.
    native = _helix(60)
    model = native.clone()
    model[50:] += torch.tensor([25.0, 25.0, 25.0], dtype=torch.float64)
    centred = (native - native.mean(0)) - (model - model.mean(0))
    naive = float((1 / (1 + centred.norm(dim=-1) ** 2)).mean())
    assert tm_score(model, native) > naive
    assert kabsch_rmsd(model, native) > 5.0


def test_prediction_cif_without_occupancy_keeps_only_first_model_ca(tmp_path):
    from foldforge.eval.structure import deposit_ca  # noqa: PLC0415

    path = tmp_path / "prediction.cif"
    path.write_text("""data_prediction
loop_
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
CA C ALA A 1 1 2 3 1
CB C ALA A 1 90 90 90 1
CA C ALA A 1 99 99 99 2
CA Ca CA B 2 100 100 100 1
#
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
CA C
#
""")
    assert deposit_ca(path) == {("A", 1): ("ALA", (1.0, 2.0, 3.0))}
