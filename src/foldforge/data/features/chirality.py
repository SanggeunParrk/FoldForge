"""Chiral-centre features, as RoseTTAFold3's atomworks featurisation builds them.

For every residue, RDKit's tetrahedral chiral centres of the CCD component
(assigned from its 3D coordinates; a P or S bonded to two oxygens excluded)
give plane pairs (c, i, j, k): the chiral centre c, two of its bonded heavy
atoms i < j, and a third k, over every side of the tetrahedron its bonded
atoms span. Each pair's target is the ideal dihedral between a tetrahedral
side and the plane through two of its atoms and the centre, arcsin(1/sqrt(3)),
signed by the component's own geometry. A pair that names an atom the residue
does not have (a hydrogen, a leaving atom) is dropped.

The diffusion encoder embeds the gradient of the squared dihedral error with
respect to the noisy coordinates (see ``modules.dense.chirality``). Indices
are flat into the dense (token, atom-slot) layout, so they survive token
bucketing, which pads tokens but keeps the slot axis.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

#: The two example keys this writes.
CENTRES, DIHEDRALS = "ref_chiral_centres", "ref_chiral_dihedrals"
_IDEAL = float(np.arcsin(1 / 3**0.5))
_OXYGEN, _PHOSPHORUS, _SULFUR = 8, 15, 16
#: A P or S centre bonded to this many oxygens is not treated as chiral.
_EXCLUDED_OXYGENS = 2


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    eps = 1e-4
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / (np.linalg.norm(b1) + eps)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return float(np.arctan2(np.dot(np.cross(b1n, v), w) + eps, np.dot(v, w) + eps))


def _component_plane_pairs(ccd: Any, name: str) -> list[tuple[tuple[str, ...], float]]:
    """(atom names c, i, j, k; signed target) for one CCD component."""
    from alphafold3.data.tools import rdkit_utils  # noqa: PLC0415 - optional dep
    from rdkit import Chem  # noqa: PLC0415 - optional dep
    from rdkit.Chem import AllChem  # noqa: PLC0415 - optional dep

    cif = ccd.get(name)
    if cif is None:
        return []
    try:
        mol = rdkit_utils.mol_from_ccd_cif(cif, remove_hydrogens=False)
        Chem.SanitizeMol(mol)
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=0)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        centres = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    except Exception:  # noqa: BLE001 - a component RDKit cannot read has no chiral rows
        return []
    tetrahedral = (
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    )
    xyz = mol.GetConformer().GetPositions()
    atom_name = [a.GetProp("atom_name") for a in mol.GetAtoms()]
    pairs = []
    for index, _ in centres:
        atom = mol.GetAtomWithIdx(index)
        if atom.GetChiralTag() not in tetrahedral:
            continue
        neighbours = sorted(b.GetOtherAtomIdx(index) for b in atom.GetBonds())
        if (
            atom.GetAtomicNum() in (_PHOSPHORUS, _SULFUR)
            and [mol.GetAtomWithIdx(n).GetAtomicNum() for n in neighbours].count(
                _OXYGEN
            )
            >= _EXCLUDED_OXYGENS
        ):
            continue
        for side in combinations(neighbours, 3):
            for i, j in combinations(side, 2):
                k = next(n for n in side if n not in (i, j))
                sign = np.sign(_dihedral(xyz[index], xyz[i], xyz[j], xyz[k]))
                names = tuple(atom_name[n] for n in (index, i, j, k))
                pairs.append((names, _IDEAL * float(sign)))
    return pairs


def chiral_features(example: dict[str, Any], ccd: Any) -> dict[str, np.ndarray]:
    """Flat dense-layout indices (n, 4) and signed targets (n,) for ``example``."""
    from alphafold3.model import feat_batch  # noqa: PLC0415 - optional dep

    layout = feat_batch.Batch.from_data_dict(example).convert_model_output
    layout = layout.token_atoms_layout
    names = np.asarray(layout.atom_name)
    slots = names.shape[-1]
    residues: dict[tuple[str, int], dict[str, int]] = {}
    resname: dict[tuple[str, int], str] = {}
    for t, a in zip(*np.nonzero(names != ""), strict=True):
        key = (str(layout.chain_id[t, a]), int(layout.res_id[t, a]))
        residues.setdefault(key, {})[str(names[t, a])] = int(t) * slots + int(a)
        resname[key] = str(layout.res_name[t, a])
    cache: dict[str, list] = {}
    centres, targets = [], []
    for key, atoms in residues.items():
        name = resname[key]
        if name not in cache:
            cache[name] = _component_plane_pairs(ccd, name)
        for pair_names, target in cache[name]:
            if all(n in atoms for n in pair_names):
                centres.append([atoms[n] for n in pair_names])
                targets.append(target)
    return {
        CENTRES: np.asarray(centres, dtype=np.int32).reshape(-1, 4),
        DIHEDRALS: np.asarray(targets, dtype=np.float32),
    }
