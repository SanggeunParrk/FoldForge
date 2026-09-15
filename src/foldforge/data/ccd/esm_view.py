# Optional chemistry/model dependencies load only at their execution boundary.
# ruff: noqa: PLC0415
"""ESMFold2 reference-coordinate view of the shared CCD database.

Adapted from pinned Biohub ESM; the model chooses conformer/atom policy,
while the database owns the chemistry, source identity and lookup caches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from esm.models.esmfold2.constants import RES_TYPE_TO_CCD

from foldforge.data.ccd.database import current_database

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rdkit.Chem.rdchem import Conformer, Mol


def load_ccd(*_args: object) -> Mapping[str, Mol]:
    """Load ccd."""
    return current_database().esm_molecules


def _get_ccd_molecules() -> Mapping[str, Mol]:
    return current_database().esm_molecules


def _get_ccd_mol_with_significant_h(
    comp_id: str,
) -> tuple[Mol, Conformer] | tuple[None, None]:
    """Get CCD molecule with only chemically significant hydrogens.

    Returns (mol, conformer) tuple or (None, None) if not available.
    """
    ccd = _get_ccd_molecules()
    if comp_id not in ccd:
        return None, None

    mol = ccd[comp_id]
    if mol.GetNumConformers() == 0:
        return None, None

    # Find the "Computed" conformer (RDKit ETKDGv3), fall back to "Ideal"
    conf_idx = 0
    for i, c in enumerate(mol.GetConformers()):
        props = c.GetPropsAsDict()
        if props.get("name") == "Computed":
            conf_idx = i
            break
    else:
        for i, c in enumerate(mol.GetConformers()):
            props = c.GetPropsAsDict()
            if props.get("name") == "Ideal":
                conf_idx = i
                break

    from rdkit import Chem

    mol_no_h = Chem.RemoveHs(mol, sanitize=False)

    if mol_no_h.GetNumConformers() == 0:
        return None, None

    return mol_no_h, mol_no_h.GetConformer(
        min(conf_idx, mol_no_h.GetNumConformers() - 1)
    )


def get_ccd_conformer(comp_id: str) -> dict[str, np.ndarray] | None:
    """Get idealized conformer as dict of atom_name -> position [3].

    Conformer priority: Computed > Ideal > first available.
    """
    if comp_id in current_database().cache("esm_ccd_conformers"):
        cached = current_database().cache("esm_ccd_conformers")[comp_id]
        return cached or None

    mol, conf = _get_ccd_mol_with_significant_h(comp_id)
    if mol is None or conf is None:
        current_database().cache("esm_ccd_conformers")[comp_id] = {}
        return None

    conformer: dict[str, np.ndarray] = {}
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_name = props.get("name")
        if not isinstance(atom_name, str) or not atom_name:
            continue
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        conformer[atom_name] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

    current_database().cache("esm_ccd_conformers")[comp_id] = conformer
    return conformer or None


def get_idealized_atom_pos(res_type: int, atom_name: str) -> np.ndarray | None:
    """Get idealized position for a standard residue atom.

    Uses res_type index to look up CCD component, then returns position.
    Returns None if not found.
    """
    cache_key = (res_type, atom_name)
    if cache_key in current_database().cache("esm_idealized_pos_cache"):
        return current_database().cache("esm_idealized_pos_cache")[cache_key]

    comp_id = RES_TYPE_TO_CCD.get(res_type)
    if comp_id:
        ccd_conformer = get_ccd_conformer(comp_id)
        if ccd_conformer and atom_name in ccd_conformer:
            pos = ccd_conformer[atom_name]
            current_database().cache("esm_idealized_pos_cache")[cache_key] = pos
            return pos

    current_database().cache("esm_idealized_pos_cache")[cache_key] = None
    return None


def get_ligand_idealized_atom_pos(res_name: str, atom_name: str) -> np.ndarray | None:
    """Get idealized position for a ligand/modified residue atom.

    Returns None if not found.
    """
    cache_key = (res_name, atom_name)
    if cache_key in current_database().cache("esm_ligand_idealized_pos_cache"):
        return current_database().cache("esm_ligand_idealized_pos_cache")[cache_key]

    ccd_conformer = get_ccd_conformer(res_name)
    if ccd_conformer and atom_name in ccd_conformer:
        pos = ccd_conformer[atom_name]
        current_database().cache("esm_ligand_idealized_pos_cache")[cache_key] = pos
        return pos

    current_database().cache("esm_ligand_idealized_pos_cache")[cache_key] = None
    return None


def get_ligand_ccd_atoms_with_charges(
    comp_id: str,
) -> list[tuple[str, str, int]] | None:
    """Get list of (atom_name, element, charge) for a CCD component.

    Uses RDKit RemoveHs(sanitize=False) to keep chemically significant hydrogens.
    Returns None if CCD data not available.
    """
    if comp_id in current_database().cache("esm_ccd_atom_cache"):
        cached = current_database().cache("esm_ccd_atom_cache")[comp_id]
        return cached or None

    mol, _ = _get_ccd_mol_with_significant_h(comp_id)
    if mol is None:
        current_database().cache("esm_ccd_atom_cache")[comp_id] = []
        return None

    atoms: list[tuple[str, str, int]] = []
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_name = props.get("name")
        if not isinstance(atom_name, str) or not atom_name:
            continue
        element = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        atoms.append((atom_name, element, charge))

    current_database().cache("esm_ccd_atom_cache")[comp_id] = atoms
    return atoms or None


def get_ligand_ccd_bonds(comp_id: str) -> list[tuple[str, str]] | None:
    """Get list of (atom1_name, atom2_name) bonds for a CCD component.

    Returns None if CCD data not available.
    """
    if comp_id in current_database().cache("esm_ccd_bonds_cache"):
        cached = current_database().cache("esm_ccd_bonds_cache")[comp_id]
        return cached or None

    mol, _ = _get_ccd_mol_with_significant_h(comp_id)
    if mol is None:
        current_database().cache("esm_ccd_bonds_cache")[comp_id] = []
        return None

    # Get included atom names
    included_atoms = set()
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_name = props.get("name")
        if isinstance(atom_name, str) and atom_name:
            included_atoms.add(atom_name)

    bonds: list[tuple[str, str]] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        n1 = a1.GetPropsAsDict().get("name")
        n2 = a2.GetPropsAsDict().get("name")
        if (
            isinstance(n1, str)
            and isinstance(n2, str)
            and n1
            and n2
            and n1 in included_atoms
            and n2 in included_atoms
        ):
            bonds.append((n1, n2))

    current_database().cache("esm_ccd_bonds_cache")[comp_id] = bonds
    return bonds or None


def get_ccd_leaving_atoms(comp_id: str) -> set[str]:
    """Get set of atom names marked as leaving atoms in CCD.

    Leaving atoms are removed during polymerization (e.g., OP3 in nucleotides).
    """
    if comp_id in current_database().cache("esm_ccd_leaving_atoms_cache"):
        return current_database().cache("esm_ccd_leaving_atoms_cache")[comp_id]

    ccd = _get_ccd_molecules()
    if comp_id not in ccd:
        current_database().cache("esm_ccd_leaving_atoms_cache")[comp_id] = set()
        return set()

    mol = ccd[comp_id]
    leaving_atoms = set()
    for atom in mol.GetAtoms():
        if atom.HasProp("leaving_atom") and atom.GetProp("leaving_atom") == "1":
            name = atom.GetProp("name") if atom.HasProp("name") else ""
            if name:
                leaving_atoms.add(name)

    current_database().cache("esm_ccd_leaving_atoms_cache")[comp_id] = leaving_atoms
    return leaving_atoms
