from __future__ import annotations

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from rdkit import Chem

from foldforge.data.bonds import replace_bond_array as _replace_bond_array
from foldforge.data.ccd.database import current_database, database_cache
from foldforge.data.ccd.substructure_perms import get_substructure_perms

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from biotite.structure import AtomArray
    from rdkit.Chem.rdchem import Mol

logger = logging.getLogger(__name__)


MAX_INTER_RESIDUE_BOND_ANGSTROM = 2.5


def _map_central_to_leaving_groups(
    component: AtomArray,
) -> dict[str, list[list[str]]] | None:
    """Map central to leaving groups.

    map each central atom (bonded atom) index to leaving atom groups in component
    (atom_array).

    Returns:
        dict[str, list[list[str]]]: central atom name to leaving atom groups (atom
        names).

    """
    comp = component.copy()
    # Eg: ions  # noqa: ERA001 - format example
    if comp.bonds is None:
        return {}
    central_to_leaving_groups = defaultdict(list)
    for c_idx in np.flatnonzero(~comp.leaving_atom_flag):
        bonds, _ = comp.bonds.get_bonds(int(c_idx))
        for l_idx in bonds:
            if comp.leaving_atom_flag[l_idx]:
                comp.bonds.remove_bond(int(c_idx), int(l_idx))
                group_idx = struc.find_connected(comp.bonds, l_idx)
                if not np.all(comp.leaving_atom_flag[group_idx]):
                    return None
                central_to_leaving_groups[comp.atom_name[c_idx]].append(
                    comp.atom_name[group_idx].tolist()
                )
    return central_to_leaving_groups


@database_cache
def get_component_atom_array(
    ccd_code: str, *, keep_leaving_atoms: bool = False, keep_hydrogens: bool = False
) -> AtomArray | None:
    """Get component atom array.

    Args:
        ccd_code (str): ccd code
        keep_leaving_atoms (bool, optional): keep leaving atoms. Defaults to False.
        keep_hydrogens (bool, optional): keep hydrogens. Defaults to False.

    Returns:
        AtomArray: Biotite AtomArray of CCD component
            with additional attribute: leaving_atom_flag (bool)

    """
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        logger.warning(
            "Warning: get_component_atom_array() can not parse %s", f"{ccd_code}"
        )
        return None
    try:
        comp = pdbx.get_component(
            ccd_cif[ccd_code], use_ideal_coord=True, allow_missing_coord=True
        )
    except biotite.InvalidFileError as e:
        # Eg: UNL without atom.
        logger.warning(
            "Warning: get_component_atom_array() can not parse %s for %s",
            f"{ccd_code}",
            f"{e}",
        )
        return None
    atom_category = ccd_cif[ccd_code]["chem_comp_atom"]
    leaving_atom_flag = atom_category["pdbx_leaving_atom_flag"].as_array()
    comp.set_annotation("leaving_atom_flag", leaving_atom_flag == "Y")

    for atom_id in ["alt_atom_id", "pdbx_component_atom_id"]:
        comp.set_annotation(atom_id, atom_category[atom_id].as_array())
    if not keep_leaving_atoms:
        comp = comp[~comp.leaving_atom_flag]
    if not keep_hydrogens:
        comp = comp[~np.isin(comp.element, ["H", "D"])]

    # Map central atom index to leaving group (atom_indices) in component (atom_array).
    comp.central_to_leaving_groups = _map_central_to_leaving_groups(comp)
    if comp.central_to_leaving_groups is None:
        logger.debug(
            (
                "CCD %s has leaving atom group bond to more than one central "
                "atom, central_to_leaving_groups is None."
            ),
            f"{ccd_code}",
        )
    return comp


@database_cache(maxsize=None)
def get_one_letter_code(ccd_code: str) -> str | None:
    """Get one_letter_code from CCD components file.

    normal return is one letter: ALA --> A, DT --> T
    unknown protein: X
    unknown DNA or RNA: N
    other unknown: None
    some ccd_code will return more than one letter:
    eg: XXY --> THG

    Args:
        ccd_code (str): _description_

    Returns:
        str: one letter code

    """
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return None
    one = ccd_cif[ccd_code]["chem_comp"]["one_letter_code"].as_item()
    if one == "?":
        return None
    return one


@database_cache(maxsize=None)
def get_mol_type(ccd_code: str) -> str:
    """Get mol_type from CCD components file.

    based on _chem_comp.type
    http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html

    not use _chem_comp.pdbx_type, because it is not consistent with _chem_comp.type
    e.g. ccd 000 --> _chem_comp.type="NON-POLYMER" _chem_comp.pdbx_type="ATOMP"
    https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic/Items/_struct_asym.pdbx_type.html

    Args:
        ccd_code (str): ccd code

    Returns:
        str: mol_type, one of {"protein", "rna", "dna", "ligand"}

    """
    ccd_cif = biotite_load_ccd_cif()
    if ccd_code not in ccd_cif:
        return "ligand"

    link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()

    if "PEPTIDE" in link_type and link_type != "PEPTIDE-LIKE":
        return "protein"
    if "DNA" in link_type:
        return "dna"
    if "RNA" in link_type:
        return "rna"
    return "ligand"


def get_all_ccd_code() -> list[str]:
    """Get all ccd code from components file."""
    ccd_cif = biotite_load_ccd_cif()
    return list(ccd_cif.keys())


@database_cache
def get_ccd_ref_info(
    ccd_code: str,
    *,
    return_perm: bool = True,
    ccd_mols: tuple[tuple[str, Chem.Mol]] | None = None,
    return_atomic_number: bool = False,
) -> dict[str, Any]:
    """Ref: AlphaFold3 SI Chapter 2.8.

    Reference features. Features derived from a residue, nucleotide or ligand's
    reference conformer.
    Given an input CCD code or SMILES string, the conformer is typically generated
    with RDKit v.2023_03_3 [25] using ETKDGv3 [26]. On error, we fall back to using the
    CCD ideal coordinates,
    or finally the representative coordinates
    if they are from before our training date cut-off (2021-09-30 unless otherwise
    stated).
    At the end, any atom coordinates still missing are set to zeros.

    Get reference atom mapping and coordinates.

    Args:
        ccd_code: CCD component identifier.
        name (str): CCD name
        return_perm (bool): return atom permutations.
        ccd_mols (Optional[tuple[tuple[str, Chem.Mol]]]): The self-defined CCD
            molecules. Defaults to None.
        return_atomic_number (bool): whether to return atomic numbers.

    Returns:
        Dict:
            ccd: ccd code
            atom_map: atom name to atom index
            coord: atom coordinates
            charge: atom formal charge
            perm: atom permutation
            atomic_number: atomic number

    """
    component_molecules = dict(ccd_mols or ())
    if ccd_code in component_molecules:
        mol = copy.deepcopy(component_molecules[ccd_code])
    else:
        mol = copy.deepcopy(get_component_rdkit_mol(ccd_code))

    if mol is None:
        return {}
    if mol.GetNumAtoms() == 0:  # eg: "UNL"
        logger.warning(
            (
                "Warning: mol %s from get_component_rdkit_mol() has no "
                "atoms,get_ccd_ref_info() return empty dict"
            ),
            f"{ccd_code}",
        )
        return {}
    conf = mol.GetConformer(mol.ref_conf_id)
    coord = conf.GetPositions()
    charge = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    results = {
        "ccd": ccd_code,  # str
        "atom_map": mol.atom_map,  # dict[str,int]: atom name to atom index
        "coord": coord,  # np.ndarray[float]: atom coordinates, shape:(n_atom,3)
        "mask": mol.ref_mask,  # np.ndarray[bool]: atom mask, shape:(n_atom,)
        "charge": charge,  # np.ndarray[int]: atom formal charge, shape:(n_atom,)
    }
    if return_atomic_number:
        results["atomic_number"] = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        )

    if return_perm:
        try:
            Chem.SanitizeMol(mol)
            perm = get_substructure_perms(mol, MaxMatches=1000)

        except (RuntimeError, ValueError):
            # Sanitize failed, permutation is unavailable
            perm = np.array(
                [
                    [
                        i
                        for i, atom in enumerate(mol.GetAtoms())
                        if atom.GetAtomicNum() != 1
                    ]
                ]
            )
        results["perm"] = (
            perm.T
        )  # np.ndarray[int]: atom permutation, shape:(n_atom_wo_h, n_perm)

    return results


# Modified from biotite to use consistent ccd components file
def _connect_inter_residue(
    atoms: AtomArray, residue_starts: np.ndarray
) -> struc.BondList:
    """Create a :class:`BondList` containing the bonds between adjacent.

    amino acid or nucleotide residues.

    Parameters
    ----------
    atoms : AtomArray
        The structure to create the :class:`BondList` for.
    residue_starts : np.ndarray
        Return value of
        ``get_residue_starts(atoms, add_exclusive_stop=True)``.

    Returns
    -------
    struc.BondList
        A bond list containing all inter residue bonds.

    """
    bonds = []

    atom_names = atoms.atom_name
    res_names = atoms.res_name
    res_ids = atoms.res_id
    chain_ids = atoms.chain_id

    # Iterate over all starts excluding:
    #   - the last residue and
    #   - exclusive end index of 'atoms'
    for i in range(len(residue_starts) - 2):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i + 1]
        after_next_start_i = residue_starts[i + 2]

        # Check if the current and next residue is in the same chain
        if chain_ids[next_start_i] != chain_ids[curr_start_i]:
            continue
        # Check if the current and next residue
        # have consecutive residue IDs
        # (Same residue ID is also possible if insertion code is used)
        if res_ids[next_start_i] - res_ids[curr_start_i] > 1:
            continue

        # Get link type for this residue from RCSB components.cif
        curr_link = get_mol_type(str(res_names[curr_start_i]))
        next_link = get_mol_type(str(res_names[next_start_i]))

        if curr_link == "protein" and next_link in "protein":
            curr_connect_atom_name = "C"
            next_connect_atom_name = "N"
        elif curr_link in ["dna", "rna"] and next_link in ["dna", "rna"]:
            curr_connect_atom_name = "O3'"
            next_connect_atom_name = "P"
        else:
            # Create no bond if the connection types of consecutive
            # residues are not compatible
            continue

        # Index in atom array for atom name in current residue
        # Addition of 'curr_start_i' is necessary, as only a slice of
        # 'atom_names' is taken, beginning at 'curr_start_i'
        curr_connect_indices = np.where(
            atom_names[curr_start_i:next_start_i] == curr_connect_atom_name
        )[0]
        curr_connect_indices += curr_start_i

        # Index in atom array for atom name in next residue
        next_connect_indices = np.where(
            atom_names[next_start_i:after_next_start_i] == next_connect_atom_name
        )[0]
        next_connect_indices += next_start_i

        if len(curr_connect_indices) == 0 or len(next_connect_indices) == 0:
            # The connector atoms are not found in the adjacent residues
            # -> skip this bond
            continue

        bonds.append(
            (curr_connect_indices[0], next_connect_indices[0], struc.BondType.SINGLE)
        )

    return struc.BondList(atoms.array_length(), np.array(bonds, dtype=np.uint32))


def add_inter_residue_bonds(  # noqa: C901 - ordered polymer bond repair
    atom_array: AtomArray,
    *,
    exclude_struct_conn_pairs: bool = False,
    remove_far_inter_chain_pairs: bool = False,
) -> AtomArray:
    """Add inter residue bonds.

    add polymer bonds (C-N or O3'-P) between adjacent residues based on auth_seq_id.

    exclude_struct_conn_pairs: if True, do not add bond between adjacent residues
        already has non-standard polymer bonds
                  on atom C or N or O3' or P.

    remove_far_inter_chain_pairs: if True, remove inter chain (based on label_asym_id)
        bonds that are far away from each other.

    Returns:
        AtomArray: Biotite AtomArray merged inter residue bonds into atom_array.bonds

    """
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    inter_bonds = _connect_inter_residue(atom_array, res_starts)

    if atom_array.bonds is None:
        atom_array.bonds = inter_bonds
        return atom_array

    select_mask = np.ones(len(inter_bonds.as_array()), dtype=bool)
    if exclude_struct_conn_pairs:
        for b_idx, (atom_i, atom_j, _b_type) in enumerate(inter_bonds.as_array()):
            atom_k = atom_i if atom_array.atom_name[atom_i] in ("N", "O3'") else atom_j
            bonds, _types = atom_array.bonds.get_bonds(atom_k)
            if len(bonds) == 0:
                continue
            for b in bonds:
                if (
                    # adjacent residues
                    abs((res_starts <= b).sum() - (res_starts <= atom_k).sum()) == 1
                    and atom_array.chain_id[b] == atom_array.chain_id[atom_k]
                    and atom_array.atom_name[b] not in ("C", "P")
                ):
                    select_mask[b_idx] = False
                    break

    if remove_far_inter_chain_pairs:
        if not hasattr(atom_array, "label_asym_id"):
            logger.warning(
                "label_asym_id not found, far inter chain bonds will not be removed"
            )
        for b_idx, (atom_i, atom_j, _b_type) in enumerate(inter_bonds.as_array()):
            if atom_array.label_asym_id[atom_i] != atom_array.label_asym_id[atom_j]:
                coord_i = atom_array.coord[atom_i]
                coord_j = atom_array.coord[atom_j]
                if np.linalg.norm(coord_i - coord_j) > MAX_INTER_RESIDUE_BOND_ANGSTROM:
                    select_mask[b_idx] = False

    # filter out removed_inter_bonds from atom_array.bonds
    remove_bonds = inter_bonds.as_array()[~select_mask]
    remove_mask = np.isin(
        atom_array.bonds.as_array()[:, 0], remove_bonds[:, 0]
    ) & np.isin(atom_array.bonds.as_array()[:, 1], remove_bonds[:, 1])
    atom_array.bonds = _replace_bond_array(
        atom_array.bonds, atom_array.bonds.as_array()[~remove_mask]
    )

    # merged normal inter_bonds into atom_array.bonds
    inter_bonds = _replace_bond_array(inter_bonds, inter_bonds.as_array()[select_mask])
    atom_array.bonds = atom_array.bonds.merge(inter_bonds)
    return atom_array


def res_names_to_sequence(res_names: Iterable[str]) -> str:
    """Convert res_names to sequences {chain_id: canonical_sequence} based on CCD.

    Return:
        str: canonical_sequence

    """
    seq = ""
    for res_name in res_names:
        one = get_one_letter_code(res_name)
        one = "X" if one is None else one
        one = "X" if len(one) > 1 else one
        seq += one
    return seq


def biotite_load_ccd_cif() -> Mapping[str, Any]:
    """Compute biotite load ccd cif."""
    return current_database().cif


def get_component_rdkit_mol(ccd_code: str) -> Mol | None:
    """Return component rdkit mol."""
    return current_database().molecule(ccd_code)


def get_ccd_cache_paths() -> tuple[str, str]:
    """Return ccd cache paths."""
    config = current_database().config
    return str(config.ccd_db), str(config.ccd_db)


def set_ccd_cache_paths(
    components_file: str | Path | None = None, rdkit_mol_pkl: str | Path | None = None
) -> None:
    # Compatibility for the upstream dataset signature; selection belongs to
    # the caller's database context and may not change behind its back.
    """Set ccd cache paths."""
    current = get_ccd_cache_paths()
    proposed = (components_file, rdkit_mol_pkl)
    if any(
        value is not None and Path(value).resolve() != Path(active).resolve()
        for value, active in zip(proposed, current, strict=True)
    ):
        msg = "Model CCD paths disagree with the active shared database"
        raise ValueError(msg)
