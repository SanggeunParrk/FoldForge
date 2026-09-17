from __future__ import annotations

import copy
import itertools
import queue
import random
import threading
import warnings
from collections import Counter
from queue import Queue
from typing import TYPE_CHECKING, Any, cast

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from foldforge.data.ccd import components as ccd
from foldforge.utils.logging import get_logger
from foldforge.utils.seed import conformer_seed

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
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research


if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

logger = get_logger(__name__)


DNA_1to3 = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "X": "DN",
    "I": "DI",  # eg: pdb 114d
    "N": "DN",  # eg: pdb 7r6t-3DR
    "U": "DU",  # eg: pdb 7sd8
}


RNA_1to3 = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "X": "N",
    "I": "I",  # eg: pdb 7wv5
    "N": "N",
}


PROTEIN_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}


def add_reference_features(atom_array: AtomArray) -> AtomArray:
    """Add reference features of each resiude to atom_array.

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with reference features
        - ref_pos: reference conformer atom positions
        - ref_charge (n): reference conformer atom charges
        - ref_mask: reference conformer atom masks

    """
    atom_count = len(atom_array)
    ref_pos = np.zeros((atom_count, 3), dtype=np.float32)
    ref_charge = np.zeros(atom_count, dtype=int)
    ref_mask = np.zeros(atom_count, dtype=int)

    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in itertools.pairwise(starts):
        res_name = atom_array.res_name[start]
        if res_name == "UNL":
            # UNL is smiles ligand, copy info from atom_array
            ref_pos[start:stop] = atom_array.coord[start:stop]
            ref_charge[start:stop] = atom_array.charge[start:stop]
            ref_mask[start:stop] = 1
            continue

        ref_info = ccd.get_ccd_ref_info(res_name)
        if ref_info:
            atom_sub_idx = [
                *map(ref_info["atom_map"].get, atom_array.atom_name[start:stop])
            ]
            ref_pos[start:stop] = ref_info["coord"][atom_sub_idx]
            ref_charge[start:stop] = ref_info["charge"][atom_sub_idx]
            ref_mask[start:stop] = ref_info["mask"][atom_sub_idx]
        else:
            logger.warning("no reference info for %s", f"{res_name}")

    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_charge", ref_charge)
    atom_array.set_annotation("ref_mask", ref_mask)
    return atom_array


def _residue_remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
    """Check polymer connections and remove non-standard leaving atoms.

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with leaving atoms removed.

    """
    if atom_array.bonds is None:
        message = "Polymer connectivity requires a bond list"
        raise ValueError(message)
    connected = np.zeros(atom_array.res_id[-1], dtype=bool)
    for i, j, _t in atom_array.bonds.as_array():
        if abs(atom_array.res_id[i] - atom_array.res_id[j]) == 1:
            connected[atom_array.res_id[[i, j]].min()] = True

    leaving_atoms = np.zeros(len(atom_array), dtype=bool)
    for res_id, conn in enumerate(connected):
        if res_id == 0 or conn:
            continue

        # Res_id start from 1
        res_name_i = atom_array.res_name[atom_array.res_id == res_id][0]
        res_name_j = atom_array.res_name[atom_array.res_id == res_id + 1][0]
        warnings.warn(
            f"No C-N or O3'-P bond between residue {res_name_i}({res_id}) "
            f"and residue {res_name_j}({res_id + 1}). \n"
            f"all leaving atoms will be removed for both residues.",
            stacklevel=2,
        )
        for idx, res_name in zip(
            [res_id, res_id + 1], [res_name_i, res_name_j], strict=False
        ):
            staying_atoms = ccd.get_component_atom_array(
                ccd_code=res_name, keep_leaving_atoms=False, keep_hydrogens=False
            ).atom_name
            if idx == 1 and ccd.get_mol_type(res_name) in ("dna", "rna"):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            if idx == atom_array.res_id[-1] and ccd.get_mol_type(res_name) == "protein":
                staying_atoms = np.append(staying_atoms, ["OXT"])
            leaving_atoms |= (atom_array.res_id == idx) & (
                ~np.isin(atom_array.atom_name, staying_atoms)
            )
    return atom_array[~leaving_atoms]


def find_range_by_index(starts: np.ndarray, atom_index: int) -> tuple[int, int]:
    """Find the residue range of an atom index.

    Args:
        starts (np.ndarray): Residue starts or Chain starts with exclusive stop.
        atom_index (int): Atom index.

    Returns:
        tuple[int, int]: range (start, stop).

    """
    for start, stop in itertools.pairwise(starts):
        if start <= atom_index < stop:
            return start, stop
    msg = f"atom_index {atom_index} not found in starts {starts}"
    raise ValueError(msg)


def remove_leaving_atoms(atom_array: AtomArray, bond_count: dict) -> AtomArray:
    """Remove leaving atoms based on ccd info.

    Args:
        atom_array (AtomArray): Biotite Atom array.
        bond_count (dict): atom index -> Bond count.

    Returns:
        AtomArray: Biotite Atom array with leaving atoms removed.

    """
    remove_indices = []
    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for centre_idx, b_count in bond_count.items():
        res_name = atom_array.res_name[centre_idx]
        centre_name = atom_array.atom_name[centre_idx]

        comp = ccd.get_component_atom_array(
            ccd_code=res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        if comp is None:
            continue

        leaving_groups = comp.central_to_leaving_groups.get(centre_name)
        if leaving_groups is None:
            continue

        if b_count > len(leaving_groups):
            warnings.warn(
                f"centre atom {centre_name=} {res_name=} {centre_idx=} has "
                f"{b_count} inter residue bonds, greater than number of "
                f"leaving groups:{leaving_groups}, remove all leaving atoms.\n"
                f"atom info: {atom_array[centre_idx]=}",
                stacklevel=2,
            )
            remove_groups = leaving_groups
        else:
            remove_groups = random.sample(leaving_groups, b_count)

        start, stop = find_range_by_index(res_starts, centre_idx)

        # Find leaving atom indices
        for group in remove_groups:
            for atom_name in group:
                leaving_idx = np.where(atom_array.atom_name[start:stop] == atom_name)[0]
                if len(leaving_idx) == 0:
                    logger.info(
                        "atom_name=%s not found in residue %s, ",
                        f"{atom_name!r}",
                        f"{res_name}",
                    )
                    continue

                remove_indices.append(leaving_idx[0] + start)

    if not remove_indices:
        return atom_array

    keep_mask = np.ones(len(atom_array), dtype=bool)
    keep_mask[remove_indices] = False
    return atom_array[keep_mask]


def _add_bonds_to_terminal_residues(atom_array: AtomArray) -> AtomArray:
    """Add bonds to terminal residues (eg: ACE, NME).

    Args:
        atom_array (AtomArray): Biotite AtomArray

    Returns:
        AtomArray: Biotite AtomArray with non-standard polymer bonds

    """
    if atom_array.bonds is None:
        atom_array.bonds = struc.BondList(len(atom_array))
    if atom_array.res_name[0] == "ACE":
        term_res_idx = atom_array.res_id[0]
        next_res_idx = term_res_idx + 1
        term_atom_idx = np.where(
            (atom_array.res_id == term_res_idx) & (atom_array.atom_name == "C")
        )[0]
        next_atom_idx = np.where(
            (atom_array.res_id == next_res_idx) & (atom_array.atom_name == "N")
        )[0]
        if len(term_atom_idx) > 0 and len(next_atom_idx) > 0:
            atom_array.bonds.add_bond(
                term_atom_idx[0], next_atom_idx[0], struc.BondType.SINGLE
            )

    if atom_array.res_name[-1] == "NME":
        term_res_idx = atom_array.res_id[-1]
        prev_res_idx = term_res_idx - 1
        term_atom_idx = np.where(
            (atom_array.res_id == term_res_idx) & (atom_array.atom_name == "N")
        )[0]
        prev_atom_idx = np.where(
            (atom_array.res_id == prev_res_idx) & (atom_array.atom_name == "C")
        )[0]
        if len(prev_atom_idx) > 0 and len(term_atom_idx) > 0:
            atom_array.bonds.add_bond(
                prev_atom_idx[0], term_atom_idx[0], struc.BondType.SINGLE
            )

    return atom_array


def _residue_build_polymer_atom_array(
    ccd_seqs: list[str],
) -> AtomArray:
    """Build polymer atom_array from ccd codes, but not remove leaving atoms.

    Args:
        ccd_seqs: a list of ccd code in sequence, ["MET", "ALA"] or ["DA", "DT"]

    Returns:
        AtomArray: Biotite AtomArray of chain
        BondList: Biotite BondList of polymer bonds (C-N or O3'-P)

    """
    chain = struc.AtomArray(0)
    for res_id, res_name in enumerate(ccd_seqs):
        # Keep all leaving atoms, will remove leaving atoms later
        residue = ccd.get_component_atom_array(
            ccd_code=res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        residue.res_id[:] = res_id + 1
        chain += residue
    res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    polymer_bonds = ccd._connect_inter_residue(chain, res_starts)  # noqa: SLF001 - Biotite extension hook

    if chain.bonds is None:
        chain.bonds = polymer_bonds
    else:
        chain.bonds = chain.bonds.merge(polymer_bonds)

    chain = _add_bonds_to_terminal_residues(chain)

    bond_count = {}
    for i, j, _t in polymer_bonds.as_array():
        bond_count[i] = bond_count.get(i, 0) + 1
        bond_count[j] = bond_count.get(j, 0) + 1

    chain = remove_leaving_atoms(chain, bond_count)

    return _residue_remove_non_std_ccd_leaving_atoms(chain)


def residue_build_polymer(entity_info: dict) -> dict:
    """Build a polymer from a polymer info dict.

    example: {
        "name": "polymer",
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",
        "count": 1
        }

    Args:
        entity_info: Polymer sequence, copies and chemical modifications.
        item (dict): polymer info dict

    Returns:
        dict: {"atom_array": biotite_AtomArray_object}

    """
    poly_type, info = next(iter(entity_info.items()))
    if poly_type == "proteinChain":
        ccd_seqs = [PROTEIN_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["ptmPosition"] - 1
                mtype = m["ptmType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    msg = f"unknown modification type: {mtype}"
                    raise ValueError(msg)
        if glycans := info.get("glycans"):
            logger.warning("glycans not supported: %s", f"{glycans}")
        chain_array = _residue_build_polymer_atom_array(ccd_seqs)

    elif poly_type in ("dnaSequence", "rnaSequence"):
        map_1to3 = DNA_1to3 if poly_type == "dnaSequence" else RNA_1to3
        ccd_seqs = [map_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                index = m["basePosition"] - 1
                mtype = m["modificationType"]
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    msg = f"unknown modification type: {mtype}"
                    raise ValueError(msg)
        chain_array = _residue_build_polymer_atom_array(ccd_seqs)

    else:
        msg = "polymer type must be proteinChain, dnaSequence or rnaSequence"
        raise ValueError(msg)
    chain_array = add_reference_features(chain_array)
    return {"atom_array": chain_array}


def residue_rdkit_mol_to_atom_array(
    mol: Chem.Mol,
    *,
    removeHs: bool = True,  # noqa: N803 - checkpoint-compatible keyword
) -> AtomArray:
    """Convert rdkit mol to biotite AtomArray.

    Args:
        mol (Chem.Mol): rdkit mol
        removeHs (bool): whether to remove hydrogens in atom_array

    Returns:
        AtomArray: biotite AtomArray

    """
    atom_count = mol.GetNumAtoms()
    atom_array = AtomArray(atom_count)
    atom_array.hetero[:] = True
    atom_array.res_name[:] = "UNL"
    atom_array.add_annotation("charge", int)

    conf = mol.GetConformer()
    coord = conf.GetPositions()

    element_count = Counter()
    for i, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"

        atom.SetProp("name", atom_name)

        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = element
        atom_array.charge[i] = atom.GetFormalCharge()
        atom_array.coord[i, :] = coord[i, :]

    bonds = []
    bonds.extend(
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()
    )
    atom_array.bonds = struc.BondList(atom_count, np.array(bonds))
    if removeHs:
        atom_array = atom_array[atom_array.element != "H"]
    return atom_array


def residue_rdkit_mol_to_atom_info(mol: Chem.Mol) -> dict[str, Any]:
    """Convert RDKit Mol to atom_info dict.

    Args:
        mol (Chem.Mol): rdkit mol

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    atom_info = {}
    atom_map_to_atom_name = {}
    atom_idx_to_atom_name = {}

    element_count = Counter()
    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"
        atom.SetProp("name", atom_name)
        if atom.GetAtomMapNum() != 0:
            atom_map_to_atom_name[atom.GetAtomMapNum()] = atom_name
        atom_idx_to_atom_name[atom.GetIdx()] = atom_name

    if atom_map_to_atom_name:
        # Atom map for input SMILES
        atom_info["atom_map_to_atom_name"] = atom_map_to_atom_name
    else:
        # Atom index for input file
        atom_info["atom_map_to_atom_name"] = atom_idx_to_atom_name

    # Atom_array without hydrogens
    atom_info["atom_array"] = residue_rdkit_mol_to_atom_array(mol=mol, removeHs=True)
    mol.atom_map = {atom.GetProp("name"): atom.GetIdx() for atom in mol.GetAtoms()}
    mol.ref_conf_id = 0
    mol.ref_mask = np.ones(mol.GetNumAtoms(), dtype=bool)
    atom_info["mol"] = mol
    return atom_info


def residue_lig_file_to_atom_info(lig_file_path: str) -> dict[str, Any]:
    """Convert ligand file to biotite AtomArray.

    Args:
        lig_file_path (str): ligand file path with one of the following suffixes: [mol,
            mol2, sdf, pdb]

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    if lig_file_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(lig_file_path)
    elif lig_file_path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(lig_file_path)
        mol = next(suppl)
    elif lig_file_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(lig_file_path)
    elif lig_file_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(lig_file_path)
    else:
        msg = f"Invalid ligand file type: .{lig_file_path.rsplit('.', maxsplit=1)[-1]}"
        raise ValueError(msg)
    if not (mol is not None):
        message = (
            f"Failed to retrieve molecule from file, "
            f"invalid ligand file: {lig_file_path}. "
            "Please provide one of: mol, mol2, sdf, pdb."
        )
        raise ValueError(message)

    if not (mol.GetConformer().Is3D()):
        message = f"3D conformer not found in ligand file: {lig_file_path}"
        raise ValueError(message)
    return residue_rdkit_mol_to_atom_info(mol)


def residue_smiles_to_atom_info(smiles: str) -> dict:
    """Convert smiles to atom_array, and atom_map_to_atom_name.

    Args:
        smiles (str): smiles string, like "CCCC", or "[C:1]NC(=O)" (use num to label
            covalent bond atom.)

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    seed = conformer_seed()

    def target(mol: Mol, q: Queue) -> None:
        """Compute target."""
        try:
            ret = rdDistGeom.EmbedMolecule(mol, randomSeed=seed)
            q.put(ret)
        except Exception as e:  # noqa: BLE001 - worker transfers the error through its result queue
            q.put(e)

    q = queue.Queue()
    t = threading.Thread(target=target, args=(mol, q))
    t.daemon = True
    t.start()

    try:
        ret_code = q.get(timeout=90)
        if isinstance(ret_code, Exception):
            raise ret_code
    except queue.Empty:
        msg = 'Conformer generation timed out.  \
                Please change the "ligand" input format to "CCD_" or "FILE_".'
        raise TimeoutError(msg) from None

    if ret_code != 0:
        # retry with random coords
        ret_code = rdDistGeom.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)

    if ret_code != 0:
        message = f"Conformer generation failed for input SMILES: {smiles}"
        raise ValueError(message)
    return residue_rdkit_mol_to_atom_info(mol)


def residue_build_ligand(entity_info: dict) -> dict:
    """Build a ligand from a ligand entity info dict.

    example1: {
        "ligand": {
          "ligand": "CCD_ATP",
          "count": 1
        }
      },
    example2:{
        "ligand": {
          "ligand": "CCC=O",  # smiles
          "count": 1
        }
      },
    example3:{
        "ion": {
          "ion": "NA",
          "count": 3
        }
      },

    Args:
        entity_info (dict): ligand entity info

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "index_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    ligand_str: str | None = None
    if info := entity_info.get("ion"):
        ccd_code = [info["ion"]]
    elif info := entity_info.get("ligand"):
        ligand_str = info["ligand"]
        if ligand_str is None:
            message = "A ligand must specify CCD codes or SMILES"
            raise ValueError(message)
        ccd_code = ligand_str[4:].split("_") if ligand_str.startswith("CCD_") else None
    else:
        message = "Expected an ion or ligand entity"
        raise ValueError(message)

    atom_info = {}
    if ccd_code is not None:
        atom_array = AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ccd_code):
            ccd_atom_array = ccd.get_component_atom_array(
                ccd_code=code, keep_leaving_atoms=True, keep_hydrogens=False
            )
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        atom_info["atom_array"] = atom_array
        atom_info["atom_array"].res_id[:] = res_ids
    else:
        if ligand_str is None:
            message = "Missing ligand specification"
            raise ValueError(message)
        if ligand_str.startswith("FILE_"):
            lig_file_path = ligand_str[5:]
            atom_info = residue_lig_file_to_atom_info(lig_file_path)
        else:
            atom_info = residue_smiles_to_atom_info(ligand_str)
        atom_info["atom_array"].res_id[:] = 1
    atom_info["atom_array"] = add_reference_features(atom_info["atom_array"])
    # add a fake sequence for ligand, which is used for msa featurizer
    atom_info["sequence"] = "-" * len(atom_info["atom_array"])
    return atom_info


def residue_add_entity_atom_array(single_job_dict: dict) -> dict:
    """Add atom_array to each entity in single_job_dict.

    Args:
        single_job_dict (dict): input job dict

    Returns:
        dict: deepcopy and updated job dict with atom_array

    """
    single_job_dict = copy.deepcopy(single_job_dict)
    sequences = single_job_dict["sequences"]
    single_job_dict["ccd_mols"] = {}
    smiles_ligand_count = 0
    for entity_info in sequences:
        if (
            (info := entity_info.get("proteinChain"))
            or (info := entity_info.get("dnaSequence"))
            or (info := entity_info.get("rnaSequence"))
        ):
            atom_info = residue_build_polymer(entity_info)
        elif info := entity_info.get("ligand"):
            atom_info = residue_build_ligand(entity_info)
            if not info["ligand"].startswith("CCD_"):
                smiles_ligand_count += 1
                if not (smiles_ligand_count <= 99):
                    message = "too many smiles ligands"
                    raise ValueError(message)
                # use lower case res_name (l01, l02, ..., l99) to avoid conflict with
                # CCD code
                atom_info["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
                if "mol" not in atom_info:
                    message = "failed to build rdkit mol from smiles/files"
                    raise ValueError(message)
                single_job_dict["ccd_mols"][f"l{smiles_ligand_count:02d}"] = atom_info[
                    "mol"
                ]
        elif info := entity_info.get("ion"):
            atom_info = residue_build_ligand(entity_info)
        else:
            msg = (
                "entity type must be proteinChain, dnaSequence, rnaSequence, "
                "ligand or ion"
            )
            raise ValueError(msg)
        info.update(atom_info)
    return single_job_dict


def _structural_remove_non_std_ccd_leaving_atoms(atom_array: AtomArray) -> AtomArray:
    """Check polymer connections and remove non-standard leaving atoms.

    Args:
        atom_array (AtomArray): biotite AtomArray

    Returns:
        AtomArray: biotite AtomArray with leaving atoms removed.

    """
    if atom_array.bonds is None:
        message = "Polymer connectivity requires a bond list"
        raise ValueError(message)
    connected = np.zeros(atom_array.res_id[-1], dtype=bool)
    for i, j, _t in atom_array.bonds.as_array():
        if abs(atom_array.res_id[i] - atom_array.res_id[j]) == 1:
            connected[atom_array.res_id[[i, j]].min()] = True

    leaving_atoms = np.zeros(len(atom_array), dtype=bool)
    for res_id, conn in enumerate(connected):
        if res_id == 0 or conn:
            continue

        # Res_id start from 1
        res_name_i = atom_array.res_name[atom_array.res_id == res_id][0]
        res_name_j = atom_array.res_name[atom_array.res_id == res_id + 1][0]
        warnings.warn(
            f"No C-N or O3'-P bond between residue {res_name_i}({res_id}) "
            f"and residue {res_name_j}({res_id + 1}). \n"
            f"all leaving atoms will be removed for both residues.",
            stacklevel=2,
        )
        for idx, res_name in zip(
            [res_id, res_id + 1], [res_name_i, res_name_j], strict=False
        ):
            component = ccd.get_component_atom_array(
                ccd_code=res_name, keep_leaving_atoms=False, keep_hydrogens=False
            )
            if component is None:
                msg = f"CCD component {res_name} could not be parsed"
                raise ValueError(msg)
            staying_atoms = component.atom_name
            if idx == 1 and ccd.get_mol_type(res_name) in ("dna", "rna"):
                staying_atoms = np.append(staying_atoms, ["OP3"])
            if idx == atom_array.res_id[-1] and ccd.get_mol_type(res_name) == "protein":
                staying_atoms = np.append(staying_atoms, ["OXT"])
            leaving_atoms |= (atom_array.res_id == idx) & (
                ~np.isin(atom_array.atom_name, staying_atoms)
            )
    return atom_array[~leaving_atoms]


def _structural_build_polymer_atom_array(ccd_seqs: list[str]) -> AtomArray:
    """Build polymer atom_array from ccd codes, but not remove leaving atoms.

    Args:
        ccd_seqs: a list of ccd code in sequence, ["MET", "ALA"] or ["DA", "DT"]

    Returns:
        AtomArray: Biotite AtomArray of chain with polymer bonds (C-N or O3'-P)

    """
    chain = struc.AtomArray(0)
    for res_id, res_name in enumerate(ccd_seqs):
        # Keep all leaving atoms, will remove leaving atoms later
        residue = ccd.get_component_atom_array(
            ccd_code=res_name, keep_leaving_atoms=True, keep_hydrogens=False
        )
        if residue is None:
            msg = f"CCD component {res_name} could not be parsed"
            raise ValueError(msg)
        residue.res_id[:] = res_id + 1
        chain += residue
    res_starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    polymer_bonds = ccd._connect_inter_residue(chain, res_starts)  # noqa: SLF001 - Biotite extension hook

    if chain.bonds is None:
        chain.bonds = polymer_bonds
    else:
        chain.bonds = chain.bonds.merge(polymer_bonds)

    chain = _add_bonds_to_terminal_residues(chain)

    bond_count = {}
    for i, j, _t in polymer_bonds.as_array():
        bond_count[i] = bond_count.get(i, 0) + 1
        bond_count[j] = bond_count.get(j, 0) + 1

    chain = remove_leaving_atoms(chain, bond_count)

    return _structural_remove_non_std_ccd_leaving_atoms(chain)


def structural_build_polymer(entity_info: dict) -> dict:
    """Build a polymer from a polymer info dict.

    example: {
        "name": "polymer",
        "sequence": "GPDSMEEVVVPEEPPKLVSALATYVQQERLCTMFLSIANKLLPLKP",
        "count": 1
        }

    Args:
        entity_info: Polymer sequence, copies and chemical modifications.
        item (dict): polymer info dict

    Returns:
        dict: {"atom_array": biotite_AtomArray_object}

    """

    def _checked_sequence_index(
        sequence: str,
        position: int,
        position_name: str,
        sequence_name: str,
        modification_type: str,
        modification_name: str,
    ) -> int:
        if not (1 <= position <= len(sequence)):
            msg = (
                f"{position_name} {position} is out of range for "
                f"{sequence_name} sequence of length {len(sequence)} "
                f"({modification_name}={modification_type}). "
                f"Valid range: 1 to {len(sequence)}."
            )
            raise ValueError(msg)
        return position - 1

    poly_type, info = next(iter(entity_info.items()))
    if poly_type == "proteinChain":
        ccd_seqs = [PROTEIN_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                mtype = m["ptmType"]
                index = _checked_sequence_index(
                    sequence=info["sequence"],
                    position=m["ptmPosition"],
                    position_name="ptmPosition",
                    sequence_name=poly_type,
                    modification_type=mtype,
                    modification_name="ptmType",
                )
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    msg = f"unknown modification type: {mtype}"
                    raise ValueError(msg)
        if glycans := info.get("glycans"):
            logger.warning("glycans not supported: %s", f"{glycans}")
        chain_array = _structural_build_polymer_atom_array(ccd_seqs)

    elif poly_type in ("dnaSequence", "rnaSequence"):
        map_1to3 = DNA_1to3 if poly_type == "dnaSequence" else RNA_1to3
        ccd_seqs = [map_1to3[x] for x in info["sequence"]]
        if modifications := info.get("modifications"):
            for m in modifications:
                mtype = m["modificationType"]
                index = _checked_sequence_index(
                    sequence=info["sequence"],
                    position=m["basePosition"],
                    position_name="basePosition",
                    sequence_name=poly_type,
                    modification_type=mtype,
                    modification_name="modificationType",
                )
                if mtype.startswith("CCD_"):
                    ccd_seqs[index] = mtype[4:]
                else:
                    msg = f"unknown modification type: {mtype}"
                    raise ValueError(msg)
        chain_array = _structural_build_polymer_atom_array(ccd_seqs)

    else:
        msg = "polymer type must be proteinChain, dnaSequence or rnaSequence"
        raise ValueError(msg)
    chain_array = add_reference_features(chain_array)
    return {"atom_array": chain_array}


def structural_rdkit_mol_to_atom_array(
    mol: Chem.Mol,
    *,
    removeHs: bool = True,  # noqa: N803 - checkpoint-compatible keyword
) -> AtomArray:
    """Convert rdkit mol to biotite AtomArray.

    Args:
        mol (Chem.Mol): rdkit mol
        removeHs (bool): whether to remove hydrogens in atom_array

    Returns:
        AtomArray: biotite AtomArray

    """
    atom_count = mol.GetNumAtoms()
    atom_array = AtomArray(atom_count)
    atom_array.hetero[:] = True
    atom_array.res_name[:] = "UNL"
    atom_array.add_annotation("charge", int)

    conf = mol.GetConformer()
    coord = conf.GetPositions()

    element_count = Counter()
    for i, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"

        atom.SetProp("name", atom_name)

        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = element
        atom_array.charge[i] = atom.GetFormalCharge()
        atom_array.coord[i, :] = coord[i, :]

    bonds = []
    bonds.extend(
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()
    )
    atom_array.bonds = cast("Any", struc).BondList(atom_count, np.array(bonds))
    if removeHs:
        atom_array = atom_array[atom_array.element != "H"]
    return atom_array


def structural_rdkit_mol_to_atom_info(mol: Chem.Mol) -> dict[str, Any]:
    """Convert RDKit Mol to atom_info dict.

    Args:
        mol (Chem.Mol): rdkit mol

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    atom_info = {}
    atom_map_to_atom_name = {}
    atom_idx_to_atom_name = {}

    element_count = Counter()
    for atom in mol.GetAtoms():
        element = atom.GetSymbol().upper()
        element_count[element] += 1
        atom_name = f"{element}{element_count[element]}"
        atom.SetProp("name", atom_name)
        if atom.GetAtomMapNum() != 0:
            atom_map_to_atom_name[atom.GetAtomMapNum()] = atom_name
        atom_idx_to_atom_name[atom.GetIdx()] = atom_name

    if atom_map_to_atom_name:
        # Atom map for input SMILES
        atom_info["atom_map_to_atom_name"] = atom_map_to_atom_name
    else:
        # Atom index for input file
        atom_info["atom_map_to_atom_name"] = atom_idx_to_atom_name

    # Atom_array without hydrogens
    atom_info["atom_array"] = structural_rdkit_mol_to_atom_array(mol=mol, removeHs=True)
    return atom_info


def structural_lig_file_to_atom_info(lig_file_path: str) -> dict[str, Any]:
    """Convert ligand file to biotite AtomArray.

    Args:
        lig_file_path (str): ligand file path with one of the following suffixes: [mol,
            mol2, sdf, pdb]

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    if lig_file_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(lig_file_path)
    elif lig_file_path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(lig_file_path)
        mol = next(suppl)
    elif lig_file_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(lig_file_path)
    elif lig_file_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(lig_file_path)
    else:
        msg = f"Invalid ligand file type: .{lig_file_path.rsplit('.', maxsplit=1)[-1]}"
        raise ValueError(msg)
    if not (mol is not None):
        message = (
            f"Failed to retrieve molecule from file, "
            f"invalid ligand file: {lig_file_path}. "
            "Please provide one of: mol, mol2, sdf, pdb."
        )
        raise ValueError(message)

    if not (mol.GetConformer().Is3D()):
        message = f"3D conformer not found in ligand file: {lig_file_path}"
        raise ValueError(message)
    return structural_rdkit_mol_to_atom_info(mol)


def structural_smiles_to_atom_info(smiles: str) -> dict:
    """Convert smiles to atom_array, and atom_map_to_atom_name.

    Args:
        smiles (str): smiles string, like "CCCC", or "[C:1]NC(=O)" (use num to label
            covalent bond atom.)

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "atom_map_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    seed = conformer_seed()
    embed_molecule = rdDistGeom.EmbedMolecule

    def target(mol: Mol, q: Queue) -> None:
        """Compute target."""
        try:
            ret = embed_molecule(mol, randomSeed=seed)
            q.put(ret)
        except Exception as e:  # noqa: BLE001 - worker transfers the error through its result queue
            q.put(e)

    q = queue.Queue()
    t = threading.Thread(target=target, args=(mol, q))
    t.daemon = True
    t.start()

    try:
        ret_code = q.get(timeout=90)
        if isinstance(ret_code, Exception):
            raise ret_code
    except queue.Empty:
        msg = 'Conformer generation timed out.  \
                Please change the "ligand" input format to "CCD_" or "FILE_".'
        raise TimeoutError(msg) from None

    if ret_code != 0:
        # retry with random coords
        ret_code = embed_molecule(mol, useRandomCoords=True, randomSeed=seed)

    if ret_code != 0:
        message = f"Conformer generation failed for input SMILES: {smiles}"
        raise ValueError(message)
    return structural_rdkit_mol_to_atom_info(mol)


def structural_build_ligand(entity_info: dict) -> dict:
    """Build a ligand from a ligand entity info dict.

    example1: {
        "ligand": {
          "ligand": "CCD_ATP",
          "count": 1
        }
      },
    example2:{
        "ligand": {
          "ligand": "CCC=O",  # smiles
          "count": 1
        }
      },
    example3:{
        "ion": {
          "ion": "NA",
          "count": 3
        }
      },

    Args:
        entity_info (dict): ligand entity info

    Returns:
        dict: info of atoms
        example: {
            "atom_array": biotite_AtomArray_object,
            "index_to_atom_name": {1: "C2"}, # only for smiles
            }

    """
    ccd_code: list[str] | None = None
    ligand_str: str | None = None
    if info := entity_info.get("ion"):
        ccd_code = [info["ion"]]
    elif info := entity_info.get("ligand"):
        ligand_str = info["ligand"]
        if ligand_str is None:
            message = "A ligand must specify CCD codes or SMILES"
            raise ValueError(message)
        if ligand_str.startswith("CCD_"):
            ccd_code = ligand_str[4:].split("_")
    else:
        msg = "entity_info must contain either 'ion' or 'ligand'"
        raise ValueError(msg)

    atom_info = {}
    if ccd_code is not None:
        atom_array = AtomArray(0)
        res_ids = []
        for idx, code in enumerate(ccd_code):
            ccd_atom_array = ccd.get_component_atom_array(
                ccd_code=code, keep_leaving_atoms=True, keep_hydrogens=False
            )
            if ccd_atom_array is None:
                msg = f"CCD component {code} could not be parsed"
                raise ValueError(msg)
            atom_array += ccd_atom_array
            res_id = idx + 1
            res_ids += [res_id] * len(ccd_atom_array)
        atom_info["atom_array"] = atom_array
        atom_info["atom_array"].res_id[:] = res_ids
    else:
        if not (ligand_str is not None):
            message = "Invalid state: ligand_str is not None"
            raise ValueError(message)
        if ligand_str.startswith("FILE_"):
            lig_file_path = ligand_str[5:]
            atom_info = structural_lig_file_to_atom_info(lig_file_path)
        else:
            atom_info = structural_smiles_to_atom_info(ligand_str)
        atom_info["atom_array"].res_id[:] = 1
    atom_info["atom_array"] = add_reference_features(atom_info["atom_array"])
    # add a fake sequence for ligand, which is used for msa featurizer
    atom_info["sequence"] = "-" * len(atom_info["atom_array"])
    return atom_info


def structural_add_entity_atom_array(single_job_dict: dict) -> dict:
    """Add atom_array to each entity in single_job_dict.

    Args:
        single_job_dict (dict): input job dict

    Returns:
        dict: deepcopy and updated job dict with atom_array

    """
    single_job_dict = copy.deepcopy(single_job_dict)
    sequences = single_job_dict["sequences"]
    smiles_ligand_count = 0
    for entity_info in sequences:
        if (
            (info := entity_info.get("proteinChain"))
            or (info := entity_info.get("dnaSequence"))
            or (info := entity_info.get("rnaSequence"))
        ):
            atom_info = structural_build_polymer(entity_info)
        elif info := entity_info.get("ligand"):
            atom_info = structural_build_ligand(entity_info)
            if not info["ligand"].startswith("CCD_"):
                smiles_ligand_count += 1
                if not (smiles_ligand_count <= 99):
                    message = "too many smiles ligands"
                    raise ValueError(message)
                # use lower case res_name (l01, l02, ..., l99) to avoid conflict with
                # CCD code
                atom_info["atom_array"].res_name[:] = f"l{smiles_ligand_count:02d}"
        elif info := entity_info.get("ion"):
            atom_info = structural_build_ligand(entity_info)
        else:
            msg = (
                "entity type must be proteinChain, dnaSequence, rnaSequence, "
                "ligand or ion"
            )
            raise ValueError(msg)
        info.update(atom_info)
    return single_job_dict
