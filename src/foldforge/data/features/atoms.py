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
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import functools
import itertools
import re
from collections import defaultdict
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray, get_residue_starts
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile

from foldforge.data.bonds import replace_bond_array as _replace_bond_array
from foldforge.data.ccd.components import biotite_load_ccd_cif
from foldforge.data.constants.chemistry import (
    DNA_STD_RESIDUES,
    PRO_STD_RESIDUES,
    RNA_STD_RESIDUES,
    STD_RESIDUES,
)
from foldforge.data.io import load_gzip_pickle, read_indices_csv
from foldforge.models.config.protenix.configs.configs_data import data_configs
from foldforge.utils.logging import get_logger

if TYPE_CHECKING:
    from foldforge.data.features.tokenizer import TokenArray


def get_antibody_clusters() -> set[str]:
    """Get the top 2 clusters of antibodies from the PDB cluster file.

    Returns:
        set[str]: A set of PDB IDs (lower case) in the top 2 clusters.

    """
    pdb_cluster_file = data_configs["pdb_cluster_file"]
    try:
        with Path(pdb_cluster_file).open() as f:
            lines = f.readlines()
    except FileNotFoundError:
        msg = (
            f"The file {pdb_cluster_file} does not exist. \n"
            "Downloading it from "
            "https://af3-dev.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt"
        )
        raise FileNotFoundError(msg) from None

    cluster_list = [line.strip().split() for line in lines]
    return set(
        [i.lower() for i in cluster_list[0]] + [i.lower() for i in cluster_list[1]]
    )


def residue_get_atom_mask_by_name(
    atom_array: AtomArray,
    entity_id: int | None = None,
    position: int | None = None,
    atom_name: str | None = None,
    copy_id: int | None = None,
) -> np.ndarray:
    """Get the atom mask of atoms with specific identifiers.

    Args:
        atom_array (AtomArray): Biotite Atom array.
        entity_id (int): Entity id.
        position (int): Residue index of the atom.
        atom_name (str): Atom name.
        copy_id (copy_id): A asym chain id in N copies of an entity.

    Returns:
        np.ndarray: Array of a bool mask.

    """
    mask = np.ones(atom_array.shape, dtype=np.bool_)

    if entity_id is not None:
        mask &= atom_array.label_entity_id == str(entity_id)
    if position is not None:
        mask &= atom_array.res_id == int(position)
    if atom_name is not None:
        mask &= atom_array.atom_name == str(atom_name)
    if copy_id is not None:
        mask &= atom_array.copy_id == int(copy_id)
    return mask


def remove_numbers(s: str) -> str:
    """Remove numbers from a string.

    Args:
        s (str): input string

    Returns:
        str: a string with numbers removed.

    """
    return re.sub(r"\d+", "", s)


def int_to_letters(n: int) -> str:
    """Convert int to letters.

    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB

    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def replace_elem_by_mapping_dict(input_array: np.ndarray, mapping: dict) -> np.ndarray:
    """Replace each element in input_array with the value in mapping.

    Args:
        input_array: np.ndarray
        mapping: a mapping dict

    Returns:
        np.ndarray: a new array with the same shape as input_array.

    """
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    sidx = k.argsort()  # k,v from approach #1

    k = k[sidx]
    v = v[sidx]

    idx = np.searchsorted(k, input_array.ravel()).reshape(input_array.shape)
    idx[idx == len(k)] = 0
    mask = k[idx] == input_array
    return np.where(mask, v[idx], 0)


@functools.lru_cache
def parse_pdb_cluster_file_to_dict(
    cluster_file: str, *, remove_uniprot: bool = True
) -> dict[str, tuple]:
    """Parse PDB cluster file, and return a pandas dataframe.

    example cluster file:
    https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt

    Args:
        remove_uniprot: Whether to exclude UniProt records from clustering.
        cluster_file (str): cluster_file path
    Returns:
        dict(str, tuple(str, str)): {pdb_id}_{entity_id} --> [cluster_id, cluster_size]

    """
    # Use the distributed wwPDB/UniProt clustering assignments.
    pdb_cluster_dict = {}
    with Path(cluster_file).open() as f:
        for line in f:
            pdb_clusters = []
            for ids in line.strip().split():
                if remove_uniprot and ids.startswith(("AF_", "MA_")):
                    continue
                pdb_clusters.append(ids)
            cluster_size = len(pdb_clusters)
            if cluster_size == 0:
                continue
            # use first member as cluster id.
            cluster_id = f"pdb_cluster_{pdb_clusters[0]}"
            for ids in pdb_clusters:
                pdb_cluster_dict[ids.lower()] = (cluster_id, cluster_size)
    return pdb_cluster_dict


@functools.lru_cache
def get_obsolete_dict() -> dict[str, str]:
    """Get a obsolete dict of pdb_id: original release date.

    Some PDB IDs have been deprecated. The release date documented in the mmCIF
    reflects the updated information,
    whereas the structure it describes had actually appeared previously.

    Returns:
        dict[str, str]: {pdb_id: first release date}

    """
    obsolete_release_date_csv = data_configs.get("obsolete_release_data_csv", None)
    if obsolete_release_date_csv is None:
        return {}
    if Path(obsolete_release_date_csv).exists():
        df = read_indices_csv(obsolete_release_date_csv)
        return dict(zip(df["successor_id"], df["release_date"], strict=False))
    return {}


def within_dist(
    query_atoms: AtomArray, env_atoms: AtomArray, distance: float
) -> np.ndarray:
    """Get mask of env_atoms within distance to query_atoms.

    Args:
        env_atoms: Environment atoms used for the distance cutoff.
        query_atoms (AtomArray): query_atoms, query_atoms not in env_atoms
        env_atoms (AtomArray):
        distance (float): within distance (Å), eg: 5

    Returns:
        array(bool): a mask of env_atoms, length = len(env_atoms)

    """
    cell_list = struc.CellList(env_atoms, cell_size=5)
    mask_2d = cell_list.get_atoms(query_atoms.coord, radius=distance, as_mask=True)
    return mask_2d.any(axis=0)


def residue_get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """Get inter residue bonds by checking chain_id and res_id.

    Args:
        atom_array (AtomArray): Biotite AtomArray, must have chain_id and res_id

    Returns:
        np.ndarray: inter residue bonds, shape = (n,2)

    """
    if atom_array.bonds is None:
        return np.empty((0, 2), dtype=int)
    idx_i = atom_array.bonds.as_array()[:, 0]
    idx_j = atom_array.bonds.as_array()[:, 1]
    chain_id_diff = atom_array.chain_id[idx_i] != atom_array.chain_id[idx_j]
    res_id_diff = atom_array.res_id[idx_i] != atom_array.res_id[idx_j]
    diff_mask = chain_id_diff | res_id_diff
    inter_residue_bonds = atom_array.bonds.as_array()[diff_mask]
    return inter_residue_bonds[:, :2]  # remove bond type


def get_starts_by(
    atom_array: AtomArray, by_annot: str, *, add_exclusive_stop: bool = False
) -> np.ndarray:
    """Get start indices by given annotation in an AtomArray.

    Args:
        atom_array (AtomArray): Biotite AtomArray
        by_annot (str): annotation to group by, eg: 'chain_id', 'res_id', 'res_name'
        add_exclusive_stop (bool, optional): add exclusive stop (len(atom_array)).
            Defaults to False.

    Returns:
        np.ndarray: start indices of each group, shape = (n,), eg: [0, 10, 20, 30, 40]

    """
    annot = getattr(atom_array, by_annot)
    # If annotation change, a new start
    annot_change_mask = annot[1:] != annot[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new residue
    starts = np.where(annot_change_mask)[0] + 1

    # The first start is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], starts, [atom_array.array_length()]))
    return np.concatenate(([0], starts))


def atom_select(
    atom_array: AtomArray, select_dict: dict, *, as_mask: bool = False
) -> np.ndarray:
    """Return index of atom_array that match select_dict.

    Args:
        atom_array (AtomArray): Biotite AtomArray
        select_dict (dict): select dict, eg: {'element': 'C'}
        as_mask (bool, optional): return mask of atom_array. Defaults to False.

    Returns:
        np.ndarray: index of atom_array that match select_dict

    """
    mask = np.ones(len(atom_array), dtype=bool)
    for k, v in select_dict.items():
        mask = mask & (getattr(atom_array, k) == v)
    if as_mask:
        return mask
    return np.where(mask)[0]


def get_polymer_polymer_bond(
    atom_array: AtomArray, entity_poly_type: dict[str, str]
) -> np.ndarray:
    """Get bonds between the bonded polymer and its parent chain.

    Args:
        atom_array (AtomArray): biotite atom array object.
        entity_poly_type (dict[str, str]): A dict of entity id to entity poly type.

    Returns:
        np.ndarray: bond records between the bonded polymer and its parent chain.
                    e.g. np.array([[atom1, atom2, bond_order]...])

    """
    # identify polymer by mol_type (protein, rna, dna, ligand)
    polymer_entities = []
    for k, v in entity_poly_type.items():
        if v in ["polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide"]:
            polymer_entities.append(k)

    poly_arr = atom_array[np.isin(atom_array.label_entity_id, polymer_entities)]
    if poly_arr.bonds is None:
        return np.empty((0, 3), dtype=int)
    bond_arr = poly_arr.bonds.as_array()
    res_id = poly_arr.res_id
    chain_id = poly_arr.chain_id

    cid_equil = chain_id[bond_arr[:, 0]] == chain_id[bond_arr[:, 1]]
    res_id_diff = res_id[bond_arr[:, 0]] - res_id[bond_arr[:, 1]]
    intra_chain_bond_mask = cid_equil & (np.abs(res_id_diff) > 1)
    polymer_polymer_bond = bond_arr[(~cid_equil) | intra_chain_bond_mask]

    if polymer_polymer_bond.size == 0:
        # no polymer-polymer bonds
        polymer_polymer_bond = np.empty((0, 3)).astype(int)

    # np.array([[atom1, atom2, bond_order]...])
    return polymer_polymer_bond


def get_ligand_polymer_bond_mask(
    atom_array: AtomArray, *, lig_include_ions: bool = False
) -> np.ndarray:
    """Ref AlphaFold3 SI Chapter 3.7.1.

    Get bonds between the bonded ligand and its parent chain.

    Args:
        atom_array (AtomArray): biotite atom array object.
        lig_include_ions (bool): whether to include ions in the ligand.

    Returns:
        np.ndarray: bond records between the bonded ligand and its parent chain.
                    e.g. np.array([[atom1, atom2, bond_order]...])

    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(
            atom_array.label_asym_id, return_counts=True
        )
        chain_id_to_count_map = dict(zip(unique_chain_id, counts, strict=False))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    # identify polymer by mol_type (protein, rna, dna, ligand)
    polymer_mask = np.isin(atom_array.mol_type, ["protein", "rna", "dna"])

    if atom_array.bonds is None:
        return np.empty((0, 3), dtype=int)
    idx_i = atom_array.bonds.as_array()[:, 0]
    idx_j = atom_array.bonds.as_array()[:, 1]

    lig_polymer_bond_indices = np.where(
        (lig_mask[idx_i] & polymer_mask[idx_j])
        | (lig_mask[idx_j] & polymer_mask[idx_i])
    )[0]
    if lig_polymer_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds.as_array()[
            lig_polymer_bond_indices
        ]  # np.array([[atom1, atom2, bond_order]...])
    return lig_polymer_bonds


def get_clean_data(atom_array: AtomArray) -> AtomArray:
    """Remove unresolved atoms from the AtomArray.

    Args:
        atom_array (AtomArray): The input AtomArray containing atoms.

    Returns:
        AtomArray: A new AtomArray with unresolved atoms removed.

    """
    atom_array.copy()
    return atom_array[atom_array.is_resolved]


def residue_save_atoms_to_cif(
    output_cif_file: str,
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    pdb_id: str,
) -> None:
    """Save atom array data to a CIF file.

    Args:
        output_cif_file (str): The output path for saving the atom array in CIF format.
        atom_array (AtomArray): The atom array to be saved.
        entity_poly_type: The entity poly type information.
        pdb_id: The PDB ID for the entry.

    """
    cifwriter = ResidueCIFWriter(atom_array, entity_poly_type)
    cifwriter.save_to_cif(
        output_path=output_cif_file,
        entry_id=pdb_id,
        include_bonds=False,
    )


def get_raw_atom_array(
    bioassembly_dict_fpath: str | Path,
) -> tuple[AtomArray, dict[str, str]]:
    """Load raw atom array and entity poly type from a bioassembly pickle file.

    Args:
        bioassembly_dict_fpath (Union[str, Path]): Path to the bioassembly pickle file.

    Returns:
        tuple[AtomArray, dict[str, str]]: A tuple containing the atom array and entity
        poly type dict.

    """
    bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)
    atom_array = bioassembly_dict["atom_array"]
    entity_poly_type = bioassembly_dict["entity_poly_type"]
    # Note: 'charge' info isn't used but causes OST DockQ parser errors. Temp set to 0
    # for compatibility.
    atom_array.charge = np.zeros(len(atom_array.charge))
    return atom_array, entity_poly_type


def residue_save_structure_cif(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
    *,
    save_wo_unresolved: bool = True,
) -> None:
    """Save the predicted structure to a CIF file.

    Args:
        atom_array (AtomArray): The original AtomArray containing the structure.
        pred_coordinate (torch.Tensor): The predicted coordinates for the structure.
        output_fpath (str): The output file path for saving the CIF file.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
        save_wo_unresolved (bool): Whether to save a version without unresolved atoms.
            Defaults to True.

    """
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    residue_save_atoms_to_cif(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved") and save_wo_unresolved:
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        residue_save_atoms_to_cif(
            output_fpath.replace(".cif", "_wounresol.cif"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


def get_lig_lig_bonds(
    atom_array: AtomArray, *, lig_include_ions: bool = False
) -> np.ndarray:
    """Get all inter-ligand bonds in order to create "token_bonds".

    Args:
        atom_array (AtomArray): biotite AtomArray object with "mol_type" attribute.
        lig_include_ions (bool, optional): . Defaults to False.

    Returns:
        np.ndarray: inter-ligand bonds, e.g. np.array([[atom1, atom2, bond_order]...])

    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(
            atom_array.label_asym_id, return_counts=True
        )
        chain_id_to_count_map = dict(zip(unique_chain_id, counts, strict=False))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    chain_res_id = np.vstack((atom_array.label_asym_id, atom_array.res_id)).T
    if atom_array.bonds is None:
        return np.empty((0, 3), dtype=int)
    idx_i = atom_array.bonds.as_array()[:, 0]
    idx_j = atom_array.bonds.as_array()[:, 1]

    ligand_ligand_bond_indices = np.where(
        (lig_mask[idx_i] & lig_mask[idx_j])
        & np.any(chain_res_id[idx_i] != chain_res_id[idx_j], axis=1)
    )[0]

    if ligand_ligand_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds.as_array()[ligand_ligand_bond_indices]
    return lig_polymer_bonds


def superimpose(
    fixed: AtomArray,
    mobile: AtomArray,
    fixed_asym_id: str,
    mobile_asym_id: str,
) -> tuple[AtomArray, struc.AffineTransformation]:
    """Superimpose mobile atom_array onto fixed atom_array based on label_asym_id.

    if fixed atom_array have attr `is_resolved==True`, only these atoms are used for
    superimposition.

    Args:
        fixed (AtomArray): fixed atom_array
        mobile (AtomArray): mobile atom_array
        fixed_asym_id (str): label_asym_id of fixed atom_array
        mobile_asym_id (str): label_asym_id of mobile atom_array

    Returns:
        AtomArray: superimposed mobile atom_array
        AffineTransformation:This object contains the affine transformation(s) that
        were
        applied on `mobile`.
        `AffineTransformation.apply()` can be used to transform
        another AtomArray in the same way.

    """
    if not (hasattr(fixed, "label_asym_id")):
        message = "fixed atom_array must have label_asym_id"
        raise ValueError(message)
    if not (hasattr(mobile, "label_asym_id")):
        message = "mobile atom_array must have label_asym_id"
        raise ValueError(message)

    if not (np.any(fixed.label_asym_id == fixed_asym_id)):
        message = f"{fixed_asym_id=} not in fixed!"
        raise ValueError(message)
    if not (np.any(mobile.label_asym_id == mobile_asym_id)):
        message = f"{mobile_asym_id=} not in mobile!"
        raise ValueError(message)
    fixed_filtered = fixed[fixed.label_asym_id == fixed_asym_id]
    mobile_filtered = mobile[mobile.label_asym_id == mobile_asym_id]

    if len(fixed_filtered) != len(mobile_filtered):
        message = f"{len(fixed_filtered)=}, {len(mobile_filtered)=}"
        raise ValueError(message)
    diff_idx = np.where(fixed_filtered.atom_name != mobile_filtered.atom_name)[0]
    if not ((fixed_filtered.atom_name == mobile_filtered.atom_name).all()):
        message = (
            f"atom_name between aligned chains must "
            f"be the same:\n {fixed_filtered[diff_idx]}\n {mobile_filtered[diff_idx]} "
        )
        raise ValueError(message)

    if hasattr(fixed_filtered, "is_resolved"):
        mobile_filtered = mobile_filtered[fixed_filtered.is_resolved]
        fixed_filtered = fixed_filtered[fixed_filtered.is_resolved]

    _fitted, transform = struc.superimpose(fixed_filtered, mobile_filtered)

    return transform.apply(mobile), transform


class _CommonCIFWriter:
    """Represent common c i f writer."""

    atom_array: AtomArray
    entity_poly_type: dict[str, str] | None

    def _get_entity_block(self) -> pdbx.CIFCategory:
        if self.entity_poly_type is None:
            return pdbx.CIFCategory()
        entity_ids_in_atom_array = np.sort(np.unique(self.atom_array.label_entity_id))
        entity_block_dict = defaultdict(list)
        for entity_id in entity_ids_in_atom_array:
            if entity_id not in self.entity_poly_type:
                entity_type = "non-polymer"
            else:
                entity_type = "polymer"
            entity_block_dict["id"].append(entity_id)
            entity_block_dict["pdbx_description"].append(".")
            entity_block_dict["type"].append(entity_type)
        return pdbx.CIFCategory(entity_block_dict)

    @staticmethod
    def get_entity_poly_and_entity_poly_seq_block(
        entity_poly_type: dict[str, str], atom_array: AtomArray
    ) -> dict[str, pdbx.CIFCategory]:
        """Return entity poly and entity poly seq block."""
        entity_poly = defaultdict(list)
        for entity_id, entity_type in entity_poly_type.items():
            label_asym_ids = np.unique(
                atom_array.label_asym_id[atom_array.label_entity_id == entity_id]
            )
            label_asym_ids_str = ",".join(label_asym_ids)

            if label_asym_ids_str == "":
                # The entity not in current atom_array
                continue

            entity_poly["entity_id"].append(entity_id)
            entity_poly["pdbx_strand_id"].append(label_asym_ids_str)
            entity_poly["type"].append(entity_type)

        if not entity_poly:
            return {}

        entity_poly_seq = defaultdict(list)
        for _entity_id, label_asym_ids_str in zip(
            entity_poly["entity_id"], entity_poly["pdbx_strand_id"], strict=False
        ):
            first_label_asym_id = label_asym_ids_str.split(",")[0]
            first_asym_chain = atom_array[
                atom_array.label_asym_id == first_label_asym_id
            ]
            chain_starts = struc.get_chain_starts(
                first_asym_chain, add_exclusive_stop=True
            )
            asym_chain = first_asym_chain[
                chain_starts[0] : chain_starts[1]
            ]  # ensure the asym chain is a single chain

            res_starts = struc.get_residue_starts(asym_chain, add_exclusive_stop=False)
            asym_chain_entity_id = asym_chain[res_starts].label_entity_id.tolist()
            asym_chain_hetero = [
                "n" if not i else "y" for i in asym_chain[res_starts].hetero
            ]
            asym_chain_res_name = asym_chain[res_starts].res_name.tolist()
            asym_chain_res_id = asym_chain[res_starts].res_id.tolist()

            entity_poly_seq["entity_id"].extend(asym_chain_entity_id)
            entity_poly_seq["hetero"].extend(asym_chain_hetero)
            entity_poly_seq["mon_id"].extend(asym_chain_res_name)
            entity_poly_seq["num"].extend(asym_chain_res_id)

        return {
            "entity_poly": pdbx.CIFCategory(entity_poly),
            "entity_poly_seq": pdbx.CIFCategory(entity_poly_seq),
        }

    def _get_chem_comp_block(self) -> pdbx.CIFCategory:
        ccd_cif = biotite_load_ccd_cif()
        all_ccd = np.unique(self.atom_array.res_name)

        chem_comp = defaultdict(list)
        chem_comp_field = [
            "id",
            "type",
            "mon_nstd_flag",
            "name",
            "pdbx_synonyms",
            "formula",
            "formula_weight",
        ]
        for ccd in all_ccd:
            if ccd not in ccd_cif:
                chem_comp["id"].append(ccd)
                chem_comp["type"].append("?")
                chem_comp["name"].append("?")
                chem_comp["mon_nstd_flag"].append("n")
                chem_comp["pdbx_synonyms"].append("?")
                chem_comp["formula"].append("?")
                chem_comp["formula_weight"].append("?")
            else:
                for i in chem_comp_field:
                    if i == "mon_nstd_flag":
                        if ccd in STD_RESIDUES and ccd not in ["N", "DN", "UNK"]:
                            mon_nstd_flag = "y"
                        elif (
                            ccd_cif[ccd]["chem_comp"]["type"].as_item() == "non-polymer"
                        ):
                            mon_nstd_flag = "."
                        else:
                            mon_nstd_flag = "n"
                        chem_comp[i].append(mon_nstd_flag)
                    else:
                        chem_comp[i].append(ccd_cif[ccd]["chem_comp"][i].as_item())
        return pdbx.CIFCategory(chem_comp)


class ResidueCIFWriter(_CommonCIFWriter):
    """Write AtomArray to cif.

    Args:
        atom_array (AtomArray): Biotite AtomArray object.
        entity_poly_type (dict[str, str], optional): A dict of label_entity_id to
            entity_poly_type. Defaults to None.
                                                     If None, "the entity_poly" and
                                                     "entity_poly_seq" will not be
                                                     written to the cif.
        atom_array_output_mask (np.ndarray, optional): A mask of atom_array. Defaults
            to None.
                                                      If None, all atoms will be
                                                      written to the cif.

    """

    def __init__(
        self,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str] | None = None,
        atom_array_output_mask: np.ndarray | None = None,
    ) -> None:
        self.atom_array = copy.deepcopy(atom_array)
        self.entity_poly_type = entity_poly_type
        self.atom_array_output_mask = atom_array_output_mask

    def _get_unresolved_block(self) -> pdbx.CIFCategory | None:
        if self.atom_array_output_mask is None or self.entity_poly_type is None:
            return None
        res_starts = get_residue_starts(self.atom_array, add_exclusive_stop=True)
        is_res_starts = np.zeros(len(self.atom_array_output_mask), dtype=bool)
        for start, stop in itertools.pairwise(res_starts):
            if not any(self.atom_array.is_resolved[start:stop]):
                is_res_starts[start] = True

        mask = (~self.atom_array_output_mask) & is_res_starts
        if not np.any(mask):
            # No unresolved atoms
            return None
        polymer_flag_bool = np.isin(
            self.atom_array.label_entity_id[mask], list(self.entity_poly_type.keys())
        )
        polymer_flag = ["Y" if i else "N" for i in polymer_flag_bool]

        unresolved_block: dict[str, np.ndarray | Sequence[str]] = {}
        unresolved_block["id"] = np.arange(mask.sum()) + 1
        unresolved_block["PDB_model_num"] = np.ones(mask.sum(), dtype=int)
        unresolved_block["polymer_flag"] = polymer_flag
        unresolved_block["occupancy_flag"] = np.ones(mask.sum(), dtype=int)
        unresolved_block["auth_asym_id"] = self.atom_array.chain_id[mask]
        unresolved_block["auth_comp_id"] = self.atom_array.res_name[mask]
        unresolved_block["auth_seq_id"] = self.atom_array.res_id[mask]
        unresolved_block["PDB_ins_code"] = ["?"] * mask.sum()
        unresolved_block["label_asym_id"] = self.atom_array.chain_id[mask]
        unresolved_block["label_comp_id"] = self.atom_array.res_name[mask]
        unresolved_block["label_seq_id"] = self.atom_array.res_id[mask]
        return pdbx.CIFCategory(unresolved_block)

    def save_to_cif(
        self,
        output_path: str | Path,
        entry_id: str | None = None,
        *,
        include_bonds: bool = False,
    ) -> None:
        """Save AtomArray to cif.

        Args:
            output_path (str): Output path of cif file.
            entry_id (str, optional): The value of "_entry.id" in cif. Defaults to
                None.
                                      If None, the entry_id will be the basename of
                                      output_path (without ".cif" extension).
            include_bonds (bool, optional): Whether to include  bonds in the cif.
                Defaults to False.
                                            If set to True and `array` has associated
                                            ``bonds`` , the
                                            intra-residue bonds will be written into
                                            the ``chem_comp_bond``
                                            category.
                                            Inter-residue bonds will be written into
                                            the ``struct_conn``
                                            independent of this parameter.

        """
        if entry_id is None:
            entry_id = Path(output_path).name.replace(".cif", "")

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        block_dict["chem_comp"] = self._get_chem_comp_block()

        if self.entity_poly_type:
            block_dict["entity"] = self._get_entity_block()
            block_dict.update(
                ResidueCIFWriter.get_entity_poly_and_entity_poly_seq_block(
                    self.entity_poly_type, self.atom_array
                )
            )

        if self.atom_array_output_mask is not None:
            unresolved_block = self._get_unresolved_block()
            if unresolved_block is not None:
                block_dict["pdbx_unobs_or_zero_occ_residues"] = unresolved_block

        block = pdbx.CIFBlock()
        for name, category in block_dict.items():
            block[name] = category
        cif = pdbx.CIFFile(
            {
                Path(output_path).name.replace(".cif", "")
                + "_predicted_by_protenix": block
            }
        )

        if self.atom_array_output_mask is not None:
            atom_array = self.atom_array[self.atom_array_output_mask]
        else:
            atom_array = self.atom_array

        if not include_bonds and atom_array.bonds is not None:
            # https://github.com/biotite-dev/biotite/pull/804
            inter_bonds = pdbx.convert._filter_bonds(atom_array, "inter")  # noqa: SLF001 - Biotite extension hook
            atom_array.bonds = _replace_bond_array(atom_array.bonds, inter_bonds)

        pdbx.set_structure(cif, atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block["atom_site"]

        occ = atom_site.get("occupancy")
        if occ is None:
            atom_site["occupancy"] = np.ones(len(atom_array), dtype=float)

        b_factor = atom_site.get("B_iso_or_equiv")
        if b_factor is None:
            atom_site["B_iso_or_equiv"] = np.round(
                np.zeros(len(atom_array), dtype=float), 2
            ).astype(str)

        if "label_entity_id" in atom_array.get_annotation_categories():
            atom_site["label_entity_id"] = atom_array.label_entity_id
        cif.write(output_path)

    @staticmethod
    def to_mmcif_string(
        atom_array: AtomArray,
        entity_poly_type: dict[str, str] | None = None,
        entry_id: str | None = None,
        *,
        include_bonds: bool = False,
    ) -> str:
        """Convert the AtomArray to an mmCIF format string.

        Parameters
        ----------
        atom_array : AtomArray
            The structure to convert.
        entity_poly_type : dict[str, str], optional
            A dictionary mapping entity IDs to their polymer types.
        entry_id : str, optional
            The value of "_entry.id" in the CIF.
        include_bonds : bool, optional
            Whether to include bonds in the output. Default is False.

        Returns
        -------
        str
            The mmCIF-formatted string representation of the structure.

        Raises
        ------
        ValueError
            If entry_id is not provided.

        """
        if entry_id is None:
            msg = "entry_id must be provided."
            raise ValueError(msg)

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        if entity_poly_type:
            block_dict.update(
                ResidueCIFWriter.get_entity_poly_and_entity_poly_seq_block(
                    atom_array=atom_array, entity_poly_type=entity_poly_type
                )
            )

        block = pdbx.CIFBlock()
        for name, category in block_dict.items():
            block[name] = category
        cif = pdbx.CIFFile({entry_id: block})  # Use a default block name
        pdbx.set_structure(cif, atom_array, include_bonds=include_bonds)

        block = cif.block
        atom_site = block["atom_site"]

        if atom_site.get("occupancy") is None:
            atom_site["occupancy"] = np.ones(len(atom_array), dtype=float)

        atom_site["label_entity_id"] = atom_array.label_entity_id

        return cif.serialize()


def remove_digits_from_label_asym_id(atom_array: AtomArray) -> AtomArray:
    """Remove digits from the label_asym_id.

    The numerical part of "label_asym_id" is added by Meson during data processing,
    with the purpose of distinguishing unique chains within the bioassembly.

    atom_array: AtomArray object with "label_asym_id" attribute.

    Returns: AtomArray object with "label_asym_id" attribute updated.
    """
    atom_array.label_asym_id = np.vectorize(remove_numbers)(atom_array.label_asym_id)
    return atom_array


def residue_make_dummy_feature[M: MutableMapping[str, torch.Tensor]](
    features_dict: M,
    dummy_feats: Sequence[str] = ("msa",),
) -> M:
    """Compute residue make dummy feature."""
    num_token = features_dict["token_index"].shape[0]
    num_atom = features_dict["atom_to_token_idx"].shape[0]
    num_msa = 1
    num_templ = 4
    num_pockets = 30
    feat_shape, _ = residue_get_data_shape_dict(
        num_token=num_token,
        num_atom=num_atom,
        num_msa=num_msa,
        num_templ=num_templ,
        num_pocket=num_pockets,
    )
    for feat_name in dummy_feats:
        if feat_name not in ["msa", "template"]:
            cur_feat_shape = feat_shape[feat_name]
            features_dict[feat_name] = torch.zeros(cur_feat_shape)
    if "msa" in dummy_feats:
        features_dict["msa"] = torch.nonzero(features_dict["restype"])[:, 1].unsqueeze(
            0
        )
        if features_dict["msa"].shape != feat_shape["msa"]:
            message = "Invalid state: features_dict['msa'].shape == feat_shape['msa']"
            raise ValueError(message)
        features_dict["has_deletion"] = torch.zeros(feat_shape["has_deletion"])
        features_dict["deletion_value"] = torch.zeros(feat_shape["deletion_value"])
        features_dict["profile"] = features_dict["restype"].clone()
        if features_dict["profile"].shape != feat_shape["profile"]:
            message = (
                "Invalid state: features_dict['profile'].shape == feat_shape['profile']"
            )
            raise ValueError(message)
        features_dict["deletion_mean"] = torch.zeros(feat_shape["deletion_mean"])
        for key in [
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ]:
            features_dict[key] = torch.tensor(0, dtype=torch.int32)

    if "template" in dummy_feats:
        features_dict["template_restype"] = (
            torch.ones(feat_shape["template_restype"]) * 31
        )  # gap
        features_dict["template_all_atom_mask"] = torch.zeros(
            feat_shape["template_all_atom_mask"]
        )
        features_dict["template_all_atom_positions"] = torch.zeros(
            feat_shape["template_all_atom_positions"]
        )
    if features_dict["msa"].dim() < 2:
        msg = f"msa must be 2D, get shape: {features_dict['msa'].shape}"
        raise ValueError(msg)
    return features_dict


def residue_data_type_transform[M: MutableMapping[str, torch.Tensor]](
    feat_or_label_dict: M,
) -> M:
    """Compute residue data type transform."""
    for key, value in feat_or_label_dict.items():
        if key in ResidueIntDataList:
            feat_or_label_dict[key] = value.to(torch.long)

    return feat_or_label_dict


ResidueIntDataList = [
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "ref_space_uid",
    "template_restype",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "frame_atom_index",
    "msa",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
]


def residue_get_data_shape_dict(
    num_token: int, num_atom: int, num_msa: int, num_templ: int, num_pocket: int
) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """Generate a dictionary containing the shapes of all data.

    Args:
        num_token (int): Number of tokens.
        num_atom (int): Number of atoms.
        num_msa (int): Number of MSA sequences.
        num_templ (int): Number of templates.
        num_pocket (int): Number of pockets to the same interested ligand.

    Returns:
        tuple[dict[str, tuple], dict[str, tuple]]: A tuple containing feat and label
        shape dictionaries.

    """
    # Features in AlphaFold3 SI Table5
    feat = {
        # Token features
        "residue_index": (num_token,),
        "token_index": (num_token,),
        "asym_id": (num_token,),
        "entity_id": (num_token,),
        "sym_id": (num_token,),
        "restype": (num_token, 32),
        # chain permutation features
        "entity_mol_id": (num_atom,),
        "mol_id": (num_atom,),
        "mol_atom_index": (num_atom,),
        # Reference features
        "ref_pos": (num_atom, 3),
        "ref_mask": (num_atom,),
        "ref_element": (num_atom, 128),  # note: 128 elem in the paper
        "ref_charge": (num_atom,),
        "ref_atom_name_chars": (num_atom, 4, 64),
        "ref_space_uid": (num_atom,),
        # Msa features
        "msa": (num_msa, num_token),
        "has_deletion": (num_msa, num_token),
        "deletion_value": (num_msa, num_token),
        "profile": (num_token, 32),
        "deletion_mean": (num_token,),
        # Template features
        "template_restype": (num_templ, num_token),
        "template_all_atom_mask": (num_templ, num_token, 37),
        "template_all_atom_positions": (num_templ, num_token, 37, 3),
        "template_pseudo_beta_mask": (num_templ, num_token),
        "template_backbone_frame_mask": (num_templ, num_token),
        "template_distogram": (num_templ, num_token, num_token, 39),
        "template_unit_vector": (num_templ, num_token, num_token, 3),
        # Bond features
        "token_bonds": (num_token, num_token),
        "is_protein": (num_atom,),  # Atom level, not token level
        "is_rna": (num_atom,),
        "is_dna": (num_atom,),
        "is_ligand": (num_atom,),
        "distogram_rep_atom_mask": (num_atom,),
        "pae_rep_atom_mask": (num_atom,),
        "plddt_m_rep_atom_mask": (num_atom,),
        "modified_res_mask": (num_atom,),
        "bond_mask": (num_atom, num_atom),
        "resolution": (1,),
    }

    # Extra features needed
    extra_feat = {
        # Input features
        "atom_to_token_idx": (num_atom,),  # after crop
        "atom_to_tokatom_idx": (num_atom,),  # after crop
        "pae_rep_atom_mask": (num_atom,),  # same as "pae_rep_atom_mask" in label_dict
        "is_distillation": (1,),
    }

    # Label
    label = {
        "coordinate": (num_atom, 3),
        "coordinate_mask": (num_atom,),
        "has_frame": (num_token,),  # move to input_feature_dict?
        "frame_atom_index": (num_token, 3),  # atom index after crop
        # Metrics
        "interested_ligand_mask": (
            num_pocket,
            num_atom,
        ),
        "pocket_mask": (
            num_pocket,
            num_atom,
        ),
    }

    # Merged
    all_feat = {**feat, **extra_feat}
    return all_feat, label


def residue_pdb_to_cif(
    input_fname: str, output_fname: str, entry_id: str | None = None
) -> None:
    """Convert PDB to CIF.

    Args:
        input_fname (str): input PDB file name
        output_fname (str): output CIF file name
        entry_id (str, optional): entry id. Defaults to None.

    """
    pdbfile = PDBFile.read(input_fname)
    atom_array = pdbfile.get_structure(model=1, include_bonds=True, altloc="first")

    # Before modifying chains, fill missing UNK residues so CIFWriter treats them as
    # unresolved
    filled_atoms = []
    output_masks = []

    chain_starts_orig = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    for c_start, c_stop in itertools.pairwise(chain_starts_orig):
        chain_arr = atom_array[c_start:c_stop]

        # Only fill for non-hetero chains
        if all(~chain_arr.hetero):
            # Determine chain polymer type to assign correct missing residue names
            prot_cnt, dna_cnt, rna_cnt = 0, 0, 0
            unique_res_names = np.unique(chain_arr.res_name)
            for name in unique_res_names:
                if name in PRO_STD_RESIDUES:
                    prot_cnt += 1
                elif name in DNA_STD_RESIDUES:
                    dna_cnt += 1
                elif name in RNA_STD_RESIDUES:
                    rna_cnt += 1

            if dna_cnt >= prot_cnt and dna_cnt >= rna_cnt and dna_cnt > 0:
                missing_res_name = "DN"
            elif rna_cnt >= prot_cnt and rna_cnt >= dna_cnt and rna_cnt > 0:
                missing_res_name = "N"
            else:
                missing_res_name = "UNK"

            res_starts = struc.get_residue_starts(chain_arr, add_exclusive_stop=True)
            for i, (r_start, r_stop) in enumerate(itertools.pairwise(res_starts)):
                # Add the actual residue
                filled_atoms.append(chain_arr[r_start:r_stop])
                output_masks.append(np.ones(r_stop - r_start, dtype=bool))

                # Check gap with next residue
                if i < len(res_starts) - 2:
                    current_res_id = chain_arr.res_id[r_start]
                    next_res_start = res_starts[i + 1]
                    next_res_id = chain_arr.res_id[next_res_start]

                    if next_res_id > current_res_id + 1:
                        # Gap detected, create dummy atoms
                        for missing_id in range(current_res_id + 1, next_res_id):
                            # Copy the first atom of the current residue as template for
                            # dummy
                            dummy = chain_arr[r_start : r_start + 1].copy()
                            dummy.res_name[:] = missing_res_name
                            dummy.res_id[:] = missing_id
                            # Important: set is_resolved to False if it doesn't exist
                            # yet,
                            # CIFWriter uses is_resolved to detect missing residues
                            if "is_resolved" not in dummy.get_annotation_categories():
                                dummy.set_annotation("is_resolved", np.array([False]))
                            else:
                                dummy.is_resolved[:] = False

                            filled_atoms.append(dummy)
                            output_masks.append(np.array([False]))
        else:
            # HETATM chains: keep as is
            filled_atoms.append(chain_arr)
            output_masks.append(np.ones(len(chain_arr), dtype=bool))

    # Reassemble atom_array with dummy UNK residues included
    if filled_atoms:
        atom_array = filled_atoms[0]
        for arr in filled_atoms[1:]:
            atom_array += arr

    atom_array_output_mask = np.concatenate(output_masks)

    # Ensure is_resolved annotation exists for original atoms as well
    if "is_resolved" not in atom_array.get_annotation_categories():
        is_resolved = np.ones(len(atom_array), dtype=bool)
        # the dummy ones created above are False, but here we just align with
        # atom_array_output_mask
        is_resolved[~atom_array_output_mask] = False
        atom_array.set_annotation("is_resolved", is_resolved)

    seq_to_entity_id = {}
    cnt = 0
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # split chains by hetero
    new_chain_starts = []
    for c_start, c_stop in itertools.pairwise(chain_starts):
        new_chain_starts.append(c_start)
        hetero_diff = np.where(
            atom_array.hetero[c_start : (c_stop - 1)]
            != atom_array.hetero[(c_start + 1) : c_stop]
        )
        if hetero_diff[0].shape[0] > 0:
            new_chain_starts_002 = c_start + hetero_diff[0] + 1
            new_chain_starts.extend(new_chain_starts_002.tolist())

    new_chain_starts.append(chain_starts[-1])

    # split HETATM chains by res id
    new_chain_starts2 = []
    for c_start, c_stop in itertools.pairwise(new_chain_starts):
        new_chain_starts2.append(c_start)
        res_id_diff = np.diff(atom_array.res_id[c_start:c_stop])
        uncont_res_starts = np.where(res_id_diff >= 1)

        if uncont_res_starts[0].shape[0] > 0:
            for res_start_atom_idx in uncont_res_starts[0]:
                new_chain_start = c_start + res_start_atom_idx + 1
                # atom_array.hetero is True if "HETATM"
                if (
                    atom_array.hetero[new_chain_start]
                    and atom_array.hetero[new_chain_start - 1]
                ):
                    new_chain_starts2.append(new_chain_start)

    chain_starts = [*new_chain_starts2, chain_starts[-1]]

    label_entity_id = np.empty(len(atom_array), dtype="<U4")
    atom_index = np.arange(len(atom_array), dtype=np.int32)
    res_id = np.empty(len(atom_array), dtype=atom_array.res_id.dtype)
    chain_id = np.empty(len(atom_array), dtype=atom_array.chain_id.dtype)

    for chain_count, (c_start, c_stop) in enumerate(
        itertools.pairwise(chain_starts), start=1
    ):
        new_chain_id = int_to_letters(chain_count)
        chain_id[c_start:c_stop] = new_chain_id

        chain_array = atom_array[c_start:c_stop]
        residue_starts = struc.get_residue_starts(chain_array, add_exclusive_stop=True)
        resname_seq = list(chain_array[residue_starts[:-1]].res_name)
        resname_str = "_".join(resname_seq)
        if (
            all(name in DNA_STD_RESIDUES for name in resname_seq)
            and resname_str in seq_to_entity_id
        ):
            resname_seq.reverse()
            resname_str = "_".join(resname_seq)
            atom_index[c_start:c_stop] = atom_index[c_start:c_stop][::-1]

        if resname_str not in seq_to_entity_id:
            cnt += 1
            seq_to_entity_id[resname_str] = str(cnt)
        label_entity_id[c_start:c_stop] = seq_to_entity_id[resname_str]

        res_cnt = 1
        for res_start, res_stop in itertools.pairwise(residue_starts):
            # Use res_cnt to determine missing sequence instead of original res_id.
            # E.g. if original res_id jumps from 708 to 766, there is a gap of 57.
            res_id[c_start:c_stop][res_start:res_stop] = res_cnt
            res_cnt += 1

    atom_array = atom_array[atom_index]
    atom_array_output_mask = atom_array_output_mask[atom_index]

    # add label entity id
    atom_array.set_annotation("label_entity_id", label_entity_id)
    entity_poly_type = {}
    for seq, entity_id in seq_to_entity_id.items():
        resname_seq = seq.split("_")

        count = defaultdict(int)
        for name in resname_seq:
            if name in PRO_STD_RESIDUES:
                count["prot"] += 1
            elif name in DNA_STD_RESIDUES:
                count["dna"] += 1
            elif name in RNA_STD_RESIDUES:
                count["rna"] += 1
            else:
                count["other"] += 1

        if count["prot"] >= 2 and count["dna"] == 0 and count["rna"] == 0:
            entity_poly_type[entity_id] = "polypeptide(L)"
        elif count["dna"] >= 2 and count["rna"] == 0 and count["prot"] == 0:
            entity_poly_type[entity_id] = "polydeoxyribonucleotide"
        elif count["rna"] >= 2 and count["dna"] == 0 and count["prot"] == 0:
            entity_poly_type[entity_id] = "polyribonucleotide"
        else:
            # other entity type: ignoring
            continue

    # add label atom id
    atom_array.set_annotation("label_atom_id", atom_array.atom_name)

    # add label asym id
    atom_array.chain_id = chain_id  # reset chain_id
    atom_array.set_annotation("label_asym_id", atom_array.chain_id)

    # add label seq id
    atom_array.res_id = res_id  # reset res_id
    atom_array.set_annotation("label_seq_id", atom_array.res_id)

    w = ResidueCIFWriter(
        atom_array=atom_array,
        entity_poly_type=entity_poly_type,
        atom_array_output_mask=atom_array_output_mask,
    )
    w.save_to_cif(
        output_fname,
        entry_id=entry_id or Path(output_fname).name,
        include_bonds=True,
    )


def dump_bioassembly_to_cif(
    bioassembly_pkl: str | Path, output_cif: str | Path
) -> None:
    """Dump a bioassembly dict to CIF.

    Args:
        bioassembly_pkl (Union[str, Path]): A *.pkl.gz file path of saved bioassembly
            dict.
        output_cif (Union[str, Path]): A *.cif file path to save.

    """
    bio_dict = load_gzip_pickle(bioassembly_pkl)
    atom_array = bio_dict["atom_array"]
    entity_poly_type = bio_dict["entity_poly_type"]
    writer = ResidueCIFWriter(atom_array=atom_array, entity_poly_type=entity_poly_type)
    writer.save_to_cif(
        output_cif, entry_id=Path(bioassembly_pkl).stem, include_bonds=False
    )


def get_atom_level_token_mask(
    token_array: TokenArray, atom_array: AtomArray
) -> np.ndarray:
    """Create a boolean mask indicating whether each atom in the atom array.

    corresponds to an atom-level token (token containing only one atom).

    Returns:
        np.ndarray: Boolean tensor of shape [N_atom] where True indicates
                     the atom belongs to an atom-level token

    """
    atom_level_mask = np.zeros(len(atom_array), dtype=bool)

    # For each token, check if it's an atom-level token (contains only one atom)
    for token in token_array:
        if len(token.atom_indices) == 1:
            # If token has only one atom, mark that atom as belonging to an atom-level
            # token
            atom_level_mask[token.atom_indices[0]] = True

    return atom_level_mask


def pad_to(arr: np.ndarray, shape: tuple, **kwargs: Any) -> np.ndarray:
    """Pad an array to a given shape. Wrapper around np.pad().

    Args:
      arr: numpy array to pad
      shape: target shape, use None for axes that should stay the same
      **kwargs: additional args for np.pad, e.g. constant_values=-1

    Returns:
      the padded array

    Raises:
      ValueError if arr and shape have a different number of axes.

    """
    if arr.ndim != len(shape):
        msg = f"arr and shape have different number of axes. {arr.shape=}, {shape=}"
        raise ValueError(msg)

    num_pad = []
    for axis, width in enumerate(shape):
        if width is None:
            num_pad.append((0, 0))
        elif width >= arr.shape[axis]:
            num_pad.append((0, width - arr.shape[axis]))
        else:
            msg = f"Can not pad to a smaller shape. {arr.shape=}, {shape=}"
            raise ValueError(msg)
    return np.pad(arr, pad_width=num_pad, **kwargs)


def residue_map_annotations_to_atom_indices(
    atom_array: AtomArray, annot_keys: list[str]
) -> dict[tuple, list[int]]:
    """Map annotations to atom indices.

    Args:
        atom_array (AtomArray): Biotite AtomArray
        annot_keys (list[str]): annotation keys, eg: ['chain_id','res_id','res_name']

    Returns:
        dict[tuple, list[int]]: annotation to atom indices map, eg: {(chain_id, res_id,
        res_name): [atom_indices]}

    """
    annot_arrays = [getattr(atom_array, k) for k in annot_keys]
    annots_to_indices = defaultdict(list)
    for i, atom_annots in enumerate(zip(*annot_arrays, strict=False)):
        annots_to_indices[atom_annots].append(i)
    return annots_to_indices


logger = get_logger(__name__)


_MIN_TERMINAL_C_OXT_DISTANCE = 1.0


_MAX_TERMINAL_C_OXT_DISTANCE = 1.7


_MIN_TERMINAL_C_CA_DISTANCE = 1.3


_MAX_TERMINAL_C_CA_DISTANCE = 1.7


_MIN_TERMINAL_C_O_DISTANCE = 1.0


_MAX_TERMINAL_C_O_DISTANCE = 1.5


_MIN_TERMINAL_O_OXT_DISTANCE = 1.8


def structural_get_atom_mask_by_name(
    atom_array: AtomArray,
    entity_id: int | None = None,
    position: int | None = None,
    atom_name: str | None = None,
    copy_id: int | None = None,
) -> np.ndarray:
    """Get the atom mask of atoms with specific identifiers.

    Args:
        atom_array (AtomArray): Biotite Atom array.
        entity_id (int): Entity id.
        position (int): Residue index of the atom.
        atom_name (str): Atom name.
        copy_id (copy_id): A asym chain id in N copies of an entity.

    Returns:
        np.ndarray: Array of a bool mask.

    """
    mask = np.ones(atom_array.shape, dtype=np.bool_)

    if entity_id is not None:
        mask &= atom_array.label_entity_id == str(entity_id)
    if position is not None:
        mask &= atom_array.res_id == int(position)
    if atom_name is not None:
        mask &= atom_array.atom_name == str(atom_name)
    if copy_id is not None:
        mask &= atom_array.copy_id == int(copy_id)
    return mask


def structural_get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """Get inter residue bonds by checking chain_id and res_id.

    Args:
        atom_array (AtomArray): Biotite AtomArray, must have chain_id and res_id

    Returns:
        np.ndarray: inter residue bonds, shape = (n,2)

    """
    if atom_array.bonds is None:
        return np.empty((0, 2), dtype=int)
    idx_i = atom_array.bonds.as_array()[:, 0]
    idx_j = atom_array.bonds.as_array()[:, 1]
    chain_id_diff = atom_array.chain_id[idx_i] != atom_array.chain_id[idx_j]
    res_id_diff = atom_array.res_id[idx_i] != atom_array.res_id[idx_j]
    diff_mask = chain_id_diff | res_id_diff
    inter_residue_bonds = atom_array.bonds.as_array()[diff_mask]
    return inter_residue_bonds[:, :2]  # remove bond type


def structural_map_annotations_to_atom_indices(
    atom_array: AtomArray, annot_keys: Sequence[str]
) -> defaultdict[tuple, list[int]]:
    """Map annotation value tuples to atom indices.

    Missing atom-name combinations are queried during residue reconstruction, so
    a defaultdict keeps absent keys cheap and explicit.
    """
    annot_arrays = [getattr(atom_array, key) for key in annot_keys]
    annots_to_indices: defaultdict[tuple, list[int]] = defaultdict(list)
    for idx, annot_values in enumerate(zip(*annot_arrays, strict=False)):
        annots_to_indices[tuple(annot_values)].append(idx)
    return annots_to_indices


def structural_save_atoms_to_cif(
    output_cif_file: str,
    atom_array: AtomArray,
    entity_poly_type: dict[str, str],
    pdb_id: str,
) -> None:
    """Save atom array data to a CIF file.

    Args:
        output_cif_file (str): The output path for saving the atom array in CIF format.
        atom_array (AtomArray): The atom array to be saved.
        entity_poly_type: The entity poly type information.
        pdb_id: The PDB ID for the entry.

    """
    cifwriter = StructuralCIFWriter(atom_array, entity_poly_type)
    cifwriter.save_to_cif(
        output_path=output_cif_file,
        entry_id=pdb_id,
        include_bonds=False,
    )


def _local_coordinate_frame(
    carbon: np.ndarray,
    alpha_carbon: np.ndarray,
    oxygen: np.ndarray,
) -> np.ndarray | None:
    """Build an orthonormal frame for one peptide carbonyl group."""
    ca_axis = alpha_carbon - carbon
    ca_norm = np.linalg.norm(ca_axis)
    if not np.isfinite(ca_norm) or ca_norm < 1e-6:
        return None
    ca_axis = ca_axis / ca_norm

    oxygen_axis = oxygen - carbon
    oxygen_axis -= np.dot(oxygen_axis, ca_axis) * ca_axis
    oxygen_norm = np.linalg.norm(oxygen_axis)
    if not np.isfinite(oxygen_norm) or oxygen_norm < 1e-6:
        return None
    oxygen_axis = oxygen_axis / oxygen_norm

    normal_axis = np.cross(ca_axis, oxygen_axis)
    return np.column_stack((ca_axis, oxygen_axis, normal_axis))


def _repair_terminal_oxt_coordinates(atom_array: AtomArray) -> int:
    """Rebuild chemically invalid OXT coordinates at free protein C termini."""
    annotations = set(atom_array.get_annotation_categories())
    if not {"mol_type", "ref_pos"}.issubset(annotations):
        return 0

    chain_ids = (
        atom_array.label_asym_id
        if "label_asym_id" in annotations
        else atom_array.chain_id
    )
    ref_mask = atom_array.ref_mask if "ref_mask" in annotations else None
    bonds = atom_array.bonds.as_array() if atom_array.bonds is not None else None
    repaired = 0

    for chain_id in np.unique(chain_ids[atom_array.mol_type == "protein"]):
        chain_mask = (chain_ids == chain_id) & (atom_array.mol_type == "protein")
        terminal_res_id = np.max(atom_array.res_id[chain_mask])
        terminal_mask = chain_mask & (atom_array.res_id == terminal_res_id)

        atom_matches = {
            atom_name: np.flatnonzero(
                terminal_mask & (atom_array.atom_name == atom_name)
            )
            for atom_name in ("CA", "C", "O", "OXT")
        }
        if any(len(indices) != 1 for indices in atom_matches.values()):
            continue
        atom_indices = {name: indices[0] for name, indices in atom_matches.items()}

        if bonds is not None:
            has_external_bond = False
            for atom_index in (
                atom_indices["C"],
                atom_indices["O"],
                atom_indices["OXT"],
            ):
                atom_bonds = bonds[np.any(bonds[:, :2] == atom_index, axis=1), :2]
                bonded_indices = atom_bonds[atom_bonds != atom_index]
                if np.any(~terminal_mask[bonded_indices]):
                    has_external_bond = True
                    break
            if has_external_bond:
                continue

        required_indices = np.array(
            [
                atom_indices["C"],
                atom_indices["CA"],
                atom_indices["O"],
                atom_indices["OXT"],
            ]
        )
        if ref_mask is not None and not np.all(ref_mask[required_indices]):
            continue

        pred_coordinates = atom_array.coord[required_indices]
        ref_coordinates = atom_array.ref_pos[required_indices]
        if not np.all(np.isfinite(ref_coordinates)):
            continue
        if not np.all(np.isfinite(pred_coordinates[:3])):
            logger.warning(
                "Skipped terminal OXT repair for chain %s residue %s because "
                "the predicted C/CA/O coordinates are not finite",
                chain_id,
                terminal_res_id,
            )
            continue

        c_oxt_distance = np.linalg.norm(pred_coordinates[3] - pred_coordinates[0])
        o_oxt_distance = np.linalg.norm(pred_coordinates[3] - pred_coordinates[2])
        if (
            np.isfinite(c_oxt_distance)
            and _MIN_TERMINAL_C_OXT_DISTANCE
            <= c_oxt_distance
            <= _MAX_TERMINAL_C_OXT_DISTANCE
            and np.isfinite(o_oxt_distance)
            and o_oxt_distance >= _MIN_TERMINAL_O_OXT_DISTANCE
        ):
            continue

        c_ca_distance = np.linalg.norm(pred_coordinates[1] - pred_coordinates[0])
        c_o_distance = np.linalg.norm(pred_coordinates[2] - pred_coordinates[0])
        if not (
            _MIN_TERMINAL_C_CA_DISTANCE <= c_ca_distance <= _MAX_TERMINAL_C_CA_DISTANCE
            and _MIN_TERMINAL_C_O_DISTANCE <= c_o_distance <= _MAX_TERMINAL_C_O_DISTANCE
        ):
            logger.warning(
                "Skipped terminal OXT repair for chain %s residue %s because "
                "the predicted C-CA/C-O distances are invalid (%.3f/%.3f A)",
                chain_id,
                terminal_res_id,
                c_ca_distance,
                c_o_distance,
            )
            continue

        pred_frame = _local_coordinate_frame(*pred_coordinates[:3])
        ref_frame = _local_coordinate_frame(*ref_coordinates[:3])
        if pred_frame is None or ref_frame is None:
            continue

        ref_c_oxt = ref_coordinates[3] - ref_coordinates[0]
        ref_c_oxt_distance = np.linalg.norm(ref_c_oxt)
        ref_o_oxt_distance = np.linalg.norm(ref_coordinates[3] - ref_coordinates[2])
        if not (
            _MIN_TERMINAL_C_OXT_DISTANCE
            <= ref_c_oxt_distance
            <= _MAX_TERMINAL_C_OXT_DISTANCE
            and ref_o_oxt_distance >= _MIN_TERMINAL_O_OXT_DISTANCE
        ):
            continue

        rotation = pred_frame @ ref_frame.T
        atom_array.coord[atom_indices["OXT"]] = (
            pred_coordinates[0] + rotation @ ref_c_oxt
        )
        repaired += 1

    return repaired


def structural_save_structure_cif(
    atom_array: AtomArray,
    pred_coordinate: torch.Tensor,
    output_fpath: str,
    entity_poly_type: dict[str, str],
    pdb_id: str,
    *,
    save_wo_unresolved: bool = False,
) -> None:
    """Save the predicted structure to a CIF file.

    Before serialization, chemically invalid OXT coordinates at free protein
    C termini are rebuilt from the CCD reference geometry when the predicted
    C/CA/O anchor geometry is valid. Caller-owned coordinates are not modified.

    Args:
        atom_array (AtomArray): The original AtomArray containing the structure.
        pred_coordinate (torch.Tensor): The predicted coordinates for the structure.
        output_fpath (str): The output file path for saving the CIF file.
        entity_poly_type (dict[str, str]): The entity poly type information.
        pdb_id (str): The PDB ID for the entry.
        save_wo_unresolved (bool): Whether to save a version without unresolved atoms.
            Defaults to False.

    """
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.detach().cpu().numpy().copy()
    pred_atom_array.coord = pred_pose
    repaired_oxt_count = _repair_terminal_oxt_coordinates(pred_atom_array)
    if repaired_oxt_count:
        logger.warning(
            "Rebuilt %d invalid terminal OXT coordinate(s) before saving %s",
            repaired_oxt_count,
            output_fpath,
        )
    structural_save_atoms_to_cif(
        output_fpath,
        pred_atom_array,
        entity_poly_type,
        pdb_id,
    )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved") and save_wo_unresolved:
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        structural_save_atoms_to_cif(
            output_fpath.replace(".cif", "_wounresol.cif"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


class StructuralCIFWriter(_CommonCIFWriter):
    """Write AtomArray to cif.

    Args:
        atom_array (AtomArray): Biotite AtomArray object.
        entity_poly_type (dict[str, str], optional): A dict of label_entity_id to
            entity_poly_type. Defaults to None.
                                                     If None, "the entity_poly" and
                                                     "entity_poly_seq" will not be
                                                     written to the cif.
        atom_array_output_mask (np.ndarray, optional): A mask of atom_array. Defaults
            to None.
                                                      If None, all atoms will be
                                                      written to the cif.

    """

    def __init__(
        self,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str] | None = None,
        atom_array_output_mask: np.ndarray | None = None,
    ) -> None:
        self.atom_array = copy.deepcopy(atom_array)
        self.entity_poly_type = entity_poly_type
        self.atom_array_output_mask = atom_array_output_mask

    def _get_unresolved_block(self) -> pdbx.CIFCategory | None:
        if self.atom_array_output_mask is None or self.entity_poly_type is None:
            return None

        res_starts = get_residue_starts(self.atom_array, add_exclusive_stop=True)
        is_res_starts = np.zeros(len(self.atom_array_output_mask), dtype=bool)
        for start, stop in itertools.pairwise(res_starts):
            if not any(self.atom_array.is_resolved[start:stop]):
                is_res_starts[start] = True

        mask = (~self.atom_array_output_mask) & is_res_starts
        if not np.any(mask):
            # No unresolved atoms
            return None
        polymer_flag_bool = np.isin(
            self.atom_array.label_entity_id[mask], list(self.entity_poly_type.keys())
        )
        polymer_flag = ["Y" if i else "N" for i in polymer_flag_bool]

        unresolved_block: dict[str, np.ndarray | Sequence[str]] = {}
        unresolved_block["id"] = np.arange(mask.sum()) + 1
        unresolved_block["PDB_model_num"] = np.ones(mask.sum(), dtype=int)
        unresolved_block["polymer_flag"] = polymer_flag
        unresolved_block["occupancy_flag"] = np.ones(mask.sum(), dtype=int)
        unresolved_block["auth_asym_id"] = self.atom_array.chain_id[mask]
        unresolved_block["auth_comp_id"] = self.atom_array.res_name[mask]
        unresolved_block["auth_seq_id"] = self.atom_array.res_id[mask]
        unresolved_block["PDB_ins_code"] = ["?"] * mask.sum()
        unresolved_block["label_asym_id"] = self.atom_array.chain_id[mask]
        unresolved_block["label_comp_id"] = self.atom_array.res_name[mask]
        unresolved_block["label_seq_id"] = self.atom_array.res_id[mask]
        return pdbx.CIFCategory(unresolved_block)

    def save_to_cif(
        self,
        output_path: str | Path,
        entry_id: str | None = None,
        *,
        include_bonds: bool = False,
    ) -> None:
        """Save AtomArray to cif.

        Args:
            output_path (str): Output path of cif file.
            entry_id (str, optional): The value of "_entry.id" in cif. Defaults to
                None.
                                      If None, the entry_id will be the basename of
                                      output_path (without ".cif" extension).
            include_bonds (bool, optional): Whether to include  bonds in the cif.
                Defaults to False.
                                            If set to True and `array` has associated
                                            ``bonds`` , the
                                            intra-residue bonds will be written into
                                            the ``chem_comp_bond``
                                            category.
                                            Inter-residue bonds will be written into
                                            the ``struct_conn``
                                            independent of this parameter.

        """
        if entry_id is None:
            entry_id = Path(output_path).name.replace(".cif", "")

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        block_dict["chem_comp"] = self._get_chem_comp_block()

        if self.entity_poly_type:
            block_dict["entity"] = self._get_entity_block()
            block_dict.update(
                StructuralCIFWriter.get_entity_poly_and_entity_poly_seq_block(
                    self.entity_poly_type, self.atom_array
                )
            )

        if self.atom_array_output_mask is not None:
            unresolved_block = self._get_unresolved_block()
            if unresolved_block is not None:
                block_dict["pdbx_unobs_or_zero_occ_residues"] = unresolved_block

        block = pdbx.CIFBlock()
        for name, category in block_dict.items():
            block[name] = category
        cif = pdbx.CIFFile(
            {
                Path(output_path).name.replace(".cif", "")
                + "_predicted_by_opendde": block
            }
        )

        if self.atom_array_output_mask is not None:
            atom_array = self.atom_array[self.atom_array_output_mask]
        else:
            atom_array = self.atom_array

        if not include_bonds and atom_array.bonds is not None:
            # https://github.com/biotite-dev/biotite/pull/804
            inter_bonds = pdbx.convert._filter_bonds(atom_array, "inter")  # noqa: SLF001 - Biotite extension hook
            atom_array.bonds = _replace_bond_array(atom_array.bonds, inter_bonds)

        pdbx.set_structure(cif, atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block["atom_site"]

        occ = atom_site.get("occupancy")
        if occ is None:
            atom_site["occupancy"] = np.ones(len(atom_array), dtype=float)

        b_factor = atom_site.get("B_iso_or_equiv")
        if b_factor is None:
            atom_site["B_iso_or_equiv"] = np.round(
                np.zeros(len(atom_array), dtype=float), 2
            ).astype(str)

        if "label_entity_id" in atom_array.get_annotation_categories():
            atom_site["label_entity_id"] = atom_array.label_entity_id
        cif.write(output_path)

    @staticmethod
    def to_mmcif_string(
        atom_array: AtomArray,
        entity_poly_type: dict[str, str] | None = None,
        entry_id: str | None = None,
        *,
        include_bonds: bool = False,
    ) -> str:
        """Convert the AtomArray to an mmCIF format string.

        Parameters
        ----------
        atom_array : AtomArray
            The structure to convert.
        entity_poly_type : dict[str, str], optional
            A dictionary mapping entity IDs to their polymer types.
        entry_id : str, optional
            The value of "_entry.id" in the CIF.
        include_bonds : bool, optional
            Whether to include bonds in the output. Default is False.

        Returns
        -------
        str
            The mmCIF-formatted string representation of the structure.

        Raises
        ------
        ValueError
            If entry_id is not provided.

        """
        if entry_id is None:
            msg = "entry_id must be provided."
            raise ValueError(msg)

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        if entity_poly_type:
            block_dict.update(
                StructuralCIFWriter.get_entity_poly_and_entity_poly_seq_block(
                    atom_array=atom_array, entity_poly_type=entity_poly_type
                )
            )

        block = pdbx.CIFBlock()
        for name, category in block_dict.items():
            block[name] = category
        cif = pdbx.CIFFile({entry_id: block})  # Use a default block name
        pdbx.set_structure(cif, atom_array, include_bonds=include_bonds)

        block = cif.block
        atom_site = block["atom_site"]

        if atom_site.get("occupancy") is None:
            atom_site["occupancy"] = np.ones(len(atom_array), dtype=float)

        atom_site["label_entity_id"] = atom_array.label_entity_id

        return cif.serialize()


def structural_make_dummy_feature(
    features_dict: dict[str, torch.Tensor],
    dummy_feats: Sequence = ("msa",),
) -> dict[str, torch.Tensor]:
    """Compute structural make dummy feature."""
    num_token = features_dict["token_index"].shape[0]
    num_atom = features_dict["atom_to_token_idx"].shape[0]
    num_msa = 1
    num_templ = 4
    num_pockets = 30
    feat_shape, _ = structural_get_data_shape_dict(
        num_token=num_token,
        num_atom=num_atom,
        num_msa=num_msa,
        num_templ=num_templ,
        num_pocket=num_pockets,
    )
    for feat_name in dummy_feats:
        if feat_name not in ["msa", "template"]:
            cur_feat_shape = feat_shape[feat_name]
            features_dict[feat_name] = torch.zeros(cur_feat_shape)
    if "msa" in dummy_feats:
        features_dict["msa"] = torch.nonzero(features_dict["restype"])[:, 1].unsqueeze(
            0
        )
        if features_dict["msa"].shape != feat_shape["msa"]:
            message = "Invalid state: features_dict['msa'].shape == feat_shape['msa']"
            raise ValueError(message)
        features_dict["has_deletion"] = torch.zeros(feat_shape["has_deletion"])
        features_dict["deletion_value"] = torch.zeros(feat_shape["deletion_value"])
        features_dict["profile"] = features_dict["restype"].clone()
        if features_dict["profile"].shape != feat_shape["profile"]:
            message = (
                "Invalid state: features_dict['profile'].shape == feat_shape['profile']"
            )
            raise ValueError(message)
        features_dict["deletion_mean"] = torch.zeros(feat_shape["deletion_mean"])
        for key in [
            "prot_paired_num_alignments",
            "prot_unpaired_num_alignments",
            "rna_paired_num_alignments",
            "rna_unpaired_num_alignments",
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ]:
            features_dict[key] = torch.tensor(0, dtype=torch.int32)

    if "template" in dummy_feats:
        features_dict["template_restype"] = torch.full(
            feat_shape["template_restype"], 31, dtype=torch.long
        )  # gap
        features_dict["template_all_atom_mask"] = torch.zeros(
            feat_shape["template_all_atom_mask"]
        )
        features_dict["template_all_atom_positions"] = torch.zeros(
            feat_shape["template_all_atom_positions"]
        )
        # Mirror the current template featurizer output so downstream code sees
        # the same schema whether templates are real or synthesized as empty.
        features_dict["template_aatype"] = torch.full(
            feat_shape["template_aatype"], 31, dtype=torch.long
        )
        features_dict["template_atom_mask"] = torch.zeros(
            feat_shape["template_atom_mask"], dtype=torch.int32
        )
        features_dict["template_atom_positions"] = torch.zeros(
            feat_shape["template_atom_positions"], dtype=torch.float32
        )
        features_dict["template_pseudo_beta_mask"] = torch.zeros(
            feat_shape["template_pseudo_beta_mask"], dtype=torch.float32
        )
        features_dict["template_backbone_frame_mask"] = torch.zeros(
            feat_shape["template_backbone_frame_mask"], dtype=torch.float32
        )
        features_dict["template_distogram"] = torch.zeros(
            feat_shape["template_distogram"], dtype=torch.float32
        )
        features_dict["template_unit_vector"] = torch.zeros(
            feat_shape["template_unit_vector"], dtype=torch.float32
        )
    if features_dict["msa"].dim() < 2:
        msg = f"msa must be 2D, get shape: {features_dict['msa'].shape}"
        raise ValueError(msg)
    return features_dict


def structural_data_type_transform(
    feat_or_label_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute structural data type transform."""
    for key, value in feat_or_label_dict.items():
        if key in StructuralIntDataList:
            feat_or_label_dict[key] = value.to(torch.long)

    return feat_or_label_dict


StructuralIntDataList = [
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "ref_space_uid",
    "template_restype",
    "template_aatype",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "template_atom_mask",
    "frame_atom_index",
    "msa",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
]


def structural_get_data_shape_dict(
    num_token: int, num_atom: int, num_msa: int, num_templ: int, num_pocket: int
) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """Generate a dictionary containing the shapes of all data.

    Args:
        num_token (int): Number of tokens.
        num_atom (int): Number of atoms.
        num_msa (int): Number of MSA sequences.
        num_templ (int): Number of templates.
        num_pocket (int): Number of pockets to the same interested ligand.

    Returns:
        tuple[dict[str, tuple], dict[str, tuple]]: A tuple containing feat and label
        shape dictionaries.

    """
    # Features in AlphaFold3 SI Table5
    feat = {
        # Token features
        "residue_index": (num_token,),
        "token_index": (num_token,),
        "asym_id": (num_token,),
        "entity_id": (num_token,),
        "sym_id": (num_token,),
        "restype": (num_token, 32),
        # chain permutation features
        "entity_mol_id": (num_atom,),
        "mol_id": (num_atom,),
        "mol_atom_index": (num_atom,),
        # Reference features
        "ref_pos": (num_atom, 3),
        "ref_mask": (num_atom,),
        "ref_element": (num_atom, 128),  # note: 128 elem in the paper
        "ref_charge": (num_atom,),
        "ref_atom_name_chars": (num_atom, 4, 64),
        "ref_space_uid": (num_atom,),
        # Msa features
        "msa": (num_msa, num_token),
        "has_deletion": (num_msa, num_token),
        "deletion_value": (num_msa, num_token),
        "profile": (num_token, 32),
        "deletion_mean": (num_token,),
        # Template features
        "template_restype": (num_templ, num_token),
        "template_all_atom_mask": (num_templ, num_token, 37),
        "template_all_atom_positions": (num_templ, num_token, 37, 3),
        "template_aatype": (num_templ, num_token),
        "template_atom_mask": (num_templ, num_token, 24),
        "template_atom_positions": (num_templ, num_token, 24, 3),
        "template_pseudo_beta_mask": (num_templ, num_token, num_token),
        "template_backbone_frame_mask": (num_templ, num_token, num_token),
        "template_distogram": (num_templ, num_token, num_token, 39),
        "template_unit_vector": (num_templ, num_token, num_token, 3),
        # Bond features
        "token_bonds": (num_token, num_token),
        "is_protein": (num_atom,),  # Atom level, not token level
        "is_rna": (num_atom,),
        "is_dna": (num_atom,),
        "is_ligand": (num_atom,),
        "distogram_rep_atom_mask": (num_atom,),
        "pae_rep_atom_mask": (num_atom,),
        "plddt_m_rep_atom_mask": (num_atom,),
        "modified_res_mask": (num_atom,),
        "bond_mask": (num_atom, num_atom),
        "resolution": (1,),
    }

    # Extra features needed
    extra_feat = {
        # Input features
        "atom_to_token_idx": (num_atom,),  # after crop
        "atom_to_tokatom_idx": (num_atom,),  # after crop
        "pae_rep_atom_mask": (num_atom,),  # same as "pae_rep_atom_mask" in label_dict
    }

    # Label
    label = {
        "coordinate": (num_atom, 3),
        "coordinate_mask": (num_atom,),
        "has_frame": (num_token,),  # move to input_feature_dict?
        "frame_atom_index": (num_token, 3),  # atom index after crop
        # Metrics
        "interested_ligand_mask": (
            num_pocket,
            num_atom,
        ),
        "pocket_mask": (
            num_pocket,
            num_atom,
        ),
    }

    # Merged
    all_feat = {**feat, **extra_feat}
    return all_feat, label


def structural_pdb_to_cif(
    input_fname: str, output_fname: str, entry_id: str | None = None
) -> None:
    """Convert PDB to CIF.

    Args:
        input_fname (str): input PDB file name
        output_fname (str): output CIF file name
        entry_id (str, optional): entry id. Defaults to None.

    """
    pdbfile = PDBFile.read(input_fname)
    atom_array = pdbfile.get_structure(model=1, include_bonds=True, altloc="first")

    seq_to_entity_id = {}
    cnt = 0
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # split chains by hetero
    new_chain_starts = []
    for c_start, c_stop in itertools.pairwise(chain_starts):
        new_chain_starts.append(c_start)
        hetero_diff = np.where(
            atom_array.hetero[c_start : (c_stop - 1)]
            != atom_array.hetero[(c_start + 1) : c_stop]
        )
        if hetero_diff[0].shape[0] > 0:
            new_chain_starts_002 = c_start + hetero_diff[0] + 1
            new_chain_starts.extend(new_chain_starts_002.tolist())

    new_chain_starts.append(chain_starts[-1])

    # split HETATM chains by res id
    new_chain_starts2 = []
    for c_start, c_stop in itertools.pairwise(new_chain_starts):
        new_chain_starts2.append(c_start)
        res_id_diff = np.diff(atom_array.res_id[c_start:c_stop])
        uncont_res_starts = np.where(res_id_diff >= 1)

        if uncont_res_starts[0].shape[0] > 0:
            for res_start_atom_idx in uncont_res_starts[0]:
                new_chain_start = c_start + res_start_atom_idx + 1
                # atom_array.hetero is True if "HETATM"
                if (
                    atom_array.hetero[new_chain_start]
                    and atom_array.hetero[new_chain_start - 1]
                ):
                    new_chain_starts2.append(new_chain_start)

    chain_starts = [*new_chain_starts2, chain_starts[-1]]

    label_entity_id = np.empty(len(atom_array), dtype="<U4")
    atom_index = np.arange(len(atom_array), dtype=np.int32)
    res_id = np.empty(len(atom_array), dtype=atom_array.res_id.dtype)
    chain_id = np.empty(len(atom_array), dtype=atom_array.chain_id.dtype)

    for chain_count, (c_start, c_stop) in enumerate(
        itertools.pairwise(chain_starts), start=1
    ):
        new_chain_id = int_to_letters(chain_count)
        chain_id[c_start:c_stop] = new_chain_id

        chain_array = atom_array[c_start:c_stop]
        residue_starts = struc.get_residue_starts(chain_array, add_exclusive_stop=True)
        resname_seq = list(chain_array[residue_starts[:-1]].res_name)
        resname_str = "_".join(resname_seq)
        if (
            all(name in DNA_STD_RESIDUES for name in resname_seq)
            and resname_str in seq_to_entity_id
        ):
            resname_seq.reverse()
            resname_str = "_".join(resname_seq)
            atom_index[c_start:c_stop] = atom_index[c_start:c_stop][::-1]

        if resname_str not in seq_to_entity_id:
            cnt += 1
            seq_to_entity_id[resname_str] = str(cnt)
        label_entity_id[c_start:c_stop] = seq_to_entity_id[resname_str]

        res_cnt = 1
        for res_start, res_stop in itertools.pairwise(residue_starts):
            res_id[c_start:c_stop][res_start:res_stop] = res_cnt
            res_cnt += 1

    atom_array = atom_array[atom_index]

    # add label entity id
    atom_array.set_annotation("label_entity_id", label_entity_id)
    entity_poly_type = {}
    for seq, entity_id in seq_to_entity_id.items():
        resname_seq = seq.split("_")

        count = defaultdict(int)
        for name in resname_seq:
            if name in PRO_STD_RESIDUES:
                count["prot"] += 1
            elif name in DNA_STD_RESIDUES:
                count["dna"] += 1
            elif name in RNA_STD_RESIDUES:
                count["rna"] += 1
            else:
                count["other"] += 1

        if count["prot"] >= 2 and count["dna"] == 0 and count["rna"] == 0:
            entity_poly_type[entity_id] = "polypeptide(L)"
        elif count["dna"] >= 2 and count["rna"] == 0 and count["prot"] == 0:
            entity_poly_type[entity_id] = "polydeoxyribonucleotide"
        elif count["rna"] >= 2 and count["dna"] == 0 and count["prot"] == 0:
            entity_poly_type[entity_id] = "polyribonucleotide"
        else:
            # other entity type: ignoring
            continue

    # add label atom id
    atom_array.set_annotation("label_atom_id", atom_array.atom_name)

    # add label asym id
    atom_array.chain_id = chain_id  # reset chain_id
    atom_array.set_annotation("label_asym_id", atom_array.chain_id)

    # add label seq id
    atom_array.res_id = res_id  # reset res_id
    atom_array.set_annotation("label_seq_id", atom_array.res_id)

    w = StructuralCIFWriter(atom_array=atom_array, entity_poly_type=entity_poly_type)
    w.save_to_cif(
        output_fname,
        entry_id=entry_id or Path(output_fname).name,
        include_bonds=True,
    )
