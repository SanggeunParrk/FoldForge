"""Compatibility import; chemistry and cache ownership live in foldforge.data.ccd."""
from foldforge.data.ccd.components import (
    _connect_inter_residue as _connect_inter_residue,
    _map_central_to_leaving_groups as _map_central_to_leaving_groups,
    biotite_load_ccd_cif as biotite_load_ccd_cif,
    get_component_atom_array as get_component_atom_array,
    get_one_letter_code as get_one_letter_code,
    get_mol_type as get_mol_type,
    get_all_ccd_code as get_all_ccd_code,
    get_component_rdkit_mol as get_component_rdkit_mol,
    get_ccd_ref_info as get_ccd_ref_info,
    add_inter_residue_bonds as add_inter_residue_bonds,
    res_names_to_sequence as res_names_to_sequence,
    get_ccd_cache_paths as get_ccd_cache_paths,
    set_ccd_cache_paths as set_ccd_cache_paths,
)
