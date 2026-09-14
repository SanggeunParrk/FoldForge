"""Build MiniWorld CCDMol records without changing atom or conformer identity."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import lmdb
import numpy as np
from biomol.core import EdgeFeature, FeatureContainer, IndexTable, NodeFeature
from biotite.structure.io.pdbx import CIFFile
from rdkit import Chem

from .lmdb_store import SCHEMA
from .mol import CCDMol


def record(name, block, molecule, af3_fields=None):
    """Keep canonical heavy atoms in views and lossless raw chemistry in metadata."""
    raw = {
        f"_{cat}.{col}": block[cat][col].as_array().tolist()
        for cat in block
        for col in block[cat]
    }

    def values(category, column, length=0, default="?"):
        return np.asarray(
            raw.get(f"_{category}.{column}", [default] * length), dtype=str
        )

    all_ids = values("chem_comp_atom", "atom_id")
    element = values("chem_comp_atom", "type_symbol", len(all_ids))
    keep = ~np.isin(element, ["H", "D"])
    ids = all_ids[keep]
    n = len(ids)
    nodes = {"id": NodeFeature(ids), "element": NodeFeature(element[keep])}
    for feature, column in [
        ("aromatic", "pdbx_aromatic_flag"),
        ("stereo", "pdbx_stereo_config"),
        ("charge", "charge"),
    ]:
        nodes[feature] = NodeFeature(
            values("chem_comp_atom", column, len(all_ids))[keep]
        )
    nodes["model_xyz"] = NodeFeature(
        np.stack(
            [
                values("chem_comp_atom", f"model_Cartn_{axis}", len(all_ids))[keep]
                for axis in "xyz"
            ],
            axis=-1,
        )
    )
    index = {name: i for i, name in enumerate(ids)}
    src_names = values("chem_comp_bond", "atom_id_1")
    dst_names = values("chem_comp_bond", "atom_id_2")
    bond_keep = np.asarray(
        [a in index and b in index for a, b in zip(src_names, dst_names, strict=True)],
        dtype=bool,
    )
    src = np.array([index[a] for a in src_names[bond_keep]], dtype=np.int64)
    dst = np.array([index[a] for a in dst_names[bond_keep]], dtype=np.int64)

    def edge(v):
        return EdgeFeature(value=np.asarray(v), src_indices=src, dst_indices=dst)

    for feature, column in [
        ("bond_type", "value_order"),
        ("bond_aromatic", "pdbx_aromatic_flag"),
        ("bond_stereo", "pdbx_stereo_config"),
    ]:
        nodes[feature] = edge(
            values("chem_comp_bond", column, len(src_names))[bond_keep]
        )
    metadata = {"ccd_cif": raw, "reference": None, "fragmentation_available": False}
    if af3_fields is not None:
        metadata["af3_text_overrides"] = {
            key: list(value)
            for key, value in af3_fields.items()
            if key != "data_" and list(value) != raw.get(key)
        }
    if molecule is not None and molecule.GetNumAtoms():
        atom_map = molecule.atom_map
        if set(atom_map) != set(all_ids) or sorted(atom_map.values()) != list(
            range(molecule.GetNumAtoms())
        ):
            raise ValueError(f"CIF/RDKit atom identity mismatch: {name}")
        molecule.GetConformer(molecule.ref_conf_id)
        if len(molecule.ref_mask) != molecule.GetNumAtoms():
            raise ValueError(f"Invalid reference mask: {name}")
        props = dict(molecule.__dict__)
        for key, value in props.items():
            if isinstance(value, np.ndarray):
                props[key] = value.tolist()
        metadata["reference"] = {
            "rdkit_binary": base64.b64encode(
                molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
            ).decode(),
            "properties": props,
        }
        # Ring membership/conjugation use the prepared, sanitized topology;
        # no new embedding or altered coordinates are generated.
        if getattr(molecule, "sanitized", False):
            rings = list(Chem.GetSymmSSSR(molecule))
            memberships = [[] for _ in ids]
            reverse = {atom_map[a]: i for i, a in enumerate(ids)}
            for ring_id, ring in enumerate(rings):
                for atom in ring:
                    if atom in reverse:
                        memberships[reverse[atom]].append(ring_id)
            sssr = np.full(
                (n, max(1, max(map(len, memberships), default=0))), -1, dtype=np.int16
            )
            for i, membership in enumerate(memberships):
                sssr[i, : len(membership)] = membership
            nodes["sssr_idx"] = NodeFeature(sssr)
            nodes["hybridization"] = NodeFeature(
                np.asarray(
                    [
                        str(molecule.GetAtomWithIdx(atom_map[a]).GetHybridization())
                        for a in ids
                    ]
                )
            )
            bonds = [
                molecule.GetBondBetweenAtoms(atom_map[ids[a]], atom_map[ids[b]])
                for a, b in zip(src, dst, strict=True)
            ]
            if any(b is None for b in bonds):
                raise ValueError(f"CIF/RDKit bond identity mismatch: {name}")
            nodes["bond_conjugation"] = edge(
                np.asarray([b.GetIsConjugated() for b in bonds], dtype=bool)
            )
            nodes["bond_aromaticity"] = edge(
                np.asarray([b.GetIsAromatic() for b in bonds], dtype=bool)
            )
            metadata["fragmentation_available"] = True
    # Missing chemistry is explicit: do not fabricate ring/conjugation values.
    residue = {
        "id": NodeFeature(values("chem_comp", "name", 1)),
        "formula": NodeFeature(values("chem_comp", "formula", 1)),
    }
    if metadata["fragmentation_available"]:
        residue["rdkit_smiles"] = NodeFeature(
            np.asarray([Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)))])
        )
    index_table = (
        IndexTable.from_parents(
            np.zeros(n, dtype=np.int64), np.zeros(1, dtype=np.int64)
        )
        if n
        else IndexTable(
            atom_to_res=np.zeros(0, dtype=np.int64),
            res_to_chain=np.zeros(1, dtype=np.int64),
            res_atom_indptr=np.zeros(2, dtype=np.int64),
            res_atom_indices=np.zeros(0, dtype=np.int64),
            chain_res_indptr=np.array([0, 1]),
            chain_res_indices=np.array([0]),
        )
    )
    return CCDMol(
        FeatureContainer(nodes),
        FeatureContainer(residue),
        FeatureContainer({"id": NodeFeature(np.asarray([name]))}),
        index_table,
        metadata,
    )


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(components: Path, rdkit: Path, destination: Path):
    """Build atomically; old databases and source assets are never overwritten."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent)
    )
    try:
        from alphafold3.cpp import cif_dict

        af3_records = cif_dict.parse_multi_data_cif(components.read_bytes())
        cif = CIFFile.read(components)
        # Explicit migration from the caller's existing trusted local asset.
        with rdkit.open("rb") as handle:
            molecules = pickle.load(handle)  # noqa: S301
        env = lmdb.open(str(temporary), map_size=32 * 1024**3)
        glycans = {"glycans_linking": [], "glycans_other": []}
        unavailable = []
        try:
            for start in range(0, len(cif), 256):
                names = list(cif)[start : start + 256]
                with env.begin(write=True) as txn:
                    for name in names:
                        mol = record(
                            name, cif[name], molecules.get(name), af3_records[name]
                        )
                        txn.put(name.encode(), mol.to_bytes())
                        if not mol.metadata["fragmentation_available"]:
                            unavailable.append(name)
                        kind = (
                            mol.metadata["ccd_cif"]
                            .get("_chem_comp.type", [""])[0]
                            .lower()
                        )
                        if "saccharide" in kind:
                            glycans[
                                "glycans_linking"
                                if "linking" in kind
                                else "glycans_other"
                            ].append(name)
                print(f"CCD {min(start + 256, len(cif))}/{len(cif)}", flush=True)
            env.sync()
        finally:
            env.close()
        manifest = {
            "schema": SCHEMA,
            "serialization": "biomol-zstd-with-content-size",
            "entries": len(cif),
            "sources": {
                "components_sha256": sha256(components),
                "rdkit_sha256": sha256(rdkit),
            },
            "files": {
                "data": {"path": "data.mdb", "sha256": sha256(temporary / "data.mdb")}
            },
            "chemical_component_sets": glycans,
            "fragmentation_unavailable": unavailable,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
