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

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from biotite.structure import AtomArray

from foldforge.data.constants.chemistry import (
    DNA_CHAIN,
    LIGAND_CHAIN_TYPES,
    PROTEIN_CHAIN,
    RNA_CHAIN,
    STANDARD_POLYMER_CHAIN_TYPES,
    STD_RESIDUES_WITH_GAP,
)
from foldforge.data.io import load_json_cached
from foldforge.data.msa.alignment import MSA_GAP_IDX as RESIDUE_MSA_GAP_IDX
from foldforge.data.msa.alignment import MSA_GAP_IDX as STRUCTURAL_MSA_GAP_IDX
from foldforge.data.msa.alignment import (
    NUM_SEQ_NUM_RES_MSA_FEATURES,
    MutableFeatureDict,
    residue_map_to_standard,
    structural_map_to_standard,
)
from foldforge.data.msa.alignment import MSAPairingEngine as ResidueMSAPairingEngine
from foldforge.data.msa.alignment import MSAPairingEngine as StructuralMSAPairingEngine
from foldforge.data.msa.alignment import RawMsa as ResidueRawMsa
from foldforge.data.msa.alignment import RawMsa as StructuralRawMsa
from foldforge.utils.logging import get_logger

logger = get_logger(__name__)


class MSASourceManager:
    """Manages MSA data retrieval and loading from multiple sources.

    Args:
        raw_paths: List of base paths for MSA storage.
        methods: List of indexing methods (e.g., 'sequence' or 'pdb_id').
        mappings: Dictionary mapping source index to its respective lookup table.
        enabled: Whether MSA loading is enabled.

    """

    def __init__(
        self,
        raw_paths: Sequence[str],
        methods: Sequence[str],
        mappings: dict[int, dict[Any, Any]],
        *,
        enabled: bool,
    ) -> None:
        self.raw_paths = raw_paths
        self.methods = methods
        self.mappings = mappings
        self.enabled = enabled

    def fetch_msas(
        self,
        sequence: str,
        pdb_id: str,
        chain_type: str,
        p_dbs: Sequence[str] | None = None,
        np_dbs: Sequence[str] | None = None,
    ) -> tuple[list[ResidueRawMsa], list[ResidueRawMsa]]:
        """Compute fetch msas.

        Fetch MSAs from the configured paths based on chain type and indexing methods.

        Args:
            sequence: Query sequence.
            pdb_id: PDB identifier.
            chain_type: Type of the chain (e.g., PROTEIN_CHAIN, RNA_CHAIN).
            p_dbs: Databases for paired MSA search.
            np_dbs: Databases for unpaired MSA search.

        Returns:
            A tuple of (unpaired_msas, paired_msas).

        """
        if not self.enabled:
            return [], []
        unpaired, paired = [], []

        # RNA-specific loading logic
        if chain_type == RNA_CHAIN:
            for path, method, m_key in zip(
                self.raw_paths, self.methods, self.mappings, strict=False
            ):
                if method != "sequence":
                    continue
                mapping = self.mappings[m_key]
                if sequence not in mapping:
                    continue
                eid = str(mapping[sequence][0])
                fpath = str(Path(path).joinpath(eid, f"{eid}_all.a3m"))
                if Path(fpath).exists():
                    with Path(fpath).open() as f:
                        content = f.read()
                    if content:
                        unpaired.append(
                            ResidueRawMsa.from_a3m(
                                sequence,
                                RNA_CHAIN,
                                content,
                                depth_limit=30000,
                                dedup=False,
                            )
                        )
            return unpaired, []

        # Protein-specific loading logic
        if chain_type == PROTEIN_CHAIN:
            p_dbs = p_dbs or []
            np_dbs = np_dbs or []
            for p_db, np_db, path, method, m_key in zip(
                p_dbs, np_dbs, self.raw_paths, self.methods, self.mappings, strict=False
            ):
                key = sequence if method == "sequence" else str(pdb_id)
                mapping = self.mappings[m_key]
                if key not in mapping:
                    continue
                dir_path = str(Path(path).joinpath(str(mapping[key])))

                if p_db:
                    for pat in [f"{p_db}.a3m"]:
                        fpath = str(Path(dir_path).joinpath(pat))
                        if Path(fpath).exists():
                            with Path(fpath).open() as f:
                                content = f.read()
                            if content:
                                paired.append(
                                    ResidueRawMsa.from_a3m(
                                        sequence, PROTEIN_CHAIN, content, dedup=False
                                    )
                                )
                            break
                if np_db:
                    for db in np_db.split("-"):
                        for pat in [f"{db}.a3m"]:
                            fpath = str(Path(dir_path).joinpath(pat))
                            if Path(fpath).exists():
                                with Path(fpath).open() as f:
                                    content = f.read()
                                if content:
                                    unpaired.append(
                                        ResidueRawMsa.from_a3m(
                                            sequence,
                                            PROTEIN_CHAIN,
                                            content,
                                            dedup=False,
                                        )
                                    )
                                break
        return unpaired, paired


class _CommonFeatureAssemblyLine:
    """Represent common feature assembly line."""

    def __init__(
        self, max_msa_size: int = 16384, max_paired_per_species: int = 600
    ) -> None:
        self.max_size = max_msa_size
        self.max_paired_per_sp = max_paired_per_species


class ResidueFeatureAssemblyLine(_CommonFeatureAssemblyLine):
    """Orchestrates the conversion of Raw MSAs into finalized Protenix features.

    Args:
        max_msa_size: Maximum number of sequences allowed in the final MSA.
        max_paired_per_species: Maximum number of paired sequences per species.

    """

    def assemble(
        self, bioassembly: Mapping[int, Mapping[str, Any]], std_idxs: np.ndarray
    ) -> "ResidueMSAFeat":
        """Execute the complete feature assembly pipeline.

        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            std_idxs: Array of standardized residue indices.

        Returns:
            An assembled MSAFeat object.

        """
        # 1. Base featurization
        unique_prot_seqs = {
            v["sequence"]
            for v in bioassembly.values()
            if v["chain_entity_type"] == PROTEIN_CHAIN
        }
        need_pairing = len(unique_prot_seqs) > 1
        active_chain_ids = {v["chain_id"] for v in bioassembly.values()}

        raw_chains = []
        for aid, info in bioassembly.items():
            ctype, seq = info["chain_entity_type"], info["sequence"]
            skip = ctype not in STANDARD_POLYMER_CHAIN_TYPES or len(seq) <= 4

            if ctype in STANDARD_POLYMER_CHAIN_TYPES:
                up_msa = ResidueRawMsa.from_a3m(
                    seq,
                    ctype,
                    (
                        info["unpaired_msa"]
                        if not skip and ctype in [PROTEIN_CHAIN, RNA_CHAIN]
                        else ""
                    ),
                    dedup=True,
                )
                p_msa = ResidueRawMsa.from_a3m(
                    seq,
                    ctype,
                    (
                        info["paired_msa"]
                        if not skip and need_pairing and ctype == PROTEIN_CHAIN
                        else ""
                    ),
                    dedup=False,
                )
            else:
                up_msa = p_msa = ResidueRawMsa(
                    seq, PROTEIN_CHAIN, [], [], deduplicate=False
                )  # Ligand placeholders

            u_f, p_f = up_msa.featurize(), p_msa.featurize()
            chain_feat = dict(u_f)
            chain_feat.update({f"{k}_all_seq": v for k, v in p_f.items()})
            chain_feat.update(
                {
                    "asym_id": np.full(len(seq), aid),
                    "chain_id": info["chain_id"],
                    "entity_id": info["entity_id"],
                }
            )
            # Compute Profile
            msa = chain_feat["msa"]
            prof = (msa[..., None] == np.arange(len(STD_RESIDUES_WITH_GAP))).sum(
                axis=0
            ) / msa.shape[0]
            chain_feat.update(
                {
                    "profile": prof.astype(np.float32),
                    "deletion_mean": np.mean(chain_feat["deletion_matrix"], axis=0),
                }
            )
            raw_chains.append(chain_feat)

        # 2. Pairing and cleanup
        max_p = self.max_size // 2
        if need_pairing:
            raw_chains = ResidueMSAPairingEngine.pair_chains_by_species(
                raw_chains, max_p, active_chain_ids, self.max_paired_per_sp
            )
            raw_chains = ResidueMSAPairingEngine.cleanup_unpaired_features(raw_chains)

        # 3. Filter all-gap rows
        nonempty_asyms = [
            c["asym_id"][0] for c in raw_chains if c["chain_id"] in active_chain_ids
        ]
        if "msa_all_seq" in raw_chains[0]:
            raw_chains = ResidueMSAPairingEngine.filter_all_gapped_rows(
                raw_chains, nonempty_asyms
            )

        # 4. Cropping and merging
        cropped = []
        for c in raw_chains:
            p_msa = c.get("msa_all_seq")
            ps = min(p_msa.shape[0], max_p) if p_msa is not None else 0
            us = max(0, min(c["msa"].shape[0], self.max_size - ps))

            cr = {
                "asym_id": c["asym_id"],
                "chain_id": c["chain_id"],
                "profile": c["profile"],
                "deletion_mean": c["deletion_mean"],
            }
            for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
                if k in c:
                    cr[k] = c[k][:us]
                if f"{k}_all_seq" in c:
                    cr[f"{k}_all_seq"] = c[f"{k}_all_seq"][:ps]
            cropped.append(cr)

        merged = {"asym_id": np.concatenate([c["asym_id"] for c in cropped])}
        for base in NUM_SEQ_NUM_RES_MSA_FEATURES:
            for f in [base, f"{base}_all_seq"]:
                if f in cropped[0]:
                    merged[f] = ResidueMSAPairingEngine.merge_chain_features(cropped, f)
        for f in ["profile", "deletion_mean"]:
            merged[f] = np.concatenate([c[f] for c in cropped])

        # 5. Depth tracking
        active_set = set(nonempty_asyms)
        max_u = max([len(c["msa"]) for c in cropped if c["asym_id"][0] in active_set])
        rna_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == RNA_CHAIN
            ]
        )
        prot_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == PROTEIN_CHAIN
            ]
        )

        merged["msa"] = merged["msa"][:max_u]
        prot_p = 1
        if "msa_all_seq" in merged:
            max_p_actual = max(
                [
                    len(c["msa_all_seq"])
                    for c in cropped
                    if c["asym_id"][0] in active_set
                ]
            )
            merged["msa_all_seq"] = merged["msa_all_seq"][:max_p_actual]
            prot_p = max_p_actual

        # 6. Final integration and coordinate mapping
        for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if k in merged and f"{k}_all_seq" in merged:
                merged[k] = np.concatenate([merged[f"{k}_all_seq"], merged[k]], axis=0)

        # Forward compatibility patch for non-protein entities
        for aid in [
            aid
            for aid, info in bioassembly.items()
            if info["chain_entity_type"] != PROTEIN_CHAIN
        ]:
            cols = np.where(merged["asym_id"] == aid)[0]
            if cols.size > 0:
                gap_mask = np.all(merged["msa"][:, cols] == RESIDUE_MSA_GAP_IDX, axis=1)
                merged["msa"][np.ix_(np.where(gap_mask)[0], cols)] = merged["msa"][
                    0, cols
                ]

        for f in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if f in merged:
                merged[f] = merged[f][:, std_idxs].copy()
        for f in ["profile", "deletion_mean"]:
            merged[f] = merged[f][std_idxs]

        def to_i8(x: np.ndarray) -> np.ndarray:
            """Convert to i8."""
            return np.clip(x, -128, 127).astype(np.int8)

        return ResidueMSAFeat(
            rows=to_i8(merged["msa"]),
            mask=np.ones_like(merged["msa"], dtype=bool),
            deletion_matrix=to_i8(merged["deletion_matrix"]),
            profile=merged["profile"],
            deletion_mean=merged["deletion_mean"],
            prot_unpaired_num_alignments=np.array(prot_u, dtype=np.int32),
            prot_paired_num_alignments=np.array(prot_p, dtype=np.int32),
            rna_unpaired_num_alignments=np.array(rna_u, dtype=np.int32),
        )


class MSAFeaturizer:
    """Represent m s a featurizer.

    Main entry point for MSA featurization, coordinating source management and
    assembly.

    Args:
        dataset_name: Name of the dataset.
        prot_seq_or_filename_to_msadir_jsons: JSON maps for protein MSA lookups.
        prot_msadir_raw_paths: Base paths for protein MSAs.
        rna_seq_or_filename_to_msadir_jsons: JSON maps for RNA MSA lookups.
        rna_msadir_raw_paths: Base paths for RNA MSAs.
        prot_pairing_dbs: List of databases for protein pairing.
        prot_non_pairing_dbs: List of databases for protein non-pairing.
        prot_indexing_methods: Methods for protein MSA indexing.
        rna_indexing_methods: Methods for RNA MSA indexing.
        enable_prot_msa: Whether to enable protein MSA processing.
        enable_rna_msa: Whether to enable RNA MSA processing.

    """

    def __init__(
        self,
        dataset_name: str = "",
        prot_seq_or_filename_to_msadir_jsons: Sequence[str] = [""],
        prot_msadir_raw_paths: Sequence[str] = [""],
        rna_seq_or_filename_to_msadir_jsons: Sequence[str] = [""],
        rna_msadir_raw_paths: Sequence[str] = [""],
        prot_pairing_dbs: Sequence[str] = [""],
        prot_non_pairing_dbs: Sequence[str] = [""],
        prot_indexing_methods: Sequence[str] = ["sequence"],
        rna_indexing_methods: Sequence[str] = ["sequence"],
        *,
        enable_prot_msa: bool = True,
        enable_rna_msa: bool = True,
    ) -> None:
        self.dataset_name = dataset_name
        super().__init__()
        # Initialize source managers for protein and RNA
        self.prot_mgr = MSASourceManager(
            prot_msadir_raw_paths,
            prot_indexing_methods,
            {
                i: load_json_cached(p)
                for i, p in enumerate(prot_seq_or_filename_to_msadir_jsons)
            },
            enabled=enable_prot_msa,
        )
        self.rna_mgr = MSASourceManager(
            rna_msadir_raw_paths,
            rna_indexing_methods,
            {
                i: load_json_cached(p)
                for i, p in enumerate(rna_seq_or_filename_to_msadir_jsons)
            },
            enabled=enable_rna_msa,
        )
        self.prot_p_dbs = prot_pairing_dbs
        self.prot_np_dbs = prot_non_pairing_dbs
        self._profile: dict[str, Any] = {}
        logger.info("MSAFeaturizer for %s initialized.", f"{dataset_name}")

    def set_last_profile(self, p: dict[str, Any]) -> None:
        """Set the internal profile state."""
        self._profile = p

    def get_last_profile(self) -> dict[str, Any]:
        """Return the internal profile state."""
        return self._profile

    def make_msa_features(
        self,
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray | None,
        entity_to_asym_id_int: Mapping[str, Sequence[int]],
    ) -> dict[str, Any]:
        """Process bioassembly information into a dictionary of MSA features.

        Args:
            bioassembly_dict: Dictionary containing biological assembly data.
            selected_indices: Optional array of indices to select from the token array.
            entity_to_asym_id_int: Mapping from entity ID to asymmetric IDs.

        Returns:
            A dictionary containing processed MSA features.

        """
        atom_array, token_array = (
            bioassembly_dict["atom_array"],
            bioassembly_dict["token_array"],
        )
        sel_tokens = (
            token_array[selected_indices]
            if selected_indices is not None
            else token_array
        )
        sel_asyms = set(
            atom_array[sel_tokens.get_annotation("centre_atom_index")].asym_id_int
        )

        # 1. Resolve metadata and fetch MSAs
        meta = {}
        poly_map = {
            "polypeptide(L)": PROTEIN_CHAIN,
            "polyribonucleotide": RNA_CHAIN,
            "polydeoxyribonucleotide": DNA_CHAIN,
        }
        for eid, asyms in entity_to_asym_id_int.items():
            for aid in [a for a in asyms if a in sel_asyms]:
                seq = bioassembly_dict["sequences"].get(eid) or (
                    "X" * (atom_array.asym_id_int == aid).sum()
                )
                ctype = poly_map.get(
                    bioassembly_dict["entity_poly_type"].get(eid, "non-polymer"),
                    "non-polymer",
                )

                up_msas, p_msas = [], []
                if ctype == RNA_CHAIN:
                    up_msas, _ = self.rna_mgr.fetch_msas(seq, "", ctype)
                elif ctype == PROTEIN_CHAIN:
                    up_msas, p_msas = self.prot_mgr.fetch_msas(
                        seq,
                        bioassembly_dict["pdb_id"],
                        ctype,
                        self.prot_p_dbs,
                        self.prot_np_dbs,
                    )

                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "chain_entity_type": ctype,
                    "paired_msa": ResidueRawMsa.merge(msas=p_msas).to_a3m()
                    if p_msas
                    else "",
                    "unpaired_msa": ResidueRawMsa.merge(msas=up_msas).to_a3m()
                    if up_msas
                    else "",
                }

        # 2. Map coordinates and assemble features
        ca = atom_array[sel_tokens.get_annotation("centre_atom_index")]
        std_idxs = residue_map_to_standard(ca.asym_id_int, ca.res_id, meta)

        res = ResidueFeatureAssemblyLine().assemble(meta, std_idxs).to_dict()
        keep = {
            "msa",
            "has_deletion",
            "deletion_value",
            "deletion_mean",
            "profile",
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        }
        return {k: v for k, v in res.items() if k in keep}

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call make_msa_features."""
        return self.make_msa_features(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ResidueMSAFeat:
    """Container for finalized numerical MSA features."""

    rows: np.ndarray
    mask: np.ndarray
    deletion_matrix: np.ndarray
    profile: np.ndarray
    deletion_mean: np.ndarray
    prot_unpaired_num_alignments: np.ndarray
    prot_paired_num_alignments: np.ndarray
    rna_unpaired_num_alignments: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert the MSA object into a standard Protenix data dictionary."""
        return {
            "msa": self.rows,
            "msa_mask": self.mask,
            "deletion_matrix": self.deletion_matrix,
            "deletion_value": (np.arctan(self.deletion_matrix / 3.0) * (2.0 / np.pi)),
            "has_deletion": np.clip(self.deletion_matrix, 0.0, 1.0),
            "profile": self.profile,
            "deletion_mean": self.deletion_mean,
            "prot_unpaired_num_alignments": self.prot_unpaired_num_alignments,
            "prot_paired_num_alignments": self.prot_paired_num_alignments,
            "rna_unpaired_num_alignments": self.rna_unpaired_num_alignments,
            "rna_pair_num_alignments": np.asarray(1, dtype=np.int32),
            "prot_pair_num_alignments": self.prot_paired_num_alignments,
            "prot_unpair_num_alignments": self.prot_unpaired_num_alignments,
            "rna_unpair_num_alignments": self.rna_unpaired_num_alignments,
        }


def ensure_ends_with_newline(s: str | None) -> str | None:
    r"""Ensure the given string ends with a newline character.

    If the string is non-empty and does not already end with '\\n',
    append '\\n'. Empty strings are returned unchanged.

    Args:
        s (str): Input string.

    Returns:
        str: The input string guaranteed to end with '\\n' when non-empty.

    """
    if not s:
        return s
    if not s.endswith("\n"):
        s += "\n"
    return s


class ResidueInferenceMSAFeaturizer:
    """Represent residue inference m s a featurizer.

    Specialized featurizer for inference scenarios, leveraging the unified assembly
    line.
    """

    @staticmethod
    def make_msa_feature(
        bioassembly: Sequence[dict[str, Any]],
        atom_array: AtomArray,
        *,
        msa_pair_as_unpair: bool = False,
        use_rna_msa: bool = True,
    ) -> dict[str, Any]:
        """Prepare MSA features during inference from bioassembly structure.

        Args:
            bioassembly: List of entities in the biological assembly.
            atom_array: Structural data array.
            msa_pair_as_unpair: Whether to treat paired MSA as unpaired.
            use_rna_msa: Whether to use MSA for RNA chains.

        Returns:
            Dictionary of processed MSA features.

        """
        meta, curr_aid = {}, 0
        for eid, info in enumerate(bioassembly):
            seq, count, ctype, u_a3m, p_a3m = "", 0, "non-polymer", None, None
            if "proteinChain" in info:
                c = info["proteinChain"]
                seq, count, ctype, u_a3m, p_a3m = (
                    c["sequence"],
                    c["count"],
                    PROTEIN_CHAIN,
                    c.get("unpairedMsa"),
                    c.get("pairedMsa"),
                )
                if u_a3m is None and c.get("unpairedMsaPath"):
                    with Path(c["unpairedMsaPath"]).open() as f:
                        u_a3m = f.read()
                if p_a3m is None and c.get("pairedMsaPath"):
                    with Path(c["pairedMsaPath"]).open() as f:
                        p_a3m = f.read()
                if u_a3m is None and (p_a3m is None) and c.get("msa"):
                    msa_dir = c["msa"].get("precomputed_msa_dir")
                    if msa_dir and Path(msa_dir).exists():
                        logger.warning(
                            "Use the old msa json format, change to "
                            "pairedMsaPath/unpairedMsaPath field for future use."
                        )
                        if Path(str(Path(msa_dir).joinpath("pairing.a3m"))).exists():
                            with Path(
                                str(Path(msa_dir).joinpath("pairing.a3m"))
                            ).open() as f:
                                p_a3m = f.read()
                        if Path(
                            str(Path(msa_dir).joinpath("non_pairing.a3m"))
                        ).exists():
                            with Path(
                                str(Path(msa_dir).joinpath("non_pairing.a3m"))
                            ).open() as f:
                                u_a3m = f.read()

            elif "rnaSequence" in info:
                c = info["rnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], RNA_CHAIN
                if use_rna_msa:
                    u_a3m = c.get("unpairedMsa")
                    if u_a3m is None and c.get("unpairedMsaPath"):
                        with Path(c["unpairedMsaPath"]).open() as f:
                            u_a3m = f.read()
            elif "dnaSequence" in info:
                c = info["dnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], DNA_CHAIN
            elif "ligand" in info:
                count, ctype, seq = (
                    info["ligand"]["count"],
                    "non-polymer",
                    "X" * (atom_array.asym_id_int == curr_aid).sum(),
                )

            p_a3m = ensure_ends_with_newline(p_a3m)
            u_a3m = ensure_ends_with_newline(u_a3m)

            if msa_pair_as_unpair and p_a3m:
                u_a3m = ResidueRawMsa.from_a3m(
                    seq, ctype, p_a3m + (u_a3m or ""), dedup=True
                ).to_a3m()

            for c_idx in range(count):
                aid = curr_aid + c_idx
                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": seq,
                    "paired_msa": p_a3m or "",
                    "unpaired_msa": u_a3m or "",
                    "chain_entity_type": ctype,
                }
            curr_aid += count

        ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
        std_idxs = residue_map_to_standard(ca.asym_id_int, ca.res_id, meta)
        return ResidueFeatureAssemblyLine().assemble(meta, std_idxs).to_dict()


class StructuralFeatureAssemblyLine(_CommonFeatureAssemblyLine):
    """Orchestrates the conversion of Raw MSAs into finalized OpenDDE features.

    Args:
        max_msa_size: Maximum number of sequences allowed in the final MSA.
        max_paired_per_species: Maximum number of paired sequences per species.

    """

    def assemble(
        self, bioassembly: Mapping[int, Mapping[str, Any]], std_idxs: np.ndarray
    ) -> "StructuralMSAFeat":
        """Execute the complete feature assembly pipeline.

        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            std_idxs: Array of standardized residue indices.

        Returns:
            An assembled MSAFeat object.

        """
        # 1. Base featurization
        unique_prot_seqs = {
            v["sequence"]
            for v in bioassembly.values()
            if v["chain_entity_type"] == PROTEIN_CHAIN
        }
        need_pairing = len(unique_prot_seqs) > 1
        active_chain_ids = {v["chain_id"] for v in bioassembly.values()}

        raw_chains: list[MutableFeatureDict] = []
        for aid, info in bioassembly.items():
            ctype, seq = info["chain_entity_type"], info["sequence"]
            skip = ctype not in STANDARD_POLYMER_CHAIN_TYPES or len(seq) <= 4

            if ctype in STANDARD_POLYMER_CHAIN_TYPES:
                up_msa = StructuralRawMsa.from_a3m(
                    seq,
                    ctype,
                    (
                        info["unpaired_msa"]
                        if not skip and ctype in [PROTEIN_CHAIN, RNA_CHAIN]
                        else ""
                    ),
                    dedup=True,
                )
                p_msa = StructuralRawMsa.from_a3m(
                    seq,
                    ctype,
                    (
                        info["paired_msa"]
                        if not skip and need_pairing and ctype == PROTEIN_CHAIN
                        else ""
                    ),
                    dedup=False,
                )
            else:
                up_msa = p_msa = StructuralRawMsa(
                    seq, PROTEIN_CHAIN, [], [], deduplicate=False
                )  # Ligand placeholders

            u_f, p_f = up_msa.featurize(), p_msa.featurize()
            chain_feat = dict(u_f)
            chain_feat.update({f"{k}_all_seq": v for k, v in p_f.items()})
            chain_feat.update(
                {
                    "asym_id": np.full(len(seq), aid),
                    "chain_id": info["chain_id"],
                    "entity_id": info["entity_id"],
                }
            )
            # Compute Profile
            msa = chain_feat["msa"]
            prof = (msa[..., None] == np.arange(len(STD_RESIDUES_WITH_GAP))).sum(
                axis=0
            ) / msa.shape[0]
            chain_feat.update(
                {
                    "profile": prof.astype(np.float32),
                    "deletion_mean": np.mean(chain_feat["deletion_matrix"], axis=0),
                }
            )
            raw_chains.append(chain_feat)

        # 2. Pairing and cleanup
        max_p = self.max_size // 2
        if need_pairing:
            raw_chains = StructuralMSAPairingEngine.pair_chains_by_species(
                raw_chains, max_p, active_chain_ids, self.max_paired_per_sp
            )
            raw_chains = StructuralMSAPairingEngine.cleanup_unpaired_features(
                raw_chains
            )

        # 3. Filter all-gap rows
        nonempty_asyms = [
            c["asym_id"][0] for c in raw_chains if c["chain_id"] in active_chain_ids
        ]
        if "msa_all_seq" in raw_chains[0]:
            raw_chains = StructuralMSAPairingEngine.filter_all_gapped_rows(
                raw_chains, nonempty_asyms
            )

        # 4. Cropping and merging
        cropped: list[MutableFeatureDict] = []
        for c in raw_chains:
            p_msa = c.get("msa_all_seq")
            ps = min(p_msa.shape[0], max_p) if p_msa is not None else 0
            us = max(0, min(c["msa"].shape[0], self.max_size - ps))

            cr = {
                "asym_id": c["asym_id"],
                "chain_id": c["chain_id"],
                "profile": c["profile"],
                "deletion_mean": c["deletion_mean"],
            }
            for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
                if k in c:
                    cr[k] = c[k][:us]
                if f"{k}_all_seq" in c:
                    cr[f"{k}_all_seq"] = c[f"{k}_all_seq"][:ps]
            cropped.append(cr)

        merged = {"asym_id": np.concatenate([c["asym_id"] for c in cropped])}
        for base in NUM_SEQ_NUM_RES_MSA_FEATURES:
            for f in [base, f"{base}_all_seq"]:
                if f in cropped[0]:
                    merged[f] = StructuralMSAPairingEngine.merge_chain_features(
                        cropped, f
                    )
        for f in ["profile", "deletion_mean"]:
            merged[f] = np.concatenate([c[f] for c in cropped])

        # 5. Depth tracking
        active_set = set(nonempty_asyms)
        max_u = max([len(c["msa"]) for c in cropped if c["asym_id"][0] in active_set])
        rna_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == RNA_CHAIN
            ]
        )
        prot_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == PROTEIN_CHAIN
            ]
        )

        merged["msa"] = merged["msa"][:max_u]
        prot_p = 1
        if "msa_all_seq" in merged:
            max_p_actual = max(
                [
                    len(c["msa_all_seq"])
                    for c in cropped
                    if c["asym_id"][0] in active_set
                ]
            )
            merged["msa_all_seq"] = merged["msa_all_seq"][:max_p_actual]
            prot_p = max_p_actual

        # 6. Final integration and coordinate mapping
        for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if k in merged and f"{k}_all_seq" in merged:
                merged[k] = np.concatenate([merged[f"{k}_all_seq"], merged[k]], axis=0)

        # Forward compatibility patch for non-protein entities
        for aid in [
            aid
            for aid, info in bioassembly.items()
            if info["chain_entity_type"] != PROTEIN_CHAIN
        ]:
            cols = np.where(merged["asym_id"] == aid)[0]
            if cols.size > 0:
                gap_mask = np.all(
                    merged["msa"][:, cols] == STRUCTURAL_MSA_GAP_IDX, axis=1
                )
                merged["msa"][np.ix_(np.where(gap_mask)[0], cols)] = merged["msa"][
                    0, cols
                ]

        for f in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if f in merged:
                merged[f] = merged[f][:, std_idxs].copy()
        for f in ["profile", "deletion_mean"]:
            merged[f] = merged[f][std_idxs]

        def to_i8(x: np.ndarray) -> np.ndarray:
            """Convert to i8."""
            return np.clip(x, -128, 127).astype(np.int8)

        return StructuralMSAFeat(
            rows=to_i8(merged["msa"]),
            mask=np.ones_like(merged["msa"], dtype=bool),
            deletion_matrix=to_i8(merged["deletion_matrix"]),
            profile=merged["profile"],
            deletion_mean=merged["deletion_mean"],
            prot_unpaired_num_alignments=np.array(prot_u, dtype=np.int32),
            prot_paired_num_alignments=np.array(prot_p, dtype=np.int32),
            rna_unpaired_num_alignments=np.array(rna_u, dtype=np.int32),
        )


@dataclasses.dataclass(frozen=True)
class StructuralMSAFeat:
    """Container for finalized numerical MSA features."""

    rows: np.ndarray
    mask: np.ndarray
    deletion_matrix: np.ndarray
    profile: np.ndarray
    deletion_mean: np.ndarray
    prot_unpaired_num_alignments: np.ndarray
    prot_paired_num_alignments: np.ndarray
    rna_unpaired_num_alignments: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert the MSA object into a standard OpenDDE data dictionary."""
        rna_paired_num_alignments = np.asarray(1, dtype=np.int32)
        return {
            "msa": self.rows,
            "msa_mask": self.mask,
            "deletion_matrix": self.deletion_matrix,
            "deletion_value": (np.arctan(self.deletion_matrix / 3.0) * (2.0 / np.pi)),
            "has_deletion": np.clip(self.deletion_matrix, 0.0, 1.0),
            "profile": self.profile,
            "deletion_mean": self.deletion_mean,
            "prot_unpaired_num_alignments": self.prot_unpaired_num_alignments,
            "prot_paired_num_alignments": self.prot_paired_num_alignments,
            "rna_unpaired_num_alignments": self.rna_unpaired_num_alignments,
            "rna_paired_num_alignments": rna_paired_num_alignments,
            "rna_pair_num_alignments": rna_paired_num_alignments,
            "prot_pair_num_alignments": self.prot_paired_num_alignments,
            "prot_unpair_num_alignments": self.prot_unpaired_num_alignments,
            "rna_unpair_num_alignments": self.rna_unpaired_num_alignments,
        }


class StructuralInferenceMSAFeaturizer:
    """Represent structural inference m s a featurizer.

    Specialized featurizer for inference scenarios, leveraging the unified assembly
    line.
    """

    @staticmethod
    def make_msa_feature(
        bioassembly: Sequence[dict[str, Any]],
        atom_array: AtomArray,
        *,
        msa_pair_as_unpair: bool = False,
        use_rna_msa: bool = True,
    ) -> dict[str, Any]:
        """Prepare MSA features during inference from bioassembly structure.

        Args:
            bioassembly: List of entities in the biological assembly.
            atom_array: Structural data array.
            msa_pair_as_unpair: Whether to treat paired MSA as unpaired.
            use_rna_msa: Whether to use MSA for RNA chains.

        Returns:
            Dictionary of processed MSA features.

        """
        meta, curr_aid = {}, 0
        for eid, info in enumerate(bioassembly):
            seq = ""
            count = 0
            ctype: Any = LIGAND_CHAIN_TYPES
            u_a3m: str | None = None
            p_a3m: str | None = None
            if "proteinChain" in info:
                c = info["proteinChain"]
                seq, count, ctype, u_a3m, p_a3m = (
                    c["sequence"],
                    c["count"],
                    PROTEIN_CHAIN,
                    c.get("unpairedMsa"),
                    c.get("pairedMsa"),
                )
                if u_a3m is None and c.get("unpairedMsaPath"):
                    with Path(c["unpairedMsaPath"]).open() as f:
                        u_a3m = f.read()
                if p_a3m is None and c.get("pairedMsaPath"):
                    with Path(c["pairedMsaPath"]).open() as f:
                        p_a3m = f.read()
                if u_a3m is None and (p_a3m is None) and c.get("msa"):
                    msa_dir = c["msa"].get("precomputed_msa_dir")
                    if msa_dir and Path(msa_dir).exists():
                        logger.warning(
                            "Use the old msa json format, change to "
                            "pairedMsaPath/unpairedMsaPath field for future use."
                        )
                        if Path(str(Path(msa_dir).joinpath("pairing.a3m"))).exists():
                            with Path(
                                str(Path(msa_dir).joinpath("pairing.a3m"))
                            ).open() as f:
                                p_a3m = f.read()
                        if Path(
                            str(Path(msa_dir).joinpath("non_pairing.a3m"))
                        ).exists():
                            with Path(
                                str(Path(msa_dir).joinpath("non_pairing.a3m"))
                            ).open() as f:
                                u_a3m = f.read()

            elif "rnaSequence" in info:
                c = info["rnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], RNA_CHAIN
                if use_rna_msa:
                    u_a3m = c.get("unpairedMsa")
                    if u_a3m is None and c.get("unpairedMsaPath"):
                        with Path(c["unpairedMsaPath"]).open() as f:
                            u_a3m = f.read()
            elif "dnaSequence" in info:
                c = info["dnaSequence"]
                seq, count, ctype = c["sequence"], c["count"], DNA_CHAIN
            elif "ligand" in info:
                count, ctype, seq = (
                    info["ligand"]["count"],
                    "non-polymer",
                    "X" * (atom_array.asym_id_int == curr_aid).sum(),
                )
            elif "ion" in info:
                count, ctype = info["ion"]["count"], LIGAND_CHAIN_TYPES

            p_a3m = ensure_ends_with_newline(p_a3m)
            u_a3m = ensure_ends_with_newline(u_a3m)

            if msa_pair_as_unpair and p_a3m:
                u_a3m = StructuralRawMsa.from_a3m(
                    seq, cast("str", ctype), p_a3m + (u_a3m or ""), dedup=True
                ).to_a3m()

            for c_idx in range(count):
                aid = curr_aid + c_idx
                chain_seq = seq
                if "ion" in info:
                    chain_seq = "X" * np.count_nonzero(
                        (atom_array.asym_id_int == aid)
                        & atom_array.centre_atom_mask.astype(bool)
                    )
                meta[aid] = {
                    "entity_id": eid,
                    "chain_id": atom_array.chain_id[atom_array.asym_id_int == aid][0],
                    "sequence": chain_seq,
                    "paired_msa": p_a3m or "",
                    "unpaired_msa": u_a3m or "",
                    "chain_entity_type": ctype,
                }
            curr_aid += count

        ca = atom_array[atom_array.centre_atom_mask.astype(bool)]
        std_idxs = structural_map_to_standard(ca.asym_id_int, ca.res_id, meta)
        return StructuralFeatureAssemblyLine().assemble(meta, std_idxs).to_dict()
