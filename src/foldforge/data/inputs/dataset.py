# Optional chemistry/model dependencies load only at their execution boundary.
# ruff: noqa: PLC0415
from __future__ import annotations

import json
import logging
import time
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from foldforge.data.ccd import components as ccd
from foldforge.data.features.atoms import (
    residue_data_type_transform,
    residue_make_dummy_feature,
    structural_data_type_transform,
    structural_make_dummy_feature,
)
from foldforge.data.inputs.features import (
    ResidueSampleDictToFeatures,
    StructuralSampleDictToFeatures,
)
from foldforge.data.inputs.validation import validate_inference_jobs
from foldforge.data.msa.featurizer import (
    ResidueInferenceMSAFeaturizer,
    StructuralInferenceMSAFeaturizer,
)
from foldforge.data.template.features import (
    ResidueTemplateHitFeaturizer,
    StructuralTemplateHitFeaturizer,
)
from foldforge.data.template.featurizer import (
    ResidueInferenceTemplateFeaturizer,
    StructuralInferenceTemplateFeaturizer,
)
from foldforge.utils.distributed import DIST_WRAPPER
from foldforge.utils.tensor import collate_fn_identity, dict_to_tensor

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
    from collections.abc import Iterable, Iterator, Mapping, Sized

    from biotite.structure import AtomArray

    from foldforge.models.config.types import ConfigNode

logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", module="biotite")


def residue_get_inference_dataloader(configs: Any) -> DataLoader:
    """Create and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the
            DataLoader.

    Returns:
        A DataLoader object configured for inference.

    """
    inference_dataset = ResidueInferenceDataset(
        configs=configs,
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    return DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_identity,
        num_workers=configs.num_workers,
    )


class ResidueInferenceDataset(Dataset):
    """Represent residue inference dataset."""

    def __init__(
        self,
        configs: ConfigNode,
    ) -> None:
        self.configs = configs

        self.input_json_path = configs.input_json_path
        self.dump_dir = configs.dump_dir
        self.use_msa = configs.use_msa
        self.msa_pair_as_unpair = configs.get("msa_pair_as_unpair", True)
        self.use_rna_msa = configs.get("use_rna_msa", True)
        self.use_template = configs.get("use_template", True)
        with Path(self.input_json_path).open() as f:
            self.inputs = json.load(f)
        json_task_name = Path(self.input_json_path).name.split(".")[0]
        self.use_rna_msa = self.use_rna_msa or any(
            chain.get("rnaSequence", {}).get("unpairedMsa")
            or chain.get("rnaSequence", {}).get("unpairedMsaPath")
            for target in self.inputs
            for chain in target["sequences"]
        )
        needs_template_search = any(
            chain.get("proteinChain", {}).get("templatesPath")
            for target in self.inputs
            for chain in target["sequences"]
        )
        if self.use_template and needs_template_search:
            template_mmcif_dir = configs.data.template.prot_template_mmcif_dir
            fetch_remote = configs.data.template.get("fetch_remote", True)
            if not fetch_remote:
                if not (
                    template_mmcif_dir is not None and Path(template_mmcif_dir).exists()
                ):
                    message = (
                        "Template search requires "
                        "data.template.prot_template_mmcif_dir "
                        "to point to an existing mmCIF directory when "
                        "data.template.fetch_remote=false. Configure that directory, "
                        "enable remote template fetching, or supply aligned templates "
                        "through template-lmdb."
                    )
                    raise ValueError(message)
            elif template_mmcif_dir:
                Path(template_mmcif_dir).mkdir(parents=True, exist_ok=True)
            self.online_template_featurizer = ResidueTemplateHitFeaturizer(
                mmcif_dir=configs.data.template.prot_template_mmcif_dir,
                template_cache_dir=configs.data.template.prot_template_cache_dir,
                max_hits=4,
                kalign_binary_path=configs.data.template.kalign_binary_path,
                max_template_date="2021-09-30",
                release_dates_path=configs.data.template.release_dates_path,
                obsolete_pdbs_path=configs.data.template.obsolete_pdbs_path,
                _shuffle_top_k_prefiltered=None,
                _max_template_candidates_num=20,
                fetch_remote=fetch_remote,
            )
        else:
            self.online_template_featurizer = None
        esm_info = configs.get("esm", {})
        configs.esm.embedding_dir = f"./esm_embeddings/{configs.esm.model_name}"
        configs.esm.sequence_fpath = (
            f"./esm_embeddings/{json_task_name}_prot_sequences.csv"
        )
        self.esm_enable = esm_info.get("enable", False)
        if self.esm_enable:
            from foldforge.data.features.embeddings import ESMFeaturizer

            Path(configs.esm.embedding_dir).mkdir(parents=True, exist_ok=True)
            Path(str(Path(configs.esm.sequence_fpath).parent)).mkdir(
                parents=True, exist_ok=True
            )
            ESMFeaturizer.precompute_esm_embedding(
                self.inputs,
                configs.esm.model_name,
                configs.esm.embedding_dir,
                configs.esm.sequence_fpath,
                configs.load_checkpoint_dir,
            )
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=esm_info.embedding_dir,
                sequence_fpath=esm_info.sequence_fpath,
                embedding_dim=esm_info.embedding_dim,
                error_dir="./esm_embeddings/",
            )

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """Process one.

        Process a single sample from the input JSON to generate features and
        statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.

        """
        # general features
        t0 = time.time()
        sample2feat = ResidueSampleDictToFeatures(
            single_sample_dict=dict(single_sample_dict),
            extract_features_for_tfg=self.configs.sample_diffusion.guidance.enable,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type_and_seqs = (
            sample2feat.entity_poly_type_and_seqs
        )  # we include ligand as well
        t1 = time.time()
        msa_features = (
            ResidueInferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                atom_array=atom_array,
                msa_pair_as_unpair=self.msa_pair_as_unpair,
                use_rna_msa=self.use_rna_msa,
            )
            if self.use_msa
            else {}
        )
        template_features = ResidueInferenceTemplateFeaturizer.make_template_feature(
            bioassembly=single_sample_dict["sequences"],
            atom_array=atom_array,
            use_template=self.use_template,
            online_template_featurizer=self.online_template_featurizer,
        )
        # Esm features
        if self.esm_enable:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array,
                bioassembly_dict=single_sample_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            features_dict.update(dict_to_tensor(template_features))
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = residue_make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = residue_data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat
        # Add dimension related items
        n_token = feat["token_index"].shape[0]
        n_atom = feat["atom_to_token_idx"].shape[0]
        n_msa = feat["msa"].shape[0]
        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )
        n_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([n_asym]),
                "N_token": torch.tensor([n_token]),
                "N_atom": torch.tensor([n_atom]),
                "N_msa": torch.tensor([n_msa]),
            }
        )

        def formatted_key(key: str) -> str:
            """Compute formatted key."""
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type_and_seqs["entity_poly_type"]})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], AtomArray | None, str]:
        single_sample_dict = self.inputs[index]
        sample_name = single_sample_dict["name"]
        data: dict[str, Any]
        try:
            logger.info("Featurizing %s...", f"{sample_name}")

            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:  # noqa: BLE001 - per-job exception is returned to the caller
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, atom_array, error_message


class InferenceJobSampler(Sampler[int]):
    """Shard whole inference jobs across DP groups and replicate within CP.

    All ranks in one group use the same ``rank`` here, so they receive
    the same job sequence. DP groups are padded to the same number of forwards
    because creates process groups collectively during model execution;
    ``owns`` identifies real assignments so padded work never writes outputs.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        num_replicas: int,
        rank: int,
        sample_indices: Iterable[int] | None = None,
    ) -> None:
        if num_replicas < 1:
            msg = "num_replicas must be >= 1"
            raise ValueError(msg)
        if not 0 <= rank < num_replicas:
            msg = "rank must be in [0, num_replicas)"
            raise ValueError(msg)
        self.data_source = data_source
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.set_sample_indices(sample_indices)

    def set_sample_indices(self, sample_indices: Iterable[int] | None) -> None:
        """Set sample indices."""
        indices = (
            list(range(len(self.data_source)))
            if sample_indices is None
            else [int(index) for index in sample_indices]
        )
        if any(index < 0 or index >= len(self.data_source) for index in indices):
            msg = "inference sample index is out of range"
            raise IndexError(msg)
        self._sample_indices = indices
        if not indices:
            self._local_indices = []
            self._owned_indices: set[int] = set()
            return
        num_samples = (len(indices) + self.num_replicas - 1) // self.num_replicas
        total_size = num_samples * self.num_replicas
        padding_size = total_size - len(indices)
        padded = indices + (indices * (padding_size // len(indices) + 1))[:padding_size]
        positions = range(self.rank, total_size, self.num_replicas)
        self._local_indices = [padded[position] for position in positions]
        self._owned_indices = {
            padded[position]
            for position in range(self.rank, len(indices), self.num_replicas)
        }

    def __iter__(self) -> Iterator[int]:
        return iter(self._local_indices)

    def __len__(self) -> int:
        return len(self._local_indices)

    def owns(self, sample_index: int) -> bool:
        """Return whether this DP rank owns, rather than pads, this job."""
        return int(sample_index) in self._owned_indices


def _data_parallel_coordinates(configs: Any) -> tuple[int, int]:  # noqa: ARG001 - shared callback or fixture signature
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()
    else:
        world_size = DIST_WRAPPER.world_size
        world_rank = DIST_WRAPPER.rank
    return (world_size, world_rank)


def structural_get_inference_dataloader(
    configs: Any, *, inputs: list[dict[str, Any]] | None = None
) -> DataLoader:
    """Create and returns a DataLoader for inference using the InferenceDataset.

    Args:
        inputs: Optional preloaded inference records.
        configs: A configuration object containing the necessary parameters for the
            DataLoader.

    Returns:
        A DataLoader object configured for inference.

    """
    inference_dataset = StructuralInferenceDataset(configs=configs, inputs=inputs)
    size_dp, dp_rank = _data_parallel_coordinates(configs)
    sampler = InferenceJobSampler(inference_dataset, num_replicas=size_dp, rank=dp_rank)
    return DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_identity,
        num_workers=configs.num_workers,
    )


class StructuralInferenceDataset(Dataset):
    """Represent structural inference dataset."""

    def __init__(
        self, configs: ConfigNode, inputs: list[dict[str, Any]] | None = None
    ) -> None:
        self.configs = configs
        self.input_json_path = configs.input_json_path
        self.dump_dir = configs.dump_dir
        self.use_msa = configs.use_msa
        self.msa_pair_as_unpair = configs.get("msa_pair_as_unpair", True)
        self.use_rna_msa = configs.get("use_rna_msa", True)
        self.use_template = configs.get("use_template", True)
        ccd.set_ccd_cache_paths(
            components_file=configs.data.ccd_components_file,
            rdkit_mol_pkl=configs.data.ccd_components_rdkit_mol_file,
        )
        if inputs is None:
            with Path(self.input_json_path).open() as f:
                inputs = validate_inference_jobs(json.load(f))
        self.inputs = cast("list[dict[str, Any]]", inputs)
        self.use_rna_msa = self.use_rna_msa or any(
            chain.get("rnaSequence", {}).get("unpairedMsa")
            or chain.get("rnaSequence", {}).get("unpairedMsaPath")
            for target in self.inputs
            for chain in target["sequences"]
        )
        needs_template_search = any(
            chain.get("proteinChain", {}).get("templatesPath")
            for target in self.inputs
            for chain in target["sequences"]
        )
        if self.use_template and needs_template_search:
            template_mmcif_dir = configs.data.template.prot_template_mmcif_dir
            fetch_remote = configs.data.template.get("fetch_remote", True)
            if not fetch_remote:
                if not (
                    template_mmcif_dir is not None and Path(template_mmcif_dir).exists()
                ):
                    message = (
                        "Template search requires "
                        "data.template.prot_template_mmcif_dir "
                        "to point to an existing mmCIF directory when "
                        "data.template.fetch_remote=false. Configure that directory, "
                        "enable remote template fetching, or supply aligned templates "
                        "through template-lmdb."
                    )
                    raise ValueError(message)
            elif template_mmcif_dir:
                Path(template_mmcif_dir).mkdir(parents=True, exist_ok=True)
            self.online_template_featurizer = StructuralTemplateHitFeaturizer(
                mmcif_dir=configs.data.template.prot_template_mmcif_dir,
                template_cache_dir=configs.data.template.prot_template_cache_dir,
                max_hits=4,
                kalign_binary_path=configs.data.template.kalign_binary_path,
                max_template_date="2021-09-30",
                release_dates_path=configs.data.template.release_dates_path,
                obsolete_pdbs_path=configs.data.template.obsolete_pdbs_path,
                _shuffle_top_k_prefiltered=None,
                _max_template_candidates_num=20,
                fetch_remote=fetch_remote,
            )
        else:
            self.online_template_featurizer = None

    def process_one(
        self, single_sample_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], AtomArray, dict[str, float]]:
        """Process one.

        Process a single sample from the input JSON to generate features and
        statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.

        """
        t0 = time.time()
        sample_diffusion_config = self.configs.sample_diffusion
        sample_diffusion_dict = sample_diffusion_config.to_dict()
        guidance_config = sample_diffusion_dict.get("guidance") or {}
        need_geometry_features = bool(guidance_config.get("enable", False))
        sample2feat = StructuralSampleDictToFeatures(
            single_sample_dict=dict(single_sample_dict),
            extract_features_for_tfg=need_geometry_features,
        )
        features_dict, atom_array, _token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type_and_seqs = sample2feat.entity_poly_type_and_seqs
        t1 = time.time()
        msa_features = (
            StructuralInferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                atom_array=atom_array,
                msa_pair_as_unpair=self.msa_pair_as_unpair,
                use_rna_msa=self.use_rna_msa,
            )
            if self.use_msa
            else {}
        )
        template_features = StructuralInferenceTemplateFeaturizer.make_template_feature(
            bioassembly=single_sample_dict["sequences"],
            atom_array=atom_array,
            use_template=self.use_template,
            online_template_featurizer=self.online_template_featurizer,
        )
        dummy_feats = []
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            features_dict.update(dict_to_tensor(template_features))
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = structural_make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        feat = structural_data_type_transform(feat_or_label_dict=features_dict)
        t2 = time.time()
        data: dict[str, Any] = {}
        data["input_feature_dict"] = feat
        n_token = feat["token_index"].shape[0]
        n_atom = feat["atom_to_token_idx"].shape[0]
        n_msa = feat["msa"].shape[0]
        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )
        n_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([n_asym]),
                "N_token": torch.tensor([n_token]),
                "N_atom": torch.tensor([n_atom]),
                "N_msa": torch.tensor([n_msa]),
            }
        )

        def formatted_key(key: str) -> str:
            """Compute formatted key."""
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type_and_seqs["entity_poly_type"]})
        time_tracker = {"parse": t1 - t0, "featurizer": t2 - t1}
        return (data, atom_array, time_tracker)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], AtomArray | None, str]:
        sample_name = f"job_{index}"
        single_sample_dict = self.inputs[index]
        sample_name = single_sample_dict["name"]
        data: dict[str, Any]
        try:
            logger.info("Featurizing %s...", f"{sample_name}")
            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:  # noqa: BLE001 - per-job exception is returned to the caller
            data, atom_array = ({}, None)
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = sample_name
        data["sample_index"] = index
        return (data, atom_array, error_message)
