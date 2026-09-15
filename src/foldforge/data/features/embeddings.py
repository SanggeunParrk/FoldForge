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
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch

from foldforge.data.features.embedding_model import (
    compute_esm_embeddings,
    load_esm_model,
)
from foldforge.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from biotite.structure import AtomArray

    from foldforge.data.features.tokenizer import TokenArray

logger = get_logger(__name__)


class ESMFeaturizer:
    """Represent e s m featurizer."""

    def __init__(
        self,
        embedding_dir: str,
        sequence_fpath: str,
        embedding_dim: int = 1028,
        error_dir: str | None = None,
    ) -> None:
        self.embedding_dir = embedding_dir
        self.sequence_fpath = sequence_fpath
        self.seq_to_filename = self.get_seq_to_filename(sequence_fpath)
        self.embedding_dim = embedding_dim
        self.error_dir = error_dir
        if self.error_dir is not None:
            self.error_dir = str(Path(self.error_dir) / ("esm_error"))
            Path(self.error_dir).mkdir(parents=True, exist_ok=True)

    def get_seq_to_filename(self, sequence_fpath: str) -> dict[str, str]:
        """Return seq to filename."""
        df = pd.read_csv(sequence_fpath)
        df["filename"] = (
            df["part_id"].astype(str) + "/" + df["seq_label"].astype(str) + ".pt"
        )
        return {
            str(key): str(value)
            for key, value in df.set_index("seq")["filename"].items()
        }

    def load_esm_embedding(self, sequence: str) -> torch.Tensor:
        """Load esm embedding."""
        x = torch.load(
            str(Path(self.embedding_dir) / (self.seq_to_filename[sequence])),
            weights_only=True,
        )
        if x.size(0) != len(sequence):
            message = (
                f"ESM embedding size {x.size(0)} not equal "
                f"to sequence length {len(sequence)}"
                "The error occurs because embeddings were "
                "previously computed for a certain 'name', "
                "and then a different JSON file with the same task "
                "name was used, causing ESM to fail loading the "
                "existing embeddings."
                "You can resolve this by deleting the local "
                "esm_embeddings directory and retrying."
                "We recommend that the 'name' field in the JSON file be unique."
            )
            raise ValueError(message)
        return x

    def save_error(self, error_sequences: list[dict[str, str]], pdb_id: str) -> None:
        """Save error."""
        if (self.error_dir is None) or (len(error_sequences) == 0):
            return
        for error_data in error_sequences:
            fpath = str(
                Path(self.error_dir) / (f"{pdb_id}_{error_data['entity_id']}.txt")
            )
            if Path(fpath).exists():
                continue
            with Path(fpath).open("w") as f:
                f.write(error_data["error"])

    def __call__(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        bioassembly_dict: Mapping[str, Any],
        *,
        inference_mode: bool = False,
    ) -> torch.Tensor:
        """Gather cached per-sequence embeddings into the cropped token layout."""
        # init as zeros
        n_token = len(token_array)
        x = torch.zeros([n_token, self.embedding_dim])

        # get one atom per token
        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atom_array = atom_array[centre_atoms_indices]

        entity_id_to_sequence: dict[str, str] = {}
        if inference_mode:  # Only contains protein entity, many-to-one mapping
            for i, entity_info_wrapper in enumerate(bioassembly_dict["sequences"]):
                entity_id = str(i + 1)
                entity_type = next(iter(entity_info_wrapper.keys()))
                entity_info = entity_info_wrapper[entity_type]
                if entity_type == "proteinChain":
                    entity_id_to_sequence[entity_id] = entity_info["sequence"]
            protein_entity_ids = set(entity_id_to_sequence.keys())
        else:
            is_protein = centre_atom_array.chain_mol_type == "protein"
            protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])

        # enumerate over the entities
        error_sequences = []
        for entity_id in protein_entity_ids:
            try:
                # Get sequence
                if inference_mode:
                    sequence = entity_id_to_sequence[entity_id]
                else:
                    sequence = bioassembly_dict["sequences"][str(entity_id)]
                x_esm = self.load_esm_embedding(sequence)
                # Get residue indices of the cropped tokens
                entity_mask = centre_atom_array.label_entity_id == entity_id
                res_index = (
                    centre_atom_array.res_id[entity_mask] - 1
                )  # res_id starts with 1
                # Get esm embeddding according to residue indices
                x[entity_mask] = x_esm[res_index]
            except Exception as e:
                error_message = f"{e}:\n{traceback.format_exc()}"
                error_sequences.append(
                    {
                        "entity_id": entity_id,
                        "error": error_message,
                    }
                )
                logger.exception(
                    "[%s] ESM error: %s",
                    f"{bioassembly_dict['pdb_id']}",
                    f"{error_message}",
                )

        id_key = "name" if inference_mode else "pdb_id"
        self.save_error(error_sequences, pdb_id=bioassembly_dict[id_key])

        return x

    @staticmethod
    def precompute_esm_embedding(
        inputs: list,
        model_name: str,
        embedding_dir: str,
        sequence_fpath: str,
        checkpoint_dir: str,
    ) -> None:
        """Compute precompute esm embedding."""
        logger.info("%s", "Precompute ESM embeddings")
        # prepare seq_label
        all_seq_dict = []
        for sample_dict in inputs:
            sample_name = sample_dict["name"]
            for i, entity_info_wrapper in enumerate(sample_dict["sequences"]):
                pdb_entity_id = sample_name + "_" + str(i + 1)
                entity_type = next(iter(entity_info_wrapper.keys()))
                entity_info = entity_info_wrapper[entity_type]
                if entity_type == "proteinChain":
                    all_seq_dict.append(
                        {
                            "seq": entity_info["sequence"],
                            "pdb_entity_id": pdb_entity_id,
                            "seq_label": pdb_entity_id,
                            "part_id": pdb_entity_id,
                        }
                    )
        df_seq = pd.DataFrame(
            all_seq_dict, columns=["seq", "pdb_entity_id", "seq_label", "part_id"]
        )
        df_seq.to_csv(sequence_fpath)
        logger.info("%s", f"Save sequence file to {sequence_fpath}")

        model, alphabet = load_esm_model(model_name, local_esm_dir=checkpoint_dir)
        error_parts = []
        part_counts = dict(df_seq["part_id"].value_counts())
        for part_id in part_counts:
            df_part = df_seq[df_seq["part_id"] == part_id]
            logger.info("%s", f"Part {part_id}: {len(df_part)} sequences.")
            labels = df_part["seq_label"].tolist()
            sequences = df_part["seq"].tolist()
            try:
                save_dir = str(Path(embedding_dir) / (part_id))
                if not Path(save_dir).exists():
                    Path(save_dir).mkdir(parents=True)
                lm_embeddings = compute_esm_embeddings(
                    model_name,
                    model,
                    alphabet,
                    labels,
                    sequences,
                    save_dir,
                    truncation_seq_length=4094,
                    toks_per_batch=16384,
                )
                logger.info(
                    "%s",
                    (
                        f"[{part_id}] Processed {len(lm_embeddings)} sequences "
                        f"in total. Done!"
                    ),
                )
            except Exception:
                logger.exception(
                    "Failed to compute ESM embeddings for part %s", part_id
                )
                error_parts.append(part_id)
        logger.info(
            "%s",
            " ".join(
                str(value)
                for value in (
                    "Error parts: ",
                    error_parts,
                )
            ),
        )
