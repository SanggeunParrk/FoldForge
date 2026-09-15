from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

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
from typing import TYPE_CHECKING, Protocol

import pandas as pd
import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType


logger = logging.getLogger(__name__)


class Alphabet(Protocol):
    """The fair-esm alphabet API used by offline embedding generation."""

    def get_batch_converter(self, truncation_seq_length: int) -> Callable:
        """Return the sequence-to-token collator."""
        ...


ESM_CONFIG = {
    "esm2-3b": {
        "type": "esm2",
        "model_path": "esm2_t36_3B_UR50D.pt",
        "emb_dim": 2560,
        "n_layers": 36,
    },
    "esm2-3b-ism": {
        "type": "esm2",
        "model_path": "esm2_t36_3B_UR50D_ism.pt",
        "emb_dim": 2560,
        "n_layers": 36,
    },  # https://www.biorxiv.org/content/10.1101/2024.11.08.622579v2
}


def _fair_esm() -> tuple[ModuleType, ModuleType]:
    esm = importlib.import_module("esm")
    pretrained = importlib.import_module("esm.pretrained")
    if not all(
        (
            hasattr(esm, "FastaBatchedDataset"),
            hasattr(pretrained, "load_model_and_alphabet_local"),
            hasattr(pretrained, "load_model_and_alphabet"),
        )
    ):
        message = (
            "Offline ESM2 embedding generation requires a separate fair-esm "
            "environment; the FoldForge ESMFold2 environment supplies ESMC. "
            "Do not install fair-esm over the ESMC package."
        )
        raise RuntimeError(message)
    return esm, pretrained


def _load_esm2_model(model_path: str) -> tuple[torch.nn.Module, Alphabet]:
    _, pretrained = _fair_esm()
    if Path(model_path).exists():
        model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    else:
        model, alphabet = pretrained.load_model_and_alphabet(Path(model_path).stem)
    return model, alphabet


def load_esm_model(
    model_name: str, local_esm_dir: str = "release_data/checkpoint"
) -> tuple[torch.nn.Module, Alphabet]:
    """Load esm model."""
    local_model_path = str(Path(local_esm_dir) / (ESM_CONFIG[model_name]["model_path"]))
    if Path(local_model_path).exists():
        logger.info(
            "%s",
            " ".join(
                str(value)
                for value in (
                    "Try to load ESM language model from ",
                    local_model_path,
                )
            ),
        )

    if "ism" in model_name and not Path(local_model_path).exists():
        msg = (
            f"esm2-3b-ism model: {local_model_path} does not exist \n"
            "this model can not be download from fair-esm, \n"
            "download it from "
            "https://af3-dev.tos-cn-beijing.volces.com/release_model/esm2_t36_3B_UR50D_ism.pt"
        )
        raise RuntimeError(msg)
    if not model_name.startswith("esm2"):
        message = f"Unsupported sequence embedding model: {model_name}"
        raise ValueError(message)
    model, alphabet = _load_esm2_model(local_model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    return model, alphabet


def _check_files_exist(save_dir: str, labels: list[str]) -> bool:
    return all(Path(str(Path(save_dir) / (label + ".pt"))).exists() for label in labels)


def compute_esm_embeddings(
    model_name: str,
    model: torch.nn.Module,
    alphabet: Alphabet,
    labels: list[str],
    sequences: list[str],
    save_dir: str,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = 1022,
) -> dict[str, torch.Tensor]:
    """Compute e s m embeddings."""
    if not model_name.startswith("esm2"):
        message = f"Unsupported sequence embedding model: {model_name}"
        raise ValueError(message)
    return compute_esm2_embeddings(
        model,
        alphabet,
        labels,
        sequences,
        save_dir,
        toks_per_batch,
        truncation_seq_length,
    )


# Adapt from Corso, Gabriele, et al. "Diffdock: Diffusion steps, twists, and turns for
# molecular docking."
# URL: https://github.com/gcorso/DiffDock/blob/main/utils/inference_utils.py
def compute_esm2_embeddings(
    model: torch.nn.Module,
    alphabet: Alphabet,
    labels: list[str],
    sequences: list[str],
    save_dir: str,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = 1022,
) -> dict[str, torch.Tensor]:
    """Compute esm2 embeddings."""
    esm, _ = _fair_esm()
    dataset = esm.FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    repr_layer = model.num_layers
    embeddings = {}
    with torch.no_grad():
        for batch_idx, (batch_labels, strs, raw_toks) in enumerate(tqdm(data_loader)):
            toks = raw_toks
            logger.info(
                "%s",
                (
                    f"Processing {batch_idx + 1} of {len(batches)} batches "
                    f"({toks.size(0)} sequences)"
                ),
            )
            if _check_files_exist(save_dir, batch_labels):
                continue
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representation = out["representations"][repr_layer].to(device="cpu")
            for i, label in enumerate(batch_labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representation[i, 1 : truncate_len + 1].clone()
                save_path = str(Path(save_dir) / (label + ".pt"))
                torch.save(embeddings[label], save_path)
    return embeddings


def pdb_sequences_iterator(
    input_path: str = "./scripts/msa/data/pdb_seqs/pdb_seq.csv",
    save_path: str = "./scripts/msa/data/pdb_seqs/pdb_labels_seqs.csv",
    start_id: int = 0,
    end_id: int = -1,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Compute pdb sequences iterator."""
    if Path(save_path).exists():
        df_seq = pd.read_csv(save_path)
    else:
        df = pd.read_csv(input_path)
        # Protein only
        df = df[df["mol_type"] == "protein"]
        # Sequence name
        df["pdb_entity_id"] = df["pdb_id"] + "_" + df["entity_id"].astype(str)
        # Group by 'seq'
        df_seq = df.groupby("seq")["pdb_entity_id"].apply(",".join).reset_index()
        # Use the first pdb_entity_id as the label
        df_seq["seq_label"] = df_seq["pdb_entity_id"].apply(lambda x: x.split(",")[0])
        if df_seq["seq_label"].nunique() != len(df_seq):
            message = "Invalid state: df_seq['seq_label'].nunique() == len(df_seq)"
            raise ValueError(message)
        # Get a part id
        df_seq["part_id"] = df_seq["pdb_entity_id"].apply(lambda x: x[1:3])
        df_seq.to_csv(save_path)

    if end_id == -1:
        end_id = len(df_seq)
    df_seq = df_seq[start_id:end_id]

    part_counts = dict(df_seq["part_id"].value_counts())
    for part_id in part_counts:
        df_part = df_seq[df_seq["part_id"] == part_id]
        logger.info("%s", f"Part {part_id}: {len(df_part)} sequences.")
        yield part_id, df_part["seq_label"].tolist(), df_part["seq"].tolist()


def process_pdb_dataset(
    model_name: str,
    root_save_dir: str,
    pdb_seq_path: str,
    pdb_seq_label_path: str,
    start_id: int = 0,
    end_id: int = -1,
) -> None:
    """Process pdb dataset."""
    model, alphabet = load_esm_model(model_name)
    seq_iterator = pdb_sequences_iterator(
        pdb_seq_path, pdb_seq_label_path, start_id, end_id
    )
    error_parts = []
    for part_id, batch_labels, sequences in seq_iterator:
        save_dir = str(Path(root_save_dir) / (f"{part_id}"))

        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        logger.info("%s", f"[{part_id}] Generating ESM language model embeddings")
        lm_embeddings = compute_esm_embeddings(
            model_name,
            model,
            alphabet,
            batch_labels,
            sequences,
            save_dir,
            truncation_seq_length=4094,
            toks_per_batch=16384,
        )
        logger.info(
            "%s",
            f"[{part_id}] Processed {len(lm_embeddings)} sequences in total. Done!",
        )

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


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=list(ESM_CONFIG.keys()))
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    args = parser.parse_args()

    save_dir = f"./esm_embeddings/{args.model_name}"
    pdb_seq_path = "./scripts/msa/data/pdb_seqs/pdb_seq.csv"
    pdb_seq_label_path = "./scripts/msa/data/pdb_seqs/pdb_labels_seqs.csv"

    if not Path(save_dir).exists():
        logger.info(
            "%s",
            " ".join(
                str(value)
                for value in (
                    "Make dir: ",
                    save_dir,
                )
            ),
        )
        Path(save_dir).mkdir(parents=True)
    process_pdb_dataset(
        args.model_name,
        save_dir,
        pdb_seq_path,
        pdb_seq_label_path,
        args.start_id,
        args.end_id,
    )


if __name__ == "__main__":
    main()
