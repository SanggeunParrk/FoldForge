# Optional chemistry/model dependencies load only at their execution boundary.
# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import json

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
import logging
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests

import foldforge.data.ccd.components as ccd
from foldforge.data.inputs.features import (
    ResidueSampleDictToFeatures as SampleDictToFeatures,
)
from foldforge.data.msa.search import HmmsearchConfig, residue_run_hmmsearch_with_a3m
from foldforge.data.web.assets import URL
from foldforge.data.web.requests import parse_fasta_string, run_mmseqs2_service
from foldforge.models.io.paths import run_directory

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MMSEQS_SERVICE_HOST_URL = os.getenv(
    "MMSEQS_SERVICE_HOST_URL", "https://protenix-server.com/api/msa"
)
MAX_ATOM_NUM = 60000
MAX_TOKEN_NUM = 5000

PROTENIX_ROOT_DIR = os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))

DATA_CACHE_DIR = f"{PROTENIX_ROOT_DIR}/common/"
CHECKPOINT_DIR = f"{PROTENIX_ROOT_DIR}/checkpoint/"


logger = logging.getLogger(__name__)


def download_tos_url(tos_url: str, local_file_path: str | Path) -> None:
    """Download completely before replacing a cached asset."""
    target = Path(local_file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        with requests.get(tos_url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                handle.writelines(response.iter_content(chunk_size=8192))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Downloaded %s to %s", tos_url, target)


class TooLargeComplexError(Exception):
    """Represent too large complex error."""

    def __init__(self, **kwargs: Any) -> None:
        if "num_atoms" in kwargs:
            message = (
                f"We can only process complexes with no "
                f"more than {MAX_ATOM_NUM} atoms, "
                f"but there are {kwargs['num_atoms']} atoms in the input."
            )
        elif "num_tokens" in kwargs:
            message = (
                f"We can only process complexes with no "
                f"more than {MAX_TOKEN_NUM} tokens, "
                f"but there are {kwargs['num_tokens']} tokens in the input."
            )
        else:
            message = ""
        super().__init__(message)


class RequestParser:
    """Represent request parser."""

    def __init__(
        self,
        request_json_path: str,
        request_dir: str,
        email: str = "",
        model_name: str = "protenix_base_default_v1.0.0",
    ) -> None:
        with Path(request_json_path).open() as f:
            self.request = json.load(f)
        self.request_dir = str(run_directory(request_dir, model="protenix"))
        self.fpath = str(Path(__file__).absolute())
        self.email = email
        self.model_name = model_name
        Path(self.request_dir).mkdir(parents=True, exist_ok=True)

    def download_data_cache(self) -> dict[str, str]:
        """Download data cache."""
        data_cache_dir = DATA_CACHE_DIR
        Path(data_cache_dir).mkdir(parents=True, exist_ok=True)
        components, molecules = ccd.get_ccd_cache_paths()
        cache_paths = {
            "ccd_components_file": components,
            "ccd_components_rdkit_mol_file": molecules,
        }
        cluster_path = Path(data_cache_dir) / "clusters-by-entity-40.txt"
        if not cluster_path.exists():
            download_tos_url(URL["pdb_cluster_file"], cluster_path)
        cache_paths["pdb_cluster_file"] = str(cluster_path)
        return cache_paths

    def download_model(self, model_name: str, checkpoint_local_path: str) -> None:
        """Download model."""
        tos_url = URL[f"{model_name}"]
        logger.info("%s", f"Downloading model checkpoing from\n {tos_url}...")
        download_tos_url(tos_url, checkpoint_local_path)

    def get_model(self) -> str:
        """Return model."""
        checkpoint_dir = CHECKPOINT_DIR
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if not Path(
            checkpoint_path := str(
                Path(checkpoint_dir).joinpath(f"{self.model_name}.pt")
            )
        ).exists():
            self.download_model(self.model_name, checkpoint_local_path=checkpoint_path)
        if Path(checkpoint_path).exists():
            return checkpoint_dir
        msg = "Failed in finding model checkpoint."
        raise ValueError(msg)

    def get_data_json(self) -> str:  # noqa: C901, PLR0912, PLR0915 - notebook entity/MSA translation stages
        """Return data json."""
        input_json_dict = {
            "name": (self.request["name"]),
            "covalent_bonds": self.request["covalent_bonds"],
        }
        input_json_path = str(Path(self.request_dir).joinpath("inputs.json"))

        sequences = []
        entity_pending_msa = {}
        for i, entity_info_wrapper in enumerate(self.request["sequences"]):
            entity_id = str(i + 1)
            entity_info_wrapper: dict[str, dict[str, Any]]
            if len(entity_info_wrapper) != 1:
                message = "Invalid state: len(entity_info_wrapper) == 1"
                raise ValueError(message)

            seq_type, seq_info = next(iter(entity_info_wrapper.items()))

            if seq_type == "proteinChain" and self.request["use_msa"]:
                entity_pending_msa[entity_id] = seq_info["sequence"]

            if seq_type not in [
                "proteinChain",
                "dnaSequence",
                "rnaSequence",
                "ligand",
                "ion",
            ]:
                raise NotImplementedError
            sequences.append({seq_type: seq_info})

        tmp_json_dict = deepcopy(input_json_dict)
        tmp_json_dict["sequences"] = sequences

        cache_paths = self.download_data_cache()
        ccd.set_ccd_cache_paths(
            cache_paths["ccd_components_file"],
            cache_paths["ccd_components_rdkit_mol_file"],
        )
        sample2feat = SampleDictToFeatures(
            single_sample_dict=tmp_json_dict,
        )
        atom_array = sample2feat.get_atom_array()
        num_atoms = len(atom_array)
        num_tokens = np.sum(atom_array.centre_atom_mask)
        if num_atoms > MAX_ATOM_NUM:
            raise TooLargeComplexError(num_atoms=num_atoms)
        if num_tokens > MAX_TOKEN_NUM:
            raise TooLargeComplexError(num_tokens=num_tokens)
        del tmp_json_dict

        if len(entity_pending_msa) > 0:
            seq_to_entity_id = defaultdict(list)
            for entity_id, seq in entity_pending_msa.items():
                seq_to_entity_id[seq].append(entity_id)
            seq_to_entity_id = dict(seq_to_entity_id)
            seqs_pending_msa = sorted(seq_to_entity_id.keys())

            Path(msa_res_dir := str(Path(self.request_dir).joinpath("msa"))).mkdir(
                parents=True, exist_ok=True
            )

            tmp_fasta_fpath = str(Path(msa_res_dir).joinpath("msa_input.fasta"))
            RequestParser.msa_search(
                seqs_pending_msa=seqs_pending_msa,
                tmp_fasta_fpath=tmp_fasta_fpath,
                msa_res_dir=msa_res_dir,
                email=self.email,
                mode="protenix",
            )
            msa_res_subdirs = RequestParser.msa_postprocess(
                seqs_pending_msa=seqs_pending_msa,
                msa_res_dir=msa_res_dir,
            )

            for seq, msa_res_dir in zip(
                seqs_pending_msa, msa_res_subdirs, strict=False
            ):
                for entity_id in seq_to_entity_id[seq]:
                    entity_index = int(entity_id) - 1
                    msa_names = []
                    if Path(
                        pairing_path := str(Path(msa_res_dir).joinpath("pairing.a3m"))
                    ).exists():
                        sequences[entity_index]["proteinChain"]["pairedMsaPath"] = (
                            pairing_path
                        )
                        msa_names.append("pairing")
                    if Path(
                        unpaired_path := str(
                            Path(msa_res_dir).joinpath("non_pairing.a3m")
                        )
                    ).exists():
                        sequences[entity_index]["proteinChain"]["unpairedMsaPath"] = (
                            unpaired_path
                        )
                        msa_names.append("non_pairing")
                    use_msa = self.request.get("use_msa")
                    use_template = self.request.get("use_template", False)
                    if msa_names and use_msa and use_template:
                        if not Path(
                            template_path := str(
                                Path(msa_res_dir).joinpath("hmmsearch.a3m")
                            )
                        ).exists():
                            database = self.request.get("template_sequence_database")
                            if not database:
                                message = (
                                    "Template search requires "
                                    "template_sequence_database"
                                )
                                raise ValueError(message)
                            a3m = "".join(
                                (Path(msa_res_dir) / f"{name}.a3m").read_text()
                                for name in msa_names
                            )
                            config = HmmsearchConfig(
                                hmmsearch_binary_path=self.request.get(
                                    "hmmsearch_binary", "hmmsearch"
                                ),
                                hmmbuild_binary_path=self.request.get(
                                    "hmmbuild_binary", "hmmbuild"
                                ),
                            )
                            Path(template_path).write_text(
                                residue_run_hmmsearch_with_a3m(
                                    database, config, None, a3m
                                )
                            )
                        sequences[entity_index]["proteinChain"]["templatesPath"] = (
                            template_path
                        )

        input_json_dict["sequences"] = sequences
        with Path(input_json_path).open("w") as f:
            json.dump([input_json_dict], f, indent=4)
        return input_json_path

    @staticmethod
    def msa_search(
        seqs_pending_msa: Sequence[str],
        tmp_fasta_fpath: str,
        msa_res_dir: str,
        email: str = "",
        mode: str = "protenix",
    ) -> list[str] | None:
        """Compute msa search."""
        lines = []
        for idx, seq in enumerate(seqs_pending_msa):
            lines.append(f">query_{idx}\n")
            lines.append(f"{seq}\n")
        if (last_line := lines[-1]).endswith("\n"):
            lines[-1] = last_line.rstrip("\n")
        with Path(tmp_fasta_fpath).open("w") as f:
            f.writelines(lines)

        with Path(tmp_fasta_fpath).open() as f:
            query_seqs = f.read()
        if mode == "protenix":
            try:
                run_mmseqs2_service(
                    query_seqs,
                    msa_res_dir,
                    use_env=True,
                    use_templates=False,
                    host_url=MMSEQS_SERVICE_HOST_URL,
                    user_agent="colabfold/1.5.5",
                    email=email,
                    server_mode=mode,
                )
            except Exception:
                error_message = (
                    f"MMSEQS2 failed with the "
                    f"following error message:\n{traceback.format_exc()}"
                )
                logger.exception("%s", error_message)

        elif mode == "colabfold":
            res_dirs = []
            fasta_dict = parse_fasta_string(query_seqs)
            for i, (seq_name, seq) in enumerate(fasta_dict.items()):
                logger.info(
                    "%s", f"Searching MSA for {seq_name} with the sequence itself."
                )
                try:
                    res_dir = run_mmseqs2_service(
                        f">{seq_name}\n{seq}",
                        str(Path(msa_res_dir) / (str(i))),
                        use_env=True,
                        use_filter=True,
                        use_templates=False,
                        filter=None,
                        use_pairing=False,
                        pairing_strategy="greedy",
                        host_url=MMSEQS_SERVICE_HOST_URL,
                        user_agent="colabfold/1.5.5",
                        email=email,
                        server_mode=mode,
                    )
                    res_dirs.append(res_dir)
                except Exception:
                    error_message = (
                        f"MMSEQS2 failed with "
                        f"the following error message:\n{traceback.format_exc()}"
                    )
                    logger.exception("%s", error_message)
            if len(fasta_dict) > 1:
                # search paired MSA
                try:
                    run_mmseqs2_service(
                        query_seqs,
                        str(Path(msa_res_dir) / ("complex")),
                        use_env=True,
                        use_filter=True,
                        use_templates=False,
                        filter=None,
                        use_pairing=True,
                        pairing_strategy="greedy",
                        host_url=MMSEQS_SERVICE_HOST_URL,
                        user_agent="colabfold/1.5.5",
                        email=email,
                        server_mode=mode,
                    )
                except Exception:
                    error_message = (
                        f"MMSEQS2 failed with "
                        f"the following error message:\n{traceback.format_exc()}"
                    )
                    logger.exception("%s", error_message)
            else:
                pairing_msa_fpath = str(Path(msa_res_dir) / ("0") / ("pairing.a3m"))
                with Path(pairing_msa_fpath).open("w") as f:
                    f.write(">query\n" + query_seqs.split("\n")[-1])
            return res_dirs
        return None

    @staticmethod
    def msa_postprocess(seqs_pending_msa: Sequence[str], msa_res_dir: str) -> list[str]:  # noqa: C901, PLR0915 - notebook entity/MSA translation stages
        """Compute msa postprocess."""

        def read_m8(m8_file: str) -> dict[str, str]:
            """Read m8."""
            uniref_to_ncbi_taxid = {}
            with Path(m8_file).open() as infile:
                for line in infile:
                    line_list = line.replace("\n", "").split("\t")
                    hit_name = line_list[1]
                    ncbi_taxid = line_list[2]
                    uniref_to_ncbi_taxid[hit_name] = ncbi_taxid
            return uniref_to_ncbi_taxid

        def read_a3m(a3m_file: str) -> tuple[list[str], list[str], int]:
            """Read a3m."""
            heads = []
            seqs = []
            # Record the row index. The index before this index is the MSA of Uniref30
            # DB,
            # and the index after this index is the MSA of ColabfoldDB.
            uniref_index = 0
            query_name: str | None = None
            with Path(a3m_file).open() as infile:
                for idx, line in enumerate(infile):
                    if line.startswith(">"):
                        heads.append(line)
                        if idx == 0:
                            query_name = line
                        elif idx > 0 and line == query_name:
                            uniref_index = idx
                    else:
                        seqs.append(line)
            return heads, seqs, uniref_index

        def make_pairing_and_non_pairing_msa(
            query_seq: str,
            seq_dir: str,
            raw_a3m_path: str,
            uniref_to_ncbi_taxid: Mapping[str, str],
        ) -> None:
            """Make pairing and non pairing msa."""
            heads, msa_seqs, uniref_index = read_a3m(raw_a3m_path)
            uniref100_lines = [">query\n", f"{query_seq}\n"]
            other_lines = [">query\n", f"{query_seq}\n"]

            for idx, (raw_head, msa_seq) in enumerate(
                zip(heads, msa_seqs, strict=False)
            ):
                head = raw_head
                if msa_seq.rstrip("\n") == query_seq:
                    continue

                uniref_id = head.split("\t")[0][1:]
                ncbi_taxid = uniref_to_ncbi_taxid.get(uniref_id, None)
                if (ncbi_taxid is not None) and (idx < (uniref_index // 2)):
                    if not uniref_id.startswith("UniRef100_"):
                        head = head.replace(
                            uniref_id, f"UniRef100_{uniref_id}_{ncbi_taxid}/"
                        )
                    else:
                        head = head.replace(uniref_id, f"{uniref_id}_{ncbi_taxid}/")
                    uniref100_lines.extend([head, msa_seq])
                else:
                    other_lines.extend([head, msa_seq])

            with Path(str(Path(seq_dir).joinpath("pairing.a3m"))).open("w") as f:
                f.writelines(uniref100_lines)
            with Path(str(Path(seq_dir).joinpath("non_pairing.a3m"))).open("w") as f:
                f.writelines(other_lines)

        def make_non_pairing_msa_only(
            query_seq: str,
            seq_dir: str,
            raw_a3m_path: str,
        ) -> None:
            """Make non pairing msa only."""
            heads, msa_seqs, _ = read_a3m(raw_a3m_path)
            other_lines = [">query\n", f"{query_seq}\n"]
            for head, msa_seq in zip(heads, msa_seqs, strict=False):
                if msa_seq.rstrip("\n") == query_seq:
                    continue
                other_lines.extend([head, msa_seq])
            with Path(str(Path(seq_dir).joinpath("non_pairing.a3m"))).open("w") as f:
                f.writelines(other_lines)

        def make_dummy_msa(
            query_seq: str, seq_dir: str, msa_type: str = "both"
        ) -> None:
            """Make dummy msa."""
            if msa_type == "both":
                fnames = ["pairing.a3m", "non_pairing.a3m"]
            elif msa_type == "pairing":
                fnames = ["pairing.a3m"]
            elif msa_type == "non_pairing":
                fnames = ["non_pairing.a3m"]
            else:
                raise NotImplementedError
            for fname in fnames:
                with Path(str(Path(seq_dir).joinpath(fname))).open("w") as f:
                    f.write(">query\n")
                    f.write(f"{query_seq}\n")

        msa_res_subdirs = []
        for seq_idx, query_seq in enumerate(seqs_pending_msa):
            Path(
                seq_dir := str(
                    Path(str(Path(msa_res_dir).joinpath(str(seq_idx)))).absolute()
                )
            ).mkdir(parents=True, exist_ok=True)
            if Path(
                raw_a3m_path := str(Path(msa_res_dir).joinpath(f"{seq_idx}.a3m"))
            ).exists():
                if Path(
                    m8_path := str(Path(msa_res_dir).joinpath("uniref_tax.m8"))
                ).exists():
                    uniref_to_ncbi_taxid = read_m8(m8_path)
                    make_pairing_and_non_pairing_msa(
                        query_seq=query_seq,
                        seq_dir=seq_dir,
                        raw_a3m_path=raw_a3m_path,
                        uniref_to_ncbi_taxid=uniref_to_ncbi_taxid,
                    )
                else:
                    make_non_pairing_msa_only(
                        query_seq=query_seq,
                        seq_dir=seq_dir,
                        raw_a3m_path=raw_a3m_path,
                    )
                    make_dummy_msa(
                        query_seq=query_seq, seq_dir=seq_dir, msa_type="pairing"
                    )

            else:
                logger.info(
                    "%s",
                    (
                        f"Failed in searching MSA for \n{query_seq}\nusing "
                        f"the sequence itself as MSA."
                    ),
                )
                make_dummy_msa(query_seq=query_seq, seq_dir=seq_dir)
            msa_res_subdirs.append(seq_dir)

        return msa_res_subdirs

    def launch(self) -> None:
        """Run notebook requests through the same checkpoint and output runtime."""
        from foldforge.data.ccd import CCDDatabase
        from foldforge.data.ccd.database import current_database
        from foldforge.models.config import ExecutionConfig, OutputConfig
        from foldforge.models.io.request import Request
        from foldforge.models.io.runtime import run
        from foldforge.utils.seed import seed_context

        database = (
            CCDDatabase(Path(self.request["ccd_db"]))
            if self.request.get("ccd_db")
            else current_database()
        )
        with database.activate():
            checkpoint = Path(self.get_model()) / f"{self.model_name}.pt"
            legacy_seeds = "model_seeds" in self.request
            if legacy_seeds and any(
                name in self.request for name in ("trunk_seeds", "diffusion_seeds")
            ):
                message = "Use either model_seeds or trunk_seeds/diffusion_seeds"
                raise ValueError(message)
            trunk_seeds = self.request.get(
                "trunk_seeds", self.request.get("model_seeds", [0])
            )
            seed_pairs = [
                (trunk_seed, diffusion_seed)
                for trunk_seed in trunk_seeds
                for diffusion_seed in self.request.get(
                    "diffusion_seeds", [trunk_seed] if legacy_seeds else [0]
                )
            ]
            input_path = None
            previous_trunk_seed = None
            for trunk_seed, diffusion_seed in seed_pairs:
                if input_path is None or trunk_seed != previous_trunk_seed:
                    with seed_context(int(trunk_seed)):
                        input_path = Path(self.get_data_json())
                    previous_trunk_seed = trunk_seed
                run(
                    Request(
                        model="protenix",
                        variant=self.model_name,
                        ccd_db=database.root,
                        input=input_path,
                        checkpoint=checkpoint,
                        out=Path(self.request_dir)
                        / f"trunk-{trunk_seed}_diffusion-{diffusion_seed}",
                        trunk_seed=int(trunk_seed),
                        diffusion_seed=int(diffusion_seed),
                        backend=self.request.get("backend", "miniworld"),
                        precision=self.request.get("precision", "bf16"),
                        recycles=int(self.request.get("N_cycle", 10)),
                        steps=int(self.request.get("N_step", 200)),
                        samples=int(self.request.get("N_sample", 5)),
                        guidance=self.request.get("use_tfg"),
                        output=OutputConfig.model_validate(
                            self.request.get("output", {})
                        ),
                        execution=ExecutionConfig.model_validate(
                            self.request.get("execution", {})
                        ),
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--request_json_path",
        type=str,
        required=True,
        help="Path to the request JSON file.",
    )
    parser.add_argument(
        "--request_dir", type=str, required=True, help="Path to the request directory."
    )
    parser.add_argument(
        "--email", type=str, required=False, default="", help="Your email address."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="protenix_base_default_v1.0.0",
        help="The model name for inference.",
    )

    args = parser.parse_args()
    parser = RequestParser(
        request_json_path=args.request_json_path,
        request_dir=args.request_dir,
        email=args.email,
        model_name=args.model_name,
    )
    parser.launch()
