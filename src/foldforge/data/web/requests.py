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
import logging
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} estimate "
    "remaining: {remaining}]"
)
logger = logging.getLogger(__name__)

username = os.environ.get("FOLDFORGE_MSA_USERNAME", "")
password = os.environ.get("FOLDFORGE_MSA_PASSWORD", "")


MAX_ERROR_RETRIES = 5


def parse_fasta_string(fasta_string: str) -> dict:
    """Parse fasta string."""
    fasta_dict: dict[str, str] = {}
    header: str | None = None
    lines = fasta_string.strip().split("\n")
    for line in lines:
        if line.startswith(">"):
            header = line[1:].strip()
            fasta_dict[header] = ""
        elif line.strip():
            if header is None:
                message = "FASTA sequence appears before its header"
                raise ValueError(message)
            fasta_dict[header] += line.strip()
    return fasta_dict


def run_mmseqs2_service(  # noqa: C901, PLR0912, PLR0915 - legacy MSA service protocol state machine
    x: list[str] | str,
    prefix: str,
    *,
    use_env: bool = True,
    use_filter: bool = True,
    use_templates: bool = False,  # noqa: ARG001 - shared callback or fixture signature
    filter: bool | None = None,  # noqa: A002 - external interface keyword
    use_pairing: bool = False,
    pairing_strategy: str = "complete",
    host_url: str = "https://api.colabfold.com",
    user_agent: str = "",
    email: str = "",
    server_mode: str = "protenix",
) -> str | None:
    """Compute run mmseqs2 service."""
    if (server_mode == "protenix") and (
        host_url != "https://protenix-server.com/api/msa"
    ):
        message = "Invalid state: host_url == 'https://protenix-server.com/api/msa'"
        raise ValueError(message)
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"
    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., "
            "'toolname/version contact@email') to help us debug in case of "
            "problems. This warning will become an error in the future."
        )

    def submit(seqs: list[str], mode: str, N: int = 101) -> dict[str, Any]:  # noqa: ARG001, N803 - checkpoint-compatible keyword
        """Compute submit."""
        n, query = start_id, ""
        for seq in seqs:
            query += f"{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a
                # multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode, "email": email},
                    timeout=6.02,
                    headers=headers,
                    auth=HTTPBasicAuth(username, password) if username else None,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    "Error while fetching result from MSA server. Retrying... (%s/5)",
                    f"{error_count}",
                )
                logger.warning("Error: %s", f"{e}")
                time.sleep(5)
                if error_count > MAX_ERROR_RETRIES:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.exception("Server didn't reply with json: %s", f"{res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID: str) -> dict[str, Any]:  # noqa: ARG001, N803 - checkpoint-compatible keyword
        """Compute status."""
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ticket_id}",
                    timeout=6.02,
                    headers=headers,
                    auth=HTTPBasicAuth(username, password) if username else None,
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    "Error while fetching result from MSA server. Retrying... (%s/5)",
                    f"{error_count}",
                )
                logger.warning("Error: %s", f"{e}")
                time.sleep(5)
                if error_count > MAX_ERROR_RETRIES:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.exception("Server didn't reply with json: %s", f"{res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID: str, path: str | Path) -> None:  # noqa: ARG001, N803 - checkpoint-compatible keyword
        """Download ."""
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ticket_id}",
                    timeout=6.02,
                    headers=headers,
                    auth=HTTPBasicAuth(username, password) if username else None,
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    "Error while fetching result from MSA server. Retrying... (%s/5)",
                    f"{error_count}",
                )
                logger.warning("Error: %s", f"{e}")
                time.sleep(5)
                if error_count > MAX_ERROR_RETRIES:
                    raise
                continue
            break
        with Path(path).open("wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        use_env = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"

    # define path
    path = prefix
    if not Path(path).is_dir():
        Path(path).mkdir()

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    start_id, redo = 101, True

    # deduplicate and keep track of order
    seqs_unique = list(dict.fromkeys(seqs))
    # lets do it!
    logger.info("Msa server is running.")
    if not Path(tar_gz_file).is_file():
        time_estimate = 100
        with tqdm(total=time_estimate, bar_format=TQDM_BAR_FORMAT) as pbar:
            ticket_id: str | None = None
            while redo:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, start_id)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 60
                    logger.error(
                        "Sleeping for %ss. Reason: %s",
                        f"{sleep_time}",
                        f"{out['status']}",
                    )
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, start_id)

                if out["status"] == "ERROR":
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your input is a "
                        "valid protein sequence. If error persists, please "
                        "try again an "
                        "hour later."
                    )
                    raise RuntimeError(msg)

                if out["status"] == "MAINTENANCE":
                    msg = (
                        "MMseqs2 API is undergoing maintenance. Please try "
                        "again in a few"
                        " minutes."
                    )
                    raise RuntimeError(msg)

                # wait for job to finish
                ticket_id, elapsed = out["id"], 0
                if ticket_id is None:
                    message = "MSA service response did not include a job ID"
                    raise ValueError(message)
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 60
                    logger.error(
                        "Sleeping for %ss. Reason: %s", f"{t}", f"{out['status']}"
                    )
                    time.sleep(t)
                    out = status(ticket_id)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        elapsed += t
                    pbar.n = min(99, int(100 * elapsed / (30.0 * 60)))
                    pbar.refresh()
                if out["status"] == "COMPLETE":
                    pbar.n = 100
                    pbar.refresh()
                    redo = False

                if out["status"] == "ERROR":
                    redo = False
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your input is a "
                        "valid protein sequence. If error persists, please "
                        "try again an "
                        "hour later."
                    )
                    raise RuntimeError(msg)

            # Download results
            if ticket_id is None:
                message = "MSA server did not return a job identifier"
                raise RuntimeError(message)
            download(ticket_id, tar_gz_file)
            with tarfile.open(tar_gz_file) as tar_gz:
                tar_gz.extractall(str(Path(tar_gz_file).parent), filter="data")
            files = [path.name for path in Path(tar_gz_file).parent.iterdir()]

            if server_mode == "protenix":
                if (
                    "0.a3m" not in files
                    or "pdb70_220313_db.m8" not in files
                    or "uniref_tax.m8" not in files
                ):
                    msg = (
                        "Files 0.a3m, pdb70_220313_db.m8, and uniref_tax.m8 "
                        "not found in "
                        "the directory."
                    )
                    raise FileNotFoundError(msg)
                logger.info("%s", "Files downloaded and extracted successfully.")
            elif server_mode == "colabfold":
                if not use_pairing:
                    env_a3m_fpath = str(
                        Path(prefix) / ("bfd.mgnify30.metaeuk30.smag30.a3m")
                    )
                    with Path(env_a3m_fpath).open() as f:
                        env_a3m_dict = parse_fasta_string(f.read().replace("\x00", ""))
                    uniref_a3m_fpath = str(Path(prefix) / ("uniref.a3m"))
                    with Path(uniref_a3m_fpath).open() as f:
                        uniref_a3m_dict = parse_fasta_string(
                            f.read().replace("\x00", "")
                        )
                    if not isinstance(x, str):
                        message = "ColabFold result conversion requires one FASTA input"
                        raise ValueError(message)
                    query_id = str(int(x.split("\n")[0].split("_")[-1]))
                    query_seq = x.split("\n")[1]
                    real_non_pairing_fpath = str(
                        Path(prefix) / (query_id) / ("non_pairing.a3m")
                    )
                    output_dir = str(Path(real_non_pairing_fpath).parent)
                    if not Path(output_dir).exists():
                        Path(output_dir).mkdir(parents=True)
                    with Path(real_non_pairing_fpath).open("w") as f:
                        f.write(f">query\n{query_seq}\n")
                        for k, v in env_a3m_dict.items():
                            if k.startswith("query_"):
                                continue
                            f.write(f">{k}\n{v}\n")
                        for k, v in uniref_a3m_dict.items():
                            if k.startswith("query_"):
                                continue
                            f.write(f">{k}\n{v}\n")
                    return str(
                        Path(str(Path(real_non_pairing_fpath).parent)).absolute()
                    )
                # pairing mode
                pair_a3m = str(Path(prefix) / ("pair.a3m"))
                with Path(pair_a3m).open() as f:
                    pair_a3m_chunks = f.read().split("\x00")
                for chunk in pair_a3m_chunks[:-1]:
                    real_pairing_fpath = str(
                        Path(prefix)
                        / (str(int(chunk.split("\n")[0].split("_")[-1])))
                        / ("pairing.a3m")
                    )
                    output_dir = str(Path(real_pairing_fpath).parent)
                    if not Path(output_dir).exists():
                        Path(output_dir).mkdir(parents=True)
                    chunk_fasta = parse_fasta_string(chunk)
                    with Path(real_pairing_fpath).open("w") as f:
                        for i, (raw_k, v) in enumerate(chunk_fasta.items()):
                            k = raw_k
                            if k.startswith("query_"):
                                f.write(f">query\n{v}\n")
                            else:
                                ks = k.split("\t")
                                ks[0] = f"{ks[0]}_{i}/"
                                k = "\t".join(ks)
                                f.write(f">{k}_{i}\n{v}\n")
    return None
