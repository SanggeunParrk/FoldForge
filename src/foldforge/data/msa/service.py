# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Credential-free client for the public ColabFold MMseqs2 MSA service.

OpenDDE does not host its own MSA server. Protein MSA search is delegated to the
public ColabFold MMseqs2 API (``https://api.colabfold.com``), the same backend
used by ColabFold and Boltz. The host can be overridden with the
``MMSEQS_SERVICE_HOST_URL`` environment variable to point at a self-hosted
MMseqs2 server that speaks the same protocol.
"""

from __future__ import annotations

import json
import logging
import os
import random
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[elapsed: {elapsed} estimate remaining: {remaining}]"
)
DEFAULT_MSA_SERVICE_HOST_URL = os.getenv(
    "MMSEQS_SERVICE_HOST_URL", "https://api.colabfold.com"
)
DEFAULT_USER_AGENT = "opendde/1.0"

logger = logging.getLogger(__name__)

MAX_REQUEST_ATTEMPTS = 5
MAX_SUBMIT_WAIT_ATTEMPTS = 60
MAX_STATUS_WAIT_ATTEMPTS = 720


def _safe_extract_tar(tar_gz: tarfile.TarFile, destination: str | Path) -> None:
    destination_path = Path(destination).resolve()
    for member in tar_gz.getmembers():
        member_path = (destination_path / member.name).resolve()
        if (
            member_path != destination_path
            and destination_path not in member_path.parents
        ):
            msg = f"Refusing to extract unsafe tar member: {member.name}"
            raise RuntimeError(msg)
    tar_gz.extractall(destination_path, filter="data")


def gather_a3m_lines(a3m_files: Sequence[str], order_ids: Sequence[int]) -> list[str]:
    r"""Merge ColabFold a3m files into one a3m string per requested sequence id.

    ColabFold returns results keyed by the numeric FASTA header used at
    submission time (``>101``, ``>102``, ...) and may pack several records into a
    single file separated by ``\\x00``. ``order_ids`` lists those numeric ids in
    the caller's input order, so the returned list is aligned with the input
    sequences.
    """
    collected: dict[int, list[str]] = {}
    for a3m_file in a3m_files:
        update_id = True
        current_id: int | None = None
        with Path(a3m_file).open() as handle:
            for raw_line in handle:
                line = raw_line
                if not line:
                    continue
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_id = True
                if line.startswith(">") and update_id:
                    current_id = int(line[1:].rstrip())
                    update_id = False
                    collected.setdefault(current_id, [])
                if current_id is not None:
                    collected[current_id].append(line)
    return ["".join(collected[seq_id]) for seq_id in order_ids]


class _MSAService:
    """ColabFold-compatible transport with bounded retries and atomic downloads."""

    def __init__(
        self,
        host_url: str,
        submission_endpoint: str,
        headers: dict[str, str],
        email: str,
    ) -> None:
        self.host_url = host_url
        self.submission_endpoint = submission_endpoint
        self.headers = headers
        self.email = email

    def request_with_retries(
        self, method: str, url: str, operation: str, **kwargs: Any
    ) -> requests.Response:
        """Retry transient HTTP failures up to the configured attempt limit."""
        last_error: BaseException | None = None
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                response = requests.request(
                    method, url, timeout=6.02, headers=self.headers, **kwargs
                )
                response.raise_for_status()
                _try_result = response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= MAX_REQUEST_ATTEMPTS:
                    raise
                sleep_time = min(30, 2**attempt)
                logger.warning(
                    "%s failed. Retrying in %ss... (%s/%s): %s",
                    operation,
                    sleep_time,
                    attempt,
                    MAX_REQUEST_ATTEMPTS,
                    exc,
                )
                time.sleep(sleep_time)
            else:
                return _try_result
        msg = f"{operation} failed after retries: {last_error}"
        raise RuntimeError(msg)

    @staticmethod
    def response_json(response: requests.Response) -> dict:
        """Decode the service response, mapping invalid JSON to an error status."""
        try:
            return response.json()
        except ValueError:
            logger.exception("Server did not reply with JSON: %s", response.text)
            return {"status": "ERROR"}

    def submit(self, seqs: Sequence[str], mode: str, start_id: int = 101) -> dict:
        """Submit numbered FASTA sequences and search mode to the service."""
        query_payload = "".join(
            f">{start_id + i}\n{seq}\n" for i, seq in enumerate(seqs)
        )
        data = {"q": query_payload, "mode": mode}
        if self.email:
            data["email"] = self.email
        response = self.request_with_retries(
            "POST",
            f"{self.host_url}/{self.submission_endpoint}",
            "Submitting to MSA server",
            data=data,
        )
        return self.response_json(response)

    def status(self, ticket_id: str) -> dict:
        """Fetch the current status of an accepted ticket."""
        response = self.request_with_retries(
            "GET",
            f"{self.host_url}/ticket/{ticket_id}",
            "Fetching MSA status",
        )
        return self.response_json(response)

    def download(self, ticket_id: str, path: str) -> None:
        """Validate the downloaded archive before replacing a cached result."""
        response = self.request_with_retries(
            "GET",
            f"{self.host_url}/result/download/{ticket_id}",
            "Downloading MSA result",
        )
        tmp_path = f"{path}.tmp"
        try:
            with Path(tmp_path).open("wb") as out:
                out.write(response.content)
            with tarfile.open(tmp_path) as tar_gz:
                tar_gz.getmembers()
            Path(tmp_path).replace(path)
        except Exception:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            raise

    def wait_for_result(
        self, out: dict[str, Any], pbar: tqdm, time_estimate: int
    ) -> str:
        """Wait for an accepted ticket and report service failures explicitly."""
        current_status = out.get("status")
        if current_status == "ERROR":
            msg = (
                "MMseqs2 API returned an error. Please confirm the input "
                "is a valid protein sequence and try again later."
            )
            raise RuntimeError(msg)
        if current_status == "MAINTENANCE":
            msg = "MMseqs2 API is undergoing maintenance. Please try again later."
            raise RuntimeError(msg)
        ticket_id = out.get("id")
        if not ticket_id:
            msg = f"MSA service response did not contain a ticket: {out}"
            raise RuntimeError(msg)

        elapsed = 0
        pbar.set_description(current_status)
        for _ in range(MAX_STATUS_WAIT_ATTEMPTS):
            if current_status not in ["UNKNOWN", "RUNNING", "PENDING"]:
                break
            sleep_time = 5 + random.randint(0, 5)
            logger.error("Sleeping for %ss. Reason: %s", sleep_time, current_status)
            time.sleep(sleep_time)
            out = self.status(ticket_id)
            current_status = out.get("status")
            pbar.set_description(current_status)
            if current_status == "RUNNING":
                elapsed += sleep_time
                pbar.n = min(time_estimate - 1, elapsed)
                pbar.refresh()
        else:
            msg = "MSA service result did not complete before timeout."
            raise RuntimeError(msg)

        if current_status == "COMPLETE":
            pbar.n = time_estimate
            pbar.refresh()
        elif current_status == "ERROR":
            msg = (
                "MMseqs2 API returned an error. Please confirm the input "
                "is a valid protein sequence and try again later."
            )
            raise RuntimeError(msg)
        else:
            msg = f"Unexpected MSA service status: {current_status}"
            raise RuntimeError(msg)

        return str(ticket_id)


def run_mmseqs2(  # noqa: C901, PLR0915 - MSA cache transaction spans polling and extraction
    query: str | Sequence[str],
    prefix: str,
    *,
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = DEFAULT_MSA_SERVICE_HOST_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    email: str = "",
) -> list[str]:
    """Run an MMseqs2 search against the ColabFold-compatible MSA service.

    Returns one merged a3m string per input sequence, in input order. Adapted
    from ColabFold/Boltz ``run_mmseqs2`` with credential-free requests and safe
    tar extraction.
    """
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"
    headers = {"User-Agent": user_agent} if user_agent else {}
    if not headers:
        logger.warning("No user agent specified for MSA service requests.")

    client = _MSAService(host_url, submission_endpoint, headers, email)

    seqs = [query] if isinstance(query, str) else list(query)

    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = "pairgreedy" if pairing_strategy == "greedy" else "paircomplete"
        if use_env:
            mode = f"{mode}-env"

    Path(prefix).mkdir(parents=True, exist_ok=True)
    tar_gz_file = str(Path(prefix) / ("out.tar.gz"))
    manifest_file = str(Path(prefix) / ("out.manifest.json"))

    start_id = 101
    seqs_unique = list(dict.fromkeys(seqs))
    order_ids = [start_id + seqs_unique.index(seq) for seq in seqs]

    if use_pairing:
        a3m_files = [str(Path(prefix) / ("pair.a3m"))]
    else:
        a3m_files = [str(Path(prefix) / ("uniref.a3m"))]
        if use_env:
            a3m_files.append(str(Path(prefix) / ("bfd.mgnify30.metaeuk30.smag30.a3m")))

    cache_signature = {
        "version": 1,
        "endpoint": submission_endpoint,
        "mode": mode,
        "start_id": start_id,
        "seqs": seqs_unique,
    }

    def cache_matches() -> bool:
        """Compute cache matches."""
        if not Path(tar_gz_file).is_file() or not Path(manifest_file).is_file():
            return False
        try:
            with Path(manifest_file).open() as handle:
                return json.load(handle) == cache_signature
        except (OSError, ValueError):
            return False

    def clear_cached_result() -> None:
        """Clear cached result."""
        for path in [tar_gz_file, manifest_file, *a3m_files]:
            if Path(path).is_file():
                Path(path).unlink()

    def wait_for_submission() -> dict:
        """Compute wait for submission."""
        out = client.submit(seqs_unique, mode, start_id)
        for _ in range(MAX_SUBMIT_WAIT_ATTEMPTS):
            current_status = out.get("status")
            if current_status not in ["UNKNOWN", "RATELIMIT"]:
                return out
            sleep_time = 5 + random.randint(0, 5)
            logger.error("Sleeping for %ss. Reason: %s", sleep_time, current_status)
            time.sleep(sleep_time)
            out = client.submit(seqs_unique, mode, start_id)
        msg = "MSA service did not accept the request before timeout."
        raise RuntimeError(msg)

    logger.info("MSA server search started against %s.", host_url)
    if not cache_matches():
        clear_cached_result()
        time_estimate = 150 * len(seqs_unique)
        with tqdm(total=time_estimate, bar_format=TQDM_BAR_FORMAT) as pbar:
            pbar.set_description("SUBMIT")
            out = wait_for_submission()
            ticket_id = client.wait_for_result(out, pbar, time_estimate)

            client.download(ticket_id, tar_gz_file)
            tmp_manifest = f"{manifest_file}.tmp"
            with Path(tmp_manifest).open("w") as handle:
                json.dump(cache_signature, handle, sort_keys=True)
            Path(tmp_manifest).replace(manifest_file)

    if any(not Path(a3m_file).is_file() for a3m_file in a3m_files):
        try:
            with tarfile.open(tar_gz_file) as tar_gz:
                _safe_extract_tar(tar_gz, prefix)
        except (tarfile.TarError, OSError):
            clear_cached_result()
            raise

    return gather_a3m_lines(a3m_files, order_ids)


def _first_a3m_sequence(a3m_text: str) -> str:
    sequence_lines = []
    in_first_record = False
    for line in a3m_text.splitlines():
        if line.startswith(">"):
            if in_first_record:
                break
            in_first_record = True
            continue
        if in_first_record:
            sequence_lines.append(line.strip())
    return "".join(sequence_lines)


def _write_query_leading_a3m(path: str, a3m_text: str | None, query_seq: str) -> None:
    """Write an a3m whose first record is the query under a ``>query`` header.

    When ``a3m_text`` is missing or does not start with the requested query, fall
    back to a query-only a3m so downstream featurization has a valid self MSA.
    """
    if a3m_text and a3m_text.strip():
        first_sequence = _first_a3m_sequence(a3m_text)
        if first_sequence.upper() == query_seq.upper():
            normalized = []
            replaced = False
            for line in a3m_text.splitlines():
                if not replaced and line.startswith(">"):
                    normalized.append(">query")
                    replaced = True
                else:
                    normalized.append(line)
            text = "\n".join(normalized).rstrip("\n") + "\n"
        else:
            logger.warning(
                "MSA query mismatch for %s; falling back to query-only MSA.", path
            )
            text = f">query\n{query_seq}\n"
    else:
        text = f">query\n{query_seq}\n"
    with Path(path).open("w") as handle:
        handle.write(text)


def search_and_build_msa(
    seqs: Sequence[str],
    msa_res_dir: str,
    host_url: str = DEFAULT_MSA_SERVICE_HOST_URL,
    email: str = "",
) -> list[str]:
    """Search MSAs for ``seqs`` and write per-sequence pairing/non-pairing a3m.

    For each input sequence, writes ``<msa_res_dir>/<i>/non_pairing.a3m`` and
    ``<msa_res_dir>/<i>/pairing.a3m``. Paired MSAs are only meaningful for
    multi-chain complexes; a single sequence gets a query-only pairing file.
    Network failures fall back to query-only MSAs so inference can still run.
    """
    Path(msa_res_dir).mkdir(parents=True, exist_ok=True)
    seqs = list(seqs)
    seq_dirs = []
    for idx in range(len(seqs)):
        seq_dir = str(Path(str(Path(msa_res_dir) / (str(idx)))).absolute())
        Path(seq_dir).mkdir(parents=True, exist_ok=True)
        seq_dirs.append(seq_dir)

    unpaired: Sequence[str | None] = [None] * len(seqs)
    try:
        unpaired = run_mmseqs2(
            seqs,
            str(Path(msa_res_dir) / ("unpaired")),
            use_env=True,
            use_pairing=False,
            host_url=host_url,
            email=email,
        )
    except Exception:
        logger.exception("Unpaired MSA search failed; falling back to query-only MSA.")

    paired: Sequence[str | None] = [None] * len(seqs)
    if len(seqs) > 1:
        try:
            paired = run_mmseqs2(
                seqs,
                str(Path(msa_res_dir) / ("paired")),
                use_env=False,
                use_pairing=True,
                pairing_strategy="greedy",
                host_url=host_url,
                email=email,
            )
        except Exception:
            logger.exception(
                "Paired MSA search failed; falling back to query-only pairing."
            )

    for idx, (seq, seq_dir) in enumerate(zip(seqs, seq_dirs, strict=False)):
        _write_query_leading_a3m(
            str(Path(seq_dir) / ("non_pairing.a3m")), unpaired[idx], seq
        )
        _write_query_leading_a3m(str(Path(seq_dir) / ("pairing.a3m")), paired[idx], seq)
    return seq_dirs
