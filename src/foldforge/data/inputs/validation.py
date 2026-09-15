# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Validation shared by inference loading and output path construction."""

from __future__ import annotations

from typing import Any

_MAX_NUMPY_SEED = 2**32 - 1


def validate_inference_seed(value: Any, *, location: str = "seed") -> int:
    """Validate a seed before it reaches NumPy/PyTorch RNG setup."""
    if isinstance(value, bool):
        msg = f"{location} must be an integer, not a boolean."
        raise TypeError(msg)
    if isinstance(value, int):
        seed = value
    elif isinstance(value, str) and value.strip() == value and value.isdecimal():
        # Preserve compatibility with older JSON/CLI inputs that quoted seeds.
        seed = int(value)
    else:
        msg = f"{location} must be an integer; got {value!r}."
        raise ValueError(msg)
    if not 0 <= seed <= _MAX_NUMPY_SEED:
        msg = f"{location} must be in [0, {_MAX_NUMPY_SEED}]; got {seed}."
        raise ValueError(msg)
    return seed


def validate_sample_name(value: Any, *, job_index: int | None = None) -> str:
    """Return a safe output path component for one inference job."""
    location = f" for job {job_index}" if job_index is not None else ""
    if not isinstance(value, str) or not value.strip():
        msg = f"Inference job name{location} must be a non-empty string."
        raise ValueError(msg)
    if value.casefold() == "err":
        msg = f"Inference job name{location} {value!r} is reserved for error reports."
        raise ValueError(msg)
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        msg = (
            f"Inference job name{location} must be a single safe path component; "
            f"got {value!r}."
        )
        raise ValueError(msg)
    return value


def validate_inference_jobs(value: Any) -> list[dict[str, Any]]:
    """Validate the top-level inference job list and collision-sensitive fields."""
    if not isinstance(value, list) or not value:
        msg = (
            "Input JSON must be a non-empty top-level list, "
            f"got {type(value).__name__}."
        )
        raise ValueError(msg)

    jobs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, job in enumerate(value):
        if not isinstance(job, dict):
            msg = f"Inference job {index} must be an object, got {type(job).__name__}."
            raise TypeError(msg)
        name = validate_sample_name(job.get("name"), job_index=index)
        if name in seen_names:
            msg = (
                f"Inference job name {name!r} is duplicated in the same input JSON; "
                "duplicate names would overwrite outputs."
            )
            raise ValueError(msg)
        seen_names.add(name)

        model_seeds = job.get("modelSeeds")
        if model_seeds is not None:
            if not isinstance(model_seeds, list):
                msg = f"modelSeeds for job {name!r} must be a list of integers."
                raise ValueError(msg)
            for seed in model_seeds:
                try:
                    validate_inference_seed(
                        seed,
                        location=f"modelSeeds for job {name!r}",
                    )
                except ValueError as exc:
                    msg = f"modelSeeds for job {name!r} contains invalid seed {seed!r}."
                    raise ValueError(msg) from exc
        jobs.append(job)
    return jobs
