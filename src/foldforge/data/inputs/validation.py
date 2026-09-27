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
        raise ValueError(msg)  # noqa: TRY004 - shared config/CLI validation contract
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
