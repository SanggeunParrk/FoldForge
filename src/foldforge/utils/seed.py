"""Request-scoped randomness shared by feature preparation and inference."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 ByteDance and/or its affiliates.
# Copyright (c) 2026 Aureka AI Research
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from foldforge.data.inputs.validation import validate_inference_seed

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class RNGState:
    """Snapshot all process RNGs, including every visible CUDA device."""

    python: tuple[Any, ...]
    numpy: Any
    cpu: torch.Tensor
    cuda: list[torch.Tensor] | None

    @classmethod
    def capture(cls) -> RNGState:
        """Read random streams without advancing them."""
        return cls(
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    def restore(self) -> None:
        """Restore the captured streams after setup or repeated measurement."""
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


def seed_all(seed: int) -> None:
    """Initialize Python, NumPy and Torch CPU/CUDA from one request seed."""
    seed = validate_inference_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def seed_context(seed: int | None) -> Iterator[None]:
    """Isolate a seeded request or feature builder from its caller's RNGs."""
    if seed is None:
        yield
        return
    state = RNGState.capture()
    try:
        seed_all(seed)
        yield
    finally:
        state.restore()


def conformer_seed(seed: int | None = None) -> int:
    """Supply RDKit a nonnegative signed seed from the request-controlled stream.

    Explicit 32-bit input seeds are folded into RDKit's signed 31-bit range.
    Without an explicit seed, draw once in the parent thread, before workers run.
    """
    if seed is None:
        return int(np.random.randint(0, 2**31))
    return validate_inference_seed(seed) % (2**31)


def seed_everything(seed: int, *, deterministic: bool) -> None:
    """Seed everything."""
    seed_all(seed)
    # These are process-wide switches. Set both branches explicitly so a
    # non-deterministic run cannot inherit True from an earlier deterministic
    # run in a long-lived Python process.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
