"""Shared utility contracts that differed between the original consumers."""

from __future__ import annotations

import random

import numpy as np
import torch

from foldforge.utils.seed import seed_everything


def test_seed_reproducibility_and_determinism_reset():
    prior = torch.are_deterministic_algorithms_enabled()
    prior_cudnn = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    import os

    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        seed_everything(5, deterministic=True)
        first = (random.random(), np.random.random(), torch.rand(3))
        seed_everything(5, deterministic=False)
        second = (random.random(), np.random.random(), torch.rand(3))
        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2])
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(prior)
        torch.backends.cudnn.deterministic = prior_cudnn
        torch.backends.cudnn.benchmark = benchmark
        if config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = config
