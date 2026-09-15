"""Retry failures must preserve their cause and stop at the retry limit."""

from typing import Never

import numpy as np
import pandas as pd
import pytest
import torch

from foldforge.data.dataset import BaseSingleDataset
from foldforge.data.features.tokenizer import Token, TokenArray


@pytest.mark.parametrize("retry", [False, True])
def test_failed_sample_recovery(monkeypatch: pytest.MonkeyPatch, *, retry: bool):
    dataset = object.__new__(BaseSingleDataset)
    dataset.random_sample_if_failed = retry
    dataset.indices_list = pd.DataFrame({"pdb_id": ["one", "two"]})
    attempts = []
    saved = []
    cause = ValueError("broken sample")

    def fail(idx: int) -> Never:
        attempts.append(idx)
        raise cause

    monkeypatch.setattr(dataset, "process_one", fail)
    monkeypatch.setattr(dataset, "save_error_data", lambda *args: saved.append(args))
    if retry:
        with pytest.raises(RuntimeError, match="10 attempts") as caught:
            dataset[0]
        assert caught.value.__cause__ is cause
        assert len(attempts) == 10
    else:
        with pytest.raises(ValueError, match="broken sample") as caught:
            dataset[0]
        assert caught.value is cause
        assert len(attempts) == 1
    assert len(saved) == len(attempts)


def test_token_indexing_supports_integer_scalars_and_index_arrays():
    tokens = [Token(index) for index in range(4)]
    array = TokenArray(tokens)
    assert array[np.int64(2)] is tokens[2]
    assert array[torch.tensor(1)] is tokens[1]
    assert array[1:3].tokens == tokens[1:3]
    assert array[np.array([3, 1])].tokens == [tokens[3], tokens[1]]
    assert array[torch.tensor([2, 0])].tokens == [tokens[2], tokens[0]]
    with pytest.raises(TypeError, match="Token indices"):
        array[1.5]


def test_balanced_sampler_accepts_explicit_rank_zero_without_process_group():
    from foldforge.data.loader import KeySumBalancedSampler

    dataset = object.__new__(BaseSingleDataset)
    dataset.indices_list = pd.DataFrame({"length": [4, 9, 2, 7]})
    sampler = KeySumBalancedSampler(
        dataset, key="length", num_replicas=2, rank=0, seed=1
    )
    assert len(list(sampler)) == 2
    dataset.indices_list = pd.DataFrame({"length": []})
    with pytest.raises(ValueError, match="empty dataset"):
        KeySumBalancedSampler(dataset, key="length", num_replicas=2, rank=0)
