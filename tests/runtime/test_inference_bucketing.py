"""Padding preserves real features, masks and public output shapes."""

import pytest
import torch
from team_gm.modules.bucketing import (
    ATOM_SHAPES,
    MSA_SHAPES,
    TOKEN_SHAPES,
    BucketShape,
    pad_axis,
    token_bucket,
)


def inputs(length=129, atoms=129, msa=3, device="cpu"):
    values = {
        name: torch.arange(length, device=device)[None]
        for name in (
            "residue_type",
            "residue_index",
            "asym_id",
            "sym_id",
            "entity_id",
            "token_index",
            "mol_type",
            "representative_atom_index",
            "deletion_mean",
        )
    }
    values.update(
        {
            name: torch.arange(atoms, device=device)[None]
            for name in ("ref_charge", "ref_element", "ref_space_uid", "atom_to_token")
        }
    )
    values.update(
        mask=torch.ones(1, length, dtype=torch.bool, device=device),
        atom_mask=torch.ones(1, atoms, dtype=torch.bool, device=device),
        token_bonds=torch.ones(1, length, length, device=device),
        ref_pos=torch.ones(1, atoms, 3, device=device),
        ref_atom_name_chars=torch.ones(1, atoms, 4, device=device),
        msa=torch.ones(1, msa, length, device=device, dtype=torch.long),
        has_deletion=torch.zeros(1, msa, length, device=device),
        deletion_value=torch.zeros(1, msa, length, device=device),
        lm_hidden_states=torch.randn(1, length, 16, device=device),
    )
    return values


def test_policy_keeps_atom_msa_edges_and_uses_128_step_tokens():
    from miniworld_engine.autotune.shape_key import ATOM_SHAPES as ENGINE_ATOMS
    from miniworld_engine.autotune.shape_key import TOKEN_SHAPES as ENGINE_TOKENS
    from miniworld_engine.autotune.shape_key import token_key

    assert ENGINE_ATOMS == ATOM_SHAPES
    assert MSA_SHAPES == (1024, 2048, 4096, 8192, 16384)
    from team_gm.modules.bucketing import ATOM_BUCKETS

    assert ATOM_BUCKETS[: len(ATOM_SHAPES)] == ATOM_SHAPES
    assert ATOM_BUCKETS[len(ATOM_SHAPES) :] == (16384, 32768)
    assert BucketShape.select(4096, 31456, 3).atom_bucket == 32768
    assert BucketShape.select(1024, 8193, 3).atom_bucket == 16384
    with pytest.raises(ValueError, match="atoms"):
        BucketShape.select(4096, 32769, 3)
    assert BucketShape.select(1140, 4591, 2049) == BucketShape(
        1140, 4591, 2049, 1152, 8192, 4096
    )
    # Physical padding never changes the engine's tuning key or its top clamp.
    assert ENGINE_TOKENS == (128, 256, 384, 512, 640, 768)
    assert token_key(1152, K=384) == token_key(768, K=384)
    assert tuple(range(128, 8193, 128)) == TOKEN_SHAPES
    for size in (0, -1):
        with pytest.raises(ValueError, match="tokens"):
            token_bucket(size)
    with pytest.raises(ValueError, match="atoms"):
        BucketShape.select(1140, 32769, None)
    with pytest.raises(ValueError, match="msa"):
        BucketShape.select(1140, 4591, 16385)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1, 128),
        (128, 128),
        (129, 256),
        (768, 768),
        (769, 896),
        (1025, 1152),
        (1140, 1152),
        (1152, 1152),
        (1153, 1280),
        (8193, 8320),
    ],
)
def test_token_bucket_boundaries(size, expected):
    assert token_bucket(size) == expected


def test_af3_featurizer_uses_same_token_policy():
    from alphafold3.model.pipeline.pipeline import calculate_bucket_size

    for size in (769, 1025, 1140, 1152, 1153, 8192):
        assert calculate_bucket_size(size, TOKEN_SHAPES) == token_bucket(size)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real compile and CUDA capture"
)
@pytest.mark.parametrize("lengths", [(129, 177), (1025, 1140)])
def test_same_bucket_reuses_graph_with_different_valid_lengths(lengths):
    from team_gm.modules.execution import ExecutedCallable
    from torch._dynamo.utils import counters

    from foldforge.models.config import ExecutionConfig

    before = counters["stats"]["unique_graphs"]
    fn = ExecutedCallable(
        lambda x, mask: (x.sin() * mask[..., None]).sum(1),
        ExecutionConfig(compile=True, cuda_graph=True, max_graphs=1),
        "bucket regression",
    )
    with torch.no_grad():
        for length in lengths:
            original = inputs(length=length, device="cuda")
            bucket = token_bucket(length)
            actual = fn(
                pad_axis(original["lm_hidden_states"], 1, bucket),
                pad_axis(original["mask"], 1, bucket),
            )
            torch.testing.assert_close(
                actual, original["lm_hidden_states"].sin().sum(1)
            )
    assert fn.captures == 1
    assert fn.replays == 2
    assert counters["stats"]["unique_graphs"] > before
