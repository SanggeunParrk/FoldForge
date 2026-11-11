"""OpenDDE's structural pair is built in row bands and equals the whole-tensor construction it replaced.

The whole-tensor version held five ``[S, S, c]`` fp32 tensors at once (12.4 GiB each at 12 chains of 5I28): the out-of-memory step.
"""

from __future__ import annotations

import pytest
import torch

from foldforge.modules.dense import structural_tokens as st

C_PAIR, RESIDUES = 8, 7
PARENT = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 6])
ROLE = torch.tensor([1, 2, 1, 2, 1, 1, 2, 4, 1, 2, 1, 2, 0])  # protein bb / sc, a DNA base, an atom


def _whole_tensor_expand_pair(module, pair, parent, features):
    """The construction this replaced, verbatim."""
    parent_pair = pair[parent][:, parent]
    role_pair = features["role_pair_type"]
    role = features["role"]
    block = role[:, None] * st.N_ROLES + role[None, :]
    delta = torch.zeros_like(parent_pair)
    for index in range(st.N_ROLES * st.N_ROLES):
        chosen = block == index
        if not bool(chosen.any()):
            continue
        masked = torch.where(chosen[..., None], parent_pair, 0.0)
        delta = delta + masked @ module.pair_block_proj[index]
    out = parent_pair + delta
    for name, table in (
        ("same_parent", module.same_parent_embedding),
        ("twin", module.same_residue_twin_embedding),
        ("prev_bb", module.prev_bb_chain_embedding),
        ("next_bb", module.next_bb_chain_embedding),
    ):
        out = out + table(features[name].to(torch.int64))
    return out + module.role_pair_type_embedding(role_pair)


@pytest.fixture
def case():
    torch.manual_seed(0)
    module = st.StructuralTokenExpander(c_single=4, c_pair=C_PAIR, c_target_feat=3)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    asym = torch.zeros(RESIDUES, dtype=torch.int64)
    prev_parent = (PARENT - 1).clamp(min=-1)  # per structural token: its parent's neighbours in the chain
    next_parent = torch.where(PARENT + 1 < RESIDUES, PARENT + 1, torch.full_like(PARENT, -1))
    features = module.pair_features(PARENT, ROLE, asym, prev_parent, next_parent)
    pair = torch.randn(RESIDUES, RESIDUES, C_PAIR)
    return module, pair, features


@pytest.mark.parametrize("band_bytes", [1 << 30, 13 * C_PAIR * 4 * 5, 1])
def test_banded_expansion_equals_the_whole_tensor_one(case, monkeypatch, band_bytes):
    module, pair, features = case
    monkeypatch.setattr(st, "_PAIR_BAND_BYTES", band_bytes)  # 1 byte: one row per band
    want = _whole_tensor_expand_pair(module, pair, PARENT, features)
    got = module.expand_pair(pair, PARENT, features)
    assert got.dtype is torch.float32
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_under_autocast_the_pair_is_stored_in_the_autocast_dtype(case, monkeypatch):
    module, pair, features = case
    monkeypatch.setattr(st, "_PAIR_BAND_BYTES", 13 * C_PAIR * 4 * 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        want = _whole_tensor_expand_pair(module, pair, PARENT, features)
        got = module.expand_pair(pair, PARENT, features)
    assert want.dtype is torch.float32  # what the old construction handed the first refiner block, which rounded it to bf16
    assert got.dtype is torch.bfloat16
    torch.testing.assert_close(got.float(), want.to(torch.bfloat16).float(), rtol=2e-2, atol=2e-2)


def test_only_one_band_of_fp32_is_alive_at_a_time(case, monkeypatch):
    module, pair, features = case
    monkeypatch.setattr(st, "_PAIR_BAND_BYTES", 13 * C_PAIR * 4 * 2)  # two rows
    seen: list[int] = []
    original = module.role_pair_type_embedding.forward

    def watch(index):
        seen.append(index.shape[0])
        return original(index)

    monkeypatch.setattr(module.role_pair_type_embedding, "forward", watch)
    module.expand_pair(pair, PARENT, features)
    assert max(seen) == 2
    assert sum(seen) == PARENT.shape[0]
