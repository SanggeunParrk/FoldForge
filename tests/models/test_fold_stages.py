"""Stage boundaries preserve residue heads, structural diffusion and recurrent state."""
# ruff: noqa: SLF001 - these tests exercise the internal stage contracts

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from foldforge.models.architectures import af3
from foldforge.modules.dense.spec import SPECS


@pytest.mark.parametrize("family", sorted(SPECS))
def test_diffusion_layout_roundtrip_keeps_residue_head_inputs(monkeypatch, family):
    model = af3.AlphaFold3.__new__(af3.AlphaFold3)
    nn.Module.__init__(model)
    model.spec = SPECS[family]
    residue = SimpleNamespace(
        num_res=2, token_features=SimpleNamespace(asym_id=torch.ones(2))
    )
    structural = SimpleNamespace(num_res=3)
    embeddings = {"pair": torch.randn(2, 2, 2), "single": torch.randn(2, 2)}
    expanded = {"pair": torch.randn(3, 3, 2)}
    gather = torch.tensor([[0, 2, -1], [1, 3, -1]])
    raw = {"struct/marker": torch.tensor(7), "structbook/residue_atom_gather": gather}

    def convert(data) -> SimpleNamespace:
        assert set(data) == {"marker"}
        return structural

    def expand(trunk, book, batch, asym) -> dict[str, torch.Tensor]:
        assert trunk is embeddings
        assert book["residue_atom_gather"] is gather
        assert batch is structural
        assert asym is residue.token_features.asym_id
        return expanded

    monkeypatch.setattr(af3.feat_batch.Batch, "from_data_dict", convert)
    model.evoformer = SimpleNamespace(expand_structural=expand)
    inputs = model._prepare_diffusion_inputs(raw, residue, embeddings)
    if model.spec.structural_tokens:
        assert inputs.batch is structural
        assert inputs.embeddings is expanded
        coords = torch.arange(36.0).reshape(2, 3, 2, 3)
        samples = {"atom_positions": coords, "mask": torch.ones(2, 3, 2)}
        restored = inputs.restore(samples)
        torch.testing.assert_close(
            restored["atom_positions"][:, 0, :2], coords.flatten(1, 2)[:, [0, 2]]
        )
        torch.testing.assert_close(
            restored["atom_positions"][:, 1, :2], coords.flatten(1, 2)[:, [1, 3]]
        )
        assert not restored["mask"][:, :, 2].any()
    else:
        assert inputs.batch is residue
        assert inputs.embeddings is embeddings
        samples = {"atom_positions": torch.randn(2, 2, 3, 3)}
        restored = inputs.restore(samples)
        assert restored is samples

    def confidence(positions, trunk, batch) -> dict[str, torch.Tensor]:
        assert positions is restored["atom_positions"]
        assert trunk is embeddings
        assert batch is residue
        return {"confidence": torch.ones(2)}

    def distogram(batch, trunk) -> dict[str, torch.Tensor]:
        assert batch is residue
        assert trunk is embeddings
        return {"bins": trunk["pair"]}

    monkeypatch.setattr(model, "_confidence_over_samples", confidence)
    model.distogram_head = distogram
    result = model._predict_heads(residue, embeddings, restored)
    assert result["diffusion_samples"] is restored
    assert result["distogram"]["bins"] is embeddings["pair"]


def test_pair_only_finalization_keeps_pre_coda_recycle():
    evo = af3.Evoformer.__new__(af3.Evoformer)
    nn.Module.__init__(evo)
    evo.spec = replace(SPECS["esmfold2"], recycle_from_initial=False, ssm_recycle=False)
    evo.prev_embedding = evo.prev_embedding_layer_norm = nn.Identity()
    evo.prev_single_embedding = evo.prev_single_embedding_layer_norm = nn.Identity()
    evo.trunk_pairformer = nn.ModuleList([])
    evo.parcae_readout = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        evo.parcae_readout.weight.copy_(2 * torch.eye(2))
    evo.trunk_coda = nn.ModuleList([])
    pair, single = torch.ones(2, 2, 2), torch.ones(2, 2)
    target = torch.zeros(2, 2)
    previous = {
        "pair": torch.full_like(pair, 50),
        "pair_pre_coda": pair,
        "single": single,
        "structure_target_feat": target,
    }
    for last in (False, True):
        result = evo._pair_only_trunk(
            prev=previous,
            pair_activations=pair,
            pair_mask=torch.ones(2, 2),
            single_init=single,
            target_feat=target,
            first_pass=False,
            last_pass=last,
        )
        torch.testing.assert_close(result["pair_pre_coda"], 2 * pair)
        torch.testing.assert_close(result["pair"], (4 if last else 2) * pair)


@pytest.mark.parametrize("apply_rate", [0.0, 1.0])
def test_trunk_mc_coin_is_drawn_once_before_all_passes(monkeypatch, apply_rate):
    model = af3.AlphaFold3.__new__(af3.AlphaFold3)
    nn.Module.__init__(model)
    model.spec = replace(SPECS["protenix1"], recycle_mc_dropout=(apply_rate, 0.25))
    model.num_recycles = 10
    model.reference_precision = False
    model.evoformer_pair_channel = model.evoformer_seq_channel = 2
    batch = SimpleNamespace(num_res=2)
    features = torch.ones(2, 2)
    monkeypatch.setattr(
        model, "create_target_feat_embedding", lambda _: (features, features)
    )
    rates = []

    def trunk(**kwargs: Any) -> dict[str, torch.Tensor]:
        rates.append(kwargs["recycle_dropout"])
        return kwargs["prev"]

    model.evoformer = trunk
    with torch.random.fork_rng():
        torch.manual_seed(17)
        torch.rand(())
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(17)
        model._run_trunk(batch)
        assert torch.equal(torch.random.get_rng_state(), expected_state)
    assert rates == [0.25 if apply_rate else 0.0] * 10
    assert model.recycle_dropout_applied == bool(apply_rate)
