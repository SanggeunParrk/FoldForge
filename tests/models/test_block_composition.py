# Chemistry and model implementations are imported at each tested boundary.
# ruff: noqa: TC003
"""Shared CCD ownership and released Pairformer composition regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from support import production_module
from team_gm.modules.checkpoints.af_family import configure_model
from team_gm.modules.checkpoints.pairformer import AF3Pairformer, ProtenixPairformer

from foldforge.data.ccd import CCDDatabase, current_database
from foldforge.data.ccd.database import database_cache


def _database(root: Path, glycan: str) -> CCDDatabase:
    root.mkdir()
    import lmdb

    env = lmdb.open(str(root), map_size=1048576)
    env.close()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "miniworld.ccd.biomol.v1",
                "sources": {"components_sha256": glycan, "rdkit_sha256": glycan},
                "files": {"data": {"path": "data.mdb", "sha256": glycan}},
                "chemical_component_sets": {
                    "glycans_linking": [glycan],
                    "glycans_other": [],
                },
            }
        )
    )
    return CCDDatabase(root)


def test_ccd_nested_sources_do_not_share_caches(tmp_path):
    a, b = _database(tmp_path / "a", "A"), _database(tmp_path / "b", "B")
    calls = []

    @database_cache
    def lookup(name) -> str:
        calls.append(name)
        return current_database().config.source_sha256

    with a.activate():
        assert lookup("ALA") == "A"
        with b.activate():
            assert lookup("ALA") == "B"
        assert lookup("ALA") == "A"
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="explicit"):
        current_database()


def test_af3_glycan_sets_follow_selected_database(tmp_path):
    from alphafold3.constants import chemical_component_sets as sets

    a, b = _database(tmp_path / "a", "A"), _database(tmp_path / "b", "B")
    with sets.use_ccd_sets(a.chemical_component_sets):
        assert set(sets.GLYCAN_LINKING_LIGANDS) == {"A"}
        with sets.use_ccd_sets(b.chemical_component_sets):
            assert set(sets.GLYCAN_LINKING_LIGANDS) == {"B"}
        assert set(sets.GLYCAN_LINKING_LIGANDS) == {"A"}


@pytest.mark.parametrize("family", ["af3", "protenix", "opendde"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU composition parity")
def test_shared_pairformer_matches_released_order(family):
    torch.manual_seed(76)
    path = "nn.pairformer" if family == "af3" else "model.modules.pairformer"
    cls = production_module(family, path).PairformerBlock
    source = cls()
    with torch.no_grad():
        for p in source.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.025)
    source = configure_model(source, "pytorch")
    converted = (AF3Pairformer if family == "af3" else ProtenixPairformer)(
        copy.deepcopy(source)
    )
    pair = torch.randn(24, 24, 128, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(24, 384, device="cuda", dtype=torch.bfloat16)
    pair_mask = torch.rand(24, 24, device="cuda") > 0.1
    with torch.inference_mode():
        if family == "af3":
            mask = torch.ones(24, device="cuda")
            expected = source(pair.clone(), pair_mask, single.clone(), mask)
            actual = converted(pair.clone(), pair_mask, single.clone(), mask)
        else:
            expected = source(
                single.clone(),
                pair.clone(),
                pair_mask.float(),
                inplace_safe=True,
                chunk_size=8,
            )
            actual = converted(
                single.clone(), pair.clone(), pair_mask, inplace_safe=True, chunk_size=8
            )
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0.01, atol=0.02)
