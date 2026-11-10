# Chemistry and model implementations are imported at each tested boundary.
# ruff: noqa: TC003
"""Shared CCD ownership regressions (the pairformer composition is tested in test_engine_graph.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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

