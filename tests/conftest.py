"""Keep prediction-writing tests inside isolated temporary run roots."""

import pytest

from foldforge.models.io import paths


@pytest.fixture(autouse=True)
def isolated_run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
