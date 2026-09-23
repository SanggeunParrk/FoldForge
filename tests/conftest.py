"""Keep prediction-writing tests inside isolated temporary run roots."""

import pytest

from foldforge.models.io import paths


@pytest.fixture(autouse=True)
def isolated_run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)


#: Every family reads a checkpoint and an input; a CLI test that is about
#: something else still has to say so. Spelt once here so a test's argument
#: list shows only what that test is actually about.
REQUIRED_CLI = ["--input", "input.json", "--checkpoint", "weights.pt"]


def cli(*extra: str) -> list[str]:
    """Return the required flags followed by whatever the test is testing."""
    return [*REQUIRED_CLI, *extra]
