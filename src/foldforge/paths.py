"""Where FoldForge keeps what is not source: weights, databases, runs.

Nothing machine-specific lives in the repository. Everything below sits under
one home directory, ``$FOLDFORGE_HOME`` (default ``~/.cache/foldforge``), the way
the Hugging Face hub keeps its cache; each part can also be pointed elsewhere by
its own variable:

=================  ==========================  ================================
part               default                     override
=================  ==========================  ================================
converted weights  ``<home>/checkpoints/``     ``FOLDFORGE_CHECKPOINT_DIR``
CCD database       ``<home>/ccd/...lmdb``      ``FOLDFORGE_CCD_DB``
prediction runs    ``<home>/runs/``            ``FOLDFORGE_RUNS``
release envs       ``<home>/releases/``        ``FOLDFORGE_RELEASE_ROOT``
benchmark inputs   ``<home>/benchmarks/``      ``FOLDFORGE_BENCHMARKS``
=================  ==========================  ================================
"""

from __future__ import annotations

import os
from pathlib import Path

#: The repository checkout (``<repo>/src/foldforge/paths.py``); only for the
#: tracked files a checkout carries, such as the release references.
REPO = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def home() -> Path:
    """Return FoldForge's home directory."""
    return _env("FOLDFORGE_HOME", Path("~/.cache/foldforge").expanduser())


def checkpoints() -> Path:
    """Return the converted-weights directory, one subdirectory per model."""
    return _env("FOLDFORGE_CHECKPOINT_DIR", home() / "checkpoints")


def ccd_db() -> Path:
    """Return the shared MiniWorld BioMol CCD database."""
    return _env("FOLDFORGE_CCD_DB", home() / "ccd" / "preprocessed_CCD.lmdb")


def runs() -> Path:
    """Return where a prediction goes when its --out is a bare run name."""
    return _env("FOLDFORGE_RUNS", home() / "runs")


def releases() -> Path:
    """Return the released implementations' environments."""
    return _env("FOLDFORGE_RELEASE_ROOT", home() / "releases")


def benchmarks() -> Path:
    """Return the benchmark input sets (replicated targets, databases)."""
    return _env("FOLDFORGE_BENCHMARKS", home() / "benchmarks")


#: The released implementations' reference structures, tracked with the tests.
RELEASE_REFERENCES = REPO / "tests" / "release_references"
