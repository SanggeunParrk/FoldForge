"""Where the weights are — asked, never hardcoded.

This resolver lives in FoldForge rather than team-gm on purpose: which
checkpoints exist on which machine is an integration concern, and team-gm is the
framework, which must not know that this cluster keeps ESMC-6B in one place and
AF3 in another.

Lookup order for ``<model>``:

1. ``$FOLDFORGE_CHECKPOINT_DIR/<model>`` — set this to keep ~44 GB off a quota'd
   home directory.
2. ``model_checkpoints/<model>`` next to the repo root.
3. the ``path`` recorded for that model in ``model_checkpoints/registry.yaml``,
   which is where a shared cluster location goes.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

#: Repo root — this file is ``<root>/src/foldforge/models/checkpoints/__init__.py``.
ROOT = Path(__file__).resolve().parents[4]

#: Per-model directories live under here unless the env var overrides it.
DEFAULT_DIR = ROOT / "model_checkpoints"

#: Default weight file inside each model's directory; Protenix names its file by
#: variant and ESMFold2 is a directory, so neither is listed.
DEFAULT_FILES = {
    "af3": "af3.bin.zst",
    "intellifold2": "intellifold2.bin.zst",
    "opendde": "opendde.pt",
}

#: Manifest of known models: ``{name: {path: ..., files: [...]}}``.
REGISTRY = DEFAULT_DIR / "registry.yaml"


def _registry() -> dict[str, dict]:
    """Return the manifest, or empty if it is not set up on this machine."""
    if not REGISTRY.is_file():
        return {}
    loaded = yaml.safe_load(REGISTRY.read_text()) or {}
    return loaded.get("models", loaded)


def _roots(model: str) -> list[Path]:
    """Candidate directories for ``model``, most specific first."""
    candidates: list[Path] = []
    override = os.environ.get("FOLDFORGE_CHECKPOINT_DIR")
    if override:
        candidates.append(Path(override) / model)
    candidates.append(DEFAULT_DIR / model)
    entry = _registry().get(model) or {}
    if entry.get("path"):
        candidates.append(Path(entry["path"]))
    return candidates


def resolve(model: str, filename: str | None = None) -> Path:
    """Absolute path to ``model``'s directory, or to one file inside it.

    Raises rather than returning a missing path: a checkpoint that silently
    resolves to somewhere empty fails much later, inside a load, with an error
    that names a tensor instead of a directory.
    """
    for root in _roots(model):
        target = root if filename is None else root / filename
        if target.exists():
            return target
    searched = "\n  ".join(str(p) for p in _roots(model))
    msg = (
        f"no checkpoint for {model!r}"
        + (f" (file {filename!r})" if filename else "")
        + f". Searched:\n  {searched}\n"
        f"Set FOLDFORGE_CHECKPOINT_DIR, or see model_checkpoints/README.md."
    )
    raise FileNotFoundError(msg)


def available() -> dict[str, bool]:
    """Which registered models are present on this machine."""
    names = set(_registry())
    if DEFAULT_DIR.is_dir():
        names |= {p.name for p in DEFAULT_DIR.iterdir() if p.is_dir()}
    return {name: _present(name) for name in sorted(names)}


def _present(model: str) -> bool:
    """Whether any candidate root for ``model`` exists."""
    return any(root.exists() for root in _roots(model))
