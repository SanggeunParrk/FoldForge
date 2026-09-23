# ruff: noqa: PLC0415 - registry listing must not import Torch
"""Predictor metadata and one lazy checkpoint-loading entry point."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Entry:
    """Architecture and input layout; loading policy is shared."""

    architecture: str | None
    layout: str | None
    summary: str
    #: Row of ``foldforge.models.config.dense.SPECS`` for dense AF3-graph families.
    family: str | None = None


_REGISTRY = {
    "af3": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "AlphaFold 3 — DeepMind; weights are access-gated",
    ),
    "intellifold2": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "IntelliFold-v2 — IntelliGen-AI; the AF3 graph at wider channels",
        "intellifold2",
    ),
    "openfold3": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "OpenFold3 v0.5.0 OpenBind — AlQuraishi Lab",
        "openbind0",
    ),
    "openfold3-preview2": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "OpenFold3 preview-2 — AlQuraishi Lab; superseded by v0.5.0",
        "openfold3",
    ),
    "rosettafold3": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "RoseTTAFold3 — RosettaCommons foundry, BSD-3-Clause",
        "rosettafold3",
    ),
    "protenix": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "Protenix v1 / v2 — ByteDance; --variant picks the release",
        # The family is the RELEASE, which --variant names, so it is resolved
        # per call rather than fixed here.
        None,
    ),
    "opendde": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "OpenDDE — Aureka AI Research; folds on structural tokens",
        "opendde",
    ),
    "esmfold2": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "ESMFold2 — biohub/ESMFold2 + ESMC-6B; folds from the language model",
        "esmfold2",
    ),
    "esmfold2-fast": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        # The caveat is in the listing because the fold looks ordinary: clean
        # geometry, no clashes, a plausible pLDDT. Only the RMSD gives it away,
        # and a user folding a novel target has nothing to compare against.
        "ESMFold2-Fast — 24 trunk blocks, no MSA stack; NOT YET ACCURATE, see "
        "docs/guides/MODEL-INTEGRATION.md",
        "esmfold2-fast",
    ),
    "boltz2": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "Boltz-2 — Wohlwend et al., MIT",
        "boltz2",
    ),
    "chai1": Entry(
        "foldforge.models.architectures.af3.AlphaFold3",
        "dense_atoms",
        "Chai-1 — Chai Discovery; its token stream is mostly ESM2-3B",
        "chai1",
    ),
}


def known_models() -> list[str]:
    """List all known predictors, including planned integrations."""
    return sorted(_REGISTRY)


def registered_models() -> list[str]:
    """List implemented predictors."""
    return sorted(name for name, entry in _REGISTRY.items() if entry.architecture)


def entry(name: str) -> Entry:
    """Return metadata without importing a network or Torch."""
    try:
        return _REGISTRY[name]
    except KeyError:
        message = f"unknown model {name!r}; known: {', '.join(known_models())}"
        raise KeyError(message) from None


def is_dense(name: str) -> bool:
    """Whether ``name`` is a family of the one dense AF3 graph."""
    return entry(name).layout == "dense_atoms"


def describe(name: str) -> str:
    """Return a short predictor description."""
    return entry(name).summary


def load(name: str, checkpoint: str | Path | None = None, **options: Any) -> Any:
    """Load any supported checkpoint with one precision/backend/strictness contract.

    ``load("protenix", path, variant="protenix-v2", backend="miniworld",
    dtype=torch.bfloat16, device="cuda")``. Importing the registry stays CPU-light.
    """
    model = entry(name)
    if model.architecture is None:
        message = (
            f"{name} is not ported yet; available: {', '.join(registered_models())}"
        )
        raise NotImplementedError(message)
    from foldforge.models.loading import load_checkpoint

    return load_checkpoint(name, checkpoint, **options)


def get_model(name: str) -> Any:
    """Compatibility callable bound to the one loader; no per-model load function."""
    if entry(name).architecture is None:
        message = (
            f"{name} is not ported yet; available: {', '.join(registered_models())}"
        )
        raise NotImplementedError(message)
    return partial(load, name)
