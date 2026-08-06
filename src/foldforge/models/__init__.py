"""The predictors, and the name -> loader table that finds them.

A registry rather than a package of imports because loading any one predictor
drags in its weights and, for some, an upstream package. Importing
``foldforge.models`` must stay cheap enough that a CLI can list what exists
without a GPU, so each implemented entry is a *thunk* that imports on call.

Every planned predictor has a package from the start, even before it is ported,
because the port is where the boundary decisions get made and they should be
written down next to the model rather than in an issue. An unported entry
carries ``loader=None``: :func:`get_model` then fails immediately and says so,
instead of handing back something that raises later from inside a forward pass.

Adding a predictor: fill in its package's ``load``, then point its entry's
``loader`` at a thunk that imports it *inside the function*, so that listing the
registry still needs neither the weights nor an upstream package::

    def _esmfold2() -> Any:
        from foldforge.models.esmfold2 import load

        return load

There is deliberately no ``FoldingModel`` Protocol yet. With one predictor
ported, any interface written now would be a guess dressed as a contract; the
second one is what tells us where the models actually agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Entry:
    """One predictor's place in the registry."""

    #: Zero-arg thunk returning the model's ``load``; ``None`` until ported.
    loader: Callable[[], Any] | None
    #: What it is and where the weights come from, for the CLI listing.
    summary: str


def _esmfold2() -> Any:
    # Imported on call, not at module scope: this pulls in torch, the engine and
    # transformers, and listing the registry must need none of them.
    from foldforge.models.esmfold2 import load  # noqa: PLC0415

    return load


#: Every predictor this repo plans to run.
_REGISTRY: dict[str, Entry] = {
    "af3": Entry(None, "AlphaFold 3 — DeepMind; weights are access-gated"),
    "boltz2": Entry(None, "Boltz-2 — MIT"),
    "chai1": Entry(None, "Chai-1 — Chai Discovery"),
    "esmfold2": Entry(_esmfold2, "ESMFold2 — biohub/ESMFold2 + ESMC-6B"),
    "opendde": Entry(None, "OpenDDE"),
    "protenix": Entry(None, "Protenix v1 / v2 — ByteDance"),
}


def known_models() -> list[str]:
    """Every predictor name in the registry, ported or not."""
    return sorted(_REGISTRY)


def registered_models() -> list[str]:
    """Return the predictors that are actually ported and loadable."""
    return sorted(name for name, entry in _REGISTRY.items() if entry.loader)


def describe(name: str) -> str:
    """One-line summary of ``name``, for listings."""
    return _entry(name).summary


def _entry(name: str) -> Entry:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_models())
        msg = f"unknown model {name!r}; known: {known}"
        raise KeyError(msg) from None


def get_model(name: str) -> Any:
    """Return the loader for ``name``, importing its package on first use."""
    entry = _entry(name)
    if entry.loader is None:
        ported = ", ".join(registered_models())
        msg = (
            f"{name} is planned but not ported yet ({entry.summary}). "
            f"Ported: {ported}. See src/foldforge/models/{name}/__init__.py."
        )
        raise NotImplementedError(msg)
    return entry.loader()
