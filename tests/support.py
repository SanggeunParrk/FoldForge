"""Shared test paths and released-source oracle lookup."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(__file__).parent / "references"
_MODULES = json.loads((REFERENCE_ROOT / "current_modules.json").read_text())

_SYMBOLS = json.loads((REFERENCE_ROOT / "current_symbols.json").read_text())


def production_module(family, path):
    key = f"foldforge.models.{family}.ported." + path.removesuffix(".py").replace(
        "/", "."
    )
    module = importlib.import_module(_MODULES[key])
    names = _SYMBOLS.get(key)
    if not names:
        return module
    return SimpleNamespace(
        **(vars(module) | {old: getattr(module, new) for old, new in names.items()})
    )


def triton_ops_run() -> int:
    """1 where the Triton-only engine ``ops`` serve a call; 0 on H100 / B200."""
    import torch
    from miniworld_engine.modules.dispatch import restricted_card

    return (
        0 if torch.cuda.is_available() and restricted_card(torch.device("cuda")) else 1
    )
