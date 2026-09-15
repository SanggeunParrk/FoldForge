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
