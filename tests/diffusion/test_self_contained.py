"""``foldforge.diffusion`` must stay movable into its own package: no import of another foldforge module or of the engine."""

from __future__ import annotations

import ast
from pathlib import Path

import foldforge.diffusion as diffusion

ROOT = Path(diffusion.__file__).parent


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_imports_stay_inside_the_subpackage():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        for module in _imports(path):
            top = module.split(".")[0]
            if top == "foldforge":
                if not (module == "foldforge.diffusion" or module.startswith("foldforge.diffusion.")):
                    offenders.append((str(path.relative_to(ROOT)), module))
            elif top == "miniworld_engine":
                offenders.append((str(path.relative_to(ROOT)), module))
    assert not offenders, offenders


def test_public_names_import():
    for name in diffusion.__all__:
        assert getattr(diffusion, name) is not None, name
