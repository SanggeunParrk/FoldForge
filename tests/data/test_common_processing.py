"""Shared chemistry processing must preserve both supported token layouts."""

import importlib
from pathlib import Path

import pytest

import foldforge

#: Every name a file must not be called after. A file named for a model is a
#: file someone will put model-specific code in, and the point of this port is
#: that a model is a ROW in a table, not a module.
_MODEL_NAMES = (
    "af3",
    "alphafold",
    "boltz",
    "chai1",
    "esmc",
    "esmfold",
    "intellifold",
    "openbind",
    "opendde",
    "openfold",
    "protenix",
    "rosettafold",
)


def test_no_file_is_named_after_a_model():
    """Across the whole package, with one exception.

    `architectures/` holds complete networks, and a network IS one model's
    topology -- naming those files after their model is what they are for.
    Nothing else gets to: a configuration, a converter, a constant table or a
    feature step belongs to a RESPONSIBILITY, and the model it came from is a
    fact for its docstring.

    Directory names are left alone. A vendored upstream config tree keeps the
    vendor's own layout so it can be diffed against the source it was taken
    from, and a provenance record is keyed by the model whose weights it
    documents.
    """
    root = Path(foldforge.__file__).parent
    architectures = root / "models" / "architectures"
    offenders = [
        path
        for path in root.rglob("*.py")
        if architectures not in path.parents
        and any(name in path.stem.lower() for name in _MODEL_NAMES)
    ]
    assert offenders == [], offenders


@pytest.mark.parametrize(
    "module",
    [
        "foldforge.data.features.protenix_constants",
        "foldforge.data.features.opendde_tokenizer",
        "foldforge.eval.protenix_clash",
        "foldforge.modules.ops.opendde_layer_norm",
    ],
)
def test_retired_modules_are_not_compatibility_wrappers(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
