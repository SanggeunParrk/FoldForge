"""The installed CPU feature package and model imports do not require JAX."""

import subprocess
import sys


def test_public_and_feature_imports_without_jax():
    script = """
import sys, importlib.abc, dataclasses
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'haiku', 'chex'}:
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, NoJax())
import numpy as np
import foldforge as ff
from alphafold3.model import features, feat_batch, merging_features
from alphafold3.model.atom_layout import atom_layout
from foldforge.models.architectures.af3 import AlphaFold3
value = np.array([1, 2], dtype=np.int32)
chains = features.Chains(
    chain_id=np.array(['A','B']), asym_id=value, entity_id=value, sym_id=value
)
changed = dataclasses.replace(chains, asym_id=value + 1)
assert (changed.asym_id == [2,3]).all()
assert (chains.asym_id == [1,2]).all()
try:
    chains.asym_id = value
except dataclasses.FrozenInstanceError:
    pass
else:
    raise AssertionError('Feature container is not frozen')
assert ff.load_inputs(ff.protein('AG')).payload['sequences']
assert not any(k.split('.')[0] in {'jax','jaxlib','haiku','chex'} for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_package_dependencies_exclude_jax_frameworks():
    import importlib.metadata
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    forbidden = {"jax", "jaxlib", "dm-haiku", "haiku", "chex"}
    for path in [root / "pyproject.toml", root / "third_party/af3-data/pyproject.toml"]:
        data = tomllib.loads(path.read_text())
        requirements = list(data["project"]["dependencies"])
        for extra in data["project"].get("optional-dependencies", {}).values():
            requirements.extend(extra)
        for requirement in requirements:
            name = (
                re.split(r"[<>=!~;\[ @]", requirement, maxsplit=1)[0]
                .lower()
                .replace("_", "-")
            )
            assert name not in forbidden, (path, requirement)
    for requirement in importlib.metadata.requires("foldforge-af3-data") or []:
        name = (
            re.split(r"[<>=!~;\[ @]", requirement, maxsplit=1)[0]
            .lower()
            .replace("_", "-")
        )
        assert name not in forbidden, requirement
