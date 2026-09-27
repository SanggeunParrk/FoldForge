# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Sets of chemical components."""

import functools
import pickle
from collections.abc import Set, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Final
from alphafold3.common import resources

_SETS_PATH = ContextVar("af3_ccd_sets_path", default=None)

@contextmanager
def use_ccd_sets(path):
  token = _SETS_PATH.set(path if isinstance(path, Mapping) else Path(path).resolve())
  try:
    yield
  finally:
    _SETS_PATH.reset(token)

@functools.lru_cache(maxsize=4)
def _read_sets(path):
  with open(path, "rb") as handle:
    return pickle.load(handle)

class _SelectedSet(Set):
  def __init__(self, name):
    self.name = name
  def _value(self):
    path = _SETS_PATH.get()
    if path is None:
      path = resources.filename(resources.ROOT / "constants/converters/chemical_component_sets.pickle")
    return (path if isinstance(path, Mapping) else _read_sets(path))[self.name]
  def __contains__(self, item):
    return item in self._value()
  def __iter__(self):
    return iter(self._value())
  def __len__(self):
    return len(self._value())

GLYCAN_LINKING_LIGANDS = _SelectedSet("glycans_linking")
GLYCAN_OTHER_LIGANDS = _SelectedSet("glycans_other")

# Each of these molecules appears in over 1k PDB structures, are used to
# facilitate crystallization conditions, but do not have biological relevance.
COMMON_CRYSTALLIZATION_AIDS: Final[frozenset[str]] = frozenset({
    'SO4', 'GOL', 'EDO', 'PO4', 'ACT', 'PEG', 'DMS', 'TRS', 'PGE', 'PG4', 'FMT',
    'EPE', 'MPD', 'MES', 'CD', 'IOD',
})  # pyformat: disable
