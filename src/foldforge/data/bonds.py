"""Public Biotite bond-list updates shared by the model input ports."""

from __future__ import annotations

from typing import Any

from biotite.structure import BondList


def replace_bond_array(bonds: BondList, array: Any) -> BondList:
    """Rebuild a bond list so its adjacency metadata agrees with its bonds."""
    return BondList(bonds.get_atom_count(), array)
