"""Bond orders for the token bond matrix, which AF3's featurisation drops.

AF3 publishes each ligand-ligand bond as a pair of token indices and nothing
else. A family that embeds the bond ORDER -- Boltz-2's token bond types --
then has to be handed one, and writing every bond as single turns benzene into
cyclohexane: 3PTB's benzamidine came out with 1.52 A ring bonds against the
release's 1.39 A. The orders are read as the release reads them: from the CCD
component's RDKit molecule after sanitisation, so aromaticity is RDKit's
perception and not the CCD's own flag.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Codes before the +1 that frees zero for "no bond"; Boltz-2's own numbering.
OTHER, SINGLE, DOUBLE, TRIPLE, AROMATIC, COVALENT = range(6)
_RDKIT = {"SINGLE": SINGLE, "DOUBLE": DOUBLE, "TRIPLE": TRIPLE, "AROMATIC": AROMATIC}
#: Example key, one entry per row of the ligand-ligand bond gather.
KEY = "ligand_ligand_bond_order"


def _component_orders(ccd: Any, name: str) -> dict[frozenset[str], int]:
    from alphafold3.data.tools import rdkit_utils  # noqa: PLC0415 - optional dep
    from rdkit import Chem  # noqa: PLC0415 - optional dep

    cif = ccd.get(name)
    if cif is None:
        return {}
    try:
        mol = rdkit_utils.mol_from_ccd_cif(cif, remove_hydrogens=False)
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - an unsanitisable component keeps single bonds
        return {}
    orders = {}
    for bond in mol.GetBonds():
        pair = frozenset(
            mol.GetAtomWithIdx(i).GetProp("atom_name")
            for i in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        )
        orders[pair] = _RDKIT.get(bond.GetBondType().name, OTHER)
    return orders


@contextlib.contextmanager
def capture(ccd: Any) -> Iterator[list[np.ndarray]]:
    """Record each example's ligand-ligand bond orders while it is featurised.

    Yields a list that fills with one int32 array per example, aligned with the
    rows of ``tokens_to_ligand_ligand_bonds``. A bond inside one residue takes
    the component's order; a bond between residues is a covalent link.
    """
    from alphafold3.model import features  # noqa: PLC0415 - optional dep

    captured: list[np.ndarray] = []
    original = features.LigandLigandBondInfo.compute_features.__func__
    cache: dict[str, dict[frozenset[str], int]] = {}

    def compute_features(
        cls: Any, all_tokens: Any, bond_layout: Any, padding_shapes: Any
    ) -> Any:
        info = original(cls, all_tokens, bond_layout, padding_shapes)
        gather = info.tokens_to_ligand_ligand_bonds
        index = np.asarray(gather.gather_idxs)
        valid = np.asarray(gather.gather_mask).all(axis=-1)
        orders = np.zeros(index.shape[0], dtype=np.int32)
        for row in np.flatnonzero(valid):
            a, b = index[row]
            if (all_tokens.chain_id[a], all_tokens.res_id[a]) != (
                all_tokens.chain_id[b],
                all_tokens.res_id[b],
            ):
                orders[row] = COVALENT
                continue
            name = str(all_tokens.res_name[a])
            if name not in cache:
                cache[name] = _component_orders(ccd, name)
            pair = frozenset(
                (str(all_tokens.atom_name[a]), str(all_tokens.atom_name[b]))
            )
            orders[row] = cache[name].get(pair, SINGLE)
        captured.append(orders)
        return info

    features.LigandLigandBondInfo.compute_features = classmethod(compute_features)
    try:
        yield captured
    finally:
        features.LigandLigandBondInfo.compute_features = classmethod(original)
