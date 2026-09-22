"""Shared chemistry processing must preserve both supported token layouts."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

import foldforge
from foldforge.data.constants.chemistry import PROTEIN_CHAIN
from foldforge.data.features.tokenizer import (
    ResidueAtomArrayTokenizer,
    StructuralAtomArrayTokenizer,
    Token,
    TokenArray,
)
from foldforge.data.msa.alignment import MSAPairingEngine, RawMsa
from foldforge.data.msa.kalign import resolve_kalign_binary
from foldforge.eval.clash import Clash

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


def test_residue_and_structural_tokens_preserve_atom_partition_and_twins():
    atoms = AtomArray(11)
    atoms.atom_name = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "C1", "O1"]
    atoms.element = ["N", "C", "C", "O", "C", "N", "C", "C", "O", "C", "O"]
    atoms.res_name = ["ALA"] * 5 + ["GLY"] * 4 + ["LIG"] * 2
    atoms.res_id = [1] * 5 + [2] * 4 + [3] * 2
    atoms.chain_id = ["A"] * 9 + ["B"] * 2
    atoms.set_annotation("mol_type", np.array(["protein"] * 9 + ["ligand"] * 2))
    atoms.set_annotation("centre_atom_mask", np.isin(np.arange(11), [1, 6, 9, 10]))
    residue = ResidueAtomArrayTokenizer(atoms).get_token_array()
    structural = StructuralAtomArrayTokenizer(atoms).get_structural_token_array()
    assert isinstance(residue, TokenArray)
    assert isinstance(structural, TokenArray)
    assert all(isinstance(token, Token) for token in [*residue, *structural])
    assert [t.atom_indices for t in residue] == [
        list(range(5)),
        list(range(5, 9)),
        [9],
        [10],
    ]
    assert [t.atom_indices for t in structural] == [
        [0, 1, 2, 3],
        [4],
        [5, 6, 7, 8],
        [9],
        [10],
    ]
    assert structural.get_annotation("parent_residue_idx") == [0, 0, 1, 2, 3]
    assert structural.get_annotation("twin_token_idx") == [1, 0, -1, -1, -1]
    for tokens in (residue, structural):
        assert sorted(i for token in tokens for i in token.atom_indices) == list(
            range(11)
        )


def test_shared_msa_preserves_rows_insertions_and_species():
    raw = RawMsa.from_a3m(
        "ARN",
        PROTEIN_CHAIN,
        ">query\nARN\n>sp|P12345|ABC_HUMAN\nARdN\n>sp|Q12345|ABC_MOUSE\nA-N\n",
        dedup=False,
    )
    features = raw.featurize()
    assert raw.depth == 3
    np.testing.assert_array_equal(
        features["deletion_matrix"], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    )
    assert features["msa_species_identifiers"].tolist() == ["", "HUMAN", "MOUSE"]
    np.testing.assert_array_equal(features["msa"][0], features["msa"][1])
    assert MSAPairingEngine().get_species_ids(raw.descs) == ["", "HUMAN", "MOUSE"]
    assert RawMsa.merge([raw, raw], deduplicate=False).depth == 6


@pytest.mark.parametrize("details_dtype", [torch.bool, torch.float32])
def test_shared_clash_preserves_requested_detail_dtype(details_dtype):
    result = Clash(compute_vdw_clash=False, details_dtype=details_dtype)(
        pred_coordinate=torch.zeros(1, 4, 3),
        asym_id=torch.tensor([0, 0, 1, 1]),
        atom_to_token_idx=torch.arange(4),
        is_ligand=torch.zeros(4),
        is_protein=torch.ones(4),
        is_dna=torch.zeros(4),
        is_rna=torch.zeros(4),
    )
    details = result["details"]["af3_clash"]
    assert details.dtype == details_dtype
    assert details.shape == (1, 2, 2, 2)
    assert details[0, 0, 1, 0] > 0
    if details_dtype == torch.float32:
        assert details[0, 0, 1, 0] > 1


def test_alignment_binary_respects_explicit_configuration(tmp_path, monkeypatch):
    default = tmp_path / "kalign"
    explicit = tmp_path / "configured-kalign"
    for path in (default, explicit):
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert resolve_kalign_binary(str(explicit)) == str(explicit)
    assert resolve_kalign_binary(None) == str(default)


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
