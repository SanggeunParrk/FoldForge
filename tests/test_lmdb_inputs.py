"""MiniWorld LMDB data roundtrips through released model featurizers."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import lmdb
import numpy as np
import pytest
import yaml
from biomol.core import FeatureContainer, IndexTable, NodeFeature
from biomol.core.utils import to_bytes

from foldforge.data.inference.build import limit_msa, load
from foldforge.data.inference.lmdb import (
    Alignment,
    load_templates,
    species_aliases,
    template_features,
)
from foldforge.data.inference.template_mol import TemplateMol


CCD_DATABASE = Path(
    os.environ.get(
        "FOLDFORGE_TEST_CCD_DB",
        str(Path(__file__).resolve().parents[1] / "data/ccd/preprocessed_CCD.lmdb"),
    )
)


def write_db(path, key, value, subdir=True):
    with lmdb.open(str(path), map_size=32 * 1024 * 1024, subdir=subdir) as env:
        with env.begin(write=True) as txn:
            txn.put(key.encode(), to_bytes(value))


def msa_record(sequence="ARNDCQEG", kind="protein"):
    from foldforge.data.inference.lmdb import ALPHABETS

    mapping = {v: k for k, v in ALPHABETS[kind].items()}
    query = np.array([mapping[x] for x in sequence], dtype=np.uint8)
    tokens = np.tile(query, (4, 1))
    tokens[1, 2] = 31
    tokens[2, 3] = 31
    tokens[3, 4] = 31
    deletions = np.zeros_like(tokens)
    deletions[1, 2] = 3
    deletions[2, 3] = 2
    deletions[3, 4] = 1
    return {
        "msa_dict": {
            "sequences": {
                "query_sequence": np.array(list(sequence)),
                "aligned_sequences": tokens,
                "deletions": deletions,
                "deletion_mean": deletions.mean(0),
                "profile": np.eye(32)[tokens].mean(0),
            },
            "headers": {
                "species": np.array(
                    ["query", "long species/1", "long species/2", "N/A"]
                )
            },
        }
    }


def template_record(sequence="ARNDCQEG", xyz=None):
    length = len(sequence)
    if xyz is None:
        xyz = np.arange(length * 12, dtype=np.float32).reshape(length, 4, 3) / 3
        xyz[2, 3] = np.nan
    return TemplateMol(
        FeatureContainer(
            {
                "id": NodeFeature(np.tile(["N", "CA", "C", "CB"], length)),
                "xyz": NodeFeature(xyz.reshape(-1, 3)),
            }
        ),
        FeatureContainer(
            {"one_letter_code_can": NodeFeature(np.array(list(sequence)))}
        ),
        FeatureContainer({"chain_id": NodeFeature(np.array(["A"]))}),
        IndexTable.from_parents(
            np.repeat(np.arange(length), 4), np.zeros(length, dtype=np.int64)
        ),
        {"id": "test", "assembly_id": "1", "model_id": 1, "alt_id": "."},
    ).to_dict()


@pytest.fixture
def target_files(tmp_path):
    (tmp_path / "A.fasta").write_text("> test | polypeptide(L) | Chain:A\nARNDCQEG\n")
    write_db(tmp_path / "msa.lmdb", "seq-id", msa_record())
    write_db(
        tmp_path / "template.lmdb",
        "pdb_A",
        {"template_mols": {"hit": template_record()}},
    )
    spec = {
        "name": "test",
        "chain_letters": {"0": "A"},
        "fasta": {"A": "A.fasta"},
        "ccd_db": str(CCD_DATABASE),
        "msa_db": {"A": "msa.lmdb"},
        "msa": {"A": "seq-id"},
        "template_db": "template.lmdb",
        "template": {"0": "pdb_A"},
        "save_trajectory": False,
    }
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path, spec


@pytest.mark.parametrize("subdir", [True, False])
def test_msa_native_roundtrip(tmp_path, subdir):
    from alphafold3.constants import mmcif_names
    from alphafold3.data import msa, msa_features

    path = tmp_path / "msa.lmdb"
    record = msa_record()
    write_db(path, "key", record, subdir)
    alignment = Alignment.load(path, "key", "protein", "ARNDCQEG")
    aliases = species_aliases([alignment])
    assert "N/A" not in aliases
    a3m = alignment.a3m(aliases)
    native = msa.Msa.from_a3m(
        query_sequence="ARNDCQEG",
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        a3m=a3m,
        deduplicate=False,
    )
    features = native.featurize()
    expected = alignment.tokens.copy()
    expected[expected == 31] = 21
    np.testing.assert_array_equal(features["msa"], expected)
    np.testing.assert_array_equal(features["deletion_matrix_int"], alignment.deletions)
    species = msa_features.extract_species_ids(native.descriptions)
    assert species == ["", aliases["long species/1"], aliases["long species/2"], ""]
    for family in ["protenix", "opendde"]:
        pairing = importlib.import_module(
            f"foldforge.models.{family}.ported.data.msa.msa_utils"
        ).MSAPairingEngine
        assert pairing.get_species_ids(native.descriptions) == species


@pytest.mark.parametrize("kind,seq", [("rna", "AUGCNAUG"), ("dna", "ATGCNATG")])
def test_nucleotide_alphabets(tmp_path, kind, seq):
    path = tmp_path / "msa"
    write_db(path, "seq", msa_record(seq, kind))
    result = Alignment.load(path, "seq", kind, seq)
    assert result.a3m(species_aliases([result])).splitlines()[1] == seq


def test_common_adapters(target_files, tmp_path):
    path, spec = target_files
    target = load(path)
    assert len(target.chains[0].templates) == 1
    af3 = target.af3()["sequences"][0]["protein"]
    family = target.af_family()[0]["sequences"][0]["proteinChain"]
    assert af3["unpairedMsa"] == family["unpairedMsa"] == family["pairedMsa"]
    assert len(af3["templates"]) == len(family["miniworldTemplates"]) == 1
    limited = limit_msa(target, 2, tmp_path / "prepared")
    assert limited.msa_text(limited.chains[0]).count(">") == 2
    assert limited.msa_species == target.msa_species
    assert (
        limited.msa_text(limited.chains[0]).splitlines()
        == target.msa_text(target.chains[0]).splitlines()[:4]
    )
    np.testing.assert_array_equal(
        limited.chains[0].alignment.deletions, target.chains[0].alignment.deletions[:2]
    )
    with pytest.raises(ValueError, match="no template conditioning"):
        target.esmfold2(4)
    spec["template"] = {}
    path.write_text(yaml.safe_dump(spec))
    esm = load(path).esmfold2(3)
    assert esm.sequences[0].msa.depth == 3


@pytest.mark.parametrize("family", ["protenix", "opendde"])
def test_template_coordinate_masks(target_files, family):
    path, _ = target_files
    target = load(path)
    template = target.chains[0].templates[0]
    features = template_features([template.payload()], "ARNDCQEG", family)[0]
    utils = importlib.import_module(
        f"foldforge.models.{family}.ported.data.template.template_utils"
    )
    indices = [utils.ATOM37_ORDER[x] for x in ["N", "CA", "C", "CB"]]
    np.testing.assert_array_equal(
        features["template_all_atom_positions"][:, indices], template.positions
    )
    np.testing.assert_array_equal(
        features["template_all_atom_masks"][:, indices], template.mask
    )
    assert features["template_all_atom_masks"][2, utils.ATOM37_ORDER["CB"]] == 0
    assert features["template_all_atom_masks"][:, utils.ATOM37_ORDER["O"]].sum() == 0


def test_af3_template_mmcif(target_files):
    from alphafold3 import structure

    path, _ = target_files
    target = load(path)
    t = target.chains[0].templates[0]
    payload = t.af3()
    parsed = structure.from_mmcif(payload["mmcif"])
    assert len(parsed.atom_x) == int(t.mask.sum())
    np.testing.assert_allclose(
        np.stack([parsed.atom_x, parsed.atom_y, parsed.atom_z], -1),
        t.positions[t.mask],
        atol=1e-5,
    )
    assert payload["queryIndices"] == list(range(8))


@pytest.mark.parametrize(
    "change,error",
    [
        ({"msa": {"A": "missing"}}, "not found"),
        ({"msa_db": {}}, "same chain letters"),
        ({"a3m": {"A": "other.a3m"}}, "Choose a3m"),
        ({"template_db": None}, "requires template_db"),
        ({"template": {"2": "pdb_A"}}, "unknown chain index"),
        ({"template_n": 5}, "less than or equal"),
    ],
)
def test_input_validation(target_files, change, error):
    path, spec = target_files
    spec.update(change)
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises((ValueError, KeyError), match=error):
        load(path)


def test_query_mismatch_and_template_length(target_files):
    path, _ = target_files
    with pytest.raises(ValueError, match="does not match"):
        Alignment.load(path.parent / "msa.lmdb", "seq-id", "protein", "ARNDCQEA")
    with pytest.raises(ValueError, match="length differs"):
        load_templates(path.parent / "template.lmdb", "pdb_A", 9, 4)


def test_zero_templates_does_not_open_db(tmp_path):
    assert load_templates(tmp_path / "absent", "missing", 8, 0) == ()


def test_structcooker_blank_gaps_and_ca_fallback(tmp_path):
    item = template_record()
    ids = item["atoms"]["nodes"]["id"]["value"].reshape(8, 4)
    xyz = item["atoms"]["nodes"]["xyz"]["value"].reshape(8, 4, 3)
    letters = item["residues"]["nodes"]["one_letter_code_can"]["value"]
    letters[1] = ""
    ids[1] = ""
    xyz[1] = np.nan
    ids[3, 3] = "CA"
    xyz[3, 3] = xyz[3, 1]
    write_db(tmp_path / "template", "gap", {"template_mols": {"hit": item}})
    t = load_templates(tmp_path / "template", "gap", 8, 4)[0]
    assert t.sequence == "A-NDCQEG"
    assert not t.mask[1].any()
    assert not t.mask[3, 3] and t.mask[3, 1]
    payload = t.af3()
    assert payload["queryIndices"] == [0, 2, 3, 4, 5, 6, 7]
    assert payload["templateIndices"] == list(range(7))
    from alphafold3 import structure

    parsed = structure.from_mmcif(payload["mmcif"])
    assert len(parsed.atom_x) == int(t.mask.sum())


@pytest.mark.parametrize("family", ["protenix", "opendde"])
@pytest.mark.skipif(
    not CCD_DATABASE.exists(), reason="prepared CCD integration database"
)
def test_dataset_consumes_lmdb_templates(target_files, tmp_path, family):
    from foldforge.data.ccd import CCDDatabase

    path, _ = target_files
    target = load(path)
    native = tmp_path / "adapter.json"
    native.write_text(json.dumps(target.af_family()))
    model = importlib.import_module(f"foldforge.models.{family}.model")
    api = importlib.import_module(
        f"foldforge.models.{family}.ported.data.inference.infer_dataloader"
    )
    cfg = model.configuration()
    cfg.input_json_path = str(native)
    cfg.dump_dir = str(tmp_path / "output")
    cfg.use_template = True
    cfg.use_msa = True
    cfg.num_workers = 0
    cfg.data.ccd_components_file = str(target.spec.ccd_db)
    cfg.data.ccd_components_rdkit_mol_file = str(target.spec.ccd_db)
    with CCDDatabase(target.spec.ccd_db).activate():
        dataset = api.InferenceDataset(cfg)
        assert dataset.online_template_featurizer is None
        data, atoms, error = dataset[0]
        assert not error, error
        features = data["input_feature_dict"]
        assert features["template_pseudo_beta_mask"].sum() > 0
        assert features["template_backbone_frame_mask"].sum() > 0
        assert features["msa"].shape[0] > 1


def test_species_aliases_are_shared_across_chain_order(tmp_path):
    from dataclasses import replace

    record = msa_record()
    write_db(tmp_path / "msa", "key", record)
    a = Alignment.load(tmp_path / "msa", "key", "protein", "ARNDCQEG")
    b = replace(a, species=("query", "long species/2", "long species/1", "N/A"))
    aliases = species_aliases([a, b])
    assert aliases == species_aliases([b, a])
    from alphafold3.data.msa_features import extract_species_ids

    sa = extract_species_ids([x[1:] for x in a.a3m(aliases).splitlines()[::2]])
    sb = extract_species_ids([x[1:] for x in b.a3m(aliases).splitlines()[::2]])
    assert sa[1] == sb[2] and sa[2] == sb[1] and sa[1] != sa[2]
    assert sa[0] == sa[3] == ""


@pytest.mark.parametrize("family", ["protenix", "opendde"])
@pytest.mark.skipif(
    not CCD_DATABASE.exists(), reason="prepared CCD integration database"
)
def test_rna_lmdb_is_not_disabled_by_default(target_files, tmp_path, family):
    from foldforge.data.ccd import CCDDatabase

    path, spec = target_files
    (tmp_path / "A.fasta").write_text(
        "> rna | polyribonucleotide | Chain:A\nAUGCAUGC\n"
    )
    write_db(tmp_path / "rna.lmdb", "rna", msa_record("AUGCAUGC", "rna"))
    spec.update(msa_db={"A": "rna.lmdb"}, msa={"A": "rna"}, template={})
    path.write_text(yaml.safe_dump(spec))
    target = load(path)
    native = tmp_path / "rna.json"
    native.write_text(json.dumps(target.af_family()))
    model = importlib.import_module(f"foldforge.models.{family}.model")
    api = importlib.import_module(
        f"foldforge.models.{family}.ported.data.inference.infer_dataloader"
    )
    cfg = model.configuration()
    cfg.input_json_path = str(native)
    cfg.dump_dir = str(tmp_path / "output")
    cfg.use_template = False
    cfg.use_msa = True
    cfg.use_rna_msa = False
    cfg.data.ccd_components_file = str(target.spec.ccd_db)
    cfg.data.ccd_components_rdkit_mol_file = str(target.spec.ccd_db)
    with CCDDatabase(target.spec.ccd_db).activate():
        dataset = api.InferenceDataset(cfg)
        assert dataset.use_rna_msa
        data, atoms, error = dataset[0]
        assert not error, error
        assert data["input_feature_dict"]["msa"].shape[0] > 1


def test_af3_native_template_feature_alignment(tmp_path):
    from alphafold3 import structure
    from alphafold3.constants import mmcif_names, atom_types
    from alphafold3.data.templates import get_polymer_features

    item = template_record()
    xyz = item["atoms"]["nodes"]["xyz"]["value"].reshape(8, 4, 3)
    ids = item["atoms"]["nodes"]["id"]["value"].reshape(8, 4)
    item["residues"]["nodes"]["one_letter_code_can"]["value"][1] = ""
    xyz[1] = np.nan
    ids[1] = ""
    write_db(tmp_path / "template", "key", {"template_mols": {"aligned": item}})
    template = load_templates(tmp_path / "template", "key", 8, 4)[0]
    payload = template.af3()
    features = get_polymer_features(
        chain=structure.from_mmcif(payload["mmcif"]),
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        query_sequence_length=8,
        query_to_hit_mapping=dict(
            zip(payload["queryIndices"], payload["templateIndices"])
        ),
    )
    order = [atom_types.ATOM37_ORDER[x] for x in ["N", "CA", "C", "CB"]]
    np.testing.assert_allclose(
        features["template_all_atom_positions"][:, order], template.positions, atol=1e-5
    )
    np.testing.assert_array_equal(
        features["template_all_atom_masks"][:, order], template.mask
    )
