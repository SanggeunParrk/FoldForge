"""Cross-project format and released-operator regression checks."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from team_gm.modules.blocks.attention_math import attention, outer_product_projection

from foldforge.data.ccd import CCDDatabase
from foldforge.data.inference.build import load

REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE = Path(
    os.environ.get(
        "MINIWORLD_TEST_SOURCE", str(REPOSITORY.parent / "MiniWorld/src/miniworld")
    )
)
DATABASE = Path(
    os.environ.get(
        "FOLDFORGE_TEST_CCD_DB", str(REPOSITORY / "data/ccd/preprocessed_CCD.lmdb")
    )
)


@pytest.mark.parametrize("efficient", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("family", ["protenix", "opendde"])
def test_released_attention_equation(family, dtype, efficient):
    module = importlib.import_module(
        f"foldforge.models.{family}.ported.model.modules.primitives"
    )
    original = json.loads(
        (
            Path(__file__).parent / "reference_af_family/attention_oracles.json"
        ).read_text()
    )[family]
    tree = ast.parse(original)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_attention"
    )
    namespace = dict(module.__dict__)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            "<pre-migration attention>",
            "exec",
        ),
        namespace,
    )
    q = torch.randn(2, 3, 7, 8, dtype=dtype)
    k, v = torch.randn(2, 3, 11, 8, dtype=dtype), torch.randn(2, 3, 11, 8, dtype=dtype)
    bias = torch.randn(2, 3, 7, 11)
    expected = namespace["_attention"](q, k, v, bias, efficient)
    actual = module._attention(q, k, v, bias, efficient)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_attention_replacement_mask_and_backward():
    q, k, v = [torch.randn(2, 3, 4, 8, requires_grad=True) for _ in range(3)]
    mask = torch.tensor(
        [
            [
                [
                    [False, False, False, False],
                    [True, False, True, False],
                    [True] * 4,
                    [True] * 4,
                ]
            ]
        ]
    )
    bias = torch.randn(1, 3, 4, 4)
    expected = (
        ((q * (8**-0.5)) @ k.transpose(-1, -2) + bias)
        .masked_fill(~mask, -1e9)
        .softmax(-1)
    ) @ v
    actual = attention(
        q, k, v, bias=bias, mask=mask, scale=8**-0.5, logits_float32=False
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(
        torch.autograd.grad(actual.sum(), (q, k, v), retain_graph=True),
        torch.autograd.grad(expected.sum(), (q, k, v)),
        strict=True,
    ):
        torch.testing.assert_close(a, b)


def test_opm_projection_bias_and_leading_batch():
    a, b = torch.randn(2, 3, 5, 7, 4), torch.randn(2, 3, 6, 7, 4)
    proj = torch.nn.Linear(16, 8)
    expected = proj(torch.einsum("...isc,...jse->...ijce", a, b).flatten(-2))
    torch.testing.assert_close(outer_product_projection(a, b, project=proj), expected)


def test_native_linear_conversion_is_idempotent():
    from team_gm.modules.precision import NativeLinear

    layer = NativeLinear(4, 3)
    layer.compute_dtype = torch.float32
    twice = NativeLinear.from_linear(layer)
    assert twice.compute_dtype == torch.float32
    assert twice.weight is layer.weight


def test_miniworld_spec_fasta_homomer_and_ligand(tmp_path):
    (tmp_path / "a.fa").write_text(">a | polypeptide(L) | Chain:A\nACD\n")
    (tmp_path / "b.fa").write_text(">b | non-polymer | Chain:B\n(ATP)\n")
    data = {
        "chain_letters": {0: "A", 1: "A", 2: "B"},
        "fasta": {"A": "a.fa", "B": "b.fa"},
        "ccd_db": str(DATABASE),
        "save_trajectory": False,
    }
    p = tmp_path / "target.yaml"
    p.write_text(json.dumps(data))
    target = load(p)
    assert [c.id for c in target.chains] == ["A", "B", "C"]
    assert target.chains[0].ccds == target.chains[1].ccds
    assert target.af3()["sequences"][2]["ligand"]["ccdCodes"] == ["ATP"]
    assert target.af_family()[0]["sequences"][2]["ligand"]["ligand"] == "CCD_ATP"
    data["presicion"] = "bf16"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="presicion"):
        load(p)


@pytest.mark.skipif(
    not DATABASE.exists() or not REFERENCE.exists(),
    reason="external MiniWorld integration fixtures",
)
@pytest.mark.parametrize("key", ["ALA", "HEM", "ATP", "NAG", "ZN", "DA", "U"])
def test_real_miniworld_ccdmol_reads_new_records(key):
    spec = importlib.util.spec_from_file_location(
        "reference_ccd_mol", REFERENCE / "data/mols/ccd_mol.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    db = CCDDatabase(DATABASE)
    mol = reference.CCDMol.from_bytes(db.lookup.raw(key))
    ours = db.lookup[key]
    np.testing.assert_array_equal(mol.atoms.id.value, ours.atom_ids)
    np.testing.assert_array_equal(mol.atoms.element.value, ours.atom_elements)
    assert list(mol.chains.id.value) == [key]
    assert not set(ours.atom_elements) & {"H", "D"}
    # RDKit metadata is independent of canonical model coordinates.
    assert db.molecule(key).atom_map
    assert db.cif[key]["chem_comp"]["id"].as_item() == key
    assert db.af3_ccd()[key]["_chem_comp.id"] == [key]


def test_empty_heavy_atom_record():
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory

    from foldforge.data.ccd.build import record

    b = CIFBlock()
    b["chem_comp"] = CIFCategory({"id": ["H"], "name": ["hydrogen"], "formula": ["H"]})
    b["chem_comp_atom"] = CIFCategory({"atom_id": ["H"], "type_symbol": ["H"]})
    m = record("H", b, None)
    assert len(m.atoms) == 0
    assert list(m.chains.id.value) == ["H"]


def test_af3_opm_mask_normalization_and_bias():
    from team_gm.modules.blocks.attention_math import af3_outer_product_mean

    left, right = torch.randn(7, 5, 3), torch.randn(7, 5, 3)
    mask = (torch.rand(7, 5, 1) > 0.2).float()
    left, right = left * mask, right * mask
    weight, bias = torch.randn(3, 3, 8), torch.randn(8)
    expected = torch.einsum("sic,sje,cef->ijf", left, right, weight) + bias
    expected = expected / (0.001 + torch.einsum("sic,sjc->ijc", mask, mask))
    actual = af3_outer_product_mean(left, right, mask, weight, bias, eps=0.001)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not DATABASE.exists(), reason="prepared integration database")
def test_lmdb_fork_and_spawn_readers():
    import multiprocessing

    from foldforge.data.ccd import CCDLookup

    reader = CCDLookup(DATABASE)
    assert reader["ALA"].n_atoms == 6
    for method in ["fork", "spawn"]:
        with multiprocessing.get_context(method).Pool(1) as pool:
            assert pool.apply(_read_ala, (reader,)) == 6
    assert reader["ALA"].n_atoms == 6


def _read_ala(reader):
    # Pool transport also checks that no live LMDB handle is pickled.
    return reader["ALA"].n_atoms


@pytest.mark.skipif(not DATABASE.exists(), reason="prepared integration database")
def test_fragmentation_levels_and_missing_chemistry():
    from foldforge.data.ccd import CCDLookup

    reader = CCDLookup(DATABASE)
    levels = reader.fragments("ATP")
    assert len(levels[0].residues) == reader["ATP"].n_atoms
    assert len(levels[max(levels)].residues) == 1
    with pytest.raises(ValueError, match="fragmentation chemistry"):
        reader.fragments("UNL")


def test_common_msa_limit_is_applied_before_adapter(tmp_path):
    from foldforge.data.inference.build import limit_msa

    (tmp_path / "a.fa").write_text(">a | polypeptide(L) | Chain:A\nACD\n")
    (tmp_path / "a.a3m").write_text(">query\nACD\n>one\nA-D\n>two\nAC-\n")
    p = tmp_path / "x.yaml"
    p.write_text(
        json.dumps(
            {
                "chain_letters": {"0": "A"},
                "fasta": {"A": "a.fa"},
                "a3m": {"A": "a.a3m"},
                "ccd_db": str(DATABASE),
                "save_trajectory": False,
            }
        )
    )
    target = limit_msa(load(p), 2, tmp_path / "prepared")
    assert target.af3()["sequences"][0]["protein"]["unpairedMsa"].count(">") == 2
    path = target.af_family()[0]["sequences"][0]["proteinChain"]["unpairedMsaPath"]
    assert Path(path).read_text().count(">") == 2
    assert (tmp_path / "a.a3m").read_text().count(">") == 3


def test_af3_adapter_uses_installed_input_dialect(tmp_path):
    from alphafold3.common.folding_input import Input as AF3Input

    (tmp_path / "a.fa").write_text(">a | polypeptide(L) | Chain:A\nACD\n")
    p = tmp_path / "x.yaml"
    p.write_text(
        json.dumps(
            {
                "chain_letters": {0: "A"},
                "fasta": {"A": "a.fa"},
                "ccd_db": str(DATABASE),
                "save_trajectory": False,
            }
        )
    )
    native = load(p)
    parsed = AF3Input.from_json(
        json.dumps(native.af3()), json_path=tmp_path / "adapter.json"
    )
    assert len(parsed.chains) == 1
    assert parsed.chains[0].sequence == "ACD"


def test_serialization_is_readable_by_older_miniworld_biomol():
    from biotite.structure.io.pdbx import CIFBlock, CIFCategory
    from zstandard import ZstdDecompressor

    from foldforge.data.ccd.build import record

    block = CIFBlock()
    block["chem_comp"] = CIFCategory(
        {"id": ["ZN"], "name": ["ZINC ION"], "formula": ["Zn"]}
    )
    block["chem_comp_atom"] = CIFCategory({"atom_id": ["ZN"], "type_symbol": ["ZN"]})
    mol = record("ZN", block, None)
    # Older BioMol calls decompress() without max_output_size, rather than a
    # streaming reader. Test that actual historical API, not only a round trip.
    payload = ZstdDecompressor().decompress(mol.to_bytes())
    header_len = int.from_bytes(payload[:8], "little")
    header = json.loads(payload[8 : 8 + header_len])
    assert "atoms" in header["template"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="engine batch regression")
def test_esm_trunk_two_samples_match_individual_kernel_calls():
    from team_gm.modules.exceptions import ImplementationType

    from foldforge.models.esmfold2.precision import inference_precision
    from foldforge.models.esmfold2.trunk import FoldingTrunk

    trunk = FoldingTrunk(
        FoldingTrunk.Config(
            d_pair=32, n_block=1, implementation=ImplementationType.MINIWORLD_ENGINE
        )
    )
    inference_precision(trunk, torch.device("cuda"), torch.bfloat16)
    x = torch.randn(2, 8, 8, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    mask[1, -2:] = False
    with torch.inference_mode():
        expected = torch.cat([trunk(x[i : i + 1], mask[i : i + 1]) for i in range(2)])
        actual = trunk(x, mask)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)


@pytest.mark.parametrize("family", ["protenix", "opendde"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triangle_attention_bias_order_matches_frozen_source(family, dtype):
    module = importlib.import_module(
        f"foldforge.models.{family}.ported.model.triangular.layers"
    )
    body = json.loads(
        (
            Path(__file__).parent / "reference_af_family/attention_oracles.json"
        ).read_text()
    )[family + "_triangle"]
    namespace = dict(module.__dict__)
    exec(compile(body, "frozen triangle attention", "exec"), namespace)
    q, k, v = [torch.randn(2, 3, 5, 8, dtype=dtype) for _ in range(3)]
    biases = [torch.randn(2, 1, 1, 5), torch.randn(1, 3, 5, 5)]
    expected = namespace["_attention"](q, k, v, biases)
    actual = module._attention(q, k, v, biases)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
