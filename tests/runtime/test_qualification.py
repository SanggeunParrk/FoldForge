"""Frozen layout equations, complex chemistry and actual runtime execution."""

from __future__ import annotations

import json
import math
from typing import Any, Union

import pytest
import torch
from support import REFERENCE_ROOT, REPOSITORY_ROOT
from team_gm.modules import tensor_layout as layout
from torch import nn

from foldforge.models.config import ExecutionConfig
from foldforge.models.execution import ExecutedCallable


def oracle():
    ns = {"torch": torch, "nn": nn, "math": math, "Any": Any, "Union": Union}
    for code in json.loads(
        (REFERENCE_ROOT / "layout_oracles.json").read_text()
    ).values():
        exec(compile(code, "<frozen original layout>", "exec"), ns)  # noqa: S102 - checked-in oracle
    return ns


@pytest.mark.parametrize("length", [31, 32, 33, 76])
def test_window_layout_against_frozen_equation(length):
    q = torch.randn(2, 3, length, 8)
    k = torch.randn(2, 3, length, 8)
    actual = layout.rearrange_qk_to_dense_trunk(q, k, -2, -2, 32, 128)
    expected = oracle()["rearrange_qk_to_dense_trunk"](q, k, -2, -2, 32, 128)
    for a, b in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in actual[2]:
        if isinstance(actual[2][key], torch.Tensor):
            torch.testing.assert_close(actual[2][key], expected[2][key])
        else:
            assert actual[2][key] == expected[2][key]


def test_batched_atom_gather_gradients():
    x = torch.randn(2, 7, 5, requires_grad=True)
    index = torch.tensor([[0, 3, 3, 6], [2, 0, 5, 1]])
    a = layout.broadcast_token_to_atom(x, index)
    b = oracle()["broadcast_token_to_atom"](x, index)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(a.sum(), x)[0],
        torch.autograd.grad(b.sum(), x)[0],
        atol=0,
        rtol=0,
    )


def test_ccd_ligand_id_is_one_code(tmp_path):
    import yaml

    from foldforge.data.inputs.build import load

    (tmp_path / "L.fasta").write_text("> ligand | non-polymer | Chain:L\n(BEN)\n")
    path = tmp_path / "in.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "chain_letters": {"0": "L"},
                "fasta": {"L": "L.fasta"},
                "ccd_db": "unused",
                "save_trajectory": False,
            }
        )
    )
    target = load(path).esmfold2(1)
    assert target.sequences[0].ccd == ["BEN"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph runtime")
def test_graph_replay_updates_inputs_and_owns_outputs():
    cfg = ExecutionConfig(cuda_graph=True)
    fn = ExecutedCallable(lambda x, scale: x.sin() * scale, cfg, "test")
    x = torch.randn(3, 7, device="cuda")
    with torch.no_grad():
        a = fn(x, scale=2.0)
        b = fn(x + 1, scale=2.0)
        torch.testing.assert_close(a, x.sin() * 2)
        torch.testing.assert_close(b, (x + 1).sin() * 2)
    assert fn.captures == 1
    assert fn.replays == 2
    with pytest.raises(RuntimeError, match="no_grad"):
        fn(x, scale=2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime")
def test_compile_is_really_invoked_and_matches():
    from torch._dynamo.utils import counters

    before = counters["stats"]["unique_graphs"]
    fn = ExecutedCallable(
        lambda x: x.sin() + x.square(), ExecutionConfig(compile=True), "compiled test"
    )
    x = torch.randn(4, 8, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(fn(x), x.sin() + x.square())
    assert counters["stats"]["unique_graphs"] > before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph runtime")
def test_graph_dataclass_metadata_and_capture_limit():
    from dataclasses import dataclass

    @dataclass
    class Inputs:
        data: torch.Tensor
        shape: torch.Tensor

    fn = ExecutedCallable(
        lambda v: v.data.reshape(tuple(v.shape.tolist())).square(),
        ExecutionConfig(cuda_graph=True, max_graphs=1),
        "metadata",
    )
    x = torch.arange(6, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        a = fn(Inputs(x, torch.tensor([2, 3])))
        b = fn(Inputs(x + 2, torch.tensor([2, 3])))
        torch.testing.assert_close(a, x.reshape(2, 3).square())
        torch.testing.assert_close(b, (x + 2).reshape(2, 3).square())
        with pytest.raises(RuntimeError, match="max_graphs"):
            fn(Inputs(x, torch.tensor([3, 2])))
        with pytest.raises(ValueError, match="integer CPU"):
            fn(Inputs(x, torch.tensor([2.0, 3.0])))


def test_full_model_graph_rejects_host_sampling():
    with pytest.raises(ValueError, match="denoiser"):
        ExecutionConfig(cuda_graph=True, scope="model")


def test_af3_atom_layout_shape_contract():
    from foldforge.modules.dense.atom_layout import GatherInfo, convert

    info = GatherInfo(torch.tensor([0, 3]), torch.tensor([True, False]), (2, 2))
    x = torch.arange(4.0).reshape(2, 2)
    torch.testing.assert_close(
        convert(info, x, layout_axes=(0, 1)), torch.tensor([0.0, 0.0])
    )
    restored = GatherInfo.from_dict(info.as_dict())
    assert restored.input_shape == (2, 2)
    with pytest.raises(ValueError, match="incompatible"):
        convert(info, torch.zeros(2, 3), layout_axes=(0, 1))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_atom_mean_mask_and_gradients(dtype):
    x = torch.randn(2, 7, 5, dtype=dtype, requires_grad=True)
    index = torch.tensor([[0, 0, 1, 1, 1, 3, 3], [0, 0, 0, 1, 1, 3, 3]])
    mask = torch.tensor([[True, False, True, True, False, True, True]]).expand(2, -1)
    actual = layout.aggregate_atom_to_token(x, index, 5, atom_mask=mask)
    reference = torch.stack(
        [
            torch.stack(
                [
                    x[b, (index[b] == token) & mask[b]].double().mean(0)
                    if ((index[b] == token) & mask[b]).any()
                    else x[b].double().sum(0) * 0
                    for token in range(5)
                ]
            )
            for b in range(2)
        ]
    ).to(dtype)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(reference.sum(), x)[0],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction")
def test_bf16_atom_mean_repeats_against_fp64():
    x = torch.randn(1, 4096, 128, device="cuda", dtype=torch.bfloat16)
    index = torch.arange(128, device="cuda").repeat_interleave(32)
    ref = x.double().reshape(1, 128, 32, 128).mean(-2).to(x.dtype)
    with torch.no_grad():
        fn = ExecutedCallable(
            lambda a: layout.aggregate_atom_to_token(a, index, 128),
            ExecutionConfig(cuda_graph=True),
            "atom mean",
        )
        for _ in range(5):
            torch.testing.assert_close(fn(x), ref, atol=0, rtol=0)


@pytest.mark.parametrize("family", ["protenix", "opendde"])
def test_shared_layout_imports_are_live(family):
    import importlib

    records = json.loads(
        (REPOSITORY_ROOT / "docs/archive/shared-layout-migration.json").read_text()
    )
    for relative, name in records:
        relative.removesuffix(".py").replace("/", ".")
        actual = importlib.import_module("team_gm.modules.tensor_layout")
        assert getattr(actual, name) is getattr(layout, name), (family, name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph RNG")
def test_graph_setup_does_not_consume_sampler_rng():
    x = torch.ones(4, 8, device="cuda")
    cfg = ExecutionConfig(cuda_graph=True)
    fn = ExecutedCallable(lambda a: a + torch.rand_like(a), cfg, "random")
    with torch.no_grad():
        torch.manual_seed(71)
        expected = x + torch.rand_like(x)
        next_expected = torch.rand_like(x)
        torch.manual_seed(71)
        result = fn(x)
        next_actual = torch.rand_like(x)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(next_actual, next_expected, atol=0, rtol=0)
