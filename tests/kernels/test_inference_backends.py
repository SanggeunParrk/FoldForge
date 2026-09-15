"""Backend selection must reach the executed operator, with one residual."""

import copy
import importlib
from unittest import mock

import pytest
import torch
from team_gm.modules.checkpoints.af_family import configure_model, triangle_residual
from team_gm.modules.exceptions import ImplementationType

from foldforge.models.config import Config
from foldforge.modules.sequence.atom_transformer import SWAAtomTransformer


@pytest.mark.parametrize("backend", ["pytorch", "cuequivariance", "miniworld"])
def test_backend_reaches_atom_attention(backend):
    impl = (
        ImplementationType.MINIWORLD_ENGINE
        if backend == "miniworld"
        else ImplementationType(backend)
    )
    assert Config(backend=backend).backend == backend
    model = SWAAtomTransformer(
        SWAAtomTransformer.Config(implementation=impl, n_block=1)
    )
    expected = "miniworld" if backend == "miniworld" else "pytorch"
    assert model.blocks[0].attn.implementation.value == expected


@pytest.mark.parametrize("family", ["af3", "protenix", "opendde"])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vendor CUDA kernels")
def test_cueq_triangle_direction_mask_and_residual(family, outgoing):
    from cuequivariance_torch import triangle_multiplicative_update

    torch.manual_seed(42)
    if family == "af3":
        cls = importlib.import_module(
            "foldforge.modules.dense.triangle_multiplication"
        ).TriangleMultiplication
        layer = cls(128, _outgoing=outgoing)
    else:
        mod = importlib.import_module("team_gm.modules.checkpoints.triangle")
        cls = (
            mod.TriangleMultiplicationOutgoing
            if outgoing
            else mod.TriangleMultiplicationIncoming
        )
        layer = cls(c_z=128, c_hidden=128)
    with torch.no_grad():
        for p in layer.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.08)
    ref = configure_model(copy.deepcopy(layer), "pytorch")
    layer = configure_model(layer, "cuequivariance")
    x = torch.randn(32, 32, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(32, 32, device="cuda") > 0.2
    with (
        torch.inference_mode(),
        mock.patch(
            "cuequivariance_torch.triangle_multiplicative_update",
            wraps=triangle_multiplicative_update,
        ) as vendor,
    ):
        expected = triangle_residual(ref, x.clone(), mask)
        actual = triangle_residual(layer, x.clone(), mask)
    assert vendor.call_count == 1
    assert (actual - expected).float().square().mean().sqrt() < 0.007


@pytest.mark.parametrize("family", ["protenix", "opendde"])
@pytest.mark.parametrize("length", [8, 32, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vendor CUDA kernels")
def test_vendor_attention_preserves_batches_and_small_length_scaling(family, length):  # noqa: ARG001 - shared callback or fixture signature
    mod = importlib.import_module("team_gm.modules.checkpoints.layers")
    layer = (
        mod.Attention(c_q=128, c_k=128, c_v=128, c_hidden=32, no_heads=4)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    with torch.no_grad():
        layer.linear_o.weight.normal_(0, 0.04)
    x = torch.randn(2, length, length, 128, device="cuda", dtype=torch.bfloat16)
    biases = [
        torch.zeros(2, length, 1, 1, length, device="cuda"),
        torch.randn(2, 1, 4, length, length, device="cuda"),
    ]
    with torch.inference_mode():
        ref = layer(x, x, biases, triangle_attention="torch")
        out = layer(x, x, biases, triangle_attention="cuequivariance")
    assert out.shape == ref.shape
    assert (out - ref).float().square().mean().sqrt() < 0.02


@pytest.mark.parametrize("ending", [False, True])
@pytest.mark.parametrize("length", [32, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vendor CUDA kernels")
def test_af3_grid_vendor_preserves_projection_and_mask(ending, length):
    from foldforge.modules.dense.attention import GridSelfAttention

    torch.manual_seed(61)
    layer = GridSelfAttention(c_pair=128, num_head=4, transpose=ending)
    reference = configure_model(copy.deepcopy(layer), "pytorch")
    vendor = configure_model(layer, "cuequivariance")
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(length, length, device="cuda") > 0.2
    with torch.inference_mode():
        expected = reference(pair, mask)
        actual = vendor(pair, mask)
    assert actual.shape == expected.shape
    assert (actual - expected).float().square().mean().sqrt() < 0.02


@pytest.mark.parametrize("family", ["protenix", "opendde"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vendor CUDA kernels")
def test_pairformer_adapter_vendor_keeps_shared_bias_with_chunk_option(family):  # noqa: ARG001 - shared callback or fixture signature
    from team_gm.modules.checkpoints.pairformer import (
        _ATTENTION_OPTIONS,
        _TriangleAttention,
    )

    mod = importlib.import_module("team_gm.modules.checkpoints.triangle")
    torch.manual_seed(73)
    source = mod.TriangleAttention(c_in=128, c_hidden=32, no_heads=4)
    with torch.no_grad():
        source.mha.linear_o.weight.normal_(0, 0.04)
    reference = _TriangleAttention(
        configure_model(copy.deepcopy(source), "pytorch"), af3=False, ending=False
    )
    vendor = _TriangleAttention(
        configure_model(source, "cuequivariance"), af3=False, ending=False
    )
    x = torch.randn(1, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 128, 128, device="cuda", dtype=torch.bool)
    token = _ATTENTION_OPTIONS.set((4, None))
    try:
        with torch.inference_mode():
            expected = reference(x.clone(), mask)
            actual = vendor(x.clone(), mask)
        assert (actual - expected).float().square().mean().sqrt() < 0.02
    finally:
        _ATTENTION_OPTIONS.reset(token)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [True, False])
def test_cueq_norm_uses_reference_with_fp32_affine(dtype, affine):
    from team_gm.modules.primitives import LayerNorm

    norm = LayerNorm(
        32, elementwise_affine=affine, implementation=ImplementationType.CUEQUIVARIANCE
    ).to(dtype)
    x = torch.randn(2, 7, 32, dtype=dtype, requires_grad=True)
    reference = copy.deepcopy(norm)
    reference.implementation = ImplementationType.PYTORCH
    ref_x = x.detach().clone().requires_grad_()
    output = norm(x)
    expected = reference(ref_x)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert output.dtype == dtype
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
    if affine:
        assert norm.weight.dtype == torch.float32
        torch.testing.assert_close(
            norm.weight.grad, reference.weight.grad, atol=0, rtol=0
        )
