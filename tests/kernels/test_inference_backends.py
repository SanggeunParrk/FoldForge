"""Backend selection must reach the executed operator, with one residual."""

import copy
from unittest import mock

import pytest
import torch
from miniworld_engine.architecture.exceptions import ImplementationType

from foldforge.models.config import Config
from foldforge.modules.dense.attention import GridSelfAttention
from foldforge.modules.engine.configure import configure_model
from foldforge.modules.engine.routing import triangle_residual


@pytest.mark.parametrize("backend", ["pytorch", "cuequivariance", "miniworld"])
def test_backend_reaches_the_executed_operator(backend):
    """The requested backend reaches the layer, not only the config object."""
    expected = (
        ImplementationType.MINIWORLD_ENGINE
        if backend == "miniworld"
        else ImplementationType(backend)
    )
    assert Config(backend=backend).backend == backend
    layer = configure_model(
        GridSelfAttention(c_pair=32, num_head=4, transpose=False),
        backend,
        torch.float32,
        "cpu",
    )
    assert layer.foldforge_implementation == expected


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vendor CUDA kernels")
def test_cueq_triangle_direction_mask_and_residual(outgoing):
    from cuequivariance_torch import triangle_multiplicative_update

    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication

    torch.manual_seed(42)
    layer = TriangleMultiplication(128, _outgoing=outgoing)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [True, False])
def test_cueq_norm_uses_reference_with_fp32_affine(dtype, affine):
    from miniworld_engine.architecture.primitives import LayerNorm

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
