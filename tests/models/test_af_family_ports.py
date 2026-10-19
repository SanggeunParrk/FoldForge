from __future__ import annotations

from support import production_module, triton_ops_run

# Tests target private upstream equations directly without loading full checkpoints.
# ruff: noqa: SLF001
"""Checkpoint-layout and residual regressions for the AF-family ports."""

import copy
import importlib
from unittest import mock

import pytest
import torch
from miniworld_engine import ops

from foldforge.modules.engine.configure import configure_model
from foldforge.modules.engine.routing import transition_residual, triangle_residual


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity")
def test_triangle_mask_and_weight_layout(outgoing):
    torch.manual_seed(42)
    mod = importlib.import_module("foldforge.modules.dense.triangle_multiplication")
    layer = mod.TriangleMultiplication(c_pair=128, _outgoing=outgoing)
    # Upstream initializes output projections to zero: randomize them so a
    # swapped direction, mask omission or incorrect interleave cannot pass.
    with torch.no_grad():
        for p in layer.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.08)
    reference = configure_model(copy.deepcopy(layer), "pytorch")
    engine = configure_model(layer, "miniworld")
    x = torch.randn(32, 32, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(32, 32, device="cuda") > 0.2
    with (
        torch.inference_mode(),
        mock.patch.object(
            ops,
            "triangle_multiplicative_update",
            wraps=ops.triangle_multiplicative_update,
        ) as call,
    ):
        expected = triangle_residual(reference, x.clone(), mask)
        actual = triangle_residual(engine, x.clone(), mask)
    assert call.call_count == triton_ops_run()
    error = (actual - expected).float()
    assert error.square().mean().sqrt() < 0.007
    assert (
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        > 0.999
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity")
def test_transition_residual_once():
    torch.manual_seed(9)
    layer = production_module("af3", "nn.primitives").Transition(128)
    with torch.no_grad():
        for p in layer.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
    ref = configure_model(copy.deepcopy(layer), "pytorch")
    engine = configure_model(layer)
    x = torch.randn(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    with (
        torch.inference_mode(),
        mock.patch.object(ops, "transition", wraps=ops.transition) as call,
    ):
        expected = transition_residual(ref, x)
        actual = transition_residual(engine, x)
    assert call.call_count == 1
    assert (actual - expected).float().square().mean().sqrt() < 0.005
    for name, p in engine.named_parameters():
        assert p.dtype == (torch.float32 if p.ndim == 1 else torch.bfloat16), name


def test_pairformer_single_transition_receives_backend():
    from miniworld_engine.architecture.blocks.pairformer import PairformerBlock
    from miniworld_engine.architecture.exceptions import ImplementationType

    impl = ImplementationType.MINIWORLD_ENGINE
    block = PairformerBlock(implementation=impl)
    assert block.transition_single.implementation.value == "miniworld"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity")
@pytest.mark.parametrize(
    ("affine", "bias"), [(False, False), (True, False), (True, True)]
)
def test_native_layernorm_optional_affine(affine, bias):
    ref = torch.nn.LayerNorm(128, elementwise_affine=affine, bias=bias)
    layer = configure_model(torch.nn.Sequential(ref))[0]
    x = torch.randn(13, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = torch.nn.functional.layer_norm(
            x.float(), (128,), layer.weight, layer.bias, layer.eps
        ).to(x.dtype)
        actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    assert set(layer.state_dict()) == (
        {"weight", "bias"} if affine and bias else {"weight"} if affine else set()
    )


def test_af3_diffusion_preserves_random_noise_field():
    from types import SimpleNamespace

    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.modules.dense.spec import ALPHAFOLD3

    class IdentityDenoiser(torch.nn.Module):
        def forward(self, positions_noisy, **_kwargs: object) -> torch.Tensor:
            return positions_noisy

    model = AlphaFold3.__new__(AlphaFold3)
    torch.nn.Module.__init__(model)
    model.num_samples, model.diffusion_steps = 2, 2
    model.spec = ALPHAFOLD3
    model.diffusion_head = IdentityDenoiser()
    batch = SimpleNamespace(
        predicted_structure_info=SimpleNamespace(atom_mask=torch.ones(4, 24))
    )
    with mock.patch(
        "miniworld_engine.architecture.diffusion.augmentation.masked_dense_rigid_motion",
        side_effect=lambda positions, _mask: positions,
    ):
        result = model._sample_diffusion(batch, {})["atom_positions"]
    assert result.shape == (2, 4, 24, 3)
    assert result.std() > 0
    assert not torch.equal(result[0], result[1])
    assert torch.unique(result).numel() == result.numel()
