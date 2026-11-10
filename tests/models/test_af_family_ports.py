"""AF-family regressions that need no checkpoint: the sampler keeps its noise field; the engine norm keeps an fp32 optional affine."""

from __future__ import annotations

# Tests target private upstream equations directly without loading full checkpoints.
# ruff: noqa: SLF001
from unittest import mock

import pytest
import torch


@pytest.mark.parametrize(("affine", "bias"), [(False, False), (True, False), (True, True)])
def test_engine_layernorm_optional_affine_keeps_its_parameters_fp32(affine, bias):
    from foldforge.modules.engine.modules import layer_norm

    layer = layer_norm(128, bias=bias, elementwise_affine=affine)
    x = torch.randn(13, 128, dtype=torch.bfloat16)
    expected = torch.nn.functional.layer_norm(x.float(), (128,), layer.weight, layer.bias, layer.eps).to(x.dtype)
    torch.testing.assert_close(layer(x), expected, rtol=0.01, atol=0.02)
    assert set(layer.state_dict()) == ({"weight", "bias"} if affine and bias else {"weight"} if affine else set())
    assert all(p.dtype == torch.float32 for p in layer.parameters())


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
    batch = SimpleNamespace(predicted_structure_info=SimpleNamespace(atom_mask=torch.ones(4, 24)))
    with mock.patch(
        "foldforge.diffusion.augmentation.masked_dense_rigid_motion",
        side_effect=lambda positions, _mask: positions,
    ):
        result = model._sample_diffusion(batch, {})["atom_positions"]
    assert result.shape == (2, 4, 24, 3)
    assert result.std() > 0
    assert not torch.equal(result[0], result[1])
    assert torch.unique(result).numel() == result.numel()
