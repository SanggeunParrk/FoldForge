"""Shared diffusion samples reach engine augmentation axes without batch mixing."""

from __future__ import annotations

import pytest
import torch
from team_gm.modules.checkpoints.af_family import configure_model
from team_gm.modules.checkpoints.transformer import AttentionPairBias

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("residual_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("batch", "samples"), [(1, 5), (2, 3), (2, 1)])
def test_shared_token_attention_preserves_structure_and_sample_axes(
    monkeypatch, batch, samples, residual_dtype
):
    from miniworld_engine import ops

    torch.manual_seed(53)
    model = configure_model(
        AttentionPairBias(c_a=192, c_s=96, c_z=32, n_heads=4), "miniworld"
    )
    a = torch.randn(batch, samples, 128, 192, device="cuda", dtype=residual_dtype)
    s = torch.randn(batch, samples, 128, 96, device="cuda", dtype=residual_dtype)
    z = torch.randn(batch, 1, 128, 128, 32, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(128, 128, device="cuda")
    bias[:, -8:] = -1e9
    calls = []
    original = ops.augmented_attention_pair_bias

    def observe(q, k, v, pair, mask) -> torch.Tensor:
        calls.append((tuple(q.shape), tuple(pair.shape)))
        return original(q, k, v, pair, mask)

    monkeypatch.setattr(ops, "augmented_attention_pair_bias", observe)
    with torch.inference_mode():
        actual = model(a, s, z, extra_attn_bias=bias)
        assert calls == [((samples, batch, 4, 128, 48), (batch, 4, 128, 128))]
        expected = torch.stack(
            [
                model(a[:, i], s[:, i], z[:, 0], extra_attn_bias=bias)
                for i in range(samples)
            ],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.04)
