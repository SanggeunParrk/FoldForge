"""AF3 samples stay independent while sharing conditioning and pair bias."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from team_gm.diffusion.augmentation import masked_dense_rigid_motion
from team_gm.modules.checkpoints.af_family import configure_model
from torch import nn

from foldforge.models.architectures.af3 import AlphaFold3
from foldforge.modules.dense.diffusion_transformer import CrossAttention, SelfAttention


def test_dense_augmentation_has_independent_sample_transforms():
    torch.manual_seed(12)
    xyz = torch.randn(3, 4, 3).expand(5, -1, -1, -1).clone()
    mask = torch.ones(3, 4)
    mask[-1, -1] = 0
    out = masked_dense_rigid_motion(xyz, mask)
    assert out.shape == xyz.shape
    assert not torch.allclose(out[0], out[1])
    assert torch.count_nonzero(out[:, -1, -1]) == 0
    indices = mask.flatten().bool()
    expected = torch.cdist(xyz[0].flatten(0, 1)[indices], xyz[0].flatten(0, 1)[indices])
    for sample in out:
        actual = torch.cdist(
            sample.flatten(0, 1)[indices], sample.flatten(0, 1)[indices]
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_sampler_calls_one_batched_denoiser_per_step():
    model = AlphaFold3.__new__(AlphaFold3)
    nn.Module.__init__(model)
    model.num_samples, model.diffusion_steps = 5, 3
    model.gamma_0, model.gamma_min, model.noise_scale, model.step_scale = (
        0.8,
        1.0,
        1.003,
        1.5,
    )
    shapes = []

    class Denoiser(nn.Module):
        def forward(self, *, positions_noisy, **_kwargs: Any) -> torch.Tensor:
            shapes.append(tuple(positions_noisy.shape))
            return positions_noisy * 0.5

    model.diffusion_head = Denoiser()
    batch = SimpleNamespace(
        predicted_structure_info=SimpleNamespace(atom_mask=torch.ones(3, 4))
    )
    out = model._sample_diffusion(batch, {})  # noqa: SLF001 - test sampling boundary
    assert shapes == [(5, 3, 4, 3)] * 3
    assert out["atom_positions"].shape == (5, 3, 4, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["pytorch", "miniworld"])
def test_token_attention_batched_matches_separate_and_routes_aug(monkeypatch, backend):
    from miniworld_engine import ops

    torch.manual_seed(7)
    model = configure_model(SelfAttention(192, 96, 4, use_single_cond=True), backend)
    x = torch.randn(5, 128, 192, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(128, device="cuda", dtype=torch.bool)
    mask[-8:] = False
    bias = torch.randn(4, 128, 128, device="cuda", dtype=torch.bfloat16)
    shapes = []
    original = ops.augmented_attention_pair_bias

    def observe(q, k, v, pair, mask) -> torch.Tensor:
        shapes.append((tuple(q.shape), tuple(pair.shape)))
        return original(q, k, v, pair, mask)

    monkeypatch.setattr(ops, "augmented_attention_pair_bias", observe)
    with torch.inference_mode():
        actual = model(x, mask, bias, cond)
        if backend == "miniworld":
            assert shapes == [((5, 1, 4, 128, 48), (1, 4, 128, 128))]
        expected = torch.stack([model(sample, mask, bias, cond) for sample in x])
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_atom_attention_broadcasts_shared_masks_and_conditioning():
    torch.manual_seed(5)
    model = configure_model(CrossAttention(), "pytorch", dtype=torch.float32)
    q = torch.randn(5, 2, 4, 128, device="cuda")
    k = torch.randn(5, 2, 8, 128, device="cuda")
    cq, ck = torch.randn_like(q[0]), torch.randn_like(k[0])
    mq = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    mk = torch.ones(2, 8, device="cuda", dtype=torch.bool)
    mk[:, -1] = False
    with torch.inference_mode():
        actual = model(q, k, mq, mk, single_cond_q=cq, single_cond_k=ck)
        expected = torch.stack(
            [
                model(a, b, mq, mk, single_cond_q=cq, single_cond_k=ck)
                for a, b in zip(q, k, strict=True)
            ]
        )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
