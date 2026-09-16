"""Shared diffusion samples reach engine augmentation axes without batch mixing."""

from __future__ import annotations

import pytest
import torch
from team_gm.modules.checkpoints.af_family import configure_model
from team_gm.modules.checkpoints.transformer import AttentionPairBias
from team_gm.modules.exceptions import ImplementationType

from foldforge.models.config.esmfold2 import AtomAttentionConfig
from foldforge.modules.sequence.atom_encoder import AtomEncoder

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


@pytest.mark.parametrize("backend", ["pytorch", "miniworld"])
def test_swa_shared_rope_and_independent_samples(monkeypatch, backend):
    from foldforge.modules.sequence import atom_encoder

    torch.manual_seed(17)
    batch, samples, atoms, tokens = 2, 3, 128, 64
    model = configure_model(
        AtomEncoder(
            AtomAttentionConfig(n_blocks=1),
            64,
            structure_prediction=True,
            implementation=(
                ImplementationType.MINIWORLD_ENGINE
                if backend == "miniworld"
                else ImplementationType.PYTORCH
            ),
        ),
        backend,
    )
    with torch.no_grad():
        model.atom_transformer.blocks[0].adaln_modulation[1].weight.normal_(0, 0.02)
    kwargs = {
        "ref_pos": torch.randn(batch, atoms, 3, device="cuda", dtype=torch.bfloat16),
        "ref_charge": torch.zeros(batch, atoms, device="cuda", dtype=torch.bfloat16),
        "ref_element": torch.zeros(
            batch, atoms, 128, device="cuda", dtype=torch.bfloat16
        ),
        "ref_atom_name_chars": torch.zeros(
            batch, atoms, 4, 64, device="cuda", dtype=torch.bfloat16
        ),
        "ref_space_uid": torch.arange(atoms, device="cuda").expand(batch, -1),
        "atom_mask": torch.ones(batch, atoms, device="cuda", dtype=torch.bool),
        "atom_to_token": (torch.arange(atoms, device="cuda") % tokens).expand(
            batch, -1
        ),
        "n_tokens": tokens,
    }
    kwargs["atom_mask"][0, -16:] = False
    kwargs["atom_mask"][1, -8:] = False
    coords = torch.randn(samples * batch, atoms, 3, device="cuda")
    calls = []
    original = atom_encoder.build_attention_params

    def observe(cos, sin, valid, num_aug) -> tuple:
        calls.append((tuple(cos.shape), tuple(valid.shape), num_aug))
        return original(cos, sin, valid, num_aug)

    monkeypatch.setattr(atom_encoder, "build_attention_params", observe)
    with torch.inference_mode():
        actual = model(**kwargs, coords=coords, num_aug=samples)
        assert calls == [((batch, atoms, 16), (samples * batch, atoms), samples)]
        separate = [
            model(**kwargs, coords=c) for c in coords.reshape(samples, batch, atoms, 3)
        ]
        for index in range(3):
            expected = torch.cat([x[index] for x in separate])
            torch.testing.assert_close(actual[index], expected, atol=0.035, rtol=0.05)
        assert torch.equal(actual[3][-1], kwargs["atom_mask"].repeat(samples, 1))
