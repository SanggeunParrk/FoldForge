"""Exercise projected attention without attached twins on the selected backend."""

from __future__ import annotations

import copy
import os

import pytest
import torch

from foldforge.modules.engine.configure import configure_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("backend", ["miniworld", "cuequivariance", "anthropic"])
@pytest.mark.parametrize("transpose", [False, True])
def test_projected_core_matches_reference_and_selects_vendor(
    monkeypatch, backend, transpose
):
    from foldforge.modules.dense.attention import GridSelfAttention
    from foldforge.modules.engine import attention_ops
    from foldforge.modules.engine.modules import engine_of

    if backend == "cuequivariance":
        pytest.importorskip("cuequivariance_torch")
    if backend == "anthropic" and not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"):
        pytest.skip("Anthropic payload not configured")
    torch.manual_seed(23)
    source = GridSelfAttention(128, 4, transpose=transpose)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.normal_(
                1.0 if "norm" in name and name.endswith("weight") else 0.0, 0.07
            )
    reference = configure_model(copy.deepcopy(source), "pytorch")
    model = configure_model(source, backend)
    assert engine_of(model) is None
    x = torch.randn(128, 128, 128, device="cuda", dtype=torch.bfloat16)
    tokens = torch.rand(128, device="cuda") > 0.2
    mask = tokens[:, None] & tokens[None, :]
    calls = []
    if backend == "anthropic":
        original = attention_ops.anthropic_attention

        def core(*args: object) -> torch.Tensor | None:
            calls.append("anthropic")
            return original(*args)

        monkeypatch.setattr(attention_ops, "anthropic_attention", core)
    elif backend == "cuequivariance":
        import cuequivariance_torch

        original = cuequivariance_torch.triangle_attention

        def core(*args: object, **options: object) -> torch.Tensor:
            calls.append("cuequivariance")
            return original(*args, **options)

        monkeypatch.setattr(cuequivariance_torch, "triangle_attention", core)
    with torch.no_grad():
        expected = reference(x, mask)
        actual = model(x, mask)
    assert calls == ([] if backend == "miniworld" else [backend])
    assert torch.isfinite(actual).all()
    # Same projected-attention bounds as test_backend_attention.compare.
    rms = (
        actual.float() - expected.float()
    ).square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-6)
    assert rms < 0.025
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.04, rtol=0.05)
