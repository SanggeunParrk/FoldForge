"""The checkpoint equations of the pieces around the engine blocks: the released output gate and the engine's ``AdaptiveRMSNorm``.

``AdaptiveRMSNorm`` (ESMFold2's atom AdaLN: an affine-free RMSNorm, SiLU of the conditioning, identity-centred scale, no biases) is the
engine class that replaces the rms-modulation kernel function FoldForge used to call; the equation is checked on its own backend.
"""

from __future__ import annotations

import pytest
import torch
from miniworld_engine.modules import AdaptiveRMSNorm
from torch.nn import functional as F

from foldforge.modules.dense.diffusion_transformer import AdaLNZero
from foldforge.modules.engine.configure import configure_model


def test_gated_projection_gradient():
    """The released output gate ``linear(sigmoid(gate) * value)``: PyTorch statements, differentiable."""
    torch.manual_seed(71)
    layer = configure_model(AdaLNZero(32, 16, None), "pytorch", torch.float32, "cpu")
    layer.requires_grad_(True)
    g, v = (torch.randn(2, 8, 32, requires_grad=True) for _ in range(2))
    a = layer(v, gate_logits=g)
    b = F.linear(torch.sigmoid(g) * v, layer.transition2.weight)
    torch.testing.assert_close(a, b)
    grad = torch.randn_like(a)
    for x, y in zip(
        torch.autograd.grad(a, (g, v), grad, retain_graph=True), torch.autograd.grad(b, (g, v), grad), strict=True
    ):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaptive_rms_norm_equation(dtype):
    torch.manual_seed(37)
    norm = AdaptiveRMSNorm(64, 64, dtype=dtype)
    with torch.no_grad():
        norm.to_scale.weight.normal_(0, 0.1)
        norm.to_bias.weight.normal_(0, 0.1)
    x, c = (torch.randn(2, 8, 64).to(dtype) for _ in range(2))
    eps = torch.finfo(torch.float32).eps
    xw, context = x.float(), F.silu(c.float())
    expected = (
        (1 + F.linear(context, norm.to_scale.weight.float())) * F.rms_norm(xw, (64,), eps=eps) + F.linear(context, norm.to_bias.weight.float())
    ).to(dtype)
    with torch.no_grad():
        actual = norm(x, c)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2 if dtype == torch.bfloat16 else 1e-5, rtol=2e-2)

