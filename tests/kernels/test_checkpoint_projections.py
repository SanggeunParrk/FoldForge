"""Native geometry and conditioning projections retain their precision contract."""

import pytest
import torch

from foldforge.modules.engine.configure import configure_model
from foldforge.modules.engine.native_linear import NativeLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA validation")


@pytest.mark.parametrize(
    ("width", "out_width", "compute"),
    [
        (267, 128, None),
        (256, 128, torch.float32),
        (512, 256, torch.float32),
        (384, 128, torch.float32),
        (128, 3, torch.float32),
        (831, 384, None),
    ],
)
def test_native_projection(width, out_width, compute):
    torch.manual_seed(54)
    norm = torch.nn.LayerNorm(width, bias=False)
    linear = NativeLinear(width, out_width, bias=False)
    linear.compute_dtype = compute
    model = configure_model(
        torch.nn.Sequential(norm, linear), "miniworld", device="cuda"
    )
    x = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(x.float(), (width,), model[0].weight, None, model[0].eps),
        model[1].weight.float(),
    )
    with torch.no_grad():
        a = model(x)
        b = reference
        error = (
            a.float() - b.float()
        ).square().mean().sqrt() / b.float().square().mean().sqrt()
        assert torch.isfinite(a).all(), error
        assert error < 0.025, error
    compiled = torch.compile(model, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), a, atol=0, rtol=0)
