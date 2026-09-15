"""Native geometry and conditioning projections retain their precision contract."""

from unittest.mock import patch

import pytest
import torch
from miniworld_engine import ops
from team_gm.modules.checkpoints.af_family import configure_model
from team_gm.modules.checkpoints.backend_attention import layernorm_projection
from team_gm.modules.checkpoints.layers import LayerNorm
from team_gm.modules.precision import NativeLinear

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
    norm = LayerNorm(width, create_offset=False)
    linear = NativeLinear(width, out_width, bias=False)
    linear.compute_dtype = compute
    model = configure_model(
        torch.nn.Sequential(norm, linear), "miniworld", device="cuda"
    )
    x = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
    with (
        torch.no_grad(),
        patch.object(ops, "layer_norm_linear", wraps=ops.layer_norm_linear) as called,
    ):
        a = layernorm_projection(model[0], model[1], x)
        b = model[1](model[0](x))
        assert called.call_count == (1 if out_width <= 16 else 0)
        error = (
            a.float() - b.float()
        ).square().mean().sqrt() / b.float().square().mean().sqrt()
        assert torch.isfinite(a).all(), error
        assert error < 0.025, error
    compiled = torch.compile(
        lambda x: layernorm_projection(model[0], model[1], x), fullgraph=True
    )
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), a, atol=0, rtol=0)
