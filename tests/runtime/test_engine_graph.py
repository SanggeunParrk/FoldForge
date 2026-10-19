"""The engine's DiT blocks run inside the denoiser's compile / CUDA-graph boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from foldforge.models.execution import ExecutedCallable
from foldforge.modules.engine.configure import configure_model
from foldforge.modules.engine.modules import engine_of, install_engine_modules

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def denoiser():
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer

    torch.manual_seed(7)
    layer = DiffusionTransformer(768, 384, 128, 16, 4)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
    layer = configure_model(layer, "miniworld")
    install_engine_modules(layer, torch.empty(0, dtype=torch.bfloat16, device="cuda"))
    assert engine_of(layer) is not None
    return layer


@pytest.mark.parametrize("compile_", [False, True])
def test_denoiser_graph_replays_the_eager_result(compile_):
    layer = denoiser()
    length = 256
    act = torch.randn(2, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(2, length, 384, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(length, device="cuda") > 0.15

    def step(x: torch.Tensor) -> torch.Tensor:
        return layer(x, mask, single, pair)

    policy = SimpleNamespace(compile=compile_, cuda_graph=True, max_graphs=2)
    graphed = ExecutedCallable(step, policy, "test.denoiser")
    with torch.no_grad():
        expected = step(act.clone())
        first = graphed(act.clone()).clone()
        second = graphed(act.clone()).clone()
    assert graphed.captures == 1
    assert graphed.replays == 2
    torch.testing.assert_close(first, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
