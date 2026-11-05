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
        changed = act * 0.9 + 0.02
        expected_changed = step(changed.clone())
        third = graphed(changed.clone()).clone()
    assert graphed.captures == 1
    assert graphed.replays == 3
    torch.testing.assert_close(first, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(third, expected_changed, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("compile_", [False, True])
def test_pairformer_adapter_graph_preserves_changed_inputs(parallel, compile_):
    from dataclasses import replace

    from foldforge.modules.dense.pairformer import PairformerBlock
    from foldforge.modules.dense.spec import ALPHAFOLD3
    from foldforge.modules.engine.pairformer import AF3Pairformer
    from foldforge.modules.engine.serving import report_observations

    torch.manual_seed(13)
    source = PairformerBlock(
        spec=replace(ALPHAFOLD3, parallel_pairformer_block=parallel)
    )
    with torch.no_grad():
        for p in source.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
    layer = AF3Pairformer(configure_model(source, "miniworld"))
    install_engine_modules(layer, torch.empty(0, dtype=torch.bfloat16, device="cuda"))
    length = 128
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(length, 384, device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.15
    mask = residues[:, None] & residues[None, :]

    def step(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return layer(x, mask, single, residues)

    policy = SimpleNamespace(compile=compile_, cuda_graph=True, max_graphs=2)
    graphed = ExecutedCallable(step, policy, "test.pairformer")
    with torch.no_grad():
        changed = pair * 0.9 + 0.02
        expected = step(pair.clone())
        expected_changed = step(changed.clone())
        first = tuple(v.clone() for v in graphed(pair.clone()))
        before_replay = report_observations(layer)
        second = tuple(v.clone() for v in graphed(changed.clone()))
    assert graphed.captures == 1
    assert graphed.replays == 2
    assert report_observations(layer) == before_replay
    for actual, want in zip(first, expected, strict=True):
        torch.testing.assert_close(actual, want, rtol=0.02, atol=0.02)
    for actual, want in zip(second, expected_changed, strict=True):
        torch.testing.assert_close(actual, want, rtol=0.02, atol=0.02)
