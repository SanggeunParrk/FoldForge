"""H100 and B200 never run an untuned Triton kernel: a Triton launch fails these tests.

The engine's ``ops`` reach Triton only; the hand-CUDA legs live in its modules. A dense
module the installer converts runs the module, and everything else runs hand-written
CUDA or PyTorch. These tests run the graph's pieces with ``JITFunction.run`` replaced
by a refusal, converted and not.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from foldforge.modules.engine.configure import configure_model
from foldforge.modules.engine.modules import install_engine_modules
from foldforge.modules.engine.pairformer import install_pairformers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def hopper_or_blackwell():
    from miniworld_engine.modules.dispatch import restricted_card

    if not restricted_card(torch.device("cuda")):
        pytest.skip("only H100 / B200 forbid Triton")


@pytest.fixture
def no_triton(monkeypatch):
    import triton.runtime.jit as jit

    def refuse(self: object, *_args: object, **_kwargs: object) -> None:
        message = f"a Triton kernel was launched: {getattr(self.fn, '__name__', self)}"
        raise AssertionError(message)

    monkeypatch.setattr(jit.JITFunction, "run", refuse)


def rms_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return (
        (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    ).item()


def randomize(module: nn.Module, std: float = 0.04) -> nn.Module:
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim > 1:
                p.normal_(0, std)
    return module


@pytest.mark.usefixtures("no_triton")
@pytest.mark.parametrize("shape", [(5, 128), (2, 7, 128), (3, 4, 5, 384)])
def test_layer_norm(shape):
    layer = nn.Sequential(nn.LayerNorm(shape[-1]))
    with torch.no_grad():
        layer[0].weight.normal_(1, 0.1)
        layer[0].bias.normal_(0, 0.1)
    model = configure_model(copy.deepcopy(layer), "miniworld")
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = torch.nn.functional.layer_norm(
            x.float(), (shape[-1],), layer[0].weight.cuda(), layer[0].bias.cuda()
        )
        actual = model(x)
    assert rms_error(actual, expected) < 0.01


def pairformer_inputs(length=128):
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(length, 384, device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.1
    return pair, residues[:, None] & residues[None, :], single, residues.float()


@pytest.mark.usefixtures("no_triton")
@pytest.mark.parametrize("installed", [False, True])
def test_pairformer_block(installed):
    from foldforge.modules.dense.pairformer import PairformerBlock

    torch.manual_seed(0)
    container = nn.Sequential(randomize(PairformerBlock()))
    ref = configure_model(copy.deepcopy(container), "pytorch")
    model = configure_model(container, "miniworld")
    if installed:
        model.foldforge_load_report = {}
        install_pairformers(model)
        install_engine_modules(
            model, torch.empty(0, dtype=torch.bfloat16, device="cuda")
        )
    pair, pair_mask, single, seq_mask = pairformer_inputs()
    with torch.no_grad():
        expected = ref[0](pair.clone(), pair_mask, single.clone(), seq_mask)
        actual = model[0](pair.clone(), pair_mask, single.clone(), seq_mask)
    real = seq_mask.bool()
    assert rms_error(actual[1][real], expected[1][real]) < 0.05


@pytest.mark.usefixtures("no_triton")
@pytest.mark.parametrize("installed", [False, True])
def test_msa_block(installed):
    from foldforge.modules.dense.pairformer import EvoformerBlock

    torch.manual_seed(1)
    container = nn.Sequential(randomize(EvoformerBlock(), 0.05))
    truth = copy.deepcopy(container).cuda().float()
    ref = configure_model(copy.deepcopy(container), "pytorch")
    model = configure_model(container, "miniworld")
    if installed:
        install_engine_modules(
            model, torch.empty(0, dtype=torch.bfloat16, device="cuda")
        )
    length = 128
    msa = torch.randn(16, length, 64, device="cuda", dtype=torch.bfloat16)
    pair, pair_mask, _, seq_mask = pairformer_inputs(length)
    msa_mask = (torch.rand(16, length, device="cuda") > 0.1).float() * seq_mask
    msa_mask[0] = seq_mask
    with torch.no_grad():
        exact = truth[0](msa.float(), pair.float(), msa_mask, pair_mask.float())
        expected = ref[0](msa.clone(), pair.clone(), msa_mask, pair_mask.float())
        actual = model[0](msa.clone(), pair.clone(), msa_mask, pair_mask.float())
    real = seq_mask.bool()
    got = rms_error(actual[1][real][:, real], exact[1][real][:, real])
    base = rms_error(expected[1][real][:, real], exact[1][real][:, real])
    assert got <= base + 0.01, (got, base)


@pytest.mark.usefixtures("no_triton")
@pytest.mark.parametrize("inference_mode", [False, True])
def test_diffusion_transformer(inference_mode):
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer

    torch.manual_seed(2)
    layer = DiffusionTransformer(768, 384, 128, 16, 4)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
    container = nn.Sequential(layer)
    ref = configure_model(copy.deepcopy(container), "pytorch")
    model = configure_model(container, "miniworld")
    install_engine_modules(model, torch.empty(0, dtype=torch.bfloat16, device="cuda"))
    length = 128
    act = torch.randn(2, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(2, length, 384, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.1
    context = torch.inference_mode if inference_mode else torch.no_grad
    with context():
        expected = ref[0](act.clone(), residues, single, pair)
        actual = model[0](act.clone(), residues, single, pair)
    assert rms_error(actual[:, residues], expected[:, residues]) < 0.05
