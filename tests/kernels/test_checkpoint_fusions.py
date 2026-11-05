"""Checkpoint equations and native BF16 contracts of the dense pieces no engine module serves.

FoldForge calls no engine kernel function for these: the output gate + projection, the LayerNorm + narrow
projection and the unconditioned diffusion transition before conversion are PyTorch statements, compiled and
graph-captured like the rest of the model.
"""

from __future__ import annotations

import copy

import pytest
import torch
from miniworld_engine.kernels.rmsnorm.interface import rms_norm_modulation
from torch.nn import functional as F

from foldforge.layers.implementation import ImplementationType
from foldforge.modules.engine.configure import configure_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA validation")


def compare(actual, expected):
    a, b = actual.float(), expected.float()
    rms = (
        (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-6)
    ).item()
    assert torch.isfinite(a).all(), rms
    assert rms < 0.025, rms


def compile_replay(fn, args):
    compiled = torch.compile(fn, fullgraph=True)
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = compiled(*args)
        graph.replay()
        torch.cuda.synchronize()
        compare(y, fn(*args))


def test_af3_atom_pair_projection():
    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )

    source = DiffusionCrossAttTransformer()
    ref = configure_model(copy.deepcopy(source), "pytorch", device="cuda")
    model = configure_model(source, "miniworld", device="cuda")
    x = torch.randn(4, 32, 128, 16, device="cuda", dtype=torch.bfloat16)

    def project(m, x):
        return m.pair_logits_projection(m.pair_input_layer_norm(x))

    with torch.no_grad():
        compare(project(model, x), project(ref, x))
    compile_replay(lambda x: project(model, x), (x,))


@pytest.mark.parametrize("width", [128, 384])
def test_af3_unconditioned_transition(width):
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransition

    source = DiffusionTransition(width, width)
    model = configure_model(copy.deepcopy(source), "miniworld", device="cuda")
    ref = configure_model(source, "pytorch", device="cuda")
    x = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        compare(model(x), ref(x))
    compile_replay(model, (x,))


def test_msa_projection():
    from foldforge.modules.dense.attention import MSAAttention

    torch.manual_seed(11)
    model = MSAAttention(c_msa=128, c_pair=384, num_head=8)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.normal_(1.0 if "norm" in name and name.endswith("weight") else 0.0, 0.05)
    model = configure_model(model, "miniworld", device="cuda")
    ref = copy.deepcopy(model)
    for child in ref.modules():
        child.foldforge_implementation = ImplementationType.PYTORCH
    x = torch.randn(16, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(16, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(128, 128, 384, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        compare(model(x, mask, z), ref(x, mask, z))


def test_gated_projection_gradient_and_graph():
    """The released output gate ``linear(sigmoid(gate) * value)``: PyTorch statements, differentiable, capturable."""
    from foldforge.modules.dense.diffusion_transformer import AdaLNZero

    torch.manual_seed(71)
    layer = configure_model(AdaLNZero(128, 64, None), "miniworld", device="cuda")
    layer.requires_grad_(True)
    g, v = [
        torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    ]
    a = layer(v, gate_logits=g)
    b = F.linear(torch.sigmoid(g) * v, layer.transition2.weight)
    compare(a, b)
    grad = torch.randn_like(a)
    for x, y in zip(
        torch.autograd.grad(a, (g, v), grad, retain_graph=True),
        torch.autograd.grad(b, (g, v), grad),
        strict=True,
    ):
        compare(x, y)
    layer.requires_grad_(False)
    compile_replay(lambda g, v: layer(v, gate_logits=g), (g.detach(), v.detach()))


def test_rms_modulation_equation_and_graph():
    torch.manual_seed(37)
    q, c = [
        torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    weights = [
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) * 0.03
        for _ in range(3)
    ]
    eps = torch.finfo(q.dtype).eps

    def fn(q, c) -> torch.Tensor:
        return rms_norm_modulation(q, c, *weights, eps=eps)

    with torch.no_grad():
        y, g = fn(q, c)
        expected = F.rms_norm(q, (128,), eps=eps) * (
            1 + F.linear(c, weights[0])
        ) + F.linear(c, weights[1])
        compare(y, expected)
        compare(g, F.linear(c, weights[2]))
    compile_replay(lambda q, c: fn(q, c)[0], (q, c))
