"""Checkpoint equations, dispatch and native BF16 contracts for added fusions."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
from miniworld_engine import ops, settings
from miniworld_engine.architecture.exceptions import ImplementationType
from support import triton_ops_run
from torch.nn import functional as F

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


@pytest.mark.parametrize("pin", ["fused", "split"])
def test_gate_gradient_and_graph(pin):
    torch.manual_seed(71)
    prior = settings.current().pin_gate_backend
    settings.configure(pin_gate_backend=pin)
    try:
        g, v = [
            torch.randn(
                2, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(2)
        ]
        w = (
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.05
        ).requires_grad_()
        a = ops.gated_linear(g, v, w)
        b = F.linear(torch.sigmoid(g) * v, w)
        compare(a, b)
        grad = torch.randn_like(a)
        da = torch.autograd.grad(a, (g, v, w), grad)
        db = torch.autograd.grad(b, (g, v, w), grad)
        for x, y in zip(da, db, strict=False):
            compare(x, y)
        compile_replay(ops.gated_linear, (g, v, w))
    finally:
        settings.configure(pin_gate_backend=prior)


@pytest.mark.parametrize(
    ("affine", "bias"), [(True, False), (False, False), (True, True)]
)
def test_layernorm_optional_affine(affine, bias):
    x = torch.randn(
        2, 128, 267, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w = (
        torch.randn(267, device="cuda", dtype=torch.float32, requires_grad=True)
        if affine
        else None
    )
    b = (
        torch.randn(267, device="cuda", dtype=torch.float32, requires_grad=True)
        if bias
        else None
    )
    y = ops.layer_norm(x, w, b, 1e-5)
    ref = F.layer_norm(x.float(), (267,), w, b, 1e-5).to(x.dtype)
    compare(y, ref)
    args = tuple(v for v in (x, w, b) if v is not None)
    grad = torch.randn_like(y)
    for ga, gb in zip(
        torch.autograd.grad(y, args, grad),
        torch.autograd.grad(ref, args, grad),
        strict=False,
    ):
        compare(ga, gb)
    compile_replay(lambda x: ops.layer_norm(x, w, b, 1e-5), (x,))


def test_af3_atom_pair_projection():
    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )
    from foldforge.modules.engine.attention_ops import (
        pair_bias_projection,
    )

    source = DiffusionCrossAttTransformer()
    model = configure_model(source, "miniworld", device="cuda")
    norm, linear = model.pair_input_layer_norm, model.pair_logits_projection
    x = torch.randn(4, 32, 128, 16, device="cuda", dtype=torch.bfloat16)
    with (
        torch.no_grad(),
        patch.object(ops, "layer_norm_linear", wraps=ops.layer_norm_linear) as called,
    ):
        a = pair_bias_projection(norm, linear, x)
        b = linear(norm(x))
        assert called.call_count == triton_ops_run()
        compare(a, b)
    compile_replay(lambda x: pair_bias_projection(norm, linear, x), (x,))


@pytest.mark.parametrize("width", [128, 384])
def test_af3_unconditioned_transition(width):
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransition

    source = DiffusionTransition(width, width)
    model = configure_model(copy.deepcopy(source), "miniworld", device="cuda")
    ref = configure_model(source, "pytorch", device="cuda")
    x = torch.randn(128, width, device="cuda", dtype=torch.bfloat16)
    with (
        torch.no_grad(),
        patch.object(ops, "swiglu_ffn", wraps=ops.swiglu_ffn) as called,
    ):
        compare(model(x), ref(x))
        assert called.call_count == triton_ops_run()
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
    with (
        torch.no_grad(),
        patch.object(ops, "gated_linear", wraps=ops.gated_linear) as called,
    ):
        compare(model(x, mask, z), ref(x, mask, z))
        assert called.call_count == triton_ops_run()


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
        return ops.rms_norm_modulation(q, c, *weights, eps=eps)

    with torch.no_grad():
        y, g = fn(q, c)
        expected = F.rms_norm(q, (128,), eps=eps) * (
            1 + F.linear(c, weights[0])
        ) + F.linear(c, weights[1])
        compare(y, expected)
        compare(g, F.linear(c, weights[2]))
    compile_replay(lambda q, c: fn(q, c)[0], (q, c))
