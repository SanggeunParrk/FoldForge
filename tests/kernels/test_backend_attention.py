"""GPU numerics, actual operator dispatch, and compile/graph checks for checkpoint
attention."""

from __future__ import annotations

import copy

import pytest
import torch
from team_gm.modules.checkpoints.af_family import configure_model
from team_gm.modules.checkpoints.backend_attention import (
    engine_dense_attention,
    pair_bias_projection,
)
from torch.profiler import ProfilerActivity, profile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA backend validation"
)


def compare(label, actual, expected):
    a, b = actual.float(), expected.float()
    assert torch.isfinite(a).all(), label
    rms = (
        (a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-6)
    ).item()
    err = (a - b).abs().max().item()
    assert rms < 0.025, (label, rms, err)
    torch.testing.assert_close(a, b, atol=0.04, rtol=0.05)


@pytest.mark.parametrize(
    "kind",
    ["af3-start", "af3-end", "shared-start", "shared-end", "af3-token", "shared-token"],
)
@pytest.mark.parametrize("mask_kind", ["mixed", "empty"])
def test_module(kind, mask_kind):
    torch.manual_seed(4)
    chain_length = 128
    if kind.startswith("af3-") and kind != "af3-token":
        from foldforge.modules.dense.attention import GridSelfAttention

        model = GridSelfAttention(128, 4, transpose=kind.endswith("end"))
        x = torch.randn(
            chain_length, chain_length, 128, device="cuda", dtype=torch.bfloat16
        )
        mask = (
            (torch.rand(chain_length, chain_length, device="cuda") > 0.2)
            if mask_kind == "mixed"
            else torch.zeros(
                chain_length, chain_length, device="cuda", dtype=torch.bool
            )
        )
        kwargs = {"mask": mask}
    elif kind.startswith("shared-") and kind != "shared-token":
        from team_gm.modules.checkpoints.triangle import TriangleAttention

        model = TriangleAttention(128, 32, 4, starting=kind.endswith("start"))
        x = torch.randn(
            2, chain_length, chain_length, 128, device="cuda", dtype=torch.bfloat16
        )
        mask = (
            (torch.rand(2, chain_length, chain_length, device="cuda") > 0.2)
            if mask_kind == "mixed"
            else torch.zeros(
                2, chain_length, chain_length, device="cuda", dtype=torch.bool
            )
        )
        kwargs = {"mask": mask.float(), "chunk_size": 16}
    elif kind == "af3-token":
        from foldforge.modules.dense.diffusion_transformer import SelfAttention

        model = SelfAttention(192, 96, 4, use_single_cond=False)
        x = torch.randn(chain_length, 192, device="cuda", dtype=torch.bfloat16)
        mask = (
            (torch.rand(chain_length, device="cuda") > 0.2)
            if mask_kind == "mixed"
            else torch.zeros(chain_length, device="cuda", dtype=torch.bool)
        )
        kwargs = {
            "mask": mask,
            "pair_logits": torch.randn(
                4, chain_length, chain_length, device="cuda", dtype=torch.bfloat16
            ),
        }
    else:
        from team_gm.modules.checkpoints.dense import Attention

        model = Attention(192, 192, 192, 48, 4, zero_init=False)
        x = torch.randn(2, chain_length, 192, device="cuda", dtype=torch.bfloat16)
        mask = (
            (torch.rand(2, chain_length, device="cuda") > 0.2)
            if mask_kind == "mixed"
            else torch.zeros(2, chain_length, device="cuda", dtype=torch.bool)
        )
        kwargs = {
            "kv_x": x,
            "attn_bias": torch.randn(
                2, 4, chain_length, chain_length, device="cuda", dtype=torch.bfloat16
            )
            + (~mask[:, None, None, :]).float() * -1e9,
        }
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.normal_(
                1.0 if "norm" in name and name.endswith("weight") else 0.0, 0.07
            )
    ref = configure_model(copy.deepcopy(model), "pytorch", device="cuda")
    model = configure_model(model, "miniworld", device="cuda")
    with torch.no_grad():
        expected = ref(x, **kwargs)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            actual = model(x, **kwargs)
    assert any(
        e.key == "miniworld_engine::augmented_attention_fwd"
        for e in prof.key_averages()
    ), kind
    compare(kind + "-" + mask_kind, actual, expected)


def test_compile_graph():
    torch.manual_seed(8)
    q, k, v = [
        torch.randn(2, 4, 128, 48, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    bias = torch.randn(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, 128, device="cuda", dtype=torch.bool)
    mask[1] = False

    def fn(q, k, v, bias, mask) -> torch.Tensor:
        return engine_dense_attention(q, k, v, bias, mask)

    compiled = torch.compile(fn, fullgraph=True)
    with torch.no_grad():
        expected = fn(q, k, v, bias, mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(q, k, v, bias, mask)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = compiled(q, k, v, bias, mask)
        graph.replay()
        torch.cuda.synchronize()
        compare("compile-graph", out, expected)


def test_fused_pair_projection():
    from team_gm.modules.checkpoints.layers import LayerNorm

    torch.manual_seed(7)
    norm = LayerNorm(128, create_offset=False)
    linear = torch.nn.Linear(128, 16, bias=False)
    container = torch.nn.Sequential(norm, linear)
    ref = configure_model(copy.deepcopy(container), "pytorch", device="cuda")
    model = configure_model(container, "miniworld", device="cuda")
    x = torch.randn(2, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = ref(x)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            actual = pair_bias_projection(model[0], model[1], x)
    assert any("layernorm_linear_pair_bias_fwd" in e.key for e in prof.key_averages())
    compare("pair-ln-linear", actual, expected)


def test_pytorch_no_custom_ops():
    from team_gm.modules.checkpoints.dense import Attention

    model = configure_model(
        Attention(128, 128, 128, 32, 4, zero_init=False), "pytorch", device="cuda"
    )
    x = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        model(x, x)
    assert not any(
        e.key.startswith(("miniworld_engine::", "cuequivariance"))
        for e in prof.key_averages()
    )


@pytest.mark.parametrize("case", ["rectangular", "head96", "fp32"])
def test_unsupported_dense_preserves_reference(case):
    from team_gm.modules.checkpoints.dense import Attention

    dim = 96 if case == "head96" else 32
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    torch.manual_seed(5)
    source = Attention(128, 128, 128, dim, 4, zero_init=False)
    ref = configure_model(copy.deepcopy(source), "pytorch", dtype=dtype, device="cuda")
    model = configure_model(source, "miniworld", dtype=dtype, device="cuda")
    q = torch.randn(2, 32, 128, device="cuda", dtype=dtype)
    k = torch.randn(
        2, 128 if case == "rectangular" else 32, 128, device="cuda", dtype=dtype
    )
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        actual = model(q, k)
        expected = ref(q, k)
    assert not any(
        e.key == "miniworld_engine::augmented_attention_fwd"
        for e in prof.key_averages()
    )
    if case == "fp32":
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        # Rectangular/head96 attention stays reference; its compatible output
        # gate+projection now uses the engine and has BF16 rounding differences.
        compare("reference-core-fused-output-" + case, actual, expected)


def test_cueq_attention_without_pairformer():
    from unittest.mock import patch

    from cuequivariance_torch.primitives.triangle import triangle_attention
    from team_gm.modules.checkpoints.triangle import TriangleAttention

    torch.manual_seed(17)
    source = TriangleAttention(128, 32, 4, starting=False)
    with torch.no_grad():
        for p in source.parameters():
            p.normal_(0, 0.07)
    ref = configure_model(copy.deepcopy(source), "pytorch", device="cuda")
    model = configure_model(source, "cuequivariance", device="cuda")
    x = torch.randn(2, 128, 128, 128, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(2, 128, 128, device="cuda") > 0.2).float()
    with (
        torch.no_grad(),
        patch(
            "cuequivariance_torch.primitives.triangle.triangle_attention",
            wraps=triangle_attention,
        ) as call,
    ):
        actual = model(x, mask, chunk_size=16)
        assert call.call_count == 1
        expected = ref(x, mask, chunk_size=16)
    compare("cueq-outside-pairformer", actual, expected)
