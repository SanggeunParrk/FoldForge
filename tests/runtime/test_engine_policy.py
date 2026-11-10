"""The guard in front of every non-pytorch engine call, the plan that picks backends, and the engine stacks inside a CUDA graph."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foldforge.modules.engine import modules, policy
from foldforge.modules.engine.backends import BackendPlan, BackendUnsupportedError


def _kernel_backend(module: nn.Module) -> str:
    return next(name for name in module.backends if name in ("cuda", "triton"))


def test_a_refused_call_runs_the_pytorch_backend_and_logs_once(caplog):
    policy.forget_logged_refusals()
    layer = modules.transition(16, 2)
    x = torch.randn(1, 4, 16)
    expected = layer(x)
    layer.set_backend(_kernel_backend(layer))  # CPU tensors: the kernel backends refuse them
    policy.guard(layer, "fallback")
    with caplog.at_level(logging.WARNING, logger="foldforge.backend"):
        first, second = layer(x), layer(x)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert layer.backend == _kernel_backend(layer)  # the chosen backend is restored after the call
    assert sum("refuses a call" in record.getMessage() for record in caplog.records) == 1


def test_the_error_policy_raises_the_engine_reason():
    layer = modules.transition(16, 2)
    layer.set_backend(_kernel_backend(layer))
    policy.guard(layer, "error")
    with pytest.raises(BackendUnsupportedError, match="Transition"):
        layer(torch.randn(1, 4, 16))


def test_the_plan_gives_each_op_module_a_backend_and_keeps_composites_on_pytorch():
    model = nn.Module()
    model.block = modules.transition(16, 2)
    model.norm = modules.layer_norm(16)
    model.plain = nn.Linear(2, 2)
    plan = BackendPlan.make("miniworld", None, "fallback")
    report = policy.apply_plan(model, plan, torch.device("cpu"))
    assert set(report) == {"block", "norm"}  # only engine op modules are in the plan
    assert {entry["backend"] for entry in report.values()} == {"pytorch"}  # no CUDA device: nothing kernel-served
    assert report["block"]["op"] == "transition"


def test_the_pairformer_block_stays_on_pytorch_and_its_children_are_planned_alone():
    from foldforge.modules.engine.pairformer import pairformer_block

    model = nn.Module()
    model.block = pairformer_block(d_pair=16, triangle_model="af3", d_single=32, n_head_single=2, n_head_tri_attention=2,
                                   d_hidden_tri_attention=8)
    report = policy.apply_plan(model, BackendPlan.make("pytorch", None, "fallback"), torch.device("cpu"))
    assert model.block.backend == "pytorch"
    assert {"block.tri_multi_outgoing", "block.tri_atten_ending", "block.transition_pair", "block.single_attention"} <= set(report)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compile_", [False, True])
def test_the_token_dit_stack_graph_replays_the_eager_result(compile_):
    from foldforge.models.execution import ExecutedCallable
    from foldforge.modules.engine.configure import configure_model
    from foldforge.modules.engine.transformers import token_dit_stack

    torch.manual_seed(7)
    stack = token_dit_stack("alphafold3", n_blocks=4, d_single=768, d_cond=384, d_pair=128, n_head=16)
    with torch.no_grad():
        for name, p in stack.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
    stack = configure_model(stack, "miniworld", torch.bfloat16, "cuda")
    policy.apply_plan(stack, BackendPlan.make("miniworld", None, "fallback"), torch.device("cuda"))
    length = 256
    act = torch.randn(2, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(2, length, 384, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(length, device="cuda") > 0.15
    key_mask = mask[None]

    def step(x: torch.Tensor) -> torch.Tensor:
        return stack(x, mask, single, pair, key_mask=key_mask)

    policy_ = SimpleNamespace(compile=compile_, cuda_graph=True, max_graphs=2)
    graphed = ExecutedCallable(step, policy_, "test.denoiser")
    with torch.no_grad():
        expected = step(act.clone())
        first = graphed(act.clone()).clone()
        second = graphed(act.clone()).clone()
    assert graphed.captures == 1
    assert graphed.replays == 2
    torch.testing.assert_close(first, expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
