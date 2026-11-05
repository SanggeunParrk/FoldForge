"""Shared policy retains legacy eligibility and distinguishes checks from execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from foldforge.layers.implementation import ImplementationType
from foldforge.modules.engine.backends import (
    BackendUnsupportedError,
    Support,
    forget_logged_refusals,
)
from foldforge.modules.engine.policy import accept_support, miniworld_support, refused
from foldforge.modules.engine.serving import report_observations


@pytest.mark.parametrize(
    "backend",
    [ImplementationType.MINIWORLD_ENGINE, ImplementationType.PYTORCH, "anthropic"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_activation_policy_matches_legacy_predicates(backend, dtype, cuda, tf32, fp32):
    module = nn.Module()
    module.foldforge_implementation = backend
    module.foldforge_tf32 = tf32
    activation = SimpleNamespace(is_cuda=cuda, dtype=dtype)
    expected = (
        backend == ImplementationType.MINIWORLD_ENGINE
        and cuda
        and (dtype == torch.float32 and tf32 if fp32 else dtype == torch.bfloat16)
    )
    assert miniworld_support(module, activation, fp32=fp32).accepted == expected


def test_policy_records_refusals_even_when_strict_raises(caplog):
    module = nn.Module()
    module.foldforge_unsupported = "fallback"
    forget_logged_refusals()
    for _ in range(2):
        assert refused(module, "transition", "anthropic", "shape refused")
    assert sum("shape refused" in r.getMessage() for r in caplog.records) == 1
    module.foldforge_unsupported = "error"
    with pytest.raises(BackendUnsupportedError, match="shape refused"):
        refused(module, "transition", "anthropic", Support(refusal="shape refused"))
    # Eligibility-only fallbacks remain silent even with error policy.
    assert not accept_support(module, Support(refusal="not_installed"))
    entry = report_observations(module)["modules"]["<root>"]
    assert entry["refused_host_checks"] == {"shape refused": 3, "not_installed": 1}
    assert entry["served_host_calls"] == 0


def test_compiled_policy_does_not_mutate_host_counters():
    module = nn.Module()
    support = Support()
    accept_support(module, support)
    before = report_observations(module)

    def call(x: torch.Tensor) -> torch.Tensor:
        return x + 1 if accept_support(module, support) else x

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.tensor(2.0)), torch.tensor(3.0))
    assert report_observations(module) == before


def test_projected_anthropic_grad_refusal_preserves_error_policy():
    from foldforge.modules.engine.attention_ops import anthropic_attention

    module = nn.Module()
    q = torch.ones(2, 1, 2, 4)
    bias = torch.ones(1, 2, 2)
    with torch.enable_grad():
        assert anthropic_attention(module, q, q, q, bias, None) is None
        module.foldforge_unsupported = "error"
        with pytest.raises(BackendUnsupportedError, match="grad is enabled"):
            anthropic_attention(module, q, q, q, bias, None)


@pytest.mark.parametrize("route", ["cuequivariance", "anthropic", "pytorch"])
def test_projected_attention_keeps_route_priority(monkeypatch, route):
    import sys

    from foldforge.modules.engine import attention_ops

    module = nn.Module()
    module.qkv_dim = 4
    module.foldforge_implementation = route
    calls = []
    q = torch.ones(2, 1, 2, 4)
    bias, mask = torch.ones(1, 2, 2), torch.tensor([[1, 0], [1, 0]])

    def vendor(*args: torch.Tensor, **options: Any) -> torch.Tensor:
        calls.append("cuequivariance")
        assert options["mask"].shape == (1, 2, 1, 1, 2)
        return args[0] + 2

    def anthropic(*_: object) -> torch.Tensor:
        calls.append("anthropic")
        return q + 3

    monkeypatch.setattr(
        attention_ops,
        "backend_support",
        lambda m, b, _x: Support(
            refusal=None if m.foldforge_implementation == b else "backend_not_selected"
        ),
    )
    monkeypatch.setattr(attention_ops, "anthropic_attention", anthropic)
    monkeypatch.setitem(
        sys.modules, "cuequivariance_torch", SimpleNamespace(triangle_attention=vendor)
    )
    result = attention_ops.serve_projected_triangle(module, q, q, q, bias, mask)
    if route == "pytorch":
        assert result is None
        assert calls == []
    else:
        assert calls == [route]
        torch.testing.assert_close(
            result, q + {"cuequivariance": 2, "anthropic": 3}[route]
        )


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("activation", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("condition", [torch.bfloat16, torch.float32])
def test_conditioning_policy_preserves_operand_compatibility(
    cuda, activation, condition
):
    from foldforge.modules.engine.conditioning import conditioning_support

    layer = SimpleNamespace(
        to_scale=SimpleNamespace(weight=SimpleNamespace(dtype=torch.bfloat16))
    )
    a, s = (
        SimpleNamespace(is_cuda=cuda, dtype=activation),
        SimpleNamespace(dtype=condition),
    )
    expected = cuda and activation == torch.bfloat16 and condition == torch.bfloat16
    assert conditioning_support(layer, a, s).accepted == expected
