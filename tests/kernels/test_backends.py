"""The backend policy on top of the engine's ``backends`` tuples: names, aliases, the table derived from the classes, the per-module
choice, and the guard that implements ``engine.unsupported`` (the engine itself never falls back)."""

from __future__ import annotations

import logging

import pytest
import torch
from miniworld_engine.modules import (
    AdaptiveLayerNorm,
    OuterProductMean,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.primitives import BackendUnsupported
from torch import nn

from foldforge.models.config import Config
from foldforge.modules.engine import backends, modules, policy
from foldforge.modules.engine.install import install_engine
from foldforge.modules.engine.serving import report_observations


@pytest.fixture(autouse=True)
def _fresh_log_state():
    policy.forget_logged_refusals()


def test_names_and_aliases():
    assert set(backends.ENGINE_BACKENDS) == {"pytorch", "cuda", "triton", "anthropic", "cueq"}
    for name in backends.BACKENDS:
        assert Config(backend=name).backend == name
    with pytest.raises(ValueError, match="Unknown backend"):
        backends.BackendPlan.make("magma")
    with pytest.raises(ValueError, match="unknown op"):
        backends.validate_ops({"nope": "pytorch"})
    with pytest.raises(ValueError, match="unknown backend"):
        backends.validate_ops({"trimul": "nope"})


def test_serves_is_derived_from_the_engine_classes():
    table = backends.SERVES
    assert set(table) == set(backends.OPS)
    for op, names in table.items():
        have = {b for cls in backends.op_classes(op) for b in cls.backends}
        assert {n for n in names if n in backends.ENGINE_BACKENDS} == have
        assert ("miniworld" in names) == bool(have & {"cuda", "triton"})
        assert ("cuequivariance" in names) == ("cueq" in have)
    assert "anthropic" not in table["conditioning"]
    assert "anthropic" not in table["msa_pwa"]


def test_choose_backend(monkeypatch):
    # no CUDA: the engine's own kernels are not available, miniworld means the reference
    monkeypatch.setattr(backends, "capability", lambda _d: None)
    assert backends.choose_backend("miniworld", Transition) == ("pytorch", None)
    assert backends.choose_backend("pytorch", Transition) == ("pytorch", None)
    name, why = backends.choose_backend("cuda", Transition)
    assert name == "pytorch"
    assert "H100" in why or "CUDA" in why
    # H100: cuda where the class has it, pytorch for a class that has none
    monkeypatch.setattr(backends, "capability", lambda _d: (9, 0))
    assert backends.choose_backend("miniworld", Transition) == ("cuda", None)
    assert backends.choose_backend("miniworld", AdaptiveLayerNorm) == ("pytorch", None)
    assert backends.choose_backend("cueq", TriangleMultiplication) == ("cueq", None)
    assert backends.choose_backend("cuequivariance", TriangleAttention) == ("cueq", None)
    name, why = backends.choose_backend("anthropic", OuterProductMean)
    assert name == "pytorch"
    assert "no 'anthropic' backend" in why
    # another card: triton, and cuda is refused by name
    monkeypatch.setattr(backends, "capability", lambda _d: (8, 0))
    assert backends.choose_backend("miniworld", Transition) == ("triton", None)
    assert backends.choose_backend("cuda", Transition)[1] == "the cuda backend is H100 only (sm_90)"


def test_plan_resolve_reports_refused_ops(monkeypatch, caplog):
    monkeypatch.setattr(backends, "capability", lambda _d: (9, 0))
    plan = backends.BackendPlan.make("pytorch", {"transition": "cuda", "msa_pwa": "anthropic"})
    with caplog.at_level(logging.WARNING, logger="foldforge.backend"):
        plan.resolve()
    assert plan.of("transition") == "cuda"
    assert plan.resolved["msa_pwa"].reason is not None
    assert plan.report()["fallbacks"].keys() == {"msa_pwa"}
    assert sum("msa_pwa" in record.message for record in caplog.records) == 1
    strict = backends.BackendPlan.make("pytorch", {"msa_pwa": "anthropic"}, "error")
    with pytest.raises(backends.BackendUnsupportedError):
        strict.resolve()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transition = modules.transition(16, 2, precision="fp32")
        self.trimul = modules.triangle_multiplication(16, outgoing=True, precision="fp32")
        self.norm = modules.layer_norm(16)


def _x(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 4, 16)


def test_apply_plan_sets_the_backend_per_module_and_guards(monkeypatch):
    monkeypatch.setattr(backends, "capability", lambda _d: (9, 0))
    model = _Model()
    plan = backends.BackendPlan.make("miniworld", {"trimul": "pytorch"})
    choices = policy.apply_plan(model, plan)
    assert model.transition.backend == "cuda"
    assert model.norm.backend == "cuda"
    assert model.trimul.backend == "pytorch"
    assert choices["transition"]["backend"] == "cuda" and choices["trimul"]["op"] == "trimul"


def test_unsupported_call_falls_back_once_and_logs_once(monkeypatch, caplog):
    monkeypatch.setattr(backends, "capability", lambda _d: (9, 0))
    model = _Model()
    expected = Transition.forward(model.transition, _x())  # pytorch backend of the same weights
    policy.apply_plan(model, backends.BackendPlan.make("cuda"))
    assert model.transition.backend == "cuda"
    # a CPU tensor is outside the cuda backend's range: the engine raises, FoldForge's guard falls back
    assert model.transition.unsupported(_x()) is not None
    with pytest.raises(BackendUnsupported):
        Transition.forward(model.transition, _x())
    with caplog.at_level(logging.WARNING, logger="foldforge.backend"):
        first = model.transition(_x())
        second = model.transition(_x())
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert model.transition.backend == "cuda"  # restored after the call
    assert sum("refuses a call" in record.message for record in caplog.records) == 1
    stats = report_observations(model)["modules"]["transition"]
    assert stats["served_host_calls"] == {"pytorch": 2}
    assert len(stats["refused_host_calls"]) == 1


def test_unsupported_error_policy_raises(monkeypatch):
    monkeypatch.setattr(backends, "capability", lambda _d: (9, 0))
    model = _Model()
    policy.apply_plan(model, backends.BackendPlan.make("cuda", unsupported="error"))
    with pytest.raises(backends.BackendUnsupportedError, match="engine.unsupported: error"):
        model.transition(_x())


def test_composite_children_are_refused_one_by_one(monkeypatch):
    """A PairformerBlock stays on pytorch itself; each child has its own backend, so one child without a kernel does not take the block."""
    from foldforge.modules.engine.pairformer import pairformer_block

    monkeypatch.setattr(backends, "capability", lambda _d: (9, 0))
    block = pairformer_block(d_pair=32, triangle_model="af3", d_single=64, n_head_single=4, precision="fp32")
    holder = nn.ModuleList([block])
    policy.apply_plan(holder, backends.BackendPlan.make("miniworld"))
    assert block.backend == "pytorch"
    assert block.tri_multi_outgoing.backend == "cuda"
    assert block.transition_pair.backend == "cuda"
    assert block.single_attention.backend == "cuda"


def test_install_engine_without_cuda_modules_runs_pytorch():
    model = _Model()
    install_engine(model, "cpu", modules=False)
    assert {m.backend for m in (model.transition, model.trimul, model.norm)} == {"pytorch"}
    report = model.foldforge_load_report["engine_backends"]
    assert report["plan"]["backend"] == "pytorch"
    assert set(report["modules"]) == {"transition", "trimul", "norm"}
