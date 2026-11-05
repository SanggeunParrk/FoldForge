"""The one serving gate every dense module with an engine twin goes through."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from foldforge.modules.engine import transformers
from foldforge.modules.engine.modules import attach
from foldforge.modules.engine.routing import served_by
from foldforge.modules.engine.serving import ADAPTER_ATTR, observe, report_observations
from foldforge.modules.engine.transformers import serve_atom, serve_token


class _Twin(nn.Module):
    """A dense module with the attribute the gate reads, and nothing else."""

    foldforge_implementation = None


def test_a_module_without_an_engine_twin_is_never_served():
    assert served_by(_Twin(), torch.zeros(2)) is None


def test_the_gate_needs_a_bf16_cuda_activation_and_the_engine_backend():
    twin = _Twin()
    attach(twin, object())
    # No backend requested (the PyTorch reference path), and a CPU float activation.
    assert served_by(twin, torch.zeros(2)) is None
    assert served_by(twin, torch.zeros(2, dtype=torch.bfloat16)) is None


def test_an_unattached_transformer_hands_back_to_the_dense_blocks():
    """None is the dense blocks' cue."""
    twin = _Twin()
    twin.window_half = None
    act, mask, cond, pair = (torch.zeros(4, 8) for _ in range(4))
    assert serve_token(twin, act, mask, cond, pair, None, None) is None
    atoms = torch.zeros(1, 32, 8)
    args = (twin, atoms, atoms[..., 0], atoms, atoms, None, None, None, None)
    assert serve_atom(*args) is None


def test_the_token_gate_refuses_a_cpu_activation_even_with_a_twin():
    twin = _Twin()
    attach(twin, [object()])
    act, mask, cond, pair = (torch.zeros(4, 8) for _ in range(4))
    assert serve_token(twin, act, mask, cond, pair, None, None) is None


class _Block(nn.Module):
    def forward(self, x, cond, _pair, _mask, **_options: object) -> torch.Tensor:
        return x + cond


class _Stack(nn.Module):
    def forward(self, x, cond, _pair, _mask, _uid=None) -> torch.Tensor:
        return x + cond


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_uniform_token_adapter_preserves_layout_and_raw_twin(
    monkeypatch, stack, batched
):
    monkeypatch.setattr(transformers, "enabled", lambda *_: True)
    monkeypatch.setattr(
        transformers, "variant_selection", lambda *_: "portable_candidate"
    )
    twin = _Twin()
    built = _Stack() if stack else [_Block()]
    attach(twin, built)
    act = torch.randn((2, 4, 2) if batched else (4, 2))
    cond, pair, mask = torch.randn(4, 2), torch.randn(4, 4, 3), torch.ones(4)
    torch.testing.assert_close(
        serve_token(twin, act, mask, cond, pair, None, None), act + cond
    )
    from foldforge.modules.engine.modules import engine_of

    assert engine_of(twin) is built
    assert not twin.state_dict()
    entry = report_observations(twin)["modules"]["<root>"]
    assert entry["adapter"] == ("token_stack" if stack else "token_blocks")
    assert entry["served_host_calls"] == 1


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("samples", [1, 2])
def test_uniform_atom_adapter_preserves_sample_and_window_layout(
    monkeypatch, stack, samples
):
    monkeypatch.setattr(transformers, "variant_selection", lambda *_: "cuda_candidate")
    twin = _Twin()
    twin.window_half = None
    attach(twin, _Stack() if stack else [_Block()])
    act, cond = torch.randn(samples, 2, 2, 2), torch.randn(2, 2, 2)
    pair, mask = torch.randn(2, 2, 4, 3), torch.ones(2, 2)
    actual = serve_atom(twin, act, mask, cond, pair, None, None, None, None)
    torch.testing.assert_close(actual, act + cond)
    entry = report_observations(twin)["modules"]["<root>"]
    assert entry["adapter"] == ("atom_stack" if stack else "atom_blocks")
    assert entry["served_host_calls"] == 1


def test_runtime_report_separates_attachment_execution_and_refusals(monkeypatch):
    from foldforge.models.config import ExecutionConfig
    from foldforge.models.execution import Execution

    monkeypatch.setattr(transformers, "enabled", lambda *_: True)
    monkeypatch.setattr(
        transformers, "variant_selection", lambda *_: "portable_candidate"
    )
    twin = _Twin()
    twin.num_recycles = 10
    attach(twin, _Stack())
    args = (
        twin,
        torch.randn(4, 2),
        torch.ones(4),
        torch.randn(4, 2),
        torch.randn(4, 4, 3),
        None,
    )
    serve_token(*args, None)
    execution = Execution(twin, "af3", ExecutionConfig())
    # Binding resets observations from any earlier calls.
    assert report_observations(twin)["modules"]["<root>"]["served_host_calls"] == 0
    assert not report_observations(twin)["modules"]["<root>"]["host_observed"]
    serve_token(*args, None)
    assert serve_token(*args, torch.zeros(4, 4)) is None
    report = execution.report()
    assert report["trunk_passes"] == 10
    dispatch = report["engine_dispatch"]
    assert not dispatch["selections_are_kernel_traces"]
    entry = dispatch["modules"]["<root>"]
    assert entry["host_observed"]
    assert entry["served_host_calls"] == 1
    assert entry["refused_host_calls"] == {"extra_pair_bias_unsupported": 1}
    assert entry["selections"] == {"portable_candidate": 1}


def test_compiled_calls_do_not_mutate_host_observations(monkeypatch):
    monkeypatch.setattr(transformers, "enabled", lambda *_: True)
    twin = _Twin()
    attach(twin, [_Block()])
    act, cond, pair, mask = (
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 4, 3),
        torch.ones(4),
    )

    serve_token(twin, act, mask, cond, pair, None, None)
    adapter = twin.__dict__[ADAPTER_ATTR]

    def call(x) -> torch.Tensor | None:
        # The existing public inference-mode probe causes a Dynamo graph break.
        # The added adapter and observer must themselves be fullgraph-safe.
        return observe(twin, adapter.try_run(twin, x, mask, cond, pair, None, None))

    before = report_observations(twin)
    compiled = torch.compile(call, backend="eager", fullgraph=True)
    changed = act + 0.3
    torch.testing.assert_close(compiled(changed), changed + cond)
    assert report_observations(twin) == before


def test_support_checks_never_claim_an_engine_execution(monkeypatch):
    from foldforge.modules.engine import routing

    twin = _Twin()
    built = object()
    attach(twin, built)
    monkeypatch.setattr(routing, "enabled", lambda *_: True)
    assert served_by(twin, torch.zeros(2)) is built
    entry = report_observations(twin)["modules"]["<root>"]
    assert entry["attached"]
    assert entry["host_observed"]
    assert entry["supported_host_checks"] == 1
    assert entry["served_host_calls"] == 0
    assert entry["execution_scope"] == "support_checks_only"
    monkeypatch.setattr(routing, "enabled", lambda *_: False)
    assert served_by(twin, torch.zeros(2)) is None
    entry = report_observations(twin)["modules"]["<root>"]
    assert entry["refused_host_checks"] == {"backend_or_dtype": 1}


def test_backend_support_preserves_reason_policy_and_plane_fallback(caplog):
    from types import SimpleNamespace

    from foldforge.modules.engine.backends import (
        BackendUnsupportedError,
        forget_logged_refusals,
    )

    twin = _Twin()
    twin.foldforge_implementation = "anthropic"
    twin.foldforge_unsupported = "fallback"
    attach(
        twin,
        SimpleNamespace(
            foldforge_op="opm", foldforge_refuses=lambda _: "test shape refused"
        ),
    )
    forget_logged_refusals()
    assert served_by(twin, torch.zeros(2)) is None
    assert served_by(twin, torch.zeros(2)) is None
    assert sum("test shape refused" in r.getMessage() for r in caplog.records) == 1
    entry = report_observations(twin)["modules"]["<root>"]
    assert entry["refused_host_checks"] == {"test shape refused": 2}
    twin.foldforge_unsupported = "error"
    with pytest.raises(BackendUnsupportedError, match="test shape refused"):
        served_by(twin, torch.zeros(2))
    attach(twin, SimpleNamespace(foldforge_plane_only=True))
    # This requests per-plane routing, not an unsupported backend error.
    assert served_by(twin, torch.zeros(2, 3, 3, 4)) is None


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("with_single", [False, True])
@pytest.mark.parametrize("samples", [1, 3])
def test_pairformer_adapter_preserves_equations_and_layout(
    parallel, with_single, samples
):
    import copy
    from dataclasses import replace

    from foldforge.modules.dense.pairformer import PairformerBlock
    from foldforge.modules.dense.spec import ALPHAFOLD3
    from foldforge.modules.engine.configure import configure_model
    from foldforge.modules.engine.pairformer import AF3Pairformer

    torch.manual_seed(11)
    source = PairformerBlock(
        n_heads=2,
        c_pair=8,
        c_single=8,
        c_hidden_mul=8,
        n_heads_pair=2,
        with_single=with_single,
        pair_qkv_dim=4,
        spec=replace(ALPHAFOLD3, parallel_pairformer_block=parallel),
    )
    with torch.no_grad():
        for parameter in source.parameters():
            if parameter.ndim > 1:
                parameter.normal_(0, 0.1)
    source = configure_model(source, "pytorch", torch.float32, "cpu").eval()
    wrapper = AF3Pairformer(copy.deepcopy(source))
    keys = set(wrapper.state_dict())
    pair = torch.randn(samples, 5, 5, 8)
    single = torch.randn(samples, 5, 8) if with_single else None
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    pair_mask = mask[:, None] * mask[None, :]
    with torch.no_grad():
        expected = [
            source(
                p.clone(),
                pair_mask,
                None if single is None else single[i].clone(),
                mask,
            )
            for i, p in enumerate(pair)
        ]
        actual = wrapper(
            pair[0] if samples == 1 else pair,
            pair_mask,
            None if single is None else single[0] if samples == 1 else single,
            mask,
        )
    if with_single:
        for index in (0, 1):
            want = (
                expected[0][index]
                if samples == 1
                else torch.stack([v[index] for v in expected])
            )
            torch.testing.assert_close(actual[index], want)
    else:
        want = expected[0] if samples == 1 else torch.stack(expected)
        torch.testing.assert_close(actual, want)
    entry = report_observations(wrapper)["modules"]["<root>"]
    assert entry["adapter"] == "pairformer"
    assert entry["execution_scope"] == "composition"
    assert entry["served_host_calls"] == 1
    assert entry["selections"] == {"engine_composition": 1}
    assert set(wrapper.state_dict()) == keys


def test_pair_mask_conversion_preserves_leading_batch_axes():
    from foldforge.modules.engine.modules import residue_mask, token_mask

    tokens = torch.tensor([[1, 0, 1], [1, 1, 0]])
    pair = tokens[:, :, None] * tokens[:, None, :]
    torch.testing.assert_close(token_mask(pair), tokens.bool())
    torch.testing.assert_close(residue_mask(pair[0]), tokens[:1].bool())
    assert token_mask(None) is None
    assert residue_mask(None) is None
