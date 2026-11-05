"""Candidate failures, legacy APIs and attachment evidence stay distinct."""

# ruff: noqa: SLF001 - install candidate contracts
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foldforge.modules.engine import modules
from foldforge.modules.engine.backends import BackendPlan
from foldforge.modules.engine.installation import (
    BuildResult,
    Candidate,
    installation_report,
)


class DiffusionTransformer(nn.Module):
    """Only the installer's structural metadata, with no inference kernels."""

    def __init__(self) -> None:
        super().__init__()
        self.num_blocks = 2
        self.c_single_cond = 384
        self.weight = nn.Parameter(torch.ones(1))


def test_norm_mismatch_has_a_named_reason():
    like = torch.empty(0, dtype=torch.bfloat16)
    triangle = SimpleNamespace(
        c_pair=4,
        center_norm=SimpleNamespace(weight=torch.ones(4), eps=1e-3),
        left_norm_input=SimpleNamespace(eps=1e-5),
    )
    assert (
        modules._build_triangle_multiplication(triangle, like).refusal
        == "norm_eps_mismatch"
    )


def test_failed_primary_then_variant_records_both_and_preserves_state_dict(monkeypatch):
    root = DiffusionTransformer()
    before = set(root.state_dict())
    twin = nn.Linear(1, 1)
    candidates = (
        Candidate(
            "engine_dit_blocks",
            lambda *_: BuildResult(refusal="block_structure_mismatch"),
        ),
        Candidate(
            "engine_dit_variant_blocks", lambda *_: BuildResult(value=twin), fp32=False
        ),
    )
    monkeypatch.setattr(
        modules,
        "_installation_candidates",
        lambda: {"DiffusionTransformer": ("dit_blocks", candidates)},
    )
    counts = modules.install_engine_modules(root, torch.empty(0, dtype=torch.bfloat16))
    row = installation_report(root)["modules"]["<root>"]
    assert counts["engine_dit_variant_blocks"] == 2
    assert row["attached_units"] == 2
    assert row["status"] == "attached"
    assert row["attempts"][0]["reason"] == "block_structure_mismatch"
    assert row["attempts"][1]["status"] == "built"
    assert modules.engine_of(root) is twin
    assert set(root.state_dict()) == before
    json.dumps(installation_report(root), allow_nan=False)


@pytest.mark.parametrize("preexisting", [False, True])
def test_skip_keeps_legacy_candidate_count_but_never_claims_attachment(
    monkeypatch, preexisting
):
    root = DiffusionTransformer()
    old = nn.Linear(1, 1)
    if preexisting:
        modules.attach(root, old)
    calls = []

    def primary(*_: object) -> BuildResult:
        calls.append("primary")
        return BuildResult(value=[nn.Linear(1, 1), nn.Linear(1, 1)])

    def variant(*_: object) -> BuildResult:
        pytest.fail("a skipped but constructed primary must still stop the chain")

    monkeypatch.setattr(
        modules,
        "_installation_candidates",
        lambda: {
            "DiffusionTransformer": (
                "dit_blocks",
                (
                    Candidate("engine_dit_blocks", primary),
                    Candidate("engine_dit_variant_blocks", variant),
                ),
            )
        },
    )
    counts = modules.install_engine_modules(
        root, torch.empty(0, dtype=torch.bfloat16), skip={"engine_dit_blocks"}
    )
    report = installation_report(root)
    row = report["modules"]["<root>"]
    assert calls == ["primary"]
    assert counts["engine_dit_blocks"] == 2
    assert report["attached_units"] == 0
    assert row["status"] == "skipped"
    assert not row["attached"]
    assert row["preexisting_twin"] == preexisting
    assert row["present_after_install"] == preexisting
    assert modules.engine_of(root) is (old if preexisting else None)


@pytest.mark.parametrize("precision", [torch.float32, torch.bfloat16])
def test_precision_and_backend_exclusions_do_not_build_candidates(
    monkeypatch, precision
):
    root = DiffusionTransformer()
    calls = []

    def build(*_: object) -> BuildResult:
        calls.append(1)
        return BuildResult(refusal="structure_mismatch")

    candidate = Candidate("engine_dit_variant_blocks", build, fp32=False)
    monkeypatch.setattr(
        modules,
        "_installation_candidates",
        lambda: {"DiffusionTransformer": ("dit_blocks", (candidate,))},
    )
    modules.install_engine_modules(root, torch.empty(0, dtype=precision))
    row = installation_report(root)["modules"]["<root>"]
    assert calls == ([] if precision == torch.float32 else [1])
    assert row["reason"] == (
        "fp32_variant_disabled" if precision == torch.float32 else "structure_mismatch"
    )
    calls.clear()
    modules.install_engine_modules(
        root, torch.empty(0, dtype=precision), plan=BackendPlan.make("pytorch")
    )
    assert calls == []
    assert (
        installation_report(root)["modules"]["<root>"]["status"]
        == "backend_without_twin"
    )


def test_factory_errors_propagate_without_trying_other_candidates(monkeypatch):
    root = DiffusionTransformer()

    def broken(*_: object) -> BuildResult:
        message = "invalid checkpoint tensor"
        raise ValueError(message)

    monkeypatch.setattr(
        modules,
        "_installation_candidates",
        lambda: {
            "DiffusionTransformer": (
                "dit_blocks",
                (Candidate("engine_dit_blocks", broken),),
            )
        },
    )
    with pytest.raises(ValueError, match="invalid checkpoint tensor"):
        modules.install_engine_modules(root, torch.empty(0, dtype=torch.bfloat16))


@pytest.mark.parametrize("enabled", [False, True])
def test_load_report_distinguishes_disabled_modules_and_skipped_phases(enabled):
    from foldforge.modules.engine.install import install_engine

    root = nn.Module()
    root.foldforge_load_report = {}
    result = install_engine(
        root,
        torch.bfloat16,
        "cpu",
        modules=enabled,
        skip={"pairformers", "conditioning"},
    )
    report = result.foldforge_load_report["engine_installation"]
    assert report["enabled"] == enabled
    assert report["modules"] == {}
    assert report["phases"]["pairformers"]["status"] == "skipped"
    assert report["phases"]["conditioning"]["status"] == "skipped"
    assert not report["is_runtime_dispatch"]


def test_parameter_conversion_phases_record_equation_exclusions():
    from foldforge.modules.engine.conditioning import install_conditioning
    from foldforge.modules.engine.installation import phase_decisions

    norm = type("AdaptiveLayerNorm", (nn.Module,), {})()
    norm.layer_norm = nn.Identity()
    norm.identity_scale = True
    norm.use_single_cond = True
    model = nn.Sequential(norm)
    model.foldforge_load_report = {}
    install_conditioning(model)
    assert model[0] is norm
    assert (
        phase_decisions(model)["conditioning"]["0"]["reason"]
        == "unconditioned_or_identity_scale"
    )
    json.dumps(phase_decisions(model), allow_nan=False)
