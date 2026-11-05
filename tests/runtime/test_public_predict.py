"""Library requests, in-memory output and CLI adaptation share one execution API."""

# ruff: noqa: SLF001 - verify the isolated engine compatibility bridge

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import torch
from click.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Iterator

import foldforge
from foldforge.cli import app
from foldforge.models.io import runtime
from foldforge.models.io.options import parse
from foldforge.models.io.output import Decoded
from foldforge.prediction import Prediction


@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    from foldforge import api
    from foldforge.data import ccd
    from foldforge.models import execution, loading

    monkeypatch.setattr(api, "_resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "test-weights.bin"
    weights.touch()
    database = tmp_path / "ccd"
    database.mkdir()
    db = SimpleNamespace(root=database, activate=nullcontext, lookup={})
    monkeypatch.setattr(ccd, "CCDDatabase", lambda _: db)
    monkeypatch.setattr(runtime, "CCDDatabase", lambda _: db)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        runtime, "measured_forward", lambda forward, _repeats: (forward(), {})
    )
    monkeypatch.setattr(
        loading,
        "load_configured",
        Mock(side_effect=lambda *_args, **_kwargs: torch.nn.Module()),
    )
    monkeypatch.setattr(
        execution,
        "Execution",
        Mock(
            side_effect=lambda *_args: SimpleNamespace(
                begin_request=lambda: None, report=lambda: {"test_execution": True}
            )
        ),
    )
    monkeypatch.setattr(runtime, "Execution", execution.Execution)
    monkeypatch.setattr(runtime, "bind_sampling_seed", Mock())
    requests = []

    def prepare(request, _database, engine) -> Iterator[runtime.Case]:
        requests.append(request)
        engine.bind(engine.load_model())

        def forward() -> torch.Tensor:
            return torch.rand(request.samples, 3, 3)

        def decode(value, _measurements) -> Decoded:
            prediction = Prediction(value, plddt=torch.full((request.samples, 2), 0.9))
            return Decoded("example", prediction, ["data_example\n"] * request.samples)

        yield runtime.Case(
            forward, decode, features={"a": torch.ones(1)}, feature_name="features.pt"
        )

    monkeypatch.setitem(
        sys.modules, "foldforge.models.io.dense_atoms", SimpleNamespace(prepare=prepare)
    )
    return {"checkpoint": weights, "ccd_db": database}, requests


def test_predict_returns_cpu_results_without_artifacts_and_preserves_inputs(
    fake_pipeline, tmp_path, capsys
):
    options, requests = fake_pipeline
    inputs = foldforge.protein("AG", name="example")
    config = {"trunk_seed": 17, "trunk": {"recycles": 2}}
    original = deepcopy((inputs, config))
    before = set(tmp_path.rglob("*"))
    state = torch.random.get_rng_state()
    result = foldforge.predict(inputs, config, samples=2, **options)
    assert isinstance(result, foldforge.PredictionResult)
    assert result.coords.shape == (2, 3, 3)
    assert result.coords.device.type == "cpu"
    assert result.output_dir is None
    assert "prediction_cifs" not in result.report
    assert result.report["confidence_units"]["plddt"] == "0..1"
    assert set(tmp_path.rglob("*")) == before
    assert torch.equal(torch.random.get_rng_state(), state)
    assert (inputs, config) == original
    assert requests[0].out is None
    assert capsys.readouterr().out == ""
    second = foldforge.predict(inputs, config, samples=2, **options)
    torch.testing.assert_close(result.coords, second.coords, rtol=0, atol=0)
    saved = result.save("saved")
    assert saved.output_dir == tmp_path / "saved"
    assert len(saved.report["prediction_cifs"]) == 2
    assert all(Path(p).is_file() for p in saved.report["prediction_cifs"])
    assert result.output_dir is None


def test_predict_explicit_output_writes_same_artifacts(fake_pipeline, tmp_path):
    options, _ = fake_pipeline
    result = foldforge.predict(foldforge.protein("AG"), out="result", **options)
    assert result.output_dir == tmp_path / "result"
    assert (result.output_dir / "model.config.json").is_file()
    assert (result.output_dir / "features.pt").is_file()
    assert (result.output_dir / "example.prediction.pt").is_file()


def test_bad_checkpoint_does_not_create_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        foldforge.predict(
            foldforge.protein("AG"), checkpoint=tmp_path / "missing", out="unused"
        )
    assert not (tmp_path / "unused").exists()


def test_cli_help_and_model_metadata_use_no_gpu_imports():
    code = (
        "import sys; from foldforge.cli import main; main(['fold','--help']); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(  # noqa: S603 - fixed trusted Python code
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--spec" in result.stdout
    assert "--input" in result.stdout
    assert "--no-compile" in result.stdout
    listing = CliRunner().invoke(app, ["models", "--json"])
    assert listing.exit_code == 0
    assert len(json.loads(listing.output)) == 12


def test_cli_only_explicit_flags_override_config(tmp_path):
    from foldforge.models.inference import resolve_cli_config

    config = tmp_path / "config.yaml"
    config.write_text(
        "execution:\n  compile: true\n  cuda_graph: true\ntrunk:\n  recycles: 7\n"
    )
    arguments = parse("af3", ["--spec", "input.yaml", "--config", str(config)])
    resolved = resolve_cli_config(arguments)
    assert resolved.execution.compile
    assert resolved.execution.cuda_graph
    assert resolved.trunk.recycles == 7
    arguments = parse(
        "af3",
        [
            "--spec",
            "input.yaml",
            "--config",
            str(config),
            "--no-compile",
            "--recycles",
            "3",
        ],
    )
    resolved = resolve_cli_config(arguments)
    assert not resolved.execution.compile
    assert resolved.execution.cuda_graph
    assert resolved.trunk.recycles == 3


def test_cli_calls_public_api_and_prints_report(monkeypatch):
    from foldforge.models import inference

    seen = []
    monkeypatch.setattr(inference, "load", lambda _: "resolved-input")

    def predict(inputs, config, **options: Any) -> SimpleNamespace:
        seen.append((inputs, config, options))
        return SimpleNamespace(report={"test": "ok"})

    monkeypatch.setattr(inference, "predict", predict)
    result = CliRunner().invoke(
        app, ["fold", "af3", "--spec", "input.yaml", "--out", "demo"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"test": "ok"}
    assert seen[0][0] == "resolved-input"
    assert seen[0][2]["model"] == "af3"


@pytest.mark.parametrize("fails", [False, True])
def test_library_restores_engine_and_precision_after_execution(
    fake_pipeline, monkeypatch, fails
):
    from miniworld_engine import settings

    options, _ = fake_pipeline
    original_policy = settings.current()
    original_precision = torch.get_float32_matmul_precision()

    def execute(_request, **_kwargs: Any) -> tuple:
        torch.set_float32_matmul_precision("high")
        settings.configure(run_autotune=not original_policy.run_autotune)
        if fails:
            message = "execution failed"
            raise RuntimeError(message)
        return (SimpleNamespace(),)

    monkeypatch.setattr(runtime, "predict", execute)
    if fails:
        with pytest.raises(RuntimeError, match="execution failed"):
            foldforge.predict(foldforge.protein("AG"), **options)
    else:
        foldforge.predict(foldforge.protein("AG"), **options)
    assert settings.current() == original_policy
    assert torch.get_float32_matmul_precision() == original_precision


def test_engine_cache_override_restores_root_and_loaded_entries(tmp_path, monkeypatch):
    from miniworld_engine.autotune import cache

    from foldforge.modules.engine.cache import cache_directory

    original_root = cache._CACHE_ROOT
    original_loaded = {("op", "gpu"): {"original": True}}
    monkeypatch.setattr(cache, "_load_cache", original_loaded.copy())

    def fail_inside_override() -> None:
        with cache_directory(tmp_path):
            assert tmp_path == cache._CACHE_ROOT
            assert cache._load_cache == {}
            cache._load_cache[("op", "gpu")] = {"temporary": True}
            message = "failed"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="failed"):
        fail_inside_override()
    assert original_root == cache._CACHE_ROOT
    assert cache._load_cache == original_loaded


def test_mapping_msa_path_does_not_acquire_conflicting_inline_default():
    from foldforge.api import _normalize_payload

    payload = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "AG", "unpairedMsaPath": "msa.a3m"}}
        ]
    }
    normalized = _normalize_payload(payload)
    assert "unpairedMsa" not in normalized["sequences"][0]["protein"]
    assert normalized["sequences"][0]["protein"]["pairedMsa"] == ""


def test_loaded_models_reuse_weights_and_independent_inputs(fake_pipeline):
    from foldforge.models import execution, loading

    options, requests = fake_pipeline
    first = foldforge.load_inputs(foldforge.protein("AG", name="first"))
    second = foldforge.load_inputs(foldforge.protein("GA", name="second"))
    snapshot = deepcopy(first.payload)
    weights = options["checkpoint"]
    database = options["ccd_db"]
    a = foldforge.load_model("af3", checkpoint=weights)
    b = foldforge.load_model("boltz2", checkpoint=weights)
    assert loading.load_configured.call_count == 2
    assert execution.Execution.call_count == 2
    for model, inputs in ((a, first), (b, first), (a, second), (a, first)):
        result = model(inputs, ccd_db=database)
        assert result.coords.device.type == "cpu"
    assert loading.load_configured.call_count == 2
    assert execution.Execution.call_count == 2
    assert first.payload == snapshot
    assert [r.model for r in requests] == ["af3", "boltz2", "af3", "af3"]
    config = a.config
    config.engine.ops["transition"] = "pytorch"
    assert a.config.engine.ops == {}
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a(first)
    b.close()


def test_loaded_json_keeps_origin_and_survives_source_removal(tmp_path):
    path = tmp_path / "input.json"
    payload = foldforge.protein("AG")
    path.write_text(json.dumps(payload))
    loaded = foldforge.load_inputs(path)
    path.unlink()
    assert loaded.origin == path
    assert loaded.payload["sequences"][0]["protein"]["sequence"] == "AG"
    assert loaded.payload["sequences"][0]["protein"]["id"] == "A"
    assert loaded.payload["name"] == payload["name"]
    assert foldforge.load_inputs(loaded) is loaded


def test_predictor_context_closes_after_failure(fake_pipeline):
    options, _ = fake_pipeline
    model = foldforge.load_model(checkpoint=options["checkpoint"])

    def fail() -> None:
        with model:
            message = "application failed"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="application failed"):
        fail()
    with pytest.raises(RuntimeError, match="closed"):
        model(foldforge.protein("AG"), ccd_db=options["ccd_db"])
