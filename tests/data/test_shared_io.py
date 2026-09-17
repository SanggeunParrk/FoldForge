"""Cross-checkpoint input identity, output units, axes and execution ownership."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
import torch

import foldforge
from foldforge.models.config import ExecutionConfig
from foldforge.models.io.cli import parse
from foldforge.models.io.output import Decoded, write_output
from foldforge.models.io.request import Request
from foldforge.prediction import Prediction

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_feature_transfer_preserves_source_and_nested_container_types():
    from foldforge.models.io.runtime import to_device

    source = {"nested": [torch.ones(2), (torch.zeros(3), None)], "name": "ligand"}
    result = to_device(source, "meta")
    assert source["nested"][0].device.type == "cpu"
    assert result["nested"][0].device.type == "meta"
    assert result["nested"][1][0].device.type == "meta"
    assert isinstance(result["nested"][1], tuple)
    assert result["nested"][1][1] is None
    assert result["name"] == "ligand"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coords": torch.zeros(1, 4, 2)},
        {"coords": torch.zeros(0, 4, 3)},
        {"plddt": torch.zeros(2, 4)},
        {"pae": torch.zeros(1, 4, 3)},
        {"ptm": torch.zeros(1, 1)},
        {"iptm": torch.zeros(2)},
        {"plddt": torch.zeros(1, 4), "pae": torch.zeros(1, 3, 3)},
        {"plddt": torch.zeros(1, 4), "distogram_logits": torch.zeros(3, 3, 8)},
    ],
)
def test_prediction_rejects_inconsistent_axes(kwargs):
    with pytest.raises(ValueError, match=r"axes|sample|square"):
        Prediction(**{"coords": torch.zeros(1, 4, 3), **kwargs})


def test_multisample_output_keeps_every_sample_and_absent_heads(tmp_path):
    prediction = Prediction(
        torch.arange(24.0).reshape(2, 4, 3), plddt=torch.full((2, 3), 0.7)
    )
    report = write_output(
        Decoded("complex", prediction, ["data_first\n", "data_second\n"]),
        tmp_path,
        expected_samples=2,
    )
    payload = torch.load(tmp_path / "complex.prediction.pt", weights_only=True)
    torch.testing.assert_close(payload["coords"], prediction.coords, atol=0, rtol=0)
    assert payload["pae"] is None
    assert payload["iptm"] is None
    assert [Path(p).read_text() for p in report["prediction_cifs"]] == [
        "data_first\n",
        "data_second\n",
    ]
    assert report["confidence_units"]["plddt"] == "0..1"


@pytest.mark.parametrize(
    "error", ["coords", "iptm", "distogram_logits", "scale", "samples", "name", "json"]
)
def test_invalid_output_fails_before_creating_prediction_artifacts(tmp_path, error):
    kwargs = {"coords": torch.zeros(1, 4, 3)}
    if error in {"coords", "iptm", "distogram_logits"}:
        shape = {"coords": (1, 4, 3), "iptm": (1,), "distogram_logits": (3, 3, 8)}[
            error
        ]
        kwargs[error] = torch.full(shape, float("nan"))
    if error == "scale":
        kwargs["plddt"] = torch.full((1, 3), 70.0)
    result = Decoded(
        "../escape" if error == "name" else "sample",
        Prediction(**kwargs),
        ["data_test\n"],
    )
    if error == "json":
        result.report["invalid"] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        write_output(result, tmp_path, expected_samples=2 if error == "samples" else 1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", ["af3", "esmfold2", "protenix", "opendde"])
def test_legacy_flags_have_one_execution_contract(model, tmp_path):
    args = [
        "--compile",
        "--cuda-graph",
        "--benchmark-repeats",
        "0",
        "--out",
        str(tmp_path),
    ]
    if model == "esmfold2":
        args += ["--dtype", "bfloat16", "--implementation", "cuequivariance"]
    else:
        args += [
            "--input",
            "input.json",
            "--checkpoint",
            "weights.pt",
            "--backend",
            "cuequivariance",
        ]
    request = parse(model, args)
    assert request.precision == "bf16"
    assert request.backend == "cuequivariance"
    assert request.execution.compile
    assert request.execution.cuda_graph
    assert request.execution.benchmark_repeats == 0


def test_resolved_input_and_msa_limit_reach_runtime_without_reparse(
    tmp_path, monkeypatch
):
    from foldforge.models import inference
    from foldforge.models.io import runtime

    spec = SimpleNamespace(
        template={},
        ccd_db=tmp_path,
        name="target",
        save_trajectory=False,
        n_trunk_samples=1,
        diffusion_batch_size=1,
        n_diffusion_samples=1,
        model_dump_json=lambda **_kw: "{}",
    )
    original = SimpleNamespace(spec=spec, chains=[])
    prepared = SimpleNamespace(spec=spec, chains=[], msa_species={}, resources=list)
    reader = Mock(return_value=original)
    limiter = Mock(return_value=prepared)
    predict = Mock(return_value=0)
    monkeypatch.setattr(inference, "load", reader)
    monkeypatch.setattr(inference, "limit_msa", limiter)
    monkeypatch.setattr(
        inference, "CCDDatabase", lambda path: SimpleNamespace(root=path)
    )
    monkeypatch.setattr(runtime, "run", predict)
    config = tmp_path / "config.yaml"
    config.write_text(
        "backend: pytorch\ntrunk:\n  msa_depth: 3\nexecution:\n  compile: "
        "true\n  cuda_graph: true\n"
    )
    inference.run(
        "esmfold2",
        ["--spec", "original.yaml", "--config", str(config), "--out", str(tmp_path)],
    )
    reader.assert_called_once()
    assert limiter.call_args.args[0] is original
    request = predict.call_args.args[0]
    assert request.resolved_input is prepared
    assert request.input_spec is None
    assert request.msa_depth == 3
    assert request.execution.compile
    assert request.execution.cuda_graph


def test_runtime_owns_forward_count_and_every_target_output(tmp_path, monkeypatch):
    from foldforge.models.io import runtime

    request = Request(
        model="esmfold2",
        ccd_db=tmp_path,
        out=tmp_path,
        execution=ExecutionConfig(benchmark_repeats=0),
    )
    executions, calls = [], []

    def measure(forward, repeats) -> tuple[Any, dict[str, float | None]]:
        calls.append(repeats)
        return forward(), {"model_seconds_cold": 0.1, "model_seconds_warm_median": None}

    def prepare(args, database, session) -> Iterator[runtime.Case]:  # noqa: ARG001 - shared callback or fixture signature
        session.bind(object())
        for i in range(2):
            yield runtime.Case(
                lambda i=i: i,
                lambda value, _measured: Decoded(
                    f"target-{value}", Prediction(torch.zeros(1, 2, 3)), ["data_test\n"]
                ),
            )

    def execution(model, name, config) -> SimpleNamespace:  # noqa: ARG001 - shared callback or fixture signature
        executions.append((name, config))
        return SimpleNamespace(report=lambda: {"effective_compile": config.compile})

    monkeypatch.setattr(runtime, "bind_sampling_seed", Mock())
    monkeypatch.setattr(runtime, "Execution", execution)
    monkeypatch.setattr(runtime, "measured_forward", measure)
    monkeypatch.setattr(
        runtime, "CCDDatabase", lambda _path: SimpleNamespace(activate=nullcontext)
    )
    monkeypatch.setattr(runtime.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda _name: SimpleNamespace(prepare=prepare),
    )
    assert runtime.run(request) == 0
    assert calls == [0, 0]
    assert len(executions) == 1
    assert len(list(tmp_path.glob("*.prediction.pt"))) == 2
    report = json.loads((tmp_path / "target-1.json").read_text())
    assert report["model"] == "esmfold2"
    assert report["autocast"] is False


def test_model_cli_shims_cannot_own_execution_or_persistence():
    root = Path(foldforge.__file__).parent
    for model in ("af3", "esmfold2", "protenix", "opendde"):
        assert not (root / f"models/{model}/inference.py").exists()
    cli = (root / "cli.py").read_text()
    assert '"foldforge.models.io.cli"' in cli
    for path in (root / "models/io").glob("*_atoms.py"):
        source = path.read_text()
        assert "torch.save(" not in source
        assert "measured_forward(" not in source


def test_artifact_comparison_detects_absent_head_regressions(tmp_path):
    import importlib.util

    script = (
        Path(foldforge.__file__).resolve().parents[2]
        / "scripts/verify_prediction_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("artifact_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old, new = tmp_path / "old", tmp_path / "new"
    payload = {"coords": torch.zeros(1, 2, 3), "iptm": None}
    for directory in (old, new):
        directory.mkdir()
        torch.save(payload, directory / "target.prediction.pt")
        (directory / "target.cif").write_text("data_test\n_atom_site.Cartn_x 0\n")
    assert module.compare(old, new)["tensor_count"] == 1
    torch.save({**payload, "iptm": torch.zeros(1)}, new / "target.prediction.pt")
    with pytest.raises(AssertionError, match="absent head"):
        module.compare(old, new)
