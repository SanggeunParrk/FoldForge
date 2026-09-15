"""Every prediction entry point must obey the same runs directory contract."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from foldforge.models.io import paths
from foldforge.models.io.cli import parse
from foldforge.models.io.output import Decoded, write_output
from foldforge.models.io.request import Request
from foldforge.prediction import Prediction


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_ROOT", root)
    return root


@pytest.mark.parametrize("name", ["experiment", "runs/experiment"])
def test_relative_output_uses_repository_root_from_any_cwd(
    name, runs, tmp_path, monkeypatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert paths.run_directory(name) == runs / "experiment"
    assert not runs.exists()


def test_default_run_names_are_unique_and_model_scoped(runs):
    first = paths.run_directory(model="af3")
    second = paths.run_directory(model="af3")
    assert first.parent == second.parent == runs / "af3"
    assert first != second
    assert not first.exists()


@pytest.mark.parametrize(
    "selection", ["../outside", "runs/../outside", "absolute", "symlink"]
)
def test_outside_destinations_rejected(selection, runs, tmp_path):
    if selection == "absolute":
        selection = tmp_path / "outside"
    elif selection == "symlink":
        runs.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (runs / "redirect").symlink_to(outside, target_is_directory=True)
        selection = runs / "redirect" / "prediction"
    with pytest.raises(ValueError, match="inside"):
        paths.run_directory(selection)


@pytest.mark.parametrize("model", ["af3", "protenix", "opendde", "esmfold2"])
def test_all_legacy_clis_default_inside_runs(model, runs):
    arguments = (
        []
        if model == "esmfold2"
        else ["--input", "input.json", "--checkpoint", "weights.pt"]
    )
    request = parse(model, arguments)
    assert request.out.parent == runs / model
    assert not request.out.exists()


def test_request_and_writer_share_resolved_destination(runs, tmp_path):
    request = Request(model="esmfold2", ccd_db=tmp_path, out=Path("case"))
    report = write_output(
        Decoded("target", Prediction(torch.zeros(1, 2, 3)), ["data_target\n"]),
        request.out,
        expected_samples=1,
    )
    assert request.out == runs / "case"
    assert Path(report["prediction_cif"]).parent == request.out
    assert (request.out / "target.prediction.pt").is_file()


def test_writer_rejects_external_directory_before_writing(runs, tmp_path):  # noqa: ARG001 - shared callback or fixture signature
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="inside"):
        write_output(
            Decoded("target", Prediction(torch.zeros(1, 2, 3)), ["data_target\n"]),
            outside,
            expected_samples=1,
        )
    assert not outside.exists()


def test_spec_cli_rejects_external_directory_before_preparation(
    runs,  # noqa: ARG001 - fixture installs the temporary run root
    tmp_path,
    monkeypatch,
):
    from foldforge.models import inference

    reader = Mock()
    monkeypatch.setattr(inference, "load", reader)
    with pytest.raises(SystemExit):
        inference.run(
            "af3", ["--spec", "input.yaml", "--out", str(tmp_path / "outside")]
        )
    reader.assert_not_called()
