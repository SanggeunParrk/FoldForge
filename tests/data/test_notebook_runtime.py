"""Notebook requests must use the shared inference runtime for every seed."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from foldforge.data.ccd import database
from foldforge.data.web.request_parser import RequestParser
from foldforge.models.io import runtime


@pytest.mark.parametrize("split", ["legacy", "both", "trunk_only"])
def test_launch_preserves_request_options(tmp_path, monkeypatch, split):
    parser = object.__new__(RequestParser)
    parser.request = {
        "model_seeds": [7, 9],
        "backend": "pytorch",
        "precision": "bf16",
        "N_cycle": 3,
        "N_step": 8,
        "N_sample": 5,
        "use_template": True,
        "use_msa": False,
        "use_tfg": True,
    }
    if split != "legacy":
        parser.request.pop("model_seeds")
        parser.request.update(trunk_seeds=[7, 9])
        if split == "both":
            parser.request["diffusion_seeds"] = [13, 17]
    parser.request_dir = str(tmp_path / "notebook")
    parser.model_name = "protenix_base_default_v1.0.0"
    ccd = Mock(root=tmp_path / "ccd")
    ccd.activate.return_value = nullcontext()
    monkeypatch.setattr(database, "current_database", lambda: ccd)
    monkeypatch.setattr(parser, "get_data_json", lambda: str(tmp_path / "input.json"))
    monkeypatch.setattr(parser, "get_model", lambda: str(tmp_path / "weights"))
    run = Mock()
    monkeypatch.setattr(runtime, "run", run)
    parser.launch()
    requests = [call.args[0] for call in run.call_args_list]
    expected = {
        "legacy": [(7, 7), (9, 9)],
        "both": [(7, 13), (7, 17), (9, 13), (9, 17)],
        "trunk_only": [(7, 0), (9, 0)],
    }
    assert [(r.trunk_seed, r.diffusion_seed) for r in requests] == expected[split]
    for request in requests:
        assert request.model == "protenix"
        assert request.backend == "pytorch"
        assert request.precision == "bf16"
        assert (request.recycles, request.steps, request.samples) == (3, 8, 5)
        assert request.guidance
        assert request.ccd_db == ccd.root
        assert request.checkpoint == tmp_path / "weights" / f"{parser.model_name}.pt"
        assert request.out == (
            tmp_path
            / "notebook"
            / f"trunk-{request.trunk_seed}_diffusion-{request.diffusion_seed}"
        )
