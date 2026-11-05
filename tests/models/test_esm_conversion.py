"""ESMFold release guards use local model specifications without JAX."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from foldforge.models.checkpoints.convert import _prepare_esmfold2
from foldforge.modules.dense.spec import SPECS


@pytest.fixture(params=["esmfold2", "esmfold2_fast"])
def release(request):
    model = request.param
    spec = SPECS[model.replace("_fast", "-fast")]
    dimensions = {
        "n_trunk": spec.trunk_layers,
        "n_msa": spec.msa_layers,
        "n_coda": spec.coda_layers,
        "n_lm_encoder": spec.lm_encoder_layers,
        "msa_head_width": 16 if spec.msa_layers else 0,
        "num_bins": spec.distogram_bins,
    }
    converter = SimpleNamespace(derive_dims=lambda _: dimensions)
    _prepare_esmfold2(converter)
    return model, converter, dimensions


def test_matching_release_dimensions_are_checked(release):
    model, converter, _ = release
    converter.check_variant_table({"confidence_head.boundaries": np.empty(38)}, model)


@pytest.mark.parametrize(
    "field",
    ["n_trunk", "n_msa", "n_coda", "n_lm_encoder", "msa_head_width", "num_bins"],
)
def test_wrong_release_dimensions_are_rejected(release, field):
    model, converter, dimensions = release
    dimensions[field] += 1
    with pytest.raises(ValueError, match=field):
        converter.check_variant_table(
            {"confidence_head.boundaries": np.empty(38)}, model
        )


def test_wrong_confidence_bins_are_rejected(release):
    model, converter, _ = release
    with pytest.raises(ValueError, match="conf_bins"):
        converter.check_variant_table(
            {"confidence_head.boundaries": np.empty(39)}, model
        )


@pytest.mark.parametrize(
    "field",
    [
        "inference_num_steps",
        "gamma_0",
        "gamma_min",
        "noise_scale",
        "step_scale",
        "inference_s_min",
        "inference_s_max",
        "inference_p",
        "max_inference_sigma",
    ],
)
def test_sampler_mismatch_is_rejected(release, field, tmp_path):
    model, converter, _ = release
    (tmp_path / "config.json").write_text(json.dumps({"structure_head": {field: 999}}))
    with pytest.raises(ValueError, match=field):
        converter.check_release_config(tmp_path, model)


def test_original_sampler_profile_is_accepted(release, tmp_path):
    model, converter, _ = release
    head = {
        "inference_num_steps": 14,
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "step_scale": 1.5,
        "inference_s_min": 0.0004,
        "inference_s_max": 160.0,
        "inference_p": 7.0,
    }
    (tmp_path / "config.json").write_text(json.dumps({"structure_head": head}))
    assert converter.check_release_config(tmp_path, model)["structure_head"] == head
    assert (
        converter.check_release_config(Path("single-file.safetensors"), model) is None
    )
