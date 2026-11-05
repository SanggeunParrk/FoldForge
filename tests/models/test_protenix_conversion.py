"""Release-versus-graph validation without a JAX model/configuration import."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from foldforge.models.checkpoints.convert import _convert_protenix
from foldforge.modules.dense.spec import SPECS


@pytest.mark.parametrize("model", ["protenix1", "protenix2"])
def test_matching_graph_is_mapped_after_all_dimension_checks(model, tmp_path):
    spec = SPECS[model]
    state = object()
    params = object()

    def load(_path) -> object:
        return state

    def derive(actual) -> dict[str, int]:
        assert actual is state
        return {
            "n_pairformer": spec.trunk_layers,
            "n_msa": spec.msa_layers,
            "n_diff_token": spec.diffusion_blocks,
            "c_z": spec.pair_channel,
        }

    def mapped(actual) -> object:
        assert actual is state
        return params

    def write(output, filename, actual, *, add_meta) -> Path:
        assert actual is params
        assert add_meta
        return output / filename

    converter = SimpleNamespace(
        load_protenix_checkpoint=load,
        derive_dims=derive,
        map_protenix2_to_af3=mapped,
        C=SimpleNamespace(write_params_blob=write),
    )
    result = _convert_protenix(converter, model, tmp_path / "raw.pt", tmp_path)
    assert result == tmp_path / f"{model}.bin.zst"


@pytest.mark.parametrize("dimension", ["n_pairformer", "n_msa", "n_diff_token", "c_z"])
@pytest.mark.parametrize("model", ["protenix1", "protenix2"])
def test_wrong_checkpoint_is_rejected_before_mapping(model, dimension, tmp_path):
    spec = SPECS[model]
    values = {
        "n_pairformer": spec.trunk_layers,
        "n_msa": spec.msa_layers,
        "n_diff_token": spec.diffusion_blocks,
        "c_z": spec.pair_channel,
    }
    values[dimension] += 1
    converter = SimpleNamespace(
        load_protenix_checkpoint=lambda _: {},
        derive_dims=lambda _: values,
        map_protenix2_to_af3=lambda _: pytest.fail("mapped a wrong checkpoint"),
    )
    with pytest.raises(ValueError, match=dimension):
        _convert_protenix(converter, model, tmp_path / "raw.pt", tmp_path)
