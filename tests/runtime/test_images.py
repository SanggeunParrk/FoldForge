"""Diagnostics stay opt-in and preserve physical units, masks and sample axes."""

import json

import numpy as np
import pytest
import torch
from conftest import cli

from foldforge.models.config import Config, OutputConfig
from foldforge.models.io import images, paths
from foldforge.models.io.cli import parse
from foldforge.models.io.confidence import from_af3, from_atom_confidence
from foldforge.models.io.output import Decoded, write_output
from foldforge.prediction import Prediction


def test_selection_defaults_validation_and_cli():
    assert Config().output.image_names == ()
    assert OutputConfig(images=("pae", "pae")).image_names == ("pae",)
    assert len(OutputConfig(images=("all",)).image_names) == 6
    with pytest.raises(ValueError, match="images"):
        Config.model_validate({"output": {"images": ["paee"]}})
    assert parse("esmfold2", cli("--save-images", "pae", "msa")).output.image_names == (
        "pae",
        "msa",
    )


@pytest.mark.parametrize("batched", [False, True])
def test_input_masks_gaps_padding_and_empty_template_slots(batched):
    features = {
        "msa": torch.tensor([[0, 1, 2, 0], [0, 31, 1, 0], [0, 0, 0, 0]]),
        "msa_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]),
        "seq_mask": torch.tensor([1, 1, 1, 0]),
        "template_atom_mask": torch.tensor(
            [[[1], [0], [1], [0]], [[0], [0], [0], [0]]]
        ),
    }
    if batched:
        features = {k: v.unsqueeze(0) for k, v in features.items()}
    result = images.input_images(
        features, ("msa", "template"), gap_id=31, batched=batched
    )
    np.testing.assert_equal(result["msa"], [[1, 1, 1], [1, np.nan, 0]])
    np.testing.assert_equal(result["msa_coverage"], [2, 1, 2])
    np.testing.assert_equal(result["template"], [[1, 0, 1]])
    assert result["msa_rows"] == 2
    assert images.input_images(features, (), gap_id=31) == {}


def test_msa_display_cap_does_not_truncate_coverage():
    msa = np.zeros((images.MAX_MSA_ROWS + 10, 2), dtype=int)
    result = images.input_images({"msa": msa}, ("msa",), gap_id=31)
    assert len(result["msa"]) == images.MAX_MSA_ROWS
    np.testing.assert_equal(result["msa_coverage"], [len(msa), len(msa)])


def test_prediction_units_and_sample_axes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    prediction = Prediction(
        coords=torch.zeros(2, 3, 3),
        plddt=torch.tensor([[0.4, 0.8], [0.3, 0.9]]),
        pae=torch.arange(8).reshape(2, 2, 2).float(),
        pde=torch.ones(2, 2, 2),
    )
    result = Decoded(
        "target",
        prediction,
        ["data_a\n", "data_b\n"],
        image_inputs={
            "msa": np.ones((1, 2)),
            "msa_rows": 1,
            "msa_coverage": np.ones(2),
            "template": np.array([[1.0, 0.0]]),
        },
        image_distogram=torch.tensor(
            [[[0.0, 2.0], [3.0, 0.0]], [[1.0, 0.0], [0.0, 4.0]]]
        ),
    )
    captured = {}
    monkeypatch.setattr(
        images, "_render", lambda plot, path: captured.update({path.name: plot})
    )
    rng = torch.random.get_rng_state().clone()
    report = write_output(result, tmp_path, expected_samples=2, images=("all",))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert len(captured) == 10
    np.testing.assert_allclose(captured["sample-000-plddt.png"].values, [40, 80])
    assert captured["sample-001-pde.png"].units == "angstrom"
    np.testing.assert_equal(captured["distogram.png"].values, [[1, 0], [0, 1]])
    assert "angstrom" not in captured["distogram.png"].units
    assert report["images"]["skipped"] == {}
    assert (
        json.loads((tmp_path / "target.json").read_text())["images"] == report["images"]
    )


def test_png_render_and_missing_heads(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    result = Decoded(
        "target", Prediction(torch.zeros(1, 1, 3), plddt=torch.ones(1, 2)), ["data_a\n"]
    )
    report = write_output(result, tmp_path, expected_samples=1, images=("all",))
    filename = report["images"]["saved"]["plddt"][0]
    assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert set(report["images"]["skipped"]) == {
        "pae",
        "pde",
        "distogram",
        "msa",
        "template",
    }


def test_disabled_images_do_not_render(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)

    def fail(*_args: object) -> None:
        pytest.fail("Images are disabled")

    monkeypatch.setattr(images, "write_images", fail)
    report = write_output(
        Decoded("target", Prediction(torch.zeros(1, 1, 3)), ["data_a\n"]),
        tmp_path,
        expected_samples=1,
    )
    assert "images" not in report
    assert not (tmp_path / "images").exists()


def test_pde_adapters_and_shape_contract():
    pde = torch.ones(1, 2, 2) * 3
    af3 = from_af3(
        {
            "predicted_lddt": torch.ones(1, 2, 1) * 70,
            "full_pae": pde * 2,
            "full_pde": pde,
        },
        torch.zeros(1, 2, 3),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(af3.pde, pde)
    flat = from_atom_confidence(
        {
            "coordinate": torch.zeros(1, 2, 3),
            "full_data": [
                {
                    "atom_plddt": torch.ones(2) * 0.7,
                    "atom_to_token_idx": torch.arange(2),
                    "token_asym_id": torch.zeros(2),
                    "token_pair_pae": pde[0] * 2,
                    "token_pair_pde": pde[0],
                }
            ],
            "summary_confidence": [{"ptm": torch.tensor(0.8)}],
        }
    )
    torch.testing.assert_close(flat.pde, pde)
    with pytest.raises(ValueError, match="square"):
        Prediction(torch.zeros(1, 2, 3), pde=torch.zeros(1, 2, 3))


def test_png_destination_cannot_escape_via_symlink(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "images").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="inside"):
        images.write_images(
            Decoded("target", Prediction(torch.zeros(1, 1, 3)), ["data_a\n"]),
            out,
            ("pae",),
        )


@pytest.mark.parametrize(
    "kind", ["pae", "pde", "distogram", "plddt", "msa", "template"]
)
def test_each_kind_renders_real_png(kind, tmp_path):
    result = Decoded(
        "render",
        Prediction(
            torch.zeros(1, 2, 3),
            plddt=torch.full((1, 2), 0.8),
            pae=torch.ones(1, 2, 2),
            pde=torch.ones(1, 2, 2),
        ),
        ["data_render\n"],
        image_inputs={
            "msa": np.array([[1.0, np.nan], [0.0, 1.0]]),
            "msa_rows": 2,
            "msa_coverage": np.array([2, 1]),
            "template": np.array([[1.0, 0.0]]),
        },
        image_distogram=torch.zeros(2, 2, 4, dtype=torch.bfloat16),
    )
    report = images.write_images(result, tmp_path, (kind,))
    for name in report["saved"][kind]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_dense_distogram_retention_preserves_existing_contact_output():
    from types import SimpleNamespace

    from foldforge.modules.dense.head import DistogramHead

    head = DistogramHead(c_pair=4, num_bins=4)
    embeddings = {"pair": torch.ones(2, 2, 4)}
    batch = SimpleNamespace(token_features=SimpleNamespace(mask=torch.ones(2)))
    normal = head(batch, embeddings)
    assert "logits" not in normal
    head.save_distogram = True
    requested = head(batch, embeddings)
    assert requested["logits"].shape == (2, 2, 4)
    torch.testing.assert_close(normal["contact_probs"], requested["contact_probs"])


@pytest.mark.parametrize("shape", [(1, 2, 2, 4), (2, 3, 4)])
def test_invalid_distogram_axes_are_not_silently_plotted(tmp_path, shape):
    result = Decoded(
        "target",
        Prediction(torch.zeros(1, 1, 3)),
        ["data_a\n"],
        image_distogram=torch.zeros(shape),
    )
    with pytest.raises(ValueError, match="logits"):
        images.write_images(result, tmp_path, ("distogram",))


def test_nonfinite_distogram_is_rejected(tmp_path):
    result = Decoded(
        "target",
        Prediction(torch.zeros(1, 1, 3)),
        ["data_a\n"],
        image_distogram=torch.full((2, 2, 4), float("nan")),
    )
    with pytest.raises(FloatingPointError, match="Non-finite"):
        images.write_images(result, tmp_path, ("distogram",))
