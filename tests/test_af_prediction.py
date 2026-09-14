"""Common confidence units and explicit AF3 checkpoint selection."""

from unittest import mock

import pytest
import torch

from foldforge.models.af_prediction import from_af3, from_atom_confidence


def test_atom_confidence_averages_tokens_and_omits_monomer_iptm():
    full = {
        "atom_plddt": torch.tensor([0.2, 0.6, 0.9]),
        "atom_to_token_idx": torch.tensor([0, 0, 1]),
        "token_asym_id": torch.tensor([0, 0]),
        "token_pair_pae": torch.zeros(2, 2),
    }
    result = from_atom_confidence(
        {
            "coordinate": torch.zeros(1, 3, 3),
            "full_data": [full],
            "summary_confidence": [
                {"ptm": torch.tensor(0.7), "iptm": torch.tensor(0.0)}
            ],
        }
    )
    torch.testing.assert_close(result.plddt, torch.tensor([[0.4, 0.9]]))
    assert result.iptm is None


def test_af3_confidence_excludes_padding_atoms():
    result = from_af3(
        {
            "predicted_lddt": torch.tensor([[[20.0, 60.0], [90.0, 0.0]]]),
            "full_pae": torch.zeros(1, 2, 2),
        },
        torch.zeros(1, 3, 3),
        torch.tensor([[1, 1], [1, 0]]),
    )
    torch.testing.assert_close(result.plddt, torch.tensor([[0.4, 0.9]]))


@pytest.mark.parametrize("directory", [False, True])
def test_af3_honors_checkpoint_path(tmp_path, directory):
    from foldforge.models.af3.ported import params  # noqa: PLC0415

    selected = tmp_path if directory else tmp_path / "custom-name.bin.zst"
    expected = tmp_path / "af3.bin.zst" if directory else selected
    with (
        mock.patch.object(
            params, "get_alphafold3_params", side_effect=OSError("stop")
        ) as read,
        pytest.raises(OSError, match="stop"),
    ):
        params.import_jax_weights_(None, selected)
    read.assert_called_once_with(expected)


@pytest.mark.parametrize("family", ["protenix", "opendde"])
def test_actual_plddt_decoder_preserves_unit_interval(family):
    import importlib  # noqa: PLC0415

    api = importlib.import_module(f"foldforge.models.{family}.model")
    decoder = importlib.import_module(
        f"foldforge.models.{family}.ported.model.sample_confidence"
    )
    settings = api.configuration()
    config = (settings.confidence if family == "opendde" else settings.loss).plddt
    atom_scores = decoder.logits_to_score(
        torch.zeros(3, config.no_bins), **decoder.get_bin_params(config)
    )
    full = {
        "atom_plddt": atom_scores,
        "atom_to_token_idx": torch.tensor([0, 0, 1]),
        "token_asym_id": torch.tensor([0, 0]),
        "token_pair_pae": torch.zeros(2, 2),
    }
    result = from_atom_confidence(
        {
            "coordinate": torch.zeros(1, 3, 3),
            "full_data": [full],
            "summary_confidence": [{"ptm": torch.tensor(0.5)}],
        }
    )
    torch.testing.assert_close(result.plddt, torch.full((1, 2), 0.5))


def test_atom_confidence_survives_cif_roundtrip(tmp_path):
    import numpy as np
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from biotite.structure import AtomArray
    from biotite.structure.io import pdbx
    from foldforge.models.af_prediction import structure_with_confidence

    atoms = AtomArray(3)
    atoms.atom_name = np.array(["N", "CA", "C"])
    atoms.element = np.array(["N", "C", "C"])
    atoms.res_name[:] = "ALA"
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    result = structure_with_confidence(
        atoms, np.zeros((3, 3)), torch.tensor([0.2, 0.6, 0.9])
    )
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, result)
    path = tmp_path / "prediction.cif"
    cif.write(path)
    decoded = MMCIF2Dict(str(path))
    np.testing.assert_allclose(
        np.array(decoded["_atom_site.B_iso_or_equiv"], dtype=float), [20.0, 60.0, 90.0]
    )
    with pytest.raises(ValueError, match="one value per output atom"):
        structure_with_confidence(atoms, np.zeros((3, 3)), torch.ones(2))
