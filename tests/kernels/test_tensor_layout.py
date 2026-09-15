"""Encoded atom layouts retain field identity, dtype and padding masks."""

import dataclasses

import pytest
import torch

from foldforge.modules.dense.atom_layout import AtomLayout, Residues


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_layout_roundtrip_uses_documented_field_order(dtype):
    fields = {
        name: torch.arange(1, 5, dtype=dtype).reshape(2, 2) + 10 * i
        for i, name in enumerate(
            (
                "atom_name",
                "atom_element",
                "res_name",
                "res_id",
                "chain_id",
                "chain_type",
            )
        )
    }
    layout = AtomLayout(**fields)
    packed = layout.to_array()
    for index, value in enumerate(fields.values()):
        torch.testing.assert_close(torch.from_numpy(packed[index]), value)
    assert AtomLayout.from_array(packed) == layout


def test_layout_padding_preserves_tensor_metadata_and_missing_fields():
    values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    layout = AtomLayout(atom_name=values, res_id=values + 5, chain_id=values + 10)
    padded = layout.copy_and_pad_to((3, 4))
    for name in ("atom_name", "res_id", "chain_id"):
        actual = getattr(padded, name)
        assert actual.dtype == values.dtype
        assert actual.device == values.device
        torch.testing.assert_close(actual[:2, :2], getattr(layout, name))
        assert not actual[2:].any()
        assert not actual[:, 2:].any()
    assert padded.atom_element is None
    with pytest.raises(ValueError, match="smaller"):
        layout.copy_and_pad_to((1, 2))


def test_residues_equality_accepts_missing_optional_fields():
    values = torch.arange(3)
    residue = Residues(
        res_name=values,
        res_id=values,
        chain_id=values,
        chain_type=values,
        is_start_terminus=values.bool(),
        is_end_terminus=values.bool(),
    )
    assert residue == dataclasses.replace(residue)
    assert residue != dataclasses.replace(residue, smiles_string=values)
