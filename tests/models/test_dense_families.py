"""Every dense family is the one AF3 graph plus declared, testable differences."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foldforge.data.features.dense_conventions import centre_conformers
from foldforge.models import entry, is_dense, registered_models
from foldforge.modules.dense.spec import ALPHAFOLD3, SPECS


def _meta_model(family: str) -> torch.nn.Module:
    from foldforge.models.architectures.af3 import AlphaFold3

    with torch.device("meta"):
        return AlphaFold3(spec=SPECS[family])


def test_every_dense_registry_row_names_a_spec_and_the_one_architecture():
    dense = [name for name in registered_models() if is_dense(name)]
    assert {"af3", "intellifold2", "openfold3", "openfold3-preview2"} <= set(dense)
    for name in dense:
        row = entry(name)
        assert row.architecture == "foldforge.models.architectures.af3.AlphaFold3"
        assert (row.family or "alphafold3") in SPECS


@pytest.mark.parametrize("family", ["intellifold2", "openbind0"])
def test_width_and_convention_families_are_the_af3_module_tree(family):
    """Widths and forward conventions never change the module tree."""
    reference = {name for name, _ in _meta_model("alphafold3").named_modules()}
    assert {name for name, _ in _meta_model(family).named_modules()} == reference


def test_per_block_pair_norm_changes_only_the_diffusion_transformer():
    reference = {name for name, _ in _meta_model("openbind0").named_modules()}
    names = {name for name, _ in _meta_model("openfold3").named_modules()}
    assert all("diffusion_head.transformer.pair_" in name for name in names ^ reference)


def test_boltz2_adds_exactly_its_declared_modules():
    model = _meta_model("boltz2")
    assert len(model.evoformer.trunk_pairformer) == 64
    assert len(model.confidence_head.confidence_pairformer) == 8
    assert type(model.evoformer.template_embedding).__name__ == "FusedTemplateEmbedding"
    assert model.confidence_head.split_heads
    assert model.evoformer.left_single.in_features == 384
    assert model.evoformer.msa_activations.in_features == 35
    state = model.state_dict()
    assert state["diffusion_head.single_cond_initial_projection.bias"].shape == (768,)
    assert "diffusion_head.transformer.transition_block.0.a_to_b.weight" in state
    assert "input_embedder.method_conditioning.weight" in state


def test_widths_follow_the_spec():
    model = _meta_model("intellifold2")
    state = model.state_dict()
    attention = "evoformer.trunk_pairformer.0.pair_attention1"
    assert state[f"{attention}.q_projection.weight"].shape == (512, 512)
    assert state[f"{attention}.pair_bias_projection.weight"].shape == (8, 512)
    assert state["evoformer.template_embedding.output_linear.weight"].shape == (
        512,
        256,
    )
    assert state["diffusion_head.pair_cond_initial_projection.weight"].shape == (
        512,
        512 + 139,
    )


def test_padded_single_conditioning_is_two_channels_wider():
    stock = _meta_model("alphafold3").state_dict()
    padded = _meta_model("openbind0").state_dict()
    key = "diffusion_head.single_cond_initial_projection.weight"
    assert stock[key].shape == (384, 831)
    assert padded[key].shape == (384, 833)


def test_per_block_pair_norm_has_one_norm_and_projection_per_block():
    transformer = _meta_model("openfold3").diffusion_head.transformer
    assert len(transformer.pair_input_layer_norm) == transformer.num_blocks == 24
    assert transformer.pair_logits_projection[0].weight.shape == (16, 128)
    stock = _meta_model("openbind0").diffusion_head.transformer
    assert len(stock.pair_logits_projection) == 6


def test_transposed_column_bias_only_touches_the_column_direction():
    from dataclasses import replace

    from foldforge.modules.dense.attention import GridSelfAttention

    flagged = replace(ALPHAFOLD3, transposed_column_pair_bias=True)
    assert not GridSelfAttention(
        c_pair=8, transpose=False, spec=flagged
    ).transposed_bias
    assert GridSelfAttention(c_pair=8, transpose=True, spec=flagged).transposed_bias
    assert not GridSelfAttention(c_pair=8, transpose=True).transposed_bias


def test_centre_conformers_is_masked_and_per_group():
    example = {
        "ref_pos": np.array(
            [
                [[1.0, 0, 0], [3.0, 0, 0], [9.0, 9, 9]],
                [[0, 2.0, 0], [0, 4.0, 0], [0, 0, 0]],
            ],
            dtype=np.float32,
        ),
        "ref_mask": np.array([[1, 1, 0], [1, 1, 0]]),
        "ref_space_uid": np.array([[0, 0, 0], [1, 1, 0]]),
    }
    centre_conformers(example)
    np.testing.assert_allclose(example["ref_pos"][0, :2, 0], [-1.0, 1.0])
    np.testing.assert_allclose(example["ref_pos"][1, :2, 1], [-1.0, 1.0])
    # Padding stays exactly zero, whatever it held before.
    assert not example["ref_pos"][0, 2].any()
    assert not example["ref_pos"][1, 2].any()
