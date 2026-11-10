"""The factories build engine classes with FoldForge's choices: model keys, master / fp32 parameters, the RoseTTAFold3 fast block."""

from __future__ import annotations

import pytest
import torch
from miniworld_engine.modules import (
    AtomDiTBoltz2,
    AtomDiTChai1,
    AtomDiTOpenFold3,
    AtomDiTProtenix,
    AtomDiTRoseTTAFold3,
    TokenDiTBoltz2,
    TokenDiTChai1,
    TokenDiTProtenix,
    TokenDiTRoseTTAFold3,
    Transition,
    TriangleAttentionAF3,
    TriangleAttentionChai1,
    TriangleAttentionChai1Confidence,
    TriangleAttentionRF3,
    TriangleMultiplication,
    TriangleMultiplicationRF3,
)
from miniworld_engine.modules.primitives import MasterWeights

from foldforge.modules.dense.spec import SPECS
from foldforge.modules.engine import modules as factory
from foldforge.modules.engine.configure import configure_model, precision_of
from foldforge.modules.engine.pairformer import pairformer_block
from foldforge.modules.engine.transformers import atom_dit_stack, engine_key_window, token_dit_stack


def test_every_dense_family_has_model_keys():
    for family in {spec.family for spec in SPECS.values()}:
        assert family in factory.MODEL_KEYS, family
    assert factory.model_keys("protenix2").triangle == "protenix"  # the triangle maps have no protenix_v2
    with pytest.raises(ValueError, match="no engine selector keys"):
        factory.model_keys("nope")


@pytest.mark.parametrize("precision", ["master", "fp32"])
def test_parameters_are_fp32_never_native_bf16(precision):
    module = factory.transition(32, 4, precision=precision)
    assert isinstance(module, Transition)
    assert {p.dtype for p in module.parameters()} == {torch.float32}
    assert module.master_weights is (precision == "master")
    module.to(torch.bfloat16)  # a bulk cast: masters and norm affines stay fp32 only under "master"
    expected = torch.float32 if precision == "master" else torch.bfloat16
    assert module.expand_a.weight.dtype == expected
    assert module.ln_in.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="precision"):
        factory.finish(Transition(8, 2), "bf16")


def test_master_flag_reaches_every_mixin_below():
    block = pairformer_block(d_pair=32, triangle_model="af3", d_single=64, n_head_single=4)
    mixins = [m for m in block.modules() if isinstance(m, MasterWeights)]
    assert mixins and all(m.master_weights for m in mixins)
    assert {p.dtype for p in block.parameters()} == {torch.float32}


def test_triangle_classes_follow_the_model_keys():
    assert type(factory.triangle_attention(32, 4, starting=True, model="af3")) is TriangleAttentionAF3
    assert type(factory.triangle_attention(32, 4, starting=False, model="rf3")) is TriangleAttentionRF3
    assert type(factory.triangle_attention(32, 4, starting=True, model="chai1")) is TriangleAttentionChai1
    assert type(factory.triangle_attention(32, 4, starting=True, model="chai1_confidence")) is TriangleAttentionChai1Confidence
    assert type(factory.triangle_multiplication(32, outgoing=True)) is TriangleMultiplication
    assert type(factory.triangle_multiplication(32, outgoing=True, divide_by_length=True)) is TriangleMultiplicationRF3
    # a template stack widens its projections only (protenix1: hidden 128 on pair 64)
    assert factory.triangle_multiplication(64, outgoing=False, d_hidden=128).to_left.weight.shape == (128, 64)


def test_opm_forms():
    grouped = factory.outer_product_mean(64, 32, 64, groups=8)
    assert grouped.product_norm.eps == 0.1 and grouped.groups == 8
    biased = factory.outer_product_mean(64, 32, 8, projection_bias=True, normalize_before_proj=False)
    assert biased.to_left.bias is not None and not biased.normalize_before_proj
    assert factory.outer_product_mean(16, 32, 8).count_offset is None  # the clamp divisor the kernels serve


def test_rosettafold3_fast_block_is_rf3_attention_with_plain_trimul():
    fast = pairformer_block(d_pair=32, triangle_model="rf3", parallel=True)
    assert type(fast.tri_atten_starting) is TriangleAttentionRF3
    assert type(fast.tri_multi_outgoing) is TriangleMultiplication and type(fast.tri_multi_incoming) is TriangleMultiplication
    assert fast.config.parallel
    exact = pairformer_block(d_pair=32, triangle_model="rf3", parallel=True, divide_by_length=True)
    assert type(exact.tri_multi_outgoing) is TriangleMultiplicationRF3


def test_chai1_block_gate_bias_and_confidence_attention():
    block = pairformer_block(d_pair=32, triangle_model="chai1_confidence", d_single=64, n_head_single=4, single_gate_bias=1.0, parallel=True)
    assert block.single_attention.gate_bias == 1.0
    assert type(block.tri_atten_starting) is TriangleAttentionChai1Confidence


@pytest.mark.parametrize(
    ("family", "token", "atom", "window"),
    [
        ("alphafold3", TokenDiTProtenix, AtomDiTOpenFold3, "slide"),
        ("protenix1", TokenDiTProtenix, AtomDiTProtenix, "slide"),
        ("boltz2", TokenDiTBoltz2, AtomDiTBoltz2, "pad"),
        ("rosettafold3", TokenDiTRoseTTAFold3, AtomDiTRoseTTAFold3, "pad"),
        ("chai1", TokenDiTChai1, AtomDiTChai1, "pad"),
    ],
)
def test_dit_stacks_are_engine_blocks(family, token, atom, window):
    d, cond = (768, 384) if family != "boltz2" else (768, 768)
    tokens = token_dit_stack(family, n_blocks=2, d_single=d, d_cond=cond, d_pair=128, n_head=16)
    atoms = atom_dit_stack(family, n_blocks=3, key_window=window, **({"adaln_eps": 1e-5} if family == "chai1" else {}))
    assert [type(b) for b in tokens.blocks] == [token, token]
    assert [type(b) for b in atoms.blocks] == [atom] * 3
    assert atoms.blocks[0].key_window == window
    assert all(k.startswith("blocks.") for k in tokens.state_dict())
    assert {p.dtype for p in tokens.parameters()} == {torch.float32}
    if family == "chai1":
        assert atoms.blocks[0].attention.ada_ln_in.ln_in.eps == 1e-5


def test_esmfold2_atom_stack_is_not_built():
    with pytest.raises(ValueError, match="no engine atom DiT"):
        atom_dit_stack("esmfold2", n_blocks=3)
    assert type(token_dit_stack("esmfold2", n_blocks=1, d_single=768, d_cond=768, d_pair=256, n_head=16).blocks[0]) is TokenDiTProtenix


def test_key_window_words():
    assert engine_key_window("slide") == ("slide", None)
    assert engine_key_window("slide_qblock") == ("slide_qblock", None)
    window, note = engine_key_window("circular")
    assert window == "pad" and "circular" in note
    with pytest.raises(ValueError, match="unknown atom key window"):
        engine_key_window("zigzag")


def test_configure_model_keeps_fp32_parameters_and_records_precision():
    model = torch.nn.Sequential(factory.transition(16, 2), torch.nn.Linear(16, 4))
    out = configure_model(model, "miniworld", torch.bfloat16, "cpu")
    assert out.foldforge_precision == "master" and out.foldforge_activation_dtype == torch.bfloat16
    assert {p.dtype for p in out.parameters()} == {torch.float32}
    assert precision_of(torch.float32) == "fp32"
    with pytest.raises(ValueError, match="no parameter precision"):
        precision_of(torch.float16)
    with pytest.raises(ValueError, match="Unknown backend"):
        configure_model(model, "magma", torch.bfloat16, "cpu")
    low = torch.nn.Linear(4, 4).to(torch.bfloat16)
    with pytest.raises(TypeError, match="never native low precision"):
        configure_model(low, "pytorch", torch.bfloat16, "cpu")
