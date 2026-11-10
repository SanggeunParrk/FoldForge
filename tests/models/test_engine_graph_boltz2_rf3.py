"""Boltz-2 and RoseTTAFold3 are built from engine classes: parity with the dense released-layout graph, and the converter's discards.

Same method as ``test_engine_graph.py`` (random weights on CPU in fp32, passed through the converter, every engine piece compared with the
dense piece it replaces), in BOTH modes: ``fast`` (AF3's computation: slid atom window, divide-after OPM, plain triangle multiplication for
RoseTTAFold3) and ``exact`` (the release's: clipped ``pad`` window, clamped OPM, RoseTTAFold3's divide-by-length multiplication).
The two families add three checks of their own: Boltz-2's / RoseTTAFold3's shared K / V AdaLN may be discarded only when it equals the
query AdaLN, and RoseTTAFold3's bias-free query only when the released bias is all zero.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from foldforge.models.architectures.af3 import AlphaFold3
from foldforge.models.checkpoints import miniworld
from foldforge.models.checkpoints.legacy import dense_model
from foldforge.modules.dense.spec import spec_for
from foldforge.modules.engine import graph, modules
from foldforge.modules.engine.modules import MODEL_KEYS
from foldforge.modules.engine.transformers import AtomDiTStack, TokenDiTStack
from foldforge.utils import atom_layout

TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}
CASES = [(family, mode) for family in ("boltz2", "rosettafold3") for mode in ("fast", "exact")]


def small(family: str, mode: str):
    spec = spec_for(family, mode)
    return replace(spec, trunk_layers=1, msa_layers=1, confidence_layers=1, diffusion_blocks=4, template_layers=1)


def randomise(model: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=generator)
            if p.ndim > 1:
                p.copy_(0.08 * noise)
            elif name.endswith("weight") or "scale" in name:
                p.copy_(1.0 + 0.1 * noise)
            else:
                p.copy_(0.05 * noise)


def tie_to_the_engine_layout(model: torch.nn.Module) -> None:
    """What a real release satisfies and random weights do not: the atom blocks' K / V AdaLN equals the query AdaLN (one shared AdaLN in the
    engine classes) and RoseTTAFold3's query projection has a zero bias (the engine class has none)."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, p in parameters.items():
            if ".q_adaptive_layernorm." in name:
                parameters[name.replace(".q_adaptive_layernorm.", ".k_adaptive_layernorm.")].copy_(p)
            if model.spec.family == "rosettafold3" and name.endswith(".q_projection.bias") and ("self_attention" in name or "cross_attention" in name):
                p.zero_()


@pytest.fixture(scope="module", params=CASES, ids=[f"{f}-{m}" for f, m in CASES])
def pair_of_models(request):
    """(dense model, engine model with the dense weights converted into it)."""
    family, mode = request.param
    modules.OPM_COUNT_OFFSET = 1e-3  # the dense graph's exact AF3 divisor (the exact mode's clamp is the spec's own)
    try:
        dense = dense_model(small(family, mode), num_recycles=1, num_samples=1, diffusion_steps=1)
        randomise(dense, 1)
        tie_to_the_engine_layout(dense)
        dense.diffusion_head.fourier_embeddings.register_buffer("weight", torch.randn(256))
        dense.diffusion_head.fourier_embeddings.register_buffer("bias", torch.randn(256))
        engine = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=small(family, mode))
    finally:
        modules.OPM_COUNT_OFFSET = None
    state = miniworld.convert_state(dense.state_dict(), engine)
    fourier = {k: state.pop(k) for k in miniworld.LOOSE}
    engine.load_state_dict(state, strict=True)
    assert set(fourier) == set(miniworld.LOOSE)
    return dense.eval(), engine.eval()


def test_the_two_families_are_ported():
    assert {"boltz2", "rosettafold3"} <= set(MODEL_KEYS)
    assert {"boltz2", "rosettafold3"} <= set(miniworld.NATIVE_FILES)


def test_the_graph_holds_engine_classes_with_engine_keys(pair_of_models):
    _, engine = pair_of_models
    family = engine.spec.family
    keys = set(engine.state_dict())
    assert "evoformer.trunk_pairformer.0.tri_multi_outgoing.to_left.weight" in keys
    assert "evoformer.msa_stack.0.msa_attention1.ln_pair.weight" in keys
    assert "diffusion_head.transformer.blocks.3.attention.to_bias.weight" in keys
    assert not any("pair_logits_projection" in k or "pair_input_layer_norm" in k or "k_adaptive_layernorm" in k for k in keys)
    assert not any("ada_ln_kv" in k for k in keys), "the atom blocks of both families share ONE AdaLN"
    assert all(p.dtype == torch.float32 for p in engine.parameters())
    assert isinstance(engine.diffusion_head.transformer, TokenDiTStack)
    assert isinstance(engine.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder, AtomDiTStack)
    if family == "boltz2":
        assert "diffusion_head.transformer.blocks.0.transition.up.weight" in keys  # the up-gated ConditionedTransition
        assert not any(k.endswith("norm_query.weight") for k in keys)
    else:
        assert "diffusion_head.transformer.blocks.0.attention.norm_query.weight" in keys  # FlatQKNorm
        assert "diffusion_head.transformer.blocks.0.attention.norm_key.weight" in keys
        assert not any(k.endswith("attention.to_query.bias") for k in keys if "diffusion_head" in k)  # bias-free query
        assert "evoformer.msa_stack.0.outer_product_mean.to_left.bias" in keys  # OPM projection_bias


def test_engine_classes_are_the_families_own(pair_of_models):
    dense, engine = pair_of_models
    family, mode = dense.spec.family, "exact" if dense.spec.triangle_mul_divide_by_length or dense.spec.opm_clamped_norm else "fast"
    block = engine.evoformer.trunk_pairformer[0]
    attention = type(block.tri_atten_starting).__name__
    multiplication = type(block.tri_multi_outgoing).__name__
    atom = type(engine.evoformer_conditioning.atom_transformer_encoder.blocks[0]).__name__
    token = type(engine.diffusion_head.transformer.blocks[0]).__name__
    if family == "boltz2":
        assert (atom, token) == ("AtomDiTBoltz2", "TokenDiTBoltz2")
        assert multiplication == "TriangleMultiplication"
    else:
        assert (atom, token) == ("AtomDiTRoseTTAFold3", "TokenDiTRoseTTAFold3")
        assert attention == "TriangleAttentionRF3"
        # fast mode builds the PLAIN multiplication explicitly, exact mode the RF3 one (contraction divided by the real length)
        assert multiplication == ("TriangleMultiplicationRF3" if mode == "exact" else "TriangleMultiplication")


def _pair_inputs(length, d_pair, d_single, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.ones(length)
    tokens[-3:] = 0
    return {
        "pair": torch.randn(length, length, d_pair, generator=generator),
        "single": torch.randn(length, d_single, generator=generator),
        "seq_mask": tokens,
        "pair_mask": tokens[:, None] * tokens[None, :],
    }


def _valid(x, mask):
    return x[mask > 0]


def test_trunk_pairformer_block_matches_dense(pair_of_models):
    dense, engine = pair_of_models
    spec = dense.spec
    x = _pair_inputs(24, spec.pair_channel, spec.seq_channel)
    with torch.no_grad():
        want_pair, want_single = dense.evoformer.trunk_pairformer[0](x["pair"], x["pair_mask"], x["single"].clone(), x["seq_mask"])
        got_pair, got_single = graph.run_block(
            engine.evoformer.trunk_pairformer[0], x["pair"], x["pair_mask"], x["single"].clone(), x["seq_mask"]
        )
    torch.testing.assert_close(_valid(got_pair, x["pair_mask"]), _valid(want_pair, x["pair_mask"]), **TOLERANCE)
    torch.testing.assert_close(_valid(got_single, x["seq_mask"]), _valid(want_single, x["seq_mask"]), **TOLERANCE)


def test_template_blocks_match_dense_and_take_a_batch_axis(pair_of_models):
    dense, engine = pair_of_models
    channels = dense.spec.template_channel
    want_block = dense.evoformer.template_embedding.tmpl_pairformer[0]
    got_block = engine.evoformer.template_embedding.tmpl_pairformer[0]
    assert got_block.supports_template_batch
    x = _pair_inputs(20, channels, 1)
    slots = torch.randn(3, 20, 20, channels)
    with torch.no_grad():
        want = torch.stack([want_block(slot, pair_mask=x["pair_mask"]) for slot in slots])
        got = graph.run_block(got_block, slots, x["pair_mask"])
    assert got.shape == slots.shape
    torch.testing.assert_close(got[:, :-3, :-3], want[:, :-3, :-3], **TOLERANCE)


def test_confidence_block_takes_samples_in_one_call(pair_of_models):
    dense, engine = pair_of_models
    spec = dense.spec
    x = _pair_inputs(16, spec.pair_channel, spec.seq_channel)
    pairs = torch.randn(2, 16, 16, spec.pair_channel)
    singles = torch.randn(2, 16, spec.seq_channel)
    got_block = engine.confidence_head.confidence_pairformer[0]
    want_block = dense.confidence_head.confidence_pairformer[0]
    assert got_block.supports_sample_batch
    with torch.no_grad():
        want = [want_block(pairs[i], x["pair_mask"], singles[i].clone(), x["seq_mask"]) for i in range(2)]
        got_pair, got_single = graph.run_block(got_block, pairs, x["pair_mask"], singles.clone(), x["seq_mask"])
    for i in range(2):
        torch.testing.assert_close(got_pair[i][:-3, :-3], want[i][0][:-3, :-3], **TOLERANCE)
        torch.testing.assert_close(got_single[i][:-3], want[i][1][:-3], **TOLERANCE)


def test_msa_block_matches_dense(pair_of_models):
    dense, engine = pair_of_models
    spec = dense.spec
    generator = torch.Generator().manual_seed(3)
    x = _pair_inputs(20, spec.pair_channel, 1)
    msa = torch.randn(6, 20, spec.msa_channel, generator=generator)
    msa_mask = (torch.rand(6, 20, generator=generator) > 0.2).float() * x["seq_mask"]
    msa_mask[0] = x["seq_mask"]
    with torch.no_grad():
        want_msa, want_pair = dense.evoformer.msa_stack[0](msa.clone(), x["pair"], msa_mask, x["pair_mask"])
        got_msa, got_pair = engine.evoformer.msa_stack[0](msa.clone(), x["pair"], msa_mask, x["pair_mask"])
    torch.testing.assert_close(got_msa[:, :-3], want_msa[:, :-3], **TOLERANCE)
    torch.testing.assert_close(_valid(got_pair, x["pair_mask"]), _valid(want_pair, x["pair_mask"]), **TOLERANCE)


def test_diffusion_transformer_matches_dense(pair_of_models):
    dense, engine = pair_of_models
    head = engine.diffusion_head
    generator = torch.Generator().manual_seed(5)
    length = 24
    act = torch.randn(2, length, head.c_act, generator=generator)
    single = torch.randn(2, length, head.seq_channel, generator=generator)
    pair = torch.randn(length, length, head.pair_channel, generator=generator)
    mask = torch.ones(length)
    mask[-3:] = 0
    key_mask = (mask > 0)[None]
    with torch.no_grad():
        want = dense.diffusion_head.transformer(act.clone(), mask, single, pair, key_mask=key_mask)
        got = head.transformer(act.clone(), mask, single, pair, key_mask=key_mask)
    torch.testing.assert_close(got[:, :-3], want[:, :-3], **TOLERANCE)


@pytest.mark.parametrize("atoms", [130, 200])
def test_atom_transformer_matches_dense(pair_of_models, atoms):
    """32 queries against 128 keys: slid inside the real atoms (fast) or at the clipped ``32 w - 48`` start (exact, ``pad``)."""
    from miniworld_engine.modules.functional import slide_starts

    dense, engine = pair_of_models
    generator = torch.Generator().manual_seed(7)
    windows = (atoms + 31) // 32
    total = windows * 32
    flat = torch.zeros(total)
    flat[:atoms] = 1
    queries_mask = flat.reshape(windows, 32)
    if dense.spec.atom_key_window == "pad":
        start = 32 * torch.arange(windows) - 48
    else:
        start = slide_starts(torch.tensor([atoms]), windows, 128)[0]
    index = start[:, None] + torch.arange(128)
    valid = (index >= 0) & (index < atoms)
    gather = atom_layout.GatherInfo(index.clamp(0, total - 1), valid, (windows, 32))
    keys_mask = atom_layout.convert(gather, queries_mask, layout_axes=(-2, -1))
    want_stack = dense.evoformer_conditioning.atom_transformer_encoder
    got_stack = engine.evoformer_conditioning.atom_transformer_encoder
    channels = got_stack.blocks[0].attention.to_query.weight.shape[1]
    act = torch.randn(windows, 32, channels, generator=generator) * queries_mask[..., None]
    cond = torch.randn(windows, 32, channels, generator=generator)
    keys_cond = atom_layout.convert(gather, cond, layout_axes=(-3, -2))
    pair = torch.randn(windows, 32, 128, 16, generator=generator)
    with torch.no_grad():
        want = want_stack(act.clone(), queries_mask, gather, keys_mask, cond, keys_cond, pair)
        got = got_stack(act.clone(), queries_mask, gather, keys_mask, cond, keys_cond, pair)
    torch.testing.assert_close(got * queries_mask[..., None], want * queries_mask[..., None], **TOLERANCE)


# ------------------------------------------------------------------------------------------------------------------ the discards
def test_shared_kv_adaln_is_discarded_only_when_equal_to_the_query_adaln(pair_of_models):
    dense, engine = pair_of_models
    state = dict(dense.state_dict())
    key = "evoformer_conditioning.atom_transformer_encoder.cross_attention.1.k_adaptive_layernorm.single_cond_scale.weight"
    assert torch.equal(state[key], state[key.replace("k_adaptive", "q_adaptive")])
    miniworld.convert_state(state, engine)  # equal: converts, and the K / V keys count as read
    state[key] = state[key] + 1e-3
    with pytest.raises(ValueError, match="differs from"):
        miniworld.convert_state(state, engine)


def test_rf3_query_bias_is_dropped_only_when_zero(pair_of_models):
    dense, engine = pair_of_models
    if dense.spec.family != "rosettafold3":
        pytest.skip("Boltz-2 keeps its query bias")
    state = dict(dense.state_dict())
    key = "diffusion_head.transformer.self_attention.2.q_projection.bias"
    assert not state[key].any()
    state[key] = state[key].clone()
    state[key][0] = 1e-3
    with pytest.raises(miniworld.ConversionError, match="not all zero"):
        miniworld.convert_state(state, engine)


def test_miniworld_file_round_trip(pair_of_models, tmp_path):
    dense, engine = pair_of_models
    family, mode = dense.spec.family, "exact" if dense.spec.opm_clamped_norm else "fast"
    state = miniworld.convert_state(dense.state_dict(), engine)
    path = tmp_path / "x.miniworld.pt"
    torch.save({"format": miniworld.FORMAT, "version": miniworld.VERSION, "family": family, "source": "t", "state_dict": state}, path)
    fresh = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=small(family, mode))
    report = miniworld.load_into(fresh, path, family=family)
    assert report["format_version"] == miniworld.VERSION
    for key, value in engine.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0, atol=0)
