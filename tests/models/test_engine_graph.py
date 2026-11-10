"""The af3-family graph is built from engine classes: parity with the dense released-layout graph, the converter and the format.

The dense graph (``spec_for(family)``) is the released layout the Haiku loader fills; the engine graph is what a ported
family runs. Both are built on CPU in fp32 with the same random weights, the weights pass through the converter, and every piece the
engine graph holds as an engine class is compared with the dense piece it replaces.
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
FAMILIES = ("alphafold3", "openfold3")  # AF3's shared pair norm; preview-2's per-block norm and transposed bias


def small(family: str):
    spec = spec_for(family)
    return replace(spec, trunk_layers=1, msa_layers=1, confidence_layers=1, diffusion_blocks=4)


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


@pytest.fixture(scope="module", params=FAMILIES)
def pair_of_models(request):
    """(dense model, engine model with the dense weights converted into it)."""
    family = request.param
    modules.OPM_COUNT_OFFSET = 1e-3  # AF3's exact divisor: the kernels' clamp differs at small row counts
    try:
        dense = dense_model(small(family), num_recycles=1, num_samples=1, diffusion_steps=1)
        randomise(dense, 1)
        dense.diffusion_head.fourier_embeddings.register_buffer("weight", torch.randn(256))
        dense.diffusion_head.fourier_embeddings.register_buffer("bias", torch.randn(256))
        engine = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=small(family))
    finally:
        modules.OPM_COUNT_OFFSET = None
    state = miniworld.convert_state(dense.state_dict(), engine)
    fourier = {k: state.pop(k) for k in miniworld.LOOSE}
    engine.load_state_dict(state, strict=True)
    assert set(fourier) == set(miniworld.LOOSE)
    return dense.eval(), engine.eval()


def test_engine_families_are_the_ported_ones():
    assert {"alphafold3", "intellifold2", "openbind0", "openfold3"} <= set(MODEL_KEYS)


def test_the_graph_holds_engine_classes_with_engine_keys(pair_of_models):
    _, engine = pair_of_models
    keys = set(engine.state_dict())
    assert "evoformer.trunk_pairformer.0.tri_multi_outgoing.to_left.weight" in keys
    assert "evoformer.trunk_pairformer.0.transition_pair.expand_a.weight" in keys
    assert "evoformer.msa_stack.0.msa_attention1.ln_pair.weight" in keys
    assert "diffusion_head.transformer.blocks.3.attention.to_bias.weight" in keys
    assert "diffusion_head.pair_transition_0.squeeze.weight" in keys
    assert "evoformer_conditioning.atom_transformer_encoder.blocks.2.attention.ada_ln_kv.to_scale.weight" in keys
    assert not any("pair_logits_projection" in k or "pair_input_layer_norm" in k for k in keys)
    assert all(p.dtype == torch.float32 for p in engine.parameters())
    assert isinstance(engine.diffusion_head.transformer, TokenDiTStack)
    assert isinstance(engine.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder, AtomDiTStack)


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
    x = _pair_inputs(24, 128, 384)
    with torch.no_grad():
        want_pair, want_single = dense.evoformer.trunk_pairformer[0](x["pair"], x["pair_mask"], x["single"].clone(), x["seq_mask"])
        got_pair, got_single = graph.run_block(
            engine.evoformer.trunk_pairformer[0], x["pair"], x["pair_mask"], x["single"].clone(), x["seq_mask"]
        )
    torch.testing.assert_close(_valid(got_pair, x["pair_mask"]), _valid(want_pair, x["pair_mask"]), **TOLERANCE)
    torch.testing.assert_close(_valid(got_single, x["seq_mask"]), _valid(want_single, x["seq_mask"]), **TOLERANCE)


def test_pair_only_blocks_match_dense_and_take_a_batch_axis(pair_of_models):
    """The template embedder's blocks (pair 64, two heads' worth of width, transition factor 2), one call for several slots."""
    dense, engine = pair_of_models
    want_block = dense.evoformer.template_embedding.single_template_embedding.template_embedding_iteration[0]
    got_block = engine.evoformer.template_embedding.single_template_embedding.template_embedding_iteration[0]
    assert got_block.supports_template_batch
    x = _pair_inputs(20, 64, 1)
    slots = torch.randn(3, 20, 20, 64)
    with torch.no_grad():
        want = torch.stack([want_block(slot, pair_mask=x["pair_mask"]) for slot in slots])
        got = graph.run_block(got_block, slots, x["pair_mask"])
    assert got.shape == slots.shape
    torch.testing.assert_close(got[:, :-3, :-3], want[:, :-3, :-3], **TOLERANCE)


def test_confidence_block_takes_samples_in_one_call(pair_of_models):
    dense, engine = pair_of_models
    x = _pair_inputs(16, 128, 384)
    pairs = torch.randn(2, 16, 16, 128)
    singles = torch.randn(2, 16, 384)
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
    generator = torch.Generator().manual_seed(3)
    x = _pair_inputs(20, 128, 1)
    msa = torch.randn(6, 20, 64, generator=generator)
    msa_mask = (torch.rand(6, 20, generator=generator) > 0.2).float() * x["seq_mask"]
    msa_mask[0] = x["seq_mask"]
    with torch.no_grad():
        want_msa, want_pair = dense.evoformer.msa_stack[0](msa.clone(), x["pair"], msa_mask, x["pair_mask"])
        got_msa, got_pair = engine.evoformer.msa_stack[0](msa.clone(), x["pair"], msa_mask, x["pair_mask"])
    torch.testing.assert_close(got_msa[:, :-3], want_msa[:, :-3], **TOLERANCE)
    torch.testing.assert_close(_valid(got_pair, x["pair_mask"]), _valid(want_pair, x["pair_mask"]), **TOLERANCE)


def test_diffusion_transformer_and_conditioning_transitions_match_dense(pair_of_models):
    dense, engine = pair_of_models
    generator = torch.Generator().manual_seed(5)
    length = 24
    act = torch.randn(2, length, 768, generator=generator)
    single = torch.randn(2, length, 384, generator=generator)
    pair = torch.randn(length, length, 128, generator=generator)
    mask = torch.ones(length)
    mask[-3:] = 0
    key_mask = (mask > 0)[None]
    with torch.no_grad():
        want = dense.diffusion_head.transformer(act.clone(), mask, single, pair, key_mask=key_mask)
        got = engine.diffusion_head.transformer(act.clone(), mask, single, pair, key_mask=key_mask)
        x = torch.randn(length, length, 128, generator=generator)
        want_t = x + dense.diffusion_head.pair_transition_0(x)
        got_t = engine.diffusion_head.pair_transition_0(x)
    torch.testing.assert_close(got[:, :-3], want[:, :-3], **TOLERANCE)
    torch.testing.assert_close(got_t, want_t, **TOLERANCE)


@pytest.mark.parametrize("atoms", [130, 200])
def test_atom_transformer_matches_dense(pair_of_models, atoms):
    """The windowed atom transformer: 32 queries against 128 keys slid inside the real atoms. (Under 128 real atoms the dense AF3 mask
    form lets a real query read the padded keys, the engine block switches them off: the two differ there by construction.)"""
    from miniworld_engine.modules.functional import slide_starts

    dense, engine = pair_of_models
    generator = torch.Generator().manual_seed(7)
    windows = (atoms + 31) // 32
    total = windows * 32
    flat = torch.zeros(total)
    flat[:atoms] = 1
    queries_mask = flat.reshape(windows, 32)
    start = slide_starts(torch.tensor([atoms]), windows, 128)[0]
    index = start[:, None] + torch.arange(128)
    valid = (index >= 0) & (index < atoms)
    gather = atom_layout.GatherInfo(index.clamp(0, total - 1), valid, (windows, 32))
    keys_mask = atom_layout.convert(gather, queries_mask, layout_axes=(-2, -1))
    act = torch.randn(windows, 32, 128, generator=generator) * queries_mask[..., None]
    cond = torch.randn(windows, 32, 128, generator=generator)
    keys_cond = atom_layout.convert(gather, cond, layout_axes=(-3, -2))
    pair = torch.randn(windows, 32, 128, 16, generator=generator)
    want_stack = dense.evoformer_conditioning.atom_transformer_encoder
    got_stack = engine.evoformer_conditioning.atom_transformer_encoder
    with torch.no_grad():
        want = want_stack(act.clone(), queries_mask, gather, keys_mask, cond, keys_cond, pair)
        got = got_stack(act.clone(), queries_mask, gather, keys_mask, cond, keys_cond, pair)
    torch.testing.assert_close(got * queries_mask[..., None], want * queries_mask[..., None], **TOLERANCE)


# ------------------------------------------------------------------------------------------------------------------ the converter
def test_the_converter_demands_every_released_key_is_read_or_a_known_discard(pair_of_models):
    dense, engine = pair_of_models
    state = dict(dense.state_dict())
    state["evoformer.trunk_pairformer.0.surprise.weight"] = torch.zeros(1)
    with pytest.raises(miniworld.ConversionError, match="surprise"):
        miniworld.convert_state(state, engine)
    state = dict(dense.state_dict())
    del state["evoformer.trunk_pairformer.0.pair_transition.transition2.weight"]
    with pytest.raises(KeyError):
        miniworld.convert_state(state, engine)


def test_known_discards_are_the_softmax_invariant_offsets(pair_of_models):
    from foldforge.models.checkpoints.legacy.released import is_discarded

    dense, _ = pair_of_models
    discarded = {k for k in dense.state_dict() if is_discarded(k) and "k_adaptive_layernorm" not in k}  # (those are read where the block has a K norm)
    assert "evoformer.msa_stack.0.msa_attention1.pair_norm.bias" in discarded
    assert "evoformer.trunk_pairformer.0.single_pair_logits_norm.bias" in discarded
    assert all(k.endswith(".bias") for k in discarded)


def test_miniworld_file_round_trip(pair_of_models, tmp_path):
    dense, engine = pair_of_models
    state = miniworld.convert_state(dense.state_dict(), engine)
    path = tmp_path / "x.miniworld.pt"
    torch.save({"format": miniworld.FORMAT, "version": miniworld.VERSION, "family": dense.spec.family, "source": "t", "state_dict": state}, path)
    fresh = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=small(dense.spec.family))
    report = miniworld.load_into(fresh, path, family=dense.spec.family)
    assert report["format_version"] == miniworld.VERSION
    for key, value in engine.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0, atol=0)
    assert set(fresh.state_dict()) == set(engine.state_dict()) | set(miniworld.LOOSE)
    with pytest.raises(ValueError, match="family"):
        miniworld.load_into(fresh, path, family="boltz2")
    torch.save({"format": "other", "state_dict": {}}, tmp_path / "y.miniworld.pt")
    with pytest.raises(ValueError, match="format"):
        miniworld.read(tmp_path / "y.miniworld.pt")
