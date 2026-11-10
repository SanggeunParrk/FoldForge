"""Chai-1 and ESMFold2 (``esmfold2``, ``esmfold2-fast``) run on engine classes: parity with the dense released-layout graph.

As ``test_engine_graph.py`` for the af3 family: the dense graph (``spec_for(family, mode)``) and the engine graph are built on
CPU in fp32 with the same random weights, the weights pass through the converter, and every piece the engine graph holds as an engine class is
compared with the dense piece it replaces. What is specific here:

* Chai-1: the parallel pairformer / MSA blocks, the grouped outer product mean, the dual-output confidence triangle attention, the identity
  AdaLN token / atom DiT (``adaln_eps``: fast 1e-5, exact 0.1), the conformer-id atom attention, and what the engine's removal of the
  ``circular`` key window means for the exact mode (``test_circular_*``),
* ESMFold2: the pair-only trunk (no triangle attention), the SSM / coda / language-model pair encoder blocks (pair-only blocks), the token
  DiT, and the atom stack that stays FoldForge-own (the engine's ESMFold2 atom block is pair-free).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from foldforge.models.architectures.af3 import AlphaFold3
from foldforge.models.checkpoints import miniworld
from foldforge.models.checkpoints.legacy import dense_model
from foldforge.modules.dense.diffusion_transformer import DiffusionCrossAttTransformer
from foldforge.modules.dense.spec import spec_for
from foldforge.modules.engine import graph, modules
from foldforge.modules.engine.modules import MODEL_KEYS
from foldforge.modules.engine.transformers import (
    AtomDiTStack,
    TokenDiTStack,
    engine_key_window,
)
from foldforge.utils import atom_layout

TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}
CASES = (("chai1", "fast"), ("chai1", "exact"), ("esmfold2", "fast"))


def small(family: str, mode: str):
    spec = spec_for(family, mode)
    return replace(
        spec,
        trunk_layers=1,
        msa_layers=1 if spec.msa_layers else 0,
        confidence_layers=1,
        diffusion_blocks=4,
        template_layers=1 if spec.template_layers else 0,
        coda_layers=1 if spec.coda_layers else 0,
        lm_encoder_layers=1 if spec.lm_encoder_layers else 0,
    )


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


@pytest.fixture(scope="module", params=CASES, ids=lambda c: f"{c[0]}:{c[1]}")
def pair_of_models(request):
    """(dense model, engine model with the dense weights converted into it)."""
    family, mode = request.param
    modules.OPM_COUNT_OFFSET = 1e-3  # AF3's exact divisor (the dense graph's), where the family divides at all
    try:
        dense = dense_model(small(family, mode), num_recycles=1, num_samples=1, diffusion_steps=1)
        randomise(dense, 1)
        if family == "chai1":  # the engine class holds ONE AdaLN for the atoms' queries and keys / values: the release ties them (the converter demands it)
            parameters = dict(dense.named_parameters())
            with torch.no_grad():
                for name, p in parameters.items():
                    if "k_adaptive_layernorm" in name:
                        p.copy_(parameters[name.replace("k_adaptive_layernorm", "q_adaptive_layernorm")])
        dense.diffusion_head.fourier_embeddings.register_buffer("weight", torch.randn(256))
        dense.diffusion_head.fourier_embeddings.register_buffer("bias", torch.randn(256))
        engine = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=small(family, mode))
    finally:
        modules.OPM_COUNT_OFFSET = None
    state = miniworld.convert_state(dense.state_dict(), engine)
    for key in miniworld.LOOSE:
        state.pop(key)
    engine.load_state_dict(state, strict=True)
    return dense.eval(), engine.eval()


def _is_chai(model) -> bool:
    return model.spec.family == "chai1"


def test_ported_families_include_chai_and_esmfold2():
    for family in ("chai1", "esmfold2", "esmfold2-fast"):
        assert family in MODEL_KEYS
        assert family in miniworld.NATIVE_FILES


def test_the_graph_holds_engine_classes_with_engine_keys(pair_of_models):
    dense, engine = pair_of_models
    keys = set(engine.state_dict())
    assert all(p.dtype == torch.float32 for p in engine.parameters())
    assert isinstance(engine.diffusion_head.transformer, TokenDiTStack)
    assert "evoformer.trunk_pairformer.0.tri_multi_outgoing.to_left.weight" in keys
    assert "diffusion_head.transformer.blocks.1.attention.to_bias.weight" in keys
    assert "diffusion_head.pair_transition_0.squeeze.weight" in keys
    assert not any("pair_logits_projection" in k or "pair_input_layer_norm" in k for k in keys if "diffusion_head.transformer" in k)
    decoder = engine.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder
    if _is_chai(engine):
        assert isinstance(decoder, AtomDiTStack)
        assert "confidence_head.confidence_pairformer.0.tri_atten_starting.to_out_t.weight" in keys  # the dual-output confidence attention
        assert "evoformer.msa_stack.0.outer_product_mean.product_norm.weight" in keys  # grouped
        assert "evoformer.trunk_pairformer.0.single_attention.to_query.weight" in keys
        assert "diffusion_head.transformer.blocks.0.attention.ada_ln_in.ln_cond.weight" not in keys  # the identity AdaLN has no cond norm
        assert not any(k.endswith("attention.to_gate.weight") for k in keys if "diffusion_head.transformer" in k)  # no gating query
        assert not any("ada_ln_kv" in k for k in keys if "atom_transformer" in k)  # one shared atom AdaLN
    else:
        # the engine's ESMFold2 atom block is pair-free SWA: this family's atom stack (atom pair + rope) is FoldForge's own
        assert isinstance(decoder, DiffusionCrossAttTransformer)
        assert not any(isinstance(m, AtomDiTStack) for m in engine.modules())
        assert not any("single_attention" in k for k in keys if "trunk_pairformer" in k)  # pair-only trunk


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


def test_trunk_block_matches_dense(pair_of_models):
    """Chai-1: the parallel two-track block (single gate bias, ungated-output triangle attention). ESMFold2: the pair-only block."""
    dense, engine = pair_of_models
    d_pair = engine.evoformer.pair_channel
    x = _pair_inputs(24, d_pair, engine.evoformer.seq_channel)
    with torch.no_grad():
        if engine.spec.trunk_single_track:
            want_pair, want_single = dense.evoformer.trunk_pairformer[0](x["pair"], x["pair_mask"], x["single"].clone(), x["seq_mask"])
            got_pair, got_single = graph.run_block(
                engine.evoformer.trunk_pairformer[0], x["pair"], x["pair_mask"], x["single"].clone(), x["seq_mask"]
            )
            torch.testing.assert_close(_valid(got_single, x["seq_mask"]), _valid(want_single, x["seq_mask"]), **TOLERANCE)
        else:
            want_pair = dense.evoformer.trunk_pairformer[0](x["pair"], pair_mask=x["pair_mask"])
            got_pair = graph.run_block(engine.evoformer.trunk_pairformer[0], x["pair"], x["pair_mask"])
    torch.testing.assert_close(_valid(got_pair, x["pair_mask"]), _valid(want_pair, x["pair_mask"]), **TOLERANCE)


def test_chai_trunk_block_is_the_parallel_one(pair_of_models):
    _, engine = pair_of_models
    if not _is_chai(engine):
        pytest.skip("Chai-1 only")
    block = engine.evoformer.trunk_pairformer[0]
    assert block.config.parallel and block.config.model == "chai1"
    assert block.config.single_gate_bias == 1.0
    assert engine.confidence_head.confidence_pairformer[0].config.model == "chai1_confidence"


def test_pair_only_blocks_match_dense_and_take_a_batch_axis(pair_of_models):
    """ESMFold2's coda / language-model encoder blocks (pair 256, no triangle attention) and Chai-1's template blocks (pair 64): one call
    for several slots."""
    dense, engine = pair_of_models
    if engine.evoformer.template_embedding is not None:
        want_block = dense.evoformer.template_embedding.single_template_embedding.template_embedding_iteration[0]
        got_block = engine.evoformer.template_embedding.single_template_embedding.template_embedding_iteration[0]
        width = 64
        assert got_block.supports_template_batch
    else:
        want_block, got_block, width = dense.evoformer.lm_encoder[0], engine.evoformer.lm_encoder[0], engine.evoformer.pair_channel
    x = _pair_inputs(20, width, 1)
    slots = torch.randn(3, 20, 20, width)
    with torch.no_grad():
        want = torch.stack([want_block(slot, pair_mask=x["pair_mask"]) for slot in slots])
        got = graph.run_block(got_block, slots, x["pair_mask"])
    assert got.shape == slots.shape
    torch.testing.assert_close(got[:, :-3, :-3], want[:, :-3, :-3], **TOLERANCE)


def test_confidence_block_takes_samples_in_one_call(pair_of_models):
    """Chai-1: the dual-output triangle attention (``W_o u + (W_t u)^T``). ESMFold2: pair attention off, the single track on."""
    dense, engine = pair_of_models
    width, single_width = engine.confidence_head.c_pair, engine.confidence_head.c_single
    x = _pair_inputs(16, width, single_width)
    pairs = torch.randn(2, 16, 16, width)
    singles = torch.randn(2, 16, single_width)
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
    """Chai-1: the parallel MSA block with the grouped outer product mean. ESMFold2: the sequential block without pair attention."""
    dense, engine = pair_of_models
    if engine.evoformer.msa_stack is None:
        pytest.skip("no MSA stack")
    if _is_chai(engine):
        assert engine.evoformer.msa_stack[0].parallel
    generator = torch.Generator().manual_seed(3)
    x = _pair_inputs(20, engine.evoformer.pair_channel, 1)
    msa = torch.randn(6, 20, engine.evoformer.msa_channel, generator=generator)
    msa_mask = (torch.rand(6, 20, generator=generator) > 0.2).float() * x["seq_mask"]
    msa_mask[0] = x["seq_mask"]
    with torch.no_grad():
        want_msa, want_pair = dense.evoformer.msa_stack[0](msa.clone(), x["pair"], msa_mask, x["pair_mask"])
        got_msa, got_pair = engine.evoformer.msa_stack[0](msa.clone(), x["pair"], msa_mask, x["pair_mask"])
    torch.testing.assert_close(got_msa[:, :-3], want_msa[:, :-3], **TOLERANCE)
    torch.testing.assert_close(_valid(got_pair, x["pair_mask"]), _valid(want_pair, x["pair_mask"]), **TOLERANCE)


def test_diffusion_transformer_and_conditioning_transitions_match_dense(pair_of_models):
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
        x = torch.randn(length, length, head.pair_channel, generator=generator)
        want_t = x + dense.diffusion_head.pair_transition_0(x)
        got_t = head.pair_transition_0(x)
    # sixteen-wide residual streams of random weights reach |a| ~ 1e2..1e3: fp32 order-of-operations noise is relative to the largest value
    assert (got[:, :-3] - want[:, :-3]).abs().max() <= 5e-5 * want.abs().max() + 2e-4
    torch.testing.assert_close(got_t, want_t, **TOLERANCE)


def test_chai_adaln_epsilon_follows_the_mode(pair_of_models):
    """FoldForge's fast mode is 1e-5, the exact mode (the release's traced ``layer_norm(x, None, None, 0.1)``) is 0.1; both DiT stacks."""
    _, engine = pair_of_models
    if not _is_chai(engine):
        pytest.skip("Chai-1 only")
    exact = engine.spec.atom_key_window == "circular"  # the exact-mode spec
    want = 0.1 if exact else 1e-5
    assert engine.spec.adaptive_norm_eps == want
    stacks = [engine.diffusion_head.transformer, engine.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder]
    for stack in stacks:
        for block in stack.blocks:
            assert block.attention.ada_ln_in.ln_in.eps == want
            assert block.transition.ada_ln_in.ln_in.eps == want


# ------------------------------------------------------------------------------------------------------------------ atom transformer
def _atom_case(atoms: int, window: str, *, seed: int = 7):
    """The inputs of one windowed atom transformer call on ``atoms`` real atoms: 24-atom conformers, the key window per FoldForge's word."""
    from miniworld_engine.modules.functional import slide_starts

    generator = torch.Generator().manual_seed(seed)
    windows = (atoms + 31) // 32
    total = windows * 32
    flat = torch.zeros(total)
    flat[:atoms] = 1
    queries_mask = flat.reshape(windows, 32)
    if window == "circular":
        index = (torch.arange(windows)[:, None] * 32 - 48 + torch.arange(128)) % atoms
        valid = torch.ones_like(index, dtype=torch.bool)
    else:
        start = slide_starts(torch.tensor([atoms]), windows, 128)[0]
        index = start[:, None] + torch.arange(128)
        valid = (index >= 0) & (index < atoms)
    gather = atom_layout.GatherInfo(index.clamp(0, total - 1), valid, (windows, 32))
    keys_mask = atom_layout.convert(gather, queries_mask, layout_axes=(-2, -1))
    uid = torch.zeros(total, dtype=torch.int64)
    uid[:atoms] = torch.arange(atoms) // 24
    uid_q = uid.reshape(windows, 32)
    uid_k = atom_layout.convert(gather, uid_q, layout_axes=(-2, -1))
    pair_mask = (uid_q[:, :, None] == uid_k[:, None, :]) & (queries_mask[:, :, None] > 0) & (keys_mask[:, None, :] > 0)
    act = torch.randn(windows, 32, 128, generator=generator) * queries_mask[..., None]
    cond = torch.randn(windows, 32, 128, generator=generator)
    keys_cond = atom_layout.convert(gather, cond, layout_axes=(-3, -2))
    pair = torch.randn(windows, 32, 128, 16, generator=generator)
    return {
        "act": act, "queries_mask": queries_mask, "gather": gather, "keys_mask": keys_mask, "cond": cond, "keys_cond": keys_cond,
        "pair": pair, "pair_mask": pair_mask, "uid": uid[None], "mask": queries_mask,
    }


def _run_atom_stacks(dense, engine, case):
    want_stack = dense.evoformer_conditioning.atom_transformer_encoder
    got_stack = engine.evoformer_conditioning.atom_transformer_encoder
    with torch.no_grad():
        want = want_stack(case["act"].clone(), case["queries_mask"], case["gather"], case["keys_mask"], case["cond"], case["keys_cond"],
                          case["pair"], pair_mask=case["pair_mask"])
        got = got_stack(case["act"].clone(), case["queries_mask"], case["gather"], case["keys_mask"], case["cond"], case["keys_cond"],
                        case["pair"], engine_inputs=(None, None, case["uid"]))
    mask = case["mask"][..., None]
    return got * mask, want * mask


@pytest.mark.parametrize("atoms", [130, 200])
def test_chai_atom_transformer_matches_dense(pair_of_models, atoms):
    """Chai-1's windowed atom transformer with the same-conformer mask (engine: the conformer ids ``uid``, shared AdaLN, parallel schedule).
    Fast mode slides the window inside the real atoms, as the dense graph does."""
    dense, engine = pair_of_models
    if not _is_chai(engine) or engine.spec.atom_key_window != "slide":
        pytest.skip("Chai-1 fast mode")
    got, want = _run_atom_stacks(dense, engine, _atom_case(atoms, "slide"))
    torch.testing.assert_close(got, want, **TOLERANCE)


def test_circular_key_window_is_the_padded_window_when_conformers_do_not_wrap(pair_of_models):
    """The engine removed Chai-1's ``circular`` key window (the exact mode wrapped the window modulo the real atom count): the exact mode
    now runs ``pad`` + the conformer ids. The two windows hold the same keys wherever a wrapped key would be masked anyway; the
    same-conformer mask masks it unless the conformer spans the wrap-around, which these 24-atom conformers do not. So here the exact
    mode equals the old output; where one conformer spans the ends (a conformer of more atoms than the window half) it does not
    (docs/ENGINE-UNSUPPORTED.md)."""
    dense, engine = pair_of_models
    if not _is_chai(engine) or engine.spec.atom_key_window != "circular":
        pytest.skip("Chai-1 exact mode")
    assert engine_key_window("circular")[0] == "pad" and engine_key_window("circular")[1]
    assert all(block.key_window == "pad" for block in engine.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder.blocks)
    got, want = _run_atom_stacks(dense, engine, _atom_case(200, "circular"))
    torch.testing.assert_close(got, want, **TOLERANCE)


def test_circular_key_window_differs_when_one_conformer_spans_the_ends(pair_of_models):
    """The documented difference of the exact mode: a conformer that spans the ends keeps its wrapped keys in the old ``circular`` window."""
    dense, engine = pair_of_models
    if not _is_chai(engine) or engine.spec.atom_key_window != "circular":
        pytest.skip("Chai-1 exact mode")
    case = _atom_case(200, "circular")
    uid = torch.zeros(case["uid"].shape[-1], dtype=torch.int64)  # ONE conformer for every atom: every wrapped key is a same-conformer key
    uid_q = uid.reshape(case["act"].shape[0], 32)
    uid_k = atom_layout.convert(case["gather"], uid_q, layout_axes=(-2, -1))
    case["pair_mask"] = (uid_q[:, :, None] == uid_k[:, None, :]) & (case["queries_mask"][:, :, None] > 0) & (case["keys_mask"][:, None, :] > 0)
    case["uid"] = uid[None]
    got, want = _run_atom_stacks(dense, engine, case)
    assert (got - want).abs().max() > 1e-3


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


def test_known_discards_are_the_softmax_invariant_offsets_and_the_shared_adaln(pair_of_models):
    from foldforge.models.checkpoints.legacy.released import is_discarded

    dense, engine = pair_of_models
    discarded = {k for k in dense.state_dict() if is_discarded(k)}
    assert "confidence_head.confidence_pairformer.0.single_pair_logits_norm.bias" in discarded
    if _is_chai(engine):
        # Chai-1's atom blocks hold one AdaLN for queries and keys / values: the released K AdaLN is the discard
        assert any("k_adaptive_layernorm" in k and "atom_transformer" in k for k in discarded)
    assert all(k.endswith(".bias") or "k_adaptive_layernorm" in k for k in discarded)


def test_miniworld_file_round_trip(pair_of_models, tmp_path):
    dense, engine = pair_of_models
    family, mode = dense.spec.family, "exact" if dense.spec.atom_key_window == "circular" else "fast"
    state = miniworld.convert_state(dense.state_dict(), engine)
    path = tmp_path / "x.miniworld.pt"
    torch.save({"format": miniworld.FORMAT, "version": miniworld.VERSION, "family": family, "source": "t", "state_dict": state}, path)
    fresh = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=1, spec=small(family, mode))
    report = miniworld.load_into(fresh, path, family=family)
    assert report["format_version"] == miniworld.VERSION
    for key, value in engine.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="family"):
        miniworld.load_into(fresh, path, family="boltz2")
