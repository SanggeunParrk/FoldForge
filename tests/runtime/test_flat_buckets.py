from __future__ import annotations

from typing import Any

import pytest
import torch
from team_gm.modules.bucketing import pad_axis
from team_gm.modules.checkpoints.denoiser import DiffusionModule
from team_gm.modules.checkpoints.embedders import RelativePositionEncoding
from team_gm.modules.checkpoints.padding import BucketedDenoiser, pad_trunk_features
from team_gm.modules.checkpoints.policy import BOUNDED_MEMORY_POLICY, DEFAULT_POLICY
from team_gm.modules.checkpoints.stacks import MSAModule, PairformerStack
from team_gm.modules.checkpoints.trunk import update_input_feature_dict


def features(tokens=7, atoms=37):
    f = {name: torch.arange(tokens) for name in ("residue_index", "token_index")}
    f.update(
        {
            name: torch.zeros(tokens, dtype=torch.long)
            for name in ("asym_id", "sym_id", "entity_id")
        }
    )
    f.update(
        ref_pos=torch.randn(atoms, 3),
        ref_space_uid=torch.arange(atoms) // 5,
        ref_charge=torch.zeros(atoms),
        ref_mask=torch.ones(atoms),
        ref_element=torch.randn(atoms, 128),
        ref_atom_name_chars=torch.randn(atoms, 4, 64),
        atom_to_token_idx=torch.arange(atoms) % tokens,
        restype=torch.randn(tokens, 32),
        profile=torch.randn(tokens, 32),
        deletion_mean=torch.zeros(tokens),
        token_bonds=torch.zeros(tokens, tokens),
        msa=torch.randint(0, 31, (3, tokens)),
        has_deletion=torch.zeros(3, tokens),
        deletion_value=torch.zeros(3, tokens),
    )
    return update_input_feature_dict(RelativePositionEncoding().generate_relp(f))


def randomize(model):
    with torch.no_grad():
        for _name, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.05)
    return model.eval()


@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("policy", [DEFAULT_POLICY, BOUNDED_MEMORY_POLICY])
def test_pair_stack_real_outputs_unchanged_with_nonzero_weights(policy, converted):
    torch.manual_seed(241)
    model = randomize(
        PairformerStack(n_blocks=2, n_heads=4, c_s=32, c_z=16, policy=policy)
    )
    if converted:
        from team_gm.modules.checkpoints.pairformer import install_pairformers

        model.foldforge_load_report = {}
        install_pairformers(model)
    s, z = torch.randn(7, 32), torch.randn(7, 7, 16)
    with torch.no_grad():
        expected = model(s, z, None)
        model.inference_bucketing = True
        actual = model(s, z, None)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("policy", [DEFAULT_POLICY, BOUNDED_MEMORY_POLICY])
def test_denoiser_masks_atoms_tokens_and_preserves_sample_shape(policy, cached):
    torch.manual_seed(271)
    model = randomize(
        DiffusionModule(
            c_atom=32,
            c_atompair=8,
            c_token=32,
            c_s=32,
            c_z=16,
            c_s_inputs=32,
            atom_encoder={"n_blocks": 1, "n_heads": 4},
            transformer={"n_blocks": 1, "n_heads": 4},
            atom_decoder={"n_blocks": 1, "n_heads": 4},
            policy=policy,
        )
    )
    f = features()
    context = {
        "input_feature_dict": f,
        "s_inputs": torch.randn(7, 32),
        "s_trunk": torch.randn(7, 32),
        "z_trunk": torch.randn(7, 7, 16),
        "pair_z": None,
        "p_lm": None,
        "c_l": None,
    }
    if cached:
        with torch.no_grad():
            context["pair_z"] = model.diffusion_conditioning.prepare_cache(
                f["relp"], context["z_trunk"], inplace_safe=False
            )
            context["p_lm"], context["c_l"] = (
                model.atom_attention_encoder.prepare_cache(
                    **{
                        k: f[k]
                        for k in (
                            "ref_pos",
                            "ref_charge",
                            "ref_mask",
                            "ref_element",
                            "ref_atom_name_chars",
                            "atom_to_token_idx",
                            "d_lm",
                            "v_lm",
                            "pad_info",
                        )
                    },
                    r_l=True,
                    z=context["pair_z"],
                    inplace_safe=False,
                )
            )
            context["z_trunk"] = None
    x = torch.randn(2, 37, 3)
    sigma = torch.full((2,), 1.0)
    with torch.no_grad():
        saved_cache = None if context["p_lm"] is None else context["p_lm"].clone()
        expected = model(x, sigma, **context)
        if saved_cache is not None:
            torch.testing.assert_close(context["p_lm"], saved_cache, atol=0, rtol=0)
        bucketed = BucketedDenoiser(model)
        actual = bucketed(x, sigma, **context)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert bucketed.shape.atom_bucket == 1024
    assert bucketed.shape.token_bucket == 128


@pytest.mark.parametrize("policy", [DEFAULT_POLICY, BOUNDED_MEMORY_POLICY])
def test_msa_padding_keeps_sample_rng_and_excludes_empty_rows(policy):
    torch.manual_seed(231)
    model = randomize(
        MSAModule(
            n_blocks=2,
            c_m=16,
            c_z=16,
            c_s_inputs=32,
            msa_configs={"strategy": "random", "sample_cutoff": {"test": 3}},
            policy=policy,
        )
    )
    f = features()
    s, z = torch.randn(7, 32), torch.randn(7, 7, 16)
    padded, count = pad_trunk_features(f)
    token = padded["bucket_token_mask"].float()
    mask = token[:, None] * token[None, :]
    with torch.no_grad():
        torch.manual_seed(51)
        expected = model(f, z, s, None)
        after = torch.get_rng_state()
        torch.manual_seed(51)
        actual = model(
            padded, pad_axis(pad_axis(z, 0, 128), 1, 128), pad_axis(s, 0, 128), mask
        )[:count, :count]
        new_after = torch.get_rng_state()
    assert torch.equal(after, new_after)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_template_axes_are_explicit_pair_masks():
    f = features()
    f.update(
        template_aatype=torch.ones(2, 7, dtype=torch.long),
        template_atom_positions=torch.randn(2, 7, 24, 3),
        template_atom_mask=torch.ones(2, 7, 24),
        template_distogram=torch.randn(2, 7, 7, 39),
        template_unit_vector=torch.randn(2, 7, 7, 3),
        template_pseudo_beta_mask=torch.ones(2, 7, 7),
        template_backbone_frame_mask=torch.ones(2, 7, 7),
    )
    padded, _ = pad_trunk_features(f)
    for name in (
        "template_distogram",
        "template_unit_vector",
        "template_pseudo_beta_mask",
        "template_backbone_frame_mask",
    ):
        assert padded[name].shape[1:3] == (128, 128)
        torch.testing.assert_close(padded[name][:, :7, :7], f[name])
    assert padded["template_aatype"].shape == (2, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reuse")
def test_two_real_shapes_share_one_denoiser_graph():
    from types import SimpleNamespace

    from team_gm.modules.execution import ExecutedCallable

    class Echo:
        def __call__(
            self, x_noisy, t_hat_noise_level, **_context: object
        ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
            return x_noisy * t_hat_noise_level[:, None, None]

    executed = ExecutedCallable(
        Echo(),
        SimpleNamespace(compile=False, cuda_graph=True, max_graphs=1),
        "bucket-test",
    )
    bucket = BucketedDenoiser(executed)
    for tokens, atoms in ((7, 37), (9, 45)):
        f = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in features(tokens, atoms).items()
        }
        f = update_input_feature_dict(f)
        context = {
            "input_feature_dict": f,
            "s_inputs": torch.randn(tokens, 32, device="cuda"),
            "s_trunk": torch.randn(tokens, 32, device="cuda"),
            "z_trunk": torch.randn(tokens, tokens, 16, device="cuda"),
            "pair_z": None,
            "p_lm": None,
            "c_l": None,
        }
        x = torch.randn(2, atoms, 3, device="cuda")
        sigma = torch.tensor([0.5, 2.0], device="cuda")
        with torch.no_grad():
            actual = bucket(x, sigma, **context)
        torch.testing.assert_close(actual, x * sigma[:, None, None], atol=0, rtol=0)
    assert executed.captures == 1
    assert executed.replays == 2


@pytest.mark.parametrize("policy", [DEFAULT_POLICY, BOUNDED_MEMORY_POLICY])
def test_template_padding_preserves_multichain_embeddings(policy):
    from team_gm.modules.checkpoints.stacks import TemplateEmbedder

    torch.manual_seed(291)
    model = randomize(TemplateEmbedder(n_blocks=2, c=16, c_z=16, policy=policy))
    f = features()
    f["asym_id"][4:] = 1
    f.update(
        template_aatype=torch.randint(0, 31, (2, 7)),
        template_distogram=torch.randn(2, 7, 7, 39),
        template_unit_vector=torch.randn(2, 7, 7, 3),
        template_pseudo_beta_mask=torch.randint(0, 2, (2, 7, 7)).float(),
        template_backbone_frame_mask=torch.randint(0, 2, (2, 7, 7)).float(),
    )
    z = torch.randn(7, 7, 16)
    padded, count = pad_trunk_features(f)
    valid = padded["bucket_token_mask"].float()
    with torch.no_grad():
        expected = model(f, z)
        actual = model(
            padded,
            pad_axis(pad_axis(z, 0, 128), 1, 128),
            pair_mask=valid[:, None] * valid[None, :],
        )[:count, :count]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_undefined_chain_pae_is_json_null_but_other_nonfinite_values_fail(tmp_path):
    import json
    import math

    from foldforge.models.io.confidence import summary_for_json
    from foldforge.models.io.output import Decoded, write_output
    from foldforge.prediction import Prediction

    source = [
        {
            "chain_pair_pae_mean": torch.tensor([[1.5, float("nan")]]),
            "chain_pair_pae_min": torch.tensor([[1.0, float("nan")]]),
            "ptm": torch.tensor(0.75),
        }
    ]
    encoded = summary_for_json(source)
    assert encoded[0]["chain_pair_pae_mean"] == [[1.5, None]]
    assert torch.isnan(source[0]["chain_pair_pae_mean"][0, 1])
    result = Decoded(
        "ligand",
        Prediction(torch.zeros(1, 2, 3)),
        ["data_test\n"],
        {"confidence": encoded},
    )
    write_output(result, tmp_path, expected_samples=1)
    assert json.loads((tmp_path / "ligand.json").read_text())["confidence"] == encoded
    for key in ("ptm", "chain_pair_pae_mean"):
        for value in (float("nan"), float("inf")):
            if key == "chain_pair_pae_mean" and math.isnan(value):
                continue
            invalid = summary_for_json([{key: torch.tensor(value)}])
            with pytest.raises(ValueError, match="Out of range float values"):
                json.dumps(invalid, allow_nan=False)


def test_repeated_forward_releases_previous_prediction(monkeypatch):
    import weakref

    from foldforge.models.execution import measured_forward

    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda: torch.zeros(1, dtype=torch.uint8)
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _state: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1)
    previous = []

    def forward() -> Any:
        if previous:
            assert previous[-1]() is None
        value = torch.ones(2)
        previous.append(weakref.ref(value))
        return value

    result, report = measured_forward(forward, 1)
    assert previous[-1]() is result
    assert len(report["model_seconds_warm"]) == 1


@pytest.mark.parametrize("starting", [False, True])
def test_triangle_score_budget_chunks_without_changing_values(monkeypatch, starting):
    from team_gm.modules.checkpoints import padding
    from team_gm.modules.checkpoints.triangle import TriangleAttention

    torch.manual_seed(312)
    layer = randomize(TriangleAttention(16, 8, 4, starting=starting))
    x = torch.randn(1, 7, 7, 16)
    mask = torch.ones(1, 7, 7)
    mask[:, -2:, :] = 0
    mask[:, :, -2:] = 0
    budget_function = padding.triangle_score_chunk
    with torch.no_grad():
        expected = layer(x, mask, inplace_safe=False)
        layer.inference_bucketing = True
        monkeypatch.setattr(
            padding,
            "triangle_score_chunk",
            lambda n, h, size, requested: budget_function(
                n, h, size, requested, budget_bytes=2 * n * n * h * 4
            ),
        )
        actual = layer(x, mask, inplace_safe=False)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for n in (1024, 2048, 4096, 8192):
        rows = budget_function(n, 16, 4, 96)
        assert rows * n * n * 16 * 4 <= max(1 << 29, n * n * 16 * 4)


def test_chain_pae_with_only_index_zero_is_defined():
    from foldforge.eval.confidence import calculate_chain_pair_pae

    scores = calculate_chain_pair_pae(
        token_pair_pae=torch.tensor([[[2.5, 3.0], [4.0, 5.0]]]),
        token_has_frame=torch.tensor([True, False]),
        asym_id=torch.tensor([0, 1]),
    )
    assert scores["chain_pair_pae_min"][0, 0, 0] == 2.5
    torch.testing.assert_close(
        scores["chain_pair_pae_mean"][0, 0, 0], torch.tensor(2.5), atol=1e-5, rtol=1e-5
    )
    assert torch.isnan(scores["chain_pair_pae_min"][0, 1, 1])


def test_next_trunk_releases_previous_request_conditions():
    import weakref
    from types import SimpleNamespace

    from team_gm.modules.checkpoints.trunk import RecycledTrunk

    bucket = BucketedDenoiser(lambda **kwargs: kwargs["x_noisy"])
    padded_pair = torch.ones(128, 128, 16)
    previous = weakref.ref(padded_pair)
    bucket.context = {"z_trunk": padded_pair}
    bucket.sources = (padded_pair,)
    bucket.features = {}
    del padded_pair

    class Trunk(RecycledTrunk):
        inference_bucketing = True
        training = False
        diffusion_module = SimpleNamespace(forward=bucket)

        def _get_pairformer_output(
            self,
            features,
            N_cycle,  # noqa: ARG002, N803 - shared checkpoint signature
            **_kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert previous() is None
            n = len(features["residue_index"])
            return torch.zeros(n, 4), torch.zeros(n, 4), torch.zeros(n, n, 4)

    Trunk().eval().get_pairformer_output(features(), 1)
    assert bucket.context is bucket.sources is bucket.features is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph pool lifecycle")
def test_graph_eviction_recaptures_and_keeps_outputs_independent():
    from types import SimpleNamespace

    from team_gm.modules.execution import ExecutedCallable

    def with_temporary(x) -> torch.Tensor:
        return (x.repeat(8192) + 1).mean().expand_as(x).clone()

    call = ExecutedCallable(
        with_temporary,
        SimpleNamespace(compile=False, cuda_graph=True, max_graphs=1),
        "eviction-test",
    )
    with torch.no_grad():
        first = call(torch.zeros(128, device="cuda"))
        assert len(call.graphs) == 1
        assert next(iter(call.graphs.values())).retained_bytes >= 4 * 2**20
        call.release_large_graphs(max_bytes=0)
        assert not call.graphs
        assert call.evictions == 1
        second = call(torch.full((128,), 3.0, device="cuda"))
    assert call.captures == call.replays == 2
    torch.testing.assert_close(first, torch.ones_like(first), atol=0, rtol=0)
    torch.testing.assert_close(second, torch.full_like(second, 4), atol=0, rtol=0)


def test_large_pair_tokens_preserve_mask_and_bias():
    from team_gm.modules.checkpoints.padding import pad_pair_stack

    tokens, extent = 1140, 1152
    single = torch.randn(tokens, 3)
    pair = torch.randn(tokens, tokens, 2)
    bias = torch.randn(tokens, tokens)
    ps, pz, mask, pb = pad_pair_stack(single, pair, None, bias)
    assert ps.shape == (extent, 3)
    assert pz.shape == (extent, extent, 2)
    torch.testing.assert_close(ps[:tokens], single)
    torch.testing.assert_close(pz[:tokens, :tokens], pair)
    torch.testing.assert_close(pb[:tokens, :tokens], bias)
    assert mask[:tokens, :tokens].all()
    assert not mask[tokens:, :].any()
    assert not mask[:, tokens:].any()
    assert not pz[tokens:, :].any()
    assert not pz[:, tokens:].any()


def test_large_trunk_and_denoiser_use_same_token_extent():
    f = features(tokens=1140, atoms=1200)
    padded, real_tokens = pad_trunk_features(f)
    assert real_tokens == 1140
    assert padded["pad_info"]["n_token"] == 1152
    assert padded["bucket_token_mask"].sum() == 1140
    assert padded["msa"].shape[-1] == 1152
    context = {
        "input_feature_dict": f,
        "s_inputs": torch.randn(1140, 3),
        "s_trunk": torch.randn(1140, 3),
        "z_trunk": torch.randn(1140, 1140, 2),
    }
    prepared, shape = BucketedDenoiser.prepare(context)
    assert (shape.token_bucket, shape.atom_bucket) == (1152, 2048)
    assert prepared["z_trunk"].shape == (1152, 1152, 2)
    torch.testing.assert_close(prepared["z_trunk"][:1140, :1140], context["z_trunk"])
    bias = prepared["input_feature_dict"]["token_attn_bias"]
    assert (bias[:, 1140:] == -1e9).all()
    assert not bias[:, :1140].any()
    assert prepared["input_feature_dict"]["pad_info"]["atom_mask"].sum() == 1200
