"""Public inference wiring and full-diffusion integration regressions."""


# GPU and optional ESMC imports stay local to their tests.

import pytest
import torch
from torch import nn

from foldforge.cli import main
from foldforge.models.precision import inference_precision


def test_cli_distinguishes_ports_from_plans(capsys):
    assert main(["models"]) == 0
    rows = capsys.readouterr().out
    assert "esmfold2\timplemented" in rows
    assert "af3\timplemented" in rows
    with pytest.raises(SystemExit) as error:
        main(["fold", "af3"])
    assert error.value.code == 2


def test_native_precision_preserves_exact_norm_weights():
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    with torch.no_grad():
        model[1].weight.fill_(1.001)
    norm_weight = model[1].weight.detach().clone()
    inference_precision(model, torch.device("cpu"), torch.bfloat16)
    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32
    torch.testing.assert_close(model[1].weight, norm_weight, atol=0, rtol=0)
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated GPU")
def test_diffusion_noise_conditioning_matches_pytorch():
    from team_gm.modules.exceptions import ImplementationType

    from foldforge.models.config.esmfold2 import ESMFold2Config
    from foldforge.modules.sequence.diffusion import DiffusionConditioning

    config = ESMFold2Config()
    dm = config.structure_head.diffusion_module
    dm.c_z, dm.c_s_inputs, dm.c_token, dm.fourier_dim = 32, 64, 64, 32
    ours = DiffusionConditioning(config, ImplementationType.MINIWORLD_ENGINE)
    reference = DiffusionConditioning(config, ImplementationType.PYTORCH)
    reference.load_state_dict(ours.state_dict())
    for model in (ours, reference):
        inference_precision(model, torch.device("cuda"), torch.bfloat16)
    single = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(2, 4, 4, 32, device="cuda", dtype=torch.bfloat16)
    sigma = torch.tensor([1.0, 16.0], device="cuda")
    with torch.no_grad():
        actual = ours(sigma, single, pair, pair)
        expected = reference(sigma, single, pair, pair)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0.03, rtol=0.03)


def test_esmc_precision_materializes_rope_and_preserves_fallback_math():
    from transformers.models.esmc.configuration_esmc import ESMCConfig
    from transformers.models.esmc.modeling_esmc import ESMCModel

    from foldforge.modules.esmc import _prepare_esmc_norms

    model = ESMCModel(ESMCConfig(d_model=64, n_heads=4, n_layers=1)).eval()
    # Match compute_lm_hidden_states: explicit chain IDs also select the
    # upstream chain-aware SDPA path on CPU.
    tokens = torch.tensor([[0, 5, 6, 2]])
    with torch.no_grad():
        expected = model(
            tokens, sequence_id=torch.zeros_like(tokens), output_hidden_states=True
        ).last_hidden_state
    _prepare_esmc_norms(model)
    inference_precision(model, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        actual = model(
            tokens, sequence_id=torch.zeros_like(tokens), output_hidden_states=True
        ).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for module in model.modules():
        if hasattr(module, "inv_freq"):
            module.inv_freq = torch.empty_like(module.inv_freq, device="meta")
    inference_precision(model, torch.device("cpu"), torch.bfloat16)
    for module in model.modules():
        if hasattr(module, "inv_freq"):
            assert not module.inv_freq.is_meta
            assert module.inv_freq.dtype == torch.float32
    with torch.no_grad():
        output = model(
            tokens, sequence_id=torch.zeros_like(tokens), output_hidden_states=True
        ).last_hidden_state
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


def test_ca_scoring_does_not_use_distogram_cb_or_ligand_ca():
    from foldforge.models.io.sequence_atoms import predicted_ca

    # Protein CB is deliberately far from CA; ligand also has an atom named CA.
    coords = torch.tensor([[[1.0, 2.0, 3.0], [80.0, 80.0, 80.0], [99.0, 99.0, 99.0]]])
    features = {
        "ref_atom_name_chars": torch.tensor(
            [[[35, 33, 0, 0], [35, 34, 0, 0], [35, 33, 0, 0]]]
        ),
        "atom_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "atom_to_token": torch.tensor([[0, 0, 1]]),
        "distogram_atom_idx": torch.tensor([[1, 2]]),
        "asym_id": torch.tensor([[0, 1]]),
        "residue_index": torch.tensor([[0, 0]]),
    }
    manifest = {"chains": [{"type": "protein", "sequence": "A"}, {"type": "ligand"}]}
    assert predicted_ca(coords, features, manifest) == {
        ("A", 0): ("ALA", (1.0, 2.0, 3.0))
    }
