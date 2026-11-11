"""The ESM-C tower loads from the Transformers-port checkpoint layout, with every weight accounted for.

The published ``esmc-6b`` directory carries ``esmc.layers.{i}.self_attn.q_proj`` ...; the vendored model reads
``transformer.blocks.{i}.attn.layernorm_qkv`` ... . A loader that matched neither left the tower at its random initialisation with a
warning only (every ESMFold2 fold since 2026-10-19 ran on it), so the layout is converted and anything unmatched raises.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from foldforge.modules import language_model as tower
from foldforge.modules.esmc import ESMCConfig, ESMCModel

D_MODEL, HEADS, LAYERS, FFN = 64, 4, 2, 96


def _tiny() -> ESMCModel:
    torch.manual_seed(0)
    config = ESMCConfig(d_model=D_MODEL, n_heads=HEADS, n_layers=LAYERS)
    config.hidden_size, config.intermediate_size, config.num_hidden_layers = D_MODEL, FFN, LAYERS
    model = ESMCModel(config).eval()
    for parameter in model.parameters():  # no all-zero tensors: a mismatched layout must show up as a different value
        torch.nn.init.normal_(parameter, std=0.1)
    return model


def _to_port_layout(model: ESMCModel) -> dict[str, torch.Tensor]:
    """The inverse of ``_esmc_from_port``, written out independently: what the Transformers port stores."""
    state = model.state_dict()
    out = {"esmc.embed_tokens.weight": state["embed.weight"], "esmc.norm.weight": state["transformer.norm.weight"]}
    for i in range(LAYERS):
        src, dst = f"transformer.blocks.{i}.", f"esmc.layers.{i}."
        q, k, v = state[src + "attn.layernorm_qkv.weight"].chunk(3)
        gate, up = state[src + "ffn.fc1_weight"].chunk(2)
        out |= {
            dst + "input_layernorm.weight": state[src + "attn.layernorm_qkv.layer_norm_weight"],
            dst + "input_layernorm.bias": state[src + "attn.layernorm_qkv.layer_norm_bias"],
            dst + "self_attn.q_proj.weight": q,
            dst + "self_attn.k_proj.weight": k,
            dst + "self_attn.v_proj.weight": v,
            dst + "self_attn.o_proj.weight": state[src + "attn.out_proj.weight"],
            dst + "self_attn.q_norm.weight": state[src + "attn.q_ln.weight"],
            dst + "self_attn.k_norm.weight": state[src + "attn.k_ln.weight"],
            dst + "post_attention_layernorm.weight": state[src + "ffn.layer_norm_weight"],
            dst + "post_attention_layernorm.bias": state[src + "ffn.layer_norm_bias"],
            dst + "mlp.gate_proj.weight": gate,
            dst + "mlp.up_proj.weight": up,
            dst + "mlp.down_proj.weight": state[src + "ffn.fc2_weight"],
        }
    out["lm_head.dense.weight"] = torch.zeros(D_MODEL, D_MODEL)  # the masked-LM head the tower does not use
    return {key: value.contiguous().clone() for key, value in out.items()}


def _write(directory, state: dict[str, torch.Tensor], config: ESMCConfig) -> str:
    config.save_pretrained(directory)
    save_file(state, str(directory / "model.safetensors"))
    return str(directory)


def test_port_layout_loads_to_the_same_weights(tmp_path):
    reference = _tiny()
    path = _write(tmp_path, _to_port_layout(reference), reference.config)

    loaded = tower._load_esmc_weights(ESMCModel, path)  # noqa: SLF001

    expected = reference.state_dict()
    got = loaded.state_dict()
    assert expected.keys() == got.keys()
    for key, value in expected.items():
        torch.testing.assert_close(got[key], value, rtol=0, atol=0, msg=key)
    assert not any(bool((p == 0).all()) for p in loaded.parameters()), "a weight was left at zero / uninitialised"


def test_port_layout_gives_the_same_forward(tmp_path):
    reference = _tiny()
    path = _write(tmp_path, _to_port_layout(reference), reference.config)
    loaded = tower._load_esmc_weights(ESMCModel, path)  # noqa: SLF001
    tokens = torch.randint(4, 30, (1, 12))
    with torch.no_grad():
        want = reference(input_ids=tokens, output_hidden_states=True).hidden_states[-1]
        got = loaded(input_ids=tokens, output_hidden_states=True).hidden_states[-1]
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_a_missing_weight_raises_instead_of_staying_random(tmp_path):
    reference = _tiny()
    state = _to_port_layout(reference)
    del state["esmc.layers.1.mlp.down_proj.weight"]
    path = _write(tmp_path, state, reference.config)
    with pytest.raises((ValueError, KeyError)):
        tower._load_esmc_weights(ESMCModel, path)  # noqa: SLF001


def test_a_wrong_shape_raises(tmp_path):
    reference = _tiny()
    state = _to_port_layout(reference)
    state["esmc.layers.0.self_attn.o_proj.weight"] = torch.zeros(D_MODEL + 1, D_MODEL)
    path = _write(tmp_path, state, reference.config)
    with pytest.raises((ValueError, RuntimeError)):
        tower._load_esmc_weights(ESMCModel, path)  # noqa: SLF001
