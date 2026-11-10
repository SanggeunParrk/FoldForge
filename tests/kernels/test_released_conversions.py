"""``released.py``: the released (dense) layout converts to the engine classes' own and the engine class computes the same thing.

Each test builds a dense FoldForge module with random weights (fp32, CPU), converts its state dict with the pure-tensor helpers, loads it
strictly into the engine class on the ``pytorch`` backend, and compares the outputs where the pair is real. The checks that cannot be a
parity (the atom transformers need the trunked atom layout) assert that the conversion loads strictly and consumes every released key
except the declared discards.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from foldforge.models.checkpoints.legacy import released
from foldforge.models.checkpoints.legacy.attention import (
    GridSelfAttention,
    MSAAttention,
)
from foldforge.models.checkpoints.legacy.primitives import OuterProductMean, Transition
from foldforge.models.checkpoints.legacy.released import (
    load_atom_stack,
    load_token_stack,
)
from foldforge.models.checkpoints.legacy.routing import triangle_residual
from foldforge.models.checkpoints.legacy.transformer import DiffusionTransformer
from foldforge.models.checkpoints.legacy.triangle_multiplication import (
    TriangleMultiplication,
)
from foldforge.modules.dense.diffusion_transformer import DiffusionCrossAttTransformer
from foldforge.modules.dense.spec import ALPHAFOLD3, SPECS
from foldforge.modules.engine import modules as factory
from foldforge.modules.engine.transformers import atom_dit_stack, token_dit_stack

L = 12


def randomize(module: torch.nn.Module, std: float = 0.15) -> torch.nn.Module:
    torch.manual_seed(3)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.ndim > 1:
                p.normal_(0, std)
            elif name.endswith("weight"):
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0, 0.1)
    return module


def masks():
    torch.manual_seed(9)
    residues = torch.ones(L, dtype=torch.bool)
    residues[-3:] = False
    return residues, residues[:, None] & residues[None, :]


def close(actual, expected, where=None, tol=2e-5):
    if where is not None:
        actual, expected = actual[where], expected[where]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication(outgoing):
    dense = randomize(TriangleMultiplication(32, _outgoing=outgoing)).eval()
    engine = factory.triangle_multiplication(32, outgoing=outgoing, precision="fp32")
    released.load(engine, released.triangle_multiplication_from_released(dense.state_dict(), outgoing=outgoing))
    x = torch.randn(L, L, 32)
    residues, mask = masks()
    with torch.no_grad():
        expected = triangle_residual(dense, x, mask)
        actual = engine(x[None], residues[None])[0]
    close(actual, expected, mask)


def test_transition():
    dense = randomize(Transition(24, 4)).eval()
    engine = factory.transition(24, 4, precision="fp32")
    released.load(engine, released.transition_from_released(dense.state_dict()))
    x = torch.randn(2, L, 24)
    with torch.no_grad():
        close(engine(x), x + dense(x))


@pytest.mark.parametrize("ending", [False, True])
def test_triangle_attention_af3(ending):
    dense = randomize(GridSelfAttention(32, 4, transpose=ending, spec=ALPHAFOLD3)).eval()
    engine = factory.triangle_attention(32, 4, starting=not ending, model="af3", precision="fp32")
    released.load(engine, released.triangle_attention_from_released(dense.state_dict()))
    x = torch.randn(L, L, 32)
    residues, mask = masks()
    with torch.no_grad():
        close(engine(x[None], residues[None])[0], dense.residual(x, mask), mask)


def test_msa_pair_weighted_averaging_discards_the_pair_norm_bias():
    dense = randomize(MSAAttention(16, 32, 4)).eval()
    engine = factory.msa_pair_weighted_averaging(16, 32, 4, precision="fp32")
    state = dense.state_dict()
    converted = released.msa_pair_weighted_averaging_from_released(state)
    assert released.is_discarded("pair_norm.bias") and "ln_pair.bias" not in converted
    released.load(engine, converted)
    msa, pair = torch.randn(5, L, 16), torch.randn(L, L, 32)
    residues, _ = masks()
    msa_mask = (torch.rand(5, L) > 0.1) & residues[None]
    msa_mask[0] = residues
    with torch.no_grad():
        expected = dense.residual(msa, msa_mask.float(), pair)
        actual = engine(msa[None], pair[None], msa_mask.any(0)[None])[0]
    close(actual, expected, (slice(None), residues))


@pytest.mark.parametrize("bias_after_norm", [False, True])
def test_outer_product_mean(bias_after_norm):
    """Dense ``bias_after_norm=False`` (AF3: the bias before the divide) is the engine's ``normalize_before_proj=False`` with AF3's
    ``n + 1e-3`` divisor; ``True`` is ``normalize_before_proj=True`` (divisor ``clamp(n, 1)`` in the dense ``clamped_norm`` form)."""
    dense = randomize(OuterProductMean(16, 32, 6, bias_after_norm=bias_after_norm, clamped_norm=bias_after_norm), 0.2).eval()
    engine = factory.outer_product_mean(
        16, 32, 6, normalize_before_proj=bias_after_norm, count_offset=None if bias_after_norm else 1e-3, precision="fp32"
    )
    released.load(engine, released.outer_product_mean_from_released(dense.state_dict()))
    msa = torch.randn(5, L, 16)
    residues, _ = masks()
    mask = ((torch.rand(5, L) > 0.1) & residues[None]).float()
    mask[0] = residues.float()
    real = residues[:, None] & residues[None, :]
    with torch.no_grad():
        expected = dense(msa, mask)
        actual = engine(msa[None], (mask > 0)[None])[0]
    close(actual, expected, real, tol=1e-4)


def test_grouped_opm_flattens_like_the_dense_product():
    dense = randomize(OuterProductMean(16, 32, 4, groups=2), 0.2).eval()
    converted = released.outer_product_mean_from_released(dense.state_dict())
    assert converted["to_out.weight"].shape == (32, 2 * 4 * 4)
    assert "product_norm.weight" in converted


@pytest.mark.parametrize("per_block_pair", [False, True])
def test_token_diffusion_transformer(per_block_pair):
    spec = dataclasses.replace(ALPHAFOLD3, per_block_pair_layer_norm=per_block_pair)
    dense = randomize(DiffusionTransformer(64, 24, 16, 16, 4, spec=spec), 0.08).eval()
    stack = token_dit_stack("alphafold3", n_blocks=4, d_single=64, d_cond=24, d_pair=16, n_head=16, precision="fp32")
    load_token_stack(stack, dense.state_dict(), n_head=16, super_block_size=4)
    act, single, pair = torch.randn(3, L, 64), torch.randn(3, L, 24), torch.randn(L, L, 16)
    residues, _ = masks()
    with torch.no_grad():
        expected = dense(act, residues, single, pair)
        actual = stack(act, residues, single, pair)
    close(actual, expected, (slice(None), residues), tol=2e-4)


def test_token_stack_state_is_the_engine_layout():
    dense = randomize(DiffusionTransformer(64, 24, 16, 16, 8, spec=ALPHAFOLD3)).eval()
    blocks = released.token_dit_blocks_from_released(dense.state_dict(), n_head=16, super_block_size=4)
    assert len(blocks) == 8
    # the ONE shared pair norm is broadcast to every block, the shared projection is sliced per block
    first = dense.state_dict()["pair_input_layer_norm.weight"]
    assert all(torch.equal(b["attention.ln_pair.weight"], first) for b in blocks)
    assert torch.equal(blocks[5]["attention.to_bias.weight"], dense.state_dict()["pair_logits_projection.1.weight"][16:32])
    assert all(not any("ln_pair.bias" in k for k in b) for b in blocks)


@pytest.mark.parametrize("family", ["alphafold3", "protenix1", "boltz2", "chai1", "rosettafold3"])
def test_atom_diffusion_transformer_loads_strictly(family):
    dense = randomize(DiffusionCrossAttTransformer(spec=SPECS[family])).eval()
    if family == "rosettafold3":  # the release has no query bias: its converted checkpoint carries zeros
        with torch.no_grad():
            for attention in dense.cross_attention:
                attention.q_projection.bias.zero_()
    if family in ("boltz2", "chai1", "rosettafold3"):  # one AdaLN for queries and keys / values: the release ties them, the loader demands it
        parameters = dict(dense.named_parameters())
        with torch.no_grad():
            for name, p in parameters.items():
                if "k_adaptive_layernorm" in name:
                    p.copy_(parameters[name.replace("k_adaptive_layernorm", "q_adaptive_layernorm")])
    kwargs = {"adaln_eps": 1e-5} if family == "chai1" else {}
    stack = atom_dit_stack(family, n_blocks=3, precision="fp32", **kwargs)
    load_atom_stack(stack, dense.state_dict(), n_head=4)
    shared = stack.blocks[0].kv_norm == "shared"
    assert shared == (not any(k.endswith("ada_ln_kv.to_scale.weight") for k in stack.state_dict()))
