from support import REFERENCE_ROOT, production_module

"""Independent released-code comparisons for the second unification pass."""

# ruff: noqa: SLF001 - lazy model imports and explicit module-state checks
import __future__

import ast
import copy
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from foldforge.models.checkpoints.legacy.reference.attention import (
    dot_product_attention,
)
from foldforge.models.checkpoints.legacy.reference.composition import msa_row_update
from foldforge.models.checkpoints.legacy import _DenseBlocks
from foldforge.modules.engine import graph as engine_graph
from foldforge.modules.engine.configure import activation_autocast, configure_model

GPU_PARITY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AF-family GPU parity"
)

ORACLES = json.loads((REFERENCE_ROOT / "forward_oracles.json").read_text())


def _oracle_gated_linear_unit(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """SwiGLU of one packed projection, as the frozen AF3 bodies call it."""
    a, b = torch.matmul(x.to(weight.dtype), weight).chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def original_module(family, path):
    source_path = Path(production_module(family, path).__file__)
    text = source_path.read_text()
    lines = text.splitlines(keepends=True)
    replacements = []
    frozen = ORACLES[f"{family}/{path}"]["methods"]
    for cls in ast.parse(text).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            key = f"{cls.name}.{getattr(fn, 'name', '')}"
            if isinstance(fn, ast.FunctionDef) and key in frozen:
                replacements.append((fn.lineno - 1, fn.end_lineno, frozen[key]))
    # A frozen method whose class moved to another file (the converter-only dense token transformer is in `legacy/transformer.py`) is
    # checked where it now lives; at least one frozen body must be in this file.
    assert replacements
    for start, end, body in sorted(replacements, reverse=True):
        lines[start:end] = body.splitlines(keepends=True)
    module = types.ModuleType(f"foldforge._reference_{family}_{source_path.stem}")
    module.__file__ = str(source_path)
    module.__package__ = production_module(family, path).__package__
    # Dataclasses resolve postponed annotations through the registered module.
    sys.modules[module.__name__] = module
    # The frozen bodies call the pre-unification `fastnn` primitives; the oracle
    # owns them.
    module.fastnn = types.SimpleNamespace(
        dot_product_attention=dot_product_attention,
        gated_linear_unit=_oracle_gated_linear_unit,
    )
    # Only versioned local test oracles are executed; production imports are untouched.
    exec(  # noqa: S102 - versioned local numerical oracle
        compile(
            "".join(lines),
            str(source_path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        module.__dict__,
    )
    return module


def randomize(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.035)
    return model


@GPU_PARITY
def test_diffusion_composition_against_original():
    torch.manual_seed(17)
    path = "nn/diffusion_transformer.py"
    oldmod = original_module("af3", path)
    newmod = production_module("af3", path)
    args = {
        "c_act": 128,
        "c_single_cond": 128,
        "c_pair_cond": 16,
        "num_head": 16,
        "num_blocks": 4,
    }
    old = randomize(oldmod.DiffusionTransformer(**args))
    new = newmod.DiffusionTransformer(**args)
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch", torch.bfloat16, "cuda")
    new = configure_model(new, "pytorch", torch.bfloat16, "cuda")
    a = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.randn_like(a)
    z = torch.randn(24, 24, 16, device="cuda", dtype=torch.bfloat16)
    mask = torch.rand(24, device="cuda") > 0.2
    with torch.inference_mode(), activation_autocast(new):
        expected = old(a.clone(), mask, s, z)
        actual = new(a.clone(), mask, s, z)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("last", [False, True])
@GPU_PARITY
def test_msa_order_against_original(last):  # noqa: ARG001 - the block has no last-block variant
    torch.manual_seed(19)
    path = "nn/pairformer.py"
    oldmod = original_module("af3", path)
    newmod = production_module("af3", path)
    old = randomize(oldmod.EvoformerBlock())
    new = newmod.EvoformerBlock()
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch", torch.bfloat16, "cuda")
    new = configure_model(new, "pytorch", torch.bfloat16, "cuda")
    m = torch.randn(7, 24, 64, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(24, 24, 128, device="cuda", dtype=torch.bfloat16)
    pairmask = (torch.rand(24, 24, device="cuda") > 0.1).float()
    mask = (torch.rand(7, 24, device="cuda") > 0.2).float()
    with torch.inference_mode(), activation_autocast(new):
        expected = old(m.clone(), z.clone(), mask, pairmask)
        actual = new(m.clone(), z.clone(), mask, pairmask)
    for a, b in zip(actual, expected, strict=True):
        if a is None:
            assert b is None
        else:
            # the released-layout block computes in its parameters' dtype (fp32 masters here), the frozen original in bf16: compare values
            torch.testing.assert_close(a.float(), b.float(), rtol=0.01, atol=0.02)


def test_msa_chunking_preserves_batch_axes_and_gradients():
    x = torch.randn(2, 7, 4, 3, requires_grad=True)
    z = torch.randn(2, 4, 4, 5)
    fn = lambda m, p: 0.2 * m.square() + p.mean()
    transition = lambda m: m + m.sigmoid()
    actual = msa_row_update(
        x,
        z,
        attention_residual=lambda m, p: m + fn(m, p),
        transition_residual=transition,
        chunk_size=3,
    )
    expected = transition(x + fn(x, z))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
    )


@pytest.mark.parametrize("count", [0, 1, 2])
@GPU_PARITY
def test_template_aggregation_and_masked_pair_stack(count):
    from foldforge.data.features.dense import Templates

    torch.manual_seed(57)
    oldmod = original_module("af3", "nn/template.py")
    newmod = production_module("af3", "nn/template.py")
    with engine_graph.use_dense_blocks(_DenseBlocks()):  # the released-layout template stack (the runtime one is engine blocks)
        old = randomize(oldmod.TemplateEmbedding())
        new = newmod.TemplateEmbedding()
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch", torch.bfloat16, "cuda")
    new = configure_model(new, "pytorch", torch.bfloat16, "cuda")
    n = 16
    z = torch.randn(n, n, 128, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(n, n, device="cuda") > 0.15).float()
    chains = torch.arange(n, device="cuda") // 8
    feature = Templates(
        aatype=torch.randint(0, 20, (count, n), device="cuda"),
        atom_positions=torch.randn(count, n, 24, 3, device="cuda"),
        atom_mask=torch.rand(count, n, 24, device="cuda") > 0.1,
    )
    same_chain = (chains[:, None] == chains[None, :]).float()
    with torch.inference_mode(), activation_autocast(new):
        actual = new(z.clone(), copy.deepcopy(feature), mask, same_chain)
        expected = old(z.clone(), copy.deepcopy(feature), mask, same_chain)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert actual.shape == z.shape