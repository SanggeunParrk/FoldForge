from support import REFERENCE_ROOT, production_module

"""Independent released-code comparisons for the second unification pass."""

# ruff: noqa: SLF001 - lazy model imports and explicit module-state checks
import __future__

import ast
import copy
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from team_gm.modules.blocks.composition import msa_row_update
from team_gm.modules.checkpoints.af_family import configure_model
from team_gm.modules.checkpoints.conditioning import (
    ReleasedConditionedTransition,
    install_conditioning,
)
from team_gm.modules.precision import NativeLinear

GPU_PARITY = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AF-family GPU parity"
)

ORACLES = json.loads((REFERENCE_ROOT / "forward_oracles.json").read_text())


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
    assert len(replacements) == len(frozen)
    for start, end, body in sorted(replacements, reverse=True):
        lines[start:end] = body.splitlines(keepends=True)
    module = types.ModuleType(f"foldforge._reference_{family}_{source_path.stem}")
    module.__file__ = str(source_path)
    module.__package__ = production_module(family, path).__package__
    # Dataclasses resolve postponed annotations through the registered module.
    sys.modules[module.__name__] = module
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
    from foldforge.data.constants.chemistry import STD_RESIDUES_WITH_GAP

    module.STD_RESIDUES_WITH_GAP = STD_RESIDUES_WITH_GAP
    if family == "opendde":
        # Old formulas named the same residual module differently. These are
        # fixture-only bridges; no released tensor expression is replaced.
        if hasattr(module, "DiffusionTransformerBlock"):
            module.DiffusionTransformerBlock.residual_path = property(
                lambda self: self.drop_path
            )
        if hasattr(module, "MSABlock"):
            # This suite validates the original single-device branch.
            module.MSABlock._maybe_foldcp_mesh = lambda _self: None
    return module


def randomize(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.035)
    return model


@pytest.mark.parametrize("family", ["af3", "protenix", "opendde"])
@pytest.mark.parametrize("backend", ["pytorch", "miniworld"])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("widths", [(128, 128), (768, 384)])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, torch.float32])
@GPU_PARITY
def test_conditioned_transition_checkpoint_mapping(
    family, backend, broadcast, widths, activation_dtype
):
    torch.manual_seed(9)
    mod = original_module(
        family,
        "nn/diffusion_transformer.py"
        if family == "af3"
        else "model/modules/transformer.py",
    )
    source = (
        mod.DiffusionTransition(*widths, use_single_cond=True)
        if family == "af3"
        else mod.ConditionedTransitionBlock(*widths)
    )
    source = configure_model(randomize(source), "pytorch")
    converted_source = copy.deepcopy(source)
    from team_gm.modules.exceptions import ImplementationType

    for m in converted_source.modules():
        m.foldforge_implementation = (
            ImplementationType.MINIWORLD_ENGINE
            if backend == "miniworld"
            else ImplementationType.PYTORCH
        )
    converted = ReleasedConditionedTransition(converted_source)
    a = torch.randn(
        (2, 24, widths[0]) if broadcast else (24, widths[0]),
        device="cuda",
        dtype=activation_dtype,
    )
    s = torch.randn(24, widths[1], device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected, actual = source(a, s), converted(a, s)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.012)
    assert not any(p.is_meta for p in converted.parameters())
    assert not any(m._forward_pre_hooks for m in converted.modules())


@pytest.mark.parametrize("family", ["af3", "protenix", "opendde"])
@GPU_PARITY
def test_diffusion_composition_against_original(family):
    torch.manual_seed(17)
    path = (
        "nn/diffusion_transformer.py"
        if family == "af3"
        else "model/modules/transformer.py"
    )
    oldmod = original_module(family, path)
    newmod = production_module(family, path)
    name = "DiffusionTransformer" if family == "af3" else "DiffusionTransformerBlock"
    args = (
        {
            "c_act": 128,
            "c_single_cond": 128,
            "c_pair_cond": 16,
            "num_head": 16,
            "num_blocks": 4,
        }
        if family == "af3"
        else {"c_a": 128, "c_s": 128, "c_z": 16, "n_heads": 4}
    )
    if family == "opendde":
        from team_gm.modules.checkpoints.policy import BOUNDED_MEMORY_POLICY

        args["policy"] = BOUNDED_MEMORY_POLICY
    old = randomize(getattr(oldmod, name)(**args))
    new = getattr(newmod, name)(**args)
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch")
    new = configure_model(new, "pytorch")
    new.foldforge_load_report = {}
    install_conditioning(new)
    a = torch.randn(24, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.randn_like(a)
    z = torch.randn(24, 24, 16, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        if family == "af3":
            mask = torch.rand(24, device="cuda") > 0.2
            expected = old(a.clone(), mask, s, z)
            actual = new(a.clone(), mask, s, z)
        else:
            options = (
                {"extra_attn_bias": torch.randn(4, 24, 24, device="cuda")}
                if family == "opendde"
                else {}
            )
            expected = old(a.clone(), s, z, **options)[0]
            actual = new(a.clone(), s, z, **options)[0]
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("family", ["af3", "protenix", "opendde"])
@pytest.mark.parametrize("last", [False, True])
@GPU_PARITY
def test_msa_order_against_original(family, last):
    torch.manual_seed(19)
    path = "nn/pairformer.py" if family == "af3" else "model/modules/pairformer.py"
    oldmod = original_module(family, path)
    newmod = production_module(family, path)
    cls = "EvoformerBlock" if family == "af3" else "MSABlock"
    args = {} if family == "af3" else {"is_last_block": last, "msa_chunk_size": 3}
    if family == "opendde":
        from team_gm.modules.checkpoints.policy import BOUNDED_MEMORY_POLICY

        args["policy"] = BOUNDED_MEMORY_POLICY
    old = randomize(getattr(oldmod, cls)(**args))
    new = getattr(newmod, cls)(**args)
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch")
    new = configure_model(new, "pytorch")
    m = torch.randn(7, 24, 64, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(24, 24, 128, device="cuda", dtype=torch.bfloat16)
    pairmask = (torch.rand(24, 24, device="cuda") > 0.1).float()
    with torch.inference_mode():
        if family == "af3":
            mask = (torch.rand(7, 24, device="cuda") > 0.2).float()
            expected = old(m.clone(), z.clone(), mask, pairmask)
            actual = new(m.clone(), z.clone(), mask, pairmask)
        else:
            expected = old(m.clone(), z.clone(), pairmask, chunk_size=8)
            actual = new(m.clone(), z.clone(), pairmask, chunk_size=8)
    for a, b in zip(actual, expected, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.02)


@pytest.mark.parametrize("precision", [None, torch.float32])
@GPU_PARITY
def test_explicit_projection_matches_previous_hook_policy(precision):
    from team_gm.modules.checkpoints.dense import Linear

    old = Linear(5, 7, precision=precision).to(device="cuda", dtype=torch.bfloat16)
    old.weight.data.normal_()
    new = NativeLinear.from_linear(old)
    x = torch.randn(3, 5, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        torch.testing.assert_close(new(x), old(x.to(old.weight.dtype)), rtol=0, atol=0)
    assert new.weight is old.weight


def test_msa_chunking_preserves_batch_axes_and_gradients():
    x = torch.randn(2, 7, 4, 3, requires_grad=True)
    z = torch.randn(2, 4, 4, 5)
    fn = lambda m, p: 0.2 * m.square() + p.mean()
    transition = lambda m: m + m.sigmoid()
    actual = msa_row_update(
        x, z, attention_delta=fn, transition_residual=transition, chunk_size=3
    )
    expected = transition(x + fn(x, z))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
    )


@pytest.mark.parametrize("family", ["af3", "protenix", "opendde"])
@pytest.mark.parametrize("count", [0, 1, 2])
@GPU_PARITY
def test_template_aggregation_and_masked_pair_stack(family, count):
    from team_gm.modules.checkpoints.pairformer import install_pairformers

    torch.manual_seed(57)
    path = "nn/template.py" if family == "af3" else "model/modules/pairformer.py"
    cls = "TemplateEmbedding" if family == "af3" else "TemplateEmbedder"
    oldmod = original_module(family, path)
    newmod = production_module(family, path)
    options = {}
    if family == "opendde":
        from team_gm.modules.checkpoints.policy import BOUNDED_MEMORY_POLICY

        options["policy"] = BOUNDED_MEMORY_POLICY
    old = randomize(getattr(oldmod, cls)(**options))
    new = getattr(newmod, cls)(**options)
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch")
    new = configure_model(new, "pytorch")
    new.foldforge_load_report = {}
    install_pairformers(new)
    n = 16
    z = torch.randn(n, n, 128, device="cuda", dtype=torch.bfloat16)
    mask = (torch.rand(n, n, device="cuda") > 0.15).float()
    chains = torch.arange(n, device="cuda") // 8
    if family == "af3":
        from foldforge.data.features.dense import Templates

        feature = Templates(
            aatype=torch.randint(0, 20, (count, n), device="cuda"),
            atom_positions=torch.randn(count, n, 24, 3, device="cuda"),
            atom_mask=torch.rand(count, n, 24, device="cuda") > 0.1,
        )
        options = ((chains[:, None] == chains[None, :]).float(),)
    else:
        feature = {
            "asym_id": chains,
            "template_aatype": torch.randint(0, 20, (count, n), device="cuda"),
            "template_distogram": torch.randn(count, n, n, 39, device="cuda"),
            "template_pseudo_beta_mask": (
                torch.rand(count, n, n, device="cuda") > 0.2
            ).float(),
            "template_unit_vector": torch.randn(count, n, n, 3, device="cuda"),
            "template_backbone_frame_mask": (
                torch.rand(count, n, n, device="cuda") > 0.2
            ).float(),
        }
    with torch.inference_mode():
        if family == "af3":
            actual = new(z.clone(), copy.deepcopy(feature), mask, *options)
            expected = old(z.clone(), copy.deepcopy(feature), mask, *options)
        else:
            actual = new(copy.deepcopy(feature), z.clone(), mask, chunk_size=8)
            expected = (
                old(copy.deepcopy(feature), z.clone(), mask, chunk_size=8)
                if count
                else torch.zeros_like(z)
            )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert actual.shape == z.shape


@pytest.mark.parametrize("family", ["protenix", "opendde"])
@GPU_PARITY
def test_atom_diffusion_local_window_matches_original(family):
    torch.manual_seed(53)
    path = "model/modules/transformer.py"
    oldmod = original_module(family, path)
    newmod = importlib.import_module("team_gm.modules.checkpoints.transformer")
    args = {
        "c_a": 128,
        "c_s": 128,
        "c_z": 16,
        "n_heads": 4,
        "cross_attention_mode": True,
    }
    old = randomize(oldmod.DiffusionTransformerBlock(**args))
    new = newmod.DiffusionTransformerBlock(**args)
    new.load_state_dict(old.state_dict(), strict=True)
    old = configure_model(old, "pytorch")
    new = configure_model(new, "pytorch")
    new.foldforge_load_report = {}
    install_conditioning(new)
    a = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    s = torch.randn_like(a)
    z = torch.randn(8, 4, 8, 16, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = old(a.clone(), s, z, n_queries=4, n_keys=8, chunk_size=8)[0]
        actual = new(a.clone(), s, z, n_queries=4, n_keys=8, chunk_size=8)[0]
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
