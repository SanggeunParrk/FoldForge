"""Dense modules reach the engine's ``nn.Module`` ops (hand-CUDA legs) and stay equal.

Each test installs the engine module on a configured BF16 dense module, then checks that
(a) the result is as close to an FP32 truth as the same module's PyTorch reference path
is, up to bf16 rounding, and (b) the engine module is what ran.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses

import pytest
import torch

from foldforge.modules.dense.spec import ALPHAFOLD3
from foldforge.modules.engine.configure import configure_model
from foldforge.modules.engine.modules import (
    _ENGINE_HAS_ATOM_VARIANT,
    _ENGINE_HAS_BIAS_ORIENTATION,
    engine_of,
    install_engine_modules,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BF16_ROUNDING = 3e-3


def rms_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return (
        (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    ).item()


def randomize(module: torch.nn.Module, std: float = 0.08) -> torch.nn.Module:
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim > 1:
                p.normal_(0, std)
    return module


def install(module: torch.nn.Module):
    """FP32 truth, BF16 reference, BF16 module with the engine module installed."""
    truth = copy.deepcopy(module).cuda().float()
    ref = configure_model(copy.deepcopy(module), "pytorch")
    engine = configure_model(module, "miniworld")
    counts = install_engine_modules(
        engine, torch.empty(0, dtype=torch.bfloat16, device="cuda")
    )
    return truth, ref, engine, counts


@contextlib.contextmanager
def engine_calls(module: torch.nn.Module):
    """Count the forward calls of the engine module(s) attached to ``module``."""
    built = engine_of(module)
    modules = built if isinstance(built, list) else [built]
    calls = [0]

    def count(*_: object) -> None:
        calls[0] += 1

    handles = [m.register_forward_hook(count) for m in modules]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def residue_and_pair_mask(length: int, drop: float = 0.15):
    residues = torch.rand(length, device="cuda") > drop
    return residues, (residues[:, None] & residues[None, :])


@pytest.mark.parametrize("length", [128, 384])
@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication(length, outgoing):
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication
    from foldforge.modules.engine.routing import triangle_residual

    torch.manual_seed(1)
    layer = randomize(TriangleMultiplication(128, _outgoing=outgoing))
    truth, ref, engine, counts = install(layer)
    assert counts["engine_trimul"] == 1
    x = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    residues, mask = residue_and_pair_mask(length)
    real = residues[:, None] & residues[None, :]
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = triangle_residual(truth, x.float(), mask)
        reference = triangle_residual(ref, x.clone(), mask)
        actual = triangle_residual(engine, x.clone(), mask)
    assert calls[0] == 1
    assert rms_error(actual[real], expected[real]) <= (
        rms_error(reference[real], expected[real]) + BF16_ROUNDING
    )


@pytest.mark.parametrize("outgoing", [True, False])
def test_template_triangle_multiplication(outgoing):
    """Template stacks: pair 64, hidden 128 (the engine's d_hidden)."""
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication
    from foldforge.modules.engine.routing import triangle_residual

    torch.manual_seed(5)
    layer = randomize(TriangleMultiplication(64, _outgoing=outgoing, hidden_dim=128))
    truth, ref, engine, counts = install(layer)
    assert counts["engine_trimul"] == 1
    x = torch.randn(256, 256, 64, device="cuda", dtype=torch.bfloat16)
    residues, mask = residue_and_pair_mask(256)
    real = residues[:, None] & residues[None, :]
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = triangle_residual(truth, x.float(), mask)
        reference = triangle_residual(ref, x.clone(), mask)
        actual = triangle_residual(engine, x.clone(), mask)
    assert calls[0] == 1
    assert rms_error(actual[real], expected[real]) <= (
        rms_error(reference[real], expected[real]) + BF16_ROUNDING
    )


@pytest.mark.parametrize("ending", [False, True])
def test_template_triangle_attention(ending):
    """Template stacks: pair 64, 4 x 32 heads (the engine's d_hidden)."""
    from foldforge.modules.dense.attention import GridSelfAttention

    torch.manual_seed(6)
    layer = randomize(
        GridSelfAttention(64, 4, transpose=ending, qkv_dim=32), 0.07
    )
    truth, ref, engine, counts = install(layer)
    assert counts["engine_triangle_attention"] == 1
    x = torch.randn(256, 256, 64, device="cuda", dtype=torch.bfloat16)
    residues, mask = residue_and_pair_mask(256)
    real = residues[:, None] & residues[None, :]
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth.residual(x.float(), mask)
        reference = ref.residual(x.clone(), mask)
        actual = engine.residual(x.clone(), mask)
    assert calls[0] == 1
    assert rms_error(actual[real], expected[real]) <= (
        rms_error(reference[real], expected[real]) + BF16_ROUNDING
    )


@pytest.mark.parametrize("ending", [False, True])
@pytest.mark.parametrize("transposed_column_pair_bias", [True, False])
def test_triangle_attention(ending, transposed_column_pair_bias):
    from foldforge.modules.dense.attention import GridSelfAttention

    if ending and not transposed_column_pair_bias and not _ENGINE_HAS_BIAS_ORIENTATION:
        pytest.skip("this engine has no ending-node bias orientation option")
    torch.manual_seed(2)
    # A large bias projection makes a wrong orientation visible.
    spec = dataclasses.replace(
        ALPHAFOLD3, transposed_column_pair_bias=transposed_column_pair_bias
    )
    layer = randomize(GridSelfAttention(128, 4, transpose=ending, spec=spec), 0.07)
    with torch.no_grad():
        layer.pair_bias_projection.weight.normal_(0, 1.5)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_triangle_attention"] == 1
    x = torch.randn(256, 256, 128, device="cuda", dtype=torch.bfloat16)
    residues, mask = residue_and_pair_mask(256)
    real = residues[:, None] & residues[None, :]
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth.residual(x.float(), mask)
        reference = ref.residual(x.clone(), mask)
        actual = engine.residual(x.clone(), mask)
    assert calls[0] == 1
    assert rms_error(actual[real], expected[real]) <= (
        rms_error(reference[real], expected[real]) + BF16_ROUNDING
    )


def test_triangle_attention_af3_ending_node_bias():
    """AF3 ending-node bias needs the engine orientation option; else no module."""
    from foldforge.modules.dense.attention import GridSelfAttention

    _, _, engine, counts = install(GridSelfAttention(128, 4, transpose=True))
    assert not engine.transposed_bias
    expected = 1 if _ENGINE_HAS_BIAS_ORIENTATION else 0
    assert counts["engine_triangle_attention"] == expected
    assert (engine_of(engine) is not None) == bool(expected)


@pytest.mark.parametrize(("rows", "length"), [(16, 256), (1024, 384)])
def test_msa_pair_weighted_averaging(rows, length):
    from foldforge.modules.dense.attention import MSAAttention

    torch.manual_seed(3)
    truth, ref, engine, counts = install(randomize(MSAAttention(64, 128, 8), 0.1))
    assert counts["engine_msa_pwa"] == 1
    msa = torch.randn(rows, length, 64, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    residues, _ = residue_and_pair_mask(length)
    keep = (torch.rand(rows, length, device="cuda") > 0.1) & residues[None]
    msa_mask = keep.float()
    msa_mask[0] = residues.float()
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth.residual(msa.float(), msa_mask, pair.float())
        reference = ref.residual(msa.clone(), msa_mask, pair)
        actual = engine.residual(msa.clone(), msa_mask, pair)
    assert calls[0] == 1
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + BF16_ROUNDING
    )


@pytest.mark.parametrize(("rows", "length"), [(1, 128), (512, 256)])
def test_outer_product_mean(rows, length):
    from foldforge.modules.dense.primitives import OuterProductMean

    torch.manual_seed(4)
    truth, ref, engine, counts = install(randomize(OuterProductMean(64, 128, 32), 0.1))
    assert counts["engine_opm"] == 1
    msa = torch.randn(rows, length, 64, device="cuda", dtype=torch.bfloat16)
    residues, _ = residue_and_pair_mask(length)
    keep = (torch.rand(rows, length, device="cuda") > 0.1) & residues[None]
    mask = keep.float()
    mask[0] = residues.float()
    real = residues[:, None] & residues[None, :]
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth(msa.float(), mask)
        reference = ref(msa.clone(), mask)
        actual = engine(msa.clone(), mask)
    assert calls[0] == 1
    # The engine divides by clamp(count, 1), the release by count + 1e-3: as close to
    # the FP32 truth as the BF16 reference is, up to the same rounding.
    assert rms_error(actual[real], expected[real]) <= (
        rms_error(reference[real], expected[real]) + BF16_ROUNDING
    )


@pytest.mark.parametrize("per_block_pair", [False, True])
def test_diffusion_transformer(per_block_pair):
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer

    torch.manual_seed(5)
    spec = dataclasses.replace(ALPHAFOLD3, per_block_pair_layer_norm=per_block_pair)
    layer = DiffusionTransformer(768, 384, 128, 16, 4, spec=spec)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
    truth, ref, engine, counts = install(layer)
    assert counts["engine_dit_blocks"] == 4
    samples, length = 5, 384
    act = torch.randn(samples, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(samples, length, 384, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.15
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth(act.float(), residues, single.float(), pair.float())
        reference = ref(act.clone(), residues, single, pair)
        actual = engine(act.clone(), residues, single, pair)
    assert calls[0] == 4
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + BF16_ROUNDING
    )
    # The engine caches its pair bias by tensor version, which an inference tensor lacks
    # have: under inference_mode the dense reference path runs instead of failing.
    with torch.inference_mode(), engine_calls(engine) as calls:
        fallback = engine(act.clone(), residues, single, pair)
    assert calls[0] == 0
    assert rms_error(fallback[:, residues], expected[:, residues]) < 0.02


def test_installer_covers_the_released_graph():
    """Every module of the AF3 graph that has an exact engine module gets one."""
    from foldforge.models.architectures.af3 import AlphaFold3

    model = AlphaFold3(
        num_recycles=1, num_samples=1, diffusion_steps=2, spec=ALPHAFOLD3
    )
    model = configure_model(model, "miniworld", torch.bfloat16, "cuda")
    counts = install_engine_modules(
        model, torch.empty(0, dtype=torch.bfloat16, device="cuda")
    )
    names = {type(m).__name__ for m in model.modules()}
    assert {"GridSelfAttention", "MSAAttention", "OuterProductMean"} <= names
    # 48 trunk + 4 confidence + 2 template pairformer blocks and 4 MSA blocks. AF3's
    # bias orientation (released code) converts only on an engine that has the option.
    blocks = 48 + 4 + 2 + 4
    assert counts["engine_trimul"] == 2 * blocks
    assert counts["engine_triangle_attention"] == (
        2 * blocks if _ENGINE_HAS_BIAS_ORIENTATION else blocks
    )
    assert counts["engine_msa_pwa"] == 4
    assert counts["engine_opm"] == 4
    assert counts["engine_dit_blocks"] == 24
    assert counts["engine_atom_blocks"] == (
        9 if _ENGINE_HAS_ATOM_VARIANT else 0
    )  # 3 atom transformers of 3 blocks


def test_pairformer_single_attention():
    from foldforge.modules.dense.pairformer import PairformerBlock
    from foldforge.modules.engine.pairformer import install_pairformers

    torch.manual_seed(6)
    block = randomize(PairformerBlock(), 0.04)
    container = torch.nn.Sequential(block)
    truth = copy.deepcopy(container).cuda().float()
    ref = configure_model(copy.deepcopy(container), "pytorch")
    ref.foldforge_load_report = {}
    install_pairformers(ref)
    engine = configure_model(container, "miniworld")
    engine.foldforge_load_report = {}
    install_pairformers(engine)
    counts = install_engine_modules(
        engine, torch.empty(0, dtype=torch.bfloat16, device="cuda")
    )
    assert counts["engine_single_attention"] == 1
    wrapper = next(
        m for m in engine.modules() if type(m).__name__ == "_SingleAttention"
    )
    length = 128
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(length, 384, device="cuda", dtype=torch.bfloat16)
    residues, pair_mask = residue_and_pair_mask(length)
    with torch.no_grad(), engine_calls(wrapper) as calls:
        expected = truth[0](
            pair.float(), pair_mask.float(), single.float(), residues.float()
        )
        reference = ref[0](pair.clone(), pair_mask, single.clone(), residues.float())
        actual = engine[0](pair.clone(), pair_mask, single.clone(), residues.float())
    assert calls[0] == 1
    assert rms_error(actual[1][residues], expected[1][residues]) <= (
        rms_error(reference[1][residues], expected[1][residues]) + BF16_ROUNDING
    )


def af3_gather(real: int, padded: int):
    """AF3's ``AtomCrossAtt.compute_features`` key windows as a query->key gather."""
    from foldforge.utils import atom_layout

    subsets = padded // 32
    centers = torch.arange(16, padded, 32, device="cuda")
    index = centers[:, None] + torch.arange(-64, 64, device="cuda")[None]
    first = index[:, :1]
    index = torch.where(first < 0, index - first, index)
    last = index[:, -1:]
    index = torch.where(last > real - 1, index - (last - (real - 1)), index)
    index = index.clamp(0, padded - 1)
    return atom_layout.GatherInfo(
        gather_idxs=index, gather_mask=index < real, input_shape=(subsets, 32)
    )


@pytest.mark.parametrize("family", ["alphafold3", "protenix1"])
@pytest.mark.parametrize("real", [512, 301, 140])
@pytest.mark.parametrize("samples", [None, 3])
def test_atom_transformer(family, real, samples):
    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )
    from foldforge.modules.dense.spec import SPECS

    if not _ENGINE_HAS_ATOM_VARIANT:
        pytest.skip("this engine has no AF3 atom attention variant")
    torch.manual_seed(7)
    layer = DiffusionCrossAttTransformer(spec=SPECS[family])
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.08)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.normal_(0, 0.05)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_atom_blocks"] == 3
    padded = 512
    subsets = padded // 32
    shape = (samples,) if samples else ()
    act = torch.randn(*shape, subsets, 32, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(subsets, 32, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(subsets, 32, 128, 16, device="cuda", dtype=torch.bfloat16)
    atoms = torch.arange(padded, device="cuda").view(subsets, 32) < real
    gather = af3_gather(real, padded)
    from foldforge.utils import atom_layout

    keys_mask = atom_layout.convert(gather, atoms, layout_axes=(-2, -1))

    def run(module: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
        keys_cond = atom_layout.convert(gather, cond.to(dtype), layout_axes=(-3, -2))
        return module(
            queries_act=act.to(dtype).clone(),
            queries_mask=atoms,
            queries_to_keys=gather,
            keys_mask=keys_mask,
            queries_single_cond=cond.to(dtype),
            keys_single_cond=keys_cond,
            pair_cond=pair.to(dtype),
        )

    with torch.no_grad(), engine_calls(engine) as calls:
        expected = run(truth, torch.float32)
        reference = run(ref, torch.bfloat16)
        actual = run(engine, torch.bfloat16)
    assert calls[0] == 3
    assert rms_error(actual[..., atoms, :], expected[..., atoms, :]) <= (
        rms_error(reference[..., atoms, :], expected[..., atoms, :]) + BF16_ROUNDING
    )


@pytest.mark.parametrize(
    "family", ["chai1", "esmfold2", "esmfold2-fast", "rosettafold3", "boltz2"]
)
def test_installer_leaves_other_families_alone(family):
    """A family whose blocks differ from AF3's keeps its reference modules."""
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.modules.dense.spec import SPECS

    model = AlphaFold3(
        num_recycles=1, num_samples=1, diffusion_steps=2, spec=SPECS[family]
    )
    model = configure_model(model, "miniworld", torch.bfloat16, "cuda")
    counts = install_engine_modules(
        model, torch.empty(0, dtype=torch.bfloat16, device="cuda")
    )
    assert counts["engine_atom_blocks"] == 0


def loaded_model(family: str = "alphafold3", *, diffusion_master: bool = False):
    """A model after the loader's installs (``diffusion_master``: FP32 diffusion)."""
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.modules.dense.spec import SPECS
    from foldforge.modules.engine.install import install_engine

    model = AlphaFold3(
        num_recycles=1, num_samples=1, diffusion_steps=2, spec=SPECS[family]
    )
    model.foldforge_load_report = {}
    model = configure_model(model, "miniworld", torch.bfloat16, "cuda")
    if diffusion_master:
        for name, p in model.named_parameters():
            if name.startswith("diffusion_head") and p.ndim > 1:
                p.data = p.data.float()
    return install_engine(model, torch.bfloat16, "cuda", modules=True)


def test_the_loader_order_keeps_the_diffusion_blocks_on_the_engine():
    """The conditioning pass renames the released transitions; read them before."""
    model = loaded_model()
    report = model.foldforge_load_report
    assert report["engine_dit_blocks"] == 24
    assert report["engine_atom_blocks"] == (9 if _ENGINE_HAS_ATOM_VARIANT else 0)


def test_fp32_master_weights_stay_fp32_in_the_diffusion_blocks():
    model = loaded_model(diffusion_master=True)
    blocks = next(
        engine_of(m)
        for m in model.modules()
        if type(m).__name__ == "DiffusionTransformer" and engine_of(m)
    )
    assert blocks[0].attention.to_query.weight.dtype == torch.float32
    trunk = next(
        engine_of(m)
        for m in model.modules()
        if type(m).__name__ == "TriangleMultiplication" and engine_of(m)
    )
    assert trunk.to_out.weight.dtype == torch.bfloat16


def test_engine_initialiser_is_skipped_only_while_installing():
    """The engine's SciPy initialiser is off while installing, and back after."""
    from miniworld_engine.modules import primitives

    from foldforge.modules.engine.install import skip_engine_init

    original = primitives.InitType.apply
    with skip_engine_init():
        assert primitives.InitType.apply is not original
    assert primitives.InitType.apply is original
    message = "boom"
    with pytest.raises(RuntimeError, match=message), skip_engine_init():
        raise RuntimeError(message)
    assert primitives.InitType.apply is original
