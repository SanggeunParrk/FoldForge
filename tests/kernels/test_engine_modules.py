"""Dense modules reach the engine's ``nn.Module`` ops (hand-CUDA legs) and stay equal.

Each test installs the engine module on a configured BF16 dense module, then checks that
(a) the result is as close to an FP32 truth as the same module's PyTorch reference path
is, up to bf16 rounding, and (b) the engine module is what ran.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import warnings

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
def force_triton(on: bool = True):
    """The engine's A/B switch (``settings.force_triton_kernels``): on an H100 it sends the portable Triton paths what the CUDA
    paths would take, so that they can be tested and timed there. A test and bench device only; FoldForge has no such key."""
    from miniworld_engine import settings

    previous = settings.current().force_triton_kernels
    settings.configure(force_triton_kernels=on)
    try:
        yield
    finally:
        settings.configure(force_triton_kernels=previous)


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


def test_engine_skip_leaves_the_module_dense():
    """``engine.skip`` of the config: the named engine_* key counts its module but attaches nothing."""
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication

    torch.manual_seed(1)
    layer = randomize(TriangleMultiplication(128, _outgoing=True))
    engine = configure_model(copy.deepcopy(layer), "miniworld")
    like = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    skipped = install_engine_modules(engine, like, skip=("engine_trimul",))
    assert skipped["engine_trimul"] == 0
    assert engine_of(engine) is None
    served = install_engine_modules(engine, like)
    assert served["engine_trimul"] == 1
    assert engine_of(engine) is not None


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
    layer = randomize(GridSelfAttention(64, 4, transpose=ending, qkv_dim=32), 0.07)
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


@pytest.mark.parametrize("outgoing", [True, False])
def test_length_divided_triangle_multiplication(outgoing):
    """RoseTTAFold3: the contraction divided by the real length, on the engine with W_left / L.

    Two calls at different real lengths: the scale is the call's, not the first call's.
    """
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication
    from foldforge.modules.engine.routing import triangle_residual

    torch.manual_seed(7)
    layer = randomize(
        TriangleMultiplication(128, _outgoing=outgoing, divide_by_length=True)
    )
    truth, ref, engine, counts = install(layer)
    assert counts["engine_trimul"] == 1
    x = torch.randn(256, 256, 128, device="cuda", dtype=torch.bfloat16)
    for drop in (0.1, 0.5):
        residues, mask = residue_and_pair_mask(256, drop)
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
@pytest.mark.parametrize("variant", ["rosettafold3", "chai1", "chai1_dual"])
def test_released_triangle_attention_variants(ending, variant):
    """RoseTTAFold3's gate / out biases, Chai-1's untransposed ending node and its confidence
    head's second out-projection: each on the engine module."""
    from foldforge.modules.dense.attention import GridSelfAttention
    from foldforge.modules.dense.spec import SPECS

    torch.manual_seed(8)
    family, dual = variant.removesuffix("_dual"), variant.endswith("_dual")
    layer = GridSelfAttention(
        128, 4, transpose=ending, spec=SPECS[family], dual_output=dual
    )
    randomize(layer, 0.07)
    with torch.no_grad():
        layer.pair_bias_projection.weight.normal_(0, 1.5)
        for linear in (
            layer.gating_query,
            layer.output_projection,
            getattr(layer, "output_projection_transposed", None),
        ):
            if linear is not None and linear.bias is not None:
                linear.bias.normal_(0, 0.3)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_triangle_attention"] == 1
    # The trunk's pair scale: a residual output there hides an update error under the pair
    # itself, so the updates are compared (an out bias added after the bf16 residual sum was
    # a second rounding at this scale, 1.4x the reference's error).
    x = (20 * torch.randn(256, 256, 128, device="cuda")).to(torch.bfloat16)
    residues, mask = residue_and_pair_mask(256)
    real = residues[:, None] & residues[None, :]
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth.residual(x.float(), mask).double() - x.double()
        reference = ref.residual(x.clone(), mask).double() - x.double()
        actual = engine.residual(x.clone(), mask).double() - x.double()
    assert calls[0] == 1
    assert rms_error(actual[real], expected[real]) <= 1.05 * rms_error(
        reference[real], expected[real]
    )


@pytest.mark.parametrize(
    ("projection_bias", "groups", "channels"), [(True, 1, 32), (False, 8, 8)]
)
@pytest.mark.parametrize(("rows", "length"), [(1, 128), (512, 256)])
def test_released_outer_product_mean_variants(
    projection_bias, groups, channels, rows, length
):
    """RoseTTAFold3's projection biases and Chai-1's eight LayerNorm'd groups on the engine."""
    from foldforge.modules.dense.primitives import OuterProductMean

    torch.manual_seed(9)
    layer = randomize(
        OuterProductMean(
            64, 128, channels, projection_bias=projection_bias, groups=groups
        ),
        0.1,
    )
    with torch.no_grad():
        if projection_bias:
            layer.left_projection.bias.normal_(0, 0.3)
            layer.right_projection.bias.normal_(0, 0.3)
        if groups > 1:
            layer.product_norm.weight.normal_(1, 0.1)
            layer.product_norm.bias.normal_(0, 0.1)
    truth, ref, engine, counts = install(layer)
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


def test_diffusion_transformer_extra_pair_bias():
    """OpenDDE's structural-token bias ([L, L], every head, every block) is folded into the engine's hoisted bias."""
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer

    torch.manual_seed(7)
    spec = dataclasses.replace(ALPHAFOLD3, per_block_pair_layer_norm=True)
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
    extra = 2 * torch.randn(length, length, device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.15
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth(act.float(), residues, single.float(), pair.float(), extra.float())
        reference = ref(act.clone(), residues, single, pair, extra)
        actual = engine(act.clone(), residues, single, pair, extra)
        without = engine(act.clone(), residues, single, pair)
    assert calls[0] == 8  # both calls reached the engine blocks, the biased one included
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + BF16_ROUNDING
    )
    # the bias is not a no-op: dropping it moves the result far outside bf16 rounding
    assert rms_error(without[:, residues], expected[:, residues]) > 10 * BF16_ROUNDING


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


def af3_gather(real: int, padded: int, keys: int = 128):
    """AF3's ``AtomCrossAtt.compute_features`` key windows as a query->key gather."""
    from foldforge.utils import atom_layout

    subsets = padded // 32
    centers = torch.arange(16, padded, 32, device="cuda")
    index = centers[:, None] + torch.arange(-keys // 2, keys // 2, device="cuda")[None]
    first = index[:, :1]
    index = torch.where(first < 0, index - first, index)
    last = index[:, -1:]
    index = torch.where(last > real - 1, index - (last - (real - 1)), index)
    index = index.clamp(0, padded - 1)
    return atom_layout.GatherInfo(
        gather_idxs=index, gather_mask=index < real, input_shape=(subsets, 32)
    )


@pytest.mark.parametrize("portable", [False, True], ids=["default", "portable"])
@pytest.mark.parametrize("family", ["alphafold3", "protenix1"])
@pytest.mark.parametrize("real", [512, 301, 140])
@pytest.mark.parametrize("samples", [None, 3])
def test_atom_transformer(family, real, samples, portable):
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

    with torch.no_grad(), engine_calls(engine) as calls, force_triton(portable):
        expected = run(truth, torch.float32)
        reference = run(ref, torch.bfloat16)
        if portable:  # the block's attention core is the engine's Triton window kernel
            assert engine_of(engine)[0].serves_portable(act)
        actual = run(engine, torch.bfloat16)
    assert calls[0] == 3
    assert rms_error(actual[..., atoms, :], expected[..., atoms, :]) <= (
        rms_error(reference[..., atoms, :], expected[..., atoms, :]) + BF16_ROUNDING
    )


@pytest.mark.parametrize("portable", [False, True], ids=["default", "portable"])
@pytest.mark.parametrize("real", [512, 301, 200])
@pytest.mark.parametrize("samples", [None, 3])
def test_esmfold2_atom_transformer(real, samples, portable):
    """ESMFold2's atom stack on the engine: RMS / SiLU AdaLN, raw gates, rotary q / k,
    32 queries x 192 keys under the rank-based +/- 64 window."""
    from foldforge.modules.dense.atom_cross_attention import (
        build_atom_rope,
        sliding_window_mask,
    )
    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )
    from foldforge.modules.dense.spec import SPECS
    from foldforge.utils import atom_layout

    spec = SPECS["esmfold2"]
    torch.manual_seed(8)
    layer = DiffusionCrossAttTransformer(spec=spec)
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
    pair = torch.randn(subsets, 32, 192, 16, device="cuda", dtype=torch.bfloat16)
    atoms = torch.arange(padded, device="cuda").view(subsets, 32) < real
    gather = af3_gather(real, padded, keys=192)

    def keys(t: torch.Tensor, axes: tuple[int, int] = (-3, -2)) -> torch.Tensor:
        return atom_layout.convert(gather, t, layout_axes=axes)

    keys_mask = keys(atoms, (-2, -1))
    pos = 5 * torch.randn(subsets, 32, 3, device="cuda")
    uid = torch.arange(padded, device="cuda").view(subsets, 32) // 9
    rope_q = build_atom_rope(pos, uid, 32, spec.atom_rope)
    rope_k = build_atom_rope(keys(pos), keys(uid, (-2, -1)), 32, spec.atom_rope)
    pair_mask = sliding_window_mask(atoms, keys_mask, gather, spec.atom_window_half)

    def run(module: torch.nn.Module, dtype: torch.dtype) -> torch.Tensor:
        return module(
            queries_act=act.to(dtype).clone(),
            queries_mask=atoms,
            queries_to_keys=gather,
            keys_mask=keys_mask,
            queries_single_cond=cond.to(dtype),
            keys_single_cond=keys(cond.to(dtype)),
            pair_cond=pair.to(dtype),
            pair_mask=pair_mask,
            rope_q=rope_q,
            rope_k=rope_k,
        )

    with torch.no_grad(), engine_calls(engine) as calls, force_triton(portable):
        expected = run(truth, torch.float32)
        reference = run(ref, torch.bfloat16)
        if portable:
            assert engine_of(engine)[0].serves_portable(act)
        actual = run(engine, torch.bfloat16)
    assert calls[0] == 3
    assert rms_error(actual[..., atoms, :], expected[..., atoms, :]) <= (
        rms_error(reference[..., atoms, :], expected[..., atoms, :]) + BF16_ROUNDING
    )


@pytest.mark.parametrize("family", ["chai1", "rosettafold3", "boltz2"])
def test_installer_leaves_other_families_alone(family):
    """A family whose blocks differ from AF3's gets no AF3 block (own stack, below)."""
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


# --- Boltz-2 / RoseTTAFold3 / Chai-1: the engine's whole-stack modules (inference) ---

VARIANT_FAMILIES = ["boltz2", "rosettafold3", "chai1"]


def variant_options(family: str, *, atom: bool = False) -> dict:
    from miniworld_engine.modules import dit_variant

    return getattr(dit_variant, family.upper() + ("_ATOM" if atom else ""))


def randomize_distinct(module: torch.nn.Module, std: float) -> None:
    """Every parameter distinct: a swapped gamma or bias changes the output."""
    with torch.no_grad():
        for _, p in module.named_parameters():
            if p.ndim > 1:
                p.normal_(0, std)
            else:
                p.normal_(0, 0.05)
        for name, p in module.named_parameters():
            if (
                p.ndim == 1
                and ("norm" in name or "layer_norm" in name)
                and name.endswith("weight")
            ):
                p.add_(1.0)


@pytest.mark.parametrize("mode", ["fast", "exact"])
@pytest.mark.parametrize("family", VARIANT_FAMILIES)
def test_variant_diffusion_transformer(family, mode, monkeypatch):
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from miniworld_engine.kernels.dit_variants import interface

    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer
    from foldforge.modules.dense.spec import spec_for

    options = variant_options(family)
    torch.manual_seed(11)
    # "exact" keeps the release's conventions (Chai-1: AdaLN epsilon 0.1).
    layer = DiffusionTransformer(
        768, options["d_cond"], options["d_pair"], 16, 4, spec=spec_for(family, mode)
    )
    randomize_distinct(layer, 0.04)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_dit_variant_blocks"] == 4
    assert counts["engine_dit_blocks"] == 0
    samples, length = 5, 384
    act = torch.randn(samples, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(length, options["d_cond"], device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(
        length, length, options["d_pair"], device="cuda", dtype=torch.bfloat16
    )
    residues = torch.rand(length, device="cuda") > 0.15
    key_mask = (residues > 0)[None]
    builds = [0]
    build = interface._bias_all  # noqa: SLF001

    def counted(*args: object) -> object:
        builds[0] += 1
        return build(*args)

    monkeypatch.setattr(interface, "_bias_all", counted)
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth(act.float(), residues, single.float(), pair.float())
        reference = ref(act.clone(), residues, single, pair)
        actual = engine(act.clone(), residues, single, pair, key_mask=key_mask)
        # A second step on the same fold: the pair bias is not made again.
        engine(act.clone(), residues, single, pair, key_mask=key_mask)
    assert calls[0] == 2
    assert builds[0] == 1
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + BF16_ROUNDING
    )
    assert rms_error(actual[:, residues], reference[:, residues]) < 0.02
    # Not served (a gradient, inference mode): the dense blocks run, not the stack.
    with engine_calls(engine) as calls:
        graded = engine(act.clone(), residues, single, pair)
    with torch.inference_mode(), engine_calls(engine) as calls_inference:
        fallback = engine(act.clone(), residues, single, pair)
    assert calls[0] == calls_inference[0] == 0
    assert rms_error(graded[:, residues], expected[:, residues]) < 0.02
    assert rms_error(fallback[:, residues], expected[:, residues]) < 0.02


def test_esmfold2_wide_diffusion_transformer(monkeypatch):
    """ESMFold2's token DiT is AF3's at a 768-wide conditioning and a 256-wide pair: the engine's variant stack serves it."""
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from miniworld_engine.kernels.dit_variants import interface

    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer
    from foldforge.modules.dense.spec import SPECS

    torch.manual_seed(13)
    layer = DiffusionTransformer(768, 768, 256, 16, 4, spec=SPECS["esmfold2"])
    randomize_distinct(layer, 0.04)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_dit_variant_blocks"] == 4
    assert counts["engine_dit_blocks"] == 0
    samples, length = 5, 384
    act = torch.randn(samples, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(length, 768, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, 256, device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.15
    key_mask = (residues > 0)[None]
    builds = [0]
    build = interface._bias_all  # noqa: SLF001

    def counted(*args: object) -> object:
        builds[0] += 1
        return build(*args)

    monkeypatch.setattr(interface, "_bias_all", counted)
    with torch.no_grad(), engine_calls(engine) as calls:
        expected = truth(act.float(), residues, single.float(), pair.float())
        reference = ref(act.clone(), residues, single, pair)
        actual = engine(act.clone(), residues, single, pair, key_mask=key_mask)
        engine(act.clone(), residues, single, pair, key_mask=key_mask)
    assert calls[0] == 2
    assert builds[0] == 1
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + BF16_ROUNDING
    )


def pad_gather(real: int, padded: int, keys: int = 128):
    """The OpenFold lineage's clipped key windows, the ones off the atoms masked."""
    from foldforge.utils import atom_layout

    subsets = padded // 32
    centers = torch.arange(16, padded, 32, device="cuda")
    index = centers[:, None] + torch.arange(-keys // 2, keys // 2, device="cuda")[None]
    return atom_layout.GatherInfo(
        gather_idxs=index.clamp(0, padded - 1),
        gather_mask=(index >= 0) & (index < real),
        input_shape=(subsets, 32),
    )


@pytest.mark.parametrize("window", ["slide", "pad"])
@pytest.mark.parametrize(("padded", "real"), [(512, 512), (512, 301), (480, 450)])
@pytest.mark.parametrize("samples", [None, 3])
@pytest.mark.parametrize("family", VARIANT_FAMILIES)
def test_variant_atom_transformer(family, samples, padded, real, window, monkeypatch):
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from miniworld_engine.kernels.dit_variants import interface

    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )
    from foldforge.modules.dense.spec import SPECS
    from foldforge.utils import atom_layout

    spec = SPECS[family]
    if window == "pad":
        spec = dataclasses.replace(
            spec, atom_key_window="pad", key_masked_atom_attention=True
        )
    torch.manual_seed(13)
    layer = DiffusionCrossAttTransformer(spec=spec)
    randomize_distinct(layer, 0.08)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_atom_variant_blocks"] == 3
    assert counts["engine_atom_blocks"] == 0
    subsets = padded // 32
    shape = (samples,) if samples else ()
    act = torch.randn(*shape, subsets, 32, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(subsets, 32, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(subsets, 32, 128, 16, device="cuda", dtype=torch.bfloat16)
    atoms = torch.arange(padded, device="cuda").view(subsets, 32) < real
    gather = (af3_gather if window == "slide" else pad_gather)(real, padded)
    keys_mask = atom_layout.convert(gather, atoms, layout_axes=(-2, -1))
    # Chai-1: a query attends only the keys of its own reference conformer.
    uid = torch.arange(padded, device="cuda").view(subsets, 32) // 11
    keys_uid = atom_layout.convert(gather, uid, layout_axes=(-2, -1))
    same_conformer = family == "chai1"
    pair_mask = (
        (uid[:, :, None] == keys_uid[:, None, :])
        & atoms[:, :, None]
        & keys_mask[:, None, :]
        if same_conformer
        else None
    )
    flat_uid = uid.reshape(padded) if same_conformer else None
    engine_inputs = (pair[None], (atoms > 0).reshape(1, padded), flat_uid, None)
    if subsets % 2:  # the fold state's copy with one empty window appended
        engine_inputs = (
            *engine_inputs[:3],
            (
                torch.nn.functional.pad(pair[None], (0, 0, 0, 0, 0, 0, 0, 1)),
                torch.nn.functional.pad(engine_inputs[1], (0, 32)),
                None
                if flat_uid is None
                else torch.nn.functional.pad(flat_uid, (0, 32), value=-1),
            ),
        )
    builds = [0]
    build = interface._atom_biases  # noqa: SLF001

    def counted(*args: object) -> object:
        builds[0] += 1
        return build(*args)

    monkeypatch.setattr(interface, "_atom_biases", counted)

    def run(
        module: torch.nn.Module, dtype: torch.dtype, **extra: object
    ) -> torch.Tensor:
        keys_cond = atom_layout.convert(gather, cond.to(dtype), layout_axes=(-3, -2))
        return module(
            queries_act=act.to(dtype).clone(),
            queries_mask=atoms,
            queries_to_keys=gather,
            keys_mask=keys_mask,
            queries_single_cond=cond.to(dtype),
            keys_single_cond=keys_cond,
            pair_cond=pair.to(dtype),
            pair_mask=pair_mask,
            **extra,
        )

    with torch.no_grad(), engine_calls(engine) as calls:
        expected = run(truth, torch.float32)
        reference = run(ref, torch.bfloat16)
        actual = run(engine, torch.bfloat16, engine_inputs=engine_inputs)
        run(engine, torch.bfloat16, engine_inputs=engine_inputs)  # the next step
    assert calls[0] == 2
    assert builds[0] == 1
    assert rms_error(actual[..., atoms, :], expected[..., atoms, :]) <= (
        rms_error(reference[..., atoms, :], expected[..., atoms, :]) + BF16_ROUNDING
    )
    assert rms_error(actual[..., atoms, :], reference[..., atoms, :]) < 0.02
    if same_conformer:
        # The mask without its conformer ids is not the engine's to serve.
        with torch.no_grad(), engine_calls(engine) as calls:
            run(engine, torch.bfloat16)
        assert calls[0] == 0


def test_variant_atom_circular_window_stays_dense():
    """Chai-1's exact mode wraps its key window; the engine knows slide and pad only."""
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )
    from foldforge.modules.dense.spec import spec_for

    layer = DiffusionCrossAttTransformer(spec=spec_for("chai1", "exact"))
    _, _, engine, counts = install(layer)
    assert counts["engine_atom_variant_blocks"] == counts["engine_atom_blocks"] == 0
    assert engine_of(engine) is None


# --- the portable path: what `miniworld` runs on a card that is not H100 / B200 ------------------------------
#
# The engine routes a variant stack itself (H100 CUDA, else its portable Triton path, else the reference);
# `variant_serves` only asks. On the H100 here the portable path is reached through the engine's A/B switch
# (`force_triton`) or through a faked card (`fake_card`), which is what the engine's own routing tests do.


@contextlib.contextmanager
def fake_card(monkeypatch, tmp_path, name: str, capability: tuple[int, int], *, recorded: bool):
    """`torch.cuda` reports another card; ``recorded`` gives it a manifest with every Triton kernel of the DiT ops `ok`."""
    import csv

    from miniworld_engine.autotune import records
    from miniworld_engine.modules import dispatch

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):  # noqa: SLF001
        cache.clear()
    if recorded:
        rows = [["kernel", "backend", "family", "file", "status", "detail", "dtype"], ["#provenance", "0", "0", "clean", "t", "t", ""]]
        seen = set()
        for op in ("local_dit", "dit_variant", "dit_variant_atom"):
            for kernel in records.triton_kernels(op, "bf16", "training"):
                if kernel not in seen:
                    seen.add(kernel)
                    rows.append([kernel, "triton", "x", "x.py", "ok", "", "bf16"])
        with (tmp_path / f"{records.card_key(name, capability)}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    with records.use_manifest_dir(tmp_path):
        try:
            yield
        finally:
            for cache in (dispatch._card_cache, dispatch._unrecorded_warned, dispatch._refused_warned):  # noqa: SLF001
                cache.clear()


def count_portable(monkeypatch, name: str) -> list[int]:
    """Count the calls of the engine's portable stack forward (``stack_forward`` / ``atom_stack_forward``)."""
    from miniworld_engine.kernels.dit_variants.triton import portable

    calls = [0]
    original = getattr(portable, name)

    def counted(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(portable, name, counted)
    return calls


def variant_token_case(family: str):
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer
    from foldforge.modules.dense.spec import spec_for

    options = variant_options(family)
    torch.manual_seed(11)
    layer = DiffusionTransformer(768, options["d_cond"], options["d_pair"], 16, 4, spec=spec_for(family, "fast"))
    randomize_distinct(layer, 0.04)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_dit_variant_blocks"] == 4
    length = 384
    act = torch.randn(5, length, 768, device="cuda", dtype=torch.bfloat16)
    single = torch.randn(length, options["d_cond"], device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(length, length, options["d_pair"], device="cuda", dtype=torch.bfloat16)
    residues = torch.rand(length, device="cuda") > 0.15
    return truth, ref, engine, act, single, pair, residues


@pytest.mark.parametrize("family", VARIANT_FAMILIES)
def test_variant_diffusion_transformer_portable(family, monkeypatch):
    """The A/B switch sends the H100 to the portable stack; its output is the dense blocks' and the reference's up to bf16 rounding."""
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    truth, ref, engine, act, single, pair, residues = variant_token_case(family)
    key_mask = residues[None]
    calls = count_portable(monkeypatch, "stack_forward")
    with torch.no_grad(), force_triton(), engine_calls(engine) as stack_calls:
        expected = truth(act.float(), residues, single.float(), pair.float())
        reference = ref(act.clone(), residues, single, pair)
        actual = engine(act.clone(), residues, single, pair, key_mask=key_mask)
        engine(act.clone(), residues, single, pair, key_mask=key_mask)
    assert calls[0] == stack_calls[0] == 2
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + BF16_ROUNDING
    )
    assert rms_error(actual[:, residues], reference[:, residues]) < 0.02
    # A call that needs a gradient is the portable path's too (the H100 kernels are inference-only).
    with force_triton(), engine_calls(engine) as graded_calls:
        graded = engine(act.clone(), residues, single, pair, key_mask=key_mask)
    assert graded_calls[0] == 1
    assert rms_error(graded[:, residues], expected[:, residues]) < 0.02
    # Inference mode stays the dense blocks' (the engine caches by tensor version).
    with torch.inference_mode(), force_triton(), engine_calls(engine) as inference_calls:
        engine(act.clone(), residues, single, pair)
    assert inference_calls[0] == 0


@pytest.mark.parametrize("family", VARIANT_FAMILIES)
@pytest.mark.parametrize(("padded", "real"), [(512, 512), (448, 301)])
@pytest.mark.parametrize("samples", [None, 3])
def test_variant_atom_transformer_portable(family, samples, padded, real, monkeypatch):
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from foldforge.modules.dense.diffusion_transformer import DiffusionCrossAttTransformer
    from foldforge.modules.dense.spec import SPECS
    from foldforge.utils import atom_layout

    torch.manual_seed(13)
    layer = DiffusionCrossAttTransformer(spec=SPECS[family])
    randomize_distinct(layer, 0.08)
    truth, ref, engine, counts = install(layer)
    assert counts["engine_atom_variant_blocks"] == 3
    subsets = padded // 32
    shape = (samples,) if samples else ()
    act = torch.randn(*shape, subsets, 32, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(subsets, 32, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(subsets, 32, 128, 16, device="cuda", dtype=torch.bfloat16)
    atoms = torch.arange(padded, device="cuda").view(subsets, 32) < real
    gather = af3_gather(real, padded)
    keys_mask = atom_layout.convert(gather, atoms, layout_axes=(-2, -1))
    uid = torch.arange(padded, device="cuda").view(subsets, 32) // 11
    keys_uid = atom_layout.convert(gather, uid, layout_axes=(-2, -1))
    same_conformer = family == "chai1"
    pair_mask = (
        (uid[:, :, None] == keys_uid[:, None, :]) & atoms[:, :, None] & keys_mask[:, None, :]
        if same_conformer
        else None
    )
    flat_uid = uid.reshape(padded) if same_conformer else None
    engine_inputs = (pair[None], atoms.reshape(1, padded), flat_uid, None)

    def run(module: torch.nn.Module, dtype: torch.dtype, **extra: object) -> torch.Tensor:
        return module(
            queries_act=act.to(dtype).clone(),
            queries_mask=atoms,
            queries_to_keys=gather,
            keys_mask=keys_mask,
            queries_single_cond=cond.to(dtype),
            keys_single_cond=atom_layout.convert(gather, cond.to(dtype), layout_axes=(-3, -2)),
            pair_cond=pair.to(dtype),
            pair_mask=pair_mask,
            **extra,
        )

    calls = count_portable(monkeypatch, "atom_stack_forward")
    with torch.no_grad(), force_triton(), engine_calls(engine) as stack_calls:
        expected = run(truth, torch.float32)
        reference = run(ref, torch.bfloat16)
        actual = run(engine, torch.bfloat16, engine_inputs=engine_inputs)
    assert calls[0] == stack_calls[0] == 1
    assert rms_error(actual[..., atoms, :], expected[..., atoms, :]) <= (
        rms_error(reference[..., atoms, :], expected[..., atoms, :]) + BF16_ROUNDING
    )
    assert rms_error(actual[..., atoms, :], reference[..., atoms, :]) < 0.02


@pytest.mark.parametrize("family", VARIANT_FAMILIES)
def test_variant_serves_asks_the_portable_path_on_another_card(family, monkeypatch, tmp_path):
    """A recorded card that is not an H100 is served by the portable path; an unrecorded one warns (the engine's warning) and keeps the
    dense blocks; an H100 / B200 never asks for it without the A/B switch."""
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from miniworld_engine.kernels.dit_variants import interface
    from miniworld_engine.modules.dispatch import UnrecordedCardWarning

    from foldforge.modules.engine.modules import variant_serves
    from foldforge.modules.engine.transformers import serve_token

    truth, ref, engine, act, single, pair, residues = variant_token_case(family)
    stack = engine_of(engine)
    key_mask = residues
    token_args = (act, single, pair, key_mask)

    with fake_card(monkeypatch, tmp_path, "NVIDIA RTX A6000", (8, 6), recorded=True):
        with torch.no_grad():
            assert not interface.serves(stack, *token_args)
            assert variant_serves(stack, *token_args)  # the stack then routes itself: portable, as interface.serves is False
        assert variant_serves(stack, *token_args)  # a gradient is served as well (the manifest holds the backward kernels)

    with fake_card(monkeypatch, tmp_path / "empty", "NVIDIA RTX A6000", (8, 6), recorded=False), torch.no_grad():
        with pytest.warns(UnrecordedCardWarning):
            assert not variant_serves(stack, *token_args)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert serve_token(engine, act, residues, single, pair, key_mask[None], None) is None  # the dense blocks run

    for name, capability in (("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA B200", (10, 0))):
        with fake_card(monkeypatch, tmp_path / name[7:11], name, capability, recorded=True):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert not variant_serves(stack, *token_args)  # a gradient: not the CUDA path either
            with force_triton():
                assert variant_serves(stack, *token_args)


def test_variant_atom_stack_asks_the_portable_path_on_another_card(monkeypatch, tmp_path):
    pytest.importorskip("miniworld_engine.modules.dit_variant")
    from foldforge.modules.dense.diffusion_transformer import DiffusionCrossAttTransformer
    from foldforge.modules.dense.spec import SPECS
    from foldforge.modules.engine.modules import variant_serves

    layer = DiffusionCrossAttTransformer(spec=SPECS["boltz2"])
    randomize_distinct(layer, 0.08)
    _, _, engine, _ = install(layer)
    stack = engine_of(engine)
    x = torch.randn(1, 512, 128, device="cuda", dtype=torch.bfloat16)
    cond = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(16, 32, 128, 16, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(512, device="cuda", dtype=torch.bool)
    with fake_card(monkeypatch, tmp_path, "NVIDIA RTX A6000", (8, 6), recorded=True), torch.no_grad():
        assert variant_serves(stack, x, cond, pair, mask, None)
        assert not variant_serves(stack, x, cond, pair[:, :, :64], mask, None)  # keys other than the atom window's
    with force_triton():
        assert variant_serves(stack, x, cond, pair, mask, None)


# --- FP32 (TF32 tensor core) diffusion transformers: precision: fp32 / af3_default -------------------------
#
# The engine's FP32 token DiT and atom blocks multiply on TF32 tensor cores, so the check is against the
# dense FP32 statements run with TF32 allowed (the same rounding), with exact FP32 as the truth.

TF32_ROUNDING = 8e-3


def install_fp32(module: torch.nn.Module, *, tf32: bool = True, diffusion_tf32: bool = False):
    """FP32 truth, dense FP32 module, FP32 module with the engine modules installed (``precision: fp32``)."""
    from foldforge.modules.engine.routing import set_tf32_policy

    truth = copy.deepcopy(module).cuda().float()
    ref = configure_model(copy.deepcopy(module), "pytorch", torch.float32, "cuda")
    engine = configure_model(module, "miniworld", torch.float32, "cuda")
    counts = install_engine_modules(
        engine, torch.empty(0, dtype=torch.float32, device="cuda")
    )
    set_tf32_policy(engine, tf32=tf32, diffusion_tf32=diffusion_tf32)
    return truth, ref, engine, counts


@contextlib.contextmanager
def tf32_matmul(on: bool = True):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if on else "highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def fp32_token_layer(blocks: int = 4):
    from foldforge.modules.dense.diffusion_transformer import DiffusionTransformer

    torch.manual_seed(5)
    layer = DiffusionTransformer(768, 384, 128, 16, blocks)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if p.ndim > 1:
                p.normal_(0, 0.04)
            elif "weight" in name:
                p.fill_(1.0)
            else:
                p.zero_()
    return layer


@pytest.mark.parametrize("length", [128, 384, 1024])
def test_fp32_diffusion_transformer(length):
    """FP32 activations take the engine's TF32 token DiT at every multiple of 128 (not only 128 .. 768)."""
    layer = fp32_token_layer()
    truth, ref, engine, counts = install_fp32(layer)
    assert counts["engine_dit_blocks"] == 4
    assert engine.foldforge_engine_fp32
    samples = 5
    act = torch.randn(samples, length, 768, device="cuda")
    single = torch.randn(samples, length, 384, device="cuda")
    pair = torch.randn(length, length, 128, device="cuda")
    residues = torch.rand(length, device="cuda") > 0.15
    with torch.no_grad(), tf32_matmul(), engine_calls(engine) as calls:
        expected = truth(act, residues, single, pair)
        reference = ref(act.clone(), residues, single, pair)
        actual = engine(act.clone(), residues, single, pair)
    assert calls[0] == 4
    assert rms_error(actual[:, residues], expected[:, residues]) <= (
        rms_error(reference[:, residues], expected[:, residues]) + TF32_ROUNDING
    )
    assert rms_error(actual[:, residues], reference[:, residues]) < TF32_ROUNDING


def test_fp32_diffusion_transformer_needs_tf32_and_whole_tiles():
    """Without TF32 allowed, or at a length the kernels do not tile, the dense FP32 statements run."""
    layer = fp32_token_layer(4)
    truth, _, engine, _ = install_fp32(layer, tf32=False)
    samples = 3
    for length, tf32 in ((256, False), (200, True)):
        from foldforge.modules.engine.routing import set_tf32_policy

        set_tf32_policy(engine, tf32=tf32)
        act = torch.randn(samples, length, 768, device="cuda")
        single = torch.randn(samples, length, 384, device="cuda")
        pair = torch.randn(length, length, 128, device="cuda")
        residues = torch.ones(length, dtype=torch.bool, device="cuda")
        with torch.no_grad(), tf32_matmul(tf32), engine_calls(engine) as calls:
            expected = truth(act, residues, single, pair)
            actual = engine(act.clone(), residues, single, pair)
        assert calls[0] == 0
        assert rms_error(actual, expected) < (TF32_ROUNDING if tf32 else 1e-5)


def test_fp32_diffusion_transformer_extra_pair_bias_stays_dense():
    """The structural-token bias is a BF16 feature of the engine's token DiT: FP32 keeps the dense statements."""
    layer = fp32_token_layer(4)
    _, _, engine, _ = install_fp32(layer)
    length = 256
    act = torch.randn(2, length, 768, device="cuda")
    single = torch.randn(2, length, 384, device="cuda")
    pair = torch.randn(length, length, 128, device="cuda")
    extra = torch.randn(length, length, device="cuda")
    residues = torch.ones(length, dtype=torch.bool, device="cuda")
    with torch.no_grad(), tf32_matmul(), engine_calls(engine) as calls:
        engine(act, residues, single, pair, extra_pair_bias=extra)
    assert calls[0] == 0


@pytest.mark.parametrize("family", ["alphafold3", "protenix1"])
@pytest.mark.parametrize("real", [512, 301])
def test_fp32_atom_transformer(family, real):
    from foldforge.modules.dense.diffusion_transformer import (
        DiffusionCrossAttTransformer,
    )
    from foldforge.modules.dense.spec import SPECS
    from foldforge.utils import atom_layout

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
    truth, ref, engine, counts = install_fp32(layer)
    assert counts["engine_atom_blocks"] == 3
    padded, samples = 512, 3
    subsets = padded // 32
    act = torch.randn(samples, subsets, 32, 128, device="cuda")
    cond = torch.randn(subsets, 32, 128, device="cuda")
    pair = torch.randn(subsets, 32, 128, 16, device="cuda")
    atoms = torch.arange(padded, device="cuda").view(subsets, 32) < real
    gather = af3_gather(real, padded)
    keys_mask = atom_layout.convert(gather, atoms, layout_axes=(-2, -1))
    keys_cond = atom_layout.convert(gather, cond, layout_axes=(-3, -2))

    def run(module: torch.nn.Module) -> torch.Tensor:
        return module(
            queries_act=act.clone(),
            queries_mask=atoms,
            queries_to_keys=gather,
            keys_mask=keys_mask,
            queries_single_cond=cond,
            keys_single_cond=keys_cond,
            pair_cond=pair,
        )

    with torch.no_grad(), tf32_matmul(), engine_calls(engine) as calls:
        expected = run(truth)
        reference = run(ref)
        actual = run(engine)
    assert calls[0] == 3
    assert rms_error(actual[..., atoms, :], expected[..., atoms, :]) <= (
        rms_error(reference[..., atoms, :], expected[..., atoms, :]) + TF32_ROUNDING
    )
    # TF32 not allowed: the dense FP32 statements
    from foldforge.modules.engine.routing import set_tf32_policy

    set_tf32_policy(engine, tf32=False)
    with torch.no_grad(), tf32_matmul(False), engine_calls(engine) as calls:
        dense = run(engine)
    assert calls[0] == 0
    assert rms_error(dense[..., atoms, :], expected[..., atoms, :]) < 1e-3


def fp32_loaded_model(family: str = "protenix1", *, af3_default: bool = False):
    """A model after the loader's installs: ``precision: fp32`` (FP32 everywhere) or af3_default (BF16 trunk)."""
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.modules.dense.spec import SPECS
    from foldforge.modules.engine.install import install_engine
    from foldforge.modules.engine.routing import set_tf32_policy

    model = AlphaFold3(
        num_recycles=1, num_samples=1, diffusion_steps=2, spec=SPECS[family]
    )
    model.foldforge_load_report = {}
    dtype = torch.bfloat16 if af3_default else torch.float32
    model = configure_model(model, "miniworld", dtype, "cuda")
    if af3_default:
        # AF3's release: BF16 trunk, FP32 diffusion (what loading.py restores with preserve_dtype)
        for name, p in model.named_parameters():
            if name.startswith("diffusion_head"):
                p.data = p.data.float()
    model = install_engine(model, dtype, "cuda", modules=True)
    set_tf32_policy(model, tf32=False, diffusion_tf32=af3_default)
    return model


def test_fp32_model_attaches_the_diffusion_transformers_only():
    """precision: fp32 -- only the modules with an FP32 engine kernel get an engine module."""
    report = fp32_loaded_model().foldforge_load_report
    assert report["engine_dit_blocks"] == 24
    assert report["engine_atom_blocks"] == (9 if _ENGINE_HAS_ATOM_VARIANT else 0)
    for trunk in (
        "engine_trimul",
        "engine_triangle_attention",
        "engine_msa_pwa",
        "engine_opm",
        "engine_single_attention",
    ):
        assert report[trunk] == 0


def test_af3_default_trunk_attaches_like_bf16_and_the_diffusion_serves_fp32():
    """af3_default (AF3's release): the BF16 trunk is on the engine as in bf16 mode, the FP32 diffusion on its TF32 kernels."""
    from foldforge.modules.engine.routing import enabled, enabled_fp32

    model = fp32_loaded_model("alphafold3", af3_default=True)
    reference = loaded_model(diffusion_master=True).foldforge_load_report
    report = model.foldforge_load_report
    for key in (
        "engine_trimul",
        "engine_triangle_attention",
        "engine_msa_pwa",
        "engine_opm",
        "engine_single_attention",
        "engine_dit_blocks",
        "engine_atom_blocks",
    ):
        assert report[key] == reference[key] > 0
    x32 = torch.empty(1, device="cuda")
    x16 = x32.bfloat16()
    dit = next(m for m in model.modules() if type(m).__name__ == "DiffusionTransformer")
    atoms = next(
        m for m in model.diffusion_head.modules() if type(m).__name__ == "DiffusionCrossAttTransformer"
    )
    tri = next(m for m in model.modules() if type(m).__name__ == "TriangleMultiplication")
    assert dit.foldforge_engine_fp32 and enabled_fp32(dit, x32) and not enabled(dit, x32)
    assert atoms.foldforge_engine_fp32 and enabled_fp32(atoms, x32)
    # the trunk: BF16 activations on the BF16 path, and no FP32 engine route (exact FP32 trunk modules stay dense)
    assert enabled(tri, x16) and not enabled_fp32(tri, x32)


def test_fp32_transition_takes_the_engine_where_tf32_is_allowed():
    from foldforge.modules.engine.routing import set_tf32_policy, transition_residual
    from foldforge.modules.dense.primitives import Transition

    torch.manual_seed(3)
    layer = Transition(128, 4)
    layer = randomize(layer).cuda().eval()
    layer = configure_model(layer, "miniworld", torch.float32, "cuda")
    x = torch.randn(1, 256, 256, 128, device="cuda")
    with torch.no_grad(), tf32_matmul():
        set_tf32_policy(layer, tf32=False)
        dense = transition_residual(layer, x)
        set_tf32_policy(layer, tf32=True)
        served = transition_residual(layer, x)
    assert rms_error(served, dense) < TF32_ROUNDING


@pytest.mark.parametrize("family", ["alphafold3", "protenix1"])
@pytest.mark.parametrize("length", [96, 200])
def test_confidence_head_runs_the_samples_as_one_batch(family, length):
    """S diffusion samples in one head call equal S sequential calls (shared trunk, own distances)."""
    from foldforge.modules.dense.head import ConfidenceHead
    from foldforge.modules.dense.spec import SPECS
    from foldforge.modules.engine.pairformer import install_pairformers
    from foldforge.utils import atom_layout

    spec = SPECS[family]
    torch.manual_seed(11)
    head = ConfidenceHead(
        c_single=spec.seq_channel,
        c_pair=spec.pair_channel,
        c_target_feat=spec.target_feat_channel,
        n_pairformer_layers=2,
        spec=spec,
    )
    randomize(head, 0.04)
    head = configure_model(head, "miniworld")
    head.foldforge_load_report = {}
    install_pairformers(head)
    install_engine_modules(head, torch.empty(0, dtype=torch.bfloat16, device="cuda"))
    atoms, samples = 24, 3
    residues = torch.rand(length, device="cuda") > 0.1
    gather = atom_layout.GatherInfo(
        gather_idxs=torch.arange(length, device="cuda") * atoms + 4,
        gather_mask=torch.ones(length, dtype=torch.bool, device="cuda"),
        input_shape=(length, atoms),
    )
    embeddings = {
        "pair": torch.randn(length, length, spec.pair_channel, device="cuda"),
        "single": torch.randn(length, spec.seq_channel, device="cuda"),
        "target_feat": torch.randn(length, spec.target_feat_channel, device="cuda"),
    }
    asym_id = (torch.arange(length, device="cuda") // (length // 2)).to(torch.int32)
    positions = 12.0 * torch.randn(samples, length, atoms, 3, device="cuda")
    kwargs = dict(
        embeddings=embeddings,
        seq_mask=residues,
        token_atoms_to_pseudo_beta=gather,
        asym_id=asym_id,
    )
    calls = [0]
    for block in head.confidence_pairformer:
        block.block.tri_atten_starting.register_forward_hook(
            lambda *_: calls.__setitem__(0, calls[0] + 1)
        )
    with torch.no_grad():
        sequential = [
            head(dense_atom_positions=positions[i], **kwargs) for i in range(samples)
        ]
        sequential_calls = calls[0]
        batched = head(dense_atom_positions=positions, **kwargs)
    assert sequential_calls == samples * 2
    assert calls[0] == sequential_calls + 2  # one batched call per block, not per sample
    assert batched.keys() == sequential[0].keys()
    for key, value in batched.items():
        expected = torch.stack([out[key] for out in sequential])
        assert value.shape == expected.shape, key
        assert rms_error(value, expected) <= BF16_ROUNDING, key
