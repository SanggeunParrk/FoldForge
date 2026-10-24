"""One seam, four backends: ``backend`` / ``engine.ops`` / ``engine.unsupported`` reach the executed kernel.

The plan tests need no GPU. The parity tests install an op on each backend that serves it, through the same
install path a loaded model takes (``configure_model`` -> ``apply_plan`` -> ``install_engine_modules``), and check
the result against the FP32 truth with the engine's own tolerance: as close as the same module's PyTorch reference
is, up to bf16 rounding. The Anthropic payloads are located by environment variables and the tests skip without them.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import pydoc
from unittest import mock

import pytest
import torch

from foldforge.models.config import Config
from foldforge.modules.engine import backends
from foldforge.modules.engine.backends import (
    BACKENDS,
    OP_CLASSES,
    OPS,
    SERVES,
    BackendPlan,
    BackendUnsupportedError,
)

LOGGER = "foldforge.backend"
#: Where the Anthropic payloads are (environment variables name locations). The engine's own ``miniworld`` modules use a
#: payload that is named, so only the tests of the anthropic backend see them.
PAYLOAD = {
    name: os.environ[name]
    for name in ("TRIMUL_NATIVE_BUILD_DIR", "OPT_CORE_DIR", "MINIWORLD_ANTHROPIC_ROOT")
    if os.environ.get(name)
}


@pytest.fixture(autouse=True)
def engine_settings_restored():
    """``apply_plan`` switches engine settings for the anthropic backend; each test starts from the defaults."""
    from miniworld_engine import settings

    before = settings.current()
    yield
    settings.configure(
        opm_train=before.opm_train,
        pwa_train=before.pwa_train,
        pwa_infer=before.pwa_infer,
    )


@pytest.fixture
def payload(monkeypatch):
    """The locations of the Anthropic payloads, for the tests of the anthropic backend."""
    for name, value in PAYLOAD.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def payload_only_for_anthropic(request, monkeypatch):
    """Every test sees the payload locations only when it is about the anthropic backend."""
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    if "payload" in request.fixturenames or params.get("backend") == "anthropic":
        for name, value in PAYLOAD.items():
            monkeypatch.setenv(name, value)
    else:
        for name in PAYLOAD:
            monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------------------------- the table and the plan


def test_the_four_backends_and_nothing_else():
    assert set(BACKENDS) == {"miniworld", "pytorch", "cuequivariance", "anthropic"}
    assert all(set(served) <= set(BACKENDS) - {"pytorch"} for served in SERVES.values())


def test_miniworld_serves_every_op_so_the_default_loses_nothing():
    assert all("miniworld" in served for served in SERVES.values())
    assert set(OP_CLASSES) <= set(OPS)


def test_the_default_plan_is_miniworld_everywhere_and_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = BackendPlan.make().resolve()
    assert {op: plan.of(op) for op in OPS} == dict.fromkeys(OPS, "miniworld")
    assert plan.report()["fallbacks"] == {}
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_pytorch_is_pytorch_everywhere_and_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = BackendPlan.make("pytorch").resolve()
    assert {plan.of(op) for op in OPS} == {"pytorch"}
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_an_op_without_a_kernel_falls_back_with_one_log_line(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = BackendPlan.make("cuequivariance").resolve()
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    for op in OPS:
        served = "cuequivariance" in SERVES[op]
        assert plan.of(op) == ("cuequivariance" if served else "pytorch")
        assert (op in lines[0]) is (not served)
    assert set(plan.report()["fallbacks"]) == {
        op for op in OPS if "cuequivariance" not in SERVES[op]
    }


def test_unsupported_error_raises_instead_of_falling_back():
    with pytest.raises(BackendUnsupportedError, match="opm"):
        BackendPlan.make("cuequivariance", unsupported="error").resolve()


def test_per_op_override_pins_one_op():
    plan = BackendPlan.make(
        "pytorch", {"trimul": "miniworld", "opm": "anthropic"}
    ).resolve()
    assert plan.requested("trimul") == "miniworld"
    assert plan.requested("transition") == "pytorch"
    assert plan.of("trimul") == "miniworld"
    assert plan.of("single_attention") == "pytorch"


@pytest.mark.parametrize(
    ("ops", "message"),
    [
        ({"nonsense": "miniworld"}, "unknown op"),
        ({"trimul": "nonsense"}, "unknown backend"),
    ],
)
def test_bad_override_is_rejected(ops, message):
    with pytest.raises(ValueError, match=message):
        BackendPlan.make("miniworld", ops)


def test_a_payload_variable_next_to_miniworld_is_named(monkeypatch, caplog):
    monkeypatch.setenv("TRIMUL_NATIVE_BUILD_DIR", "/nonexistent")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        BackendPlan.make("miniworld").resolve()
    assert "TRIMUL_NATIVE_BUILD_DIR" in caplog.text


def test_config_defaults_and_values():
    config = Config()
    assert config.backend == "miniworld"
    assert config.engine.ops == {}
    assert config.engine.unsupported == "fallback"
    for backend in BACKENDS:
        assert Config(backend=backend).backend == backend
    config = Config.model_validate(
        {
            "backend": "pytorch",
            "engine": {"ops": {"trimul": "anthropic"}, "unsupported": "error"},
        }
    )
    assert config.engine.ops == {"trimul": "anthropic"}
    assert config.engine.unsupported == "error"


@pytest.mark.parametrize(
    "bad",
    [
        {"backend": "nonsense"},
        {"backend": ""},
        {"engine": {"ops": {"trimul": "nonsense"}}},
        {"engine": {"ops": {"flash": "miniworld"}}},
        {"engine": {"unsupported": "ignore"}},
    ],
)
def test_config_rejects_what_it_does_not_know(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate(bad)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_cli_takes_each_backend_and_only_those(backend):
    from foldforge.models.io import cli

    arguments = [
        "--ccd-db",
        "ccd",
        "--checkpoint",
        "weights.pt",
        "--input",
        "input.json",
    ]
    assert cli.parse("af3", [*arguments, "--backend", backend]).backend == backend
    with pytest.raises(SystemExit):
        cli.parse("af3", [*arguments, "--backend", "nonsense"])


# ------------------------------------------------------------------------------------------------ module-level parity

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
needs_anthropic = pytest.mark.skipif(
    len(PAYLOAD) < 3,
    reason="TRIMUL_NATIVE_BUILD_DIR, OPT_CORE_DIR and MINIWORLD_ANTHROPIC_ROOT name the Anthropic payloads",
)

BF16_ROUNDING = 3e-3
LIKE = None


def rms_error(actual, expected) -> float:
    actual, expected = actual.float(), expected.float()
    return (
        (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    ).item()


def randomize(module, std=0.08):
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim > 1:
                p.normal_(0, std)
    return module


def install(module, backend, ops=None, unsupported="fallback"):
    """The path a loaded model takes for one dense module: configure, apply the plan, attach the engine modules."""
    from foldforge.modules.engine.configure import configure_model
    from foldforge.modules.engine.install import apply_plan
    from foldforge.modules.engine.modules import install_engine_modules

    plan = BackendPlan.make(backend, ops, unsupported).resolve(torch.device("cuda"))
    module = configure_model(module, backend)
    apply_plan(module, plan)
    like = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    if plan.uses("miniworld", "anthropic"):
        install_engine_modules(module, like, plan=plan)
    return module


def masks(length, drop=0.15):
    residues = torch.rand(length, device="cuda") > drop
    return residues, residues[:, None] & residues[None, :]


@contextlib.contextmanager
def kernel_calls(built, target):
    """Count the calls of the kernel a backend is supposed to run: a patched entry point, or ``"engine"`` for the engine
    modules attached to ``built`` (forward hooks)."""
    if target == "engine":
        from foldforge.modules.engine.modules import engine_of

        attached = engine_of(built)
        modules = attached if isinstance(attached, list) else [attached]
        calls = [0]
        handles = [
            m.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
            for m in modules
        ]
        try:
            yield lambda: calls[0]
        finally:
            for handle in handles:
                handle.remove()
        return
    with mock.patch(target, wraps=pydoc.locate(target)) as patched:
        yield lambda: patched.call_count


#: (op, backend) -> what must be called for the backend to have served the op.
KERNEL = {
    ("trimul", "miniworld"): "engine",
    ("trimul", "cuequivariance"): "cuequivariance_torch.triangle_multiplicative_update",
    (
        "trimul",
        "anthropic",
    ): "miniworld_engine.integrations.anthropic_trimul.update_unidirectional",
    ("triangle_attention", "miniworld"): "engine",
    ("triangle_attention", "cuequivariance"): "cuequivariance_torch.triangle_attention",
    (
        "triangle_attention",
        "anthropic",
    ): "miniworld_engine.integrations.anthropic.triangle_attention",
    ("opm", "miniworld"): "engine",
    (
        "opm",
        "anthropic",
    ): "miniworld_engine.integrations.anthropic_msa.outer_product_mean",
    ("msa_pwa", "miniworld"): "engine",
    (
        "msa_pwa",
        "anthropic",
    ): "miniworld_engine.integrations.anthropic_msa.pair_weighted_averaging",
    ("transition", "miniworld"): "miniworld_engine.ops.transition",
    ("transition", "anthropic"): "miniworld_engine.integrations.anthropic.transition",
    ("single_attention", "cuequivariance"): "cuequivariance_torch.attention_pair_bias",
}


def parity(make, run, backend, op, *, real, tolerance=BF16_ROUNDING):
    """FP32 truth vs the pytorch backend vs ``backend``: no worse than the reference plus bf16 rounding, and the
    backend's own kernel is what ran."""
    torch.manual_seed(7)
    layer = randomize(make())
    truth = copy.deepcopy(layer).cuda().float()
    from foldforge.modules.engine.configure import configure_model

    reference_layer = configure_model(copy.deepcopy(layer), "pytorch")
    with torch.no_grad():
        expected = run(truth, float_input=True)
        reference = run(reference_layer)
    reference_error = rms_error(reference[real], expected[real])
    built = install(copy.deepcopy(layer), backend)
    with torch.no_grad(), kernel_calls(built, KERNEL[op, backend]) as calls:
        actual = run(built)
    assert calls() >= 1, f"{backend} did not run its {op} kernel"
    assert torch.isfinite(actual.float()).all(), backend
    error = rms_error(actual[real], expected[real])
    assert error <= reference_error + tolerance, (backend, error, reference_error)
    return error


def pair_case(length=256):
    x = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    residues, mask = masks(length)
    return x, mask, residues[:, None] & residues[None, :]


@needs_cuda
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize(
    "backend",
    ["miniworld", "cuequivariance", pytest.param("anthropic", marks=needs_anthropic)],
)
def test_trimul_on_each_backend(backend, outgoing):
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication
    from foldforge.modules.engine.routing import triangle_residual

    x, mask, real = pair_case()

    def run(layer, *, float_input=False) -> torch.Tensor:
        return triangle_residual(layer, (x.float() if float_input else x).clone(), mask)

    parity(
        lambda: TriangleMultiplication(128, _outgoing=outgoing),
        run,
        backend,
        "trimul",
        real=real,
    )


@needs_cuda
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    "backend",
    ["miniworld", "cuequivariance", pytest.param("anthropic", marks=needs_anthropic)],
)
def test_triangle_attention_on_each_backend(backend, transpose):
    from foldforge.modules.dense.attention import GridSelfAttention

    x, mask, real = pair_case()

    def run(layer, *, float_input=False) -> torch.Tensor:
        return layer.residual((x.float() if float_input else x).clone(), mask)

    parity(
        lambda: GridSelfAttention(c_pair=128, num_head=4, transpose=transpose),
        run,
        backend,
        "triangle_attention",
        real=real,
    )


@needs_cuda
@pytest.mark.parametrize(
    "backend", ["miniworld", pytest.param("anthropic", marks=needs_anthropic)]
)
def test_msa_ops_on_each_backend(backend):
    from foldforge.modules.dense.attention import MSAAttention
    from foldforge.modules.dense.primitives import OuterProductMean

    rows, length = 64, 256
    msa = torch.randn(rows, length, 64, device="cuda", dtype=torch.bfloat16)
    msa_mask = torch.ones(rows, length, device="cuda")
    msa_mask[:, ::17] = 0
    pair = torch.randn(length, length, 128, device="cuda", dtype=torch.bfloat16)
    real = torch.ones(rows, length, 64, device="cuda", dtype=torch.bool)
    real[:, ::17] = False

    def run_pwa(layer, *, float_input=False) -> torch.Tensor:
        m, z = (msa.float(), pair.float()) if float_input else (msa, pair)
        return layer.residual(m.clone(), msa_mask, z)

    # the release's cell is (d_msa 64, d_pair 128, 8 heads, head width 32); AF3's own head width is 8 (below)
    width = 32 if backend == "anthropic" else None
    parity(
        lambda: MSAAttention(64, 128, 8, value_dim=width),
        run_pwa,
        backend,
        "msa_pwa",
        real=real,
    )

    def run_opm(layer, *, float_input=False) -> torch.Tensor:
        return layer(msa.float() if float_input else msa, msa_mask)

    pair_real = torch.ones(length, length, 128, device="cuda", dtype=torch.bool)
    pair_real[::17] = False
    pair_real[:, ::17] = False
    # the release's epilogue divides by the pair count before it adds the bias: the OpenFold-lineage convention (AF3's is below)
    convention = (
        {"bias_after_norm": True, "clamped_norm": True}
        if backend == "anthropic"
        else {}
    )
    parity(
        lambda: OuterProductMean(64, 128, 32, **convention),
        run_opm,
        backend,
        "opm",
        real=pair_real,
    )


@needs_cuda
@pytest.mark.parametrize(
    "backend", ["miniworld", pytest.param("anthropic", marks=needs_anthropic)]
)
@pytest.mark.parametrize("width", [128, 384])
def test_transition_on_each_backend(backend, width):
    from foldforge.modules.dense.primitives import Transition
    from foldforge.modules.engine.routing import transition_residual

    x = torch.randn(256, width, device="cuda", dtype=torch.bfloat16)
    real = torch.ones_like(x, dtype=torch.bool)

    def run(layer, *, float_input=False) -> torch.Tensor:
        return transition_residual(layer, (x.float() if float_input else x).clone())

    parity(lambda: Transition(width, 4), run, backend, "transition", real=real)


@needs_cuda
def test_single_attention_on_cuequivariance_and_miniworld():
    from foldforge.modules.dense.diffusion_transformer import SelfAttention
    from foldforge.modules.engine.pairformer import _SingleAttention

    length = 256
    single = torch.randn(1, length, 384, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(1, length, length, 128, device="cuda", dtype=torch.bfloat16)
    residues, _ = masks(length)
    mask = residues[None].float()
    real = residues[None, :, None].expand_as(single)

    class Source(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.single_pair_logits_norm = torch.nn.LayerNorm(128)
            self.single_pair_logits_projection = torch.nn.Linear(128, 16, bias=False)
            self.single_attention_ = SelfAttention(c_x=384, num_head=16)

    def make() -> torch.nn.Module:
        return _SingleAttention(Source())

    def run(layer, *, float_input=False) -> torch.Tensor:
        s, z = (single.float(), pair.float()) if float_input else (single, pair)
        return layer(s.clone(), z, mask)

    parity(make, run, "cuequivariance", "single_attention", real=real)


# ---------------------------------------------------------------- the fallback is stated, never silent; overrides work


@needs_cuda
def test_the_engine_module_of_an_op_runs_only_on_its_backend():
    """``backend: pytorch`` with ``trimul: miniworld`` attaches the engine TriMul to the trimul module and nothing else."""
    from foldforge.modules.dense.primitives import OuterProductMean
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication
    from foldforge.modules.engine.modules import engine_of

    mixed = torch.nn.ModuleDict(
        {
            "trimul": TriangleMultiplication(128, _outgoing=True),
            "opm": OuterProductMean(64, 128, 32),
        }
    )
    mixed = install(mixed, "pytorch", {"trimul": "miniworld"})
    assert engine_of(mixed["trimul"]) is not None
    assert engine_of(mixed["opm"]) is None
    assert mixed["opm"].foldforge_implementation.value == "pytorch"
    assert mixed["trimul"].foldforge_implementation.value == "miniworld_engine"


@needs_cuda
def test_backend_without_a_kernel_attaches_nothing_and_equals_pytorch():
    """cuequivariance has no OuterProductMean kernel: the op is the PyTorch reference, bit for bit."""
    from foldforge.modules.dense.primitives import OuterProductMean
    from foldforge.modules.engine.configure import configure_model
    from foldforge.modules.engine.modules import engine_of

    torch.manual_seed(3)
    layer = randomize(OuterProductMean(64, 128, 32))
    msa = torch.randn(32, 128, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(32, 128, device="cuda")
    reference = configure_model(copy.deepcopy(layer), "pytorch")
    built = install(copy.deepcopy(layer), "cuequivariance")
    assert engine_of(built) is None
    with torch.no_grad():
        assert torch.equal(built(msa, mask), reference(msa, mask))


@needs_cuda
@pytest.mark.skipif(len(PAYLOAD) < 3, reason="the Anthropic payloads are not named")
@pytest.mark.usefixtures("payload")
def test_af3_msa_head_width_is_refused_by_the_release_and_stated(caplog):
    """AF3's pair weighted averaging has 8-wide heads; the release's cell is 32 wide: one line, then the reference."""
    from foldforge.modules.dense.attention import MSAAttention
    from foldforge.modules.engine.configure import configure_model

    torch.manual_seed(2)
    layer = randomize(MSAAttention(64, 128, 8))
    msa = torch.randn(32, 128, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(32, 128, device="cuda")
    pair = torch.randn(128, 128, 128, device="cuda", dtype=torch.bfloat16)
    reference = configure_model(copy.deepcopy(layer), "pytorch")
    backends.forget_logged_refusals()
    built = install(copy.deepcopy(layer), "anthropic")
    with torch.no_grad(), caplog.at_level(logging.WARNING, logger=LOGGER):
        out = built.residual(msa.clone(), mask, pair)
        expected = reference.residual(msa.clone(), mask, pair)
    assert (
        len([r for r in caplog.records if "refuses a msa_pwa call" in r.getMessage()])
        == 1
    )
    assert torch.equal(out, expected)


@needs_cuda
@pytest.mark.skipif(len(PAYLOAD) < 3, reason="the Anthropic payloads are not named")
@pytest.mark.usefixtures("payload")
def test_af3_opm_bias_convention_is_refused_by_the_release_and_stated(caplog):
    """AF3 adds the OPM output bias before the division by the pair count, the release's epilogue after: one line, then the reference."""
    from foldforge.modules.dense.primitives import OuterProductMean
    from foldforge.modules.engine.configure import configure_model

    torch.manual_seed(2)
    layer = randomize(OuterProductMean(64, 128, 32))
    msa = torch.randn(32, 128, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(32, 128, device="cuda")
    reference = configure_model(copy.deepcopy(layer), "pytorch")
    backends.forget_logged_refusals()
    built = install(copy.deepcopy(layer), "anthropic")
    with torch.no_grad(), caplog.at_level(logging.WARNING, logger=LOGGER):
        out = built(msa, mask)
        expected = reference(msa, mask)
    assert (
        len([r for r in caplog.records if "refuses a opm call" in r.getMessage()]) == 1
    )
    assert torch.equal(out, expected)


@needs_cuda
@pytest.mark.skipif(len(PAYLOAD) < 3, reason="the Anthropic payloads are not named")
@pytest.mark.usefixtures("payload")
def test_a_call_the_payload_refuses_runs_the_reference_with_one_line_or_raises(caplog):
    """A pair width the payload has no TriMul unit for (192): stated once, then the reference."""
    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication
    from foldforge.modules.engine.configure import configure_model
    from foldforge.modules.engine.routing import triangle_residual

    torch.manual_seed(5)
    layer = randomize(TriangleMultiplication(192, _outgoing=True))
    x = torch.randn(128, 128, 192, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(128, 128, device="cuda", dtype=torch.bool)
    reference = configure_model(copy.deepcopy(layer), "pytorch")
    backends.forget_logged_refusals()
    built = install(copy.deepcopy(layer), "anthropic")
    with torch.no_grad(), caplog.at_level(logging.WARNING, logger=LOGGER):
        out = triangle_residual(built, x.clone(), mask)
        triangle_residual(built, x.clone(), mask)
        expected = triangle_residual(reference, x.clone(), mask)
    refusals = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and "refuses a trimul call" in r.getMessage()
    ]
    assert len(refusals) == 1, caplog.text
    assert torch.equal(out, expected)
    strict = install(
        copy.deepcopy(layer), "pytorch", {"trimul": "anthropic"}, unsupported="error"
    )
    backends.forget_logged_refusals()
    with torch.no_grad(), pytest.raises(BackendUnsupportedError):
        triangle_residual(strict, x.clone(), mask)


@needs_cuda
def test_sampler_ops_are_off_unless_miniworld():
    """The engine's fused rigid-motion / pooling kernels are miniworld's: other backends take the torch ops."""
    from foldforge.modules.dense.atom_cross_attention import AtomCrossAttEncoder

    assert "AtomCrossAttEncoder" in OP_CLASSES["sampler_ops"]
    holder = torch.nn.Module()
    holder.child = torch.nn.Module()
    from foldforge.modules.engine.install import apply_plan

    for backend, expected in (
        ("miniworld", True),
        ("pytorch", False),
        ("cuequivariance", False),
        ("anthropic", False),
    ):
        apply_plan(holder, BackendPlan.make(backend).resolve(torch.device("cuda")))
        assert holder.foldforge_sampler_ops is expected
    assert AtomCrossAttEncoder  # the class the flag is set on
