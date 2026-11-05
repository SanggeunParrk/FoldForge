"""Check sampler glue and hoisted conditioning against the torch path."""

from __future__ import annotations

import contextlib

import pytest
import torch

from foldforge.models import sampler_graph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _glue_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from miniworld_engine.kernels.sampler_ops import interface
    except ImportError:
        return False
    return torch.cuda.get_device_capability() == (9, 0) and interface is not None


needs_glue = pytest.mark.skipif(
    not _glue_available(), reason="engine sampler_ops on an H100 required"
)

CONFIGS = {
    "af3": {"gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003, "step_scale": 1.5},
    "edm": {
        "gamma_0": 0.8,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "step_scale": 1.5,
        "churn_gamma": 0.4,
        "churn_sigma_min": 0.05,
        "churn_sigma_max": 50.0,
        "variance_floor": 1e-6,
    },
}


def _scene(samples=3, tokens=16, atoms=24, real=300) -> tuple:
    from foldforge.diffusion.augmentation import (
        masked_dense_rigid_motion,
    )
    from foldforge.models.architectures.af3 import weighted_rigid_align

    torch.manual_seed(1)
    mask = (
        (torch.arange(tokens * atoms, device="cuda") < real)
        .reshape(tokens, atoms)
        .float()
    )
    coords = 20 * torch.randn(samples, tokens, atoms, 3, device="cuda") + 5

    def denoise(noisy, sigma, row=None) -> torch.Tensor:
        # a smooth stand-in for the network: shrinks towards a fixed point, in the frame
        # it is given
        scale = (
            1.0 / (1.0 + sigma)
            if row is None
            else 1.0 / (1.0 + sigma) + 0 * row.float().sum()
        )
        return (0.8 * noisy + 0.3 * torch.sin(noisy / 9.0) * scale) * mask[..., None]

    return mask, coords, denoise, masked_dense_rigid_motion, weighted_rigid_align


@needs_glue
@pytest.mark.parametrize("churn", list(CONFIGS))
@pytest.mark.parametrize("sigmas", [(160.0, 120.0), (3.0, 2.2), (0.04, 0.03)])
def test_fused_step_equals_the_torch_step(churn, sigmas):
    from foldforge.diffusion.edm.sampling import EulerSampler

    mask, coords, denoise, augment, align = _scene()
    sampler = EulerSampler(EulerSampler.Config(**CONFIGS[churn]))
    sigma, sigma_next = (torch.tensor(v, device="cuda") for v in sigmas)

    def build(rigid_mask) -> object:
        return sampler_graph.build_step(
            sampler,
            denoise,
            generator=None,
            augment=lambda c: augment(c, mask),
            align=lambda noisy, denoised: align(noisy, denoised, mask),
            coordinate_dims=2,
            rigid_mask=rigid_mask,
        )

    results = []
    for rigid_mask in (None, mask):
        torch.manual_seed(5)  # the same random draws, in the same order
        results.append(build(rigid_mask)(coords, sigma, sigma_next))
    expected, got = results
    error = (got - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    assert error < 2e-5, error.item()


@needs_glue
def test_the_graph_replays_the_fused_step():
    from foldforge.diffusion.edm.sampling import EulerSampler

    mask, coords, denoise, augment, align = _scene()
    sampler = EulerSampler(EulerSampler.Config(**CONFIGS["edm"]))
    sigmas = torch.tensor([30.0, 12.0, 5.0, 2.0, 0.5], device="cuda")
    kwargs = {
        "device": coords.device,
        "generator": None,
        "augment": lambda c: augment(c, mask),
        "align": lambda noisy, denoised: align(noisy, denoised, mask),
        "coordinate_dims": 2,
        "scopes": contextlib.nullcontext,
    }
    outputs = []
    for rigid_mask in (None, mask):
        torch.manual_seed(9)
        outputs.append(
            sampler_graph.sample(
                sampler,
                denoise,
                tuple(coords.shape),
                sigmas,
                rigid_mask=rigid_mask,
                **kwargs,
            )
        )
    expected, got = outputs
    assert torch.isfinite(got).all()
    assert (
        (got - expected).square().mean().sqrt() / expected.square().mean().sqrt()
    ) < 1e-3


def test_conditioning_rows_replace_the_per_step_computation():
    from foldforge.diffusion.edm.sampling import EulerSampler

    _, coords, _, _, _ = _scene()
    sampler = EulerSampler(EulerSampler.Config(**CONFIGS["af3"]))
    sigmas = torch.tensor([30.0, 12.0, 5.0, 2.0, 0.5, 0.1], device="cuda")

    def noise_term(level) -> torch.Tensor:
        return torch.cos(0.3 * level)

    def denoise(noisy, sigma, row=None) -> torch.Tensor:
        # the noise-level term, made here from sigma_hat or read from the step's row
        term = noise_term(sigma) if row is None else row
        return 0.9 * noisy + 0.05 * term

    kwargs = {
        "device": coords.device,
        "generator": None,
        "augment": None,
        "align": None,
        "coordinate_dims": 2,
        "scopes": contextlib.nullcontext,
    }
    outputs = []
    for conditioning in (None, noise_term):
        torch.manual_seed(4)
        outputs.append(
            sampler_graph.sample(
                sampler,
                denoise,
                tuple(coords.shape),
                sigmas,
                conditioning=conditioning,
                **kwargs,
            )
        )
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-6, atol=1e-6)


def test_hatted_sigmas_follow_the_churn_rule():
    from foldforge.diffusion.edm.sampling import EulerSampler

    sigmas = torch.tensor([160.0, 60.0, 20.0, 0.9, 0.3], device="cuda")
    af3 = sampler_graph._hatted(  # noqa: SLF001 - verify private churn schedule
        EulerSampler(EulerSampler.Config(**CONFIGS["af3"])).config, sigmas
    )
    # gamma_0 is on while sigma_next exceeds gamma_min
    torch.testing.assert_close(
        af3, sigmas[:-1] * torch.tensor([1.8, 1.8, 1.0, 1.0], device="cuda")
    )
    edm = sampler_graph._hatted(  # noqa: SLF001 - verify private churn schedule
        EulerSampler(EulerSampler.Config(**CONFIGS["edm"])).config, sigmas
    )
    torch.testing.assert_close(
        edm, sigmas[:-1] * torch.tensor([1.0, 1.0, 1.4, 1.4], device="cuda")
    )


@needs_glue
def test_encoder_pooling_equals_gather_relu_mask_mean():
    from foldforge.modules.dense import utils
    from foldforge.modules.dense.atom_cross_attention import _pool_engine
    from foldforge.utils import atom_layout

    torch.manual_seed(3)
    samples, subsets, queries, channels, tokens, atoms = 3, 12, 32, 768, 16, 24
    act = torch.randn(
        samples, subsets, queries, channels, device="cuda", dtype=torch.bfloat16
    )
    idx = torch.randint(0, subsets * queries, (tokens, atoms), device="cuda")
    atom_mask = (torch.rand(tokens, atoms, device="cuda") > 0.2).float()
    gather = atom_layout.GatherInfo(idx, atom_mask > 0, (subsets, queries))
    token_atoms = atom_layout.convert(gather, act, layout_axes=(-3, -2))
    expected = utils.mask_mean(
        atom_mask.reshape(1, tokens, atoms, 1), torch.relu(token_atoms), dim=-2
    ).to(token_atoms.dtype)
    got = _pool_engine(act, gather, atom_mask)
    assert got is not None
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected, rtol=2**-7, atol=2**-7)
