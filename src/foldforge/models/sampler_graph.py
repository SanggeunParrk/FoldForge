"""The whole EDM sampler step as one CUDA graph.

A step is: re-pose the iterate (random rigid motion), re-noise it, denoise, align,
take the Euler step. Run eagerly it is ~100 small launches around the denoiser and
one host synchronisation (the churn branch reads ``sigma_next`` on the host), so the
GPU idles between steps. Here the branch is a ``where`` on device scalars, the
alignment needs no SVD (``utils.rigid_align``), the random numbers come from
generators registered with the graph, and the denoiser runs inside the same capture.
Nothing in a step reads the device on the host, so one replay is one step.

The graph is built for ONE fold: it reads the fold's conditioning tensors directly
(they stand for the life of the sampler call), so a sampler call builds its own and
drops it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import contextlib
    from collections.abc import Callable
    from types import ModuleType

    from foldforge.diffusion.edm.sampling import EulerSampler

#: Eager steps run before the capture (kernels autotune, caches fill); their random
#:numbers are rolled back.
WARMUP_STEPS = 2


def _expand(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return value.reshape(*value.shape, *(1,) * (target.ndim - value.ndim))


def _hatted(config: EulerSampler.Config, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute hatted noise levels for the complete schedule.

    ``sigma_hat`` of every step of the schedule (the arithmetic of ``step``, over
    the whole vector).
    """
    sigma, sigma_next = sigmas[:-1], sigmas[1:]
    one = torch.ones((), dtype=sigmas.dtype, device=sigmas.device)
    if config.churn_gamma is None:
        on = sigma_next > config.gamma_min
        factor = torch.where(on, one * (1 + config.gamma_0), one)
    else:
        on = (sigma >= config.churn_sigma_min) & (sigma <= config.churn_sigma_max)
        factor = torch.where(on, one * (1 + config.churn_gamma), one)
    return sigma * factor


def _rigid_glue(
    sampler: EulerSampler, rigid_mask: torch.Tensor | None, coordinate_dims: int
) -> tuple[ModuleType, torch.Tensor, dict[str, Any]] | None:
    """Select fused sampler glue and its churn settings.

    The engine's fused rigid-motion glue (``sampler_ops``) and its churn settings,
    or None when it does not serve.

    It serves the dense layout's step: fp32 coordinates, one random rigid motion per
    sample before the noise, one Kabsch alignment after the denoiser, on an H100
    whose engine ships the kernels (an older engine, another GPU or a batched sigma
    keeps the torch ops).
    """
    config = sampler.config
    if rigid_mask is None or config.batched_sigma or coordinate_dims != 2:  # noqa: PLR2004 - dense token and atom axes
        return None
    try:
        from miniworld_engine.kernels.sampler_ops import (  # noqa: PLC0415 - optional backend
            interface,
        )
    except ImportError:
        return None
    mask = rigid_mask.reshape(-1).to(torch.float32).contiguous()
    window = config.churn_gamma is not None
    churn = {
        "window": window,
        "gamma": config.churn_gamma if window else config.gamma_0,
        "gamma_min": config.gamma_min,
        "sigma_min": config.churn_sigma_min,
        "sigma_max": config.churn_sigma_max,
        "noise_scale": config.noise_scale,
        "variance_floor": config.variance_floor,
        "clamp": bool(config.clamp_variance or config.variance_floor),
    }
    return interface, mask, churn


def build_step(
    sampler: EulerSampler,
    denoise: Callable,
    *,
    generator: torch.Generator | None,
    augment: Callable | None,
    align: Callable | None,
    coordinate_dims: int,
    rigid_mask: torch.Tensor | None = None,
) -> Callable:
    """Build a sampler step whose branches use device scalars.

    ``step(coords, sigma, sigma_next)``: ``EulerSampler.step`` with every branch on
    device scalars.

    The arithmetic is that of ``EulerSampler.prepare`` / ``update`` operation for
    operation: the churn factor ``1 + gamma`` is the same float32 constant the eager
    path multiplies by.

    ``rigid_mask`` (the atom mask) declares that ``augment`` and ``align`` are the
    dense layout's random rigid motion and masked Kabsch alignment; the engine then
    runs both, with the churn arithmetic and the Euler update, as six kernels
    (random numbers drawn in the same order from the same generator).
    """
    config = sampler.config
    glue = (
        _rigid_glue(sampler, rigid_mask, coordinate_dims)
        if augment is not None and align is not None
        else None
    )

    def fused_step(
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        row: torch.Tensor | None,
    ) -> torch.Tensor:
        interface, mask, churn = glue
        batch = coords.shape[0]
        flat = coords.reshape(batch, -1, 3)
        frame = torch.randn((batch, 2, 3), device=coords.device, dtype=coords.dtype)
        shift = torch.randn((batch, 3), device=coords.device, dtype=coords.dtype)
        noise = torch.randn(
            flat.shape, device=coords.device, dtype=coords.dtype, generator=generator
        )
        noisy, scalars = interface.augment_noise(
            flat, mask, frame, shift, noise, sigma, sigma_next, **churn
        )
        noisy = noisy.view(coords.shape)
        denoised = (
            denoise(noisy, scalars[0])
            if row is None
            else denoise(noisy, scalars[0], row)
        )
        flat_out = interface.align_euler(
            noisy.view(batch, -1, 3),
            denoised.reshape(batch, -1, 3).to(torch.float32),
            mask,
            scalars,
            sigma_next,
            config.step_scale,
        )
        return flat_out.view(coords.shape)

    def step(
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        row: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            glue is not None
            and coords.ndim == coordinate_dims + 2
            and glue[0].serves(coords.reshape(coords.shape[0], -1, 3), glue[1])
        ):
            return fused_step(coords, sigma, sigma_next, row)
        if augment is not None:
            coords = augment(coords)
        one = torch.ones((), dtype=sigma.dtype, device=sigma.device)
        if config.churn_gamma is None:
            on = sigma_next > config.gamma_min
            factor = torch.where(on, one * (1 + config.gamma_0), one)
        else:
            on = (sigma >= config.churn_sigma_min) & (sigma <= config.churn_sigma_max)
            factor = torch.where(on, one * (1 + config.churn_gamma), one)
        sigma_hat = sigma * factor
        variance = sigma_hat**2 - sigma**2
        if config.clamp_variance or config.variance_floor:
            variance = variance.clamp(min=config.variance_floor)
        magnitude = config.noise_scale * torch.sqrt(variance)
        noisy = coords + magnitude * torch.randn(
            coords.shape, device=coords.device, dtype=coords.dtype, generator=generator
        )
        if config.batched_sigma:
            batch = coords.shape[: -(coordinate_dims + 1)]
            sigma_hat = sigma_hat.reshape((1,) * len(batch)).expand(batch)
        denoised = (
            denoise(noisy, sigma_hat) if row is None else denoise(noisy, sigma_hat, row)
        )
        if align is not None:
            noisy = align(noisy, denoised)
        direction = (noisy - denoised) / _expand(sigma_hat, noisy)
        dt = _expand(sigma_next - sigma_hat, noisy)
        return noisy + config.step_scale * dt * direction

    return step


class _CapturedStep:
    def __init__(
        self,
        step: Callable,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        row: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None,
        scopes: Callable[[], contextlib.AbstractContextManager],
    ) -> None:
        self.scopes = scopes
        self.row = None if row is None else row.detach().clone()
        self.coords = coords.detach().clone()
        # (sigma, sigma_next) of the step in one buffer: one copy per replay;
        # the step's result is carried into ``coords`` inside the graph
        self.pair = torch.stack([sigma, sigma_next]).detach().clone()
        self.sigma, self.sigma_next = self.pair[0], self.pair[1]
        device = torch.cuda.current_device()
        with scopes():
            # The warmup consumes random numbers the real run must still
            # see: roll the streams back.
            default_state = torch.cuda.get_rng_state(device)
            generator_state = None if generator is None else generator.get_state()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(WARMUP_STEPS):
                    step(self.coords, self.sigma, self.sigma_next, self.row)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            torch.cuda.set_rng_state(default_state, device)
            if generator is not None:
                generator.set_state(generator_state)
            self.graph = torch.cuda.CUDAGraph()
            if generator is not None:
                self.graph.register_generator_state(generator)
            with torch.cuda.graph(self.graph):
                self.out = step(self.coords, self.sigma, self.sigma_next, self.row)
                self.coords.copy_(self.out)
            torch.cuda.synchronize()
        if hasattr(scopes, "captures"):
            scopes.captures += 1

    def __call__(self, pair: torch.Tensor, row: torch.Tensor | None = None) -> None:
        """Advance the iterate using the current sigma pair.

        One step: ``coords`` (the iterate this object carries) advance by the step
        at ``pair`` = (sigma, sigma_next).
        """
        if row is not None:
            self.row.copy_(row)
        self.pair.copy_(pair)
        self.graph.replay()
        if hasattr(self.scopes, "replays"):
            self.scopes.replays += 1


def sample(
    sampler: EulerSampler,
    denoise: Callable,
    shape: tuple[int, ...],
    sigmas: torch.Tensor,
    *,
    device: torch.device,
    generator: torch.Generator | None,
    augment: Callable | None,
    align: Callable | None,
    coordinate_dims: int,
    scopes: Callable[[], contextlib.AbstractContextManager],
    dtype: torch.dtype = torch.float32,
    rigid_mask: torch.Tensor | None = None,
    conditioning: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sample a trajectory with the complete step captured as a graph.

    ``EulerSampler.sample`` (one chunk, all samples drawn first) with the step in
    one graph.

    ``conditioning(sigma_hat)`` makes everything the denoiser needs that depends on
    the noise level and not on the iterate, for every step at once (``[steps,
    ...]``); the step then reads its row as ``denoise(coords, sigma_hat, row)``
    instead of making it per step.
    """
    step = build_step(
        sampler,
        denoise,
        generator=generator,
        augment=augment,
        align=align,
        coordinate_dims=coordinate_dims,
        rigid_mask=rigid_mask,
    )
    coords = sigmas[0] * torch.randn(
        shape, device=device, dtype=dtype, generator=generator
    )
    table = (
        None if conditioning is None else conditioning(_hatted(sampler.config, sigmas))
    )
    captured = None
    pairs = torch.stack([sigmas[:-1], sigmas[1:]], 1)
    for index in range(len(pairs)):
        row = None if table is None else table[index]
        if captured is None:
            captured = _CapturedStep(
                step,
                coords,
                sigmas[0],
                sigmas[1],
                row,
                generator=generator,
                scopes=scopes,
            )
        captured(pairs[index], row)
    return captured.coords.clone()
