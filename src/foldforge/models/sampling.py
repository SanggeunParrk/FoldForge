# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Checkpoint-facing argument adapters for team-gm's common diffusion runtime.

This module owns argument names only. Noise, augmentation, schedules, sample
chunking, integration and training corruption live in team_gm.diffusion.
"""

from __future__ import annotations

from functools import partial, wraps
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from team_gm.diffusion.augmentation import centre_random_augmentation
from team_gm.diffusion.edm.sampling import EulerSampler
from team_gm.diffusion.edm.training import sample_training

from foldforge.models import entry, is_dense
from foldforge.utils.seed import seed_context

if TYPE_CHECKING:
    from collections.abc import Callable


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor | None,
    p_lm: torch.Tensor | None,
    c_l: torch.Tensor | None,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,  # noqa: N803 - checkpoint-compatible keyword
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: int | None = None,
    *,
    inplace_safe: bool = False,
    attn_chunk_size: int | None = None,
    enable_efficient_fusion: bool = False,
    guidance_configs: Any = None,
    rollout_seed: int | None = None,
    guidance_factory: Callable | None = None,
    parse_guidance: Callable | None = None,
) -> torch.Tensor:
    """Adapt released argument layouts to one clean-coordinate denoiser."""
    device, dtype = (s_inputs.device, s_inputs.dtype)
    generator = (
        None
        if rollout_seed is None
        else torch.Generator(device=device).manual_seed(int(rollout_seed))
    )
    numpy_rng = (
        None if rollout_seed is None else np.random.default_rng(int(rollout_seed))
    )
    context = {
        "input_feature_dict": input_feature_dict,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "pair_z": pair_z,
        "p_lm": p_lm,
        "c_l": c_l,
        "chunk_size": attn_chunk_size,
        "inplace_safe": inplace_safe,
        "enable_efficient_fusion": enable_efficient_fusion,
    }

    def augment(coords: torch.Tensor) -> torch.Tensor:
        """Compute augment."""
        return (
            centre_random_augmentation(
                coords, N_sample=1, torch_generator=generator, numpy_rng=numpy_rng
            )
            .squeeze(-3)
            .to(dtype)
        )

    def denoise(coords: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute denoise."""
        return denoise_net(x_noisy=coords, t_hat_noise_level=sigma, **context)

    if parse_guidance is None:
        from team_gm.diffusion.guidance.config import parse_tfg_config

        parse_guidance = parse_tfg_config
    if guidance_factory is None:
        from team_gm.diffusion.guidance.compat import TFGEngine

        guidance_factory = TFGEngine
    guidance: (
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int], torch.Tensor]
        | None
    ) = None
    if parse_guidance is not None:
        config = parse_guidance(guidance_configs)
        if config.enable:
            engine = guidance_factory(config, device=device, dtype=dtype)
            extras = {"torch_generator": generator}

            def apply_guidance(
                coords: torch.Tensor,
                sigma: torch.Tensor,
                sigma_next: torch.Tensor,
                step_index: int,
                num_steps: int,
            ) -> torch.Tensor:
                return engine.step(
                    denoise_net,
                    x=coords,
                    t_hat=sigma,
                    c_tau=sigma_next,
                    step_i=step_index,
                    num_diffusion_steps=num_steps,
                    step_scale_eta=step_scale_eta,
                    **context,
                    **extras,
                )

            guidance = apply_guidance

    sampler = EulerSampler(
        EulerSampler.Config(
            gamma_0=gamma0,
            gamma_min=gamma_min,
            noise_scale=noise_scale_lambda,
            step_scale=step_scale_eta,
            batched_sigma=True,
        )
    )
    shape = (
        *s_inputs.shape[:-2],
        N_sample,
        input_feature_dict["atom_to_token_idx"].size(-1),
        3,
    )
    result = sampler.sample(
        denoise,
        shape,
        noise_schedule,
        device=device,
        dtype=dtype,
        generator=generator,
        chunk_size=diffusion_chunk_size,
        augment=augment,
        sigma_dtype=dtype,
        guidance=guidance,
    )
    if not isinstance(result, torch.Tensor):
        message = "Sampling without trajectories must return a Tensor"
        raise TypeError(message)
    return result


def sample_diffusion_training(
    noise_sampler: Callable,
    denoise_net: Callable,
    label_dict: dict[str, torch.Tensor],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor | None,
    p_lm: torch.Tensor | None,
    c_l: torch.Tensor | None,
    N_sample: int = 1,  # noqa: N803 - checkpoint-compatible keyword
    diffusion_chunk_size: int | None = None,
    *,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt training model arguments; corruption and chunking are shared."""
    network = partial(
        denoise_net,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        use_conditioning=use_conditioning,
        enable_efficient_fusion=enable_efficient_fusion,
    )

    def denoise(coords: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute denoise."""
        return network(x_noisy=coords, t_hat_noise_level=sigma)

    return sample_training(
        denoise,
        label_dict["coordinate"],
        label_dict["coordinate_mask"],
        noise_sampler,
        num_samples=N_sample,
        chunk_size=diffusion_chunk_size,
    )


def bind_sampling_seed(model: torch.nn.Module, family: str, seed: int) -> None:
    """Isolate the complete diffusion trajectory from trunk and confidence RNGs.

    This host boundary encloses initial noise, augmentation, churn and guidance.
    The denoiser's compiled/CUDA-graph boundary stays inside it.
    """
    sequence = entry(family).layout == "sequence_atoms"
    if sequence:
        owner, method = model.get_submodule("structure_head"), "sample"
    else:
        owner, method = model, "_sample_diffusion"
    original = getattr(owner, method)
    device = next(model.parameters()).device

    @torch.compiler.disable
    @wraps(original)
    def sample(*args: Any, **kwargs: Any) -> Any:
        with seed_context(seed):
            if sequence:
                # Do not consume the explicit generator already used by the trunk.
                kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
            elif not is_dense(family):
                # The flat graph carries an optional seed in its feature
                # dictionary; override it. Asked by LAYOUT rather than by name,
                # because the same family now has a dense graph that takes its
                # seed from the context above and has no such argument.
                kwargs["rollout_seed"] = seed
            return original(*args, **kwargs)

    setattr(owner, method, sample)
