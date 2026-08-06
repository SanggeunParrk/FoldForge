"""ESMFold2's reverse-diffusion solver.

:class:`~team_gm.diffusion.AF3Solver` covers the shape of AlphaFold-3
Algorithm 18 — churn, denoise, one Euler step — but ESMFold2 runs three
variations that the shared solver cannot express:

* the iterate is re-posed (centred, randomly rotated, randomly translated)
  before *every* denoiser call, not merely centred;
* the noisy iterate is Kabsch-aligned onto the denoised prediction before the
  step is taken;
* the schedule is clipped at a maximum sigma, and the walk starts from the cap.

Rather than widen the shared solver for one model, those live here. If a second
model turns out to need the same three, that is the moment to promote them.

Randomness is injected as a :class:`torch.Generator`. The shared solver seeds
the global RNG from its constructor and disables cuDNN; both are overridden,
since a model object should not reach out and change process-wide state.
"""

import torch
from pydantic import BaseModel
from team_gm.diffusion import AF3Solver, DiffusionScheduler
from team_gm.utils.transform import weighted_kabsch_align

from .diffusion_utils import DenoiseFn, expand_to_coords, random_rigid_motion


class ESMFold2Solver(AF3Solver):
    """AF3 Algorithm 18 with ESMFold2's per-step re-posing and alignment.

    Parameters
    ----------
    config : Config
        Solver settings, all drawn from the released model config rather than
        hardcoded as in the shared solver.
    scheduler : DiffusionScheduler
        Supplies the noise levels and the EDM preconditioning.

    """

    class Config(BaseModel):
        """Configuration for :class:`ESMFold2Solver`."""

        # Churn: re-noise to sigma * (1 + gamma) before denoising, but only
        # while there is enough noise left for it to help.
        gamma_0: float = 0.8
        gamma_min: float = 1.0
        noise_scale: float = 1.003
        step_scale: float = 1.5
        # Re-pose before each denoiser call (AF3 Algorithm 19).
        augment_per_step: bool = True
        translation_noise: float = 1.0
        # Kabsch-align the noisy iterate onto the denoised prediction.
        align_to_denoised: bool = True
        # Drop schedule entries above this level and start from the cap. The top
        # of the power-law schedule sits far above any real structure's scale,
        # so those steps mostly burn compute.
        max_sigma: float | None = 256.0

    def __init__(self, config: Config, scheduler: DiffusionScheduler) -> None:
        # AF3Solver.__init__ hardcodes the hyperparameters and seeds globally;
        # bypass it and wire the base class up directly.
        self.config = config
        self.scheduler = scheduler
        self.gamma_0 = config.gamma_0
        self.gamma_min = config.gamma_min
        self._lambda = config.noise_scale
        self.step_scale = config.step_scale
        self.center_per_step = False

    def _set_seed(self, seed: int) -> None:
        """Do nothing: randomness comes from an explicit generator.

        The shared implementation reseeds the global RNG and sets
        ``torch.backends.cudnn.enabled = False``, which would silently change
        every other computation in the caller's process.
        """

    def solver_sigmas(
        self, num_steps: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Build the schedule, clipped to ``max_sigma``.

        Clipping drops every level above the cap and prepends the cap itself, so
        the walk starts there and runs fewer steps than ``num_steps``.

        Parameters
        ----------
        num_steps : int
            Requested number of steps before clipping.
        device : torch.device or None
            Device for the returned tensor.

        Returns
        -------
        Tensor
            Descending noise levels, ending at zero.

        """
        sigmas = self.scheduler.sampling_time_steps(num_steps).to(device)
        cap = self.config.max_sigma
        if cap is None:
            return sigmas
        kept = sigmas[sigmas <= cap]
        return torch.cat([kept.new_full((1,), cap), kept])

    def denoise_step(
        self,
        denoise_fn: DenoiseFn,
        coords: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one step.

        Named apart from :meth:`AF3Solver.step` because the contract differs:
        the denoiser is handed the raw sigma rather than a conditioning value,
        and the noise levels are passed directly instead of being indexed out of
        a schedule.

        Parameters
        ----------
        denoise_fn : DenoiseFn
            Called with the scaled coordinates and the noise level.
        coords : Tensor
            Current iterate.
        sigma, sigma_next : Tensor
            Current and next noise levels, scalars.
        generator : torch.Generator or None
            Source of randomness for re-posing and churn.
        mask : Tensor or None
            Atom validity, used by re-posing and alignment.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(next_coords, denoised_coords)``.

        """
        config = self.config
        if config.augment_per_step:
            coords = random_rigid_motion(
                coords, mask, config.translation_noise, generator=generator
            )

        gamma = config.gamma_0 if float(sigma_next) > config.gamma_min else 0.0
        sigma_hat = sigma * (1.0 + gamma)
        churn = config.noise_scale * torch.sqrt(
            (sigma_hat**2 - sigma**2).clamp(min=0.0)
        )
        noisy = coords + churn * torch.randn(
            coords.shape, device=coords.device, dtype=coords.dtype, generator=generator
        )

        batch_sigma = sigma_hat.reshape(1).expand(noisy.shape[0])
        scale = expand_to_coords(self.scheduler.input_scale(batch_sigma), noisy)
        update = denoise_fn(noisy * scale, batch_sigma)
        skip = expand_to_coords(self.scheduler.skip_scale(batch_sigma), noisy)
        out = expand_to_coords(self.scheduler.output_scale(batch_sigma), noisy)
        denoised = skip * noisy + out * update

        if config.align_to_denoised:
            weights = (
                mask.to(denoised.dtype)
                if mask is not None
                else denoised.new_ones(denoised.shape[:-1])
            )
            noisy = weighted_kabsch_align(
                noisy.float(), denoised.float(), weights, None
            ).to(denoised.dtype)

        direction = (noisy - denoised) / sigma_hat
        return (
            noisy + config.step_scale * (sigma_next - sigma_hat) * direction,
            denoised,
        )

    def sample(
        self,
        denoise_fn: DenoiseFn,
        shape: torch.Size | tuple[int, ...],
        num_steps: int,
        device: torch.device | None = None,
        *,
        generator: torch.Generator | None = None,
        mask: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Integrate from pure noise down to a structure.

        Parameters
        ----------
        denoise_fn : DenoiseFn
            The denoiser to query at each step.
        shape : torch.Size or tuple[int, ...]
            Coordinate tensor shape, ending in ``3``.
        num_steps : int
            Solver steps before schedule clipping.
        device : torch.device or None
            Device to sample on.
        generator : torch.Generator or None
            Source of randomness; ``None`` uses the global RNG.
        mask : Tensor or None
            Atom validity.
        return_trajectory : bool
            Also return the iterate after every step.

        Returns
        -------
        Tensor or tuple[Tensor, list[Tensor]]
            Final coordinates, and the trajectory when requested.

        """
        sigmas = self.solver_sigmas(num_steps, device)
        coords = (
            torch.randn(tuple(shape), device=device, generator=generator) * sigmas[0]
        )
        trajectory: list[torch.Tensor] = []
        for index in range(len(sigmas) - 1):
            coords, _ = self.denoise_step(
                denoise_fn,
                coords,
                sigmas[index],
                sigmas[index + 1],
                generator=generator,
                mask=mask,
            )
            if return_trajectory:
                trajectory.append(coords.clone())
        return (coords, trajectory) if return_trajectory else coords
