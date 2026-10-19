# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TFG (Training-Free Guidance) engine.

This module implements the core *runtime* logic for applying TFG potentials
inside the diffusion sampler.

High-level idea
---------------
- A set of guidance "terms" (potentials) define an energy function E(x).
- We treat p(x) ~ exp(-E(x)) as a soft constraint/conditioning signal.
- During sampling, we:
  1) optionally guide x_t via gradients that flow through the denoiser
     ("denoiser-path guidance" controlled by `rho`),
  2) optionally refine the denoiser's x0 prediction directly (controlled by
     `mu`), and
  3) optionally project x0 onto constraint manifolds (projection steps).
"""

import math
from collections.abc import Callable, Mapping
from logging import getLogger
from typing import Any

import torch

from foldforge.guidance.config import TFGConfig, validate_features
from foldforge.guidance.potentials import Potential

logger = getLogger(__name__)


def _sample_eps(
    std: float,
    shape: torch.Size,
    *,
    k: int,
    device,
    dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Gaussian perturbations used for Monte-Carlo log-prob estimation.

    Args:
        std: Standard deviation of the noise. If 0, returns all zeros.
        shape: Base tensor shape to perturb (typically `x0.shape`).
        k: Number of Monte-Carlo samples.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor of shape `[k, *shape]`.

    """
    if std == 0.0:
        return torch.zeros((1, *shape), device=device, dtype=dtype)
    return std * torch.randn(
        (k, *shape), device=device, dtype=dtype, generator=generator
    )


def _logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable `log(mean(exp(x)))`.

    This is commonly used when aggregating Monte-Carlo samples in log-space.
    """
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


class GuidanceEngine:
    """TFG guidance engine.

    This class evaluates configured TFG terms and integrates them into the
    diffusion sampling step.
    """

    Config = TFGConfig

    def __init__(self, cfg: TFGConfig, *, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self._projection_terms_sorted = sorted(
            self.cfg.terms,
            key=lambda x: (
                0
                if x.name == "ChiralAtomPotential"
                else 1
                if x.name == "PairwiseDistancePotential"
                else 2
            ),
        )

    def _energy_and_grad(
        self, coords: torch.Tensor, feats: Mapping[str, Any], *, t: float, step_i: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the total energy and gradient for a coordinate tensor.

        The energy is defined by summing all *active* terms in `self.cfg.terms`.

        Args:
            coords: Atomic coordinates, shape `[*batch, N_atom, 3]`.
            feats: Feature dict used by terms (e.g. atom metadata, bonds,
                restraints, templates). This is validated once in `step()`.
            t: Diffusion "time" in [0, 1], where larger means noisier.
            step_i: 0-based diffusion step index.

        Returns:
            (energy, grad)
            - energy: shape `[*batch]`
            - grad: shape `[*batch, N_atom, 3]`

        """
        energy = torch.zeros(
            coords.shape[:-2], device=coords.device, dtype=coords.dtype
        )
        grad = torch.zeros_like(coords)
        for term in self.cfg.terms:
            if not term.active(step_i):
                continue
            e, g = term.energy_and_grad(coords, feats, t)
            energy = energy + e
            grad = grad + g
        return (energy, grad)

    def _energy(
        self, coords: torch.Tensor, feats: Mapping[str, Any], *, t: float, step_i: int
    ) -> torch.Tensor:
        """Compute total energy only (no explicit analytic gradients).

        This is useful for denoiser-path guidance where we rely on autograd to
        backprop through the denoiser and only need a scalar objective.
        """
        energy = torch.zeros(
            coords.shape[:-2], device=coords.device, dtype=coords.dtype
        )
        for term in self.cfg.terms:
            if not term.active(step_i):
                continue
            energy = energy + term.energy(coords, feats, t)
        return energy

    def _logp_x0(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        feats: Mapping[str, Any],
        *,
        t: float,
        step_i: int,
    ) -> torch.Tensor:
        """Estimate `log p(x0)` via Monte-Carlo samples (energy-only path)."""
        k = eps.shape[0]
        x0_eps = (x0.unsqueeze(0) + eps).reshape(-1, *x0.shape[-2:])
        e = self._energy(x0_eps, feats, t=t, step_i=step_i)
        e = e.reshape(k, *x0.shape[:-2])
        logp = -e
        if k == 1:
            return logp.squeeze(0)
        return _logmeanexp(logp, dim=0)

    def _logp_and_grad_x0(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        feats: Mapping[str, Any],
        *,
        t: float,
        step_i: int,
        log_components: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate `log p(x0)` and `d/dx0 log p(x0)` via Monte-Carlo samples.

        We interpret the configured TFG terms as an unnormalized density:

            p(x0) ~ exp(-E(x0))

        and compute a Monte-Carlo approximation of `log p(x0)` under additive
        perturbations `eps` (reparameterization used for stability / smoothing).

        Args:
            x0: Coordinates to evaluate, shape `[*batch, N_atom, 3]`.
            eps: Noise samples, shape `[K, *batch, N_atom, 3]`.
            feats: Feature dict consumed by the energy terms.
            t: Diffusion "time" in [0, 1].
            step_i: 0-based diffusion step index.
            log_components: If True, logs per-term energies for debugging at the
                final iteration.

        Returns:
            (avg_logp, grad_x0)
            - avg_logp: shape `[*batch]`, approximating log-mean-exp over K.
            - grad_x0: shape `[*batch, N_atom, 3]`.

        """
        k = eps.shape[0]
        x0_eps = (x0.unsqueeze(0) + eps).reshape(-1, *x0.shape[-2:])
        e, g = self._energy_and_grad(x0_eps, feats, t=t, step_i=step_i)
        e = e.reshape(k, *x0.shape[:-2])
        g = g.reshape(k, *x0.shape)
        logp = -e
        if k == 1:
            avg_logp = logp.squeeze(0)
            grad = (-g).squeeze(0)
        else:
            avg_logp = _logmeanexp(logp, dim=0)
            w = torch.softmax(logp, dim=0)
            dlogp = -g
            grad = (w.unsqueeze(-1).unsqueeze(-1) * dlogp).sum(dim=0)
        if log_components:
            with torch.no_grad():
                for term in self.cfg.terms:
                    if not term.active(step_i):
                        continue
                    e_t = term.energy(x0, feats, t)
                    logger.info(
                        "TFG last-step energy %s: %s",
                        term.name,
                        e_t.detach().cpu().tolist(),
                    )
        return (avg_logp, grad)

    def _project(
        self, coords: torch.Tensor, feats: Mapping[str, Any], *, t: float, step_i: int
    ) -> torch.Tensor:
        """Run iterative projection steps for projection-enabled terms.

        Projection is treated as an additive coordinate update `delta`.
        Each projection-enabled term may propose a small correction, and we
        accumulate these corrections for `projection_steps` iterations.

        Args:
            coords: Coordinates to project, shape `[*batch, N_atom, 3]`.
            feats: Feature dict consumed by projection terms.
            t: Diffusion "time" in [0, 1].
            step_i: 0-based diffusion step index.

        Returns:
            A delta tensor with the same shape as `coords`.

        """
        if (
            not self.cfg.enable
            or self.cfg.projection_outer_steps <= 0
            or self.cfg.projection_inner_steps <= 0
        ):
            return torch.zeros_like(coords)
        delta = torch.zeros_like(coords)
        sorted_terms = [
            term
            for term in self._projection_terms_sorted
            if type(term._potential)._project is not Potential._project
        ]
        for _ in range(self.cfg.projection_outer_steps):
            for term in sorted_terms:
                if not term.active(step_i) or not term.enable_projection:
                    continue
                for _ in range(self.cfg.projection_inner_steps):
                    d = term.project(coords + delta, feats, t)
                    if d is not None:
                        delta = delta + d
        return delta

    def project(
        self,
        coords: torch.Tensor,
        feats: Mapping[str, Any],
        *,
        step_i: int,
        num_diffusion_steps: int,
    ) -> torch.Tensor:
        """Public projection hook.

        This is used when the caller wants to apply projections without running
        the full TFG-aware diffusion update.

        Args:
            coords: Coordinates, shape `[*batch, N_atom, 3]`.
            feats: Feature dict.
            step_i: 0-based diffusion step index.
            num_diffusion_steps: Total number of diffusion steps.

        Returns:
            Projection delta with the same shape as `coords`.

        """
        t = 1.0 - float(step_i) / float(max(1, num_diffusion_steps))
        return self._project(coords, feats, t=t, step_i=step_i)

    def step(
        self,
        denoise_net: Callable,
        *,
        x: torch.Tensor,
        t_hat: torch.Tensor,
        c_tau: torch.Tensor,
        step_scale_eta: float,
        step_i: int,
        num_diffusion_steps: int,
        input_feature_dict: Mapping[str, Any],
        torch_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run one TFG-aware diffusion update: `x_t -> x_{t-1}`.

        The update combines:
        - denoiser-path guidance (controlled by `cfg.rho`),
        - direct refinement on the predicted x0 (controlled by `cfg.mu`),
        - optional constraint projection (controlled by `cfg.projection_steps`),
        - and a predictor-corrector style diffusion update with stochasticity.

        Args:
            denoise_net: Callable denoiser that maps `(x_t, t_hat, feats, ...)`
                to a predicted `x0`.
            x: Current noisy coordinates `x_t`, shape `[*batch, N_atom, 3]`.
            t_hat: Current noise level (scalar per batch), shape `[*batch]`.
            c_tau: Next noise level (scalar per batch), shape `[*batch]`.
            step_scale_eta: Global step size used by the sampler.
            step_i: 0-based diffusion step index.
            num_diffusion_steps: Total number of diffusion steps.
            input_feature_dict: Feature dict shared by denoiser and TFG terms.
            denoise_net: Callable accepting only coordinates and noise levels;
                checkpoint-specific conditioning is bound by the caller.
            torch_generator: Optional RNG used for reproducible stochastic guidance.

        Returns:
            Updated noisy coordinates `x_{t-1}` with shape `[*batch, N_atom, 3]`.

        """

        def _prepare_guidance_step():
            if step_i == 0:
                validate_features(input_feature_dict, self.cfg.terms)
            prepared_t = 1.0 - float(step_i) / float(max(1, num_diffusion_steps))
            prepared_eps = _sample_eps(
                self.cfg.eps_std,
                x.shape,
                k=self.cfg.eps_batch,
                device=self.device,
                dtype=self.dtype,
                generator=torch_generator,
            )
            return (prepared_t, prepared_eps)

        t, eps = _prepare_guidance_step()
        x_work = x
        for outer in range(self.cfg.outer_steps):

            def _prepare_guidance_shift(x_work=x_work):
                if self.cfg.rho != 0.0:
                    with torch.enable_grad():
                        x_var = x_work.detach().requires_grad_(requires_grad=True)
                        x0_pred = denoise_net(x_var, t_hat)
                        avg_logp = self._logp_x0(
                            x0_pred, eps, input_feature_dict, t=t, step_i=step_i
                        )
                        grad_xt = torch.autograd.grad(
                            avg_logp.sum(),
                            x_var,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                        if grad_xt is None:
                            grad_xt = torch.zeros_like(x_var)
                    return grad_xt.detach() * float(self.cfg.rho)
                return torch.zeros_like(x_work)

            xt_shift = _prepare_guidance_shift()
            shifted_x_work = x_work + xt_shift
            with torch.no_grad():
                x0 = denoise_net(shifted_x_work, t_hat)

            def _complete_guidance_outer(
                x_work=x_work, xt_shift=xt_shift, x0=x0, outer=outer
            ):
                x0_ref = x0.detach()
                x0_ref = x0_ref + self._project(
                    x0_ref, input_feature_dict, t=t, step_i=step_i
                )
                for inner in range(self.cfg.inner_steps):
                    if self.cfg.mu == 0.0:
                        break
                    log_components = (
                        self.cfg.log_last_step_energy
                        and step_i == num_diffusion_steps - 1
                        and (outer == self.cfg.outer_steps - 1)
                        and (inner == self.cfg.inner_steps - 1)
                    )
                    _, grad_x0 = self._logp_and_grad_x0(
                        x0_ref,
                        eps,
                        input_feature_dict,
                        t=t,
                        step_i=step_i,
                        log_components=log_components,
                    )
                    x0_ref = x0_ref + grad_x0 * float(self.cfg.mu)
                direction = (x_work + xt_shift - x0_ref) / t_hat[..., None, None]
                dt = c_tau - t_hat
                x_next_local = (
                    x_work
                    + xt_shift
                    + float(step_scale_eta) * dt[..., None, None] * direction
                )
                sigma = torch.sqrt(t_hat**2 - c_tau**2)
                next_x_work = x_next_local + sigma[..., None, None] * torch.randn(
                    size=x_next_local.shape,
                    device=x_next_local.device,
                    dtype=x_next_local.dtype,
                    generator=torch_generator,
                )
                return (x_next_local, next_x_work)

            x_next, x_work = _complete_guidance_outer()
        return x_next
