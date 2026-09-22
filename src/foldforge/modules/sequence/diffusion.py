"""ESMFold2 diffusion denoiser: conditioning, atom decoder, one denoising step.

The token-level denoiser is team-gm's existing
:class:`~team_gm.modules.blocks.DiffusionTransformer` — adaptive LayerNorm,
gated pair-biased attention and a conditioned SwiGLU transition all match the
reference block for block, so only the weight packing differs. The noise
conditioning reuses :class:`~team_gm.modules.layers.Transition`, and the atom
tracks reuse :class:`~team_gm.models.esmfold2.AtomEncoder`.

What is new here is the diffusion plumbing: the Fourier noise embedding, the
skip-connected atom decoder, and the EDM-style preconditioning that turns the
network's coordinate update into a denoised structure.

:class:`DiffusionModule` is one denoising step; :class:`DiffusionStructureHead`
wraps it with the shared solver from :mod:`team_gm.diffusion` to sample.

Sampling runs the denoiser a few hundred times over inputs that mostly do not
change. :class:`DenoiseCache` holds the parts that are provably independent of
the iterate and the noise level, so they are built once per structure instead of
once per step. It is an exact hoist, not an approximation: see the individual
docstrings for why each piece qualifies.
"""

from typing import NamedTuple, NotRequired, TypedDict, Unpack

import torch
from jaxtyping import Bool, Float, Int
from miniworld_engine.modules import Transition
from team_gm import typecheck
from team_gm.diffusion import EDMScheduler
from team_gm.diffusion.edm.preconditioned import (
    PreconditionedEulerSolver as ESMFold2Solver,
)
from team_gm.diffusion.edm.preconditioning import (
    reconstruct_coordinates,
    scale_coordinates,
)
from team_gm.modules.blocks import DiffusionTransformer
from team_gm.modules.blocks._engine_impl import to_engine_impl
from team_gm.modules.blocks.fourier_math import fourier_features
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.primitives import LayerNorm, Linear
from torch import nn

from foldforge.models.config.sequence import ESMFold2Config
from foldforge.modules.sequence.atom_encoder import XYZ_DIMS, AtomEncoder, AtomStatics
from foldforge.modules.sequence.atom_transformer import SWAAtomTransformer
from foldforge.modules.sequence.tokens import gather_token_to_atom


class DenoiseCache(NamedTuple):
    """Per-structure work hoisted out of the sampling loop.

    Every field is a function of the trunk output and the reference conformer
    only — never of the coordinates being denoised or the noise level — so
    reusing it across steps changes nothing about the result.

    Attributes
    ----------
    conditioned_pair : Tensor
        ``[B, L, L, c_z]``; two pair transitions, by far the largest saving.
    projected_single : Tensor
        ``[BS, L, c_token]`` token inputs before the noise embedding is added.
    atom_statics : AtomStatics
        Atom conditioning and the 3D RoPE / sliding-window attention params.
    """

    conditioned_pair: torch.Tensor
    projected_single: torch.Tensor
    atom_statics: AtomStatics


class FourierEmbedding(nn.Module):
    """Random-feature Fourier embedding of a scalar, ``cos(2*pi*(t*w + b))``.

    ``w`` and ``b`` are frozen buffers drawn once at construction, so the
    embedding is a fixed random projection rather than something learned.

    Parameters
    ----------
    dim : int
        Number of Fourier features.
    """

    w: torch.Tensor
    b: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("w", torch.randn(dim))
        self.register_buffer("b", torch.randn(dim))

    @typecheck
    def forward(self, t: torch.Tensor) -> Float[torch.Tensor, "BS dim"]:
        """Compute the module output."""
        return fourier_features(t, self.w, self.b, cast_input=True)


class DiffusionConditioning(nn.Module):
    """Condition the pair and single tracks on the current noise level.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the transitions.
    """

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        dm = config.structure_head.diffusion_module
        self.sigma_data = dm.sigma_data
        n = dm.transition_multiplier

        self.ln_pair_in = LayerNorm(2 * dm.c_z, implementation=implementation)
        self.pair_proj = Linear(2 * dm.c_z, dm.c_z, bias=False, init="default")
        self.pair_transitions = nn.ModuleList(
            [
                Transition(dm.c_z, n=n, implementation=to_engine_impl(implementation))
                for _ in range(2)
            ]
        )

        self.ln_single_in = LayerNorm(dm.c_s_inputs, implementation=implementation)
        self.single_proj = Linear(dm.c_s_inputs, dm.c_token, bias=False, init="default")
        self.fourier = FourierEmbedding(dm.fourier_dim)
        self.ln_noise = LayerNorm(dm.fourier_dim, implementation=implementation)
        self.noise_proj = Linear(dm.fourier_dim, dm.c_token, bias=False, init="default")
        self.single_transitions = nn.ModuleList(
            [
                Transition(
                    dm.c_token, n=n, implementation=to_engine_impl(implementation)
                )
                for _ in range(2)
            ]
        )

    @typecheck
    def condition_pair(
        self,
        pair: Float[torch.Tensor, "B L L c_z"],
        relative_position_encoding: Float[torch.Tensor, "B L L c_z"],
    ) -> Float[torch.Tensor, "B L L c_z"]:
        """Fold the relative position encoding into the trunk pair track.

        Reads no noise level, so a sampler can call this once and reuse the
        result for every solver step. This is the single largest saving
        available: two ``[B, L, L, c_z]`` transitions per step otherwise.
        """
        # Concatenate in fp32 (the trunk hands these over at mixed precision),
        # then enter the projection in its own dtype.
        weight_dtype = self.pair_proj.weight.dtype
        conditioned = self.pair_proj(
            self.ln_pair_in(
                torch.cat(
                    [pair.float(), relative_position_encoding.float()], dim=-1
                ).to(weight_dtype)
            )
        )
        # No `conditioned +` here: the engine's Transition owns its self-residual
        # (ARCHITECTURE.md rule 2). Adding it again tripled the conditioned pair
        # (|x| 15.1 -> 46.1), over-compacted the structure (Rg 15.9 -> 14.9 A on
        # 3PTB) and dropped pLDDT 0.97 -> 0.75 — with nothing raised anywhere.
        for block in self.pair_transitions:
            conditioned = block(conditioned)
        return conditioned

    @typecheck
    def project_single(
        self, single_inputs: Float[torch.Tensor, "BS L d_inputs"]
    ) -> Float[torch.Tensor, "BS L c_token"]:
        """Project the token inputs, before the noise embedding is added.

        Also noise-free; the transitions after it are not, because they see the
        summed noise embedding.
        """
        return self.single_proj(
            self.ln_single_in(single_inputs.to(self.single_proj.weight.dtype))
        )

    @typecheck
    def forward(
        self,
        noise_level: torch.Tensor,
        single_inputs: Float[torch.Tensor, "BS L d_inputs"],
        pair: Float[torch.Tensor, "B L L c_z"],
        relative_position_encoding: Float[torch.Tensor, "B L L c_z"],
        conditioned_pair: Float[torch.Tensor, "B L L c_z"] | None = None,
        projected_single: Float[torch.Tensor, "BS L c_token"] | None = None,
    ) -> tuple[Float[torch.Tensor, "BS L c_token"], Float[torch.Tensor, "B L L c_z"]]:
        """Forward pass.

        Parameters
        ----------
        noise_level : Tensor
            ``[BS]`` current sigma per sample.
        single_inputs : Tensor
            Token-level input features, already expanded to the sample axis.
        pair : Tensor
            Trunk pair representation.
        relative_position_encoding : Tensor
            Relative position encoding reused from the trunk.
        conditioned_pair, projected_single : Tensor or None
            Precomputed :meth:`condition_pair` / :meth:`project_single` output.
            Pass them to skip that work; ``None`` computes it here.

        Returns
        -------
        tuple[Tensor, Tensor]
            Conditioned single and pair tracks. The pair track is independent of
            the noise level, so a sampler can compute it once per structure.
        """
        if conditioned_pair is None:
            conditioned_pair = self.condition_pair(pair, relative_position_encoding)
        single = (
            self.project_single(single_inputs)
            if projected_single is None
            else projected_single
        )

        scaled = 0.25 * torch.log((noise_level / self.sigma_data).clamp(min=1e-20))
        # One noise embedding per sample: retain the explicit singleton token axis
        # so the engine can form a semantic (batch, length) autotune key.
        noise = self.noise_proj(self.ln_noise(self.fourier(scaled).unsqueeze(1)))
        single = single + noise
        # Same contract as the pair transitions above: the op owns the residual.
        for block in self.single_transitions:
            single = block(single)
        return single, conditioned_pair


class AtomDecoder(nn.Module):
    """Turn token features back into a per-atom coordinate update.

    Runs on the encoder's atom features and conditioning, so the encoder's
    sliding-window attention parameters are reused rather than rebuilt.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the atom track.
    """

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        atom = config.inputs.atom_encoder
        dm = config.structure_head.diffusion_module

        self.token_to_atom_linear = Linear(
            dm.c_token, dm.c_atom, bias=False, init="default"
        )
        self.atom_transformer = SWAAtomTransformer(
            SWAAtomTransformer.Config(
                d_atom=dm.c_atom,
                d_cond=dm.c_atom,
                n_block=dm.atom_num_blocks,
                n_head=dm.atom_num_heads,
                swa_window_size=atom.swa_window_size,
                expansion_ratio=atom.expansion_ratio,
                n_spatial_rope_pairs_per_axis=atom.n_spatial_rope_pairs_per_axis,
                spatial_rope_base_frequency=atom.spatial_rope_base_frequency,
                n_uid_rope_pairs=atom.n_uid_rope_pairs,
                uid_rope_base_frequency=atom.uid_rope_base_frequency,
                implementation=implementation,
            )
        )
        self.norm = LayerNorm(dm.c_atom, implementation=implementation)
        self.output_linear = Linear(dm.c_atom, XYZ_DIMS, bias=False, init="default")

    @typecheck
    def forward(
        self,
        token_features: Float[torch.Tensor, "BS L c_token"],
        atom_features: Float[torch.Tensor, "BS A c_atom"],
        atom_conditioning: Float[torch.Tensor, "BS A c_atom"],
        attention_params: tuple,
        atom_to_token: Int[torch.Tensor, "BS A"],
    ) -> Float[torch.Tensor, "BS A 3"]:
        """Forward pass."""
        broadcast = gather_token_to_atom(
            self.token_to_atom_linear(token_features), atom_to_token
        )
        atoms = self.atom_transformer(
            atom_features + broadcast, atom_conditioning, attention_params
        )
        return self.output_linear(self.norm(atoms))


class DenoiseInputs(TypedDict):
    """Static features forwarded to the coordinate denoiser."""

    single_inputs: torch.Tensor
    pair: torch.Tensor
    relative_position_encoding: torch.Tensor
    ref_pos: torch.Tensor
    ref_charge: torch.Tensor
    ref_element: torch.Tensor
    ref_atom_name_chars: torch.Tensor
    ref_space_uid: torch.Tensor
    atom_mask: torch.Tensor
    atom_to_token: torch.Tensor
    mask: torch.Tensor
    num_diffusion_samples: NotRequired[int]
    cache: NotRequired[DenoiseCache | None]


class DiffusionModule(nn.Module):
    """One denoising step: noisy coordinates in, denoised coordinates out.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the shared layers.
    """

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.config = config
        dm = config.structure_head.diffusion_module
        self.sigma_data = dm.sigma_data

        self.conditioning = DiffusionConditioning(config, implementation)
        self.atom_encoder = AtomEncoder(
            config.inputs.atom_encoder,
            d_token_out=dm.c_token,
            structure_prediction=True,
            implementation=implementation,
        )
        self.atom_decoder = AtomDecoder(config, implementation)
        self.single_to_token = Linear(dm.c_token, dm.c_token, bias=False, init="zero")
        self.token_transformer = DiffusionTransformer(
            DiffusionTransformer.Config(
                d_single=dm.c_token,
                d_cond=dm.c_token,
                d_pair=dm.c_z,
                n_head=dm.token_num_heads,
                n_block=dm.token_num_blocks,
                implementation=implementation,
            )
        )
        self.ln_single_step = LayerNorm(dm.c_token, implementation=implementation)
        self.ln_token = LayerNorm(dm.c_token, implementation=implementation)

    def build_cache(
        self,
        single_inputs: torch.Tensor,
        pair: torch.Tensor,
        relative_position_encoding: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
        num_diffusion_samples: int = 1,
    ) -> DenoiseCache:
        """Precompute everything :meth:`denoise` would otherwise redo each step.

        Takes the same tensors as :meth:`denoise` minus the iterate and the
        noise level — which is exactly the point: nothing here can depend on
        them.

        Parameters
        ----------
        single_inputs, pair, relative_position_encoding : Tensor
            Trunk outputs, one row per structure.
        ref_pos, ref_charge, ref_element, ref_atom_name_chars, ref_space_uid : Tensor
            Reference-conformer atom features.
        atom_mask : Tensor
            Atom validity.
        num_diffusion_samples : int
            Samples per structure. Single features retain the sampler's B-major
            order; atom features use sample-major order. Atom reference features
            and RoPE are computed once per structure before expansion.

        Returns
        -------
        DenoiseCache
            Pass to :meth:`denoise` as ``cache=``.
        """
        samples = num_diffusion_samples

        def expand(x: torch.Tensor) -> torch.Tensor:
            """Expand ."""
            return x if samples == 1 else x.repeat_interleave(samples, 0)

        return DenoiseCache(
            conditioned_pair=self.conditioning.condition_pair(
                pair, relative_position_encoding
            ),
            projected_single=self.conditioning.project_single(expand(single_inputs)),
            atom_statics=self.atom_encoder.static_features(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                atom_mask=atom_mask,
                num_aug=samples,
            ),
        )

    @typecheck
    def denoise(
        self,
        scaled_coords: Float[torch.Tensor, "BS A 3"],
        noise_level: torch.Tensor,
        single_inputs: Float[torch.Tensor, "B L d_inputs"],
        pair: Float[torch.Tensor, "B L L c_z"],
        relative_position_encoding: Float[torch.Tensor, "B L L c_z"],
        ref_pos: Float[torch.Tensor, "B A 3"],
        ref_charge: Float[torch.Tensor, "B A"],
        ref_element: Float[torch.Tensor, "B A 128"],
        ref_atom_name_chars: Float[torch.Tensor, "B A 4 64"],
        ref_space_uid: Int[torch.Tensor, "B A"],
        atom_mask: Bool[torch.Tensor, "B A"],
        atom_to_token: Int[torch.Tensor, "B A"],
        mask: Bool[torch.Tensor, "B L"],
        num_diffusion_samples: int = 1,
        cache: DenoiseCache | None = None,
    ) -> Float[torch.Tensor, "BS A 3"]:
        """Run the denoiser network and return its raw coordinate update.

        Conforms to :data:`~team_gm.diffusion.DenoiseFn`, so a solver can drive
        it directly and apply the EDM preconditioning itself.

        Parameters
        ----------
        scaled_coords : Tensor
            Coordinates already divided by ``sqrt(sigma^2 + sigma_data^2)``.
        noise_level : Tensor
            ``[BS]`` current sigma per sample.
        single_inputs, pair, relative_position_encoding : Tensor
            Trunk outputs, one row per structure.
        ref_pos, ref_charge, ref_element, ref_atom_name_chars, ref_space_uid : Tensor
            Reference-conformer atom features.
        atom_mask, atom_to_token, mask : Tensor
            Atom validity, atom-to-token map and token validity.
        num_diffusion_samples : int
            Samples per structure.
        cache : DenoiseCache or None
            Output of :meth:`build_cache` for these same inputs. Reuses the
            iterate-independent work instead of redoing it; ``None`` redoes it.

        Returns
        -------
        Tensor
            Raw per-atom coordinate update.
        """
        samples = num_diffusion_samples
        n_tokens = mask.shape[1]

        def expand(x: torch.Tensor) -> torch.Tensor:
            """Expand ."""
            return x if samples == 1 else x.repeat_interleave(samples, 0)

        single_inputs_s = expand(single_inputs)
        single, conditioned_pair = self.conditioning(
            noise_level,
            single_inputs_s,
            pair,
            relative_position_encoding,
            conditioned_pair=None if cache is None else cache.conditioned_pair,
            projected_single=None if cache is None else cache.projected_single,
        )

        # The sampler flattens [B, samples]; atom SWA and token attention use
        # [samples, B]. Build shared reference features once per structure.
        batch = pair.shape[0]

        def to_sample_axis(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch, samples, *x.shape[1:]).transpose(0, 1)

        atom_to_token_s = atom_to_token.repeat(samples, 1)
        token_features, atom_features, atom_conditioning, attention_params = (
            self.atom_encoder(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                atom_mask=atom_mask,
                atom_to_token=atom_to_token,
                n_tokens=n_tokens,
                coords=to_sample_axis(scaled_coords).flatten(0, 1),
                statics=None if cache is None else cache.atom_statics,
                num_aug=samples,
            )
        )
        single_aug = to_sample_axis(single)
        token_features = token_features.reshape(samples, batch, n_tokens, -1)
        token_features = token_features + self.single_to_token(
            self.ln_single_step(single_aug)
        )
        denoised_tokens = self.token_transformer(
            token_features, single_aug, conditioned_pair, mask
        )
        token_features = self.ln_token(denoised_tokens.flatten(0, 1))
        update = self.atom_decoder(
            token_features,
            atom_features,
            atom_conditioning,
            attention_params,
            atom_to_token_s,
        )
        # Restore B-major order at the shared solver boundary.
        return (
            update.reshape(samples, batch, *update.shape[1:])
            .transpose(0, 1)
            .flatten(0, 1)
        )

    @typecheck
    def forward(
        self,
        noisy_coords: Float[torch.Tensor, "BS A 3"],
        noise_level: torch.Tensor,
        **kwargs: Unpack[DenoiseInputs],
    ) -> Float[torch.Tensor, "BS A 3"]:
        """Denoise one step, applying the EDM preconditioning in place.

        Kept for the reference-parity path, which does the input scaling and the
        skip/update blend inside the module. Solvers should call
        :meth:`denoise` instead and precondition through the scheduler.

        Parameters
        ----------
        noisy_coords : Tensor
            Coordinates at the current noise level.
        noise_level : Tensor
            Current sigma per sample.
        **kwargs
            Forwarded to :meth:`denoise`.

        Returns
        -------
        Tensor
            Denoised coordinates.
        """
        update = self.denoise(
            scale_coordinates(noisy_coords, noise_level, self.sigma_data),
            noise_level,
            **kwargs,
        )
        return reconstruct_coordinates(
            noisy_coords, update, noise_level, self.sigma_data
        )


class DiffusionStructureHead(nn.Module):
    """Sample coordinates by driving :class:`DiffusionModule` with a solver.

    Everything about the sampling process — the noise schedule, the churn, the
    Euler step, the per-step re-posing and Kabsch alignment — comes from
    :mod:`team_gm.diffusion`. This class only supplies the denoiser and the
    hyperparameters the released config asks for.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the shared layers.
    max_sigma : float or None
        Clip the schedule at this noise level. The reference defaults to 256.
    use_inference_cache : bool
        Hoist the iterate-independent work out of the sampling loop; see
        :class:`DenoiseCache`. Exact, and on by default.
    """

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
        max_sigma: float | None = 256.0,
        *,
        use_inference_cache: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        head = config.structure_head
        dm = head.diffusion_module

        # On by default: the hoist is exact, so there is no accuracy reason to
        # opt out. The flag exists to A/B it and to pin the equivalence in a test.
        self.use_inference_cache = use_inference_cache
        self.diffusion_module = DiffusionModule(config, implementation)
        self.scheduler = EDMScheduler(
            EDMScheduler.EDMSchedulerConfig(
                sigma_data=dm.sigma_data,
                P_mean=head.train_noise_log_mean,
                P_std=head.train_noise_log_std,
                sigma_max=head.inference_s_max,
                sigma_min=head.inference_s_min,
                rho=head.inference_p,
            )
        )
        self.solver = ESMFold2Solver(
            ESMFold2Solver.Config(
                gamma_0=head.gamma_0,
                gamma_min=head.gamma_min,
                noise_scale=head.noise_scale,
                step_scale=head.step_scale,
                max_sigma=max_sigma,
            ),
            self.scheduler,
        )

    @torch.no_grad()
    def sample(
        self,
        single_inputs: Float[torch.Tensor, "B L d_inputs"],
        pair: Float[torch.Tensor, "B L L c_z"],
        relative_position_encoding: Float[torch.Tensor, "B L L c_z"],
        ref_pos: Float[torch.Tensor, "B A 3"],
        ref_charge: Float[torch.Tensor, "B A"],
        ref_element: Float[torch.Tensor, "B A 128"],
        ref_atom_name_chars: Float[torch.Tensor, "B A 4 64"],
        ref_space_uid: Int[torch.Tensor, "B A"],
        atom_mask: Bool[torch.Tensor, "B A"],
        atom_to_token: Int[torch.Tensor, "B A"],
        mask: Bool[torch.Tensor, "B L"],
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Float[torch.Tensor, "BS A 3"]:
        """Run reverse diffusion to a set of structures.

        Parameters
        ----------
        single_inputs, pair, relative_position_encoding : Tensor
            Trunk outputs.
        ref_pos, ref_charge, ref_element, ref_atom_name_chars, ref_space_uid : Tensor
            Reference-conformer atom features.
        atom_mask, atom_to_token, mask : Tensor
            Atom validity, atom-to-token map and token validity.
        num_diffusion_samples : int
            Structures to sample per input.
        num_sampling_steps : int or None
            Solver steps before schedule clipping; defaults to the config value.
        generator : torch.Generator or None
            Source of randomness. Pass one to make a run reproducible.

        Returns
        -------
        Tensor
            Sampled coordinates, ``batch * num_diffusion_samples`` rows.
        """
        steps = num_sampling_steps or self.config.structure_head.inference_num_steps
        static = {
            "single_inputs": single_inputs,
            "pair": pair,
            "relative_position_encoding": relative_position_encoding,
            "ref_pos": ref_pos,
            "ref_charge": ref_charge,
            "ref_element": ref_element,
            "ref_atom_name_chars": ref_atom_name_chars,
            "ref_space_uid": ref_space_uid,
            "atom_mask": atom_mask,
            "atom_to_token": atom_to_token,
            "mask": mask,
            "num_diffusion_samples": num_diffusion_samples,
        }
        # Built once here, reused by all `steps` denoiser calls. Exact: every
        # cached tensor is a function of these same arguments, none of which
        # changes inside the loop.
        cache = (
            self.diffusion_module.build_cache(
                single_inputs=single_inputs,
                pair=pair,
                relative_position_encoding=relative_position_encoding,
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                atom_mask=atom_mask,
                num_diffusion_samples=num_diffusion_samples,
            )
            if self.use_inference_cache
            else None
        )

        def denoise_fn(scaled: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            """Compute denoise fn."""
            return self.diffusion_module.denoise(scaled, sigma, cache=cache, **static)

        batch, n_atoms = atom_mask.shape
        sample_mask = (
            atom_mask
            if num_diffusion_samples == 1
            else atom_mask.repeat_interleave(num_diffusion_samples, 0)
        )
        result = self.solver.sample(
            denoise_fn,
            (batch * num_diffusion_samples, n_atoms, 3),
            steps,
            device=ref_pos.device,
            generator=generator,
            mask=sample_mask,
        )
        if not isinstance(result, torch.Tensor):
            message = "Sampling without a trajectory must return coordinate tensors"
            raise TypeError(message)
        return result
