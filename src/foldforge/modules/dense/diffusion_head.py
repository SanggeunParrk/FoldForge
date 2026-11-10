import torch
import torch.nn as nn

from foldforge.data.features import dense_batch as feat_batch
from foldforge.diffusion.edm.schedules import PowerLawSchedule
from foldforge.modules.dense import featurization
from foldforge.modules.dense.atom_cross_attention import (
    AtomCrossAttDecoder,
    AtomCrossAttEncoder,
)
from foldforge.modules.dense.preconditioning import (
    reconstruct_coordinates,
    scale_coordinates,
)
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec
from foldforge.modules.engine import graph as engine_graph
from foldforge.modules.engine import modules as engine_modules

# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


def _plus_transition(transition: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``x + transition(x)``.

    The engine ``Transition`` owns its residual and returns the sum itself: adding its delta back would
    round twice in bf16.
    """
    if isinstance(transition, engine_modules.Transition):
        return transition(x)
    return x + transition(x)


# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0

#: Rows (steps x tokens) the batched conditioning of a whole trajectory passes through
#:the transitions at once.
MAX_TABLE_ROWS = 32768


def fourier_features(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``cos(2 pi (value * weight + bias))`` for scalar noise levels, with the checkpoint's fixed projection."""
    return torch.cos(2 * torch.pi * (value[..., None] * weight + bias))


class FourierEmbeddings(nn.Module):
    """Represent fourier embeddings."""

    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        if not hasattr(self, "weight") or not hasattr(self, "bias"):
            msg = "FourierEmbeddings not initialized"
            raise RuntimeError(msg)
        return fourier_features(x, self.weight, self.bias)


class DiffusionHead(nn.Module):
    """Represent diffusion head."""

    def __init__(self, spec: DenseSpec = ALPHAFOLD3) -> None:
        super().__init__()

        self.spec = spec
        pair_channel = spec.diffusion_pair_channel
        seq_channel = spec.diffusion_seq_channel
        trunk_pair_channel = spec.pair_channel
        padded_single_cond = spec.padded_single_cond
        self.padded_single_cond = padded_single_cond
        affine = spec.affine_norms

        self.c_act = 768
        self.pair_channel = pair_channel
        self.seq_channel = seq_channel

        # Trunk pair plus the 139 relative-position features, raw or projected.
        self.relpe_projection = (
            nn.Linear(139, pair_channel, bias=False)
            if spec.diffusion_projected_relpos
            else None
        )
        self.pair_init_cond = spec.diffusion_pair_init_cond

        # A family whose denoiser pair is NARROWER than the trunk's compresses
        # the trunk pair to that width first, rather than norming the whole
        # concatenation at the trunk's width. The joint norm couples the two
        # halves, so which width it runs at is a real difference in the path.
        self.compress_trunk_pair = trunk_pair_channel != pair_channel
        if self.compress_trunk_pair:
            self.z_trunk_norm = nn.LayerNorm(trunk_pair_channel, bias=False)
            self.z_trunk_projection = nn.Linear(
                trunk_pair_channel, pair_channel, bias=False
            )
            first = pair_channel
        else:
            first = trunk_pair_channel
        self.c_pair_cond_initial = first + (
            trunk_pair_channel
            if self.pair_init_cond
            else (pair_channel if spec.diffusion_projected_relpos else 139)
        )
        self.pair_cond_initial_norm = nn.LayerNorm(
            self.c_pair_cond_initial, bias="pair_cond_initial_norm" in affine
        )
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False
        )

        self.pair_transition_0 = self._transition(spec, self.pair_channel)
        self.pair_transition_1 = self._transition(spec, self.pair_channel)

        self.cond_final_norm = spec.diffusion_cond_final_norm
        if self.cond_final_norm:
            self.pair_cond_final_norm = nn.LayerNorm(self.pair_channel, bias=True)

        # Trunk single plus the target features -- unless there is no trunk
        # single at all, in which case the conditioning is the features alone.
        self.pair_only_trunk = not spec.trunk_single_track
        self.c_single_cond_initial = (
            spec.single_cond_channel
            if self.pair_only_trunk
            else spec.seq_channel + spec.target_feat_channel + 2 * padded_single_cond
        )
        self.single_cond_initial_norm = nn.LayerNorm(
            self.c_single_cond_initial, bias="single_cond_initial_norm" in affine
        )
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial,
            self.seq_channel,
            bias=spec.single_cond_projection_bias,
        )

        self._fold = None
        self._fold_epoch = 0
        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = nn.LayerNorm(
            self.c_noise_embedding, bias="noise_embedding_initial_norm" in affine
        )
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_noise_embedding, self.seq_channel, bias=False
        )

        self.single_transition_0 = self._transition(spec, self.seq_channel)
        self.single_transition_1 = self._transition(spec, self.seq_channel)
        if spec.diffusion_cond_final_norm:
            self.single_cond_final_norm = nn.LayerNorm(self.seq_channel, bias=True)

        self.atom_cross_att_encoder = AtomCrossAttEncoder(
            per_token_channels=self.c_act,
            with_token_atoms_act=True,
            with_trunk_pair_cond=True,
            with_trunk_single_cond=True,
            trunk_pair_channels=pair_channel,
            trunk_single_channels=spec.seq_channel,
            spec=spec,
        )

        self.single_cond_embedding_norm = (
            nn.LayerNorm(self.seq_channel, bias="single_cond_embedding_norm" in affine)
            if spec.single_cond_embedding_norm
            else None
        )
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False
        )

        self.transformer = engine_graph.make_token_transformer(
            spec, c_act=self.c_act, d_cond=seq_channel, d_pair=pair_channel, num_blocks=spec.diffusion_blocks
        )

        self.output_norm = nn.LayerNorm(self.c_act, bias="output_norm" in affine)

        self.atom_cross_att_decoder = AtomCrossAttDecoder(spec=spec)

        self.fourier_embeddings = FourierEmbeddings(dim=256)

    @staticmethod
    def _transition(spec: DenseSpec, width: int) -> nn.Module:  # noqa: ARG004
        """A conditioning transition (no conditioning input, factor 2): the engine `Transition`."""
        return engine_graph.make_conditioning_transition(width)

    def _fold_conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run  fold conditioning.

        The pair conditioning and the noise-independent part of the single
        conditioning.

        Both read only the trunk output and the batch, never the noise level or the
        coordinates, so they are the same at every sampler step of a fold.
        """
        # Official AF3 hands FP32 trunk embeddings to its FP32 denoiser.
        # Keep this boundary explicit when the trunk itself uses BF16.
        if self.single_cond_initial_projection.weight.dtype == torch.float32:
            embeddings = {name: value.float() for name, value in embeddings.items()}
        single_embedding = use_conditioning * embeddings["single"]
        pair_embedding = use_conditioning * embeddings["pair"]

        if self.compress_trunk_pair:
            pair_embedding = self.z_trunk_projection(self.z_trunk_norm(pair_embedding))

        if self.pair_init_cond:
            second = embeddings["pair_init"].to(dtype=pair_embedding.dtype)
        else:
            second = featurization.create_relative_encoding(
                batch.token_features,
                max_relative_idx=32,
                max_relative_chain=2,
                chain_bucket_on_same_chain=self.spec.chain_bucket_on_same_chain,
            ).to(dtype=pair_embedding.dtype)
            if self.relpe_projection is not None:
                second = self.relpe_projection(second)
        features_2d = torch.concatenate([pair_embedding, second], dim=-1)

        pair_cond = self.pair_cond_initial_projection(self.pair_cond_initial_norm(features_2d))

        pair_cond = _plus_transition(self.pair_transition_0, pair_cond)
        pair_cond = _plus_transition(self.pair_transition_1, pair_cond)
        if self.cond_final_norm:
            pair_cond = self.pair_cond_final_norm(pair_cond)

        # The diffusion module takes the structure projection where the family
        # trained one; every other family hands it the same tensor as the trunk.
        target_feat = embeddings.get("structure_target_feat", embeddings["target_feat"])
        if self.spec.single_cond_layout == "esm":
            target_feat = featurization.widen_to_esm_classes(target_feat)
        features_1d = (
            target_feat
            if self.pair_only_trunk
            else torch.concatenate([single_embedding, target_feat], dim=-1)
        )
        if self.padded_single_cond:
            # One zero column after each 31-class block (restype, then profile).
            pad = torch.zeros_like(features_1d[..., :1])
            restype_end = single_embedding.shape[-1] + 31
            features_1d = torch.concatenate(
                [
                    features_1d[..., :restype_end],
                    pad,
                    features_1d[..., restype_end : restype_end + 31],
                    pad,
                    features_1d[..., restype_end + 31 :],
                ],
                dim=-1,
            )
        single_cond = self.single_cond_initial_projection(self.single_cond_initial_norm(features_1d))

        return single_cond, pair_cond

    def _fold_atom_state(self, batch, embeddings, pair_cond):
        """Run  fold atom state.

        The atom encoder's fold-invariant part (atom conditioning, atom-pair
        activations, rotary tables).
        """
        return self.atom_cross_att_encoder.prepare(
            batch,
            trunk_single_cond=embeddings["single"],
            trunk_pair_cond=pair_cond,
        )

    def _noise_term(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Run  noise term.

        The noise-level embedding the single conditioning adds (one row per noise
        level).
        """
        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )
        return self.noise_embedding_initial_projection(self.noise_embedding_initial_norm(noise_embedding))

    def _finish_single_conditioning(self, single_cond: torch.Tensor) -> torch.Tensor:
        """Run  finish single conditioning.

        The conditioning transitions (and the closing norm) over the fold's single
        plus the noise term.
        """
        single_cond = _plus_transition(self.single_transition_0, single_cond)
        single_cond = _plus_transition(self.single_transition_1, single_cond)
        if self.cond_final_norm:
            single_cond = self.single_cond_final_norm(single_cond)
        return single_cond

    def _complete_single_conditioning(
        self, single_base: torch.Tensor, noise_level: torch.Tensor
    ) -> torch.Tensor:
        """Run  complete single conditioning.

        Add the noise-level embedding to the fold's single conditioning (per step).
        """
        return self._finish_single_conditioning(
            single_base + self._noise_term(noise_level)
        )

    def conditioning_table(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        noise_levels: torch.Tensor,
    ) -> torch.Tensor:
        """Run conditioning table.

        The single conditioning of every step of a trajectory at once: ``[steps,
        *single_base.shape]``.

        It depends on the fold and the noise level only, not on the iterate, so a
        sampler that knows its whole schedule makes it in one batched pass instead
        of one small pass per step (``_step(..., single_cond=table[i])``).
        """
        single_base = self._conditioned(batch, embeddings, use_conditioning=True)[0]
        # The transitions widen every row 8x: bound the rows of one pass, not
        # the length of the trajectory.
        per_pass = max(
            1, MAX_TABLE_ROWS // max(1, single_base.numel() // single_base.shape[-1])
        )
        rows = []
        for levels in noise_levels.split(per_pass):
            term = self._noise_term(levels)
            term = term.reshape(
                term.shape[0], *(1,) * (single_base.ndim - 1), term.shape[-1]
            )
            rows.append(self._finish_single_conditioning(single_base[None] + term))
        return rows[0] if len(rows) == 1 else torch.cat(rows)

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
        single_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run forward.

        Compute the module output (``single_cond``: this step's row of
        :meth:`conditioning_table`, else made here).
        """
        single_base, pair_cond, lite, epoch, key_mask, atom_state = self._conditioned(
            batch, embeddings, use_conditioning
        )
        return self._step(
            positions_noisy,
            noise_level,
            batch,
            lite,
            single_base,
            pair_cond,
            key_mask,
            atom_state,
            fold_epoch=epoch,
            single_cond=single_cond,
        )

    def _conditioned(self, batch, embeddings, use_conditioning):
        """Run  conditioned.

        The fold's conditioning, made once and reused by every step that reads the
        same trunk output.

        The key is the identity and version of every trunk tensor (and the batch): a
        new fold has new tensors, an in-place change bumps a version. The entry
        keeps its tensors alive, so an address cannot be reused while the entry
        stands.
        """
        try:
            key = (
                id(batch),
                bool(use_conditioning),
                tuple(
                    (name, id(value), value._version, tuple(value.shape))
                    for name, value in sorted(embeddings.items())
                ),
            )
        except RuntimeError:  # inference tensors carry no version counter
            key = (
                id(batch),
                bool(use_conditioning),
                tuple(
                    (name, id(value), value.data_ptr(), tuple(value.shape))
                    for name, value in sorted(embeddings.items())
                ),
            )
        entry = self._fold
        if entry is None or entry[0] != key:
            single_base, pair_cond = self._fold_conditioning(
                batch, embeddings, use_conditioning
            )
            lite = {
                name: embeddings[name]
                for name in ("single", "structural_pair_attn_bias")
                if name in embeddings
            }
            self._fold_epoch += 1
            # The key mask is one tensor for the whole fold: the engine
            # keys its hoisted pair bias
            # on its address, and a mask made inside the (compiled,
            # captured) step is a new
            # tensor at every call.
            key_mask = (batch.token_features.mask > 0)[None]
            atom_state = self._fold_atom_state(batch, embeddings, pair_cond)
            entry = self._fold = (
                key,
                single_base,
                pair_cond,
                lite,
                self._fold_epoch,
                key_mask,
                atom_state,
                (batch, embeddings),
            )
        return entry[1], entry[2], entry[3], entry[4], entry[5], entry[6]

    def _step(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        single_base: torch.Tensor,
        trunk_pair_cond: torch.Tensor,
        key_mask: torch.Tensor,
        atom_state,
        fold_epoch: int = 0,
        single_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run  step.

        One denoiser evaluation given the fold's conditioning (the part CUDA graphs
        capture).

        ``fold_epoch`` only keys the graph: it changes when the conditioning was
        remade for another fold, so a captured graph never outlives the tensors it
        was captured with.
        """
        trunk_single_cond = (
            single_cond
            if single_cond is not None
            else self._complete_single_conditioning(single_base, noise_level)
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Position features
        act = scale_coordinates(
            positions_noisy, noise_level, SIGMA_DATA, mask=atom_mask
        )

        enc = self.atom_cross_att_encoder.step(batch, act, atom_state)
        act = enc.token_act

        if self.single_cond_embedding_norm is None:
            # The conditioning already closed this track; re-normalising is not a
            # no-op even at scale one, because it re-centres and re-scales.
            act = act + self.single_cond_embedding_projection(trunk_single_cond)
        else:
            act += self.single_cond_embedding_projection(self.single_cond_embedding_norm(trunk_single_cond))

        act = self.transformer(
            act=act,
            single_cond=trunk_single_cond,
            mask=sequence_mask,
            pair_cond=trunk_pair_cond,
            # Present only on the structural-token path: the expander's own
            # statement of which token pairs belong together.
            extra_pair_bias=embeddings.get("structural_pair_attn_bias"),
            key_mask=key_mask,
        )
        act = self.output_norm(act)

        # (Possibly) atom-granularity decoder
        position_update = self.atom_cross_att_decoder(
            batch=batch,
            token_act=act,
            enc=enc,
        )

        return reconstruct_coordinates(
            positions_noisy, position_update, noise_level, SIGMA_DATA, mask=atom_mask
        )


# Compatibility names; shared implementations preserve the dense checkpoint layout.


def noise_schedule(t, smin: float = 0.0004, smax: float = 160.0, p: int = 7):
    """Compute noise schedule."""
    return PowerLawSchedule(
        PowerLawSchedule.Config(
            sigma_data=SIGMA_DATA,
            sigma_min=smin,
            sigma_max=smax,
            rho=p,
            terminal_zero=False,
        )
    ).levels(t)
