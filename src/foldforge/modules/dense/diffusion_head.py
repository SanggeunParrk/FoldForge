import torch
import torch.nn as nn
from team_gm.diffusion.edm.preconditioning import (
    reconstruct_coordinates,
    scale_coordinates,
)
from team_gm.diffusion.edm.schedules import PowerLawSchedule
from team_gm.modules.blocks.fourier_math import fourier_features
from team_gm.modules.checkpoints.backend_attention import layernorm_projection

from foldforge.data.features import dense_batch as feat_batch
from foldforge.modules import ops as fastnn
from foldforge.modules.dense import featurization
from foldforge.modules.dense.atom_cross_attention import (
    AtomCrossAttDecoder,
    AtomCrossAttEncoder,
)
from foldforge.modules.dense.diffusion_transformer import (
    DiffusionTransformer,
    DiffusionTransition,
)
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec

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


# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


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
            self.z_trunk_norm = fastnn.LayerNorm(trunk_pair_channel, bias=False)
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
        self.pair_cond_initial_norm = fastnn.LayerNorm(
            self.c_pair_cond_initial, bias="pair_cond_initial_norm" in affine
        )
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False
        )

        self.pair_transition_0 = DiffusionTransition(
            self.pair_channel, c_single_cond=None
        )
        self.pair_transition_1 = DiffusionTransition(
            self.pair_channel, c_single_cond=None
        )

        self.cond_final_norm = spec.diffusion_cond_final_norm
        if self.cond_final_norm:
            self.pair_cond_final_norm = fastnn.LayerNorm(self.pair_channel, bias=True)

        # Trunk single plus the target features -- unless there is no trunk
        # single at all, in which case the conditioning is the features alone.
        self.pair_only_trunk = not spec.trunk_single_track
        self.c_single_cond_initial = (
            spec.single_cond_channel
            if self.pair_only_trunk
            else spec.seq_channel + spec.target_feat_channel + 2 * padded_single_cond
        )
        self.single_cond_initial_norm = fastnn.LayerNorm(
            self.c_single_cond_initial, bias="single_cond_initial_norm" in affine
        )
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial,
            self.seq_channel,
            bias=spec.single_cond_projection_bias,
        )

        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = fastnn.LayerNorm(
            self.c_noise_embedding, bias="noise_embedding_initial_norm" in affine
        )
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_noise_embedding, self.seq_channel, bias=False
        )

        self.single_transition_0 = DiffusionTransition(
            self.seq_channel, c_single_cond=None
        )
        self.single_transition_1 = DiffusionTransition(
            self.seq_channel, c_single_cond=None
        )
        if spec.diffusion_cond_final_norm:
            self.single_cond_final_norm = fastnn.LayerNorm(self.seq_channel, bias=True)

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
            fastnn.LayerNorm(
                self.seq_channel, bias="single_cond_embedding_norm" in affine
            )
            if spec.single_cond_embedding_norm
            else None
        )
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False
        )

        self.transformer = DiffusionTransformer(
            c_single_cond=seq_channel,
            c_pair_cond=pair_channel,
            num_blocks=spec.diffusion_blocks,
            spec=spec,
        )

        self.output_norm = fastnn.LayerNorm(self.c_act, bias="output_norm" in affine)

        self.atom_cross_att_decoder = AtomCrossAttDecoder(spec=spec)

        self.fourier_embeddings = FourierEmbeddings(dim=256)

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Official AF3 hands FP32 trunk embeddings to its FP32 denoiser.
        # Keep this boundary explicit when the trunk itself uses BF16.
        if self.single_cond_initial_projection.weight.dtype == torch.float32:
            embeddings = {name: value.float() for name, value in embeddings.items()}
        single_embedding = use_conditioning * embeddings["single"]
        pair_embedding = use_conditioning * embeddings["pair"]

        if self.compress_trunk_pair:
            pair_embedding = layernorm_projection(
                self.z_trunk_norm, self.z_trunk_projection, pair_embedding
            )

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

        pair_cond = layernorm_projection(
            self.pair_cond_initial_norm, self.pair_cond_initial_projection, features_2d
        )

        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)
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
        single_cond = layernorm_projection(
            self.single_cond_initial_norm,
            self.single_cond_initial_projection,
            features_1d,
        )

        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )

        single_cond += layernorm_projection(
            self.noise_embedding_initial_norm,
            self.noise_embedding_initial_projection,
            noise_embedding,
        )

        single_cond += self.single_transition_0(single_cond)
        single_cond += self.single_transition_1(single_cond)
        if self.cond_final_norm:
            single_cond = self.single_cond_final_norm(single_cond)

        return single_cond, pair_cond

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool,
    ) -> torch.Tensor:
        # Get conditioning
        """Compute the module output."""
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Position features
        act = scale_coordinates(
            positions_noisy, noise_level, SIGMA_DATA, mask=atom_mask
        )

        enc = self.atom_cross_att_encoder(
            batch=batch,
            token_atoms_act=act,
            trunk_single_cond=embeddings["single"],
            trunk_pair_cond=trunk_pair_cond,
        )
        act = enc.token_act

        if self.single_cond_embedding_norm is None:
            # The conditioning already closed this track; re-normalising is not a
            # no-op even at scale one, because it re-centres and re-scales.
            act = act + self.single_cond_embedding_projection(trunk_single_cond)
        else:
            act += layernorm_projection(
                self.single_cond_embedding_norm,
                self.single_cond_embedding_projection,
                trunk_single_cond,
            )

        act = self.transformer(
            act=act,
            single_cond=trunk_single_cond,
            mask=sequence_mask,
            pair_cond=trunk_pair_cond,
            # Present only on the structural-token path: the expander's own
            # statement of which token pairs belong together.
            extra_pair_bias=embeddings.get("structural_pair_attn_bias"),
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
