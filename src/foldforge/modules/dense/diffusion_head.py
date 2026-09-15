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

    def __init__(self) -> None:
        super().__init__()

        self.c_act = 768
        self.pair_channel = 128
        self.seq_channel = 384

        self.c_pair_cond_initial = 267
        self.pair_cond_initial_norm = fastnn.LayerNorm(
            self.c_pair_cond_initial, bias=False
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

        self.c_single_cond_initial = 831
        self.single_cond_initial_norm = fastnn.LayerNorm(
            self.c_single_cond_initial, bias=False
        )
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial, self.seq_channel, bias=False
        )

        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = fastnn.LayerNorm(
            self.c_noise_embedding, bias=False
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

        self.atom_cross_att_encoder = AtomCrossAttEncoder(
            per_token_channels=self.c_act,
            with_token_atoms_act=True,
            with_trunk_pair_cond=True,
            with_trunk_single_cond=True,
        )

        self.single_cond_embedding_norm = fastnn.LayerNorm(self.seq_channel, bias=False)
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False
        )

        self.transformer = DiffusionTransformer()

        self.output_norm = fastnn.LayerNorm(self.c_act, bias=False)

        self.atom_cross_att_decoder = AtomCrossAttDecoder()

        self.fourier_embeddings = FourierEmbeddings(dim=256)

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_embedding = use_conditioning * embeddings["single"]
        pair_embedding = use_conditioning * embeddings["pair"]

        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=32, max_relative_chain=2
        ).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)

        pair_cond = layernorm_projection(
            self.pair_cond_initial_norm, self.pair_cond_initial_projection, features_2d
        )

        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)

        target_feat = embeddings["target_feat"]
        features_1d = torch.concatenate([single_embedding, target_feat], dim=-1)
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
