"""ESMFold2 atom encoder and the token-level input embedder.

The atom track runs on team-gm's existing
:class:`~team_gm.modules.blocks.SWAAtomTransformer` unchanged — its parameter
names and shapes already match the released checkpoint exactly, so conversion
for that stack is the identity. What lives here is the featuriser around it: the
389-dimensional reference-conformer encoding, the atom-to-token pooling, and the
concatenation that produces the 451-dimensional token inputs the pair trunk
consumes.

One behavioural note: the reference attention casts q/k/v to bf16 regardless of
the input dtype, while team-gm's :class:`SWA3DRoPEAttention` keeps whatever it
is given. Run in bf16 to reproduce released numbers exactly; fp32 differs by the
usual bf16 rounding.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from miniworld_engine.modules.swa_atom_attention import build_attention_params
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.primitives import LayerNorm, Linear
from torch import nn

from foldforge.models.config.esmfold2 import AtomAttentionConfig, ESMFold2Config
from foldforge.modules.sequence.atom_transformer import SWAAtomTransformer
from foldforge.modules.sequence.tokens import scatter_mean_to_token


class AtomStatics(NamedTuple):
    """Atom-track work that does not depend on the diffusion iterate.

    The reference-conformer encoding and the 3D RoPE are built from ``ref_pos``,
    not from the coordinates being denoised, so both are identical at every
    solver step. Computing them once and passing them back in is exact, not an
    approximation.
    """

    conditioning: torch.Tensor
    attention_params: tuple


XYZ_DIMS = 3
MAX_ATOMIC_NUMBER = 128
CHAR_VOCAB_SIZE = 64
MAX_CHARS = 4
# ref_pos(3) + charge(1) + mask(1) + element one-hot(128) + name chars(4 x 64)
ATOM_FEATURE_DIM = XYZ_DIMS + 1 + 1 + MAX_ATOMIC_NUMBER + CHAR_VOCAB_SIZE * MAX_CHARS
NUM_RES_TYPES = 33


class AtomEncoder(nn.Module):
    """Encode reference-conformer atoms and pool them into tokens.

    Parameters
    ----------
    config : AtomAttentionConfig
        Atom-track widths, depth and 3D-RoPE settings.
    d_token_out : int
        Width of the pooled per-token output.
    structure_prediction : bool
        Add the coordinate projection used by the diffusion module. The input
        embedder does not need it.
    implementation : ImplementationType
        Kernel backend. The atom track has no fused op yet, so this is passed
        through for interface parity and takes effect the moment one lands.
    """

    def __init__(
        self,
        config: AtomAttentionConfig,
        d_token_out: int,
        *,
        structure_prediction: bool = False,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.config = config
        self.structure_prediction = structure_prediction

        self.atom_linear = Linear(
            ATOM_FEATURE_DIM, config.d_atom, bias=False, init="default"
        )
        self.atom_norm = LayerNorm(config.d_atom, implementation=implementation)
        if structure_prediction:
            # Current and predicted coordinates, concatenated.
            self.coords_linear = Linear(
                2 * XYZ_DIMS, config.d_atom, bias=False, init="default"
            )
        self.atom_transformer = SWAAtomTransformer(
            SWAAtomTransformer.Config(
                d_atom=config.d_atom,
                d_cond=config.d_atom,
                n_block=config.n_blocks,
                n_head=config.n_heads,
                swa_window_size=config.swa_window_size,
                expansion_ratio=config.expansion_ratio,
                n_spatial_rope_pairs_per_axis=config.n_spatial_rope_pairs_per_axis,
                spatial_rope_base_frequency=config.spatial_rope_base_frequency,
                n_uid_rope_pairs=config.n_uid_rope_pairs,
                uid_rope_base_frequency=config.uid_rope_base_frequency,
                implementation=implementation,
            )
        )
        self.atom_to_token_linear = Linear(
            config.d_atom, d_token_out, bias=False, init="default"
        )

    @typecheck
    def atom_features(
        self,
        ref_pos: Float[torch.Tensor, "B A 3"],
        ref_charge: Float[torch.Tensor, "B A"],
        ref_element: Float[torch.Tensor, "B A 128"],
        ref_atom_name_chars: Float[torch.Tensor, "B A 4 64"],
        atom_mask: Bool[torch.Tensor, "B A"],
    ) -> Float[torch.Tensor, "B A 389"]:
        """Concatenate the reference-conformer features for every atom.

        Parameters
        ----------
        ref_pos : Tensor
            Reference conformer coordinates.
        ref_charge : Tensor
            Formal charge.
        ref_element : Tensor
            One-hot atomic number. Padding rows must be zeroed — the downstream
            projection is bias-free, so stray padding would leak in.
        ref_atom_name_chars : Tensor
            One-hot atom-name characters, likewise zeroed on padding.
        atom_mask : Tensor
            Atom validity, which is itself a feature.

        Returns
        -------
        Tensor
            Per-atom feature vectors.
        """
        batch, n_atoms = ref_pos.shape[:2]
        return torch.cat(
            [
                ref_pos,
                ref_charge.unsqueeze(-1),
                atom_mask.to(ref_pos.dtype).unsqueeze(-1),
                ref_element,
                ref_atom_name_chars.reshape(
                    batch, n_atoms, MAX_CHARS * CHAR_VOCAB_SIZE
                ),
            ],
            dim=-1,
        )

    def static_features(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> AtomStatics:
        """Build the iterate-independent half of the atom track.

        Split out of :meth:`forward` so a diffusion sampler can hoist it out of
        the solver loop: none of it reads the coordinates being denoised.

        Parameters
        ----------
        ref_pos, ref_charge, ref_element, ref_atom_name_chars : Tensor
            Reference-conformer features; see :meth:`atom_features`.
        ref_space_uid : Tensor
            Per-atom residue id, used by the UID half of the 3D RoPE.
        atom_mask : Tensor
            Atom validity. Valid atoms must be front-packed in each row.

        Returns
        -------
        AtomStatics
            Atom conditioning and the sliding-window attention parameters.
        """
        features = self.atom_features(
            ref_pos, ref_charge, ref_element, ref_atom_name_chars, atom_mask
        )
        cos, sin = self.atom_transformer.build_rope(ref_pos, ref_space_uid)
        return AtomStatics(
            conditioning=self.atom_norm(self.atom_linear(features)),
            attention_params=build_attention_params(cos, sin, atom_mask, num_aug=1),
        )

    def forward(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_space_uid: torch.Tensor,
        atom_mask: torch.Tensor,
        atom_to_token: torch.Tensor,
        n_tokens: int,
        coords: torch.Tensor | None = None,
        predicted_coords: torch.Tensor | None = None,
        statics: AtomStatics | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Forward pass.

        Parameters
        ----------
        ref_pos, ref_charge, ref_element, ref_atom_name_chars : Tensor
            Reference-conformer features; see :meth:`atom_features`.
        ref_space_uid : Tensor
            Per-atom residue id, used by the UID half of the 3D RoPE.
        atom_mask : Tensor
            Atom validity. Valid atoms must be front-packed in each row.
        atom_to_token : Tensor
            Token index of each atom.
        n_tokens : int
            Number of tokens to pool into.
        coords, predicted_coords : Tensor or None
            Current and predicted coordinates; only used when the encoder was
            built with ``structure_prediction=True``.
        statics : AtomStatics or None
            Precomputed output of :meth:`static_features`. Pass it to skip
            rebuilding the conditioning and the RoPE; ``None`` builds them here.

        Returns
        -------
        tuple
            ``(token_features, atom_features, atom_conditioning,
            attention_params)``. The last three feed the atom decoder.
        """
        if statics is None:
            statics = self.static_features(
                ref_pos,
                ref_charge,
                ref_element,
                ref_atom_name_chars,
                ref_space_uid,
                atom_mask,
            )
        conditioning, attention_params = statics

        atoms = conditioning
        if self.structure_prediction and coords is not None:
            if predicted_coords is None:
                predicted_coords = torch.zeros_like(coords)
            # The solver deliberately keeps the diffusion iterate in fp32 —
            # coordinates accumulate over every step and bf16 has ~3 decimal
            # digits — so the cast to the network's dtype belongs here, at the
            # one projection that reads them, not in the solver.
            atoms = atoms + self.coords_linear(
                torch.cat([coords, predicted_coords], dim=-1).to(
                    self.coords_linear.weight.dtype
                )
            )

        atoms = self.atom_transformer(atoms, conditioning, attention_params)
        pooled = scatter_mean_to_token(
            F.relu(self.atom_to_token_linear(atoms)),
            atom_to_token,
            n_tokens,
            atom_mask=atom_mask,
        )
        return pooled, atoms, conditioning, attention_params


class InputsEmbedder(nn.Module):
    """Build the token-level input features the pair trunk consumes.

    Half of the 451 dimensions come from the pooled atom encoding; the rest are
    the residue one-hot, the MSA profile and the mean deletion count.

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
        self.config = config
        self.atom_encoder = AtomEncoder(
            config.inputs.atom_encoder,
            d_token_out=config.inputs.atom_encoder.d_token // 2,
            structure_prediction=False,
            implementation=implementation,
        )

    @typecheck
    def forward(
        self,
        residue_type: Float[torch.Tensor, "B L 33"],
        profile: Float[torch.Tensor, "B L 33"],
        deletion_mean: Float[torch.Tensor, "B L"],
        ref_pos: Float[torch.Tensor, "B A 3"],
        ref_charge: Float[torch.Tensor, "B A"],
        ref_element: Float[torch.Tensor, "B A 128"],
        ref_atom_name_chars: Float[torch.Tensor, "B A 4 64"],
        ref_space_uid: Int[torch.Tensor, "B A"],
        atom_mask: Bool[torch.Tensor, "B A"],
        atom_to_token: Int[torch.Tensor, "B A"],
    ) -> Float[torch.Tensor, "B L d_inputs"]:
        """Forward pass.

        Parameters
        ----------
        residue_type : Tensor
            One-hot residue type per token.
        profile : Tensor
            MSA residue-type profile per token; equals ``residue_type`` when no
            MSA is available.
        deletion_mean : Tensor
            Mean deletion count per token.
        ref_pos, ref_charge, ref_element, ref_atom_name_chars, ref_space_uid : Tensor
            Reference-conformer atom features.
        atom_mask, atom_to_token : Tensor
            Atom validity and atom-to-token map.

        Returns
        -------
        Tensor
            Token-level input features.
        """
        pooled, _, _, _ = self.atom_encoder(
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            n_tokens=residue_type.shape[1],
        )
        return torch.cat(
            [pooled, residue_type, profile, deletion_mean.unsqueeze(-1)], dim=-1
        )
