"""ESMFold2 MSA encoder, built on team-gm's shared layers.

Each block folds the MSA into the pair track with an outer-product mean, then
updates the MSA rows with pair-weighted averaging and runs the same triangle
pair update as the trunk. The final block drops the MSA-side updates: its
output is discarded, so only the pair contribution is computed.

MSA tensors use team-gm's ``[B, M, L, d]`` layout. The reference implementation
pre-transposes to ``[B, L, M, d]`` before calling; :func:`msa_from_reference`
handles that boundary.
"""

import torch
from jaxtyping import Bool, Float
from miniworld_engine.modules import (
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleMultiplication,
)
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules.blocks._engine_impl import to_engine_impl
from team_gm.modules.exceptions import ImplementationType
from torch import nn

# One-hot residue types (33) + has_deletion + deletion_value.
MSA_FEATURE_DIM = 35


class MSAEncoderBlock(nn.Module):
    """One MSA encoder block.

    Parameters
    ----------
    d_msa : int
        MSA representation width.
    d_pair : int
        Pair representation width.
    d_hidden : int
        Outer-product-mean hidden width.
    n_heads_msa : int
        Heads for pair-weighted averaging.
    msa_head_width : int
        Per-head width for pair-weighted averaging.
    is_final_block : bool
        Skip the MSA-side updates. The last block's MSA output is unused, so
        computing it would be wasted work.
    implementation : ImplementationType
        Kernel backend for the shared layers.

    """

    def __init__(
        self,
        d_msa: int = 128,
        d_pair: int = 256,
        d_hidden: int = 32,
        n_heads_msa: int = 8,
        msa_head_width: int = 16,
        *,
        is_final_block: bool = False,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.is_final_block = is_final_block
        engine = to_engine_impl(implementation)

        # ESMFold2 divides by n_valid AFTER to_out, which scales that
        # projection's bias by 1/n too. AF3 (and so the engine's default) means
        # first and projects after, leaving the bias unscaled; the two differ by
        # the whole bias term, so this flag is load-bearing, not cosmetic.
        self.outer_product_mean = OuterProductMean(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hidden=d_hidden,
            normalize_before_proj=False,
            implementation=engine,
        )
        if not is_final_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa=d_msa,
                d_pair=d_pair,
                n_head=n_heads_msa,
                d_hidden=msa_head_width,
                implementation=engine,
            )
            self.msa_transition = Transition(d_msa, n=4, implementation=engine)
        self.tri_mul_out = TriangleMultiplication(
            d_pair=d_pair, outgoing=True, implementation=engine
        )
        self.tri_mul_in = TriangleMultiplication(
            d_pair=d_pair, outgoing=False, implementation=engine
        )
        self.pair_transition = Transition(d_pair, n=4, implementation=engine)

    @typecheck
    def forward(
        self,
        msa: Float[torch.Tensor, "B M L d_msa"],
        pair: Float[torch.Tensor, "B L L d_pair"],
        msa_mask: Bool[torch.Tensor, "B M L"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "B M L d_msa"],
        Float[torch.Tensor, "B L L d_pair"],
    ]:
        """Forward pass.

        Two different residual conventions here, both set by the engine
        (ARCHITECTURE.md rule 2). The outer product mean is a CROSS-TENSOR
        residual — it reads ``msa`` and adds into ``pair`` — so the op returns a
        raw delta and the model hands it the target via ``residual=``. Everything
        else is a self-residual the op owns and fuses; adding it again here would
        silently double it.
        """
        pair = self.outer_product_mean(msa, msa_mask, residual=pair)
        if not self.is_final_block:
            msa = self.msa_pair_weighted_averaging(msa, pair, mask)
            msa = self.msa_transition(msa)
        pair = self.tri_mul_out(pair, mask)
        pair = self.tri_mul_in(pair, mask)
        return msa, self.pair_transition(pair)


class MSAEncoder(nn.Module):
    """Stack of :class:`MSAEncoderBlock`; returns the updated pair only."""

    class Config(BaseModel):
        """Configuration for the MSA encoder."""

        d_msa: int = 128
        d_pair: int = 256
        d_inputs: int = 451
        d_hidden: int = 32
        n_block: int = 4
        n_heads_msa: int = 8
        msa_head_width: int = 16
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Linear(MSA_FEATURE_DIM, config.d_msa, bias=False)
        self.project_inputs = nn.Linear(config.d_inputs, config.d_msa, bias=False)
        self.blocks = nn.ModuleList(
            [
                MSAEncoderBlock(
                    d_msa=config.d_msa,
                    d_pair=config.d_pair,
                    d_hidden=config.d_hidden,
                    n_heads_msa=config.n_heads_msa,
                    msa_head_width=config.msa_head_width,
                    is_final_block=(i == config.n_block - 1),
                    implementation=config.implementation,
                )
                for i in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single_inputs: Float[torch.Tensor, "B L d_inputs"],
        msa_features: Float[torch.Tensor, "B M L 35"],
        msa_mask: Bool[torch.Tensor, "B M L"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass.

        Parameters
        ----------
        pair : Tensor
            Pair representation to condition.
        single_inputs : Tensor
            Token-level input features, broadcast across MSA rows.
        msa_features : Tensor
            One-hot residue types concatenated with ``has_deletion`` and
            ``deletion_value``. Padding rows must be zeroed: the embedding is
            bias-free, so non-zero padding would leak into every row.
        msa_mask : Tensor
            Per-row, per-column validity.

        Returns
        -------
        Tensor
            The conditioned pair representation.

        """
        msa = self.embed(msa_features) + self.project_inputs(single_inputs).unsqueeze(1)
        # Row 0 is the query, so its column mask is the token mask.
        mask = msa_mask[:, 0]
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask, mask)
        return pair


def msa_from_reference(
    msa_oh: torch.Tensor,
    has_deletion: torch.Tensor,
    deletion_value: torch.Tensor,
) -> torch.Tensor:
    """Assemble MSA features from the reference implementation's tensors.

    The reference passes ``[B, L, M, ...]``; team-gm layers use ``[B, M, L, ...]``.

    Parameters
    ----------
    msa_oh : Tensor
        ``[B, L, M, 33]`` one-hot residue types.
    has_deletion : Tensor
        ``[B, L, M]`` deletion indicator.
    deletion_value : Tensor
        ``[B, L, M]`` deletion magnitude.

    Returns
    -------
    Tensor
        ``[B, M, L, 35]`` feature tensor in team-gm layout.

    """
    feats = torch.cat(
        [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1
    )
    return feats.permute(0, 2, 1, 3)
