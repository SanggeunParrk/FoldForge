"""ESMFold2 pair-track trunk, built on team-gm's shared layers.

The released trunk is pair-only: no single representation and no triangle
attention, just outgoing/incoming triangle multiplication and a SwiGLU
transition, repeated ``n_layers`` times. That is exactly
:class:`~team_gm.modules.layers.TriangleMultiplication` and
:class:`~team_gm.modules.layers.Transition`, so this module contributes the
wiring and nothing else — no second copy of the triangle kernels.

Three stacks in ESMFold2 are this same block at different depths: the main
trunk (48), the LM-side pair encoder (4) and the parcae coda (2), plus a
fourth inside the confidence head (4).
"""

import torch
from jaxtyping import Bool, Float
from miniworld_engine.modules import Transition, TriangleMultiplication
from pydantic import BaseModel, model_validator
from team_gm import typecheck
from team_gm.modules.blocks._engine_impl import to_engine_impl
from team_gm.modules.exceptions import ImplementationType
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
from typing_extensions import Self


class PairUpdateBlock(nn.Module):
    """One ESMFold2 trunk block: outgoing trimul, incoming trimul, transition.

    Parameters
    ----------
    d_pair : int
        Pair representation width.
    expansion_ratio : int
        Transition hidden expansion factor.
    implementation : ImplementationType
        Kernel backend for the triangle and transition ops.

    """

    def __init__(
        self,
        d_pair: int = 256,
        expansion_ratio: int = 4,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        # Two separate modules rather than BidirectionalTriangleMultiplication:
        # the bidirectional block shares one input LayerNorm across both flows,
        # so it cannot hold ESMFold2's per-flow weights. Use the bidirectional
        # variant when training this architecture from scratch.
        engine = to_engine_impl(implementation)
        self.tri_mul_out = TriangleMultiplication(
            d_pair=d_pair,
            outgoing=True,
            implementation=engine,
        )
        self.tri_mul_in = TriangleMultiplication(
            d_pair=d_pair,
            outgoing=False,
            implementation=engine,
        )
        self.pair_transition = Transition(
            d_pair,
            n=expansion_ratio,
            implementation=engine,
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass.

        No ``pair + ...`` here: every engine op owns its own self-residual and
        fuses it into its kernel epilogue (ARCHITECTURE.md rule 2). Re-adding it
        would double it, and nothing would raise — it produces a plausible, wrong
        structure.
        """
        pair = self.tri_mul_out(pair, mask)
        pair = self.tri_mul_in(pair, mask)
        return self.pair_transition(pair)


class FoldingTrunk(nn.Module):
    """Stack of :class:`PairUpdateBlock`."""

    class Config(BaseModel):
        """Configuration for the folding trunk."""

        d_pair: int = 256
        expansion_ratio: int = 4
        n_block: int = 48
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE
        n_checkpoint_segments: int | None = None

        @model_validator(mode="after")
        def check_checkpoint_segments(self) -> Self:
            """Check n_checkpoint_segments is valid."""
            if self.n_checkpoint_segments is None:
                return self
            if (
                self.n_checkpoint_segments > self.n_block
                or self.n_checkpoint_segments < 1
            ):
                msg = (
                    "n_checkpoint_segments must be between 1 and n_block. "
                    f"Got n_checkpoint_segments={self.n_checkpoint_segments} "
                    f"and n_block={self.n_block}."
                )
                raise ValueError(msg)
            return self

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                PairUpdateBlock(
                    d_pair=config.d_pair,
                    expansion_ratio=config.expansion_ratio,
                    implementation=config.implementation,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"] | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass, keeping independent samples within the kernel batch limit."""
        if (
            not self.training
            and pair.shape[0] > 1
            and self.config.implementation == ImplementationType.MINIWORLD_ENGINE
        ):
            # The pinned inference TriMul front accepts B=1. Confidence expands
            # B by the diffusion-sample count; evaluate those independent items
            # separately while preserving order, masks and the selected backend.
            return torch.cat(
                [
                    self.forward(
                        pair[i : i + 1], None if mask is None else mask[i : i + 1]
                    )
                    for i in range(pair.shape[0])
                ],
                dim=0,
            )
        if self.config.n_checkpoint_segments is None:
            for block in self.blocks:
                pair = block(pair, mask)
            return pair

        def run_module(module):  # noqa: ANN001, ANN202
            def forward(x):  # noqa: ANN001, ANN202
                return module(x, mask)

            return forward

        return checkpoint_sequential(
            [run_module(b) for b in self.blocks],
            segments=self.config.n_checkpoint_segments,
            use_reentrant=False,
            input=pair,
        )
