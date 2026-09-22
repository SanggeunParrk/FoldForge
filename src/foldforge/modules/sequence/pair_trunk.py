"""ESMFold2's pair trunk: everything between the input features and ``z``.

The trunk is a linear recurrence (the reference calls it *parcae*) wrapped
around the shared :class:`~team_gm.models.esmfold2.FoldingTrunk`. Each step
rebuilds an injection term from the static inputs, the language model and the
MSA, mixes it into a decaying state, then runs the trunk::

    z <- a * z + B @ norm(inject)
    z <- folding_trunk(z)

``a`` and ``B`` are discretised from learned continuous-time parameters, so the
state decays at a learned per-channel rate rather than a fixed one.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.primitives import Linear
from torch import nn

from foldforge.models.config.sequence import ESMFold2Config
from foldforge.modules.sequence.embeddings import (
    LanguageModelShim,
    RelativePositionEncoding,
)
from foldforge.modules.sequence.msa_encoder import MSAEncoder
from foldforge.modules.sequence.trunk import FoldingTrunk


class PairTrunkOutput(NamedTuple):
    """Trunk outputs consumed by the structure and confidence heads."""

    pair: Float[torch.Tensor, "B L L d_pair"]
    #: ``None`` unless the caller asked for it. The distogram is a training
    #: auxiliary — nothing downstream of the trunk reads it — so at inference it
    #: is pure cost: a ``[B, L, L, d_pair]`` symmetrisation plus a projection to
    #: ``bins``, which at L=594 moves ~0.5 GB for a tensor that is discarded.
    distogram_logits: Float[torch.Tensor, "B L L bins"] | None
    relative_position_encoding: Float[torch.Tensor, "B L L d_pair"]
    token_bonds_encoding: Float[torch.Tensor, "B L L d_pair"]


class PairTrunk(nn.Module):
    """Input embedding, recurrent pair refinement and the distogram head.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the shared layers inside every trunk stack.
    """

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.config = config
        d_pair = config.d_pair
        d_inputs = config.inputs.d_inputs

        self.z_init_row = Linear(d_inputs, d_pair, bias=False, init="default")
        self.z_init_col = Linear(d_inputs, d_pair, bias=False, init="default")
        self.rel_pos = RelativePositionEncoding(
            d_pair=d_pair,
            r_max=config.n_relative_residx_bins,
            s_max=config.n_relative_chain_bins,
        )
        self.token_bonds = Linear(1, d_pair, bias=False, init="default")

        self.language_model = LanguageModelShim(
            d_pair=d_pair, d_model=config.lm_d_model, n_layers=config.lm_num_layers
        )
        if not config.lm_encoder.enabled:
            # Measured on 4YX2 (594 tokens, bf16, A6000): disabling it takes
            # pLDDT 0.842 -> 0.550, pTM 0.791 -> 0.392, and CA RMSD to the
            # deposited structure 4.38 -> 11.25 A, for 6% less compute. Both
            # released checkpoints enable it. Refusing here also lets the
            # recurrence hoist the MSA encoder out of the loop unconditionally
            # (see forward), which the disabled path would silently invalidate.
            msg = (
                "lm_encoder.enabled=False is not supported: it destroys the "
                "prediction (4YX2 pLDDT 0.842 -> 0.550, RMSD 4.4 -> 11.3 A) and "
                "only saves ~6% of the fold."
            )
            raise ValueError(msg)
        self.lm_encoder = FoldingTrunk(
            FoldingTrunk.Config(
                d_pair=d_pair,
                n_block=config.lm_encoder.n_layers,
                implementation=implementation,
            )
        )
        self.msa_encoder = (
            MSAEncoder(
                MSAEncoder.Config(
                    d_msa=config.msa_encoder.d_msa,
                    d_pair=d_pair,
                    d_inputs=d_inputs,
                    d_hidden=config.msa_encoder.d_hidden,
                    n_block=config.msa_encoder.n_layers,
                    n_heads_msa=config.msa_encoder.n_heads_msa,
                    msa_head_width=config.msa_encoder.msa_head_width,
                    implementation=implementation,
                )
            )
            if config.msa_encoder.enabled
            else None
        )

        self.ln_inject = nn.LayerNorm(d_pair)
        self.log_decay_rate = nn.Parameter(torch.zeros(d_pair))
        decay_init = math.sqrt(1.0 / 5.0)
        self.log_step = nn.Parameter(
            torch.full((d_pair,), _inverse_softplus(-math.log(decay_init)))
        )
        self.input_matrix = nn.Parameter(torch.eye(d_pair))
        self.readout = Linear(d_pair, d_pair, bias=False, init="default")
        nn.init.eye_(self.readout.weight)

        self.folding_trunk = FoldingTrunk(
            FoldingTrunk.Config(
                d_pair=d_pair,
                n_block=config.folding_trunk.n_layers,
                implementation=implementation,
            )
        )
        self.coda = FoldingTrunk(
            FoldingTrunk.Config(
                d_pair=d_pair,
                n_block=config.parcae.coda_n_layers,
                implementation=implementation,
            )
        )
        self.distogram_head = Linear(
            d_pair, config.structure_head.distogram_bins, init="default"
        )

    def discretized_dynamics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretise the continuous-time recurrence.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-channel decay ``a`` of shape ``[d_pair]`` and input matrix ``B``
            of shape ``[d_pair, d_pair]``.
        """
        step = F.softplus(self.log_step)
        decay = torch.exp(-step * torch.exp(self.log_decay_rate))
        return decay, step[:, None] * self.input_matrix

    def init_pair_state(
        self,
        reference: Float[torch.Tensor, "B L L d_pair"],
        generator: torch.Generator | None = None,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Draw the initial recurrent state, truncated at three sigma.

        Sampled by inverse CDF rather than ``nn.init.trunc_normal_``, which
        takes no generator and would silently read the global RNG — leaving a
        run unreproducible even when the caller supplied a seed.

        Parameters
        ----------
        reference : Tensor
            Tensor whose shape, device and dtype the state should match.
        generator : torch.Generator or None
            Source of randomness.

        Returns
        -------
        Tensor
            Truncated-normal state.
        """
        std = math.sqrt(2.0 / (5.0 * reference.shape[-1]))
        # Uniform over the CDF between -3 and +3 sigma, then invert.
        low = 0.5 * (1.0 + math.erf(-3.0 / math.sqrt(2.0)))
        uniform = torch.rand(
            reference.shape,
            device=reference.device,
            dtype=torch.float32,
            generator=generator,
        )
        uniform = low + uniform * (1.0 - 2.0 * low)
        state = std * torch.special.ndtri(uniform)
        return state.to(dtype=reference.dtype)

    def drop_lm_pair(
        self,
        lm_pair: torch.Tensor | None,
        lm_dropout: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | None:
        """Apply this pass's LM dropout. The only randomness in the recurrence.

        Kept out of :meth:`recurrence_step` on purpose: that method is meant to
        be CUDA-graph capturable, and a capture would freeze this draw into the
        graph — every replay would then reuse one mask, silently turning four
        independently regularised passes into the same pass four times.

        The reference forces training-mode dropout even under eval, matching how
        the LM features were regularised during training. Drawn explicitly
        rather than through ``F.dropout``, which takes no generator and would
        read the global RNG.
        """
        if lm_pair is None or lm_dropout <= 0:
            return lm_pair
        keep = (
            torch.rand(
                lm_pair.shape,
                device=lm_pair.device,
                dtype=lm_pair.dtype,
                generator=generator,
            )
            >= lm_dropout
        )
        return lm_pair * keep / (1.0 - lm_dropout)

    def recurrence_step(
        self,
        pair: torch.Tensor,
        injection_base: torch.Tensor,
        lm_pair: torch.Tensor | None,
        mask: torch.Tensor,
        decay: torch.Tensor,
        input_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """One parcae pass: LM encoder, linear recurrence, folding trunk.

        The whole loop body, as a single tensors-in/tensor-out call with no
        randomness, no data-dependent control flow and no host synchronisation —
        so it can be wrapped once with
        :func:`team_gm.modules.cudagraph.capture_method` and replayed for every
        pass. That is a much better capture unit than the folding trunk alone:
        the LM encoder (4 blocks), the injection LayerNorm and the recurrence
        arithmetic all sit between the trunk calls and would otherwise stay
        eager.

        Every argument except ``pair`` is identical on all passes; ``pair`` keeps
        its shape and only changes value, which is exactly what a static input
        buffer wants.
        """
        inject = injection_base
        if lm_pair is not None:
            # lm_encoder is mandatory (see __init__), so the LM contribution is
            # always the refined one and is always added after the MSA term.
            inject = inject + self.lm_encoder(lm_pair.to(inject.dtype), mask).to(
                inject.dtype
            )
        inject = self.ln_inject(inject)
        pair = decay * pair + F.linear(inject.to(pair.dtype), input_matrix)
        return self.folding_trunk(pair, mask)

    def injection_base(
        self,
        pair_init: torch.Tensor,
        single_inputs: torch.Tensor,
        msa_features: torch.Tensor | None,
        msa_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the MSA-conditioned injection for one parcae pass.

        The MSA encoder reads ``pair_init``, ``single_inputs``, ``msa_features``
        and ``msa_mask`` and contains nothing stochastic. With the released
        whole-MSA input those are loop-invariant, so the term is bit-identical on
        all ``num_loops + 1`` passes and :meth:`forward` computes it once. With
        ``msa_rows_per_loop`` set, :func:`sample_msa_rows` draws a new subset per
        pass and this runs every loop, as the reference did.

        This holds only because ``lm_encoder`` is mandatory here. With it
        disabled the reference adds the *dropped-out* LM pair into the MSA
        encoder's input, which would make this loop-variant — which is a second
        reason that configuration is refused in :meth:`__init__`.
        """
        if self.msa_encoder is None or msa_features is None:
            return pair_init
        if msa_mask is None:
            msg = "msa_mask is required when msa_features is given"
            raise ValueError(msg)
        msa_pair = self.msa_encoder(pair_init, single_inputs, msa_features, msa_mask)
        return (
            msa_pair.to(pair_init.dtype)
            if self.config.msa_encoder_overwrite
            else pair_init + msa_pair.to(pair_init.dtype)
        )

    @typecheck
    def forward(
        self,
        single_inputs: Float[torch.Tensor, "B L d_inputs"],
        residue_index: Int[torch.Tensor, "B L"],
        asym_id: Int[torch.Tensor, "B L"],
        sym_id: Int[torch.Tensor, "B L"],
        entity_id: Int[torch.Tensor, "B L"],
        token_index: Int[torch.Tensor, "B L"],
        token_bonds: Float[torch.Tensor, "B L L"],
        mask: Bool[torch.Tensor, "B L"],
        lm_hidden_states: Float[torch.Tensor, "B L n_lm d_model"] | None = None,
        msa_features: Float[torch.Tensor, "B M L 35"] | None = None,
        msa_mask: Bool[torch.Tensor, "B M L"] | None = None,
        num_loops: int | None = None,
        pair_state: Float[torch.Tensor, "B L L d_pair"] | None = None,
        *,
        return_distogram: bool = False,
        generator: torch.Generator | None = None,
    ) -> PairTrunkOutput:
        """Forward pass.

        Parameters
        ----------
        single_inputs : Tensor
            Token-level input features from the inputs embedder.
        residue_index, asym_id, sym_id, entity_id, token_index : Tensor
            Token indexing features for the relative position encoding.
        token_bonds : Tensor
            Pairwise bond indicator.
        mask : Tensor
            Token validity.
        lm_hidden_states : Tensor or None
            Cached ESMC hidden states. Omit to run without the language model.
        msa_features, msa_mask : Tensor or None
            MSA inputs in team-gm ``[B, M, L, ...]`` layout.
        num_loops : int or None
            Recurrence loops; the trunk runs ``num_loops + 1`` steps. Defaults
            to the config value.
        return_distogram : bool
            Run the distogram head. ``False`` by default: the distogram is a
            training auxiliary, no other head consumes it, and computing it
            costs a full symmetrised pair tensor plus a projection.
        pair_state : Tensor or None
            Initial recurrent state. Defaults to a fresh truncated-normal draw.
        generator : torch.Generator or None
            Source of randomness for the initial state and the per-loop LM
            dropout. Both are stochastic at inference, so a run is only
            reproducible when this is supplied.

        Returns
        -------
        PairTrunkOutput
            Refined pair, distogram logits, and the two encodings the
            confidence head re-uses.
        """
        loops = self.config.num_loops if num_loops is None else num_loops
        steps = max(1, loops + 1)

        pair_init = self.z_init_row(single_inputs).unsqueeze(2) + self.z_init_col(
            single_inputs
        ).unsqueeze(1)
        relative_position_encoding = self.rel_pos(
            residue_index, asym_id, sym_id, entity_id, token_index
        )
        token_bonds_encoding = self.token_bonds(token_bonds.unsqueeze(-1))
        pair_init = pair_init + relative_position_encoding + token_bonds_encoding

        lm_pair = (
            self.language_model(lm_hidden_states)
            if lm_hidden_states is not None
            else None
        )

        pair = (
            self.init_pair_state(pair_init, generator)
            if pair_state is None
            else pair_state
        )
        decay, input_matrix = self.discretized_dynamics()
        decay = decay.view(1, 1, 1, -1).to(device=pair.device, dtype=pair.dtype)
        input_matrix = input_matrix.to(device=pair.device, dtype=pair.dtype)

        lm_dropout = (
            self.config.lm_encoder.lm_dropout
            if self.config.lm_encoder.per_loop_lm_dropout
            else 0.0
        )
        rows = self.config.msa_rows_per_loop
        resample = rows is not None and msa_features is not None
        # Without per-loop sampling the MSA term is identical on every pass, so
        # the reference's per-pass recomputation of the MSA encoder is pure waste.
        base = (
            None
            if resample
            else self.injection_base(pair_init, single_inputs, msa_features, msa_mask)
        )
        for _ in range(steps):
            # The draws stay out here so recurrence_step remains capturable.
            kept = self.drop_lm_pair(lm_pair, lm_dropout, generator)
            if base is None:
                assert msa_features is not None  # noqa: S101 - narrowed above
                assert rows is not None  # noqa: S101 - narrowed above
                features, rows_mask = sample_msa_rows(
                    msa_features, msa_mask, rows, generator
                )
                inject = self.injection_base(
                    pair_init, single_inputs, features, rows_mask
                )
            else:
                inject = base
            pair = self.recurrence_step(pair, inject, kept, mask, decay, input_matrix)

        pair = self.coda(self.readout(pair), mask)
        distogram_logits = (
            self.distogram_head(pair + pair.transpose(-2, -3))
            if return_distogram
            else None
        )
        return PairTrunkOutput(
            pair=pair,
            distogram_logits=distogram_logits,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
        )


def sample_msa_rows(
    msa_features: torch.Tensor,
    msa_mask: torch.Tensor | None,
    rows: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``rows`` MSA rows, valid rows first in uniformly random order.

    Mirrors the AF3 MSA module: rows with at least one unmasked, non-gap token
    are shuffled ahead of padded or all-gap rows, then the leading ``rows`` are
    kept. Each call re-draws, so recurrence loops see different alignments. The
    ``[B, M, L, 35]`` features carry the one-hot residue types first, so the gap
    channel is ``MSA_GAP_TOKEN_ID``.
    """
    from esm.models.esmfold2.constants import (  # noqa: PLC0415 - optional input package
        MSA_GAP_TOKEN_ID,
    )

    if msa_mask is None:
        msa_mask = torch.ones(
            msa_features.shape[:3], dtype=torch.bool, device=msa_features.device
        )
    not_gap = msa_features[..., MSA_GAP_TOKEN_ID] == 0
    valid = (msa_mask & not_gap).any(dim=-1)  # [B, M]
    scores = torch.rand(valid.shape, device=valid.device, generator=generator)
    scores = torch.where(valid, scores, scores - 2.0)
    order = torch.argsort(scores, dim=1, descending=True)[
        :, : min(rows, valid.shape[1])
    ]
    features = torch.gather(
        msa_features, 1, order[:, :, None, None].expand(-1, -1, *msa_features.shape[2:])
    )
    rows_mask = torch.gather(
        msa_mask, 1, order[:, :, None].expand(-1, -1, msa_mask.shape[2])
    )
    return features, rows_mask


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
