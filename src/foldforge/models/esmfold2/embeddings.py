"""ESMFold2 pair-track embeddings: relative position, LM projection, pooling."""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules.primitives import Linear
from torch import nn


class RelativePositionEncoding(nn.Module):
    """Relative position / token / chain / entity features, projected to pair.

    Deliberately not :class:`~team_gm.modules.layers.RelativePositionEmbedding`:
    that one differs from the AlphaFold-3 recipe ESMFold2 follows in two ways
    that both change the result, so the two cannot share weights.

    1. The chain feature is bucketed on ``same_chain`` here (same chain gets the
       out-of-range bin, different chains get the clipped ``sym_id`` delta);
       team-gm's buckets on ``same_entity``, with the branches the other way up.
    2. Feature concatenation order is (residue, token, same-entity, chain);
       team-gm's is (residue, token, chain, same-entity), which permutes the
       projection's input columns.

    Parameters
    ----------
    d_pair : int
        Output pair width.
    r_max : int
        Residue/token relative-position clip range.
    s_max : int
        Chain relative-position clip range.

    """

    def __init__(self, d_pair: int = 256, r_max: int = 32, s_max: int = 2) -> None:
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        n_features = 2 * (2 * r_max + 2) + 1 + (2 * s_max + 2)
        self.embed = Linear(n_features, d_pair, bias=False, init="default")

    @typecheck
    def forward(
        self,
        residue_index: Int[torch.Tensor, "B L"],
        asym_id: Int[torch.Tensor, "B L"],
        sym_id: Int[torch.Tensor, "B L"],
        entity_id: Int[torch.Tensor, "B L"],
        token_index: Int[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        with torch.no_grad():
            same_chain = asym_id[:, :, None] == asym_id[:, None, :]
            same_residue = residue_index[:, :, None] == residue_index[:, None, :]
            same_entity = entity_id[:, :, None] == entity_id[:, None, :]

            d_residue = residue_index[:, :, None] - residue_index[:, None, :]
            d_residue = torch.clip(d_residue + self.r_max, 0, 2 * self.r_max)
            d_residue = torch.where(same_chain, d_residue, 2 * self.r_max + 1)

            d_token = token_index[:, :, None] - token_index[:, None, :]
            d_token = torch.clip(d_token + self.r_max, 0, 2 * self.r_max)
            d_token = torch.where(
                same_chain & same_residue, d_token, 2 * self.r_max + 1
            )

            d_chain = sym_id[:, :, None] - sym_id[:, None, :]
            d_chain = torch.clip(d_chain + self.s_max, 0, 2 * self.s_max)
            d_chain = torch.where(same_chain, 2 * self.s_max + 1, d_chain)

            features = torch.cat(
                [
                    F.one_hot(d_residue.long(), 2 * self.r_max + 2).float(),
                    F.one_hot(d_token.long(), 2 * self.r_max + 2).float(),
                    same_entity.float().unsqueeze(-1),
                    F.one_hot(d_chain.long(), 2 * self.s_max + 2).float(),
                ],
                dim=-1,
            )
        # One-hots are built in fp32 for exactness, then handed over in the
        # projection's own dtype: under a bf16 model a hardcoded fp32 feature
        # block collides with bf16 weights at this Linear.
        return self.embed(features.to(self.embed.weight.dtype))


class SingleToPair(nn.Module):
    """Outer product and difference of a single track, mixed into a pair track.

    Parameters
    ----------
    d_single : int
        Input single width.
    d_hidden : int
        Width the single track is projected to before the outer op.
    d_pair : int
        Output pair width.

    """

    def __init__(self, d_single: int, d_hidden: int, d_pair: int) -> None:
        super().__init__()
        self.downproject = Linear(d_single, d_hidden, init="default")
        self.expand = Linear(2 * d_hidden, d_pair, init="default")
        self.project = Linear(d_pair, d_pair, init="default")

    @typecheck
    def forward(
        self,
        single: Float[torch.Tensor, "B L d_single"],
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass."""
        x = self.downproject(single)
        pair = torch.cat(
            [x.unsqueeze(2) * x.unsqueeze(1), x.unsqueeze(2) - x.unsqueeze(1)],
            dim=-1,
        )
        return self.project(F.gelu(self.expand(pair)))


class LanguageModelShim(nn.Module):
    """Turn cached ESMC hidden states into a pair representation.

    The per-layer mixing weights are a softmax over a learned vector, so the
    shim picks its own blend of the LM's 81 hidden layers.

    Parameters
    ----------
    d_pair : int
        Output pair width.
    d_model : int
        LM hidden width.
    n_layers : int
        LM depth; the shim mixes ``n_layers + 1`` hidden states.

    """

    def __init__(
        self, d_pair: int = 256, d_model: int = 2560, n_layers: int = 80
    ) -> None:
        super().__init__()
        self.ln_lm = nn.LayerNorm(d_model)
        self.to_pair_single = Linear(d_model, d_pair, bias=False, init="default")
        self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))
        self.single_to_pair = SingleToPair(d_pair, d_pair, d_pair)
        self.ln_pair = nn.LayerNorm(d_pair)

    @typecheck
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B L n_layers_plus_1 d_model"],
        lm_dropout: float = 0.0,
    ) -> Float[torch.Tensor, "B L L d_pair"]:
        """Forward pass.

        Parameters
        ----------
        hidden_states : Tensor
            Per-layer ESMC hidden states, detached upstream.
        lm_dropout : float
            Dropout on the resulting pair track. The reference applies this at
            inference too (forced ``training=True``), so it is an explicit
            argument rather than a module-level ``nn.Dropout``.

        Returns
        -------
        Tensor
            Pair representation contributed by the language model.

        """
        single = self.to_pair_single(self.ln_lm(hidden_states))
        weights = self.layer_weights.softmax(0).to(single.dtype)
        single = torch.einsum("n,blnd->bld", weights, single)
        pair = self.ln_pair(self.single_to_pair(single))
        if lm_dropout > 0:
            pair = F.dropout(pair, p=lm_dropout, training=True)
        return pair


class RowAttentionPooling(nn.Module):
    """Collapse a pair track to a single track by attention over each row.

    Parameters
    ----------
    d_pair : int
        Input pair width.
    d_single : int
        Output single width.

    """

    def __init__(self, d_pair: int = 256, d_single: int = 384) -> None:
        super().__init__()
        self.to_score = Linear(d_pair, 1, bias=False, init="default")
        self.to_out = Linear(d_pair, d_single, bias=False, init="default")

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        mask: Bool[torch.Tensor, "B L"],
    ) -> Float[torch.Tensor, "B L d_single"]:
        """Forward pass."""
        scores = self.to_score(pair).squeeze(-1)
        # -1e9 rather than finfo.min: matches the reference, and stays finite
        # under bf16 autocast where finfo.min would round to -inf.
        scores = scores.masked_fill(~mask[:, None, :], -1e9)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, pair)
        return self.to_out(pooled)
