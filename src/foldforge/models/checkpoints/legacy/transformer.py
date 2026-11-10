"""The dense released-layout token transformer (``DiffusionTransformer``) and its self attention (converter graph only).

Runtime graphs hold ``TokenDiT*`` engine blocks (:mod:`foldforge.modules.engine.transformers`); this is the released AF3 layout the
Haiku blob is read into before :mod:`foldforge.models.checkpoints.legacy.released` converts it.
"""

import einops
import torch
import torch.nn as nn

from foldforge.models.checkpoints.legacy.reference.attention import (
    dot_product_attention,
)
from foldforge.modules.dense.diffusion_transformer import (
    AdaLNZero,
    AdaptiveLayerNorm,
    conditioned_residual,
    conditioned_transition,
)
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec


class SelfAttention(nn.Module):
    """Represent self attention."""

    def __init__(
        self,
        c_x: int = 768,
        c_single_cond: int = 384,
        num_head: int = 16,
        use_single_cond: bool = False,
        kq_norm: bool = False,
        identity_scale: bool = False,
        norm_eps: float = 1e-5,
        gating_query: bool = True,
        project_output: bool = True,
        gate_bias: float = 0.0,
    ) -> None:

        super().__init__()
        self.kq_norm = kq_norm
        self.use_gating_query = gating_query
        # Added to the gate LOGITS, before the sigmoid. A bias there cannot be
        # folded into the projection that produces them, because a sigmoid does
        # not commute with a linear map.
        self.gate_bias = gate_bias
        if kq_norm:
            self.query_layer_norm = nn.LayerNorm(c_x)
            self.key_layer_norm = nn.LayerNorm(c_x)

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            identity_scale=identity_scale,
            eps=norm_eps,
        )

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        if self.use_gating_query:
            self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond,
            project=project_output,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_logits: torch.Tensor | None = None,
        single_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:

        x (torch.Tensor): (num_tokens, ch)
        mask (torch.Tensor): (num_tokens,)
        pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """
        if (single_cond is None) != (self.use_single_cond is False):
            message = (
                "Invalid state: (single_cond is None) == (self.use_single_cond is"
                " False)"
            )
            raise ValueError(message)

        x = self.adaptive_layernorm(x, single_cond)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)
        if self.kq_norm:
            # Over the flattened head axis, before the per-head scaling.
            q, k = self.query_layer_norm(q), self.key_layer_norm(k)

        q, k, v = (
            einops.rearrange(t, "... n (h c) -> ... h n c", h=self.num_head)
            if x.ndim > 2
            else einops.rearrange(t, "n (h c) -> h n c", h=self.num_head).unsqueeze(0)
            for t in [q, k, v]
        )

        weighted_avg = dot_product_attention(q, k, v, mask=mask, bias=pair_logits)

        if x.ndim == 2:
            weighted_avg = weighted_avg.squeeze(0)
        weighted_avg = einops.rearrange(weighted_avg, "... h q c -> ... q (h c)")

        gate_logits = self.gating_query(x) if self.use_gating_query else None
        if gate_logits is not None and self.gate_bias:
            gate_logits = gate_logits + self.gate_bias
        return self.adaptive_zero_init(
            weighted_avg, single_cond, gate_logits=gate_logits
        )


class DiffusionTransformer(nn.Module):
    """Represent diffusion transformer."""

    def __init__(
        self,
        c_act: int = 768,
        c_single_cond: int = 384,
        c_pair_cond: int = 128,
        num_head: int = 16,
        num_blocks: int = 24,
        super_block_size: int = 4,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:

        super().__init__()
        self.per_block_pair = spec.per_block_pair_layer_norm
        self.parallel = spec.parallel_attention_transition

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        if self.per_block_pair:
            # One pair norm and one projection per block, as the vendor trained it.
            self.pair_input_layer_norm = nn.ModuleList(
                [
                    nn.LayerNorm(
                        self.c_pair_cond,
                        bias="pair_input_layer_norm" in spec.affine_norms,
                    )
                    for _ in range(self.num_blocks)
                ]
            )
            self.pair_logits_projection = nn.ModuleList(
                [
                    nn.Linear(self.c_pair_cond, self.num_head, bias=False)
                    for _ in range(self.num_blocks)
                ]
            )
        else:
            self.pair_input_layer_norm = nn.LayerNorm(
                self.c_pair_cond,
                bias="pair_input_layer_norm" in spec.affine_norms,
            )
            self.pair_logits_projection = nn.ModuleList(
                [
                    nn.Linear(
                        self.c_pair_cond,
                        self.super_block_size * self.num_head,
                        bias=False,
                    )
                    for _ in range(self.num_super_blocks)
                ]
            )

        self.self_attention = nn.ModuleList(
            [
                SelfAttention(
                    self.c_act,
                    self.c_single_cond,
                    use_single_cond=True,
                    kq_norm=spec.attention_kq_norm,
                    identity_scale=spec.adaptive_identity_scale,
                    gating_query=spec.token_attention_gating_query,
                    norm_eps=spec.adaptive_norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.transition_block = nn.ModuleList(
            [
                conditioned_transition(spec)(
                    self.c_act,
                    self.c_single_cond,
                    use_single_cond=True,
                    identity_scale=spec.adaptive_identity_scale,
                    norm_eps=spec.adaptive_norm_eps,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(
        self,
        act: torch.Tensor,
        mask: torch.Tensor,
        single_cond: torch.Tensor,
        pair_cond: torch.Tensor,
        extra_pair_bias: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the module output.

        ``key_mask`` is ``[1, N_token]`` bool: ``mask > 0`` made by the caller. The engine keys its
        hoisted pair bias on that tensor's address, so a caller that runs the sampler's steps
        passes one tensor for all of them; made here, it would be a new tensor at every call
        (and inside a compiled region, at every replay).

        ``extra_pair_bias`` is (N_token, N_token), added to every head's pair
        logits in every block. Only the structural-token family supplies one:
        its expander states the token relationships the pair itself cannot,
        having just been built rather than trained on this token set.
        """
        if self.per_block_pair:
            for idx in range(self.num_blocks):
                pair_logits = self.pair_logits_projection[idx](
                    self.pair_input_layer_norm[idx](pair_cond)
                ).permute(2, 0, 1)
                if extra_pair_bias is not None:
                    pair_logits = pair_logits + extra_pair_bias[None].to(
                        pair_logits.dtype
                    )
                if self.parallel:
                    act = (
                        act
                        + self.self_attention[idx](act, mask, pair_logits, single_cond)
                        + self.transition_block[idx](act, single_cond)
                    )
                    continue
                act = conditioned_residual(
                    act,
                    single_cond,
                    attention=lambda value, idx=idx, pair_logits=pair_logits: (
                        self.self_attention[idx](value, mask, pair_logits, single_cond)
                    ),
                    transition=self.transition_block[idx],
                )
            return act

        pair_act = self.pair_input_layer_norm(pair_cond)

        for super_block_i in range(self.num_super_blocks):
            pair_logits = self.pair_logits_projection[super_block_i](pair_act)
            pair_logits = einops.rearrange(
                pair_logits, "n s (b h) -> b h n s", h=self.num_head
            )
            if extra_pair_bias is not None:
                pair_logits = pair_logits + extra_pair_bias[None, None].to(
                    pair_logits.dtype
                )
            for j in range(self.super_block_size):
                idx = super_block_i * self.super_block_size + j
                act = conditioned_residual(
                    act,
                    single_cond,
                    attention=lambda value, idx=idx, j=j, pair_logits=pair_logits: (
                        self.self_attention[idx](
                            value, mask, pair_logits[j], single_cond
                        )
                    ),
                    transition=self.transition_block[idx],
                )

        return act


