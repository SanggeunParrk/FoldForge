# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""ESMFold2 ``SWAAtomTransformer`` (Algorithm 8) — DiT adaLN-Zero blocks with
sliding-window 3D-RoPE attention and no atom-pair tensor.

This module adapts sequence-layout checkpoint weights to the shared engine
attention, normalization/modulation and SwiGLU operations.

Each block is a canonical DiT adaLN-Zero block (Peebles & Xie, arXiv:2212.09748):
the conditioning predicts ``shift / scale / gate`` for both the attention and the
FFN sub-layer; the residual branches are gated by a zero-initialised modulation
so they start as the identity. This is intentionally *not* the AF3 conditioning
(``sigmoid``-scale AdaLN + ``bias=-2`` output gate) used by
``miniworld_engine.modules.AugmentedAttentionPairBias``; it mirrors ESMFold2,
which reverts the atom transformer to the original DiT block while keeping AF3's
``AttentionPairBias`` only at the token level.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from miniworld_engine.modules.swa_atom_attention import (
    SWA3DRoPEAttention,
    build_3d_rope,
)
from pydantic import BaseModel
from team_gm import typecheck
from team_gm.modules._engine_impl import to_engine_impl
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.layers.ops import mp_sum, mp_swish_gate
from team_gm.modules.primitives import Linear, MPLinear


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with hidden size rounded up to a multiple of 256."""

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int = 2,
        *,
        magnitude_preserving: bool = False,
        implementation: ImplementationType = ImplementationType.PYTORCH,
    ) -> None:
        super().__init__()
        hidden = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.implementation = ImplementationType(implementation)
        self.magnitude_preserving = magnitude_preserving
        dense = MPLinear if magnitude_preserving else Linear
        self.w_up = dense(d_model, 2 * hidden, bias=False, init="normal")
        self.w_down = dense(hidden, d_model, bias=False, init="normal")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        if (
            self.implementation == ImplementationType.MINIWORLD_ENGINE
            and not self.training
            and not self.magnitude_preserving
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.w_up.weight.dtype == x.dtype
        ):
            from miniworld_engine import ops

            wa, wb = self.w_up.weight.chunk(2, dim=0)
            return ops.swiglu_ffn(x, wa, wb, self.w_down.weight)
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        x = mp_swish_gate(x1, x2) if self.magnitude_preserving else F.silu(x1) * x2
        return self.w_down(x)


class SWAAtomBlock(nn.Module):
    """adaLN-Zero + sliding-window 3D-RoPE attention + SwiGLU FFN."""

    def __init__(
        self,
        d_atom: int,
        d_cond: int,
        n_head: int,
        *,
        half_window: int = 64,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
        expansion_ratio: int = 2,
        magnitude_preserving: bool = False,
        mp_full: bool = False,
        mp_residual: bool = False,
        residual_t: float = 0.3,
    ) -> None:
        super().__init__()
        magnitude_preserving = magnitude_preserving or mp_full
        self.implementation = ImplementationType(implementation)
        self.mp_full = mp_full
        self.mp_residual = mp_residual or mp_full
        self.residual_t = residual_t
        self.attn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        self.ffn_norm = nn.RMSNorm(d_atom, elementwise_affine=False)
        # Plain ESMFold2 uses adaLN-Zero. Under full MP, this modulation is an
        # active MP projection; residual magnitude is controlled by mp_sum.
        modulation = (
            MPLinear(d_cond, 6 * d_atom, bias=False, init="normal")
            if mp_full
            else Linear(d_cond, 6 * d_atom, bias=False, init="zero")
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            modulation,
        )
        self.attn = SWA3DRoPEAttention(
            d_atom,
            n_head,
            half_window=half_window,
            implementation=to_engine_impl(
                ImplementationType.PYTORCH
                if implementation == ImplementationType.CUEQUIVARIANCE
                else implementation
            ),
            magnitude_preserving=magnitude_preserving,
            mp_full=mp_full,
        )
        self.ffn = SwiGLUFFN(
            d_atom,
            expansion_ratio,
            magnitude_preserving=magnitude_preserving,
            implementation=implementation,
        )

    def forward(
        self,
        q: Float[torch.Tensor, "N S d_atom"],
        c: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        """Compute the module output."""
        # Sequential erases the concrete type of its children; construction
        # guarantees that the modulation projection is Linear or its MP subclass.
        modulation = cast("nn.Linear", self.adaln_modulation[1])
        if (
            self.implementation == ImplementationType.MINIWORLD_ENGINE
            and not self.training
            and not self.mp_full
            and q.is_cuda
            and q.dtype == c.dtype == torch.bfloat16
            and modulation.weight.dtype == q.dtype
        ):
            from miniworld_engine import ops

            activated = F.silu(c)
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
                modulation.weight.chunk(6, dim=0)
            )
            # PyTorch 2.10 RMSNorm uses the accumulation dtype epsilon when eps
            # is omitted: BF16 accumulates in FP32 (also under torch.compile).
            eps_a = self.attn_norm.eps
            eps_f = self.ffn_norm.eps
            attn_in, gate = ops.rms_norm_modulation(
                q,
                activated,
                scale_a,
                shift_a,
                gate_a,
                self.attn_norm.weight,
                torch.finfo(torch.float32).eps if eps_a is None else eps_a,
            )
            attn = gate * self.attn(attn_in, attention_params)
            q = mp_sum(q, attn, self.residual_t) if self.mp_residual else q + attn
            ffn_in, gate = ops.rms_norm_modulation(
                q,
                activated,
                scale_f,
                shift_f,
                gate_f,
                self.ffn_norm.weight,
                torch.finfo(torch.float32).eps if eps_f is None else eps_f,
            )
            ffn = gate * self.ffn(ffn_in)
            return mp_sum(q, ffn, self.residual_t) if self.mp_residual else q + ffn
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_modulation(
            c
        ).chunk(6, dim=-1)
        attn_in = self.attn_norm(q) * (1 + scale_a) + shift_a
        attn = gate_a * self.attn(attn_in, attention_params)
        q = mp_sum(q, attn, self.residual_t) if self.mp_residual else q + attn
        ffn_in = self.ffn_norm(q) * (1 + scale_f) + shift_f
        ffn = gate_f * self.ffn(ffn_in)
        q = mp_sum(q, ffn, self.residual_t) if self.mp_residual else q + ffn
        return q


class SWAAtomTransformer(nn.Module):
    """Stack of :class:`SWAAtomBlock` (ESMFold2 Algorithm 8)."""

    class Config(BaseModel):
        """Configuration for the SWA atom transformer."""

        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE
        d_atom: int = 128
        d_cond: int = 128
        n_block: int = 3
        n_head: int = 4
        swa_window_size: int = 128
        expansion_ratio: int = 2
        # "esmfold2": ESMFold2 SWAAtomBlock (adaLN-Zero + SwiGLU).
        # "af3": keep MiniWorld's AF3 atom block (AdaLN + sigmoid gate + to_scale
        #   + ConditionedTransition), swap ONLY the attention core to 3D-RoPE
        #   sliding-window (no atom pair). See RoPESWAAF3Transformer.
        block_style: str = "esmfold2"
        # global_attn: with block_style "af3", use GLOBAL self-attention (all
        # atoms attend to all) instead of the sliding window — keeps 3D RoPE and
        # no atom pair. O(L_atom^2); for training-crop-size experiments only.
        global_attn: bool = False
        # local_structure_attn: with block_style "af3", attend to explicit
        # neighbors = nearest atoms in sequence order plus nearest atoms by the
        # current noisy coordinates x_t, excluding sequence neighbors. This uses
        # a gather-based sparse attention path because FA window kernels cannot
        # express arbitrary dynamic neighbors.
        local_structure_attn: bool = False
        seq_neighbors: int = 128
        structure_neighbors: int = 128
        structure_query_chunk_size: int = 128
        sparse_attention_query_chunk_size: int = 64
        n_spatial_rope_pairs_per_axis: int = 2
        spatial_rope_base_frequency: float = 20.0
        n_uid_rope_pairs: int = 10
        uid_rope_base_frequency: float = 10000.0
        magnitude_preserving: bool = False
        use_rotation: bool = False
        mp_full: bool = False
        mp_residual: bool = False
        residual_t: float = 0.3

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_atom // config.n_head
        self.blocks = nn.ModuleList(
            [
                SWAAtomBlock(
                    config.d_atom,
                    config.d_cond,
                    config.n_head,
                    half_window=config.swa_window_size // 2,
                    implementation=config.implementation,
                    expansion_ratio=config.expansion_ratio,
                    magnitude_preserving=config.magnitude_preserving,
                    mp_full=config.mp_full,
                    mp_residual=config.mp_residual,
                    residual_t=config.residual_t,
                )
                for _ in range(config.n_block)
            ],
        )

    @typecheck
    def build_rope(
        self,
        ref_pos: Float[torch.Tensor, "B N 3"],
        ref_space_uid: Int[torch.Tensor, "B N"],
    ) -> tuple[Float[torch.Tensor, "B N half"], Float[torch.Tensor, "B N half"]]:
        """Build the (cos, sin) 3D RoPE tensors for one batch (no augment axis)."""
        return build_3d_rope(
            ref_pos,
            ref_space_uid,
            self.head_dim,
            n_spatial_per_axis=self.config.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.config.n_uid_rope_pairs,
            spatial_base_freq=self.config.spatial_rope_base_frequency,
            uid_base_freq=self.config.uid_rope_base_frequency,
        )

    @typecheck
    def forward(
        self,
        q: Float[torch.Tensor, "N S d_atom"],
        c: Float[torch.Tensor, "N S d_cond"],
        attention_params: tuple,
    ) -> Float[torch.Tensor, "N S d_atom"]:
        """Forward pass over the flattened ``N = A * B`` atom sequence."""
        for block in self.blocks:
            q = block(q, c, attention_params)
        return q
