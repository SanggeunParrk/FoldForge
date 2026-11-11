"""The ported families' graph pieces: engine classes with FoldForge's layout adapters.

FoldForge's graph holds the miniworld-engine classes directly: a pairformer block IS
``miniworld_engine.modules.PairformerBlock``, a token / atom diffusion transformer is a list of ``TokenDiT*`` / ``AtomDiT*`` blocks, a
transition is ``Transition``. Their ``state_dict`` keys are the engine's. What is FoldForge's is the layout around them:

* the engine is batched (``[B, L, ...]`` activations, bool key masks ``[B, L]``, diffusion ``single`` / ``cond`` ``[A, B, L, d]``), the
  graph is unbatched (``[L, ...]`` activations, ``[L, L]`` pair masks, a leading sample axis ``[S, ...]`` where there are samples);
  :func:`run_block` and the stacks' ``forward`` add and remove the batch axis,
* the stacks (:class:`~foldforge.modules.engine.transformers.TokenDiTStack` / ``AtomDiTStack``) take the keyword arguments the dense
  transformers took, so the diffusion head and the atom encoder / decoder call either alike,
* :class:`EvoformerBlock` composes the engine's MSA-stack parts in the AF3 order (the engine has no MSA block).

Every family is ported. The released dense layout lives only in the offline converter (``foldforge.models.checkpoints.legacy``), which
builds its Haiku-blob graph by installing a :class:`DenseBlocks` factory for the duration of a conversion (:func:`use_dense_blocks`);
nothing here imports it, and with no factory installed (every runtime path) the graph is engine classes only.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Protocol

import torch
from miniworld_engine.modules import PairformerBlock
from torch import nn

from foldforge.modules.engine import modules as factories
from foldforge.modules.engine.configure import engine_activation
from foldforge.modules.engine.pairformer import pairformer_block
from foldforge.modules.engine.transformers import token_dit_stack

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldforge.modules.dense.spec import DenseSpec


class DenseBlocks(Protocol):
    """The released-layout blocks of the offline converter's graph (implemented in ``foldforge.models.checkpoints.legacy``)."""

    def pairformer_block(self, spec: DenseSpec, **options: object) -> nn.Module:
        """One dense pairformer block (``options`` are :func:`make_pairformer_block`'s resolved arguments)."""

    def evoformer_block(self, spec: DenseSpec, c_msa: int, c_pair: int, n_heads_pair: int) -> nn.Module:
        """One dense MSA-stack block."""

    def token_transformer(self, spec: DenseSpec, c_act: int, c_single_cond: int, c_pair_cond: int, num_blocks: int) -> nn.Module:
        """The dense token transformer."""

    def conditioning_transition(self, width: int) -> nn.Module:
        """The dense conditioning transition."""


_dense: DenseBlocks | None = None


@contextlib.contextmanager
def use_dense_blocks(blocks: DenseBlocks) -> Iterator[None]:
    """Build graphs from ``blocks`` (the converter's released layout) while active; never used at inference."""
    global _dense  # noqa: PLW0603 - the converter's one switch
    previous, _dense = _dense, blocks
    try:
        yield
    finally:
        _dense = previous


def uses_engine() -> bool:
    """Whether graphs are built from engine classes (always, except inside the converter's :func:`use_dense_blocks`)."""
    return _dense is None


# ------------------------------------------------------------------------------------------------------------------ chai1 / esmfold2 helpers
def chai1_adaln_eps(spec: DenseSpec) -> float | None:
    """Chai-1's DiT AdaLN epsilon (``spec.adaptive_norm_eps``: fast 1e-5, exact 0.1 = the release); ``None`` for every other family (their
    engine classes have no such argument)."""
    return spec.adaptive_norm_eps if spec.family == "chai1" else None


def has_engine_atom_dit(spec: DenseSpec) -> bool:
    """Whether the engine has an atom DiT class that serves ``spec``'s atom stack (not ESMFold2's: pair-free SWA vs atom pair + rope)."""
    return factories.model_keys(spec.family).atom_dit is not None


# ------------------------------------------------------------------------------------------------------------------ pairformer
def make_pairformer_block(
    spec: DenseSpec,
    *,
    d_pair: int,
    d_single: int = 0,
    n_head_pair: int | None = None,
    qkv_dim: int | None = None,
    transition_n: int | None = None,
    single_transition_n: int | None = None,
    n_head_single: int = 16,
    pair_attention: bool = True,
    confidence: bool = False,
    batched: str | None = None,
    tri_hidden: int | None = None,
) -> PairformerBlock:
    """One engine ``PairformerBlock`` of ``spec``'s family (parameters fp32 masters).

    ``d_single > 0``: the single track. ``qkv_dim``: the per-head width of the triangle attention (default ``d_pair / heads``).
    ``confidence``: the confidence head's block (``ModelKeys.confidence`` where the family has a different one). ``batched``:
    ``"sample"`` / ``"template"`` marks the block as one that takes a leading batch axis of samples / template slots in one call
    (:func:`run_block` always does; the head / the template embedder read the attribute)."""
    keys = factories.model_keys(spec.family)
    heads = n_head_pair or spec.pair_heads
    factor = spec.pairformer_transition_factor if transition_n is None else transition_n
    if _dense is not None:
        return _dense.pairformer_block(  # type: ignore[return-value]
            spec,
            d_pair=d_pair,
            d_single=d_single,
            n_head_pair=heads,
            qkv_dim=qkv_dim,
            transition_n=factor,
            single_transition_n=factor if single_transition_n is None else single_transition_n,
            n_head_single=n_head_single,
            pair_attention=pair_attention,
            confidence=confidence,
            tri_hidden=tri_hidden,
        )
    block = pairformer_block(
        d_pair=d_pair,
        triangle_model=(keys.confidence if confidence and keys.confidence else keys.triangle) or "af3",
        d_single=d_single,
        d_hidden_tri_multi=tri_hidden,  # the template stack of protenix1 widens only its multiplications (hidden 128 on pair 64)
        d_hidden_tri_attention=qkv_dim or d_pair // heads,
        n_head_tri_attention=heads,
        n_head_single=n_head_single,
        single_gate_bias=spec.single_attention_gate_bias,
        pair_transition_n=factor,
        single_transition_n=factor if single_transition_n is None else single_transition_n,
        use_triangle_attention=pair_attention,
        parallel=spec.parallel_pairformer_block,
        divide_by_length=spec.triangle_mul_divide_by_length,
        norm_eps=None,
    )
    if batched == "sample":
        block.supports_sample_batch = True
    elif batched == "template":
        block.supports_template_batch = True
    return block


def run_block(
    block: nn.Module,
    pair: torch.Tensor,
    pair_mask: torch.Tensor,
    single: torch.Tensor | None = None,
    seq_mask: torch.Tensor | None = None,
    extra_pair_bias: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One pairformer block on FoldForge's layout.

    ``pair`` ``[L, L, d]`` (or ``[S, L, L, d]``: S samples / template slots through one call), ``pair_mask`` the ``[L, L]`` outer-product
    mask, ``single`` ``[L, d]`` (``[S, L, d]``), ``seq_mask`` ``[L]``, ``extra_pair_bias`` ``[L, L]`` (added to every head of the single
    attention). Returns ``pair`` (a block without the single track) or ``(pair, single)``, in the layout it was given."""
    if not isinstance(block, PairformerBlock):  # the converter's released-layout block carries its own call (``run_dense``)
        return block.run_dense(pair, pair_mask, single, seq_mask, extra_pair_bias)  # type: ignore[operator]
    pair, single = engine_activation(pair), engine_activation(single)  # the engine's master blocks take bf16 activations, not fp32
    batched = pair.ndim == 4
    if not batched:
        pair = pair[None]
        single = None if single is None else single[None]
    count = pair.shape[0]
    key_mask = (seq_mask > 0)[None] if seq_mask is not None else factories.residue_mask(pair_mask)
    if count > 1:
        key_mask = key_mask.expand(count, -1).contiguous()  # the kernels read a dense mask, not a stride-0 view
    bias = None
    if extra_pair_bias is not None:
        heads = block.config.n_head_single
        bias = extra_pair_bias.to(pair.dtype)[None, None].expand(count, heads, -1, -1)
    if single is None:
        out = block(pair, key_mask)
        return out if batched else out[0]
    pair, single = block(pair, key_mask, single, bias) if bias is not None else block(pair, key_mask, single)
    return (pair, single) if batched else (pair[0], single[0])


# ------------------------------------------------------------------------------------------------------------------ MSA stack
class EvoformerBlock(nn.Module):
    """AF3's MSA-stack block from engine parts (the engine has no MSA block): outer product mean -> MSA pair-weighted averaging ->
    MSA transition (or the Boltz order, ``spec.msa_update_before_opm``), then the pair track (two triangle multiplications, two
    triangle attentions, the transition). Every part owns its residual except the outer product mean, whose raw update is added here.

    Same call as the dense block: ``(msa [M, L, d], pair [L, L, d], msa_mask [M, L], pair_mask [L, L]) -> (msa, pair)``."""

    def __init__(self, spec: DenseSpec, c_msa: int, c_pair: int, n_heads_pair: int) -> None:
        super().__init__()
        self.parallel = spec.parallel_msa_block  # Chai-1: every update of a stage reads the stage's input (chai1_* parallel stages)
        keys = factories.model_keys(spec.family)
        self.msa_first = spec.msa_update_before_opm
        self.outer_product_mean = factories.outer_product_mean(
            c_msa,
            c_pair,
            spec.opm_channel * spec.opm_groups,  # the engine's d_hidden is the TOTAL width (Chai-1: 8 groups of 8)
            groups=spec.opm_groups,
            projection_bias=spec.opm_projection_bias,
            normalize_before_proj=spec.opm_bias_after_norm,
            count_offset=None if spec.opm_clamped_norm or spec.opm_groups > 1 else factories.OPM_COUNT_OFFSET,  # grouped (Chai-1): no division
        )
        self.msa_attention1 = factories.msa_pair_weighted_averaging(c_msa, c_pair, 8, spec.msa_value_dim)
        self.msa_transition = factories.transition(c_msa, 4)
        self.triangle_multiplication_outgoing = factories.triangle_multiplication(
            c_pair, outgoing=True, divide_by_length=spec.triangle_mul_divide_by_length
        )
        self.triangle_multiplication_incoming = factories.triangle_multiplication(
            c_pair, outgoing=False, divide_by_length=spec.triangle_mul_divide_by_length
        )
        self.with_pair_attention = spec.pair_attention
        if self.with_pair_attention:
            self.pair_attention1 = factories.triangle_attention(c_pair, n_heads_pair, starting=True, model=keys.triangle)
            self.pair_attention2 = factories.triangle_attention(c_pair, n_heads_pair, starting=False, model=keys.triangle)
        self.pair_transition = factories.transition(c_pair, 4)

    def forward(
        self, msa: torch.Tensor, pair: torch.Tensor, msa_mask: torch.Tensor, pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One block (batch axis of 1 added for the engine and removed again)."""
        m, z = msa[None], pair[None]
        rows = (msa_mask > 0)[None]
        columns = (msa_mask.max(dim=0).values > 0)[None]  # a column is a key when any row has it
        tokens = factories.residue_mask(pair_mask)
        if self.parallel:
            return self._forward_parallel(m, z, rows, columns, tokens)
        if self.msa_first:
            m = self.msa_transition(self.msa_attention1(m, z, columns))
        z = z + self.outer_product_mean(m, rows)
        if not self.msa_first:
            m = self.msa_transition(self.msa_attention1(m, z, columns))
        z = self.triangle_multiplication_outgoing(z, tokens)
        z = self.triangle_multiplication_incoming(z, tokens)
        if self.with_pair_attention:
            z = self.pair_attention1(z, tokens)
            z = self.pair_attention2(z, tokens)
        z = self.pair_transition(z)
        return m[0], z[0]

    def _forward_parallel(
        self, m: torch.Tensor, z: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chai-1's block (``spec.parallel_msa_block``): the parts own their residuals, so a stage's update is the sum of the parts' deltas
        (``part(x) - x``) on the stage's input: MSA attention + MSA transition on the MSA entering the block (reading the pair after the
        outer product mean), the two triangle multiplications + the pair transition on the post-outer-product pair, then the two triangle
        attentions on that result."""
        if self.msa_first:
            raise ValueError("a parallel MSA block runs the outer product mean first")
        z = z + self.outer_product_mean(m, rows)
        m = m + (self.msa_attention1(m, z, columns) - m) + (self.msa_transition(m) - m)
        z = z + (
            (self.triangle_multiplication_outgoing(z, tokens) - z)
            + (self.triangle_multiplication_incoming(z, tokens) - z)
            + (self.pair_transition(z) - z)
        )
        if self.with_pair_attention:
            z = z + ((self.pair_attention1(z, tokens) - z) + (self.pair_attention2(z, tokens) - z))
        return m[0], z[0]


def make_evoformer_block(spec: DenseSpec, c_msa: int, c_pair: int, n_heads_pair: int) -> nn.Module:
    """One MSA-stack block of ``spec``'s family (:class:`EvoformerBlock`)."""
    if _dense is not None:
        return _dense.evoformer_block(spec, c_msa, c_pair, n_heads_pair)
    return EvoformerBlock(spec, c_msa, c_pair, n_heads_pair)


# ------------------------------------------------------------------------------------------------------------------ diffusion head
def make_token_transformer(spec: DenseSpec, *, c_act: int, d_cond: int, d_pair: int, num_blocks: int) -> nn.Module:
    """The token diffusion transformer of ``spec``'s family: ``TokenDiT*`` blocks (Chai-1: its AdaLN epsilon)."""
    if _dense is not None:
        return _dense.token_transformer(spec, c_act, d_cond, d_pair, num_blocks)
    return token_dit_stack(
        spec.family, n_blocks=num_blocks, d_single=c_act, d_cond=d_cond, d_pair=d_pair, n_head=16, adaln_eps=chai1_adaln_eps(spec)
    )


def make_conditioning_transition(width: int) -> nn.Module:
    """A conditioning transition of the diffusion head (no conditioning input, factor 2): the engine ``Transition``."""
    if _dense is not None:
        return _dense.conditioning_transition(width)
    return factories.transition(width, 2)
