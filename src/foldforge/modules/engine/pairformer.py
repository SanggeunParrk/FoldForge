"""The engine ``PairformerBlock`` as FoldForge builds it: one explicit constructor per family, the stack assembled here.

The engine's ``PairformerConfig.model`` is ONE key for both the triangle attention and the triangle multiplication of a block. That is
right for every family but RoseTTAFold3's fast mode, which needs the RF3 triangle attention and the PLAIN triangle multiplication (the
RF3 class divides the contraction by the length); :func:`pairformer_block` builds the block the engine's docstring describes (the
engine class, then its two multiplications replaced) instead of asking the engine for a mixed key.
"""

from __future__ import annotations

from miniworld_engine.modules import PairformerBlock, PairformerConfig

from foldforge.modules.engine.modules import Precision, finish, triangle_multiplication


def pairformer_block(
    *,
    d_pair: int,
    triangle_model: str,
    d_single: int = 0,
    d_hidden_tri_multi: int | None = None,
    d_hidden_tri_attention: int = 32,
    n_head_tri_attention: int = 4,
    n_head_single: int = 16,
    single_gate_bias: float = 0.0,
    pair_transition_n: int = 4,
    single_transition_n: int = 4,
    use_triangle_attention: bool = True,
    parallel: bool = False,
    divide_by_length: bool = False,
    precision: Precision = "master",
    norm_eps: float | None = None,
) -> PairformerBlock:
    """One block (residuals included). ``triangle_model`` is a ``ModelKeys.triangle`` (or ``confidence``) key. ``parallel``: Chai-1 /
    RoseTTAFold3's schedule. ``divide_by_length``: RoseTTAFold3's EXACT mode (RF3 triangle multiplication); fast mode leaves it False
    and gets the plain one. ``d_single > 0`` adds the single track (``AttentionPairBias`` + a single ``Transition``)."""
    rf3_plain = triangle_model == "rf3" and not divide_by_length
    config = PairformerConfig(
        d_pair=d_pair,
        d_hidden_tri_multi=d_pair if d_hidden_tri_multi is None else d_hidden_tri_multi,
        d_hidden_tri_attention=d_hidden_tri_attention,
        n_head_tri_attention=n_head_tri_attention,
        p_drop=0.0,
        use_triangle_attention=use_triangle_attention,
        model=triangle_model,
        d_single=d_single,
        n_head_single=n_head_single,
        single_gate_bias=single_gate_bias,
        pair_transition_n=pair_transition_n,
        single_transition_n=single_transition_n,
        parallel=parallel,
    )
    block = PairformerBlock(config)
    if rf3_plain:
        hidden = config.d_hidden_tri_multi
        block.tri_multi_outgoing = triangle_multiplication(d_pair, outgoing=True, d_hidden=hidden, precision=precision)
        block.tri_multi_incoming = triangle_multiplication(d_pair, outgoing=False, d_hidden=hidden, precision=precision)
    return finish(block, precision, norm_eps=norm_eps)
