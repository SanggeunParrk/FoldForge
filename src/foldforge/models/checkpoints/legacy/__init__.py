"""The released-layout dense graph, kept only for the offline checkpoint converter.

A released Haiku blob (``af3.bin.zst`` ...) is read into the DENSE released-layout graph in this package (:mod:`.haiku` maps the blob's
records onto its parameters), and :func:`foldforge.models.checkpoints.miniworld.convert_haiku` converts that graph's ``state_dict`` into the
engine-class graph's layout (the miniworld checkpoint format). Nothing at inference imports this package: ``load_model`` / ``predict`` /
training build engine-class graphs and read miniworld-format files only.

* :func:`dense_model` builds the dense graph of a family (the runtime ``AlphaFold3`` class with this package's blocks installed).
* The blocks themselves (``attention``, ``pairformer``, ``primitives``, ``triangle_multiplication``, ``transformer``) are the released AF3 modules;
  ``routing`` and ``reference`` bind their parameter layout to the reference equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from foldforge.modules.dense.spec import DenseSpec


class _DenseBlocks:
    """:class:`foldforge.modules.engine.graph.DenseBlocks` of the released layout."""

    def pairformer_block(self, spec: DenseSpec, **o: Any) -> nn.Module:
        from foldforge.models.checkpoints.legacy.pairformer import PairformerBlock

        return PairformerBlock(
            n_heads=o["n_head_single"],
            c_pair=o["d_pair"],
            c_single=o["d_single"] or 384,
            n_heads_pair=o["n_head_pair"],
            num_intermediate_factor=o["transition_n"],
            single_intermediate_factor=o["single_transition_n"],
            with_single=o["d_single"] > 0,
            with_pair_attention=o["pair_attention"],
            spec=spec,
            pair_qkv_dim=o["qkv_dim"],
            dual_output=bool(o["confidence"] and spec.confidence_dual_output),
            tri_hidden_dim=o["tri_hidden"],
        )

    def evoformer_block(self, spec: DenseSpec, c_msa: int, c_pair: int, n_heads_pair: int) -> nn.Module:
        from foldforge.models.checkpoints.legacy.pairformer import EvoformerBlock

        return EvoformerBlock(c_msa=c_msa, c_pair=c_pair, n_heads_pair=n_heads_pair, spec=spec)

    def token_transformer(self, spec: DenseSpec, c_act: int, c_single_cond: int, c_pair_cond: int, num_blocks: int) -> nn.Module:
        from foldforge.models.checkpoints.legacy.transformer import DiffusionTransformer

        return DiffusionTransformer(c_act=c_act, c_single_cond=c_single_cond, c_pair_cond=c_pair_cond, num_blocks=num_blocks, spec=spec)

    def conditioning_transition(self, width: int) -> nn.Module:
        from foldforge.modules.dense.diffusion_transformer import DiffusionTransition

        return DiffusionTransition(width, c_single_cond=None)


def dense_model(spec: DenseSpec, **kwargs: Any) -> nn.Module:
    """The dense released-layout ``AlphaFold3`` graph of ``spec`` (parameters named as the Haiku blob's records)."""
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.modules.engine import graph as engine_graph

    with engine_graph.use_dense_blocks(_DenseBlocks()):
        return AlphaFold3(spec=spec, **kwargs)
