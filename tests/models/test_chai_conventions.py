"""Conventions Chai-1 needs that no parameter records, so no tree diff can see."""

from __future__ import annotations

from torch import nn

from foldforge.modules.dense.pairformer import PairformerBlock
from foldforge.modules.dense.spec import SPECS


def _loaded(block: nn.Module) -> nn.Module:
    holder = nn.Module()
    holder.block = block
    holder.foldforge_load_report = {}
    return holder


def test_install_keeps_a_block_with_its_own_schedule():
    """The shared block is sequential; a parallel one must keep its forward.

    Swapping it ran Chai-1's 48 trunk blocks under the wrong composition for
    every real fold while a harness that skipped the loader matched the
    release, and cost 18 pLDDT points on 5I28.
    """
    from team_gm.modules.checkpoints.pairformer import install_pairformers

    kwargs = {"c_pair": 16, "c_single": 32, "n_heads": 2, "n_heads_pair": 2}
    parallel = _loaded(PairformerBlock(spec=SPECS["chai1"], **kwargs).eval())
    sequential = _loaded(PairformerBlock(spec=SPECS["alphafold3"], **kwargs).eval())

    install_pairformers(parallel)
    install_pairformers(sequential)

    assert type(parallel.block) is PairformerBlock
    assert parallel.foldforge_load_report["team_gm_pairformer_blocks"] == 0
    assert type(sequential.block) is not PairformerBlock
    assert sequential.foldforge_load_report["team_gm_pairformer_blocks"] == 1
