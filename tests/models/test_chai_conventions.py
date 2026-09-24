"""Conventions Chai-1 needs that no parameter records, so no tree diff can see."""

from __future__ import annotations

import torch
from torch import nn

from foldforge.data.features.dense import MSA
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


def _msa(rows: torch.Tensor, deletions: torch.Tensor | None = None) -> MSA:
    depth, length = rows.shape
    return MSA(
        rows=rows,
        mask=torch.ones(depth, length),
        deletion_matrix=torch.zeros(depth, length) if deletions is None else deletions,
        profile=torch.rand(length, 31),
        deletion_mean=torch.rand(length),
        num_alignments=torch.tensor(depth),
    )


def test_query_alone_is_no_alignment_however_often_it_repeats():
    query = torch.tensor([3, 1, 4, 1, 5])
    assert not _msa(query[None]).has_alignment()
    assert not _msa(torch.stack([query, query])).has_alignment()

    other = query.clone()
    other[2] = 0
    assert _msa(torch.stack([query, other])).has_alignment()

    deletions = torch.zeros(2, 5)
    deletions[1, 0] = 1.0
    assert _msa(torch.stack([query, query]), deletions).has_alignment()


def test_without_alignment_empties_the_statistics():
    """A profile of the query alone is a one-hot of the sequence, not nothing."""
    msa = _msa(torch.tensor([[3, 1, 4], [3, 1, 4]])).without_alignment()

    assert not msa.mask.any()
    assert not msa.profile.any()
    assert not msa.deletion_mean.any()
    assert not msa.has_alignment()
