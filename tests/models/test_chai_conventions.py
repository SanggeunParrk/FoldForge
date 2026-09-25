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


def test_relative_encoding_tells_chains_and_entities_apart():
    """A pair across chains must not read as one inside a chain.

    Folded into the bias at their single-chain values, Chai-1 saw every
    protein-ligand pair as intra-chain.
    """
    from types import SimpleNamespace

    import torch

    from foldforge.modules.dense.featurization import chai_relative_encoding

    # Two copies of one protein (entity 1) and a ligand (entity 2).
    tokens = SimpleNamespace(
        residue_index=torch.tensor([1, 2, 1, 2, 1]),
        token_index=torch.arange(5),
        asym_id=torch.tensor([1, 1, 2, 2, 3]),
        entity_id=torch.tensor([1, 1, 1, 1, 2]),
        sym_id=torch.tensor([1, 1, 2, 2, 1]),
    )
    feat = chai_relative_encoding(tokens, torch.float32)
    assert feat.shape == (5, 5, SPECS["chai1"].relpos_channel)
    chain = feat[..., 134:140].argmax(-1)
    entity = feat[..., 140:143].argmax(-1)
    assert chain[0, 1] == 2  # same chain
    assert chain[0, 2] == 1  # the next copy of the same entity
    assert chain[0, 4] == 5  # across entities
    assert entity[0, 1] == 1
    assert (entity[0, 4], entity[4, 0]) == (0, 2)


def test_only_protein_chains_reach_the_language_model():
    """A ligand's atoms are not a protein; the release gives them zero rows."""
    import torch

    from foldforge.modules.language_model import embed_chains

    seen = []

    def tower(tokens: torch.Tensor) -> torch.Tensor:
        seen.append(tokens.shape[1])
        return torch.ones(1, tokens.shape[1], 3)

    out = embed_chains(
        tower,
        aatype=torch.tensor([0, 1, 2, 20, 20]),
        asym_id=torch.tensor([1, 1, 1, 2, 2]),
        mask=torch.ones(5),
        is_protein=torch.tensor([1, 1, 1, 0, 0]),
    )
    assert seen == [5]  # the protein chain, wrapped; the ligand never ran
    assert out[:3].eq(1).all()
    assert out[3:].eq(0).all()
