"""``models.grouped``: the parts of the trunk_seed x diffusion_seed sampler that need no GPU (refusals and the conditioning table)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foldforge.models.grouped import GroupedSampler, GroupedUnsupported


def _stack(blocks: int = 2) -> SimpleNamespace:
    return SimpleNamespace(blocks=nn.ModuleList([nn.Identity() for _ in range(blocks)]))


def _head(**override):
    enc = SimpleNamespace(atom_transformer_encoder=_stack())
    dec = SimpleNamespace(atom_transformer_decoder=_stack())
    head = SimpleNamespace(
        transformer=_stack(3),
        atom_cross_att_encoder=enc,
        atom_cross_att_decoder=dec,
        _conditioned=lambda *a, **k: None,
        _noise_term=lambda levels: levels[:, None].expand(-1, 3) * 1.0,
        _finish_single_conditioning=lambda x: x,
    )
    for key, value in override.items():
        setattr(head, key, value)
    return head


def test_a_model_without_a_dense_diffusion_head_is_refused_before_anything_is_sampled():
    with pytest.raises(GroupedUnsupported, match="dense diffusion head"):
        GroupedSampler(SimpleNamespace())
    with pytest.raises(GroupedUnsupported, match="dense diffusion head"):
        GroupedSampler(SimpleNamespace(diffusion_head=SimpleNamespace()))


def test_a_transformer_that_is_not_a_stack_of_engine_blocks_is_refused():
    with pytest.raises(GroupedUnsupported, match="stack of engine blocks"):
        GroupedSampler(
            SimpleNamespace(diffusion_head=_head(transformer=SimpleNamespace()))
        )
    with pytest.raises(GroupedUnsupported, match="stack of engine blocks"):
        GroupedSampler(
            SimpleNamespace(
                diffusion_head=_head(
                    transformer=SimpleNamespace(blocks=nn.ModuleList())
                )
            )
        )


def test_the_blocks_of_the_three_stacks_are_taken_as_they_are():
    sampler = GroupedSampler(SimpleNamespace(diffusion_head=_head()))
    assert (len(sampler.built_tok), len(sampler.built_enc), len(sampler.built_dec)) == (
        3,
        2,
        2,
    )
    assert sampler.atoms_grouped


def test_the_conditioning_table_is_one_row_per_noise_level_per_fold():
    sampler = GroupedSampler(SimpleNamespace(diffusion_head=_head()))
    base = torch.zeros(4, 3)
    folds = [{"base": base}, {"base": base + 1}]
    sampler.state["cond"] = folds
    levels = torch.tensor([0.5, 1.5, 2.5])
    table = sampler._table(None, None, levels)  # noqa: SLF001
    assert table.shape == (3, 2, 4, 3), "[steps, folds, tokens, channels]"
    assert torch.allclose(table[:, 0], levels[:, None, None].expand(3, 4, 3))
    assert torch.allclose(table[:, 1], levels[:, None, None].expand(3, 4, 3) + 1)
    # one level (the per-step call a graph-less run makes) gives the same row as the full table
    assert torch.equal(sampler._table(None, None, levels[:1])[0], table[0])  # noqa: SLF001
