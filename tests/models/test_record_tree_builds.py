"""Every family's parameter tree must BUILD, without a checkpoint to read.

The converter reads attributes off the graph it is handed. Rename or drop one
and nothing fails until a blob is loaded -- which needs weights, a GPU-sized
machine and minutes, so it runs far from the edit that broke it. Building each
family's translation dict on the meta device costs neither and catches exactly
that: a stale attribute, a record with no module behind it, a family whose
optional stage the builder still reaches for unconditionally.

It does NOT check that the names or shapes match a released checkpoint. That is
`scripts/diff_dense_checkpoint.py`, and it needs the weights.
"""

from __future__ import annotations

import pytest
import torch

from foldforge.modules.dense.spec import SPECS


def _translation(family: str) -> dict:
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.models.checkpoints import haiku

    with torch.device("meta"):
        model = AlphaFold3(spec=SPECS[family])
    return haiku.get_translation_dict(model)


@pytest.mark.parametrize("family", sorted(SPECS))
def test_every_family_builds_its_record_tree(family):
    records = _translation(family)
    assert records, f"{family} produced no records"
    # Flattening is what the loader does, and it is where a malformed subtree
    # surfaces rather than at construction.
    flat = haiku_flatten(records)
    assert all(isinstance(key, str) for key in flat)
    assert len(flat) == len(set(flat)), f"{family} names a record twice"


def haiku_flatten(records: dict) -> list[str]:
    """Return every record path the loader would ask the checkpoint for."""
    from foldforge.models.checkpoints import haiku

    return list(haiku._process_translations_dict(d=records, _key_prefix=""))  # noqa: SLF001 - the loader's own walk


def test_a_widened_family_names_the_same_records():
    """IntelliFold-v2's port is a widening, so its tree is AF3's tree.

    Not its SPEC, though -- it also carries featurisation conventions (its key
    window, its dropped atoms, its self-MSA). Those carry no parameters, which
    is exactly why the two trees agree while the two rows do not, and why a
    convention cannot be checked by a diff.
    """
    assert set(haiku_flatten(_translation("alphafold3"))) == set(
        haiku_flatten(_translation("intellifold2"))
    )


def test_a_family_with_no_msa_stack_names_no_msa_records():
    fast = set(haiku_flatten(_translation("esmfold2-fast")))
    full = set(haiku_flatten(_translation("esmfold2")))
    assert not any("msa_stack" in name for name in fast)
    assert any("msa_stack" in name for name in full)
    # Dropping the stack RENUMBERS what follows it, so the trunk moves.
    assert any("__layer_stack_no_per_layer/trunk_pairformer" in n for n in fast)
    assert any("__layer_stack_no_per_layer_1/trunk_pairformer" in n for n in full)
