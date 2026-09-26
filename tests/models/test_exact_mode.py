"""The exact mode: each release's conventions that the fast mode unifies to AF3's.

The fast mode is what these conventions measured as (the release's own fold
spread absorbs every one); the exact mode runs them, so a fold can be compared
with its release computation for computation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foldforge.data.features.dense import MSA
from foldforge.modules.dense.spec import (
    ALPHAFOLD3,
    EXACT_CONVENTIONS,
    MODES,
    SPECS,
    spec_for,
)


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


def test_fast_is_the_registered_spec_and_exact_adds_only_its_conventions():
    assert MODES == ("fast", "exact")
    for family, spec in SPECS.items():
        assert spec_for(family, "fast") is spec
        exact = spec_for(family, "exact")
        for name, value in EXACT_CONVENTIONS.get(family, {}).items():
            assert getattr(exact, name) == value
            assert getattr(ALPHAFOLD3, name) != value or name == "drop_atoms"
    # AF3 is its own release: nothing to add.
    assert spec_for("alphafold3", "exact") == ALPHAFOLD3
    with pytest.raises(ValueError, match="unknown mode"):
        spec_for("boltz2", "faithful")


def test_exact_conventions_are_the_releases_own():
    """Spot values read off each release (see EXACT_CONVENTIONS)."""
    assert spec_for("chai1", "exact").atom_key_window == "circular"
    assert spec_for("chai1", "exact").adaptive_norm_eps == 0.1
    assert spec_for("boltz2", "exact").template_visibility_by_coverage
    assert spec_for("rosettafold3", "exact").triangle_mul_divide_by_length
    assert spec_for("protenix2", "exact").key_masked_atom_attention
    assert spec_for("intellifold2", "exact").atom_key_window == "slide_qblock"
    # The fast mode runs AF3's computation for all of them.
    assert SPECS["chai1"].atom_key_window == "slide"


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


def test_length_divided_triangle_equals_scaling_the_norm_epsilon():
    from team_gm.modules.checkpoints.af_family import triangle_residual

    from foldforge.modules.dense.triangle_multiplication import TriangleMultiplication

    torch.manual_seed(0)
    plain = TriangleMultiplication(c_pair=8)
    divided = TriangleMultiplication(c_pair=8, divide_by_length=True)
    divided.load_state_dict(plain.state_dict())
    pair, mask = 1e-3 * torch.randn(6, 6, 8), torch.ones(6, 6)
    plain.center_norm.eps = plain.center_norm.eps * 36
    torch.testing.assert_close(
        triangle_residual(divided, pair, mask), triangle_residual(plain, pair, mask)
    )
    assert divided.center_norm.eps == 1e-5


def test_the_outer_product_divisor_and_its_bias_placement_are_stated_apart():
    """One family takes the clamped divisor without the bias placement."""
    from foldforge.modules.dense import primitives

    torch.manual_seed(0)
    msa, mask = torch.randn(2, 5, 8), torch.ones(2, 5)

    def parts(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        x, m = module.layer_norm_input(msa), mask.unsqueeze(-1)
        outer = torch.einsum(
            "acb,ade->dceb",
            (m * module.left_projection(x)).permute(0, 2, 1),
            m * module.right_projection(x),
        )
        return (
            torch.einsum("dceb,cef->dbf", outer, module.output_w),
            torch.einsum("abc,adc->bdc", m, m),
        )

    def build(**kw: bool) -> primitives.OuterProductMean:
        return primitives.OuterProductMean(
            c_msa=8, num_output_channel=4, num_outer_channel=3, **kw
        )

    af3 = build()
    out, count = parts(af3)
    expected = (out + af3.output_b).permute(1, 0, 2) / (af3.epsilon + count)
    assert torch.allclose(af3(msa, mask), expected, atol=1e-6)

    clamped = build(clamped_norm=True)
    out, count = parts(clamped)
    expected = (out + clamped.output_b).permute(1, 0, 2) / count.clamp_min(1.0)
    assert torch.allclose(clamped(msa, mask), expected, atol=1e-6)

    both = build(bias_after_norm=True, clamped_norm=True)
    out, count = parts(both)
    expected = out.permute(1, 0, 2) / count.clamp_min(1.0) + both.output_b
    assert torch.allclose(both(msa, mask), expected, atol=1e-6)

    assert spec_for("esmfold2", "exact").opm_clamped_norm
    assert not spec_for("esmfold2", "exact").opm_bias_after_norm


@pytest.mark.parametrize("policy", ["pad", "slide_qblock", "circular"])
def test_key_window_policies_differ_from_af3_only_at_the_ends(policy):
    """A block wholly inside the atoms gets AF3's window under every policy."""
    from foldforge.data.features.dense_conventions import key_window

    subsets, queries, keys, atoms = 6, 32, 128, 150
    flat = np.arange(subsets * queries)
    mask = (flat < atoms).reshape(subsets, queries)
    starts = np.arange(subsets) * queries + (queries - keys) // 2
    af3 = np.clip(starts, 0, atoms - keys)[:, None] + np.arange(keys)[None]
    example = {
        "token_atoms_to_queries:gather_mask": mask,
        "queries_to_keys:gather_idxs": af3.copy(),
        "queries_to_keys:gather_mask": np.ones_like(af3, dtype=bool),
        "tokens_to_queries:gather_idxs": (flat // 4).reshape(subsets, queries),
        "tokens_to_queries:gather_mask": mask,
        "tokens_to_keys:gather_idxs": af3 // 4,
        "tokens_to_keys:gather_mask": np.ones_like(af3, dtype=bool),
    }
    key_window(example, policy)
    window = example["queries_to_keys:gather_idxs"]
    middle = 2  # starts at 64 - 48 = 16, ends at 144: inside 150 atoms
    np.testing.assert_array_equal(window[middle], af3[middle])
    # ...and somewhere at an end, its own.
    assert not (
        np.array_equal(window[0], af3[0]) and np.array_equal(window[-1], af3[-1])
    )
