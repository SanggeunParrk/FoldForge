"""One AF3-style MSA policy applied to every predictor."""

import pytest

from foldforge.models import msa_policy, registered_models
from foldforge.models.config import Config
from foldforge.models.io.request import Request
from foldforge.modules.dense.spec import SPECS


def test_policy_record_matches_af3_pipeline():
    assert msa_policy.RECORD == {
        "name": "af3-msa-v1",
        "prepared_rows": 16384,
        "sampled_rows_per_recycle": 1024,
        "templates_per_chain": 4,
    }


def test_every_family_states_its_subsampling_rule():
    """The released adapters disagreed; each disagreement is one spec field.

    ``shuffle`` is the AF3 rule. Chai-1 takes the prepared rows in order and
    ESMFold2 keeps the query row and samples the rest, so a checkpoint that
    was trained under its own rule still reads its own rule here -- from the
    family row, never from a branch on the family's name.
    """
    assert {spec.msa_subsample for spec in SPECS.values()} == {
        "shuffle",
        "ordered",
        "keep_query",
    }
    assert SPECS["alphafold3"].msa_subsample == "shuffle"
    assert SPECS["chai1"].msa_subsample == "ordered"
    assert SPECS["esmfold2"].msa_subsample == "keep_query"


def test_a_family_without_an_msa_stack_still_carries_the_rule():
    """``esmfold2-fast`` drops the stack; the row stays truthful about the rule."""
    assert SPECS["esmfold2-fast"].msa_layers == 0
    assert SPECS["esmfold2-fast"].msa_subsample == SPECS["esmfold2"].msa_subsample


@pytest.mark.parametrize("model", sorted(registered_models()))
def test_prepared_rows_default_everywhere(model, tmp_path):
    """No family carries its own prepared depth; every request starts at 16384."""
    assert Config().trunk.msa_depth is None
    request = Request(
        model=model,
        ccd_db=tmp_path,
        input=tmp_path / "input.json",
        checkpoint=tmp_path / "weights.pt",
        recycles=10,
        steps=200,
    )
    assert request.msa_depth == msa_policy.PREPARED_ROWS == 16384
