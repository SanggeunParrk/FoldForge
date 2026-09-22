"""One AF3-style MSA policy applied to every predictor."""

import pytest
import torch

from foldforge.models import msa_policy
from foldforge.models.config import Config, configuration
from foldforge.models.config.sequence import ESMFold2Config
from foldforge.models.io.request import Request
from foldforge.modules.sequence.pair_trunk import sample_msa_rows

GAP = pytest.importorskip("esm.models.esmfold2.constants").MSA_GAP_TOKEN_ID


def test_policy_record_matches_af3_pipeline():
    assert msa_policy.RECORD == {
        "name": "af3-msa-v1",
        "prepared_rows": 16384,
        "sampled_rows_per_recycle": 1024,
        "templates_per_chain": 4,
    }


@pytest.mark.parametrize(
    ("name", "variant"), [("opendde", None), ("protenix", "protenix-v2")]
)
def test_flat_configs_sample_1024_rows_per_pass(name, variant):
    configs = configuration(name, variant)
    released = configs.data["msa"].get("msa_depth")
    assert released in {None, 1280}
    msa_policy.apply_flat(configs)
    assert configs.data["msa"]["msa_depth"] == 1024


def test_esmfold2_config_resamples_per_loop():
    configs = ESMFold2Config()
    assert configs.msa_rows_per_loop is None
    msa_policy.apply_esmfold2(configs)
    assert configs.msa_rows_per_loop == 1024


def test_prepared_rows_default_everywhere(tmp_path):
    assert Config().trunk.msa_depth is None
    assert Request(model="esmfold2", ccd_db=tmp_path).msa_depth == 16384


def _features(
    valid_rows: int, gap_rows: int, pad_rows: int, length: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = valid_rows + gap_rows + pad_rows
    msa = torch.randint(0, 33, (1, rows, length))
    msa[msa == GAP] = (GAP + 1) % 33
    msa[:, valid_rows : valid_rows + gap_rows] = GAP
    one_hot = torch.nn.functional.one_hot(msa, num_classes=33).float()
    mask = torch.ones(1, rows, length, dtype=torch.bool)
    mask[:, valid_rows + gap_rows :] = False
    one_hot = one_hot * mask.unsqueeze(-1)
    features = torch.cat([one_hot, torch.zeros(1, rows, length, 2)], dim=-1)
    return features, mask


def test_sample_msa_rows_prefers_valid_rows_and_redraws():
    features, mask = _features(valid_rows=40, gap_rows=5, pad_rows=15)
    generator = torch.Generator().manual_seed(3)
    picked, picked_mask = sample_msa_rows(features, mask, 32, generator)
    assert picked.shape == (1, 32, 6, 35)
    assert picked_mask.shape == (1, 32, 6)
    # Every selected row is a real, non-gap row when enough of them exist.
    assert picked_mask.all()
    assert (picked[..., GAP] == 0).all()
    again, _ = sample_msa_rows(features, mask, 32, generator)
    assert not torch.equal(picked, again)
    replay, _ = sample_msa_rows(features, mask, 32, torch.Generator().manual_seed(3))
    assert torch.equal(picked, replay)


def test_sample_msa_rows_falls_back_to_invalid_rows_only_when_short():
    features, mask = _features(valid_rows=10, gap_rows=4, pad_rows=6)
    picked, picked_mask = sample_msa_rows(features, mask, 12, None)
    assert picked.shape[1] == 12
    # Ten valid rows lead; the remainder are invalid rows (all-gap or padding),
    # which the AF3 rule ranks equally.
    assert picked_mask[:, :10].all()
    assert (picked[:, :10, :, GAP] == 0).all()
    all_gap = (picked[:, 10:, :, GAP] == 1).all(dim=-1)
    padding = ~picked_mask[:, 10:].any(dim=-1)
    assert (all_gap | padding).all()
    picked, _ = sample_msa_rows(features, mask, 64, None)
    assert picked.shape[1] == 20
