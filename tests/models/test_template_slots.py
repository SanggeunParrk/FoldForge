"""An absent template slot still contributes when the vendor runs it."""

from __future__ import annotations

import torch

from foldforge.data.features.dense import Templates
from foldforge.data.features.dense_conventions import _GAP_RESTYPE
from foldforge.modules.dense.fused_template import FusedTemplateEmbedding
from foldforge.modules.dense.spec import SPECS


def _empty(slots: int, tokens: int) -> Templates:
    return Templates(
        aatype=torch.full((slots, tokens), _GAP_RESTYPE, dtype=torch.int64),
        atom_positions=torch.zeros(slots, tokens, 24, 3),
        atom_mask=torch.zeros(slots, tokens, 24, dtype=torch.bool),
    )


def test_absent_slots_are_averaged_not_skipped():
    """Protenix runs every slot and divides by the slot count.

    With no template at all the normed query pair still drives the stack, so
    the term is live. Skipping absent slots returned zero and removed a pair
    term of RMS 12-17 from every recycle of a template-free fold.
    """
    torch.manual_seed(0)
    spec = SPECS["protenix1"]
    module = FusedTemplateEmbedding(spec).eval()
    tokens = 6
    query = torch.randn(tokens, tokens, spec.pair_channel)
    ones = torch.ones(tokens, tokens)

    with torch.no_grad():
        one = module(query, _empty(1, tokens), ones, ones)
        four = module(query, _empty(4, tokens), ones, ones)

    assert one.abs().sum() > 0
    # Identical absent slots average to the same mean whatever their number.
    torch.testing.assert_close(one, four)
