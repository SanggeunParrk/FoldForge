"""One inference MSA policy for every predictor, following the AF3 pipeline.

Inputs are prepared with up to ``PREPARED_ROWS`` alignment rows, the official
AF3 ``msa_crop_size``. Every trunk pass then embeds a fresh uniformly random
subset of ``SAMPLED_ROWS`` valid rows, the released AF3 ``num_msa``; rows are
drawn without replacement and re-drawn on each recycle, so passes see different
alignments while profile and deletion statistics keep the full prepared MSA.

Released adapters differed: AF3 already behaves this way, OpenDDE sampled 1280
rows per pass, Protenix drew a random-size subset (Uniform[1, n] rows, capped at
16384) per pass in training and inference alike, and ESMFold2 embedded every
prepared row on every recurrence loop. Every family now reads the rule from its
``DenseSpec`` row instead, so a benchmark row records ``msa_policy`` rather than
each checkpoint's own default. Templates are capped at ``TEMPLATES_PER_CHAIN``
by the input spec, the AF3 ``max_templates``.
"""

from __future__ import annotations

from typing import Any

PREPARED_ROWS = 16384
SAMPLED_ROWS = 1024
TEMPLATES_PER_CHAIN = 4

RECORD: dict[str, Any] = {
    "name": "af3-msa-v1",
    "prepared_rows": PREPARED_ROWS,
    "sampled_rows_per_recycle": SAMPLED_ROWS,
    "templates_per_chain": TEMPLATES_PER_CHAIN,
}
