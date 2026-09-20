"""One AlphaFold 3 graph, many released weights.

Every AF3-family predictor that loads into the dense graph is this graph at some
width plus, for some families, declared branches. A family is a row here; it is
never a copy of the network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DenseSpec:
    """Widths, head counts and sampler constants of one dense-graph family."""

    family: str = "alphafold3"
    pair_channel: int = 128
    seq_channel: int = 384
    msa_channel: int = 64
    template_channel: int = 64
    #: Pair width inside the diffusion conditioning; AF3 keeps it at the trunk width.
    diffusion_pair_channel: int = 128
    #: Heads of every pair-axis attention: trunk, MSA stack, templates, confidence.
    pair_heads: int = 4
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5


ALPHAFOLD3 = DenseSpec()

#: IntelliFold-v2 runs the AF3 graph verbatim at wider channels and eight pair heads.
INTELLIFOLD2 = replace(
    ALPHAFOLD3,
    family="intellifold2",
    pair_channel=512,
    msa_channel=256,
    template_channel=256,
    diffusion_pair_channel=512,
    pair_heads=8,
)

SPECS = {spec.family: spec for spec in (ALPHAFOLD3, INTELLIFOLD2)}
