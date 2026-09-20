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
    #: OpenFold3 lineage: a bond sets both [i, j] and [j, i] of the token bond matrix.
    symmetric_bonds: bool = False
    #: OpenFold3 lineage: a padded key atom never counts as the query's own residue.
    #: Its zero-filled ref_space_uid would otherwise collide with token 0.
    key_masked_offsets: bool = False
    #: The vendor's single conditioning spans 833 channels: its restype and profile
    #: blocks carry one class AF3 lacks, re-inserted as zero columns before the norm.
    padded_single_cond: bool = False
    #: Reference conformers are centred per residue before they reach the network.
    centre_ref_conformers: bool = False
    #: Column-wise pair attention takes its pair bias transposed, Linear(z[k, q]).
    transposed_column_pair_bias: bool = False
    #: The diffusion transformer norms and projects the pair conditioning in every
    #: block; AF3 norms once and projects once per super block.
    per_block_pair_layer_norm: bool = False
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

#: OpenFold3 v0.5.0 "OpenBind". It adopted AF3's single pair norm in the diffusion
#: transformer, so only the lineage conventions remain.
OPENBIND0 = replace(
    ALPHAFOLD3,
    family="openbind0",
    symmetric_bonds=True,
    key_masked_offsets=True,
    padded_single_cond=True,
    centre_ref_conformers=True,
)

#: OpenFold3 preview-2, kept because earlier results used it.
OPENFOLD3_PREVIEW2 = replace(
    OPENBIND0,
    family="openfold3",
    transposed_column_pair_bias=True,
    per_block_pair_layer_norm=True,
)

SPECS = {
    spec.family: spec
    for spec in (ALPHAFOLD3, INTELLIFOLD2, OPENBIND0, OPENFOLD3_PREVIEW2)
}
