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
    #: Atom names the vendor's tokenizer never creates on standard residues.
    drop_atoms: tuple[str, ...] = ()
    #: A chain with no alignments gets a depth-one MSA instead of AF3's two query rows.
    dedupe_self_msa: bool = False
    #: End policy of the atom-attention key window: "slide", "pad" or "slide_qblock".
    atom_key_window: str = "slide"
    #: Column-wise pair attention takes its pair bias transposed, Linear(z[k, q]).
    transposed_column_pair_bias: bool = False
    #: The diffusion transformer norms and projects the pair conditioning in every
    #: block; AF3 norms once and projects once per super block.
    per_block_pair_layer_norm: bool = False
    trunk_layers: int = 48
    #: Expansion factor of the pairformer transitions, in the trunk and the
    #: confidence head. AF3 is 4; chai-1 halves both.
    pairformer_transition_factor: int = 4
    #: Per-head width of the TRUNK pair attention; None is channel / heads.
    pair_qkv_dim: int | None = None
    #: Blocks in the diffusion token transformer.
    diffusion_blocks: int = 24
    #: Groups of the outer product mean. AF3 takes one C x C outer product over
    #: `opm_channel` channels a side; chai-1 projects to G groups of K and
    #: contracts WITHIN each group, giving G*K*K products. AF3 is G = 1.
    opm_groups: int = 1
    #: Width of each outer-product projection.
    opm_channel: int = 32
    #: chai-1's grouped outer product SUMS the depth axis without dividing, and
    #: its product LayerNorm carries eps 0.1 to absorb the missing scale.
    opm_sum_without_norm: bool = False
    confidence_layers: int = 4
    #: Per-head value width of the MSA pair-weighted averaging; None is msa / heads.
    msa_value_dim: int | None = None
    #: Single conditioning width inside the diffusion module.
    diffusion_seq_channel: int = 384
    #: LayerNorms that carry a trained offset where AF3's are scale-only, by name.
    affine_norms: frozenset[str] = frozenset()
    #: Which form of input embedder the weights were trained with. "af3"
    #: concatenates the 447 target features with the atom encoder's token output;
    #: "summed" adds bias-free restype, profile and conditioning projections onto
    #: that output; "chai1" builds its own token stream and projects the pair
    #: [token_act, stream] twice. The last two emit one seq_channel vector.
    input_embedder: str = "af3"
    #: Columns of the relative-position feature. A family that folds constant
    #: members of its token-pair stream into this projection carries a bias.
    #: Which relative-position encoding the weights were trained with.
    relpos: str = "af3"
    relpos_channel: int = 139
    relpos_bias: bool = False
    #: Columns of the MSA feature when the family does not build AF3's set.
    msa_feat_columns: int | None = None
    #: Value of the appended MSA "is paired" column on the query row; None omits it.
    msa_query_paired: float | None = None
    #: Pair init also embeds token bond orders and contact conditioning.
    bond_type_and_contact_init: bool = False
    #: Which template embedder the weights were trained with.
    template: str = "af3"
    template_layers: int = 2
    template_heads: int = 4
    #: Per-head width of the template pair attention; None is channel / heads.
    template_qkv_dim: int | None = None
    template_transition_factor: int = 2
    #: Whether homolog templates from the input reach the embedder. RoseTTAFold3's
    #: template channel is distance-distribution CONDITIONING with a noise level: a
    #: homolog fed as an exact condition is obeyed, not weighed (5I28 with four
    #: homologs: CA RMSD 1.9 A and strained peptide bonds, against 0.7 A without).
    use_input_templates: bool = True
    #: Fused templates: a template's visibility follows what it covers, not chains.
    template_visibility_by_coverage: bool = False
    #: Fused templates: the stack input is added once more around the whole stack.
    template_stack_outer_residual: bool = False
    #: Which confidence embedding and heads the weights were trained with.
    confidence: str = "af3"
    #: The pair entering the trunk is added once more after the MSA stack.
    msa_double_add: bool = False
    #: An MSA block updates the MSA before its outer product mean reads it.
    msa_update_before_opm: bool = False
    #: The outer product divides by the pair count clamped at one, then adds its bias.
    opm_bias_after_norm: bool = False
    #: The distogram projection carries a trained bias.
    distogram_bias: bool = False
    #: Conditioned transitions multiply the SwiGLU output by a linear up-gate.
    transition_up_gate: bool = False
    #: Diffusion pair conditioning concatenates the trunk pair with PROJECTED
    #: relative-position features.
    diffusion_projected_relpos: bool = False
    #: Diffusion pair conditioning concatenates the trunk pair with the token
    #: embedder's OWN pair output instead of a relative-position encoding; the
    #: relative features are already inside it.
    diffusion_pair_init_cond: bool = False
    #: Each conditioning track closes with an affine LayerNorm. Both feed every
    #: adaptive norm downstream, which scale by (s + 1), so omitting them lets
    #: the token transformer run away.
    diffusion_cond_final_norm: bool = False
    #: The token transformer re-normalises the single conditioning before
    #: projecting it. A family that already closed that track does not.
    single_cond_embedding_norm: bool = True
    #: The diffusion single conditioning projection carries a bias.
    single_cond_projection_bias: bool = False
    #: The fused atom feature embedding carries a bias.
    atom_features_bias: bool = False
    #: Atom queries are the per-atom features before the trunk single is added;
    #: only the conditioning sees the trunk.
    pre_trunk_atom_query: bool = False
    #: The reference charge enters raw; AF3 feeds arcsinh(charge).
    raw_ref_charge: bool = False
    #: The atom-pair conditioning is ONE projection over a distance-class
    #: one-hot, an inverse-square distance and a validity column, instead of
    #: separate offset, distance and validity embeddings; its MLP is two
    #: layers rather than three.
    atom_pair_distogram_feature: bool = False
    #: The atom decoder conditions on a second, affine norm over the encoder's
    #: atom conditioning rather than reusing it unchanged.
    post_atom_cond_norm: bool = False
    #: A padded key atom is masked from every query, not only from padded queries.
    key_masked_atom_attention: bool = False
    #: Atom transformers norm and project their pair conditioning in every block.
    per_block_atom_pair_layer_norm: bool = False
    #: Diffusion attentions LayerNorm the projected queries and keys (all heads flat).
    attention_kq_norm: bool = False
    #: Diffusion conditioning uses the identity-centred (s + 1) scale rather than
    #: sigmoid(s), and leaves the conditioning unnormalised.
    adaptive_identity_scale: bool = False
    #: Epsilon of the activation LayerNorm inside the diffusion blocks.
    adaptive_norm_eps: float = 1e-5
    #: The diffusion token transformer's attention carries an output gate.
    token_attention_gating_query: bool = True
    #: The atom transformers' attention carries an output gate, and projects its
    #: concatenated heads. Without the projection the raw heads are multiplied by
    #: the conditioning gate and that is the whole output.
    atom_attention_gating_query: bool = True
    atom_attention_project_output: bool = True
    #: Diffusion blocks feed the transition the PRE-attention activation and add
    #: both deltas in one residual: x + attention(x) + transition(x).
    parallel_attention_transition: bool = False
    #: Triangle attention's gate and output projections carry trained biases.
    triangle_attention_bias: bool = False
    #: Triangle multiplication divides by the sequence length before its centre norm.
    triangle_mul_divide_by_length: bool = False
    #: The outer product's left and right projections carry trained biases.
    opm_projection_bias: bool = False
    distogram_bins: int = 64
    #: The distogram head is an MLP (norm, hidden, GELU) rather than AF3's
    #: single projection, and symmetrises with the mean rather than the sum.
    distogram_hidden: bool = False
    distogram_mean_symmetrised: bool = False
    #: A constant the vendor's conformer-embedding MLP emits for an all-zero input.
    conformer_embedding_bias: bool = False
    #: The diffusion atom encoder embeds chirality gradients of the noisy coordinates.
    atom_chiral_features: bool = False
    #: chai-1's pairformer block is PARALLEL: every pair update reads the pair
    #: entering the block and their results are summed into it, and the single
    #: attention and transition both read the entering single. AF3 threads each
    #: update through the running activation.
    parallel_pairformer_block: bool = False
    #: chai-1's MSA block is parallel in two stages, and its pair transition
    #: sits in the FIRST: the two triangle multiplications and the transition
    #: all read the post-OPM pair and are summed in, then both attention
    #: directions read that result and are summed in turn.
    parallel_msa_block: bool = False
    #: chai-1's two pair-attention directions are one module whose single output
    #: projection reads them in mixed orientation, so the ending-node direction
    #: is NOT transposed back before the sum.
    untransposed_column_pair_output: bool = False
    #: The confidence head embeds the predicted structure as this many distance
    #: classes over (min, max); None keeps AF3's own distogram features.
    confidence_dgram: tuple[float, float, int] | None = None
    #: pLDDT is predicted over this many atom slots; None is the dense layout's.
    plddt_atom_slots: int | None = None
    #: The confidence head's pair attention carries a per-direction output
    #: projection pair combined as `kept + transpose(other)`.
    confidence_dual_output: bool = False
    #: MSA pair-weighted averaging masks its logits with the TOKEN PAIR mask at
    #: -10000 and zeroes the value where the MSA mask is false, instead of
    #: deriving a per-token mask from the MSA rows.
    msa_pair_mask_logits: bool = False
    #: The MSA feature embedding carries a trained bias.
    msa_activations_bias: bool = False
    #: The MSA stack's single term is the RECYCLED single, not the target feat.
    msa_single_from_recycle: bool = False
    #: The token-pair stream carries no bond feature, so no bond embedder runs.
    no_bond_embedding: bool = False
    #: The recycle carry starts at the INITIAL representations rather than zeros,
    #: so pass one already adds `recycle_proj(norm(z_init))`.
    recycle_from_initial: bool = False
    #: The structure module reads its own projection of the token features
    #: rather than sharing the trunk's.
    separate_structure_target_feat: bool = False
    #: The template feature embedding carries a trained bias.
    template_feature_bias: bool = False
    #: The template distogram's top class is a MASK class for pairs the template
    #: does not cover, instead of AF3's all-zero row there.
    template_mask_class: bool = False
    #: A template residue the template does not COVER is the gap restype, not
    #: the query's own residue.
    template_gap_uncovered: bool = False
    #: Templates are averaged over the PRESENT slots, not over every slot.
    template_present_denominator: bool = False
    #: Atom activations are re-masked in every atom-transformer block.
    mask_atom_act_per_block: bool = False
    #: Whether the confidence head predicts experimentally-resolved atoms.
    resolved_head: bool = True
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    rho: float = 7.0

    @property
    def target_feat_channel(self) -> int:
        """Width of the per-token input features that feed every embedder."""
        return 447 if self.input_embedder == "af3" else self.seq_channel

    @property
    def msa_feat_channel(self) -> int:
        """Restype one-hot, has-deletion, deletion value and the optional paired flag."""
        if self.msa_feat_columns is not None:
            return self.msa_feat_columns
        return 34 + (self.msa_query_paired is not None)


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
    key_masked_atom_attention=True,
    drop_atoms=("OXT", "OP3", "O3P"),
    dedupe_self_msa=True,
    atom_key_window="slide_qblock",
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
    drop_atoms=("OXT", "OP3", "O3P"),
)

#: OpenFold3 preview-2, kept because earlier results used it.
OPENFOLD3_PREVIEW2 = replace(
    OPENBIND0,
    family="openfold3",
    transposed_column_pair_bias=True,
    per_block_pair_layer_norm=True,
)

#: Boltz-2. OpenFold3 lineage, with its own input embedder, pair init, template
#: module, confidence re-embedding and sampler constants.
BOLTZ2 = replace(
    OPENFOLD3_PREVIEW2,
    family="boltz2",
    padded_single_cond=False,
    trunk_layers=64,
    confidence_layers=8,
    msa_value_dim=32,
    diffusion_seq_channel=768,
    affine_norms=frozenset(
        {
            "pair_cond_initial_norm",
            "single_cond_initial_norm",
            "noise_embedding_initial_norm",
            "single_cond_embedding_norm",
            "output_norm",
            "lnorm_trunk_single_cond",
            "lnorm_trunk_pair_cond",
            "atom_features_layer_norm",
        }
    ),
    input_embedder="summed",
    msa_query_paired=1.0,
    bond_type_and_contact_init=True,
    template="boltz2",
    template_qkv_dim=32,
    template_transition_factor=4,
    template_visibility_by_coverage=True,
    template_stack_outer_residual=True,
    confidence="boltz2",
    msa_double_add=True,
    msa_update_before_opm=True,
    opm_bias_after_norm=True,
    distogram_bias=True,
    transition_up_gate=True,
    diffusion_projected_relpos=True,
    single_cond_projection_bias=True,
    atom_features_bias=True,
    pre_trunk_atom_query=True,
    raw_ref_charge=True,
    key_masked_atom_attention=True,
    drop_atoms=("OXT",),
    dedupe_self_msa=True,
    atom_key_window="pad",
    gamma_0=0.605,
    gamma_min=1.107,
    noise_scale=0.901,
    step_scale=1.638,
    rho=8.0,
)

#: RoseTTAFold3 (RosettaCommons foundry). OpenFold3 lineage by its forward
#: conventions, with its own template conditioning, attention details and confidence
#: input normalisation. Its MSA stack is one block's weights run four times; the
#: converter replicates them.
ROSETTAFOLD3 = replace(
    OPENBIND0,
    family="rosettafold3",
    centre_ref_conformers=False,
    per_block_pair_layer_norm=True,
    msa_value_dim=32,
    affine_norms=BOLTZ2.affine_norms,
    msa_query_paired=0.0,
    template="rf3",
    use_input_templates=False,
    template_qkv_dim=64,
    template_transition_factor=4,
    confidence="rf3",
    opm_bias_after_norm=True,
    opm_projection_bias=True,
    distogram_bias=True,
    distogram_bins=65,
    diffusion_projected_relpos=True,
    pre_trunk_atom_query=True,
    raw_ref_charge=True,
    key_masked_atom_attention=True,
    per_block_atom_pair_layer_norm=True,
    attention_kq_norm=True,
    parallel_attention_transition=True,
    triangle_attention_bias=True,
    triangle_mul_divide_by_length=True,
    conformer_embedding_bias=True,
    atom_chiral_features=True,
    dedupe_self_msa=True,
    atom_key_window="pad",
)

#: Chai-1. Not OpenFold3 lineage: its own pairformer schedule (parallel on both
#: tracks), grouped outer product, fused two-direction pair attention, sixteen
#: diffusion blocks and an ESM2-3B token stream. Its token-pair stream carries no
#: bond feature and its confidence head predicts no resolved atoms.
CHAI1 = replace(
    ALPHAFOLD3,
    family="chai1",
    input_embedder="chai1",
    msa_feat_columns=41,
    relpos="chai1",
    relpos_channel=134,
    distogram_bias=True,
    distogram_hidden=True,
    distogram_mean_symmetrised=True,
    diffusion_pair_init_cond=True,
    diffusion_cond_final_norm=True,
    single_cond_embedding_norm=False,
    adaptive_identity_scale=True,
    adaptive_norm_eps=0.1,
    token_attention_gating_query=False,
    atom_attention_gating_query=False,
    atom_attention_project_output=False,
    atom_pair_distogram_feature=True,
    post_atom_cond_norm=True,
    parallel_attention_transition=True,
    relpos_bias=True,
    pair_channel=256,
    diffusion_pair_channel=256,
    pairformer_transition_factor=2,
    msa_value_dim=32,
    opm_groups=8,
    opm_channel=8,
    opm_sum_without_norm=True,
    diffusion_blocks=16,
    template_qkv_dim=32,
    template_feature_bias=True,
    template_mask_class=True,
    template_gap_uncovered=True,
    template_present_denominator=True,
    confidence="chai1",
    confidence_dual_output=True,
    confidence_dgram=(3.375, 21.375, 16),
    plddt_atom_slots=37,
    affine_norms=frozenset(
        {
            "pair_cond_initial_norm",
            "single_cond_initial_norm",
            "noise_embedding_initial_norm",
            "output_norm",
            "lnorm_trunk_single_cond",
            "lnorm_trunk_pair_cond",
            "atom_features_layer_norm",
            "pair_input_layer_norm",
        }
    ),
    per_block_pair_layer_norm=True,
    parallel_pairformer_block=True,
    parallel_msa_block=True,
    untransposed_column_pair_output=True,
    msa_pair_mask_logits=True,
    msa_activations_bias=True,
    msa_single_from_recycle=True,
    no_bond_embedding=True,
    recycle_from_initial=True,
    separate_structure_target_feat=True,
    mask_atom_act_per_block=True,
    resolved_head=False,
    msa_double_add=True,
    pre_trunk_atom_query=True,
    raw_ref_charge=True,
    drop_atoms=("OXT",),
    atom_key_window="circular",
    sigma_max=80.0,
)

SPECS = {
    spec.family: spec
    for spec in (
        ALPHAFOLD3,
        INTELLIFOLD2,
        OPENBIND0,
        OPENFOLD3_PREVIEW2,
        BOLTZ2,
        ROSETTAFOLD3,
        CHAI1,
    )
}
