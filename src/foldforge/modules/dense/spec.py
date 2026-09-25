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
    #: The vendor's single conditioning spans 833 channels: its restype and profile
    #: blocks carry one class AF3 lacks, re-inserted as zero columns before the norm.
    padded_single_cond: bool = False
    #: Whose idealised residue geometry fills `ref_pos`. "af3" is the CCD
    #: ideal; a family trained on its own frame was trained on THAT one, and
    #: the feature goes straight into a Linear, so it is not pose-invariant.
    ref_conformers: str = "af3"
    #: Reference conformers are centred per residue before they reach the network.
    #: The AF3 SI (Table 5) says ref_pos carries "a random rotation and
    #: translation"; AF3's inference code poses nothing, and this graph follows
    #: the code. The OpenFold3 releases centre (and randomly rotate) per residue.
    centre_ref_conformers: bool = False
    #: Atom names the vendor's tokenizer never creates on standard residues.
    drop_atoms: tuple[str, ...] = ()
    #: How a family pads an ABSENT template: "first" gives one gap-restype
    #: template and zero-pads the rest, "all" makes every slot one. None
    #: keeps AF3's zero class. Live even with no template supplied, because
    #: these families' template embedders divide by the padded slot count.
    empty_template_gap: str | None = None
    #: A chain with no alignments gets a depth-one MSA instead of AF3's two query rows.
    dedupe_self_msa: bool = False
    #: Column-wise pair attention takes its pair bias transposed, Linear(z[k, q]).
    #:
    #: The AF3 SI and AF3's own code DISAGREE here. SI Algorithm 15 writes the
    #: ending-node logits as q_ij.k_kj + b_ki, which is also AF2/OpenFold's
    #: convention (bias taken after transposing the pair). AF3's code projects
    #: the bias BEFORE transposing, i.e. b_ik -- that is this graph's default
    #: because it is what AF3's weights saw. Families built from the SI or the
    #: OpenFold lineage (Protenix, OpenDDE, OpenFold3 preview-2, Boltz-2) set
    #: this. Off: -2 to -28 pLDDT.
    transposed_column_pair_bias: bool = False
    #: The diffusion transformer norms and projects the pair conditioning in every
    #: block; AF3 norms once and projects once per super block.
    per_block_pair_layer_norm: bool = False
    #: Whether a recycle setting counts TOTAL trunk passes rather than the
    #: additional ones AF3 counts. Clamped at one either way.
    recycles_are_total: bool = False
    trunk_layers: int = 48
    #: Whether the pairformer blocks carry the two pair-axis ATTENTIONS. A
    #: family that folds from a language model keeps only the triangle
    #: multiplications and the transition.
    pair_attention: bool = True
    #: Whether the TRUNK carries a single track at all. Without one there is no
    #: `single` to recycle, to condition the denoiser on, or to hand a head:
    #: every downstream single is built from the target features instead.
    trunk_single_track: bool = True
    #: The trunk recycles through a discretised diagonal SSM rather than an
    #: addition: `z = decay * z_prev + proj(norm(z_inject))` against AF3's
    #: `z = z_inject + proj(norm(z_prev))`. The same two modules applied to the
    #: OTHER operand, plus a per-channel decay; at decay one with the operands
    #: swapped back it is AF3's own recycling.
    ssm_recycle: bool = False
    #: Pair blocks run ONCE after the recycle loop, reading a projection of the
    #: stack's output. AF3 has no post-trunk stack.
    coda_layers: int = 0
    #: Pair blocks over the language model's pair representation, before it is
    #: added to the trunk pair.
    lm_encoder_layers: int = 0
    #: Dropout kept on the language-model pair AT INFERENCE, resampled every
    #: recycle pass. Not polish: the family trained with it.
    lm_pair_dropout: float = 0.0
    #: The summed per-atom reference features are normalised. AF3 sums bias-free
    #: per-feature projections and leaves the result unnormalised.
    normed_atom_features: bool = False
    #: Blocks of the MSA stack. A family that folds from a language model alone
    #: has none, and must not BUILD one: its checkpoint carries no weights for it.
    msa_layers: int = 4
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
    #: The relative-CHAIN bucket is keyed on same-CHAIN, sending the MATCH to
    #: the pad class, where AF3 keys on same-ENTITY and sends the MISMATCH
    #: there. Read by all three call sites -- trunk, diffusion conditioning and
    #: confidence re-embedding -- because the vendor builds them from one module.
    chain_bucket_on_same_chain: bool = False
    relpos_bias: bool = False
    #: Which protein language model supplies this family's token stream, if
    #: any. Its embeddings arrive on the batch like any other input.
    language_model: str | None = None
    #: Which MSA feature stream the weights were trained on.
    msa_feat_layout: str = "af3"
    #: Columns of the MSA feature when the family does not build AF3's set.
    msa_feat_columns: int | None = None
    #: The MSA is taken in the order given, query first, and not shuffled.
    #: How the MSA is cut to the trunk's depth. "shuffle" is AF3's gumbel
    #: permutation followed by a truncation; "ordered" takes the alignment's
    #: first rows as given; "keep_query" draws a random subset but pins the
    #: query at row 0 and re-sorts. The three differ only above the depth
    #: limit, which is why a single-sequence gate cannot tell them apart.
    msa_subsample: str = "shuffle"
    #: Value of the appended MSA "is paired" column on the query row; None omits it.
    msa_query_paired: float | None = None
    #: Pair init also embeds token bond orders and contact conditioning.
    bond_type_and_contact_init: bool = False
    #: Which template embedder the weights were trained with.
    template: str = "af3"
    template_layers: int = 2
    #: Heads of the template pair attention; None ties it to the trunk's, which
    #: is what AF3 does and what every family on its template embedder kept.
    template_heads: int | None = None
    #: Per-head width of the template pair attention; None is channel / heads.
    template_qkv_dim: int | None = None
    #: Width of the template triangle multiplication's projections, where it
    #: is NOT the channel count AF3 ties it to.
    template_hidden_dim: int | None = None
    template_transition_factor: int = 2
    #: Whether homolog templates from the input reach the embedder. RoseTTAFold3's
    #: template channel is distance-distribution CONDITIONING with a noise level: a
    #: homolog fed as an exact condition is obeyed, not weighed (5I28 with four
    #: homologs: CA RMSD 1.9 A and strained peptide bonds, against 0.7 A without).
    use_input_templates: bool = True
    #: Which NAMES the confidence head's weights were written under. The head
    #: itself is chosen by `confidence`; this is a rename of its records.
    confidence_records: str = "af3"
    #: The confidence head re-embeds the pair from the inputs (Boltz lineage)
    #: rather than reading the trunk's. Everything a family does on top of that
    #: is a field of its own below. This was once a string naming each family's
    #: head; only "boltz2" was ever read -- resetting "chai1", "rf3" or
    #: "protenix2" left PAE, PDE and pLDDT bit-identical.
    confidence_reembed_pair: bool = False
    #: The trunk inputs are normalised over the WHOLE tensor of real tokens
    #: before the head reads them, rather than per position.
    confidence_global_norm: bool = False
    #: The trunk single is clamped and normalised before ANY use, so the
    #: confidence pairformer and every head see the normalised one. AF3 uses it
    #: raw, and an unnormalised trunk single enters this head at std 211.
    confidence_single_clamp: bool = False
    #: A raw, unbinned distance term rides alongside the binned one, carrying
    #: the sub-bin resolution the one-hot throws away.
    confidence_raw_distance: bool = False
    #: The predicted structure is embedded as forty TOKEN-CENTRE distance
    #: classes from 3.25 to 50.75 A, masked, rather than from the pseudo-beta
    #: gather AF3 uses.
    confidence_centre_dgram: bool = False
    #: Layout of the per-token features the DIFFUSION conditioning and the
    #: confidence re-embedding read. "af3" is the trunk's own 447. "esm"
    #: widens the restype and profile blocks from 31 classes to 33: the extra
    #: columns cannot be dropped from the weights the way a zero input column
    #: can be dropped from a bias-free Linear, because the LayerNorm after them
    #: divides by the width and subtracts the mean over it, so normalising 447
    #: channels instead of 451 rescales the ENTIRE conditioning.
    single_cond_layout: str = "af3"
    #: How the distance-error head symmetrises. AF3 projects the pair and
    #: symmetrises the LOGITS; "pair" symmetrises inside the norm, which a
    #: LayerNorm makes a different function; "none" does neither, reading the
    #: pair exactly as its PAE head does. Summing a logit with its transpose
    #: doubles an expectation, so the wrong one is not a small error.
    pde_symmetrise: str = "logits"
    #: Bias added to the single-attention GATE logits in a pairformer block,
    #: before the sigmoid. Chai-1's parallel block opens its gate by 1.0, and no
    #: weight can express that -- a sigmoid does not fold into a linear map.
    #: Missing it, every block's attention contributes less, and the shortfall
    #: compounds: over 48 blocks the trunk's single reaches the confidence head
    #: at 0.58x the released scale. Undocumented; ``sigmoid(g + CONSTANTS.c0)``
    #: with c0 = 1 in the traced trunk.
    single_attention_gate_bias: float = 0.0
    #: Separate intra- and inter-chain heads for the distance error and the PAE.
    #: Boltz-2 established the re-embedded pair; a family can take that path
    #: without splitting its heads, so the two are stated apart.
    confidence_split_heads: bool = False
    #: Whether the four heads norm their input. Boltz-2 dropped every one of
    #: them; the families that reuse its re-embedding did not all follow.
    confidence_head_norms: bool = True
    #: Distance classes the confidence re-embedding bins the prediction into,
    #: with its OWN trained boundaries rather than a constant range. The class
    #: count is one more than the boundary count and has to be static.
    confidence_learned_bins: int | None = None
    #: A trained attention pooling over each token's pair ROW, projected back
    #: onto the single before the per-token heads read it.
    confidence_row_pool: bool = False
    #: The pair entering the MSA module is added once more after it.
    #:
    #: A literal reading of a TYPO in the AF3 Supplementary Information, kept
    #: because the weights were trained with it. SI Algorithm 1 line 10 writes
    #: ``{z} += MsaModule(...)`` while Algorithm 8 already returns the UPDATED
    #: pair (line 15, ``return {z}``), so the input is counted twice. AF3's own
    #: code assigns (``z = msa_module(z)``) and the Protenix report lists this
    #: line as an erratum (Table 1: "an additional residual update would be
    #: redundant"). Boltz-2 (``z = z + msa_module(z)``, boltz2.py) and Chai-1
    #: (traced trunk) implemented the typo; neither report mentions it. Turning
    #: it off costs Boltz-2 9 pLDDT on 5I28 -- do not "fix" it here, report it.
    msa_double_add: bool = False
    #: An MSA block updates the MSA before its outer product mean reads it.
    #:
    #: A DOCUMENTED design change, not an artifact. AF3 SI Algorithm 8 runs the
    #: outer product mean first (line 6). Boltz-1 reorders it to
    #: PairWeightedAveraging -> MSATransition -> OuterProductMean so the MSA
    #: transition's output reaches the pair in the same block (Boltz-1 report,
    #: section 3.1 "Architectural modifications"; tested on a smaller model, no
    #: ablation published). Boltz-2 keeps it; OpenDDE copies it ("Boltz-style
    #: MSA block" in its pairformer.py). Off: Boltz-2 -46, OpenDDE -18 pLDDT.
    msa_update_before_opm: bool = False
    #: The distogram projection carries a trained bias.
    distogram_bias: bool = False
    #: Conditioned transitions multiply the SwiGLU output by a linear up-gate.
    transition_up_gate: bool = False
    #: Diffusion pair conditioning concatenates the trunk pair with PROJECTED
    #: relative-position features.
    diffusion_projected_relpos: bool = False
    #: The diffusion and the confidence heads run on STRUCTURAL tokens: each
    #: residue splits into a backbone and a sidechain token, expanded from the
    #: residue-level trunk and refined by its own pairformer stack.
    structural_tokens: bool = False
    structural_refiner_layers: int = 4
    #: The refiner stack's single-attention heads and transition factor, where
    #: the family trained it narrower than the trunk pairformer it reuses.
    structural_refiner_heads: int = 16
    structural_refiner_transition_factor: int = 4
    #: The pair initialisation reads the single activations rather than the
    #: target features, so the single is built before the pair.
    pair_init_from_single: bool = False
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
    #:
    #: The AF3 SI and AF3's own code DISAGREE here. SI Algorithm 5 copies
    #: ``q_l = c_l`` (line 7) BEFORE ``c_l += LinearNoBias(LayerNorm(s_trunk))``
    #: (line 9); AF3's code takes the query from the conditioning AFTER the trunk
    #: term, which is this graph's default. Boltz-2, Chai-1 and RF3 follow the
    #: SI. Off: Boltz-2 -43 pLDDT / 19 A.
    pre_trunk_atom_query: bool = False
    #: The reference charge enters raw; AF3 feeds arcsinh(charge).
    raw_ref_charge: bool = False
    #: The atom-pair conditioning is ONE projection over a distance-class
    #: one-hot, an inverse-square distance and a validity column, instead of
    #: separate offset, distance and validity embeddings; its MLP is two
    #: layers rather than three.
    atom_pair_distogram_feature: bool = False
    #: Atom attention opens only within a token. AF3 lets every atom attend
    #: across the whole window, spreading the softmax over some eighty keys
    #: where this opens nine; the damage is intra-residue geometry.
    #: Undocumented Chai-1 behaviour, read from its traced graphs.
    same_token_atom_attention: bool = False
    #: The atom conditioning SUM is normalised, without parameters. No blob
    #: names it, and every adaptive norm downstream scales by (s + 1) off it.
    #: Undocumented Chai-1 behaviour, read from its traced token embedder.
    atom_cond_norm: bool = False
    #: The atom decoder conditions on a second, affine norm over the encoder's
    #: atom conditioning rather than reusing it unchanged.
    post_atom_cond_norm: bool = False
    #: The atom attention CHAINS its two adaptive normalisations: the keys are
    #: gathered from the already-normed queries rather than from the raw
    #: activation. Two chained norms are not two parallel ones.
    chained_atom_key_norm: bool = False
    #: Atom transformers norm and project their pair conditioning in every block.
    per_block_atom_pair_layer_norm: bool = False
    #: Diffusion attentions LayerNorm the projected queries and keys (all heads flat).
    attention_kq_norm: bool = False
    #: Diffusion conditioning uses the identity-centred (s + 1) scale rather than
    #: sigmoid(s), and leaves the conditioning unnormalised.
    adaptive_identity_scale: bool = False
    #: The same, for the ATOM transformer alone; None follows the token one. A
    #: family whose atom attention is conditioned differently from its token
    #: attention says so here rather than by widening the field above.
    atom_adaptive_identity_scale: bool | None = None
    #: Whether the ATOM transformer's output gate carries a trained bias.
    atom_adaptive_zero_bias: bool = True
    #: The ATOM blocks normalise their activation with an affine-free RMSNorm,
    #: pass the conditioning through SiLU rather than a LayerNorm, and take the
    #: output gate RAW instead of through a sigmoid. None of the three carries
    #: a parameter, so a tree diff cannot see any of them.
    atom_rms_conditioning: bool = False
    #: The atom attention's ENTIRE positional signal is a 3D rotary built from
    #: the reference conformer and its space uid, applied after an affine-free
    #: RMSNorm on the queries and keys. Empty for every other family.
    atom_rope: dict[str, float] | None = None
    #: Half-width, by RANK among valid atoms, of the atom attention's sliding
    #: window. AF3's window is only BLOCK-aligned, so this is a different mask
    #: at the same width, and the key subset has to be wide enough to hold it.
    atom_window_half: int | None = None
    #: The sampler rigid-aligns the noisy coordinates onto the denoised
    #: prediction before each Euler step.
    realign_sampler: bool = False
    #: Atoms per key subset. AF3 takes 128; a family with a wider window needs
    #: a subset that covers it.
    atom_keys_subset: int = 128
    #: The diffusion token transformer's attention carries an output gate.
    token_attention_gating_query: bool = True
    #: The atom transformers' attention carries an output gate, and projects its
    #: concatenated heads. Without the projection the raw heads are multiplied by
    #: the conditioning gate and that is the whole output.
    atom_attention_gating_query: bool = True
    atom_attention_project_output: bool = True
    #: Diffusion blocks feed the transition the PRE-attention activation and add
    #: both deltas in one residual: x + attention(x) + transition(x).
    #:
    #: Follows the AF3 SI's "unusual order" (Algorithm 23: ``a <- b +
    #: ConditionedTransitionBlock(a)``), which the Boltz-1, Protenix and RF3
    #: reports all call a problem and replace with two sequential residuals.
    #: Chai-1 keeps the parallel form (traced code). RF3's RELEASE does too --
    #: ``no_residual_connection_between_attention_and_transition: true`` in
    #: rf3_net.yaml -- although its report (appendix A.3.2) says it switched to
    #: the sequential form: the paper and the released code disagree.
    parallel_attention_transition: bool = False
    #: Triangle attention's gate and output projections carry trained biases.
    triangle_attention_bias: bool = False
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
    #:
    #: Undocumented (AF3 SI Algorithm 17 is sequential; the Chai-1 report says
    #: only that it "largely follows" AF3), read from the traced trunk. team-gm's
    #: install_pairformers must NOT swap such a block for its sequential one.
    parallel_pairformer_block: bool = False
    #: chai-1's MSA block is parallel in two stages, and its pair transition
    #: sits in the FIRST: the two triangle multiplications and the transition
    #: all read the post-OPM pair and are summed in, then both attention
    #: directions read that result and are summed in turn. The MSA row update is
    #: parallel as well: its transition reads the MSA entering the block.
    parallel_msa_block: bool = False
    #: chai-1's two pair-attention directions are one module whose single output
    #: projection reads them in mixed orientation, so the ending-node direction
    #: is NOT transposed back before the sum.
    #:
    #: Undocumented: in neither the AF3 SI nor the Chai-1 report, read from the
    #: release's traced trunk (trunk.pt forward_256). Off: -47 pLDDT.
    untransposed_column_pair_output: bool = False
    #: The confidence head embeds the predicted structure as this many distance
    #: classes over (min, max); None keeps AF3's own distogram features.
    confidence_dgram: tuple[float, float, int] | None = None
    #: pLDDT is predicted over this many atom slots; None is the dense layout's.
    plddt_atom_slots: int | None = None
    #: The confidence head's pair attention carries a per-direction output
    #: projection pair combined as `kept + transpose(other)`.
    confidence_dual_output: bool = False
    #: The MSA feature embedding carries a trained bias.
    msa_activations_bias: bool = False
    #: The MSA stack's single term is the single after its recycle add -- the
    #: one the pairformer starts from -- not the target feat.
    msa_single_from_recycle: bool = False
    #: The token-pair stream carries no bond feature, so no bond embedder runs.
    no_bond_embedding: bool = False
    #: The recycle carry starts at the INITIAL representations rather than zeros,
    #: so pass one already adds `recycle_proj(norm(z_init))`.
    recycle_from_initial: bool = False
    #: The template feature embedding carries a trained bias.
    template_feature_bias: bool = False
    #: Each template's normalised embedding is multiplied by its own coverage
    #: before the sum. Not a no-op: the norm has a bias, so an uncovered pair
    #: is nonzero after it.
    template_coverage_mask: bool = False
    #: Templates are averaged over the PRESENT slots, not over every slot.
    template_present_denominator: bool = False
    #: Whether the confidence head predicts experimentally-resolved atoms.
    resolved_head: bool = True
    #: EDM's own churn: one constant over a sigma window, rather than AF3's
    #: gamma_0 switched on above gamma_min. None keeps AF3's form.
    churn_total: float | None = None
    churn_sigma_min: float = 0.0
    churn_sigma_max: float = float("inf")
    #: Floor of the re-noising variance; a no-churn step still adds this.
    sampler_variance_floor: float = 0.0
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
    def single_cond_channel(self) -> int:
        """Width of the per-token features the diffusion and confidence read."""
        if self.single_cond_layout == "esm":
            # Two extra classes in each of the restype and profile blocks.
            return self.target_feat_channel + 4
        return self.target_feat_channel

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
    dedupe_self_msa=True,
)

#: OpenFold3 v0.5.0 "OpenBind". It adopted AF3's single pair norm in the diffusion
#: transformer, so only the lineage conventions remain.
OPENBIND0 = replace(
    ALPHAFOLD3,
    family="openbind0",
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

#: Boltz-2. OpenFold3 lineage, with its own input embedder, pair init, template
#: module and confidence re-embedding. Its sampler is AF3's except sigma_min.
BOLTZ2 = replace(
    OPENFOLD3_PREVIEW2,
    family="boltz2",
    confidence_split_heads=True,
    confidence_head_norms=False,
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
    template_heads=4,
    template_qkv_dim=32,
    template_transition_factor=4,
    confidence_reembed_pair=True,
    msa_double_add=True,
    msa_update_before_opm=True,
    distogram_bias=True,
    transition_up_gate=True,
    diffusion_projected_relpos=True,
    single_cond_projection_bias=True,
    atom_features_bias=True,
    pre_trunk_atom_query=True,
    raw_ref_charge=True,
    dedupe_self_msa=True,
    # Boltz-2's released sampler is Boltz2DiffusionParams (boltz main.py): AF3's
    # gamma_0/gamma_min/noise_scale/step_scale and rho, with sigma_min 1e-4.
    # The 0.605/1.107/0.901/rho 8/1.638 set once here is BoltzDiffusionParams --
    # Boltz-1's -- and made our samples 2.7x tighter than the release's.
    sigma_min=1e-4,
)

#: RoseTTAFold3 (RosettaCommons foundry). OpenFold3 lineage by its forward
#: conventions, with its own template conditioning, attention details and confidence
#: input normalisation. Its MSA stack is one block's weights run four times; the
#: converter replicates them.
ROSETTAFOLD3 = replace(
    OPENBIND0,
    family="rosettafold3",
    # The release's n_recycles is the number of trunk passes (range(n_recycles)).
    recycles_are_total=True,
    centre_ref_conformers=False,
    per_block_pair_layer_norm=True,
    msa_value_dim=32,
    affine_norms=BOLTZ2.affine_norms,
    msa_query_paired=0.0,
    template="rf3",
    template_heads=4,
    use_input_templates=False,
    template_qkv_dim=64,
    template_transition_factor=4,
    confidence_global_norm=True,
    confidence_centre_dgram=True,
    opm_projection_bias=True,
    distogram_bias=True,
    distogram_bins=65,
    diffusion_projected_relpos=True,
    pre_trunk_atom_query=True,
    raw_ref_charge=True,
    per_block_atom_pair_layer_norm=True,
    attention_kq_norm=True,
    parallel_attention_transition=True,
    triangle_attention_bias=True,
    conformer_embedding_bias=True,
    atom_chiral_features=True,
    dedupe_self_msa=True,
)

#: Chai-1. Not OpenFold3 lineage: its own pairformer schedule (parallel on both
#: tracks), grouped outer product, fused two-direction pair attention, sixteen
#: diffusion blocks and an ESM2-3B token stream. Its token-pair stream carries no
#: bond feature and its confidence head predicts no resolved atoms.
CHAI1 = replace(
    ALPHAFOLD3,
    family="chai1",
    input_embedder="chai1",
    recycles_are_total=True,
    language_model="esm2",
    msa_feat_layout="chai1",
    msa_feat_columns=41,
    msa_subsample="ordered",
    relpos="chai1",
    relpos_channel=134,
    distogram_bias=True,
    distogram_hidden=True,
    distogram_mean_symmetrised=True,
    diffusion_pair_init_cond=True,
    diffusion_cond_final_norm=True,
    single_cond_embedding_norm=False,
    adaptive_identity_scale=True,
    token_attention_gating_query=False,
    atom_attention_gating_query=False,
    atom_attention_project_output=False,
    atom_pair_distogram_feature=True,
    same_token_atom_attention=True,
    atom_cond_norm=True,
    post_atom_cond_norm=True,
    parallel_attention_transition=True,
    relpos_bias=True,
    pair_channel=256,
    diffusion_pair_channel=256,
    pairformer_transition_factor=2,
    msa_value_dim=32,
    opm_groups=8,
    opm_channel=8,
    diffusion_blocks=16,
    template_qkv_dim=32,
    template_feature_bias=True,
    template_present_denominator=True,
    template_coverage_mask=True,
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
    # Its parallel block opens the single-attention gate by 1.0, which no weight
    # can express. The released block also masks that attention's residual; the
    # mask changes no output (padded rows are masked downstream) and is dropped.
    single_attention_gate_bias=1.0,
    parallel_msa_block=True,
    untransposed_column_pair_output=True,
    msa_activations_bias=True,
    msa_single_from_recycle=True,
    no_bond_embedding=True,
    recycle_from_initial=True,
    resolved_head=False,
    msa_double_add=True,
    pre_trunk_atom_query=True,
    raw_ref_charge=True,
    sigma_max=80.0,
    churn_total=80.0,
    # Kept although a same-seed reset moves one sample by only 0.5 A: the churn
    # window and variance floor set how widely the samples spread (0.32 A with
    # them, 0.55 A without, against the release's 0.39 A on 5I28).
    churn_sigma_min=4e-4,
    churn_sigma_max=80.0,
    sampler_variance_floor=1e-6,
)

#: Protenix v2. OpenFold3 lineage by its forward conventions, widened to a
#: 256-channel pair with eight pair heads, and carrying its own fused template
#: features and two confidence divergences.
PROTENIX2 = replace(
    OPENBIND0,
    family="protenix2",
    # The release's N_cycle is the number of trunk passes (range(N_cycle)).
    recycles_are_total=True,
    chained_atom_key_norm=True,
    dedupe_self_msa=True,
    # Kept after the release re-fold: without it protenix1 on 5I28 reads 69.93
    # pLDDT against its release's 68.43. With no template the template term is
    # still live, and the empty slot's restype changes it.
    empty_template_gap="first",
    pair_channel=256,
    msa_channel=128,
    diffusion_pair_channel=256,
    pair_heads=8,
    msa_value_dim=8,
    per_block_pair_layer_norm=True,
    per_block_atom_pair_layer_norm=True,
    transposed_column_pair_bias=True,
    diffusion_projected_relpos=True,
    distogram_bias=True,
    template="protenix2",
    template_heads=2,
    pde_symmetrise="pair",
    confidence_single_clamp=True,
    confidence_raw_distance=True,
)

#: Protenix v1. The same graph at AlphaFold 3's widths, but its TEMPLATE stack
#: is uniformly twice as wide as its channel count implies: a 128-wide triangle
#: multiplication and 4 x 32 attention on a 64-channel template pair.
PROTENIX1 = replace(
    PROTENIX2,
    family="protenix1",
    pair_channel=128,
    msa_channel=64,
    diffusion_pair_channel=128,
    pair_heads=4,
    msa_value_dim=None,
    template_heads=4,
    template_qkv_dim=32,
    template_hidden_dim=128,
)

#: OpenDDE. Protenix lineage, widened again to a 384-channel pair with twelve
#: pair heads, a 96-bin distogram, and its own diffusion pair conditioning: it
#: compresses the trunk pair and the relative features SEPARATELY before the
#: joint norm, where AF3 projects one concatenation.
OPENDDE = replace(
    PROTENIX2,
    family="opendde",
    structural_tokens=True,
    confidence_records="opendde",
    dedupe_self_msa=False,
    pair_channel=384,
    #: The trunk is 384 wide but the denoiser conditions on a 128-wide pair,
    #: which is what makes it compress the trunk pair before concatenating.
    diffusion_pair_channel=128,
    pair_heads=12,
    structural_refiner_heads=8,
    structural_refiner_transition_factor=2,
    distogram_bins=96,
    msa_update_before_opm=True,
    diffusion_projected_relpos=True,
    pair_init_from_single=True,
    template="af3",
)

#: ESMFold2 folds from ESM-C, not from an MSA: its trunk is PAIR-ONLY at 256
#: channels, it recycles through a discretised diagonal SSM rather than an
#: addition, and it closes with a post-loop "coda" of pair blocks.
ESMFOLD2 = replace(
    ALPHAFOLD3,
    family="esmfold2",
    pair_channel=256,
    msa_channel=128,
    diffusion_pair_channel=256,
    diffusion_seq_channel=768,
    diffusion_blocks=12,
    trunk_layers=48,
    msa_layers=4,
    template_layers=0,
    coda_layers=2,
    lm_encoder_layers=4,
    lm_pair_dropout=0.25,
    pair_attention=False,
    trunk_single_track=False,
    ssm_recycle=True,
    normed_atom_features=True,
    distogram_bins=64,
    distogram_bias=True,
    diffusion_projected_relpos=True,
    msa_subsample="keep_query",
    chain_bucket_on_same_chain=True,
    pde_symmetrise="none",
    raw_ref_charge=True,
    # No terminal OXT: this family builds its atom list from a fixed table that
    # carries none, so AF3's extra oxygen is an atom it has never seen. Worse
    # here than elsewhere, because the atom window is +/-64 by RANK -- one
    # spurious atom at the END of the list corrupts the last ~64 atoms'
    # attention, and nothing before them.
    drop_atoms=("OXT",),
    # Its self-MSA is the query ONCE, where AF3 hands a chain with no
    # alignments two identical rows. Worth 4.3% of the trunk's MSA injection.
    dedupe_self_msa=True,
    ref_conformers="esmfold2",
    language_model="esmc",
    confidence_reembed_pair=True,
    confidence_learned_bins=39,
    single_cond_layout="esm",
    atom_adaptive_identity_scale=True,
    atom_adaptive_zero_bias=False,
    atom_rms_conditioning=True,
    # 3 axes x 2 spatial pairs + 10 uid pairs = 16, which is head_dim / 2 for
    # this family's four 32-channel atom heads exactly.
    atom_rope={"n_spatial": 2, "n_uid": 10, "spatial_base": 20.0, "uid_base": 1e4},
    atom_window_half=64,
    # 32 queries plus 2x64 of context needs 160; 192 is the next size AF3's
    # gather machinery takes.
    atom_keys_subset=192,
    realign_sampler=True,
    confidence_row_pool=True,
    affine_norms=frozenset(
        {
            "pair_cond_initial_norm",
            "single_cond_initial_norm",
            "noise_embedding_initial_norm",
            "single_cond_embedding_norm",
            "output_norm",
            "atom_features_layer_norm",
        }
    ),
)

#: The same family at 24 trunk blocks and no MSA encoder: it folds from the
#: language model alone, so it must not BUILD an MSA stack it has no weights for.
ESMFOLD2_FAST = replace(
    ESMFOLD2,
    family="esmfold2-fast",
    trunk_layers=24,
    msa_layers=0,
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
        PROTENIX1,
        PROTENIX2,
        OPENDDE,
        ESMFOLD2,
        ESMFOLD2_FAST,
    )
}
