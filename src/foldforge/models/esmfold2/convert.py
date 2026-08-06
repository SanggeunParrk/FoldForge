"""Remap released ESMFold2 weights onto team-gm's shared layers.

The two implementations compute the same functions but pack projections
differently: ESMFold2 fuses the triangle-multiplication input projections into
one ``proj_bundle`` and the SwiGLU gate/value into one ``w12``, while team-gm
keeps them as separate ``Linear`` modules so each can be swapped for a fused
kernel independently. Conversion is therefore pure slicing — no transposes, no
reordering within a slice.

``proj_bundle`` is ``[4 * d_hidden, d_pair]`` and splits as::

    [0 : d]        -> to_left          (signal, left)
    [d : 2d]       -> to_right         (signal, right)
    [2d : 3d]      -> to_left_gate     (gate, left)
    [3d : 4d]      -> to_right_gate    (gate, right)

matching the reference's ``bundled.split(2d)`` into (signal, gate) followed by
``routed.chunk(2)`` into (left, right).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from .config import ESMFold2Config

StateDict = dict[str, "Tensor"]


class MissingReferenceKeyError(KeyError):
    """Raised when the reference state dict lacks a key the mapping needs."""

    def __init__(self, key: str) -> None:
        msg = f"Reference state dict has no key '{key}'."
        super().__init__(msg)


def _take(reference: StateDict, key: str) -> Tensor:
    if key not in reference:
        raise MissingReferenceKeyError(key)
    return reference[key]


def convert_triangle_multiplication(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``TriangleMultiplicativeBlock`` engine.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix of the engine, e.g.
        ``"folding_trunk.blocks.0.tri_mul_out._engine."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.modules.layers.TriangleMultiplication`.

    """
    bundle = _take(reference, f"{prefix}proj_bundle.weight")
    d_hidden = bundle.shape[0] // 4
    return {
        "ln_pair.weight": _take(reference, f"{prefix}norm_start.weight"),
        "ln_pair.bias": _take(reference, f"{prefix}norm_start.bias"),
        "to_left.weight": bundle[:d_hidden],
        "to_right.weight": bundle[d_hidden : 2 * d_hidden],
        "to_left_gate.weight": bundle[2 * d_hidden : 3 * d_hidden],
        "to_right_gate.weight": bundle[3 * d_hidden :],
        "ln_out.weight": _take(reference, f"{prefix}norm_mix.weight"),
        "ln_out.bias": _take(reference, f"{prefix}norm_mix.bias"),
        "to_out.weight": _take(reference, f"{prefix}proj_emit.weight"),
        "to_gate.weight": _take(reference, f"{prefix}proj_gate.weight"),
    }


def convert_transition(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``Transition`` / ``PairTransition``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"folding_trunk.blocks.0.pair_transition."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.modules.layers.Transition`.

    """
    w12 = _take(reference, f"{prefix}ffn.w12.weight")
    hidden = w12.shape[0] // 2
    return {
        "ln_in.weight": _take(reference, f"{prefix}norm.weight"),
        "ln_in.bias": _take(reference, f"{prefix}norm.bias"),
        "expand_a.weight": w12[:hidden],
        "expand_b.weight": w12[hidden:],
        "squeeze.weight": _take(reference, f"{prefix}ffn.w3.weight"),
    }


def convert_outer_product_mean(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``OuterProductMean``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"msa_encoder.blocks.0.outer_product_mean."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.modules.layers.OuterProductMean`.

    """
    w = _take(reference, f"{prefix}W.weight")
    d_hidden = w.shape[0] // 2
    return {
        "ln_msa.weight": _take(reference, f"{prefix}norm.weight"),
        "ln_msa.bias": _take(reference, f"{prefix}norm.bias"),
        "to_left.weight": w[:d_hidden],
        "to_right.weight": w[d_hidden:],
        "to_out.weight": _take(reference, f"{prefix}Wout.weight"),
        "to_out.bias": _take(reference, f"{prefix}Wout.bias"),
    }


def convert_msa_pair_weighted_averaging(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``MSAPairWeightedAveraging``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g.
        ``"msa_encoder.blocks.0.msa_pair_weighted_averaging."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.modules.layers.MSAPairWeightedAveraging`.

    """
    return {
        "ln_msa.weight": _take(reference, f"{prefix}norm_single.weight"),
        "ln_msa.bias": _take(reference, f"{prefix}norm_single.bias"),
        "ln_pair.weight": _take(reference, f"{prefix}compute_bias.0.weight"),
        "ln_pair.bias": _take(reference, f"{prefix}compute_bias.0.bias"),
        "to_bias.weight": _take(reference, f"{prefix}compute_bias.1.weight"),
        "to_value.weight": _take(reference, f"{prefix}Wv.weight"),
        "to_gate.weight": _take(reference, f"{prefix}Wgate.weight"),
        "to_out.weight": _take(reference, f"{prefix}Wout.weight"),
    }


def _prefixed(mapping: StateDict, prefix: str) -> StateDict:
    return {f"{prefix}{k}": v for k, v in mapping.items()}


def convert_pair_update_block(reference: StateDict, prefix: str) -> StateDict:
    """Convert one trunk block (two triangle updates plus a transition).

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"folding_trunk.blocks.0."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.models.esmfold2.PairUpdateBlock`.

    """
    out: StateDict = {}
    for name in ("tri_mul_out", "tri_mul_in"):
        out.update(
            _prefixed(
                convert_triangle_multiplication(reference, f"{prefix}{name}._engine."),
                f"{name}.",
            )
        )
    out.update(
        _prefixed(
            convert_transition(reference, f"{prefix}pair_transition."),
            "pair_transition.",
        )
    )
    return out


def convert_folding_trunk(reference: StateDict, prefix: str, n_block: int) -> StateDict:
    """Convert a whole :class:`~team_gm.models.esmfold2.FoldingTrunk` stack.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Stack prefix, e.g. ``"folding_trunk."``, ``"lm_encoder."`` or
        ``"parcae_coda."``.
    n_block : int
        Number of blocks in the stack.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for the trunk module.

    """
    out: StateDict = {}
    for i in range(n_block):
        out.update(
            _prefixed(
                convert_pair_update_block(reference, f"{prefix}blocks.{i}."),
                f"blocks.{i}.",
            )
        )
    return out


def convert_msa_encoder(reference: StateDict, prefix: str, n_block: int) -> StateDict:
    """Convert a whole :class:`~team_gm.models.esmfold2.MSAEncoder`.

    The final block carries only the pair-side submodules; the reference drops
    its MSA-side weights, and so does this mapping.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Encoder prefix, normally ``"msa_encoder."``.
    n_block : int
        Number of blocks in the encoder.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for the MSA encoder module.

    """
    out: StateDict = {
        "embed.weight": _take(reference, f"{prefix}embed.weight"),
        "project_inputs.weight": _take(reference, f"{prefix}project_inputs.weight"),
    }
    for i in range(n_block):
        block_prefix = f"{prefix}blocks.{i}."
        block: StateDict = {}
        block.update(
            _prefixed(
                convert_outer_product_mean(
                    reference, f"{block_prefix}outer_product_mean."
                ),
                "outer_product_mean.",
            )
        )
        if i != n_block - 1:
            block.update(
                _prefixed(
                    convert_msa_pair_weighted_averaging(
                        reference, f"{block_prefix}msa_pair_weighted_averaging."
                    ),
                    "msa_pair_weighted_averaging.",
                )
            )
            block.update(
                _prefixed(
                    convert_transition(reference, f"{block_prefix}msa_transition."),
                    "msa_transition.",
                )
            )
        for name in ("tri_mul_out", "tri_mul_in"):
            block.update(
                _prefixed(
                    convert_triangle_multiplication(
                        reference, f"{block_prefix}{name}._engine."
                    ),
                    f"{name}.",
                )
            )
        block.update(
            _prefixed(
                convert_transition(reference, f"{block_prefix}pair_transition."),
                "pair_transition.",
            )
        )
        out.update(_prefixed(block, f"blocks.{i}."))
    return out


def convert_language_model_shim(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``LanguageModelShim``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, normally ``"language_model."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.models.esmfold2.embeddings.LanguageModelShim`.

    """
    mlp = f"{prefix}base_z_mlp."
    return {
        "ln_lm.weight": _take(reference, f"{prefix}base_z_linear.0.weight"),
        "ln_lm.bias": _take(reference, f"{prefix}base_z_linear.0.bias"),
        "to_pair_single.weight": _take(reference, f"{prefix}base_z_linear.1.weight"),
        "layer_weights": _take(reference, f"{prefix}base_z_combine"),
        "single_to_pair.downproject.weight": _take(
            reference, f"{mlp}0.downproject.weight"
        ),
        "single_to_pair.downproject.bias": _take(reference, f"{mlp}0.downproject.bias"),
        "single_to_pair.expand.weight": _take(reference, f"{mlp}0.output_mlp.0.weight"),
        "single_to_pair.expand.bias": _take(reference, f"{mlp}0.output_mlp.0.bias"),
        "single_to_pair.project.weight": _take(
            reference, f"{mlp}0.output_mlp.2.weight"
        ),
        "single_to_pair.project.bias": _take(reference, f"{mlp}0.output_mlp.2.bias"),
        "ln_pair.weight": _take(reference, f"{mlp}1.weight"),
        "ln_pair.bias": _take(reference, f"{mlp}1.bias"),
    }


def convert_row_attention_pooling(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``RowAttentionPooling``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"confidence_head.row_attention_pooling."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.models.esmfold2.embeddings.RowAttentionPooling`.

    """
    return {
        "to_score.weight": _take(reference, f"{prefix}attn_proj.weight"),
        "to_out.weight": _take(reference, f"{prefix}out_proj.weight"),
    }


def convert_pair_trunk(reference: StateDict, config: ESMFold2Config) -> StateDict:
    """Convert the whole pair trunk, including every stack it owns.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict (top-level keys, no prefix).
    config : ESMFold2Config
        Config the target :class:`~team_gm.models.esmfold2.PairTrunk` was built
        from; supplies each stack's depth.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for the pair trunk.

    """
    out: StateDict = {
        "z_init_row.weight": _take(reference, "z_init_1.weight"),
        "z_init_col.weight": _take(reference, "z_init_2.weight"),
        "rel_pos.embed.weight": _take(reference, "rel_pos.embed.weight"),
        "token_bonds.weight": _take(reference, "token_bonds.weight"),
        # The recurrence's continuous-time parameters.
        "ln_inject.weight": _take(reference, "parcae_input_norm.weight"),
        "ln_inject.bias": _take(reference, "parcae_input_norm.bias"),
        "log_decay_rate": _take(reference, "parcae_log_a"),
        "log_step": _take(reference, "parcae_log_delta"),
        "input_matrix": _take(reference, "parcae_b_cont"),
        "readout.weight": _take(reference, "parcae_readout.weight"),
        "distogram_head.weight": _take(reference, "distogram_head.weight"),
        "distogram_head.bias": _take(reference, "distogram_head.bias"),
    }
    out.update(
        _prefixed(
            convert_language_model_shim(reference, "language_model."), "language_model."
        )
    )
    out.update(
        _prefixed(
            convert_folding_trunk(
                reference, "folding_trunk.", config.folding_trunk.n_layers
            ),
            "folding_trunk.",
        )
    )
    out.update(
        _prefixed(
            convert_folding_trunk(
                reference, "parcae_coda.", config.parcae.coda_n_layers
            ),
            "coda.",
        )
    )
    if config.lm_encoder.enabled:
        out.update(
            _prefixed(
                convert_folding_trunk(
                    reference, "lm_encoder.", config.lm_encoder.n_layers
                ),
                "lm_encoder.",
            )
        )
    if config.msa_encoder.enabled:
        out.update(
            _prefixed(
                convert_msa_encoder(
                    reference, "msa_encoder.", config.msa_encoder.n_layers
                ),
                "msa_encoder.",
            )
        )
    return out


def convert_confidence_head(
    reference: StateDict, prefix: str, config: ESMFold2Config
) -> StateDict:
    """Convert the confidence head.

    ``{prefix}s_norm``, ``{prefix}s_inputs_to_single`` and
    ``{prefix}s_input_to_s`` are skipped: the release forward pass never reads
    them, so the team-gm head does not carry them.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, normally ``"confidence_head."``.
    config : ESMFold2Config
        Config the target head was built from; supplies its trunk depth.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.models.esmfold2.ConfidenceHead`.

    """
    out: StateDict = {
        "boundaries": _take(reference, f"{prefix}boundaries"),
        "distance_bin_embed.weight": _take(
            reference, f"{prefix}dist_bin_pairwise_embed.weight"
        ),
        "ln_single_inputs.weight": _take(reference, f"{prefix}s_inputs_norm.weight"),
        "ln_single_inputs.bias": _take(reference, f"{prefix}s_inputs_norm.bias"),
        "ln_pair.weight": _take(reference, f"{prefix}z_norm.weight"),
        "ln_pair.bias": _take(reference, f"{prefix}z_norm.bias"),
        "single_to_pair_row.weight": _take(reference, f"{prefix}s_to_z.weight"),
        "single_to_pair_col.weight": _take(
            reference, f"{prefix}s_to_z_transpose.weight"
        ),
        "single_to_pair_prod_row.weight": _take(
            reference, f"{prefix}s_to_z_prod_in1.weight"
        ),
        "single_to_pair_prod_col.weight": _take(
            reference, f"{prefix}s_to_z_prod_in2.weight"
        ),
        "single_to_pair_prod_out.weight": _take(
            reference, f"{prefix}s_to_z_prod_out.weight"
        ),
        "ln_plddt.weight": _take(reference, f"{prefix}plddt_ln.weight"),
        "ln_plddt.bias": _take(reference, f"{prefix}plddt_ln.bias"),
        "plddt_weight": _take(reference, f"{prefix}plddt_weight"),
        "ln_pae.weight": _take(reference, f"{prefix}pae_ln.weight"),
        "ln_pae.bias": _take(reference, f"{prefix}pae_ln.bias"),
        "pae_head.weight": _take(reference, f"{prefix}pae_head.weight"),
        "ln_pde.weight": _take(reference, f"{prefix}pde_ln.weight"),
        "ln_pde.bias": _take(reference, f"{prefix}pde_ln.bias"),
        "pde_head.weight": _take(reference, f"{prefix}pde_head.weight"),
        "ln_resolved.weight": _take(reference, f"{prefix}resolved_ln.weight"),
        "ln_resolved.bias": _take(reference, f"{prefix}resolved_ln.bias"),
        "resolved_weight": _take(reference, f"{prefix}resolved_weight"),
    }
    out.update(
        _prefixed(
            convert_folding_trunk(
                reference,
                f"{prefix}folding_trunk.",
                config.confidence_head.folding_trunk.n_layers,
            ),
            "folding_trunk.",
        )
    )
    out.update(
        _prefixed(
            convert_row_attention_pooling(reference, f"{prefix}row_attention_pooling."),
            "row_attention_pooling.",
        )
    )
    return out


def convert_atom_transformer(
    reference: StateDict, prefix: str, n_block: int
) -> StateDict:
    """Convert one SWA atom transformer stack.

    This is the identity: team-gm's
    :class:`~team_gm.modules.blocks.SWAAtomTransformer` already uses the
    reference's parameter names and shapes, so nothing is repacked. The function
    exists to keep every stack's weights flowing through one place and to fail
    loudly if the reference layout ever drifts.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Stack prefix, e.g.
        ``"inputs_embedder.atom_attention_encoder.atom_transformer."``.
    n_block : int
        Number of blocks in the stack.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for the atom transformer.

    """
    names = (
        "adaln_modulation.1.weight",
        "attn.Wqkv.weight",
        "attn.out_proj.weight",
        "attn.gate_proj.weight",
        "ffn.w_up.weight",
        "ffn.w_down.weight",
    )
    return {
        f"blocks.{i}.{name}": _take(reference, f"{prefix}blocks.{i}.{name}")
        for i in range(n_block)
        for name in names
    }


def convert_atom_encoder(
    reference: StateDict, prefix: str, n_block: int, *, structure_prediction: bool
) -> StateDict:
    """Convert one atom encoder.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Encoder prefix, e.g. ``"inputs_embedder.atom_attention_encoder."``.
    n_block : int
        Atom transformer depth.
    structure_prediction : bool
        Whether the encoder carries the coordinate projection.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.models.esmfold2.AtomEncoder`.

    """
    out: StateDict = {
        "atom_linear.weight": _take(reference, f"{prefix}atom_linear.weight"),
        "atom_norm.weight": _take(reference, f"{prefix}atom_norm.weight"),
        "atom_norm.bias": _take(reference, f"{prefix}atom_norm.bias"),
        "atom_to_token_linear.weight": _take(
            reference, f"{prefix}atom_to_token_linear.weight"
        ),
    }
    if structure_prediction:
        out["coords_linear.weight"] = _take(reference, f"{prefix}coords_linear.weight")
    out.update(
        _prefixed(
            convert_atom_transformer(reference, f"{prefix}atom_transformer.", n_block),
            "atom_transformer.",
        )
    )
    return out


def convert_inputs_embedder(
    reference: StateDict, prefix: str, config: ESMFold2Config
) -> StateDict:
    """Convert the token-level input embedder.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Embedder prefix, normally ``"inputs_embedder."``.
    config : ESMFold2Config
        Config the target embedder was built from.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.models.esmfold2.InputsEmbedder`.

    """
    return _prefixed(
        convert_atom_encoder(
            reference,
            f"{prefix}atom_attention_encoder.",
            config.inputs.atom_encoder.n_blocks,
            structure_prediction=False,
        ),
        "atom_encoder.",
    )


def convert_adaptive_layer_norm(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``AdaptiveLayerNorm``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"...attn_blocks.0.adaln."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.modules.layers.AdaptiveLayerNorm`.

    """
    return {
        "ln_cond.weight": _take(reference, f"{prefix}s_scale"),
        "to_scale.weight": _take(reference, f"{prefix}s_gate.weight"),
        "to_scale.bias": _take(reference, f"{prefix}s_gate.bias"),
        "to_bias.weight": _take(reference, f"{prefix}s_shift.weight"),
    }


def convert_attention_pair_bias(reference: StateDict, prefix: str) -> StateDict:
    """Convert one conditioned ``AttentionPairBias``.

    ``{prefix}pair_norm.bias`` is carried across. It is mathematically inert —
    it reaches the logits only through a bias-free projection, so it adds a
    per-head constant identical for every ``(i, j)`` and cancels in the softmax
    — but the engine's ``ln_pair`` has a bias parameter, and leaving it
    unwritten would silently substitute zeros for a trained tensor. Dropping a
    weight because it "does not matter" is an argument that has to be re-checked
    every time the consumer changes; copying it is not.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"...token_transformer.attn_blocks.0."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~miniworld_engine.modules.AugmentedAttentionPairBias`.

    """
    kv = _take(reference, f"{prefix}kv_proj.weight")
    width = kv.shape[0] // 2
    out = {
        "to_query.weight": _take(reference, f"{prefix}q_proj.weight"),
        "to_query.bias": _take(reference, f"{prefix}q_proj.bias"),
        "to_key.weight": kv[:width],
        "to_value.weight": kv[width:],
        "ln_pair.weight": _take(reference, f"{prefix}pair_norm.weight"),
        "ln_pair.bias": _take(reference, f"{prefix}pair_norm.bias"),
        "to_bias.weight": _take(reference, f"{prefix}pair_bias_proj.weight"),
        "to_gate.weight": _take(reference, f"{prefix}g_proj.weight"),
        "to_out.weight": _take(reference, f"{prefix}out_proj.weight"),
        "to_scale.weight": _take(reference, f"{prefix}out_gate.weight"),
        "to_scale.bias": _take(reference, f"{prefix}out_gate.bias"),
    }
    out.update(
        _prefixed(
            convert_adaptive_layer_norm(reference, f"{prefix}adaln."), "ada_ln_in."
        )
    )
    return out


def convert_conditioned_transition(reference: StateDict, prefix: str) -> StateDict:
    """Convert one ``ConditionedTransitionBlock``.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"...token_transformer.transition_blocks.0."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.modules.layers.ConditionedTransition`.

    """
    swish = _take(reference, f"{prefix}lin_swish.weight")
    hidden = swish.shape[0] // 2
    out = {
        "expand_a.weight": swish[:hidden],
        "expand_b.weight": swish[hidden:],
        "squeeze.weight": _take(reference, f"{prefix}lin_out.weight"),
        "to_scale.weight": _take(reference, f"{prefix}output_gate.weight"),
        "to_scale.bias": _take(reference, f"{prefix}output_gate.bias"),
    }
    out.update(
        _prefixed(
            convert_adaptive_layer_norm(reference, f"{prefix}adaln."), "ada_ln_in."
        )
    )
    return out


def convert_diffusion_transformer(
    reference: StateDict, prefix: str, n_block: int
) -> StateDict:
    """Convert the token-level diffusion transformer stack.

    The reference keeps attention and transition in two parallel lists; team-gm
    pairs them inside one block, so the indices are re-interleaved here.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Stack prefix, e.g. ``"...diffusion_module.token_transformer."``.
    n_block : int
        Number of blocks.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.modules.blocks.DiffusionTransformer`.

    """
    out: StateDict = {}
    for i in range(n_block):
        out.update(
            _prefixed(
                convert_attention_pair_bias(reference, f"{prefix}attn_blocks.{i}."),
                f"blocks.{i}.attention_pair_bias.",
            )
        )
        out.update(
            _prefixed(
                convert_conditioned_transition(
                    reference, f"{prefix}transition_blocks.{i}."
                ),
                f"blocks.{i}.transition.",
            )
        )
    return out


def convert_transition_layer(reference: StateDict, prefix: str) -> StateDict:
    """Convert one reference ``TransitionLayer`` (unpacked a/b projections).

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"...conditioning.z_transitions.0."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.modules.layers.Transition`.

    """
    return {
        "ln_in.weight": _take(reference, f"{prefix}norm.weight"),
        "ln_in.bias": _take(reference, f"{prefix}norm.bias"),
        "expand_a.weight": _take(reference, f"{prefix}a_proj.weight"),
        "expand_b.weight": _take(reference, f"{prefix}b_proj.weight"),
        "squeeze.weight": _take(reference, f"{prefix}out_proj.weight"),
    }


def convert_diffusion_conditioning(reference: StateDict, prefix: str) -> StateDict:
    """Convert the noise-conditioning module.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"...diffusion_module.conditioning."``.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for
        :class:`~team_gm.models.esmfold2.diffusion.DiffusionConditioning`.

    """
    out: StateDict = {
        "ln_pair_in.weight": _take(reference, f"{prefix}z_input_norm.weight"),
        "ln_pair_in.bias": _take(reference, f"{prefix}z_input_norm.bias"),
        "pair_proj.weight": _take(reference, f"{prefix}z_proj.weight"),
        "ln_single_in.weight": _take(reference, f"{prefix}s_input_norm.weight"),
        "ln_single_in.bias": _take(reference, f"{prefix}s_input_norm.bias"),
        "single_proj.weight": _take(reference, f"{prefix}s_proj.weight"),
        "fourier.w": _take(reference, f"{prefix}fourier.w"),
        "fourier.b": _take(reference, f"{prefix}fourier.b"),
        "ln_noise.weight": _take(reference, f"{prefix}noise_norm.weight"),
        "ln_noise.bias": _take(reference, f"{prefix}noise_norm.bias"),
        "noise_proj.weight": _take(reference, f"{prefix}noise_proj.weight"),
    }
    for i in range(2):
        out.update(
            _prefixed(
                convert_transition_layer(reference, f"{prefix}z_transitions.{i}."),
                f"pair_transitions.{i}.",
            )
        )
        out.update(
            _prefixed(
                convert_transition_layer(reference, f"{prefix}s_transitions.{i}."),
                f"single_transitions.{i}.",
            )
        )
    return out


def convert_atom_decoder(reference: StateDict, prefix: str, n_block: int) -> StateDict:
    """Convert the atom decoder.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, e.g. ``"...diffusion_module.atom_decoder."``.
    n_block : int
        Atom transformer depth.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.models.esmfold2.AtomDecoder`.

    """
    out: StateDict = {
        "token_to_atom_linear.weight": _take(
            reference, f"{prefix}token_to_atom_linear.weight"
        ),
        "norm.weight": _take(reference, f"{prefix}norm.weight"),
        "norm.bias": _take(reference, f"{prefix}norm.bias"),
        "output_linear.weight": _take(reference, f"{prefix}output_linear.weight"),
    }
    out.update(
        _prefixed(
            convert_atom_transformer(reference, f"{prefix}atom_transformer.", n_block),
            "atom_transformer.",
        )
    )
    return out


def convert_diffusion_module(
    reference: StateDict, prefix: str, config: ESMFold2Config
) -> StateDict:
    """Convert the whole diffusion denoiser.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    prefix : str
        Key prefix, normally ``"structure_head.diffusion_module."``.
    config : ESMFold2Config
        Config the target module was built from.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.models.esmfold2.DiffusionModule`.

    """
    dm = config.structure_head.diffusion_module
    out: StateDict = {
        "single_to_token.weight": _take(reference, f"{prefix}s_to_token.weight"),
        "ln_single_step.weight": _take(reference, f"{prefix}s_step_norm.weight"),
        "ln_single_step.bias": _take(reference, f"{prefix}s_step_norm.bias"),
        "ln_token.weight": _take(reference, f"{prefix}token_norm.weight"),
        "ln_token.bias": _take(reference, f"{prefix}token_norm.bias"),
    }
    out.update(
        _prefixed(
            convert_diffusion_conditioning(reference, f"{prefix}conditioning."),
            "conditioning.",
        )
    )
    out.update(
        _prefixed(
            convert_atom_encoder(
                reference,
                f"{prefix}atom_encoder.",
                dm.atom_num_blocks,
                structure_prediction=True,
            ),
            "atom_encoder.",
        )
    )
    out.update(
        _prefixed(
            convert_atom_decoder(
                reference, f"{prefix}atom_decoder.", dm.atom_num_blocks
            ),
            "atom_decoder.",
        )
    )
    out.update(
        _prefixed(
            convert_diffusion_transformer(
                reference, f"{prefix}token_transformer.", dm.token_num_blocks
            ),
            "token_transformer.",
        )
    )
    return out


def convert_model(reference: StateDict, config: ESMFold2Config) -> StateDict:
    """Convert a whole released ESMFold2 checkpoint.

    Parameters
    ----------
    reference : dict[str, Tensor]
        Released ESMFold2 state dict.
    config : ESMFold2Config
        Config the target model was built from.

    Returns
    -------
    dict[str, Tensor]
        Weights keyed for :class:`~team_gm.models.esmfold2.ESMFold2Model`.

    """
    out: StateDict = {}
    out.update(
        _prefixed(
            convert_inputs_embedder(reference, "inputs_embedder.", config),
            "inputs_embedder.",
        )
    )
    out.update(_prefixed(convert_pair_trunk(reference, config), "pair_trunk."))
    out.update(
        _prefixed(
            convert_diffusion_module(
                reference, "structure_head.diffusion_module.", config
            ),
            "structure_head.diffusion_module.",
        )
    )
    out.update(
        _prefixed(
            convert_confidence_head(reference, "confidence_head.", config),
            "confidence_head.",
        )
    )
    return out
