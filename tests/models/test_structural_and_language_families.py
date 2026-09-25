"""The two families whose differences a parameter tree cannot see.

OpenDDE folds on structural tokens and ESMFold2 folds from a language model.
Both read exact on a scope diff while carrying differences that live entirely
in the forward pass -- which is the way Chai-1 once passed every gate and still
folded to 18 A. Everything here is one of those differences, pinned.
"""

from __future__ import annotations

import pytest
import torch

from foldforge.modules.dense.spec import SPECS

#: Both releases of the family that folds from a language model. Membership is
#: stated per FAMILY, not per release: a convention named on one release is a
#: convention the next release silently loses, which is the mistake one shared
#: filename already invited here once.
_LANGUAGE_MODEL_FAMILY = frozenset({"esmfold2", "esmfold2-fast"})


def test_esm_class_widening_is_a_permutation_not_a_shift():
    """ESMFold2 puts the gap BELOW the residues; AF3 puts it above them.

    Padding with leading zeros instead lands AF3's gap column on the first
    nucleic class and shifts every nucleic class down one -- invisible to any
    single-sequence protein fold, because both are then identically zero.
    """
    from foldforge.modules.dense.featurization import widen_to_esm_classes

    features = torch.arange(2 * 447, dtype=torch.float32).reshape(2, 447)
    wide = widen_to_esm_classes(features)
    assert wide.shape == (2, 451)

    for start, target in ((0, 0), (447 - 447 + 31, 33)):
        block = features[:, start : start + 31]
        out = wide[:, target : target + 33]
        assert torch.equal(out[:, 0], torch.zeros(2))  # ESM slot 0 is unused
        assert torch.equal(out[:, 1], block[:, 21])  # the gap, below the residues
        assert torch.equal(out[:, 2:23], block[:, :21])  # residues and UNK
        assert torch.equal(out[:, 23:32], block[:, 22:31])  # the nucleic acids
        assert torch.equal(out[:, 32], torch.zeros(2))  # DN, which AF3 lacks

    # Deletion mean and the atom encoder's token output ride through untouched.
    assert torch.equal(wide[:, 66:], features[:, 62:])


def test_rotary_tables_fill_half_the_atom_head_exactly():
    """3 axes x 2 spatial pairs + 10 uid pairs = 16 = head_dim / 2."""
    from foldforge.modules.dense.atom_cross_attention import build_atom_rope

    spec = SPECS["esmfold2"]
    assert spec.atom_rope is not None
    cos, sin = build_atom_rope(
        torch.randn(3, 5, 3), torch.arange(15).reshape(3, 5), 32, spec.atom_rope
    )
    assert cos.shape == sin.shape == (3, 5, 16)
    # No padding was needed, which is what "exactly" means: a table short of
    # half the head would be zero-filled and rotate nothing on those channels.
    assert not torch.equal(cos[..., -1], torch.ones(3, 5))


def test_rotary_is_tiled_not_interleaved():
    """`[c|c]` pairs with the split-into-halves rotation.

    Interleaving instead reads correlation 0.88 on the atom encoder: high
    enough to look like noise, low enough to ruin the fold.
    """
    from foldforge.modules.dense.diffusion_transformer import apply_rope

    x = torch.randn(2, 3, 1, 8)
    cos, sin = torch.ones(2, 3, 4), torch.zeros(2, 3, 4)
    # At sin = 0 the rotation is the identity, whatever the tiling.
    assert torch.allclose(apply_rope(x, cos, sin), x)

    # At cos = 0, sin = 1 it is the half-swap: [a, b] -> [-b, a].
    out = apply_rope(x, torch.zeros(2, 3, 4), torch.ones(2, 3, 4))
    a, b = torch.split(x, 4, dim=-1)
    assert torch.allclose(out, torch.concatenate([-b, a], dim=-1))


def test_sliding_window_counts_valid_atoms_not_slots():
    """The window is +/-N by RANK among valid atoms, not by position."""
    from foldforge.modules.dense import atom_layout
    from foldforge.modules.dense.atom_cross_attention import sliding_window_mask

    queries_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    identity = atom_layout.GatherInfo(
        gather_idxs=torch.arange(4)[None],
        gather_mask=queries_mask.to(torch.bool),
        input_shape=torch.tensor([1, 4]),
    )
    mask = sliding_window_mask(queries_mask, queries_mask, identity, half=1)

    # Slot 1 is padding, so slots 0, 2 and 3 have ranks 0, 1 and 2: slot 0 and
    # slot 2 are ADJACENT by rank though two apart by position.
    assert bool(mask[0, 0, 2])
    assert not bool(mask[0, 0, 3])
    # Nothing attends to or from a padded slot.
    assert not bool(mask[0, 1].any())
    assert not bool(mask[0, :, 1].any())


def test_rigid_align_recovers_a_known_rotation_and_translation():
    from foldforge.models.architectures.af3 import weighted_rigid_align

    torch.manual_seed(0)
    reference = torch.randn(4, 5, 3)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        # A reflection is not a rotation, and the alignment must not return one.
        rotation[:, 0] *= -1
    mobile = reference @ rotation + torch.tensor([3.0, -1.0, 7.0])
    aligned = weighted_rigid_align(mobile, reference, torch.ones(4, 5))
    assert torch.allclose(aligned, reference, atol=1e-4)


def test_rigid_align_gives_every_sample_its_own_rotation():
    """A leading axis is a separate structure, not more atoms of this one.

    The alignment flattened all leading axes into the atom list, so a request
    for five diffusion samples crashed -- 5x3072 weighted against 3072 -- and
    the crash was the lucky outcome. Had the shapes broadcast it would have
    solved one Kabsch problem across every sample at once, silently.
    """
    from foldforge.models.architectures.af3 import weighted_rigid_align

    torch.manual_seed(11)
    reference = torch.randn(4, 5, 3)
    weight = (torch.rand(4, 5) > 0.3).float()
    rotations = []
    for _ in range(3):
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        rotations.append(rotation)
    samples = torch.stack([reference @ r + float(i) for i, r in enumerate(rotations)])

    batched = weighted_rigid_align(samples, reference.expand(3, 4, 5, 3), weight)
    looped = torch.stack(
        [weighted_rigid_align(samples[i], reference, weight) for i in range(3)]
    )
    assert batched.shape == samples.shape
    # Close, not equal: a batched SVD reduces in a different order from three
    # separate ones, so on CUDA the two agree to float precision and not to the
    # bit. Demanding equality here passed on CPU and failed on a GPU node.
    torch.testing.assert_close(batched, looped, atol=1e-5, rtol=1e-5)
    # Each sample came back to the reference, which one shared rotation across
    # three different rotations could not do.
    assert torch.allclose(
        batched * weight[..., None], reference * weight[..., None], atol=1e-4
    )


def test_structural_diffusion_output_returns_to_the_residue_layout():
    """Exact, not a choice: every residue atom sits in one structural slot."""
    from foldforge.models.architectures.af3 import _to_residue_layout

    max_atoms = 4
    # Residue 0 splits into structural tokens 0 and 1; residue 1 does not.
    gather = torch.tensor(
        [[0, 1, 1 * max_atoms + 0, -1], [2 * max_atoms + 0, -1, -1, -1]]
    )
    positions = torch.arange(3 * max_atoms * 3, dtype=torch.float32)
    positions = positions.reshape(1, 3, max_atoms, 3)
    out = _to_residue_layout(
        {"atom_positions": positions}, {"structbook/residue_atom_gather": gather}, 2
    )["atom_positions"]

    flat = positions.reshape(1, -1, 3)
    assert torch.equal(out[0, 0, 0], flat[0, 0])
    assert torch.equal(out[0, 0, 2], flat[0, max_atoms])
    assert torch.equal(out[0, 1, 0], flat[0, 2 * max_atoms])
    # Padded slots carry nothing rather than whatever index -1 would reach.
    assert torch.equal(out[0, 0, 3], torch.zeros(3))
    assert torch.equal(out[0, 1, 1:], torch.zeros(3, 3))


def test_a_family_that_folds_from_a_language_model_refuses_to_fold_without_one():
    """Silence here is the failure mode, not the exception.

    Without its language model this family still folds, and folds to something
    plausible; the reference ran four models that way for weeks, and what gave
    it away was that supplying the tower changed the answer by nothing at all.
    """
    from foldforge.models.architectures.af3 import Evoformer

    spec = SPECS["esmfold2"]
    assert spec.language_model is not None
    with torch.device("meta"):
        evoformer = Evoformer(spec)
    batch = type("Batch", (), {"lm_pair": None})()
    with pytest.raises(ValueError, match="folds from a language model"):
        evoformer._embed_lm_pair(batch, torch.zeros(2, 2, 256), torch.ones(2, 2))  # noqa: SLF001 - the guard is the subject


@pytest.mark.parametrize(
    ("residue", "expected"),
    [("A", "N9"), ("G", "N9"), ("C", "N1"), ("U", "N1"), ("DT", "N1"), ("DG", "N9")],
)
def test_nucleic_base_centre_follows_the_ring_system(residue, expected):
    """Purines carry an N1 too, in the six-membered ring.

    One preference list cannot express this: a plain [N1, N9, ...] silently
    picks N1 for every purine -- the wrong representative atom on about half of
    all nucleotides, and invisible to protein.
    """
    from foldforge.data.features.structural_tokens import _base_centre

    assert _base_centre(residue)[0] == expected


def test_structural_roles_split_every_standard_residue_but_glycine():
    from foldforge.data.features import structural_tokens as st

    assert st._BACKBONE["protein"] == frozenset(["N", "CA", "C", "O", "OXT"])  # noqa: SLF001 - the table is the subject
    # Glycine is standard and splits to one token because its sidechain set is
    # empty, not because it is excluded from the standard table.
    assert "GLY" in st._STANDARD["protein"]  # noqa: SLF001 - the table is the subject


def test_the_language_model_shim_mixes_the_tower_s_last_layers():
    """The mix peaks on the LAST states, so the tower cannot be truncated."""
    from foldforge.models.checkpoints import DEFAULT_DIR
    from foldforge.modules.language_model import load_pair_shim

    path = DEFAULT_DIR / "esmfold2" / "esmfold2.lm.npz"
    if not path.is_file():
        pytest.skip("the shim ships beside converted ESMFold2 weights")
    shim = load_pair_shim(path)
    assert float(shim.combine.sum()) == pytest.approx(1.0, abs=1e-5)
    assert set(torch.topk(shim.combine, 3).indices.tolist()) == {78, 79, 80}
    pair = shim(torch.randn(6, shim.combine.shape[0], 2560))
    assert pair.shape == (6, 6, 256)
    assert bool(torch.isfinite(pair).all())


def test_the_language_model_packs_the_protein_tokens_it_is_given():
    """The selector is a BOOLEAN, and an empty pack is an error, not a fold.

    It used to be a molecule-type CLASS compared against ``PROTEIN_MOL_TYPE``,
    which is 0, while the caller had ``is_protein``, which is 1 on protein.
    So the comparison selected exactly the tokens it meant to drop: on an
    all-protein input nothing was packed, the tower saw nothing, and the shim
    turned its zeros into ONE constant vector repeated at all 16384 pair
    positions. Every ESMFold2 fold ran without its language model and still
    produced a structure, which is why nothing caught it.
    """
    from foldforge.modules.language_model import build_lm_inputs

    protein = torch.tensor([[True, True, True, False]])
    ids = torch.tensor([[5, 7, 9, 3]])
    asym = torch.zeros(1, 4, dtype=torch.long)
    residue = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones(1, 4, dtype=torch.bool)

    packed, _sequence_id, position_map = build_lm_inputs(
        ids, asym, residue, protein, mask
    )
    # The three protein tokens reached the tower and the fourth did not.
    assert (position_map[0, :3] >= 0).all()
    assert int(position_map[0, 3]) == -1
    assert packed.shape[0] == 1

    # An all-protein input is the common case and must pack everything.
    everything = build_lm_inputs(
        ids, asym, residue, torch.ones(1, 4, dtype=torch.bool), mask
    )[2]
    assert (everything[0] >= 0).all()

    # Nothing to pack is an error: silence here is what hid the mix-up.
    with pytest.raises(ValueError, match="No protein token"):
        build_lm_inputs(ids, asym, residue, torch.zeros(1, 4, dtype=torch.bool), mask)


def test_pair_only_families_build_neither_single_track_nor_pair_attention():
    from foldforge.models.architectures.af3 import Evoformer

    spec = SPECS["esmfold2"]
    with torch.device("meta"):
        evoformer = Evoformer(spec)
    block = evoformer.trunk_pairformer[0]
    assert not block.with_single
    assert not block.with_pair_attention
    assert not hasattr(block, "pair_attention1")
    # ...but the single is still PROJECTED and recycled: the checkpoint carries
    # both projections, the trunk simply never updates it.
    assert evoformer.single_activations is not None
    assert evoformer.template_embedding is None
    assert len(evoformer.trunk_coda) == spec.coda_layers
    assert len(evoformer.lm_encoder) == spec.lm_encoder_layers


def test_only_the_atom_blocks_take_the_rms_conditioning():
    """Gating on the family alone would strip the token blocks that have it."""
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.modules.dense.diffusion_transformer import RMSNorm

    with torch.device("meta"):
        model = AlphaFold3(spec=SPECS["esmfold2"])
    atom = model.diffusion_head.atom_cross_att_encoder.atom_transformer_encoder
    token = model.diffusion_head.transformer

    assert isinstance(atom.cross_attention[0].q_adaptive_layernorm.layer_norm, RMSNorm)
    assert atom.cross_attention[0].adaptive_zero_init.raw_gate
    assert not isinstance(
        token.self_attention[0].adaptive_layernorm.layer_norm, RMSNorm
    )
    assert not token.self_attention[0].adaptive_zero_init.raw_gate


def test_structural_token_families_carry_their_expander_and_refiner():
    from foldforge.models.architectures.af3 import Evoformer

    spec = SPECS["opendde"]
    with torch.device("meta"):
        evoformer = Evoformer(spec)
    assert evoformer.structural_token_expander is not None
    assert len(evoformer.structural_token_refiner) == spec.structural_refiner_layers
    block = evoformer.structural_token_refiner[0]
    # The refiner narrows only its PAIR transition; the single one keeps the
    # trunk's factor, which is what upstream hardcodes.
    assert block.num_intermediate_factor == spec.structural_refiner_transition_factor
    assert block.n_heads == spec.structural_refiner_heads

    # Every other family builds neither.
    with torch.device("meta"):
        plain = Evoformer(SPECS["alphafold3"])
    assert plain.structural_token_expander is None


def test_only_the_language_model_family_widens_its_key_subset():
    """A window of +/-64 by rank needs a subset that contains it."""
    wide = {name for name, spec in SPECS.items() if spec.atom_keys_subset != 128}
    assert wide == _LANGUAGE_MODEL_FAMILY
    for name in _LANGUAGE_MODEL_FAMILY:
        assert SPECS[name].atom_keys_subset == 192
        assert SPECS[name].atom_window_half == 64


def test_no_spec_field_is_declared_and_never_read():
    """The trap this port has hit three times.

    A field that nothing reads is a difference someone wrote down and did not
    wire, and it fails as a quietly worse fold rather than as an error.

    A substring scan for the ATTRIBUTE access, so a row that merely sets the
    field does not count as reading it -- `msa_feat_columns=41` has no dot.
    It can miss a `getattr`, which makes this a guard rather than a proof.
    """
    import dataclasses
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "foldforge"
    sources = "\n".join(path.read_text() for path in root.rglob("*.py"))
    unread = [
        field.name
        for field in dataclasses.fields(SPECS["alphafold3"])
        if field.name != "family" and f".{field.name}" not in sources
    ]
    assert unread == []


def test_keep_query_subsampling_never_drops_the_query():
    """AF3's shuffle-then-truncate drops it about half the time.

    The gumbel shuffle ranks EVERY row including row 0, so the query lands
    somewhere random and the truncation then discards it with probability
    1 - num_msa/depth. A trunk handed an alignment with no query in it folds
    WORSE the moment it is given a real one, which is how this was found.
    """
    from foldforge.data.features import dense as features
    from foldforge.modules.dense import featurization

    depth, width, keep = 40, 6, 8
    # Row i is all i, so a subsample can be read straight off the first column.
    rows = torch.arange(depth).reshape(depth, 1).repeat(1, width)
    msa = features.MSA(
        rows=rows,
        mask=torch.ones(depth, width),
        deletion_matrix=torch.zeros(depth, width),
        profile=torch.zeros(width, 31),
        deletion_mean=torch.zeros(width),
        num_alignments=torch.tensor(depth),
    )

    torch.manual_seed(0)
    draws = set()
    for _ in range(8):
        index = featurization.subsample_msa_keep_query(msa, keep).rows[:, 0].tolist()
        assert index[0] == 0, "the query must stay at row 0"
        assert index == sorted(index), "an a3m's row order is information"
        assert len(set(index)) == keep, "row 0 must not also be drawn into the tail"
        draws.add(tuple(index))
    assert len(draws) > 1, "the tail is a random subset, not a truncation"


def test_the_three_msa_subsampling_policies_are_stated_apart():
    """They differ only ABOVE the depth limit, so one gate cannot tell them apart."""
    policies = {name: spec.msa_subsample for name, spec in SPECS.items()}
    assert policies["alphafold3"] == "shuffle"
    assert policies["chai1"] == "ordered"
    assert policies["esmfold2"] == "keep_query"
    assert set(policies.values()) <= {"shuffle", "ordered", "keep_query"}


def test_the_relative_chain_bucket_flips_every_pair_of_a_monomer():
    """Keyed on same-CHAIN sending the MATCH to the pad class, not AF3's entity.

    Same-chain is universally true on a monomer, so one convention makes the
    whole one-hot the pad class and the other the zero-offset class. A constant
    column either way -- but a DIFFERENT one into a trained projection, and the
    reference measured it at 1.522 -> 0.719 A when it was first found.
    """
    from foldforge.data.features import dense as features
    from foldforge.modules.dense import featurization

    n = 4
    tokens = features.TokenFeatures(
        residue_index=torch.arange(n),
        token_index=torch.arange(n),
        asym_id=torch.ones(n),
        entity_id=torch.ones(n),
        sym_id=torch.zeros(n),
        aatype=torch.zeros(n, dtype=torch.long),
        is_protein=torch.ones(n),
        is_rna=torch.zeros(n),
        is_dna=torch.zeros(n),
        is_ligand=torch.zeros(n),
        is_water=torch.zeros(n),
        is_nonstandard_polymer_chain=torch.zeros(n),
        mask=torch.ones(n),
        seq_length=torch.tensor(n),
    )
    entity = featurization.create_relative_encoding(tokens, 32, 2)
    chain = featurization.create_relative_encoding(
        tokens, 32, 2, chain_bucket_on_same_chain=True
    )
    assert entity.shape[-1] == chain.shape[-1] == 139
    assert entity[0, 0, -6:].tolist() == [0, 0, 1, 0, 0, 0]
    assert chain[0, 0, -6:].tolist() == [0, 0, 0, 0, 0, 1]
    assert bool((entity != chain).any(-1).all())

    families = {n for n, s in SPECS.items() if s.chain_bucket_on_same_chain}
    # Boltz-2 looked like a member until its own checkpoint settled it: its
    # `fix_sym_check` flag IS this convention, and True means AF3's.
    assert families == _LANGUAGE_MODEL_FAMILY


def test_the_outer_product_mean_is_afs_for_every_family():
    """One OPM normalisation: AF3's mean over rows, 1e-3 offset, bias first.

    The clamped divisor and the bias-after-divide some releases use were
    unified: resetting either moved no fold beyond 0.1 A.
    """
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

    build = lambda **kw: primitives.OuterProductMean(
        c_msa=8, num_output_channel=4, num_outer_channel=3, **kw
    )

    af3 = build()
    out, count = parts(af3)
    expected = (out + af3.output_b).permute(1, 0, 2) / (af3.epsilon + count)
    assert torch.allclose(af3(msa, mask), expected, atol=1e-6)


def test_every_pde_symmetrisation_is_named():
    """Summing a logit with its transpose doubles an expectation."""
    named = {name: spec.pde_symmetrise for name, spec in SPECS.items()}
    assert named["alphafold3"] == "logits"
    assert named["protenix2"] == "pair"
    assert named["esmfold2"] == "none"
    assert set(named.values()) <= {"logits", "pair", "none"}


def test_the_reference_charge_convention_covers_every_family_that_takes_it_raw():
    raw = {name for name, spec in SPECS.items() if spec.raw_ref_charge}
    assert raw == {"chai1", "boltz2", "rosettafold3"} | _LANGUAGE_MODEL_FAMILY


def test_the_reference_geometry_override_replaces_by_atom_name():
    """Its atom encoder was trained on ITS frame, and ref_pos is not pose-invariant."""
    import numpy as np

    from foldforge.data.constants import residue_geometry
    from foldforge.data.features.dense_conventions import override_ref_conformers

    frame = residue_geometry.as_conformers("esmfold2")
    names, positions = frame["ALA"]
    encode = lambda atom: [ord(c) - 32 for c in atom.ljust(4)]

    # Two tokens: an alanine the table covers, and a ligand atom it does not.
    example = {
        "ref_pos": np.zeros((2, 4, 3), np.float32),
        "ref_mask": np.array([[1, 1, 0, 0], [1, 0, 0, 0]], np.float32),
        "ref_atom_name_chars": np.array(
            [
                [encode(names[0]), encode(names[1]), encode(""), encode("")],
                [encode("ZZ9"), encode(""), encode(""), encode("")],
            ]
        ),
        # Alanine's index in AF3's polymer table, then something past the end.
        "aatype": np.array([0, 40]),
    }
    replaced = override_ref_conformers(example, frame)

    assert replaced == 2
    assert np.allclose(example["ref_pos"][0, 0], positions[0])
    assert np.allclose(example["ref_pos"][0, 1], positions[1])
    # A residue the table does not carry keeps what it had, rather than zeroing.
    assert np.allclose(example["ref_pos"][1], 0.0)
    # A masked slot is never written, even where the name would match.
    assert np.allclose(example["ref_pos"][0, 2:], 0.0)


def test_the_language_model_family_takes_every_featurisation_knob_it_needs():
    """Getting one wrong is silent, and shows as a fold that is merely mediocre."""
    spec = SPECS["esmfold2"]
    # No terminal OXT: worse here than elsewhere, because the atom window is
    # +/-64 by RANK, so one spurious atom at the END corrupts the last ~64.
    assert spec.drop_atoms == ("OXT",)
    # Its self-MSA is the query ONCE; AF3 hands it two identical rows.
    assert spec.dedupe_self_msa
    assert spec.ref_conformers == "esmfold2"
    assert spec.atom_keys_subset == 192


def test_a_family_with_no_msa_stack_builds_neither_the_stack_nor_its_inputs():
    """Skipping the CALL, not only the blocks, is what keeps the tree clean.

    Layer-stack scope names are POSITIONAL, so dropping the MSA stack also
    shifts the trunk and the coda down one -- which is a fact about the NAMES,
    and a blob written one way cannot be read the other.
    """
    from foldforge.models.architectures.af3 import Evoformer

    with torch.device("meta"):
        fast = Evoformer(SPECS["esmfold2-fast"])
        full = Evoformer(SPECS["esmfold2"])

    assert SPECS["esmfold2-fast"].msa_layers == 0
    assert fast.msa_stack is None
    assert not hasattr(fast, "msa_activations")
    assert not hasattr(fast, "extra_msa_target_feat")

    assert full.msa_stack is not None
    assert len(full.msa_stack) == SPECS["esmfold2"].msa_layers
    # The two releases differ in exactly the trunk depth and the MSA stack.
    assert SPECS["esmfold2-fast"].trunk_layers == 24
    assert SPECS["esmfold2"].trunk_layers == 48


def test_the_confidence_head_branches_on_facts_not_on_family_names():
    """One selector was gating several unrelated behaviours.

    `confidence` now picks a STRUCTURE -- whether the pair is re-embedded from
    the inputs or read from the trunk -- and everything a family does on top of
    that is its own field. Membership is asserted because the split has to be
    behaviour-preserving: these are exactly the families the old strings named.
    """
    import pathlib as _pathlib

    named = lambda field: {name for name, spec in SPECS.items() if getattr(spec, field)}
    protenix = {"protenix1", "protenix2", "opendde"}
    assert named("confidence_single_clamp") == protenix
    assert named("confidence_raw_distance") == protenix
    assert named("confidence_global_norm") == {"rosettafold3"}
    assert named("confidence_centre_dgram") == {"rosettafold3"}

    # ...and the head no longer decides anything by a family's name, except the
    # one comparison that genuinely selects which embedding was trained.
    head = (
        _pathlib.Path(__file__).resolve().parents[2]
        / "src/foldforge/modules/dense/head.py"
    )
    source = head.read_text()
    names = [n for n in SPECS if f'== "{n}"' in source]
    assert names == ["boltz2"]
