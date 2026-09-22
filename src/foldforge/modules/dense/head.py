# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from foldforge.data.constants import atom_types
from foldforge.data.features import dense_batch as feat_batch
from foldforge.modules import ops as fastnn
from foldforge.modules.dense import atom_layout, featurization, pairformer, template
from foldforge.modules.dense.pair_init import ContactConditioning, token_bond_types
from foldforge.modules.dense.spec import ALPHAFOLD3, DenseSpec

_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3


class DistogramHead(nn.Module):
    """Represent distogram head."""

    breaks: torch.Tensor
    is_contact_bin: torch.Tensor

    def __init__(
        self,
        c_pair: int = 128,
        num_bins: int = 64,
        first_break: float = 2.3125,
        last_break: float = 21.6875,
        bias: bool = False,
        hidden: bool = False,
        mean_symmetrised: bool = False,
    ) -> None:
        super().__init__()

        self.c_pair = c_pair
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break
        self.mean_symmetrised = mean_symmetrised

        # A trained bias passes through the symmetrisation below, so it enters twice
        # unless the family takes the mean.
        self.hidden = None
        width = self.c_pair
        if hidden:
            # A head trained post-hoc on a frozen trunk, as an MLP rather than the
            # single projection AF3 reads the pair with.
            self.input_layer_norm = fastnn.LayerNorm(self.c_pair)
            self.hidden = nn.Linear(self.c_pair, 2 * self.c_pair, bias=True)
            width = 2 * self.c_pair
        self.half_logits = nn.Linear(width, self.num_bins, bias=bias)

        breaks = torch.linspace(
            self.first_break,
            self.last_break,
            self.num_bins - 1,
        )

        self.register_buffer("breaks", breaks)

        bin_tops = torch.cat(
            (breaks, (breaks[-1] + (breaks[-1] - breaks[-2])).reshape(1))
        )
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)

        self.register_buffer("is_contact_bin", is_contact_bin)

    def forward(
        self, batch: feat_batch.Batch, embeddings: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Args:

            pair (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, num_bins]
        """
        pair_act = embeddings["pair"]
        seq_mask = batch.token_features.mask.to(dtype=torch.bool)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]

        act = pair_act
        if self.hidden is not None:
            # The exact erf GELU, not the tanh approximation: the difference is
            # silent and this head was trained against torch's default.
            act = F.gelu(
                self.hidden(self.input_layer_norm(pair_act)), approximate="none"
            )
        left_half_logits = self.half_logits(act)
        right_half_logits = left_half_logits
        logits = left_half_logits + right_half_logits.transpose(-2, -3)
        if self.mean_symmetrised:
            # The mean, not the sum. Once the softmax sees it this is not a rescale.
            logits = logits / 2
        probs = torch.softmax(logits, dim=-1)
        contact_probs = torch.einsum(
            "ijk,k->ij", probs.float(), self.is_contact_bin.float()
        )

        contact_probs = pair_mask * contact_probs

        return {
            **({"logits": logits} if getattr(self, "save_distogram", False) else {}),
            "bin_edges": self.breaks,
            "contact_probs": contact_probs,
        }


def masked_global_norm(
    x: torch.Tensor, mask: torch.Tensor, width: int | None = None
) -> torch.Tensor:
    """Normalise by the mean and variance over the REAL tokens and every feature.

    A statistic that reduces over more than the feature axis is padding-sensitive;
    the vendor never pads, so padding must not enter it. ``width`` is the vendor's
    feature width where it exceeds ours: each missing all-zero column still adds
    ``mean**2`` to the variance sum per real token.
    """
    value = x.float()
    weight = mask.float()[..., None]
    width = value.shape[-1] if width is None else width
    count = (weight.sum() * width).clamp_min(1.0)
    mean = (value * weight).sum() / count
    missing = (width - value.shape[-1]) * mask.float().sum()
    variance = (
        ((value - mean).square() * weight).sum() + missing * mean.square()
    ) / count
    return ((value - mean) / (variance + 1e-5).sqrt()).to(x.dtype)


class ConfidenceReembedding(nn.Module):
    """Rebuild single and pair from the normed trunk outputs before the confidence stack.

    AF3 adds two target-feature projections and a distogram embedding to the trunk
    pair. This form LayerNorms both trunk outputs and sums nine terms into the pair,
    among them a product of two single projections and the pair-init terms the trunk
    itself used (relative positions, bonds, bond orders, contact conditioning).
    """

    def __init__(
        self,
        c_single: int,
        c_pair: int,
        c_target_feat: int,
        learned_bins: int | None = None,
        esm_classes: bool = False,
    ) -> None:
        super().__init__()
        self.esm_classes = esm_classes
        if esm_classes:
            c_target_feat += 4
        self.s_inputs_norm = fastnn.LayerNorm(c_target_feat)
        self.s_norm = fastnn.LayerNorm(c_single)
        self.s_input_to_s = nn.Linear(c_target_feat, c_single, bias=False)
        self.z_norm = fastnn.LayerNorm(c_pair)
        self.rel_pos_project = nn.Linear(139, c_pair, bias=False)
        self.token_bonds_project = nn.Linear(1, c_pair, bias=False)
        self.token_bonds_type_embed = nn.Linear(7, c_pair, bias=False)
        self.contact_conditioning = ContactConditioning(c_pair)
        self.left_target_feat_project = nn.Linear(c_target_feat, c_pair, bias=False)
        self.right_target_feat_project = nn.Linear(c_target_feat, c_pair, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(c_target_feat, c_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(c_target_feat, c_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(c_pair, c_pair, bias=False)
        # A family that trained its OWN boundaries bins the prediction with
        # them; the constant 2..22 A over 63 edges is Boltz-2's.
        self.learned_bins = learned_bins
        bins = learned_bins or 64
        if learned_bins is not None:
            self.distogram_boundaries = nn.Parameter(torch.zeros(bins - 1))
        self.distogram_feat_project = nn.Linear(bins, c_pair, bias=False)

    def forward(
        self,
        pair: torch.Tensor,
        single: torch.Tensor,
        target_feat: torch.Tensor,
        positions: torch.Tensor,
        pair_mask: torch.Tensor,
        batch: feat_batch.Batch,
        symmetric_bonds: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the re-embedded (pair, single)."""
        dtype = pair.dtype
        if self.esm_classes:
            target_feat = featurization.widen_to_esm_classes(target_feat)
        inputs = self.s_inputs_norm(target_feat)
        single = self.s_norm(single) + self.s_input_to_s(inputs)

        bonds = token_bond_types(batch, symmetric=symmetric_bonds)
        pair = self.z_norm(pair)
        pair = pair + self.rel_pos_project(
            featurization.create_relative_encoding(
                batch.token_features, max_relative_idx=32, max_relative_chain=2
            ).to(dtype)
        )
        pair = pair + self.token_bonds_project((bonds > 0)[..., None].to(dtype))
        pair = pair + self.token_bonds_type_embed(
            torch.nn.functional.one_hot(bonds, 7).to(dtype)
        )
        pair = pair + self.contact_conditioning(pair.shape[0], pair)
        # Orientation is load-bearing: the row-indexed term is the RIGHT projection.
        pair = pair + self.right_target_feat_project(inputs)[:, None]
        pair = pair + self.left_target_feat_project(inputs)[None]
        pair = pair + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(inputs)[:, None] * self.s_to_z_prod_in2(inputs)[None]
        )
        distance = (
            (positions[:, None] - positions[None]).square().sum(-1) + 1e-10
        ).sqrt()
        edges = (
            self.distogram_boundaries.to(distance.dtype)
            if self.learned_bins is not None
            else torch.linspace(2.0, 22.0, 63, device=distance.device)
        )
        distogram = torch.nn.functional.one_hot(
            (distance[..., None] > edges).sum(-1), self.learned_bins or 64
        ).to(dtype)
        pair = pair + self.distogram_feat_project(distogram * pair_mask[..., None])
        return pair, single


class ConfidenceHead(nn.Module):
    """Implement Algorithm 31 in AF3."""

    bin_centers: torch.Tensor
    distance_breaks: torch.Tensor
    pae_bin_centers: torch.Tensor
    pae_breaks: torch.Tensor
    pae_step: torch.Tensor
    plddt_bin_centers: torch.Tensor
    step: torch.Tensor

    def __init__(
        self,
        c_single: int = 384,
        c_pair: int = 128,
        c_target_feat: int = 447,
        n_pairformer_layers: int = 4,
        spec: DenseSpec = ALPHAFOLD3,
    ) -> None:
        super().__init__()

        self.c_single = c_single
        self.c_pair = c_pair
        self.c_target_feat = c_target_feat
        self.spec = spec
        #: "boltz2": the pair is RE-EMBEDDED from the inputs rather than read
        #: from the trunk. Whether the heads then split by chain and whether
        #: they norm their input are stated separately: ESMFold2 takes this
        #: embedding and does neither.
        self.reembed_pair = spec.confidence == "boltz2"
        self.split_heads = spec.confidence_split_heads
        #: "protenix2": the trunk single is clamped and normalised before ANY
        #: use, the distance-error head normalises the SYMMETRISED pair, and a
        #: raw-distance term rides alongside the binned one.
        self.protenix_confidence = spec.confidence == "protenix2"
        #: "rf3": trunk inputs normalised over the WHOLE tensor of real tokens, and
        #: the predicted structure embedded as 40 CA-CA distance bins.
        self.global_norm_inputs = spec.confidence == "rf3"

        self.dgram_features_config = template.DistogramFeaturesConfig()
        #: Distance classes the head embeds the predicted structure as.
        self.confidence_dgram = spec.confidence_dgram
        self.confidence_dgram_bins = (
            40
            if self.global_norm_inputs
            else (
                self.confidence_dgram[2]
                if self.confidence_dgram is not None
                else self.dgram_features_config.num_bins
            )
        )

        self.num_bins = 64
        self.max_error_bin = 31.0

        self.pae_num_bins = 64
        self.pae_max_error_bin = 31.0

        self.num_plddt_bins = 50
        self.num_atom = atom_types.DENSE_ATOM_NUM
        #: A head trained to predict pLDDT over a different atom table keeps its
        #: own slot count; the gather back onto the dense layout is by atom name.
        self.plddt_slots = spec.plddt_atom_slots or self.num_atom
        self.bin_width = 1.0 / self.num_plddt_bins

        self._build_input_embedding(c_single, c_pair, c_target_feat)

        self.row_pool_attn = None
        if spec.confidence_row_pool:
            self.row_pool_attn = nn.Linear(self.c_pair, 1, bias=False)
            self.row_pool_out = nn.Linear(self.c_pair, self.c_single, bias=False)

        self.confidence_pairformer = nn.ModuleList(
            [
                pairformer.PairformerBlock(
                    c_single=self.c_single,
                    c_pair=self.c_pair,
                    n_heads_pair=spec.pair_heads,
                    num_intermediate_factor=spec.pairformer_transition_factor,
                    with_single=True,
                    with_pair_attention=spec.pair_attention,
                    spec=spec,
                    dual_output=spec.confidence_dual_output,
                )
                for _ in range(n_pairformer_layers)
            ]
        )

        head_norm = fastnn.LayerNorm if spec.confidence_head_norms else nn.Identity
        self.logits_ln = head_norm(self.c_pair)
        self.left_half_distance_logits = nn.Linear(
            self.c_pair, self.num_bins, bias=False
        )
        if self.split_heads:
            self.inter_half_distance_logits = nn.Linear(
                self.c_pair, self.num_bins, bias=False
            )
            self.pae_inter_logits = nn.Linear(self.c_pair, 64, bias=False)

        self.register_buffer(
            "distance_breaks",
            torch.linspace(0.0, self.max_error_bin, self.num_bins - 1),
        )
        self.register_buffer("step", self.distance_breaks[1] - self.distance_breaks[0])
        self.register_buffer("bin_centers", self.distance_breaks + self.step / 2)
        self.bin_centers = torch.concatenate(
            [self.bin_centers, self.bin_centers[-1:] + self.step], dim=0
        )

        self.pae_logits_ln = head_norm(self.c_pair)
        self.pae_logits = nn.Linear(self.c_pair, self.pae_num_bins, bias=False)

        self.register_buffer(
            "pae_breaks",
            torch.linspace(0.0, self.pae_max_error_bin, self.pae_num_bins - 1),
        )
        self.register_buffer("pae_step", self.pae_breaks[1] - self.pae_breaks[0])

        pae_bin_centers_ = self.pae_breaks + self.pae_step / 2
        self.register_buffer(
            "pae_bin_centers",
            torch.concatenate(
                [pae_bin_centers_, pae_bin_centers_[-1:] + self.pae_step], dim=0
            ),
        )

        self.register_buffer(
            "plddt_bin_centers", torch.arange(0.5 * self.bin_width, 1.0, self.bin_width)
        )

        self.plddt_logits_ln = head_norm(self.c_single)
        self.plddt_logits = nn.Linear(
            self.c_single, self.plddt_slots * self.num_plddt_bins, bias=False
        )

        self.resolved_head = spec.resolved_head
        if self.resolved_head:
            self.experimentally_resolved_ln = head_norm(self.c_single)
            self.experimentally_resolved_logits = nn.Linear(
                self.c_single, self.num_atom * 2, bias=False
            )

    def _build_input_embedding(
        self, c_single: int, c_pair: int, c_target_feat: int
    ) -> None:
        """Projections that put the trunk and the predicted structure on the pair."""
        if self.reembed_pair:
            self.reembedding = ConfidenceReembedding(
                c_single,
                c_pair,
                c_target_feat,
                learned_bins=self.spec.confidence_learned_bins,
                esm_classes=self.spec.single_cond_layout == "esm",
            )
        else:
            self.left_target_feat_project = nn.Linear(
                self.c_target_feat, self.c_pair, bias=False
            )
            self.right_target_feat_project = nn.Linear(
                self.c_target_feat, self.c_pair, bias=False
            )
            self.distogram_feat_project = nn.Linear(
                self.confidence_dgram_bins, self.c_pair, bias=False
            )
            if self.protenix_confidence:
                # Unbinned, so it carries the sub-bin resolution the one-hot
                # throws away.
                self.distance_feat_project = nn.Linear(1, self.c_pair, bias=False)
        if self.protenix_confidence:
            self.input_single_norm = fastnn.LayerNorm(self.c_single)

    def _embed_features(
        self,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:

        out = (
            self.left_target_feat_project(target_feat)[..., None, :, :]
            + self.right_target_feat_project(target_feat)[..., None, :]
        )

        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )

        if self.global_norm_inputs:
            # Token-centre (CA, dense atom 1) distances over 39 edges from 3.25 A.
            ca = dense_atom_positions[:, 1, :]
            distance = ((ca[:, None] - ca[None]).square().sum(-1) + 1e-10).sqrt()
            edges = torch.arange(39, device=ca.device) * ((50.75 - 3.25) / 39.0) + 3.25
            dgram = torch.nn.functional.one_hot(
                (distance[..., None] > edges).sum(-1), 40
            ).to(target_feat.dtype)
        elif self.confidence_dgram is not None:
            # Distances binned by a searchsorted over evenly spaced boundaries,
            # and NOT masked: the family that trained this does not mask it.
            low, high, classes = self.confidence_dgram
            distance = (
                (positions[:, None] - positions[None]).square().sum(-1) + 1e-10
            ).sqrt()
            edges = torch.linspace(
                low, high, classes - 1, device=positions.device, dtype=distance.dtype
            )
            return out + self.distogram_feat_project(
                torch.nn.functional.one_hot(
                    (distance[..., None] > edges).sum(-1), classes
                ).to(target_feat.dtype)
            )
        else:
            dgram = template.dgram_from_positions(positions, self.dgram_features_config)

        dgram *= pair_mask[..., None]

        out += self.distogram_feat_project(dgram)
        if self.protenix_confidence:
            distance = (
                (positions[:, None] - positions[None]).square().sum(-1) + 1e-10
            ).sqrt()
            out = out + self.distance_feat_project(distance[..., None].to(out.dtype))

        return out

    def forward(  # noqa: PLR0915 - one head sequence per released variant
        self,
        dense_atom_positions: torch.Tensor,
        embeddings: dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        asym_id: torch.Tensor,
        batch: feat_batch.Batch | None = None,
    ) -> dict[str, torch.Tensor]:
        """Args:

        target_feat (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        dense_atom_positions (torch.Tensor): array of positions.
            [N_tokens, N_atom, 3]
        pair_mask (torch.Tensor): pair mask
            [..., N_token, N_token]
        token_atoms_to_pseudo_beta (atom_layout.GatherInfo): Pseudo beta info for
            atom tokens.
        """
        dtype = self.left_half_distance_logits.weight.dtype
        if not self.reembed_pair:
            dtype = self.left_target_feat_project.weight.dtype

        seq_mask_cast = seq_mask.to(dtype=dtype)
        pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
        pair_mask = pair_mask.to(dtype=dtype)

        pair_act = embeddings["pair"].clone().to(dtype=dtype)
        single_act = embeddings["single"].clone().to(dtype=dtype)
        target_feat = embeddings["target_feat"].clone().to(dtype=dtype)

        if self.global_norm_inputs:
            real = seq_mask.bool()
            pair_act = masked_global_norm(pair_act, real[:, None] & real[None])
            single_act = masked_global_norm(single_act, real)
            # The vendor's input features are 449 wide; the two classes FoldForge's
            # alphabet lacks are zero but still enter its mean and variance.
            target_feat = masked_global_norm(target_feat, real, width=449)
        if self.protenix_confidence:
            # Clamped and normalised before ANY use: the confidence pairformer
            # and every head see the normalised single. AF3 uses it raw, and an
            # unnormalised trunk single enters this head at std 211.
            single_act = self.input_single_norm(single_act.clamp(-512.0, 512.0))

        if self.reembed_pair:
            if batch is None:
                message = "The re-embedding confidence head needs the feature batch"
                raise ValueError(message)
            positions = atom_layout.convert(
                token_atoms_to_pseudo_beta, dense_atom_positions, layout_axes=(-3, -2)
            )
            pair_act, single_act = self.reembedding(
                pair_act,
                single_act,
                target_feat,
                positions.to(dtype),
                pair_mask,
                batch,
                self.spec.symmetric_bonds,
            )
        else:
            pair_act += self._embed_features(
                dense_atom_positions, token_atoms_to_pseudo_beta, pair_mask, target_feat
            )

        # pairformer stack
        for layer in self.confidence_pairformer:
            pair_act, single_act = layer(pair_act, pair_mask, single_act, seq_mask)

        if self.row_pool_attn is not None:
            # The single is pooled FROM THE PAIR, and after the stack, not
            # before it. Pooling the pre-stack pair leaves every pair-derived
            # output close and every PER-ATOM one uncorrelated, because pLDDT
            # and experimentally-resolved are the only heads reading the single.
            scores = self.row_pool_attn(pair_act)[..., 0]
            scores = torch.where(
                seq_mask.to(torch.bool)[None, :],
                scores,
                torch.full_like(scores, -1e9),
            )
            single_act = self.row_pool_out(
                torch.einsum("nm,nmd->nd", torch.softmax(scores, dim=-1), pair_act)
            )

        pair_act = pair_act.to(self.left_half_distance_logits.weight.dtype)
        single_act = single_act.to(self.plddt_logits.weight.dtype)

        # Produce logits to predict a distogram of pairwise distance errors
        # between the input prediction and the ground truth.

        same_chain = (asym_id[:, None] == asym_id[None])[..., None].to(pair_act.dtype)
        if self.split_heads:
            # Symmetrise FIRST, then route each pair to its chain-relation head.
            symmetric = pair_act + pair_act.transpose(-2, -3)
            distance_logits = self.left_half_distance_logits(
                symmetric
            ) * same_chain + self.inter_half_distance_logits(symmetric) * (
                1 - same_chain
            )
        elif self.protenix_confidence:
            # The symmetrisation is INSIDE the norm rather than outside the
            # projection, and a LayerNorm is not linear, so the two differ.
            distance_logits = self.left_half_distance_logits(
                self.logits_ln(pair_act + pair_act.transpose(-2, -3))
            )
        else:
            left_distance_logits = self.left_half_distance_logits(
                self.logits_ln(pair_act)
            )
            distance_logits = left_distance_logits + torch.transpose(
                left_distance_logits, -2, -3
            )

        distance_probs = torch.softmax(distance_logits, dim=-1)
        pred_distance_error = (
            torch.sum(distance_probs * self.bin_centers, dim=-1) * pair_mask
        )
        average_pred_distance_error = torch.sum(
            pred_distance_error, dim=[-2, -1]
        ) / torch.sum(pair_mask, dim=[-2, -1])

        # Predicted aligned error
        pae_outputs = {}
        pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
        if self.split_heads:
            pae_logits = pae_logits * same_chain + self.pae_inter_logits(pair_act) * (
                1 - same_chain
            )
        pae_probs = torch.softmax(pae_logits, dim=-1)

        pair_mask_bool = pair_mask.to(dtype=torch.bool)

        pae = torch.sum(pae_probs * self.pae_bin_centers, dim=-1) * pair_mask_bool
        pae_outputs.update(
            {
                "full_pae": pae,
            }
        )

        tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = (
            self._get_tmscore_adjusted_pae(
                asym_id=asym_id,
                seq_mask=seq_mask,
                pair_mask=pair_mask_bool,
                bin_centers=self.pae_bin_centers,
                pae_probs=pae_probs,
            )
        )

        pae_outputs.update(
            {
                "tmscore_adjusted_pae_global": tmscore_adjusted_pae_global,
                "tmscore_adjusted_pae_interface": tmscore_adjusted_pae_interface,
            }
        )

        # pLDDT
        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))
        plddt_logits = einops.rearrange(
            plddt_logits,
            "... (n_atom n_bins) -> ... n_atom n_bins",
            n_bins=self.num_plddt_bins,
        )
        if self.plddt_slots != self.num_atom:
            plddt_logits = self._gather_plddt_by_atom_name(plddt_logits, batch)
        predicted_lddt = torch.sum(
            torch.softmax(plddt_logits, dim=-1) * self.plddt_bin_centers, dim=-1
        )
        predicted_lddt = predicted_lddt * 100.0

        # Experimentally resolved. A family that trained no such head reports
        # nothing rather than a random-init number that would read like one.
        predicted_experimentally_resolved = None
        if self.resolved_head:
            experimentally_resolved_logits = self.experimentally_resolved_logits(
                self.experimentally_resolved_ln(single_act)
            )
            experimentally_resolved_logits = einops.rearrange(
                experimentally_resolved_logits,
                "... (n_atom n_bins) -> ... n_atom n_bins",
                n_bins=2,
            )

            predicted_experimentally_resolved = torch.softmax(
                experimentally_resolved_logits, dim=-1
            )[..., 1]

        return {
            "predicted_lddt": predicted_lddt,
            **(
                {"predicted_experimentally_resolved": predicted_experimentally_resolved}
                if predicted_experimentally_resolved is not None
                else {}
            ),
            "full_pde": pred_distance_error,
            "average_pde": average_pred_distance_error,
            **pae_outputs,
        }

    def _gather_plddt_by_atom_name(
        self, logits: torch.Tensor, batch: feat_batch.Batch | None
    ) -> torch.Tensor:
        """Put logits predicted over a canonical atom table on the dense slots.

        The two orders differ -- CB is canonical slot 3 and dense slot 4, and
        tryptophan's NE1 is 24 -- and the permutation depends on the residue
        type, so no reshape of the weight can express it. The atom NAME is
        carried per dense slot, so match on that.
        """
        if batch is None:
            message = "Gathering pLDDT by atom name needs the batch's atom names"
            raise ValueError(message)
        names = batch.ref_structure.atom_name_chars.to(torch.int64)
        table = torch.tensor(
            [[ord(c) - 32 for c in name.ljust(4)] for name in atom_types.ATOM37],
            dtype=names.dtype,
            device=names.device,
        )
        hit = (names[:, :, None, :] == table[None, None]).all(-1)
        # (n_token, dense slots) -> the canonical slot each one names.
        index = hit.to(torch.int64).argmax(-1)
        index = index[..., None].expand(*index.shape, logits.shape[-1])
        while index.ndim < logits.ndim:
            index = index[None].expand(logits.shape[-index.ndim - 1], *index.shape)
        return torch.take_along_dim(logits, index, dim=-2)

    def _get_tmscore_adjusted_pae(
        self,
        asym_id: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        bin_centers: torch.Tensor,
        pae_probs: torch.Tensor,
    ):

        def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
            # Clip to avoid negative/undefined d0.
            """Return tmscore adjusted pae."""
            clipped_num_res = torch.clamp(num_interface_tokens, min=19)

            # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
            # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
            # Yang & Skolnick "Scoring function for automated
            # assessment of protein structure template quality" 2004.
            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

            # Make compatible with [num_tokens, num_tokens, num_bins]
            d0 = d0[:, :, None]
            bin_centers = bin_centers[None, None, :]

            # TM-Score term for every bin.
            tm_per_bin = 1.0 / (1 + torch.square(bin_centers) / torch.square(d0))
            # E_distances tm(distance).
            predicted_tm_term = torch.sum(pae_probs * tm_per_bin, dim=-1)
            return predicted_tm_term

        # Interface version
        x = asym_id[None, :] == asym_id[:, None]
        num_chain_tokens = torch.sum(x * pair_mask, dim=-1, dtype=torch.int32)
        num_interface_tokens = num_chain_tokens[None, :] + num_chain_tokens[:, None]
        # Don't double-count within a single chain
        num_interface_tokens -= x * (num_interface_tokens // 2)
        num_interface_tokens = num_interface_tokens * pair_mask

        num_global_tokens = torch.ones(
            size=pair_mask.shape, dtype=torch.int32, device=x.device
        )
        num_global_tokens *= seq_mask.sum()

        if num_global_tokens.dtype != torch.int32:
            message = "Invalid state: num_global_tokens.dtype == torch.int32"
            raise ValueError(message)
        if num_interface_tokens.dtype != torch.int32:
            message = "Invalid state: num_interface_tokens.dtype == torch.int32"
            raise ValueError(message)
        global_apae = get_tmscore_adjusted_pae(
            num_global_tokens, bin_centers, pae_probs
        )
        interface_apae = get_tmscore_adjusted_pae(
            num_interface_tokens, bin_centers, pae_probs
        )
        return global_apae, interface_apae
