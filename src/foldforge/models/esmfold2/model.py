"""Top-level ESMFold2: input features to coordinates and confidence."""

from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_file
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType
from torch import nn

from .atom_encoder import NUM_RES_TYPES, InputsEmbedder
from .confidence import ConfidenceHead, ConfidenceOutput
from .config import ESMFold2Config
from .convert import convert_model
from .diffusion import DiffusionStructureHead
from .lm import LanguageModel, compute_lm_hidden_states
from .pair_trunk import PairTrunk


class ESMFold2Output(NamedTuple):
    """Predicted structure and the confidence estimates that go with it."""

    coords: torch.Tensor
    #: ``None`` unless ``return_distogram=True``. The distogram head is a
    #: training auxiliary; no other head reads it, so it is off by default.
    distogram_logits: torch.Tensor | None
    confidence: ConfidenceOutput


class ESMFold2Model(nn.Module):
    """ESMFold2 assembled from team-gm's shared layers.

    The ESMC language model is not owned by this module. Either pass
    ``lm_hidden_states`` — cheap to cache, since they do not change across
    diffusion samples — or attach a model with :meth:`set_language_model`.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the shared layers.

    """

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        self.config = config
        self.inputs_embedder = InputsEmbedder(config, implementation)
        self.pair_trunk = PairTrunk(config, implementation)
        self.structure_head = DiffusionStructureHead(config, implementation)
        self.confidence_head = ConfidenceHead(config, implementation)
        self._language_model: LanguageModel | None = None

    def set_language_model(self, language_model: LanguageModel | None) -> None:
        """Attach (or detach) the ESMC model used to embed sequences.

        Parameters
        ----------
        language_model : LanguageModel or None
            Anything matching the protocol in :mod:`.lm`.

        """
        self._language_model = language_model

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path | None = None,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> "ESMFold2Model":
        """Build a model and load released weights into it.

        Parameters
        ----------
        path : str or Path or None
            Checkpoint directory. Defaults to whatever
            :func:`team_gm.checkpoints.resolve` finds for ``"esmfold2"``.
        implementation : ImplementationType
            Kernel backend for the shared layers.

        Returns
        -------
        ESMFold2Model
            Model with the released weights loaded.

        """
        from foldforge import checkpoints  # noqa: PLC0415

        directory = Path(path) if path is not None else checkpoints.resolve("esmfold2")
        config = ESMFold2Config.from_json(directory)
        model = cls(config, implementation)
        model.load_state_dict(
            convert_model(load_file(directory / "model.safetensors"), config)
        )
        return model

    @staticmethod
    def _profile(
        residue_type: torch.Tensor,
        msa: torch.Tensor | None,
        msa_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Average residue-type distribution over the MSA, or the query alone.

        Returns ``residue_type``'s dtype. The averaging itself runs in fp32 —
        the row count can exceed bf16's integer-exact range — but the result has
        to come back, because this is concatenated with the pooled atom features
        and ``torch.cat`` promotes the whole tensor to the widest member: one
        fp32 column would silently drag a bf16 model's entire token track back to
        fp32 and cost 2.1x at the first fused kernel.
        """
        if msa is None:
            return residue_type
        one_hot = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
        if msa_mask is None:
            return one_hot.mean(dim=1).to(residue_type.dtype)
        weights = msa_mask.float().unsqueeze(-1)
        profile = (one_hot * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return profile.to(residue_type.dtype)

    @typecheck
    @torch.no_grad()
    def forward(
        self,
        residue_type: Int[torch.Tensor, "B L"],
        residue_index: Int[torch.Tensor, "B L"],
        asym_id: Int[torch.Tensor, "B L"],
        sym_id: Int[torch.Tensor, "B L"],
        entity_id: Int[torch.Tensor, "B L"],
        token_index: Int[torch.Tensor, "B L"],
        mol_type: Int[torch.Tensor, "B L"],
        token_bonds: Float[torch.Tensor, "B L L"],
        mask: Bool[torch.Tensor, "B L"],
        ref_pos: Float[torch.Tensor, "B A 3"],
        ref_charge: Float[torch.Tensor, "B A"],
        ref_element: Int[torch.Tensor, "B A"],
        ref_atom_name_chars: Int[torch.Tensor, "B A 4"],
        ref_space_uid: Int[torch.Tensor, "B A"],
        atom_mask: Bool[torch.Tensor, "B A"],
        atom_to_token: Int[torch.Tensor, "B A"],
        representative_atom_index: Int[torch.Tensor, "B L"],
        msa: Int[torch.Tensor, "B M L"] | None = None,
        msa_mask: Bool[torch.Tensor, "B M L"] | None = None,
        has_deletion: Float[torch.Tensor, "B M L"] | None = None,
        deletion_value: Float[torch.Tensor, "B M L"] | None = None,
        deletion_mean: Float[torch.Tensor, "B L"] | None = None,
        lm_hidden_states: Float[torch.Tensor, "B L n_lm d_model"] | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
        pair_state: Float[torch.Tensor, "B L L d_pair"] | None = None,
        *,
        return_distogram: bool = False,
        generator: torch.Generator | None = None,
    ) -> ESMFold2Output:
        """Predict a structure.

        Parameters
        ----------
        residue_type : Tensor
            Residue-type index per token.
        residue_index, asym_id, sym_id, entity_id, token_index, mol_type : Tensor
            Token indexing features.
        token_bonds : Tensor
            Pairwise bond indicator.
        mask : Tensor
            Token validity.
        ref_pos, ref_charge, ref_element, ref_atom_name_chars, ref_space_uid : Tensor
            Reference-conformer atom features. ``ref_element`` and
            ``ref_atom_name_chars`` are indices; they are one-hot encoded here.
        atom_mask, atom_to_token : Tensor
            Atom validity and atom-to-token map.
        representative_atom_index : Tensor
            Atom standing in for each token when measuring distances.
        msa, msa_mask, has_deletion, deletion_value, deletion_mean : Tensor or None
            MSA features in team-gm's ``[B, M, L]`` layout. Omit to fold without
            an MSA.
        lm_hidden_states : Tensor or None
            Cached ESMC hidden states. Computed from a model attached with
            :meth:`set_language_model` when omitted.
        num_loops : int or None
            Trunk recurrence loops.
        num_diffusion_samples : int
            Structures to sample.
        num_sampling_steps : int or None
            Diffusion solver steps.
        return_distogram : bool
            Run the trunk's distogram head. ``False`` by default: it is a
            training auxiliary that no other head consumes, so at inference it
            only costs a symmetrised pair tensor and a projection.
        pair_state : Tensor or None
            Initial recurrent state; pass one to pin the trunk's randomness.
        generator : torch.Generator or None
            Source of randomness for the trunk's initial state, the per-loop LM
            dropout and diffusion sampling — every stochastic step. Supply one
            to make a run reproducible.

        Returns
        -------
        ESMFold2Output
            Coordinates, distogram logits and confidence estimates.

        """
        # One-hot features enter bias-free projections, so they must arrive in the
        # model's own dtype: a hardcoded fp32 here is an fp32-vs-bf16 collision at
        # the first Linear of a bf16 model.
        param_dtype = next(self.parameters()).dtype
        residue_one_hot = F.one_hot(residue_type.long(), num_classes=NUM_RES_TYPES).to(
            param_dtype
        ) * mask.unsqueeze(-1)
        # Bias-free projections downstream, so padding has to be zeroed.
        atom_weight = atom_mask.unsqueeze(-1).to(param_dtype)
        element = (
            F.one_hot(ref_element.long(), num_classes=128).to(param_dtype) * atom_weight
        )
        name_chars = F.one_hot(ref_atom_name_chars.long(), num_classes=64).to(
            param_dtype
        ) * atom_weight.unsqueeze(-1)
        if deletion_mean is None:
            deletion_mean = residue_type.new_zeros(
                residue_type.shape, dtype=param_dtype
            )

        single_inputs = self.inputs_embedder(
            residue_type=residue_one_hot,
            profile=self._profile(residue_one_hot, msa, msa_mask),
            deletion_mean=deletion_mean,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_element=element,
            ref_atom_name_chars=name_chars,
            ref_space_uid=ref_space_uid,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
        )

        if lm_hidden_states is None and self._language_model is not None:
            lm_hidden_states = compute_lm_hidden_states(
                self._language_model,
                residue_type,
                asym_id,
                residue_index,
                mol_type,
                mask,
            ).to(single_inputs.dtype)

        msa_features = None
        if msa is not None:
            msa_features = _msa_features(
                msa, has_deletion, deletion_value, msa_mask, param_dtype
            )

        trunk = self.pair_trunk(
            single_inputs=single_inputs,
            residue_index=residue_index,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            token_index=token_index,
            token_bonds=token_bonds,
            mask=mask,
            lm_hidden_states=lm_hidden_states,
            msa_features=msa_features,
            msa_mask=msa_mask,
            num_loops=num_loops,
            pair_state=pair_state,
            return_distogram=return_distogram,
            generator=generator,
        )

        coords = self.structure_head.sample(
            single_inputs=single_inputs,
            pair=trunk.pair,
            relative_position_encoding=trunk.relative_position_encoding,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_element=element,
            ref_atom_name_chars=name_chars,
            ref_space_uid=ref_space_uid,
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            mask=mask,
            num_diffusion_samples=num_diffusion_samples,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        confidence = self.confidence_head(
            pair=trunk.pair,
            single_inputs=single_inputs,
            coords=coords,
            representative_atom_index=representative_atom_index,
            atom_to_token=atom_to_token,
            atom_mask=atom_mask,
            mask=mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=num_diffusion_samples,
            relative_position_encoding=trunk.relative_position_encoding,
            token_bonds_encoding=trunk.token_bonds_encoding,
        )
        return ESMFold2Output(
            coords=coords,
            distogram_logits=trunk.distogram_logits,
            confidence=confidence,
        )


def _msa_features(
    msa: torch.Tensor,
    has_deletion: torch.Tensor | None,
    deletion_value: torch.Tensor | None,
    msa_mask: torch.Tensor | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One-hot the MSA and append the two deletion channels, in ``dtype``."""
    one_hot = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).to(dtype)
    if msa_mask is not None:
        one_hot = one_hot * msa_mask.unsqueeze(-1).to(dtype)
    zeros = one_hot.new_zeros(msa.shape)
    deletions = has_deletion if has_deletion is not None else zeros
    values = deletion_value if deletion_value is not None else zeros
    if msa_mask is not None:
        weight = msa_mask.to(dtype)
        deletions, values = deletions * weight, values * weight
    return torch.cat([one_hot, deletions.unsqueeze(-1), values.unsqueeze(-1)], dim=-1)
