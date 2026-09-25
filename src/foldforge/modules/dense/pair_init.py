"""Pair and single initialisation terms some dense-graph families add to AF3's."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from foldforge.data.features import dense_batch as feat_batch

#: Bond-order codes written into the token bond type matrix, plus one: zero is
#: "no bond" and its embedding row is a learned vector, not a no-op.
BOND_SINGLE = 1
BOND_COVALENT = 5


def token_bond_types(batch: feat_batch.Batch) -> torch.Tensor:
    """(tokens, tokens) bond-order classes for the token bond type embedding.

    Polymer-ligand links are covalent by construction. Ligand-ligand rows mix a
    residue's own bond graph with inter-residue links; the featurised batch carries
    no bond order, so they are written as single bonds.
    """
    tokens = batch.token_features.token_index.shape[0]
    device = batch.token_features.token_index.device
    matrix = torch.zeros((tokens, tokens), dtype=torch.int64, device=device)
    for gather, code in (
        (batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds, BOND_COVALENT),
        (batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds, BOND_SINGLE),
    ):
        valid = gather.gather_mask.prod(dim=1).to(torch.int64)
        index = gather.gather_idxs.to(torch.int64) * valid[:, None]
        value = (code + 1) * valid
        matrix[index[:, 0], index[:, 1]] = value
    # Every padded bond row points at [0, 0].
    matrix[0, 0] = 0
    return matrix


class ContactConditioning(nn.Module):
    """Distance-restraint conditioning evaluated for an unconstrained input.

    The contact classes are (unspecified, unselected, three restraint kinds). The
    first two bypass the encoder and contribute a learned constant, so an input with
    no restraints still adds ``encoding_unspecified`` to every pair. FoldForge's
    inputs carry no restraints; the encoder is kept so the checkpoint loads strictly
    and a restraint feature can reach it later.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.contact_fourier = nn.Linear(1, channels, bias=True)
        self.contact_encoder = nn.Linear(3 + 1 + channels, channels, bias=True)
        self.encoding_unspecified = nn.Parameter(torch.zeros(channels))
        self.encoding_unselected = nn.Parameter(torch.zeros(channels))

    def forward(self, tokens: int, like: torch.Tensor) -> torch.Tensor:
        """Return the (tokens, tokens, channels) term for no restraints at all."""
        del tokens
        return self.encoding_unspecified.to(like.dtype).expand(like.shape)

    def encode(self, contact: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Full conditioning for one-hot ``contact`` (…, 5) and distance ``threshold``."""
        dtype = self.contact_encoder.weight.dtype
        scaled = ((threshold - 4.0) / (20.0 - 4.0))[..., None].to(dtype)
        fourier = torch.cos(2 * math.pi * self.contact_fourier(scaled))
        encoded = self.contact_encoder(
            torch.cat([contact[..., 2:].to(dtype), scaled, fourier], dim=-1)
        )
        selected = contact[..., 0:2].sum(-1, keepdim=True).to(dtype)
        return (
            encoded * (1 - selected)
            + self.encoding_unspecified * contact[..., 0:1].to(dtype)
            + self.encoding_unselected * contact[..., 1:2].to(dtype)
        )


class SummedInputEmbedder(nn.Module):
    """Input embedder that sums onto the atom encoder's token output.

    AF3 concatenates target features with the atom encoder output; this form adds
    bias-free projections of the restype, the MSA profile and four conditioning
    classes to it and hands every downstream embedder one ``channels``-wide vector.
    """

    METHOD_XRAY = 1

    #: Blob record name for each projection, so the importer needs no model list.
    RECORDS = (
        "res_type_encoding",
        "msa_profile_encoding",
        "mol_type_conditioning",
        "cyclic_conditioning",
        "method_conditioning",
        "modified_conditioning",
    )
    RECORD_PREFIX = "boltz2_"

    def records(self) -> dict[str, nn.Linear]:
        """Map each blob record name onto the projection that holds it."""
        return {self.RECORD_PREFIX + name: getattr(self, name) for name in self.RECORDS}

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res_type_encoding = nn.Linear(31, channels, bias=False)
        self.msa_profile_encoding = nn.Linear(32, channels, bias=False)
        self.mol_type_conditioning = nn.Linear(4, channels, bias=False)
        self.cyclic_conditioning = nn.Linear(1, channels, bias=False)
        self.method_conditioning = nn.Linear(12, channels, bias=False)
        self.modified_conditioning = nn.Linear(2, channels, bias=False)

    def forward(self, batch: feat_batch.Batch, token_act: torch.Tensor) -> torch.Tensor:
        """Sum every term; conditioning rows are learned vectors even at class zero."""
        dtype = self.res_type_encoding.weight.dtype
        features = batch.token_features
        tokens = token_act.shape[0]

        def one_hot(index: torch.Tensor, classes: int) -> torch.Tensor:
            return F.one_hot(index.to(torch.int64), classes).to(dtype)

        out = token_act.to(dtype)
        out = out + self.res_type_encoding(one_hot(features.aatype, 31))
        profile = torch.cat(
            [batch.msa.profile.to(dtype), batch.msa.deletion_mean[..., None].to(dtype)],
            dim=-1,
        )
        out = out + self.msa_profile_encoding(profile)
        mol_type = (
            features.is_dna.to(torch.int64)
            + 2 * features.is_rna.to(torch.int64)
            + 3 * features.is_ligand.to(torch.int64)
        )
        out = out + self.mol_type_conditioning(one_hot(mol_type, 4))
        # No cyclic, modified-residue or experimental-method feature reaches the
        # batch: not cyclic, not modified, X-ray (the vendor's prediction default).
        out = out + self.cyclic_conditioning(out.new_zeros(tokens, 1))
        method = torch.full((tokens,), self.METHOD_XRAY, device=out.device)
        out = out + self.method_conditioning(one_hot(method, 12))
        modified = torch.zeros(tokens, dtype=torch.int64, device=out.device)
        return out + self.modified_conditioning(one_hot(modified, 2))


class ChaiTokenEmbedder(nn.Module):
    """Input embedder that builds its own token stream, then projects it twice.

    Where AF3 concatenates 447 target features with the atom encoder's token
    output, this form builds a ``channels``-wide stream of its own -- restype,
    MSA profile and, when the input carries them, protein language-model
    embeddings -- concatenates THAT with the token output and reads the pair
    through two separate projections. The trunk and the confidence head take the
    first; the diffusion module takes the second, which is a different vector
    from the same inputs and is what those weights were trained against.

    The restype projection carries a bias because the rest of the vendor's token
    stream is constant for a fold and the converter folds it in there.
    """

    RECORDS = (
        "token_feature_embedding",
        "msa_profile_embedding",
        "esm_embedding",
        "single_proj_in_trunk",
        "single_proj_in_structure",
    )
    RECORD_PREFIX = "chai1_"

    def records(self) -> dict[str, nn.Linear]:
        """Map each blob record name onto the projection that holds it."""
        return {self.RECORD_PREFIX + name: getattr(self, name) for name in self.RECORDS}

    #: Restype classes: the polymer types plus unknown and gap.
    RESTYPES = 31
    #: MSA profile columns plus the deletion mean.
    PROFILE = 32

    def __init__(self, channels: int, lm_channels: int = 2560) -> None:
        super().__init__()
        self.token_feature_embedding = nn.Linear(self.RESTYPES, channels, bias=True)
        self.msa_profile_embedding = nn.Linear(self.PROFILE, channels, bias=False)
        self.esm_embedding = nn.Linear(lm_channels, channels, bias=False)
        self.single_proj_in_trunk = nn.Linear(2 * channels, channels, bias=False)
        self.single_proj_in_structure = nn.Linear(2 * channels, channels, bias=False)

    def forward(
        self,
        batch: feat_batch.Batch,
        token_act: torch.Tensor,
        lm_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the trunk and the structure projections, in that order."""
        dtype = self.token_feature_embedding.weight.dtype
        aatype = batch.token_features.aatype.to(torch.int64)
        stream = self.token_feature_embedding(
            F.one_hot(aatype, self.RESTYPES).to(dtype)
        )
        profile = torch.cat(
            [batch.msa.profile.to(dtype), batch.msa.deletion_mean[..., None].to(dtype)],
            dim=-1,
        )
        stream = stream + self.msa_profile_embedding(profile)
        if lm_embeddings is not None:
            # Absent, the term is simply not added, which keeps a family that
            # supplies no language model byte-identical.
            stream = stream + self.esm_embedding(lm_embeddings.to(dtype))
        pair = torch.cat([token_act.to(dtype), stream], dim=-1)
        return self.single_proj_in_trunk(pair), self.single_proj_in_structure(pair)
