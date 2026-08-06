"""ESMFold2 confidence head: pLDDT, PAE, PDE, resolved, pTM and ipTM.

The head re-reads the trunk's pair representation together with the sampled
coordinates, so its estimates are conditioned on the structure that was actually
produced rather than on the trunk alone. It runs one more
:class:`~team_gm.models.esmfold2.FoldingTrunk` (4 blocks) over that conditioned
pair, then pools to a single track for the per-atom heads.

Three tensors in the released checkpoint — ``s_norm``, ``s_inputs_to_single``
and ``s_input_to_s`` — are never read by the release forward pass, so they are
not carried here and :func:`~team_gm.models.esmfold2.convert.convert_confidence_head`
does not emit them.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules.exceptions import ImplementationType
from team_gm.modules.primitives import Linear
from torch import nn

from .config import ESMFold2Config
from .embeddings import RowAttentionPooling
from .tokens import (
    categorical_mean,
    gather_representative_atoms,
    gather_token_to_atom,
    intra_token_index,
    mean_per_token,
)
from .trunk import FoldingTrunk

# Token type id for anything that is not a polymer residue.
NON_POLYMER_ID = 4
# Atom slots per token that the pLDDT/resolved heads have weights for.
MAX_ATOMS_PER_TOKEN = 23
# PAE/PDE bins span 0-32 A.
ERROR_RANGE = 32.0
# Contact cutoff for the interface-weighted pLDDT.
CONTACT_CUTOFF = 8.0
EPS = 1e-6


class ConfidenceOutput(NamedTuple):
    """Confidence estimates for one batch of sampled structures.

    The leading axis of every field is ``batch * num_diffusion_samples``; ``L``
    counts tokens and ``A`` atoms.

    Attributes
    ----------
    plddt : Tensor
        ``[BS, L]`` per-token pLDDT.
    plddt_ca : Tensor
        ``[BS, L]`` pLDDT at each token's representative atom.
    plddt_per_atom : Tensor
        ``[BS, A]`` per-atom pLDDT.
    plddt_logits : Tensor
        ``[BS, A, num_plddt_bins]`` raw per-atom logits.
    complex_plddt : Tensor
        ``[BS]`` mean pLDDT over all valid atoms.
    complex_iplddt : Tensor
        ``[BS]`` interface-weighted mean pLDDT.
    pae, pde : Tensor
        ``[BS, L, L]`` expected aligned / distance error in angstrom.
    pae_logits, pde_logits : Tensor
        ``[BS, L, L, bins]`` raw logits behind those expectations.
    resolved_logits : Tensor
        ``[BS, A, 2]`` unresolved/resolved logits per atom.
    ptm, iptm : Tensor
        ``[BS]`` predicted TM-score, overall and across chain interfaces.

    """

    plddt: torch.Tensor
    plddt_ca: torch.Tensor
    plddt_per_atom: torch.Tensor
    plddt_logits: torch.Tensor
    complex_plddt: torch.Tensor
    complex_iplddt: torch.Tensor
    pae: torch.Tensor
    pae_logits: torch.Tensor
    pde: torch.Tensor
    pde_logits: torch.Tensor
    resolved_logits: torch.Tensor
    ptm: torch.Tensor
    iptm: torch.Tensor


class ConfidenceHead(nn.Module):
    """Predict per-atom, per-pair and whole-complex confidence.

    Parameters
    ----------
    config : ESMFold2Config
        Released model configuration.
    implementation : ImplementationType
        Kernel backend for the head's own trunk stack.

    """

    boundaries: torch.Tensor

    def __init__(
        self,
        config: ESMFold2Config,
        implementation: ImplementationType = ImplementationType.MINIWORLD_ENGINE,
    ) -> None:
        super().__init__()
        head = config.confidence_head
        d_pair, d_single = config.d_pair, config.d_single
        d_inputs = config.inputs.d_inputs
        self.config = config

        self.register_buffer(
            "boundaries",
            torch.linspace(head.min_dist, head.max_dist, head.distogram_bins - 1),
        )
        self.distance_bin_embed = nn.Embedding(head.distogram_bins, d_pair)

        self.ln_single_inputs = nn.LayerNorm(d_inputs)
        self.ln_pair = nn.LayerNorm(d_pair)
        self.single_to_pair_row = Linear(d_inputs, d_pair, bias=False, init="default")
        self.single_to_pair_col = Linear(d_inputs, d_pair, bias=False, init="default")
        self.single_to_pair_prod_row = Linear(
            d_inputs, d_pair, bias=False, init="default"
        )
        self.single_to_pair_prod_col = Linear(
            d_inputs, d_pair, bias=False, init="default"
        )
        self.single_to_pair_prod_out = Linear(
            d_pair, d_pair, bias=False, init="default"
        )

        self.folding_trunk = FoldingTrunk(
            FoldingTrunk.Config(
                d_pair=d_pair,
                n_block=head.folding_trunk.n_layers,
                implementation=implementation,
            )
        )
        self.row_attention_pooling = RowAttentionPooling(
            d_pair=d_pair, d_single=d_single
        )

        self.ln_plddt = nn.LayerNorm(d_single)
        self.plddt_weight = nn.Parameter(
            torch.zeros(MAX_ATOMS_PER_TOKEN, d_single, head.num_plddt_bins)
        )
        self.ln_pae = nn.LayerNorm(d_pair)
        self.pae_head = Linear(d_pair, head.num_pae_bins, bias=False, init="default")
        self.ln_pde = nn.LayerNorm(d_pair)
        self.pde_head = Linear(d_pair, head.num_pde_bins, bias=False, init="default")
        self.ln_resolved = nn.LayerNorm(d_single)
        # 2 logits: [unresolved, resolved].
        self.resolved_weight = nn.Parameter(
            torch.zeros(MAX_ATOMS_PER_TOKEN, d_single, 2)
        )

    def _condition_pair(
        self,
        pair: torch.Tensor,
        single_inputs: torch.Tensor,
        relative_position_encoding: torch.Tensor | None,
        token_bonds_encoding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fold the token-level inputs back into the trunk's pair track."""
        single = self.ln_single_inputs(single_inputs)
        conditioned = self.ln_pair(pair)
        if relative_position_encoding is not None:
            conditioned = conditioned + relative_position_encoding
        if token_bonds_encoding is not None:
            conditioned = conditioned + token_bonds_encoding
        conditioned = conditioned + self.single_to_pair_row(single).unsqueeze(2)
        conditioned = conditioned + self.single_to_pair_col(single).unsqueeze(1)
        return conditioned + self.single_to_pair_prod_out(
            self.single_to_pair_prod_row(single).unsqueeze(2)
            * self.single_to_pair_prod_col(single).unsqueeze(1)
        )

    def _per_atom_logits(
        self,
        single: torch.Tensor,
        atom_to_token: torch.Tensor,
        norm: nn.LayerNorm,
        weight: nn.Parameter,
    ) -> torch.Tensor:
        """Score each atom with the weight matrix for its slot inside its token."""
        per_atom = norm(gather_token_to_atom(single, atom_to_token))
        slot = intra_token_index(atom_to_token).clamp(max=weight.shape[0] - 1)
        return torch.einsum("...c,...cb->...b", per_atom, weight[slot])

    def _tm_expected(
        self, pae_logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the expected TM contribution of each token pair from the PAE."""
        n_bins = pae_logits.shape[-1]
        width = ERROR_RANGE / n_bins
        centers = torch.arange(
            0.5 * width, ERROR_RANGE, width, device=pae_logits.device
        )
        n_res = mask.float().sum(dim=-1, keepdim=True)
        d0 = 1.24 * (n_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
        per_bin = 1 / (1 + (centers / d0) ** 2)
        return (F.softmax(pae_logits, dim=-1) * per_bin[:, None, None, :]).sum(dim=-1)

    @typecheck
    def forward(
        self,
        pair: Float[torch.Tensor, "B L L d_pair"],
        single_inputs: Float[torch.Tensor, "B L d_inputs"],
        coords: Float[torch.Tensor, "BS A 3"],
        representative_atom_index: Int[torch.Tensor, "B L"],
        atom_to_token: Int[torch.Tensor, "B A"],
        atom_mask: Bool[torch.Tensor, "B A"],
        mask: Bool[torch.Tensor, "B L"],
        asym_id: Int[torch.Tensor, "B L"],
        mol_type: Int[torch.Tensor, "B L"],
        num_diffusion_samples: int = 1,
        relative_position_encoding: Float[torch.Tensor, "B L L d_pair"] | None = None,
        token_bonds_encoding: Float[torch.Tensor, "B L L d_pair"] | None = None,
    ) -> ConfidenceOutput:
        """Forward pass.

        Parameters
        ----------
        pair : Tensor
            Trunk pair representation.
        single_inputs : Tensor
            Token-level input features.
        coords : Tensor
            Sampled atom coordinates; the batch axis carries
            ``num_diffusion_samples`` samples per input.
        representative_atom_index : Tensor
            Atom standing in for each token when measuring distances.
        atom_to_token, atom_mask : Tensor
            Atom-to-token map and atom validity.
        mask : Tensor
            Token validity.
        asym_id, mol_type : Tensor
            Chain id and molecule type per token.
        num_diffusion_samples : int
            Samples per input in ``coords``.
        relative_position_encoding, token_bonds_encoding : Tensor or None
            Encodings reused from the trunk.

        Returns
        -------
        ConfidenceOutput
            Confidence estimates, one row per sample.

        """
        conditioned = self._condition_pair(
            pair, single_inputs, relative_position_encoding, token_bonds_encoding
        )

        def repeat(x: torch.Tensor) -> torch.Tensor:
            return (
                x
                if num_diffusion_samples == 1
                else x.repeat_interleave(num_diffusion_samples, 0)
            )

        conditioned = repeat(conditioned)
        atom_to_token = repeat(atom_to_token)
        atom_mask = repeat(atom_mask)
        representative_atom_index = repeat(representative_atom_index)
        mask = repeat(mask)
        asym_id = repeat(asym_id)
        mol_type = repeat(mol_type)

        representative = gather_representative_atoms(coords, representative_atom_index)
        distances = torch.cdist(
            representative,
            representative,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        bins = (distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        conditioned = conditioned + self.distance_bin_embed(bins)
        # The trunk already applies residuals inside every block, so adding its
        # output back is a second, whole-stack residual. That is what the
        # released weights were trained with — not a transcription slip.
        conditioned = conditioned + self.folding_trunk(conditioned, mask)
        single = self.row_attention_pooling(conditioned, mask)

        plddt_logits = self._per_atom_logits(
            single, atom_to_token, self.ln_plddt, self.plddt_weight
        )
        plddt_per_atom = categorical_mean(plddt_logits, start=0.0, end=1.0)
        plddt = mean_per_token(
            plddt_per_atom, atom_to_token, atom_mask, single.shape[1]
        )

        atom_weight = atom_mask.to(plddt_per_atom.dtype)
        complex_plddt = (plddt_per_atom * atom_weight).sum(dim=-1) / (
            atom_weight.sum(dim=-1) + EPS
        )

        # Interface-weighted pLDDT: ligands always count, polymer tokens only
        # where they contact another chain.
        is_ligand = (mol_type == NON_POLYMER_ID).float()
        inter_chain = (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)).float()
        in_contact = (distances < CONTACT_CUTOFF).float()
        interface = (in_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(
            dim=-1
        )
        token_weight = torch.where(
            is_ligand.bool(), torch.full_like(interface, 2.0), interface
        )
        interface_weight = atom_weight * gather_token_to_atom(
            token_weight.unsqueeze(-1), atom_to_token
        ).squeeze(-1)
        complex_iplddt = (plddt_per_atom * interface_weight).sum(dim=-1) / (
            interface_weight.sum(dim=-1) + EPS
        )

        pae_logits = self.pae_head(self.ln_pae(conditioned))
        pde_logits = self.pde_head(self.ln_pde(conditioned))
        resolved_logits = self._per_atom_logits(
            single, atom_to_token, self.ln_resolved, self.resolved_weight
        )

        tm = self._tm_expected(pae_logits, mask)
        pair_mask = mask.float().unsqueeze(-1) * mask.float().unsqueeze(-2)
        ptm = ((tm * pair_mask).sum(-1) / (pair_mask.sum(-1) + EPS)).max(-1).values
        inter_mask = inter_chain * pair_mask
        iptm = ((tm * inter_mask).sum(-1) / (inter_mask.sum(-1) + EPS)).max(-1).values

        return ConfidenceOutput(
            plddt=plddt,
            plddt_ca=plddt_per_atom.gather(1, representative_atom_index),
            plddt_per_atom=plddt_per_atom,
            plddt_logits=plddt_logits,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            pae=categorical_mean(pae_logits, start=0.0, end=ERROR_RANGE),
            pae_logits=pae_logits,
            pde=categorical_mean(pde_logits, start=0.0, end=ERROR_RANGE),
            pde_logits=pde_logits,
            resolved_logits=resolved_logits,
            ptm=ptm,
            iptm=iptm,
        )


def chain_pair_iptm(
    tm: Float[torch.Tensor, "BS L L"],
    asym_id: Int[torch.Tensor, "BS L"],
    mask: Bool[torch.Tensor, "BS L"],
) -> Float[torch.Tensor, "BS C C"]:
    """Per-chain-pair ipTM matrix.

    Separate from :meth:`ConfidenceHead.forward` because it is only needed when
    ranking individual interfaces, and its cost grows with the square of the
    chain count.

    Parameters
    ----------
    tm : Tensor
        Expected per-pair TM contributions.
    asym_id : Tensor
        Chain id per token.
    mask : Tensor
        Token validity.

    Returns
    -------
    Tensor
        ``[batch, n_chains, n_chains]`` mean TM between every chain pair.

    """
    batch = tm.shape[0]
    n_chains = int(asym_id.max().item()) + 1 if batch else 0
    out = torch.zeros(batch, n_chains, n_chains, device=tm.device, dtype=tm.dtype)
    mask_f = mask.float()
    for first in range(n_chains):
        rows = (asym_id == first).float() * mask_f
        if rows.sum() == 0:
            continue
        for second in range(n_chains):
            cols = (asym_id == second).float() * mask_f
            weights = rows.unsqueeze(-1) * cols.unsqueeze(-2)
            out[:, first, second] = (tm * weights).sum(dim=(-1, -2)) / (
                weights.sum(dim=(-1, -2)) + EPS
            )
    return out
