"""Atom/token index utilities shared by the ESMFold2 heads."""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from team_gm import typecheck
from team_gm.modules.tensor_layout import aggregate_atom_to_token


@typecheck
def gather_token_to_atom(
    token_features: Float[torch.Tensor, "B L d"],
    atom_to_token: Int[torch.Tensor, "B A"],
) -> Float[torch.Tensor, "B A d"]:
    """Broadcast per-token features onto their atoms.

    Parameters
    ----------
    token_features : Tensor
        Per-token features.
    atom_to_token : Tensor
        Token index of each atom.

    Returns
    -------
    Tensor
        Per-atom features.

    """
    index = atom_to_token.unsqueeze(-1).expand(-1, -1, token_features.shape[-1])
    return torch.gather(token_features, 1, index)


@typecheck
def gather_representative_atoms(
    coords: Float[torch.Tensor, "B A 3"],
    representative_atom_index: Int[torch.Tensor, "B L"],
) -> Float[torch.Tensor, "B L 3"]:
    """Pick each token's representative atom coordinate.

    Parameters
    ----------
    coords : Tensor
        Per-atom coordinates.
    representative_atom_index : Tensor
        Atom index standing in for each token.

    Returns
    -------
    Tensor
        Per-token coordinates.

    """
    index = representative_atom_index.unsqueeze(-1).expand(-1, -1, coords.shape[-1])
    return torch.gather(coords, 1, index)


@typecheck
def intra_token_index(
    atom_to_token: Int[torch.Tensor, "B A"],
) -> Int[torch.Tensor, "B A"]:
    """Index each atom within its own token, starting at zero.

    Atoms of one token are contiguous, so this is a running count that resets at
    every token boundary — no per-token loop needed.

    Parameters
    ----------
    atom_to_token : Tensor
        Token index of each atom.

    Returns
    -------
    Tensor
        Position of each atom inside its own token.

    """
    same_as_previous = F.pad(
        atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False
    )
    running = torch.cumsum(torch.ones_like(atom_to_token), dim=-1)
    group_start = torch.cummax(running.masked_fill(same_as_previous, 0), dim=-1).values
    return running - group_start


@typecheck
def categorical_mean(
    logits: Float[torch.Tensor, "*batch bins"],
    start: float,
    end: float,
) -> Float[torch.Tensor, "*batch"]:
    """Take the expected value of a categorical distribution over even bins.

    Parameters
    ----------
    logits : Tensor
        Unnormalised bin logits.
    start, end : float
        Outer edges of the bin range.

    Returns
    -------
    Tensor
        Expected value per distribution.

    """
    n_bins = logits.shape[-1]
    edges = torch.linspace(
        start, end, n_bins + 1, device=logits.device, dtype=torch.float32
    )
    centers = (edges[:-1] + edges[1:]) / 2
    # Softmax and the expectation stay fp32 for precision; the result goes back
    # to the caller's dtype so a bf16 track does not get an fp32 tensor injected.
    return (logits.float().softmax(-1) @ centers).to(logits.dtype)


@typecheck
def mean_per_token(
    per_atom: Float[torch.Tensor, "B A"],
    atom_to_token: Int[torch.Tensor, "B A"],
    atom_mask: Bool[torch.Tensor, "B A"],
    n_tokens: int,
) -> Float[torch.Tensor, "B L"]:
    """Average a per-atom quantity over each token's unmasked atoms.

    Parameters
    ----------
    per_atom : Tensor
        Per-atom values.
    atom_to_token : Tensor
        Token index of each atom.
    atom_mask : Tensor
        Atom validity.
    n_tokens : int
        Number of tokens.

    Returns
    -------
    Tensor
        Per-token means; tokens with no valid atoms come out as zero.

    """
    return aggregate_atom_to_token(
        per_atom.unsqueeze(-1), atom_to_token, n_tokens, atom_mask=atom_mask
    ).squeeze(-1)


@typecheck
def scatter_mean_to_token(
    atom_features: Float[torch.Tensor, "B A d"],
    atom_to_token: Int[torch.Tensor, "B A"],
    n_tokens: int,
    atom_mask: Bool[torch.Tensor, "B A"] | None = None,
) -> Float[torch.Tensor, "B L d"]:
    """Average per-atom features into their tokens.

    Masked atoms are routed to a scratch bucket past the last token rather than
    being zeroed, so they do not drag a token's mean toward zero.

    Parameters
    ----------
    atom_features : Tensor
        Per-atom features.
    atom_to_token : Tensor
        Token index of each atom.
    n_tokens : int
        Number of tokens.
    atom_mask : Tensor or None
        Atom validity; ``None`` treats every atom as valid.

    Returns
    -------
    Tensor
        Per-token means.

    """
    return aggregate_atom_to_token(
        atom_features, atom_to_token, n_tokens, atom_mask=atom_mask
    )
