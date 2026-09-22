"""Protein language-model embeddings, for the families whose token stream needs them.

Two AF3-family predictors read a language model: one takes ESM2's residue
embeddings as most of its token stream, the other conditions on ESM-C. The two
alphabets are the same thirty-three tokens in the same order and differ in
exactly one slot -- index 31 is ESM2's ``<null_1>`` and ESM-C's chain separator
-- so they are written out separately rather than shared. Neither token appears
in a single-chain input, and a silent one-token drift is precisely the failure
that would survive every shape check.

The models themselves are not reimplemented. Anything that maps a token tensor
to per-token hidden states will do; :func:`load_traced` wraps a released
TorchScript trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from foldforge.data.constants import residue_names

if TYPE_CHECKING:
    from pathlib import Path

_COMMON = (
    "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . -"
)
#: ESM2's alphabet.
ESM2_VOCAB: tuple[str, ...] = tuple((_COMMON + " <null_1> <mask>").split())
#: ESM-C's alphabet; slot 31 is a chain separator where ESM2 has a spare.
ESMC_VOCAB: tuple[str, ...] = tuple((_COMMON + " | <mask>").split())

BOS, PAD, EOS, MASK = 0, 1, 2, 32

#: One-letter code of each protein residue type, indexed as the structure side
#: numbers them; everything past the protein types has no residue letter.
_ONE_LETTER = residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP


class TokenEmbedder(Protocol):
    """A model mapping a token tensor to per-token hidden states."""

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, length, channels]`` for ``[batch, length]`` tokens."""
        ...


def residue_tokens(
    aatype: torch.Tensor, vocab: tuple[str, ...] = ESM2_VOCAB
) -> torch.Tensor:
    """Map structure-side residue types onto ``vocab``'s ids.

    A token with no residue letter -- a nucleotide or a ligand -- takes the
    unknown-residue id, which is what the alphabet has for "not one of these".
    """
    lookup = torch.full((len(residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP),), 3)
    for index, letter in enumerate(_ONE_LETTER):
        if letter in vocab:
            lookup[index] = vocab.index(letter)
    return lookup.to(aatype.device)[aatype.to(torch.int64).clamp(0, len(lookup) - 1)]


def embed_chains(
    model: TokenEmbedder,
    aatype: torch.Tensor,
    asym_id: torch.Tensor,
    mask: torch.Tensor,
    vocab: tuple[str, ...] = ESM2_VOCAB,
) -> torch.Tensor:
    """Return one embedding per token, each chain wrapped and run on its own.

    Every tower wraps a chain as ``[BOS, ids..., EOS]``. Running the chains
    separately rather than packed keeps attention inside a chain without needing
    the tower to honour a sequence id, and the rows come back in token order.
    """
    tokens = residue_tokens(aatype, vocab)
    real = mask.to(torch.bool)
    out: torch.Tensor | None = None
    for chain in torch.unique(asym_id[real]):
        where = real & (asym_id == chain)
        ids = tokens[where]
        wrapped = torch.cat(
            [
                ids.new_full((1,), BOS),
                ids,
                ids.new_full((1,), EOS),
            ]
        )[None]
        # Drop the wrapper rows: they are the tower's, not the structure's.
        hidden = model(wrapped)[0, 1:-1]
        if out is None:
            out = hidden.new_zeros((aatype.shape[0], hidden.shape[-1]))
        out[where] = hidden.to(out.dtype)
    if out is None:
        message = "No real token to embed; the mask is empty"
        raise ValueError(message)
    return out


def load_traced(path: str | Path, device: torch.device | None = None) -> TokenEmbedder:
    """Load a released TorchScript tower, frozen and in eval mode."""
    model = torch.jit.load(str(path), map_location=device or "cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad=False)
    return model
