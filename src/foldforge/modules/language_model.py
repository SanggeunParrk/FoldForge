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

import numpy as np
import torch
import torch.nn.functional as F

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


class PairShim(torch.nn.Module):
    """Turn a protein language model's hidden states into a pair representation.

    A SEPARATE graph from the folding trunk: the last few layers of a tower, not
    part of AF3's. Its weights ride beside the blob rather than inside it -- 40
    MB against 800 -- and a fold that skips the tower never loads them. What the
    trunk then does with the result, the per-pass dropout and the encoder
    blocks, is ordinary pair work and stays there.

    The layer mix arrives already softmaxed: it is a constant, so it folds to a
    plain (num_layers,) array. It peaks on the LAST layers -- for ESM-C, 79, 80
    and 78 hold 58% of the mass -- so the tower cannot be truncated from the
    top.
    """

    # Declared so the type checker can see what `register_buffer` installs.
    combine: torch.Tensor
    lm_norm_scale: torch.Tensor
    lm_norm_offset: torch.Tensor
    lm_projection_weights: torch.Tensor
    downproject_weights: torch.Tensor
    downproject_bias: torch.Tensor
    pair_mlp_1_weights: torch.Tensor
    pair_mlp_1_bias: torch.Tensor
    pair_mlp_2_weights: torch.Tensor
    pair_mlp_2_bias: torch.Tensor
    pair_norm_scale: torch.Tensor
    pair_norm_offset: torch.Tensor

    #: Every weight the shim needs, by the name the converter writes.
    RECORDS = (
        "combine",
        "lm_norm.scale",
        "lm_norm.offset",
        "lm_projection.weights",
        "downproject.weights",
        "downproject.bias",
        "pair_mlp_1.weights",
        "pair_mlp_1.bias",
        "pair_mlp_2.weights",
        "pair_mlp_2.bias",
        "pair_norm.scale",
        "pair_norm.offset",
    )

    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        missing = [name for name in self.RECORDS if name not in weights]
        if missing:
            message = f"Language-model shim is missing {', '.join(missing)}"
            raise ValueError(message)
        for name in self.RECORDS:
            self.register_buffer(name.replace(".", "_"), weights[name])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map (num_tokens, num_layers, width) to (num_tokens, num_tokens, c_pair).

        In float32 throughout: the trunk amplifies an injection error by around
        fifty, so TF32 matmuls here land at the same magnitude as a featurisation
        bug. The precision costs nothing measurable at this size.
        """
        x = F.layer_norm(
            hidden.float(), hidden.shape[-1:], self.lm_norm_scale, self.lm_norm_offset
        )
        # The mix is applied BEFORE the projection: the projection is linear and
        # shared across layers, so the two orders agree exactly, and this does
        # one matmul where the other does one per layer.
        x = torch.einsum("k,lkc->lc", self.combine, x) @ self.lm_projection_weights
        x = x @ self.downproject_weights + self.downproject_bias
        # The outer product carries BOTH a product and a difference, so the pair
        # sees magnitude and direction rather than only agreement.
        z = torch.concatenate(
            [x[:, None] * x[None, :], x[:, None] - x[None, :]], dim=-1
        )
        z = z @ self.pair_mlp_1_weights + self.pair_mlp_1_bias
        z = F.gelu(z, approximate="none") @ self.pair_mlp_2_weights
        z = z + self.pair_mlp_2_bias
        return F.layer_norm(
            z, z.shape[-1:], self.pair_norm_scale, self.pair_norm_offset
        )


def load_pair_shim(path: Path) -> PairShim:
    """Build the shim from the `.lm.npz` the converter writes beside the blob.

    The variants of a family share the TOWER; they do not share this. Each
    trains its own, and feeding one variant another's reads correlation 0.03
    against native -- which is exactly the mistake one shared filename invites,
    so the file is named per MODEL.
    """
    with np.load(path) as blob:
        return PairShim({name: torch.from_numpy(blob[name]) for name in blob})
