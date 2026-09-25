# A released tower loads through a pinned Transformers fork; import it only
# where one is actually asked for.
# ruff: noqa: PLC0415
"""Protein language models: their alphabets, their towers, and what reads them.

Three predictors here read one. Two take residue embeddings as part of their
token stream; the third conditions on a PAIR built from every hidden layer, and
the shim that builds it is at the bottom of this file.

The two
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
from jaxtyping import Bool, Float, Int
from team_gm import typecheck

from foldforge.data.constants import residue_names

if TYPE_CHECKING:
    from collections.abc import Sequence
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
    is_protein: torch.Tensor,
    vocab: tuple[str, ...] = ESM2_VOCAB,
) -> torch.Tensor:
    """Return one embedding per token, each protein chain wrapped and run on its own.

    Every tower wraps a chain as ``[BOS, ids..., EOS]``. Running the chains
    separately rather than packed keeps attention inside a chain without needing
    the tower to honour a sequence id, and the rows come back in token order.

    Only protein chains are embedded; every other token keeps a zero row, as the
    release gives it. Run on everything, a ligand's atoms went in as a string of
    unknown residues and came back as a protein's embedding.
    """
    tokens = residue_tokens(aatype, vocab)
    real = mask.to(torch.bool)
    protein = is_protein.to(torch.bool)
    out: torch.Tensor | None = None
    for chain in torch.unique(asym_id[real]):
        where = real & (asym_id == chain)
        if not protein[where].any():
            continue
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
        if not real.any():
            message = "No real token to embed; the mask is empty"
            raise ValueError(message)
        # No protein chain at all: nothing for the tower to read, and only
        # its width to learn, from an empty wrapped sequence.
        width = model(tokens.new_tensor([[BOS, EOS]])).shape[-1]
        out = torch.zeros((aatype.shape[0], width), device=aatype.device)
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


class LanguageModelOutput(Protocol):
    """Hidden representations returned when the language model exposes its layers."""

    @property
    def hidden_states(self) -> Sequence[torch.Tensor] | None:
        """Return representations ordered from embedding to final layer."""
        ...


class NormLinearParameters(Protocol):
    """Parameter contract of the Biohub fused norm-linear module."""

    @property
    def d_in(self) -> int:
        """Return the original module d in."""
        ...

    @property
    def eps(self) -> float:
        """Return the original module eps."""
        ...

    @property
    def layer_norm_weight(self) -> torch.Tensor:
        """Return the original module layer norm weight."""
        ...

    @property
    def layer_norm_bias(self) -> torch.Tensor | None:
        """Return the original module layer norm bias."""
        ...

    @property
    def weight(self) -> torch.Tensor:
        """Return the original module weight."""
        ...


class NormMLPParameters(Protocol):
    """Parameter contract of the Biohub fused norm-SwiGLU module."""

    @property
    def hidden_size(self) -> int:
        """Return the original module hidden size."""
        ...

    @property
    def eps(self) -> float:
        """Return the original module eps."""
        ...

    @property
    def layer_norm_weight(self) -> torch.Tensor:
        """Return the original module layer norm weight."""
        ...

    @property
    def layer_norm_bias(self) -> torch.Tensor | None:
        """Return the original module layer norm bias."""
        ...

    @property
    def fc1_weight(self) -> torch.Tensor:
        """Return the original module fc1 weight."""
        ...

    @property
    def fc2_weight(self) -> torch.Tensor:
        """Return the original module fc2 weight."""
        ...


class LanguageModel(Protocol):
    """Minimal interface ESMFold2 needs from a protein language model."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        sequence_id: torch.Tensor,
        *,
        output_hidden_states: bool = True,
    ) -> LanguageModelOutput:
        """Return an object exposing ``hidden_states`` as ``[layers, B, L, D]``."""
        ...


@typecheck
def build_lm_inputs(
    input_ids: Int[torch.Tensor, "B L"],
    asym_id: Int[torch.Tensor, "B L"],
    residue_index: Int[torch.Tensor, "B L"],
    is_protein: Bool[torch.Tensor, "B L"],
    mask: Bool[torch.Tensor, "B L"],
    pad_to_multiple: int | None = None,
) -> tuple[
    Int[torch.Tensor, "B M"], Int[torch.Tensor, "B M"], Int[torch.Tensor, "B L"]
]:
    """Collapse structure tokens to residues and pack chains for the LM.

    Parameters
    ----------
    input_ids : Tensor
        Per-token residue ids.
    asym_id, residue_index : Tensor
        Chain id and residue number; together they identify a residue.
    is_protein : Tensor
        True on the protein tokens; only those reach a tower.

        A BOOLEAN, not a molecule-type class. It was a class, compared
        against ``PROTEIN_MOL_TYPE == 0``, and every caller had the boolean --
        so ``mol_type == 0`` selected exactly the tokens it meant to drop. On
        an all-protein input that left nothing to pack, and the tower returned
        zeros that the shim turned into one constant vector repeated at every
        pair position. The fold still ran, and looked like a fold.
    mask : Tensor
        Token validity.
    pad_to_multiple : int or None
        Round the packed length up to this multiple. fp8 attention kernels
        require the flattened token count to be divisible by 8.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Packed ``input_ids``, the matching ``sequence_id`` (``-1`` on padding),
        and a ``[B, L]`` map from structure token to packed position, ``-1``
        where a token has no LM counterpart.
    """
    batch, length = input_ids.shape
    device = input_ids.device
    protein = is_protein.bool() & mask
    if not bool(protein.any()):
        # Silence here is what hid the class-versus-flag mix-up above: an empty
        # pack gives zero hidden states, a constant pair, and a fold.
        message = "No protein token reached the language model"
        raise ValueError(message)

    packed: list[torch.Tensor] = []
    position_maps: list[torch.Tensor] = []
    for index in range(batch):
        keep = protein[index]
        ids = input_ids[index][keep]
        chains = asym_id[index][keep]
        residues = residue_index[index][keep]

        residue_of_token, first_token = _collapse_to_residues(chains, residues)
        residue_ids = ids[first_token]
        residue_chains = chains[first_token]

        sequence, residue_positions = _pack_chains(residue_ids, residue_chains, device)
        packed.append(sequence)

        position_map = torch.full((length,), -1, device=device, dtype=torch.long)
        position_map[keep.nonzero(as_tuple=True)[0]] = residue_positions[
            residue_of_token
        ]
        position_maps.append(position_map)

    width = max(int(sequence.numel()) for sequence in packed)
    if pad_to_multiple:
        width = -(-width // pad_to_multiple) * pad_to_multiple
    lm_input_ids = torch.full((batch, width), PAD, device=device, dtype=input_ids.dtype)
    for index, sequence in enumerate(packed):
        lm_input_ids[index, : sequence.numel()] = sequence

    # One id per chain, so attention stays inside a chain. Padding gets -1.
    sequence_id = (lm_input_ids == BOS).cumsum(dim=1) - 1
    sequence_id = sequence_id.masked_fill(lm_input_ids == PAD, -1)
    return lm_input_ids, sequence_id, torch.stack(position_maps)


def _collapse_to_residues(
    chains: torch.Tensor, residues: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each token to a residue index, keeping first-appearance order."""
    device = chains.device
    keys = torch.stack((chains, residues), dim=1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    positions = torch.arange(keys.shape[0], device=device, dtype=torch.long)
    first = torch.full(
        (unique.shape[0],), keys.shape[0], device=device, dtype=torch.long
    )
    first.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    # torch.unique sorts by key; re-order so residues follow the input order.
    order = torch.argsort(first)
    relabel = torch.empty_like(order)
    relabel[order] = torch.arange(unique.shape[0], device=device, dtype=torch.long)
    return relabel[inverse], first[order]


def _pack_chains(
    residue_ids: torch.Tensor, residue_chains: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join chains as ``[BOS] c1 [EOS] [BOS] c2 ... [EOS]``."""
    parts = [torch.tensor([BOS], device=device, dtype=residue_ids.dtype)]
    positions = torch.empty(residue_ids.shape[0], device=device, dtype=torch.long)
    cursor = 1  # position 0 is the leading BOS
    chain_ids = residue_chains.unique(sorted=True)
    for order, chain in enumerate(chain_ids):
        members = (residue_chains == chain).nonzero(as_tuple=True)[0]
        parts.append(residue_ids[members])
        positions[members] = torch.arange(
            cursor, cursor + members.shape[0], device=device, dtype=torch.long
        )
        cursor += members.shape[0]
        if order < len(chain_ids) - 1:
            parts.append(
                torch.tensor([EOS, BOS], device=device, dtype=residue_ids.dtype)
            )
            cursor += 2
    parts.append(torch.tensor([EOS], device=device, dtype=residue_ids.dtype))
    return torch.cat(parts), positions


@typecheck
def scatter_lm_hidden_states(
    hidden_states: Float[torch.Tensor, "n_lm B M D"],
    position_map: Int[torch.Tensor, "B L"],
) -> Float[torch.Tensor, "B L n_lm D"]:
    """Spread packed LM hidden states back onto the structure tokens.

    Parameters
    ----------
    hidden_states : Tensor
        Per-layer LM outputs over the packed sequence.
    position_map : Tensor
        Structure token to packed position, ``-1`` where there is none.

    Returns
    -------
    Tensor
        Per-token, per-layer hidden states; zero for tokens the LM never saw.
    """
    n_layers, batch, _, width = hidden_states.shape
    length = position_map.shape[1]
    out = hidden_states.new_zeros(batch, length, n_layers, width)
    for index in range(batch):
        present = position_map[index] >= 0
        if not bool(present.any()):
            continue
        taken = hidden_states[:, index, position_map[index][present], :]
        out[index, present.nonzero(as_tuple=True)[0]] = taken.permute(1, 0, 2)
    return out


def mask_lm_inputs(
    input_ids: torch.Tensor,
    fraction: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly replace residues with the mask token, as at training time.

    BOS/EOS/PAD are never masked, so the chain structure survives.

    Parameters
    ----------
    input_ids : Tensor
        Packed LM input ids.
    fraction : float
        Probability of masking each residue.
    generator : torch.Generator or None
        Source of randomness.

    Returns
    -------
    Tensor
        Input ids with some residues replaced.
    """
    if fraction <= 0.0:
        return input_ids
    special = (input_ids == BOS) | (input_ids == PAD) | (input_ids == EOS)
    draw = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    return input_ids.masked_fill((draw < fraction) & ~special, MASK)


@torch.no_grad()
def compute_lm_hidden_states(
    language_model: LanguageModel,
    input_ids: torch.Tensor,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    is_protein: torch.Tensor,
    mask: torch.Tensor,
    *,
    mask_fraction: float = 0.0,
    pad_to_multiple: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the language model and return per-structure-token hidden states.

    Parameters
    ----------
    language_model : LanguageModel
        Anything matching the :class:`LanguageModel` protocol.
    input_ids, asym_id, residue_index, is_protein, mask : Tensor
        Structure-token features; see :func:`build_lm_inputs`.
    mask_fraction : float
        Residue masking probability applied before the LM runs.
    pad_to_multiple : int or None
        Pad the packed length to this multiple.
    generator : torch.Generator or None
        Source of randomness for masking.

    Returns
    -------
    Tensor
        ``[B, L, n_layers + 1, d_model]`` hidden states, detached.
    """
    lm_input_ids, sequence_id, position_map = build_lm_inputs(
        input_ids, asym_id, residue_index, is_protein, mask, pad_to_multiple
    )
    lm_input_ids = mask_lm_inputs(lm_input_ids, mask_fraction, generator=generator)
    output = language_model(
        lm_input_ids, sequence_id=sequence_id, output_hidden_states=True
    )
    if output.hidden_states is None:
        message = "Language model did not return requested hidden states"
        raise ValueError(message)
    hidden = torch.stack(list(output.hidden_states))
    return scatter_lm_hidden_states(hidden, position_map).detach()


class _FP32NormLinear(torch.nn.Module):
    """ESMC's PyTorch fallback with explicit FP32 norm and native-dtype GEMM."""

    def __init__(self, original: NormLinearParameters) -> None:
        super().__init__()
        self.d_in, self.eps = original.d_in, original.eps
        self.layer_norm_weight = original.layer_norm_weight
        self.layer_norm_bias = original.layer_norm_bias
        self.weight = original.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        normalized = torch.nn.functional.layer_norm(
            x.float(),
            (self.d_in,),
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
        ).to(x.dtype)
        return torch.nn.functional.linear(normalized, self.weight)


class _FP32NormMLP(torch.nn.Module):
    """ESMC's SwiGLU fallback with explicit FP32 normalization."""

    def __init__(self, original: NormMLPParameters) -> None:
        super().__init__()
        self.hidden_size, self.eps = original.hidden_size, original.eps
        self.layer_norm_weight = original.layer_norm_weight
        self.layer_norm_bias = original.layer_norm_bias
        self.fc1_weight, self.fc2_weight = original.fc1_weight, original.fc2_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the module output."""
        normalized = torch.nn.functional.layer_norm(
            x.float(),
            (self.hidden_size,),
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
        ).to(x.dtype)
        a, b = torch.nn.functional.linear(normalized, self.fc1_weight).chunk(2, dim=-1)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(a) * b, self.fc2_weight
        )


def _prepare_esmc_norms(model: torch.nn.Module) -> None:
    from transformers.models.esmc.modeling_esmc import (
        _PyTorchLayerNormLinear,
        _PyTorchLayerNormMLP,
    )

    for name, child in list(model.named_children()):
        if isinstance(child, _PyTorchLayerNormLinear):
            setattr(model, name, _FP32NormLinear(child))
        elif isinstance(child, _PyTorchLayerNormMLP):
            setattr(model, name, _FP32NormMLP(child))
        else:
            _prepare_esmc_norms(child)


def load_esmc(
    path: str, device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
) -> LanguageModel:
    """Load the released ESMC model through ``transformers``.

    The environment pins the Biohub Transformers fork that implements this
    checkpoint API. Import it lazily when a language model is requested.

    Parameters
    ----------
    path : str
        Checkpoint directory or Hugging Face id.
    device : torch.device or None
        Device to place the model on.
    dtype : torch.dtype
        Weight dtype; the reference uses bfloat16.

    Returns
    -------
    LanguageModel
        The loaded model, frozen and in eval mode.
    """
    from transformers.models.esmc.modeling_esmc import ESMCModel

    from foldforge.models.precision import inference_precision

    model = ESMCModel.from_pretrained(path)
    _prepare_esmc_norms(model)
    inference_precision(model, device or torch.device("cpu"), dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # noqa: FBT003 - PyTorch binding is positional
    return model
