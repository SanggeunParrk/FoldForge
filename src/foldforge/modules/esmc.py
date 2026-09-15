# Optional chemistry/GPU backends load at the selected execution boundary.
# ruff: noqa: PLC0415
"""Feeding ESMFold2's structure tokens to the ESMC language model.

The LM was trained on one token per residue, but the structure side tokenises
some residues into several atoms — a modified residue such as HYP or MSE spans
multiple structure tokens that share one ``(asym_id, residue_index)`` key. So
the token stream has to be collapsed to per-residue before the LM sees it and
scattered back afterwards.

Chains are packed into a single sequence as
``[BOS] chain1 [EOS] [BOS] chain2 ... [EOS]`` with a ``sequence_id`` that keeps
attention from crossing chain boundaries.

The LM itself is not reimplemented here. ESMC is a plain transformer that shares
nothing with the folding stack, so there is no kernel to unify — anything
matching :class:`LanguageModel` will do, and :func:`load_esmc` provides the
released one.
"""

from collections.abc import Sequence
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Int
from team_gm import typecheck

BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MASK_TOKEN_ID = 32
#: ``mol_type`` value marking a protein token; only these reach the LM.
PROTEIN_MOL_TYPE = 0


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
    mol_type: Int[torch.Tensor, "B L"],
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
    mol_type : Tensor
        Molecule type per token; non-protein tokens are dropped.
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
    is_protein = (mol_type == PROTEIN_MOL_TYPE) & mask

    packed: list[torch.Tensor] = []
    position_maps: list[torch.Tensor] = []
    for index in range(batch):
        keep = is_protein[index]
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
    lm_input_ids = torch.full(
        (batch, width), PAD_TOKEN_ID, device=device, dtype=input_ids.dtype
    )
    for index, sequence in enumerate(packed):
        lm_input_ids[index, : sequence.numel()] = sequence

    # One id per chain, so attention stays inside a chain. Padding gets -1.
    sequence_id = (lm_input_ids == BOS_TOKEN_ID).cumsum(dim=1) - 1
    sequence_id = sequence_id.masked_fill(lm_input_ids == PAD_TOKEN_ID, -1)
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
    parts = [torch.tensor([BOS_TOKEN_ID], device=device, dtype=residue_ids.dtype)]
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
                torch.tensor(
                    [EOS_TOKEN_ID, BOS_TOKEN_ID], device=device, dtype=residue_ids.dtype
                )
            )
            cursor += 2
    parts.append(torch.tensor([EOS_TOKEN_ID], device=device, dtype=residue_ids.dtype))
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
    special = (
        (input_ids == BOS_TOKEN_ID)
        | (input_ids == PAD_TOKEN_ID)
        | (input_ids == EOS_TOKEN_ID)
    )
    draw = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    return input_ids.masked_fill((draw < fraction) & ~special, MASK_TOKEN_ID)


@torch.no_grad()
def compute_lm_hidden_states(
    language_model: LanguageModel,
    input_ids: torch.Tensor,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    mol_type: torch.Tensor,
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
    input_ids, asym_id, residue_index, mol_type, mask : Tensor
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
        input_ids, asym_id, residue_index, mol_type, mask, pad_to_multiple
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
