"""Shared block ordering, independent of model feature and attention layouts.

Callables adapt masks, windows and released projection layouts at the model
boundary. Delta branches and branches owning their residual are deliberately
different contracts; no fused residual is undone by subtraction.
"""

from collections.abc import Callable
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

Tensor = torch.Tensor


class MSAUpdateConfig(BaseModel):
    """Specify the checkpoint's communication order and last-block policy."""

    model_config = ConfigDict(frozen=True)
    order: Literal["opm_first", "msa_first"] = "opm_first"
    update_msa: bool = True
    return_msa: bool = True


def conditioned_residual(
    single: Tensor,
    cond: Tensor,
    *,
    attention: Callable[[Tensor], Tensor],
    transition: Callable[[Tensor, Tensor], Tensor],
    residual_path: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Compose raw attention and transition deltas with one residual each."""
    delta = attention(single)
    single = single + (delta if residual_path is None else residual_path(delta))
    delta = transition(single, cond)
    return single + (delta if residual_path is None else residual_path(delta))


def msa_pair_update(
    msa: Tensor,
    pair: Tensor,
    *,
    config: MSAUpdateConfig,
    outer_product_delta: Callable[[Tensor], Tensor],
    msa_residual: Callable[[Tensor, Tensor], Tensor],
    pair_residual: Callable[[Tensor], Tensor],
) -> tuple[Tensor | None, Tensor]:
    """Run AF3/Protenix or Boltz/OpenDDE ordering without changing its algorithm."""
    if config.order == "msa_first" and config.update_msa:
        msa = msa_residual(msa, pair)
    pair = pair + outer_product_delta(msa)
    if config.order == "opm_first" and config.update_msa:
        msa = msa_residual(msa, pair)
    pair = pair_residual(pair)
    return (msa if config.return_msa else None), pair


def msa_row_update(
    msa: Tensor,
    pair: Tensor,
    *,
    attention_residual: Callable[[Tensor, Tensor], Tensor],
    transition_residual: Callable[[Tensor], Tensor],
    chunk_size: int | None = None,
) -> Tensor:
    """Update MSA rows, preserving independent leading batch dimensions."""
    size = msa.shape[-3]
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("MSA chunk size must be positive")
    if size == 0:
        return msa
    chunks = []
    for start in range(0, size, chunk_size or size):
        value = msa[..., start : start + (chunk_size or size), :, :]
        value = attention_residual(value, pair)
        chunks.append(transition_residual(value))
    return torch.cat(chunks, dim=-3)


def template_embedding_mean(
    count: int,
    *,
    embed: Callable[[int], Tensor],
    empty: Tensor,
    project: Callable[[Tensor], Tensor],
    eps: float = 1e-7,
    present: Callable[[int], bool] | None = None,
) -> Tensor:
    """Aggregate model-specific template features after their shared pair stack.

    Sequential accumulation preserves the released BF16 reduction order. The
    empty tensor supplies shape/device/dtype when there are no templates.

    ``present`` makes the divisor the number of slots that carry a template
    rather than the number of slots. The two agree on a fully populated input
    and differ by the padding factor on any other, which is a rescaling of the
    whole term and not a small one.
    """
    if count < 0:
        raise ValueError("Template count cannot be negative")
    total = empty
    filled = 0
    for index in range(count):
        total = total + embed(index)
        if present is not None and present(index):
            filled += 1
    divisor = max(float(filled), 1.0) if present is not None else count + eps
    return project(torch.relu(total / divisor))
