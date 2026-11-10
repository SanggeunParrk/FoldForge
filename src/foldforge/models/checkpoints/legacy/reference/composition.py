"""Dense-graph block ordering: the MSA-stack update orders of the released layout (converter graph only)."""

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


