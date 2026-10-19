"""Shared triangle contraction for the released (interleaved) projection layout."""

from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict

Tensor = torch.Tensor
Projection = Callable[[Tensor], Tensor]


class TriangleConfig(BaseModel):
    """The contraction order is a property of the weights."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    outgoing: bool = True


def triangle_update(
    x: Tensor,
    mask: Tensor | None,
    *,
    normalize: Projection,
    project_a: Projection,
    gate_a: Projection,
    center: Projection,
    project_out: Projection,
    gate_out: Projection,
    config: TriangleConfig,
) -> Tensor:
    """Return a raw delta while preserving masked-product rounding order.

    ``project_a`` / ``gate_a`` hold both branches, interleaved per hidden channel.
    """
    value = normalize(x)
    projection = project_a(value).movedim(-1, -3)
    if mask is not None:
        projection = projection * mask.unsqueeze(-3)
    projection = projection * torch.sigmoid(gate_a(value).movedim(-1, -3))
    a, b = projection.unflatten(-3, (-1, 2)).unbind(-3)
    equation = "...cik,...cjk->...cij" if config.outgoing else "...ckj,...cki->...cij"
    contracted = torch.einsum(equation, a, b).movedim(-3, -1)
    return project_out(center(contracted)) * torch.sigmoid(gate_out(value))
