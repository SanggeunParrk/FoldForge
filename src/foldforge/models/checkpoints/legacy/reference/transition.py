"""Dense-graph transition equation with released projection and chunking layouts (converter graph only)."""

from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict, Field

from foldforge.modules.dense.diffusion_transformer import swiglu

Tensor = torch.Tensor
Projection = Callable[[Tensor], Tensor]


class TransitionConfig(BaseModel):
    """Memory policy; projection layout is supplied by checkpoint adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    flatten: bool = False
    chunk_training: bool = False
    chunk_threshold: int = Field(default=3200, gt=0)
    chunks: int = Field(default=8, gt=0)
    max_rows: int | None = Field(default=None, gt=0)


def transition(
    x: Tensor,
    *,
    normalize: Projection,
    project_a: Projection,
    project_b: Projection | None,
    project_out: Projection,
    config: TransitionConfig,
    training: bool,
) -> Tensor:
    """Return a delta, preserving inference chunk boundaries and output dtype."""

    def compute(value: Tensor) -> Tensor:
        return project_out(swiglu(normalize(value), project_a, project_b))

    if not config.flatten or (training and not config.chunk_training):
        return compute(x)
    shape = x.shape
    value = x.reshape(-1, shape[-1])
    if config.max_rows is not None and value.shape[0] > config.max_rows:
        chunks = value.split(config.max_rows, dim=-2)
    elif shape[-2] >= config.chunk_threshold:
        chunks = value.chunk(config.chunks, dim=-2)
    else:
        chunks = (value,)
    result = torch.empty_like(value)
    start = 0
    for chunk in chunks:
        output = compute(chunk)
        result[start : start + len(chunk)] = output
        start += len(chunk)
    return result.reshape(shape)
