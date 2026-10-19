"""Shared transition equations with released projection and chunking layouts."""

from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.nn import functional as F

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


def swiglu(
    x: Tensor, project_a: Projection, project_b: Projection | None = None
) -> Tensor:
    """Preserve one packed GEMM or two separate checkpoint projections."""
    if project_b is None:
        a, b = project_a(x).chunk(2, dim=-1)
    else:
        a, b = project_a(x), project_b(x)
    return F.silu(a) * b


def adaptive_normalization(
    x: Tensor,
    cond: Tensor | None,
    *,
    normalize: Projection,
    normalize_cond: Projection | None = None,
    scale: Projection | None = None,
    shift: Projection | None = None,
    identity_scale: bool = False,
    activate_cond: bool = False,
) -> Tensor:
    """Conditioned normalization, with the original separate normalization steps.

    ``identity_scale`` selects the identity-centred form, ``(s + 1) * x``, over
    the sigmoid one. Both start near the identity at initialisation, so which a
    checkpoint was trained under is invisible until its weights are loaded.
    Families on that form also leave the conditioning unnormalised, which is why
    ``normalize_cond`` may be absent here and is not a no-op when it is not:
    a LayerNorm re-centres and re-scales even at unit scale.

    ``activate_cond`` passes the conditioning through SiLU before the scale and
    shift read it, which one family's atom blocks do in place of normalising it.
    """
    value = normalize(x)
    if cond is None:
        return value
    if scale is None or shift is None:
        msg = "Conditioned normalization requires a scale and a shift"
        raise ValueError(msg)
    if normalize_cond is None and not identity_scale:
        msg = "Sigmoid conditioning requires the condition norm"
        raise ValueError(msg)
    context = cond if normalize_cond is None else normalize_cond(cond)
    if activate_cond:
        context = torch.nn.functional.silu(context)
    gate = scale(context) + 1.0 if identity_scale else torch.sigmoid(scale(context))
    return gate * value + shift(context)


def gated_projection(
    x: Tensor, cond: Tensor | None, *, project: Projection, gate: Projection | None
) -> Tensor:
    """Project the branch and optionally apply its sigmoid output gate."""
    output = project(x)
    return output if cond is None else torch.sigmoid(gate(cond)) * output


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
