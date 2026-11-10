"""Factories of the engine classes FoldForge's graph holds as its own modules (miniworld-engine 2.5.0).

FoldForge always takes its modules from the engine: a Transition, a TriangleMultiplication, a PairformerBlock ... in a FoldForge model IS
the engine class, so its ``state_dict`` keys are the engine's. This module is the one place that decides HOW such a module is built:

* the **model key** (``MODEL_KEYS``: which ``TriangleAttention(model=...)`` / ``TokenDiT(model=...)`` / ``AtomDiT(model=...)`` class a
  FoldForge family uses),
* the **dtype**: parameters are ``"master"`` (fp32 master weights, bf16 activations under autocast: the engine's ``MasterWeights``) or
  ``"fp32"`` (plain fp32); never native bf16 parameters. The trunk (pairformer / evoformer side) is always ``"master"``,
* the **backend**: every factory builds on ``pytorch`` (the reference); :func:`foldforge.modules.engine.policy.apply_plan` gives the
  modules their backend afterwards (per class, with the guard that implements ``engine.unsupported``).

The factories are thin on purpose: the engine classes own the math, the offline converter (``models/checkpoints/legacy``) owns the released checkpoint layout, this module owns only the
choices FoldForge makes (the keys below, the precision, the options a family's equations need).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from miniworld_engine.modules import (
    LayerNorm,
    MSAPairWeightedAveraging,
    OuterProductMean,
    Transition,
    TriangleAttention,
    TriangleMultiplication,
)
from miniworld_engine.modules.primitives import MasterWeights

if TYPE_CHECKING:
    from torch import nn

Precision = Literal["master", "fp32"]
PRECISIONS = ("master", "fp32")

#: AF3's outer-product-mean divisor is ``n + 1e-3``; the engine kernels serve ``clamp(n, 1)`` only (``count_offset=None``, the default here).
#: They differ by 1e-3 / n relative: below 1e-4 from ~10 valid rows, and at pairs with no valid row (padding). Set to ``1e-3`` for the
#: exact AF3 divisor (the engine then runs that module on its pytorch backend; ``docs/ENGINE-UNSUPPORTED.md``). Read when a block is built.
OPM_COUNT_OFFSET: float | None = None


@dataclass(frozen=True)
class ModelKeys:
    """The engine selector keys of one FoldForge family (``DenseSpec.family``).

    ``triangle`` is ``TriangleAttention(model=)``; ``triangle_multiplication`` is ``TriangleMultiplication(model=)`` (``None``: the plain
    class, which every key but ``rf3`` builds); ``confidence`` the triangle attention of the confidence head where it differs (Chai-1).
    ``token_dit`` / ``atom_dit`` are the ``TokenDiT(model=)`` / ``AtomDiT(model=)`` keys (``None``: the engine has no class: see
    ``docs/ENGINE-UNSUPPORTED.md``)."""

    triangle: str | None
    token_dit: str | None
    atom_dit: str | None
    triangle_multiplication: str | None = None
    confidence: str | None = None


#: ``DenseSpec.family`` -> engine selector keys. The FoldForge registry's ``openfold3`` is the family ``openbind0`` (AF3 bias direction) and
#: its ``openfold3-preview2`` the family ``openfold3`` (transposed bias); the engine keys name the lineage (``openfold3`` /
#: ``openfold3_preview2``).
MODEL_KEYS: dict[str, ModelKeys] = {
    "alphafold3": ModelKeys("af3", "af3", "af3"),
    "intellifold2": ModelKeys("intellifold2", "intellifold2", "intellifold2"),
    "openbind0": ModelKeys("openfold3", "openfold3", "openfold3"),
    "openfold3": ModelKeys("openfold3_preview2", "openfold3_preview2", "openfold3_preview2"),
    "protenix1": ModelKeys("protenix", "protenix", "protenix"),
    "protenix2": ModelKeys("protenix", "protenix_v2", "protenix_v2"),  # triangle map has no ``protenix_v2``
    "opendde": ModelKeys("opendde", "opendde", "opendde"),
    "boltz2": ModelKeys("boltz2", "boltz2", "boltz2"),
    "rosettafold3": ModelKeys("rf3", "rosettafold3", "rosettafold3"),
    "chai1": ModelKeys("chai1", "chai1", "chai1", confidence="chai1_confidence"),
    "esmfold2": ModelKeys(None, "esmfold2", None),
    "esmfold2-fast": ModelKeys(None, "esmfold2", None),
}


def model_keys(family: str) -> ModelKeys:
    """The selector keys of ``family``."""
    try:
        return MODEL_KEYS[family]
    except KeyError:
        known = ", ".join(sorted(MODEL_KEYS))
        raise ValueError(f"no engine selector keys for family {family!r}; known: {known}") from None


def set_norm_eps(module: nn.Module, eps: float) -> None:
    """Set ``eps`` on every LayerNorm of ``module`` that still has the default (released LayerNorms are ``1e-5`` unless a family says
    otherwise; a norm with its own non-default eps, e.g. the grouped OPM ``product_norm`` 0.1, is left alone)."""
    for sub in module.modules():
        if isinstance(sub, torch.nn.LayerNorm) and sub.eps == 1e-5:
            sub.eps = eps


# -------------------------------------------------------------------------------------------------------------------- precision
def finish(module: nn.Module, precision: Precision = "master", *, norm_eps: float | None = None) -> nn.Module:
    """Give a freshly built engine module FoldForge's parameter dtype: ``master`` pins fp32 masters (``set_master_weights``; a module
    without the mixin is plain fp32, its norms already fp32-pinned), ``fp32`` is plain fp32. ``norm_eps`` sets the eps of the LayerNorms
    that still have the default (a family whose released norms differ)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "master" and isinstance(module, MasterWeights):
        module.set_master_weights()
    else:
        module.float()
    if norm_eps is not None:
        set_norm_eps(module, norm_eps)
    return module


# -------------------------------------------------------------------------------------------------------------------- pair-track ops
def transition(
    d_hidden: int, n: int = 4, *, precision: Precision = "master", norm_eps: float | None = None
) -> Transition:
    """``Transition`` (residual included): pair, single, MSA and template transitions of every family."""
    return finish(Transition(d_hidden, n), precision, norm_eps=norm_eps)


def triangle_multiplication(
    d_pair: int,
    *,
    outgoing: bool,
    d_hidden: int | None = None,
    divide_by_length: bool = False,
    precision: Precision = "master",
    norm_eps: float | None = None,
) -> nn.Module:
    """``TriangleMultiplication`` (residual included). ``divide_by_length``: RoseTTAFold3's exact form (``TriangleMultiplicationRF3``, the
    contraction divided by the real length); its fast mode and every other family build the plain class. ``d_hidden``: a stack that widens
    only its projections (the protenix1 template: hidden 128 on pair 64)."""
    module = TriangleMultiplication(
        d_pair, d_hidden=d_hidden, outgoing=outgoing, p_drop=0.0, model="rf3" if divide_by_length else None
    )
    return finish(module, precision, norm_eps=norm_eps)


def triangle_attention(
    d_pair: int,
    n_head: int,
    *,
    starting: bool,
    model: str,
    d_hidden: int | None = None,
    precision: Precision = "master",
    norm_eps: float | None = None,
) -> nn.Module:
    """``TriangleAttention(model=...)`` (residual included); ``model`` is a ``ModelKeys.triangle`` / ``.confidence`` key. ``d_hidden`` is
    the TOTAL hidden width (heads may span more than the pair: the template stacks, pair 64 with 4 x 32)."""
    module = TriangleAttention(d_pair, n_head, d_hidden=d_hidden, starting=starting, p_drop=0.0, model=model)
    return finish(module, precision, norm_eps=norm_eps)


def outer_product_mean(
    d_msa: int,
    d_pair: int,
    d_hidden: int = 32,
    *,
    groups: int = 1,
    projection_bias: bool = False,
    normalize_before_proj: bool = True,
    product_norm_eps: float = 0.1,
    count_offset: float | None = None,
    precision: Precision = "master",
    norm_eps: float | None = None,
) -> OuterProductMean:
    """``OuterProductMean`` (the raw update, no residual). ``normalize_before_proj=False`` is AF3's bias-after-norm form,
    ``projection_bias`` RoseTTAFold3's, ``groups`` Chai-1's. ``count_offset`` (AF3's ``n + 1e-3`` divisor) is served by the pytorch backend
    only: the default ``None`` is the engine's ``clamp(n, 1)`` divisor the kernels serve (see ``docs/ENGINE-UNSUPPORTED.md``)."""
    module = OuterProductMean(
        d_msa, d_pair, d_hidden, normalize_before_proj=normalize_before_proj, projection_bias=projection_bias, groups=groups,
        product_norm_eps=product_norm_eps, count_offset=count_offset,
    )
    return finish(module, precision, norm_eps=norm_eps)


def msa_pair_weighted_averaging(
    d_msa: int, d_pair: int, n_head: int = 8, d_hidden: int | None = None, *, precision: Precision = "master", norm_eps: float | None = None
) -> MSAPairWeightedAveraging:
    """``MSAPairWeightedAveraging`` (residual included)."""
    return finish(MSAPairWeightedAveraging(d_msa, d_pair, n_head, d_hidden, p_drop=0.0), precision, norm_eps=norm_eps)


def layer_norm(
    d: int | tuple[int, ...], *, eps: float = 1e-5, bias: bool = True, elementwise_affine: bool = True
) -> LayerNorm:
    """The engine ``LayerNorm`` (fp32 affine under any cast)."""
    return LayerNorm(d, eps=eps, bias=bias, elementwise_affine=elementwise_affine)


# -------------------------------------------------------------------------------------------------------------------- masks
def token_mask(pair_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Token masks from outer-product pair masks with any leading batch axes."""
    if pair_mask is None:
        return None
    return (pair_mask.diagonal(dim1=-2, dim2=-1) > 0).contiguous()


def residue_mask(pair_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Key mask ``[1, L]`` (bool) from an outer-product pair mask ``[L, L]``: every pair mask of the graph is the outer product of the
    token mask, so its diagonal is the token mask (no host sync)."""
    mask = token_mask(pair_mask)
    return None if mask is None else mask[None]
