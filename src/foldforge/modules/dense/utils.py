# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import numbers
from collections import abc

import torch


def mask_mean(mask, value, dim=None, keepdim: bool = False, eps: float = 1e-10):
    """Masked mean."""
    mask_shape = mask.shape
    value_shape = value.shape

    if len(mask_shape) != len(value_shape):
        message = f"Shapes are not compatible, shapes: {mask_shape}, {value_shape}"
        raise ValueError(message)

    if isinstance(dim, numbers.Integral):
        dim = [dim]
    elif dim is None:
        dim = list(range(len(mask_shape)))
    if not (isinstance(dim, abc.Iterable)):
        message = 'axis needs to be either an iterable, integer or "None"'
        raise TypeError(message)

    dim = tuple(int(axis) for axis in dim)
    broadcast_factor = 1.0
    for dim_ in dim:
        value_size = value_shape[dim_]
        mask_size = mask_shape[dim_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            error = f"Shapes are not compatible, shapes: {mask_shape}, {value_shape}"
            if mask_size != value_size:
                message = error
                raise ValueError(message)

    return torch.sum(mask * value, keepdim=keepdim, dim=dim) / (
        torch.clamp(
            torch.sum(mask, keepdim=keepdim, dim=dim) * broadcast_factor, min=eps
        )
    )
