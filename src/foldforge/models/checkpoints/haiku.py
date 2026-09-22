# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Model param loading."""

from __future__ import annotations

import bisect
import collections
import contextlib
import io
import logging
import operator
import os
import pathlib
import re
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
import zstandard

from foldforge.models.config.fourier import _BIAS, _WEIGHT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from _typeshed import WriteableBuffer


logger = logging.getLogger(__name__)


class BinaryReader(Protocol):
    """A blocking binary input stream used by checkpoint records."""

    def read(self, size: int = -1, /) -> bytes | None:
        """Read bytes, returning empty bytes at end of stream."""
        ...


class ParameterModule(Protocol):
    """Expose named children and parameters used by checkpoint translation."""

    def __getattr__(self, name: str, /) -> Any:
        """Read a checkpoint-specific parameter or child module."""
        ...


type ParamTree = dict[str, Any]


class RecordError(Exception):
    """Error reading a record."""


def encode_record(scope: str, name: str, arr: np.ndarray) -> bytes:
    """Encode a single haiku param as bytes, preserving non-numpy dtypes."""
    scope_bytes = scope.encode("utf-8")
    name_bytes = name.encode("utf-8")
    shape = arr.shape
    dtype = str(arr.dtype).encode("utf-8")
    arr = np.ascontiguousarray(arr)
    if sys.byteorder == "big":
        arr = arr.byteswap()
    arr_buffer = arr.tobytes("C")
    header = struct.pack(
        "<5i",
        len(scope_bytes),
        len(name_bytes),
        len(dtype),
        len(shape),
        len(arr_buffer),
    )
    return header + b"".join(
        (
            scope_bytes,
            name_bytes,
            dtype,
            struct.pack(f"<{len(shape)}i", *shape),
            arr_buffer,
        )
    )


_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _read_record(stream: BinaryReader) -> tuple[str, str, torch.Tensor] | None:
    """Read a record encoded by `_encode_record` from a byte stream."""
    header_size = struct.calcsize("<5i")
    header = stream.read(header_size)
    if not header:
        return None
    if len(header) < header_size:
        msg = f"Incomplete header: {len(header)=} < {header_size=}"
        raise RecordError(msg)
    (scope_len, name_len, dtype_len, shape_len, arr_buffer_len) = struct.unpack(
        "<5i", header
    )
    fmt = f"<{scope_len}s{name_len}s{dtype_len}s{shape_len}i"
    payload_size = struct.calcsize(fmt) + arr_buffer_len
    payload = stream.read(payload_size)
    if payload is None:
        message = "Checkpoint stream returned no data while reading a record"
        raise RecordError(message)
    if len(payload) < payload_size:
        msg = f"Incomplete payload: {len(payload)=} < {payload_size=}"
        raise RecordError(msg)
    scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
    scope = scope.decode("utf-8")
    name = name.decode("utf-8")
    dtype = dtype.decode("utf-8")
    arr = torch.frombuffer(
        bytearray(payload[-arr_buffer_len:]), dtype=_DTYPE_MAP[dtype]
    )
    arr = torch.reshape(arr, shape)
    return scope, name, arr


def read_records(stream: BinaryReader) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Read the contents of a byte stream."""
    while record := _read_record(stream):
        yield record


class _MultiFileIO(io.RawIOBase):
    """A file-like object that presents a concatenated view of multiple files."""

    def __init__(self, files: list[pathlib.Path]) -> None:
        if not files:
            message = "At least one checkpoint shard is required"
            raise ValueError(message)
        self._files = files
        self._stack = contextlib.ExitStack()
        self._handles = [self._stack.enter_context(file.open("rb")) for file in files]
        self._sizes = []
        for handle in self._handles:
            handle.seek(0, os.SEEK_END)
            self._sizes.append(handle.tell())
        self._length = sum(self._sizes)
        self._offsets = [0]
        for s in self._sizes[:-1]:
            self._offsets.append(self._offsets[-1] + s)
        self._abspos = 0
        self._relpos = self._abs_to_rel(0)

    def _abs_to_rel(self, pos: int) -> tuple[int, int]:
        idx = bisect.bisect_right(self._offsets, pos) - 1
        return idx, pos - self._offsets[idx]

    def close(self) -> None:
        """Close every shard and mark the stream closed."""
        try:
            self._stack.close()
        finally:
            super().close()

    def fileno(self) -> int:
        """Compute fileno."""
        message = "A concatenated checkpoint has no single file descriptor"
        raise io.UnsupportedOperation(message)

    def readable(self) -> bool:
        """Compute readable."""
        return True

    def tell(self) -> int:
        """Compute tell."""
        if self.closed:
            message = "I/O operation on a closed checkpoint stream"
            raise ValueError(message)
        return self._abspos

    def seek(self, pos: int, whence: int = os.SEEK_SET, /) -> int:
        """Move within the concatenated shards and return the absolute position."""
        if self.closed:
            message = "I/O operation on a closed checkpoint stream"
            raise ValueError(message)
        position = operator.index(pos)
        if whence == os.SEEK_CUR:
            position += self._abspos
        elif whence == os.SEEK_END:
            position += self._length
        elif whence != os.SEEK_SET:
            message = f"Invalid whence: {whence}"
            raise ValueError(message)
        if position < 0:
            message = "Checkpoint stream position cannot be negative"
            raise ValueError(message)
        self._abspos = position
        self._relpos = self._abs_to_rel(position)
        return position

    def seekable(self) -> bool:
        """Return whether the underlying shards support seeking."""
        return True

    def readinto(self, b: WriteableBuffer, /) -> int:
        """Read across shard boundaries, including empty shards and end-of-file."""
        if self.closed:
            message = "I/O operation on a closed checkpoint stream"
            raise ValueError(message)
        result = 0
        mem = memoryview(b).cast("B")
        while mem and self._abspos < self._length:
            handle = self._handles[self._relpos[0]]
            handle.seek(self._relpos[1])
            count = handle.readinto(mem)
            if count == 0:
                message = "Checkpoint shard ended before its recorded size"
                raise EOFError(message)
            result += count
            self._abspos += count
            self._relpos = self._abs_to_rel(self._abspos)
            mem = mem[count:]
        return result


@contextlib.contextmanager
def open_for_reading(
    model_files: list[pathlib.Path], *, is_compressed: bool
) -> Iterator[BinaryReader]:
    """Compute open for reading."""
    with io.BufferedReader(_MultiFileIO(model_files)) as f:
        if is_compressed:
            yield zstandard.ZstdDecompressor().stream_reader(f)
        else:
            yield f


def _match_model(
    paths: list[pathlib.Path], pattern: re.Pattern[str]
) -> dict[str, list[pathlib.Path]]:
    """Match files in a directory with a pattern, and group by model name."""
    models = collections.defaultdict(list)
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match:
            models[match.group("model_name")].append(path)
    return {
        name: sorted(shards, key=lambda path: int(re.findall(r"\d+", path.name)[-1]))
        for name, shards in models.items()
    }


def select_model_files(
    model_dir: pathlib.Path, model_name: str | None = None
) -> tuple[list[pathlib.Path], bool]:
    """Select the model files from a model directory."""
    files = [file for file in model_dir.iterdir() if file.is_file()]

    for pattern, is_compressed in (
        (r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$", True),
        (r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$", True),
        (r"(?P<model_name>.*)\.[0-9]+\.bin$", False),
        (r"(?P<model_name>.*)\.bin\.[0-9]+$", False),
        (r"(?P<model_name>.*)\.bin\.zst$", True),
        (r"(?P<model_name>.*)\.bin$", False),
    ):
        models = _match_model(files, re.compile(pattern))
        if model_name is not None:
            if model_name in models:
                return models[model_name], is_compressed
        elif models:
            if len(models) > 1:
                msg = f"Multiple models matched in {model_dir}"
                raise RuntimeError(msg)
            _, model_files = models.popitem()
            return model_files, is_compressed
    msg = f"No models matched in {model_dir}"
    raise FileNotFoundError(msg)


def get_alphafold3_params(checkpoint_path: pathlib.Path) -> dict[str, torch.Tensor]:
    """Return alphafold3 params."""
    if not Path(checkpoint_path).exists():
        msg = f"Given checkpoint path not exist [{checkpoint_path}]"
        raise RuntimeError(msg)
    logger.info("%s", f"Loading from {checkpoint_path}")
    is_compressed = False
    if checkpoint_path.suffix == ".zst":
        is_compressed = True
    params = {}
    with open_for_reading(
        [pathlib.Path(checkpoint_path)], is_compressed=is_compressed
    ) as stream:
        for scope, name, arr in read_records(stream):
            params[f"{scope}/{name}"] = arr
    return params


def stacked(
    param_dict_list: list[ParamTree], out: ParamTree | None = None
) -> ParamTree:
    """Stack corresponding parameters while retaining the checkpoint key tree.

    Args:
        out: Optional destination parameter tree to populate.
        param_dict_list: Parameter trees with identical keys to stack.
        A list of (nested) Param dicts to stack. The structure of
        each dict must be the identical (down to the ParamTypes of
        "parallel" Params). There must be at least one dict
        in the list.

    """
    if out is None:
        out = {}
    template = param_dict_list[0]
    for k in template:
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = {}
            stacked(v, out=out[k])
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )

            out[k] = stacked_param

    return out


def _process_translations_dict(
    d: ParamTree, _key_prefix: str, *, top_layer: bool = True
) -> dict[str, Param]:
    flat = {}
    for k, v in d.items():
        if type(v) is dict:
            prefix = _key_prefix if top_layer else ""
            sub_flat = {
                (prefix + f"{k}/{k_prime}"): v_prime
                for k_prime, v_prime in _process_translations_dict(
                    d=v, _key_prefix=_key_prefix, top_layer=False
                ).items()
            }
            flat.update(sub_flat)
        else:
            flat[(_key_prefix if top_layer else "") + k] = v

    return flat


def _copy_parameter(
    target: torch.Tensor, weight: torch.Tensor, *, preserve_dtype: bool
) -> None:
    if preserve_dtype:
        target.data = weight.to(device=target.device).clone()
    else:
        target.copy_(weight)


def assign(
    translation_dict: dict[str, Param],
    param_to_load: dict[str, torch.Tensor],
    *,
    preserve_dtype: bool = False,
) -> None:
    """Copy validated checkpoint arrays into matching single or stacked parameters."""
    with torch.no_grad():
        for key, param in translation_dict.items():
            source = torch.as_tensor(param_to_load[key])
            reference = param.param
            if param.stacked:
                if not isinstance(reference, list):
                    message = f"{key}: a stacked parameter requires a Tensor list"
                    raise TypeError(message)
                targets = reference
                if len(targets) == source.shape[0]:
                    weights = list(torch.unbind(source, 0))
                elif (
                    source.ndim > 1
                    and len(targets) == source.shape[0] * source.shape[1]
                ):
                    weights = list(
                        torch.unbind(source.reshape(-1, *source.shape[2:]), 0)
                    )
                else:
                    message = f"{key}: stacked parameter count mismatch"
                    raise ValueError(message)
            else:
                if not isinstance(reference, torch.Tensor):
                    message = f"{key}: an unstacked parameter requires a Tensor"
                    raise TypeError(message)
                targets = [reference]
                weights = [source]
            transformed = [
                param.param_type.transformation(weight) for weight in weights
            ]
            if len(targets) != len(transformed):
                message = f"{key}: stacked parameter count mismatch"
                raise ValueError(message)
            for target, weight in zip(targets, transformed, strict=True):
                if target.shape != weight.shape:
                    message = f"{key}: {target.shape} != {weight.shape}"
                    raise ValueError(message)
                _copy_parameter(target, weight, preserve_dtype=preserve_dtype)


# With Param, a poor man's enum with attributes (Rust-style)
class ParamType(Enum):
    """Represent param type."""

    linear_weight = partial(  # partial preserves callable enum values
        lambda w: w.transpose(-1, -2)
    )
    linear_weight_mha = partial(
        lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2)
    )
    LinearWeightNoTransposeMHA = partial(lambda w: w.reshape(-1, w.shape[-1]))
    linear_bias_mha = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearFlat = partial(lambda w: w.unsqueeze(-1))
    Other = partial(lambda w: w)

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.transformation = fn


def cat_params(params: ParamTree, prefix: str) -> ParamTree:
    """Compute cat params."""
    return {f"{prefix}{k}": v for k, v in params.items()}


@dataclass
class Param:
    """Represent param."""

    param: torch.Tensor | list[torch.Tensor]
    param_type: ParamType = ParamType.Other
    stacked: bool = False


def build_linear_weight(
    l: torch.Tensor, *, already_transpose_weights: bool = False
) -> Param:
    """Compute linear weight."""
    if already_transpose_weights is True:
        return Param(l)
    return Param(l, param_type=ParamType.linear_weight)


def build_linear_weight_mha(
    l: torch.Tensor, *, already_transpose_weights: bool = False
) -> Param:
    """Compute linear weight m h a."""
    if already_transpose_weights is True:
        return Param(l, param_type=ParamType.LinearWeightNoTransposeMHA)
    return Param(l, param_type=ParamType.linear_weight_mha)


def build_linear_bias_mha(b: torch.Tensor) -> Param:
    """Compute linear bias m h a."""
    return Param(b, param_type=ParamType.linear_bias_mha)


def build_linear_params(
    l: ParameterModule,
    *,
    use_bias: bool = False,
    already_transpose_weights: bool = False,
) -> ParamTree:
    """Compute linear params."""
    d = {
        "weights": build_linear_weight(
            l=l.weight, already_transpose_weights=already_transpose_weights
        )
    }

    if use_bias:
        d["bias"] = Param(l.bias)

    return d


def build_flat_linear(l: ParameterModule, *, use_bias: bool = False) -> ParamTree:
    """Compute linearfrom flat params."""
    d = {"weights": Param(l.weight, param_type=ParamType.LinearFlat)}

    if use_bias:
        d["bias"] = Param(l.bias)

    return d


def build_linear_hma_params(
    l: ParameterModule,
    *,
    use_bias: bool = False,
    already_transpose_weights: bool = False,
) -> ParamTree:
    """Compute linear h m a params."""
    d = {
        "weights": build_linear_weight_mha(
            l=l.weight, already_transpose_weights=already_transpose_weights
        )
    }

    if use_bias:
        d["bias"] = build_linear_bias_mha(l.bias)
    return d


def build_layer_norm_params(l: ParameterModule, *, use_bias: bool = True) -> ParamTree:
    """Compute layer norm params."""
    d = {
        "scale": Param(l.weight),
    }
    if use_bias:
        d["offset"] = Param(l.bias)

    return d


def build_scale_norm_params(l: ParameterModule) -> ParamTree:
    """Map a norm that is scale-only in AF3 and affine in some families."""
    return build_layer_norm_params(l=l, use_bias=l.bias is not None)


def build_adaptive_layer_norm_params(
    aln: ParameterModule, *, use_single_cond: bool = False
) -> ParamTree:
    """Compute adaptive layer norm params."""
    if use_single_cond is False:
        return {
            "layer_norm": build_layer_norm_params(l=aln.layer_norm),
        }
    return {
        "single_cond_layer_norm": build_layer_norm_params(
            l=aln.single_cond_layer_norm, use_bias=False
        ),
        "single_cond_scale": build_linear_params(
            l=aln.single_cond_scale, use_bias=True
        ),
        "single_cond_bias": build_linear_params(l=aln.single_cond_bias),
    }


def build_ada_ln_zero_params(
    ada_ln_zero: ParameterModule, *, use_single_cond: bool = False
) -> ParamTree:
    """Compute ada l n zero params."""
    d = {
        "transition2": build_linear_params(l=ada_ln_zero.transition2),
    }

    if use_single_cond is True:
        d.update(
            {
                "adaptive_zero_cond": build_linear_params(
                    l=ada_ln_zero.adaptive_zero_cond, use_bias=True
                ),
            }
        )

    return d


def build_tri_mul_params(tri_mul: ParameterModule) -> ParamTree:
    """Compute tri mul params."""
    return {
        "left_norm_input": build_layer_norm_params(l=tri_mul.left_norm_input),
        "projection": build_linear_params(l=tri_mul.projection),
        "gate": build_linear_params(l=tri_mul.gate),
        "center_norm": build_layer_norm_params(l=tri_mul.center_norm),
        "output_projection": build_linear_params(l=tri_mul.output_projection),
        "gating_linear": build_linear_params(l=tri_mul.gating_linear),
    }


def build_outer_product_mean_params(outer_product_mean: ParameterModule) -> ParamTree:
    """Compute outer product mean params."""
    return {
        "layer_norm_input": build_layer_norm_params(
            l=outer_product_mean.layer_norm_input
        ),
        "left_projection": build_linear_params(
            l=outer_product_mean.left_projection,
            use_bias=outer_product_mean.left_projection.bias is not None,
        ),
        "right_projection": build_linear_params(
            l=outer_product_mean.right_projection,
            use_bias=outer_product_mean.right_projection.bias is not None,
        ),
        "output_w": Param(outer_product_mean.output_w),
        "output_b": Param(outer_product_mean.output_b),
    }


def build_transition_params(transition: ParameterModule) -> ParamTree:
    """Compute transition params."""
    return {
        "input_layer_norm": build_layer_norm_params(l=transition.input_layer_norm),
        "transition1": build_linear_params(l=transition.transition1),
        "transition2": build_linear_params(l=transition.transition2),
    }


def build_grid_self_attention_params(pair_attention: ParameterModule) -> ParamTree:
    """Compute grid self attention params."""
    return {
        "act_norm": build_layer_norm_params(l=pair_attention.act_norm),
        "pair_bias_projection": build_linear_params(
            l=pair_attention.pair_bias_projection
        ),
        "q_projection": build_linear_hma_params(
            l=pair_attention.q_projection, already_transpose_weights=True
        ),
        "k_projection": build_linear_hma_params(
            l=pair_attention.k_projection, already_transpose_weights=True
        ),
        "v_projection": build_linear_hma_params(l=pair_attention.v_projection),
        "gating_query": build_linear_params(
            l=pair_attention.gating_query,
            already_transpose_weights=True,
            use_bias=pair_attention.gating_query.bias is not None,
        ),
        "output_projection": build_linear_params(
            l=pair_attention.output_projection,
            use_bias=pair_attention.output_projection.bias is not None,
        ),
    }


def build_kq_norm_params(attention: ParameterModule) -> ParamTree:
    """Map the query and key norms of a diffusion attention that has them."""
    if not getattr(attention, "kq_norm", False):
        return {}
    return {
        "query_layer_norm": build_layer_norm_params(l=attention.query_layer_norm),
        "key_layer_norm": build_layer_norm_params(l=attention.key_layer_norm),
    }


def build_self_attention_params(
    self_attention: ParameterModule, *, use_single_cond: bool = False
) -> ParamTree:
    """Compute self attention params."""
    return {
        "q_projection": build_linear_hma_params(
            l=self_attention.q_projection, use_bias=True
        ),
        "k_projection": build_linear_hma_params(l=self_attention.k_projection),
        "v_projection": build_linear_hma_params(l=self_attention.v_projection),
        "gating_query": build_linear_params(l=self_attention.gating_query),
        **build_kq_norm_params(self_attention),
        "transition2": build_linear_params(
            l=self_attention.adaptive_zero_init.transition2
        ),
        **build_adaptive_layer_norm_params(
            aln=self_attention.adaptive_layernorm, use_single_cond=use_single_cond
        ),
        **build_ada_ln_zero_params(
            ada_ln_zero=self_attention.adaptive_zero_init,
            use_single_cond=use_single_cond,
        ),
    }


def build_cross_attention_params(cross_attention: ParameterModule) -> ParamTree:
    """Compute cross attention params."""
    return {
        **cat_params(
            build_adaptive_layer_norm_params(
                aln=cross_attention.q_adaptive_layernorm, use_single_cond=True
            ),
            "q",
        ),
        **cat_params(
            build_adaptive_layer_norm_params(
                aln=cross_attention.k_adaptive_layernorm, use_single_cond=True
            ),
            "k",
        ),
        "q_projection": build_linear_hma_params(
            l=cross_attention.q_projection, use_bias=True
        ),
        "k_projection": build_linear_hma_params(l=cross_attention.k_projection),
        "v_projection": build_linear_hma_params(l=cross_attention.v_projection),
        "gating_query": build_linear_params(l=cross_attention.gating_query),
        **build_kq_norm_params(cross_attention),
        **build_ada_ln_zero_params(
            ada_ln_zero=cross_attention.adaptive_zero_init, use_single_cond=True
        ),
    }


def build_msa_attention_params(msa_attention: ParameterModule) -> ParamTree:
    """Compute m s a attention params."""
    return {
        "act_norm": build_layer_norm_params(l=msa_attention.act_norm),
        "pair_norm": build_layer_norm_params(l=msa_attention.pair_norm),
        "pair_logits": build_linear_params(l=msa_attention.pair_logits),
        "v_projection": build_linear_hma_params(l=msa_attention.v_projection),
        "gating_query": build_linear_params(l=msa_attention.gating_query),
        "output_projection": build_linear_params(l=msa_attention.output_projection),
    }


def build_diffusion_transition_params(
    transition: ParameterModule, *, use_single_cond: bool = False
) -> ParamTree:
    """Compute diffusion transition params."""
    return {
        **build_adaptive_layer_norm_params(
            aln=transition.adaptive_layernorm, use_single_cond=use_single_cond
        ),
        "transition1": build_linear_params(l=transition.transition1),
        **(
            {"a_to_b": build_linear_params(l=transition.a_to_b)}
            if hasattr(transition, "a_to_b")
            else {}
        ),
        **build_ada_ln_zero_params(
            ada_ln_zero=transition.adaptive_zero_init, use_single_cond=use_single_cond
        ),
    }


def build_diffusion_transformer_params(transformer: ParameterModule) -> ParamTree:
    """Compute diffusion transformer params."""
    self_attention_params = stacked(
        [
            build_self_attention_params(self_attention=l, use_single_cond=True)
            for l in transformer.self_attention
        ]
    )
    transistion_params = stacked(
        [
            build_diffusion_transition_params(transition=l, use_single_cond=True)
            for l in transformer.transition_block
        ]
    )

    if getattr(transformer, "per_block_pair", False):
        stack = "__layer_stack_no_per_layer/__layer_stack_no_per_layer/"
        return {
            stack + "pair_input_layer_norm": stacked(
                [
                    build_layer_norm_params(l=l, use_bias=l.bias is not None)
                    for l in transformer.pair_input_layer_norm
                ]
            ),
            stack + "pair_logits_projection": stacked(
                [build_linear_params(l=l) for l in transformer.pair_logits_projection]
            ),
            **cat_params(self_attention_params, stack + "transformer"),
            **cat_params(transistion_params, stack + "transformerffw_"),
        }
    return {
        "pair_input_layer_norm": build_layer_norm_params(
            l=transformer.pair_input_layer_norm, use_bias=False
        ),
        "__layer_stack_with_per_layer/pair_logits_projection": stacked(
            [build_linear_hma_params(l=l) for l in transformer.pair_logits_projection]
        ),
        **cat_params(
            self_attention_params,
            "__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer",
        ),
        **cat_params(
            transistion_params,
            "__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_",
        ),
    }


def build_diffusion_cross_att_transformer_params(
    transformer: ParameterModule, prefix: str = "diffusion_atom_transformer_encoder"
) -> ParamTree:
    """Compute diffusion cross att transformer params."""
    cross_attention_params = stacked(
        [build_cross_attention_params(l) for l in transformer.cross_attention]
    )
    transistion_params = stacked(
        [
            build_diffusion_transition_params(transition=l, use_single_cond=True)
            for l in transformer.transition_block
        ]
    )

    if getattr(transformer, "per_block_pair", False):
        stack = "__layer_stack_no_per_layer/"
        return {
            stack + "pair_input_layer_norm": stacked(
                [
                    build_layer_norm_params(l=l, use_bias=l.bias is not None)
                    for l in transformer.pair_input_layer_norm
                ]
            ),
            stack + "pair_logits_projection": stacked(
                [build_linear_params(l=l) for l in transformer.pair_logits_projection]
            ),
            **cat_params(cross_attention_params, stack + prefix),
            **cat_params(transistion_params, f"{stack}{prefix}ffw_"),
        }
    return {
        "pair_input_layer_norm": build_layer_norm_params(
            l=transformer.pair_input_layer_norm, use_bias=False
        ),
        "pair_logits_projection": build_linear_hma_params(
            l=transformer.pair_logits_projection
        ),
        **cat_params(cross_attention_params, f"__layer_stack_with_per_layer/{prefix}"),
        **cat_params(transistion_params, f"__layer_stack_with_per_layer/{prefix}ffw_"),
    }


def build_atom_cross_att_encoder_params(
    encoder: ParameterModule,
    *,
    with_token_atoms_act: bool = False,
    with_trunk_single_cond: bool = False,
    with_trunk_pair_cond: bool = False,
    prefix: str = "evoformer_conditioning_atom_transformer_encoder",
) -> ParamTree:
    """Compute atom cross att encoder params."""
    d = {
        "embed_ref_pos": build_linear_params(l=encoder.embed_ref_pos),
        "embed_ref_mask": build_linear_params(l=encoder.embed_ref_mask),
        "embed_ref_element": build_linear_params(l=encoder.embed_ref_element),
        "embed_ref_charge": build_linear_params(l=encoder.embed_ref_charge),
        "embed_ref_atom_name": build_linear_params(l=encoder.embed_ref_atom_name),
        "single_to_pair_cond_row": build_linear_params(
            l=encoder.single_to_pair_cond_row
        ),
        "single_to_pair_cond_col": build_linear_params(
            l=encoder.single_to_pair_cond_col
        ),
        "embed_pair_offsets": build_linear_params(l=encoder.embed_pair_offsets),
        "embed_pair_distances": build_linear_params(l=encoder.embed_pair_distances),
        "single_to_pair_cond_row_1": build_linear_params(
            l=encoder.single_to_pair_cond_row_1
        ),
        "single_to_pair_cond_col_1": build_linear_params(
            l=encoder.single_to_pair_cond_col_1
        ),
        "embed_pair_offsets_1": build_linear_params(l=encoder.embed_pair_offsets_1),
        "embed_pair_distances_1": build_linear_params(l=encoder.embed_pair_distances_1),
        "embed_pair_offsets_valid": build_linear_params(
            l=encoder.embed_pair_offsets_valid
        ),
        "pair_mlp_1": build_linear_params(l=encoder.pair_mlp_1),
        "pair_mlp_2": build_linear_params(l=encoder.pair_mlp_2),
        "pair_mlp_3": build_linear_params(l=encoder.pair_mlp_3),
        "atom_transformer_encoder": build_diffusion_cross_att_transformer_params(
            encoder.atom_transformer_encoder, prefix=prefix
        ),
        "project_atom_features_for_aggr": build_linear_params(
            l=encoder.project_atom_features_for_aggr
        ),
    }

    if hasattr(encoder, "embed_atom_features_bias"):
        d["embed_atom_features_bias"] = Param(encoder.embed_atom_features_bias)
    if hasattr(encoder, "conformer_embedding_bias"):
        d["conformer_embedding_bias"] = Param(encoder.conformer_embedding_bias)
    if hasattr(encoder, "atom_chiral_to_features"):
        d["atom_chiral_to_features"] = build_linear_params(
            l=encoder.atom_chiral_to_features
        )

    if with_token_atoms_act is True:
        d.update(
            {
                "atom_positions_to_features": build_linear_params(
                    l=encoder.atom_positions_to_features
                ),
            }
        )

    if with_trunk_single_cond is True:
        d.update(
            {
                "lnorm_trunk_single_cond": build_scale_norm_params(
                    encoder.lnorm_trunk_single_cond
                ),
                "embed_trunk_single_cond": build_linear_params(
                    l=encoder.embed_trunk_single_cond
                ),
            }
        )

    if with_trunk_pair_cond:
        d.update(
            {
                "lnorm_trunk_pair_cond": build_scale_norm_params(
                    encoder.lnorm_trunk_pair_cond
                ),
                "embed_trunk_pair_cond": build_linear_params(
                    l=encoder.embed_trunk_pair_cond
                ),
            }
        )

    return d


def build_atom_cross_att_decoder_params(decoder: ParameterModule) -> ParamTree:
    """Compute atom cross att decoder params."""
    return {
        "project_token_features_for_broadcast": build_linear_params(
            l=decoder.project_token_features_for_broadcast
        ),
        "atom_transformer_decoder": build_diffusion_cross_att_transformer_params(
            decoder.atom_transformer_decoder,
            prefix="diffusion_atom_transformer_decoder",
        ),
        "atom_features_layer_norm": build_scale_norm_params(
            decoder.atom_features_layer_norm
        ),
        "atom_features_to_position_update": build_linear_params(
            l=decoder.atom_features_to_position_update
        ),
    }


def build_template_embedding_params(template_embedding: ParameterModule) -> ParamTree:
    """Compute template embedding params."""
    pairformer_params = stacked(
        [
            build_pairformer_block_params(b=b, with_single=False)
            for b in (
                template_embedding.single_template_embedding.template_embedding_iteration
            )
        ]
    )

    return {
        "single_template_embedding/query_embedding_norm": build_layer_norm_params(
            l=template_embedding.single_template_embedding.query_embedding_norm
        ),
        "single_template_embedding/template_pair_embedding_0": build_linear_params(
            l=template_embedding.single_template_embedding.template_pair_embedding_0
        ),
        "single_template_embedding/template_pair_embedding_1": build_flat_linear(
            l=template_embedding.single_template_embedding.template_pair_embedding_1
        ),
        "single_template_embedding/template_pair_embedding_2": build_linear_params(
            l=template_embedding.single_template_embedding.template_pair_embedding_2
        ),
        "single_template_embedding/template_pair_embedding_3": build_linear_params(
            l=template_embedding.single_template_embedding.template_pair_embedding_3
        ),
        "single_template_embedding/template_pair_embedding_4": build_flat_linear(
            l=template_embedding.single_template_embedding.template_pair_embedding_4
        ),
        "single_template_embedding/template_pair_embedding_5": build_flat_linear(
            l=template_embedding.single_template_embedding.template_pair_embedding_5
        ),
        "single_template_embedding/template_pair_embedding_6": build_flat_linear(
            l=template_embedding.single_template_embedding.template_pair_embedding_6
        ),
        "single_template_embedding/template_pair_embedding_7": build_flat_linear(
            l=template_embedding.single_template_embedding.template_pair_embedding_7
        ),
        "single_template_embedding/template_pair_embedding_8": build_linear_params(
            l=template_embedding.single_template_embedding.template_pair_embedding_8
        ),
        **cat_params(
            pairformer_params,
            (
                "single_template_embedding/__layer_stack_no_per_layer/template_em"
                "bedding_iteration/"
            ),
        ),
        "single_template_embedding/output_layer_norm": build_layer_norm_params(
            l=template_embedding.single_template_embedding.output_layer_norm
        ),
        "output_linear": build_linear_params(l=template_embedding.output_linear),
    }


def build_contact_conditioning_params(module: ParameterModule) -> ParamTree:
    """Contact conditioning; the two class constants are bare parameters."""
    return {
        "contact_fourier": build_linear_params(l=module.contact_fourier, use_bias=True),
        "contact_encoder": build_linear_params(l=module.contact_encoder, use_bias=True),
    }


def build_fused_template_params(template: ParameterModule) -> ParamTree:
    """Fused template embedder shared by the Boltz-2, Protenix and RF3 weights."""
    tree = {
        "z_norm": build_layer_norm_params(l=template.z_norm),
        "z_proj": build_linear_params(l=template.z_proj),
        "a_proj": build_linear_params(l=template.a_proj),
        "v_norm": build_layer_norm_params(l=template.v_norm),
        "u_proj": build_linear_params(l=template.u_proj),
    }
    if len(template.tmpl_pairformer):
        tree.update(
            cat_params(
                stacked(
                    [
                        build_pairformer_block_params(b=b, with_single=False)
                        for b in template.tmpl_pairformer
                    ]
                ),
                "__layer_stack_no_per_layer/tmpl_pairformer/",
            )
        )
    return tree


def build_pairformer_block_params(
    b: ParameterModule, *, with_single: bool = False
) -> ParamTree:
    """Compute pairformer block params."""
    d: dict[str, Any] = {
        "triangle_multiplication_outgoing": build_tri_mul_params(
            b.triangle_multiplication_outgoing
        ),
        "triangle_multiplication_incoming": build_tri_mul_params(
            b.triangle_multiplication_incoming
        ),
        "pair_attention1": build_grid_self_attention_params(b.pair_attention1),
        "pair_attention2": build_grid_self_attention_params(b.pair_attention2),
        "pair_transition": build_transition_params(b.pair_transition),
    }

    if with_single is True:
        d.update(
            {
                "single_pair_logits_norm": build_layer_norm_params(
                    l=b.single_pair_logits_norm
                ),
                "single_pair_logits_projection": build_linear_params(
                    l=b.single_pair_logits_projection
                ),
                **cat_params(
                    build_self_attention_params(self_attention=b.single_attention_),
                    "single_attention_",
                ),
                "single_transition": build_transition_params(b.single_transition),
            }
        )

    return d


def build_evoformer_block_params(b: ParameterModule) -> ParamTree:
    """Compute evoformer block params."""
    return {
        "outer_product_mean": build_outer_product_mean_params(b.outer_product_mean),
        "msa_attention1": build_msa_attention_params(b.msa_attention1),
        "msa_transition": build_transition_params(b.msa_transition),
        "triangle_multiplication_outgoing": build_tri_mul_params(
            b.triangle_multiplication_outgoing
        ),
        "triangle_multiplication_incoming": build_tri_mul_params(
            b.triangle_multiplication_incoming
        ),
        "pair_attention1": build_grid_self_attention_params(b.pair_attention1),
        "pair_attention2": build_grid_self_attention_params(b.pair_attention2),
        "pair_transition": build_transition_params(b.pair_transition),
    }


def build_diffusion_head_params(head: ParameterModule) -> ParamTree:
    """Compute diffusion head params."""
    return {
        "pair_cond_initial_norm": build_scale_norm_params(head.pair_cond_initial_norm),
        "pair_cond_initial_projection": build_linear_params(
            l=head.pair_cond_initial_projection
        ),
        **cat_params(
            build_diffusion_transition_params(transition=head.pair_transition_0),
            "pair_transition_0ffw_",
        ),
        **cat_params(
            build_diffusion_transition_params(transition=head.pair_transition_1),
            "pair_transition_1ffw_",
        ),
        "single_cond_initial_norm": build_scale_norm_params(
            head.single_cond_initial_norm
        ),
        "single_cond_initial_projection": build_linear_params(
            l=head.single_cond_initial_projection,
            use_bias=head.single_cond_initial_projection.bias is not None,
        ),
        **(
            {"relpe_projection": build_linear_params(l=head.relpe_projection)}
            if head.relpe_projection is not None
            else {}
        ),
        "noise_embedding_initial_norm": build_scale_norm_params(
            head.noise_embedding_initial_norm
        ),
        "noise_embedding_initial_projection": build_linear_params(
            l=head.noise_embedding_initial_projection
        ),
        **cat_params(
            build_diffusion_transition_params(transition=head.single_transition_0),
            "single_transition_0ffw_",
        ),
        **cat_params(
            build_diffusion_transition_params(transition=head.single_transition_1),
            "single_transition_1ffw_",
        ),
        **cat_params(
            build_atom_cross_att_encoder_params(
                head.atom_cross_att_encoder,
                with_token_atoms_act=True,
                with_trunk_pair_cond=True,
                with_trunk_single_cond=True,
                prefix="diffusion_atom_transformer_encoder",
            ),
            "diffusion_",
        ),
        "single_cond_embedding_norm": build_scale_norm_params(
            head.single_cond_embedding_norm
        ),
        "single_cond_embedding_projection": build_linear_params(
            l=head.single_cond_embedding_projection
        ),
        "transformer": build_diffusion_transformer_params(head.transformer),
        "output_norm": build_scale_norm_params(head.output_norm),
        **cat_params(
            build_atom_cross_att_decoder_params(head.atom_cross_att_decoder),
            "diffusion_",
        ),
    }


def build_confidence_head_params(head: ParameterModule) -> ParamTree:
    """Compute confidence head params."""
    pairformer_blocks_params = stacked(
        [
            build_pairformer_block_params(b=b, with_single=True)
            for b in head.confidence_pairformer
        ]
    )
    resolved_params = (
        {
            "experimentally_resolved_logits": build_linear_hma_params(
                l=head.experimentally_resolved_logits
            )
        }
        if head.resolved_head
        else {}
    )

    if getattr(head, "split_heads", False):
        re = head.reembedding
        scope = "~_boltz2_reembed/"
        stack = "__layer_stack_no_per_layer/confidence_pairformer"
        linears = (
            "s_input_to_s",
            "rel_pos_project",
            "token_bonds_project",
            "token_bonds_type_embed",
            "left_target_feat_project",
            "right_target_feat_project",
            "s_to_z_prod_in1",
            "s_to_z_prod_in2",
            "s_to_z_prod_out",
            "distogram_feat_project",
        )
        return {
            **{
                scope + name: build_layer_norm_params(l=getattr(re, name))
                for name in ("s_inputs_norm", "s_norm", "z_norm")
            },
            **{
                scope + name: build_linear_params(l=getattr(re, name))
                for name in linears
            },
            **cat_params(
                build_contact_conditioning_params(re.contact_conditioning), scope
            ),
            "contact_encoding_unspecified": Param(
                re.contact_conditioning.encoding_unspecified
            ),
            "contact_encoding_unselected": Param(
                re.contact_conditioning.encoding_unselected
            ),
            stack: pairformer_blocks_params,
            "left_half_distance_logits": build_linear_params(
                l=head.left_half_distance_logits
            ),
            "inter_half_distance_logits": build_linear_params(
                l=head.inter_half_distance_logits
            ),
            "pae_logits": build_linear_params(l=head.pae_logits),
            "pae_inter_logits": build_linear_params(l=head.pae_inter_logits),
            "plddt_logits": build_linear_hma_params(l=head.plddt_logits),
            **resolved_params,
        }
    return {
        "~_embed_features/left_target_feat_project": build_linear_params(
            l=head.left_target_feat_project
        ),
        "~_embed_features/right_target_feat_project": build_linear_params(
            l=head.right_target_feat_project
        ),
        "~_embed_features/distogram_feat_project": build_linear_params(
            l=head.distogram_feat_project
        ),
        "__layer_stack_no_per_layer/confidence_pairformer": pairformer_blocks_params,
        "logits_ln": build_layer_norm_params(l=head.logits_ln),
        "left_half_distance_logits": build_linear_params(
            l=head.left_half_distance_logits
        ),
        "pae_logits_ln": build_layer_norm_params(l=head.pae_logits_ln),
        "pae_logits": build_linear_params(l=head.pae_logits),
        "plddt_logits_ln": build_layer_norm_params(l=head.plddt_logits_ln),
        "plddt_logits": build_linear_hma_params(l=head.plddt_logits),
        **(
            {
                "experimentally_resolved_ln": build_layer_norm_params(
                    l=head.experimentally_resolved_ln
                ),
            }
            if head.resolved_head
            else {}
        ),
        **resolved_params,
    }


def build_evoformer_params(evoformer: ParameterModule) -> ParamTree:
    """Compute evoformer params."""
    msa_stack_params = stacked(
        [build_evoformer_block_params(b) for b in evoformer.msa_stack]
    )

    trunk_pairformer_params = stacked(
        [
            build_pairformer_block_params(b=b, with_single=True)
            for b in evoformer.trunk_pairformer
        ]
    )

    return {
        "left_single": build_linear_params(l=evoformer.left_single),
        "right_single": build_linear_params(l=evoformer.right_single),
        "prev_embedding_layer_norm": build_layer_norm_params(
            l=evoformer.prev_embedding_layer_norm
        ),
        "prev_embedding": build_linear_params(l=evoformer.prev_embedding),
        "~_relative_encoding/position_activations": build_linear_params(
            l=evoformer.position_activations
        ),
        **(
            {"bond_embedding": build_linear_params(l=evoformer.bond_embedding)}
            if hasattr(evoformer, "bond_embedding")
            else {}
        ),
        "template_embedding": (
            build_fused_template_params(evoformer.template_embedding)
            if hasattr(evoformer.template_embedding, "tmpl_pairformer")
            else build_template_embedding_params(evoformer.template_embedding)
        ),
        **(
            {
                "token_bonds_type_embed": build_linear_params(
                    l=evoformer.token_bonds_type_embed
                ),
                **build_contact_conditioning_params(evoformer.contact_conditioning),
                "contact_encoding_unspecified": Param(
                    evoformer.contact_conditioning.encoding_unspecified
                ),
                "contact_encoding_unselected": Param(
                    evoformer.contact_conditioning.encoding_unselected
                ),
            }
            if hasattr(evoformer, "contact_conditioning")
            else {}
        ),
        "msa_activations": build_linear_params(
            l=evoformer.msa_activations,
            use_bias=evoformer.msa_activations.bias is not None,
        ),
        "extra_msa_target_feat": build_linear_params(l=evoformer.extra_msa_target_feat),
        **cat_params(msa_stack_params, "__layer_stack_no_per_layer/msa_stack/"),
        "single_activations": build_linear_params(l=evoformer.single_activations),
        "prev_single_embedding_layer_norm": build_layer_norm_params(
            l=evoformer.prev_single_embedding_layer_norm
        ),
        "prev_single_embedding": build_linear_params(l=evoformer.prev_single_embedding),
        **cat_params(
            trunk_pairformer_params, "__layer_stack_no_per_layer_1/trunk_pairformer/"
        ),
    }


def get_translation_dict(model: ParameterModule) -> ParamTree:
    """Return translation dict."""
    return {
        **cat_params(
            build_atom_cross_att_encoder_params(model.evoformer_conditioning),
            "evoformer_conditioning_",
        ),
        "evoformer": build_evoformer_params(model.evoformer),
        "~/diffusion_head": build_diffusion_head_params(model.diffusion_head),
        "distogram_head/half_logits": build_linear_params(
            l=model.distogram_head.half_logits,
            use_bias=model.distogram_head.half_logits.bias is not None,
        ),
        "confidence_head": build_confidence_head_params(model.confidence_head),
        **(
            {
                record: build_linear_params(l=l, use_bias=l.bias is not None)
                for record, l in model.input_embedder.records().items()
            }
            if hasattr(model, "input_embedder")
            else {}
        ),
    }


def import_jax_weights_(
    model: ParameterModule, model_path: pathlib.Path, *, preserve_dtype: bool = False
) -> dict[str, int]:
    """Compute import jax weights."""
    checkpoint = model_path / "af3.bin.zst" if model_path.is_dir() else model_path
    params = get_alphafold3_params(checkpoint)
    # AF3's noise Fourier features are fixed constants; ported families train
    # theirs and carry them in the blob.
    trained_fourier = {
        name: params.pop(f"diffuser/~/diffusion_head/fourier_embedding_{name}")
        for name in ("weight", "bias")
        if f"diffuser/~/diffusion_head/fourier_embedding_{name}" in params
    }
    if len(trained_fourier) == 1:
        msg = f"Checkpoint carries only one Fourier record: {sorted(trained_fourier)}"
        raise ValueError(msg)

    translations = get_translation_dict(model)

    flat = _process_translations_dict(d=translations, _key_prefix="diffuser/")

    missing = sorted(set(flat) - set(params))
    extra = sorted(name for name in set(params) - set(flat) if "__meta__/" not in name)
    if missing or extra:
        msg = f"AF3 checkpoint mapping mismatch: missing={missing}, extra={extra}"
        raise ValueError(msg)
    assign(flat, params, preserve_dtype=preserve_dtype)
    setattr(model, "__identifier__", params["__meta__/__identifier__"])  # noqa: B010 - dynamic checkpoint metadata

    fourier = model.diffusion_head.fourier_embeddings
    fourier.register_buffer(
        "weight",
        trained_fourier["weight"].to(torch.float32).reshape(-1)
        if trained_fourier
        else torch.tensor(_WEIGHT, dtype=torch.float32),
    )
    fourier.register_buffer(
        "bias",
        trained_fourier["bias"].to(torch.float32).reshape(-1)
        if trained_fourier
        else torch.tensor(_BIAS, dtype=torch.float32),
    )
    return {
        "source_records": len(params) + len(trained_fourier),
        "mapped_records": len(flat),
        "trained_fourier": bool(trained_fourier),
    }
