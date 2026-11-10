"""The miniworld checkpoint format: an fp32 ``state_dict`` whose keys are the FoldForge graph's own (engine-class) keys.

Every model's graph is built from miniworld-engine classes, so its parameters have the engine's names and layouts. Its checkpoint is that ``state_dict`` plus a format tag, saved with ``torch.save`` and read with ``weights_only=True``::

    {"format": "miniworld", "version": 1, "family": "alphafold3", "source": "af3.bin.zst", "state_dict": {name: fp32 tensor}}

* **Loading** (:func:`load_into`) is ``model.load_state_dict(state_dict, strict=True)``: no translation tree.
* **Converting** (:func:`convert_haiku`, offline, once per released blob) is the only place the released layout is dealt with: the Haiku blob
  is read into the DENSE released-layout graph (:mod:`foldforge.models.checkpoints.legacy`, ``haiku.import_jax_weights_``), and every module the engine graph holds as an engine class is
  converted with the rules of :mod:`foldforge.models.checkpoints.legacy.released` (split / broadcast / discard). The conversion refuses to write a file
  unless every released key was either read, copied or a known discard (``released.DISCARDED``) and the result has exactly the keys and
  shapes of the engine graph.

The Fourier noise features (``diffusion_head.fourier_embeddings.{weight,bias}``, fixed constants or trained, depending on the family) are
buffers registered at load time; they are in the file and set by :func:`load_into`.

    python -m foldforge.models.checkpoints.miniworld convert af3 [--blob PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

FORMAT = "miniworld"
VERSION = 1
SUFFIX = ".miniworld.pt"

#: Registered model name -> the converted file in its checkpoint directory (the Haiku blob it is converted from is ``DEFAULT_FILES``).
NATIVE_FILES = {
    "af3": "af3.miniworld.pt",
    "intellifold2": "intellifold2.miniworld.pt",
    "openfold3": "openbind0.miniworld.pt",
    "openfold3-preview2": "openfold3.miniworld.pt",
    "boltz2": "boltz2.miniworld.pt",
    "rosettafold3": "rosettafold3.miniworld.pt",
    "chai1": "chai1.miniworld.pt",
    "esmfold2": "esmfold2.miniworld.pt",
    "esmfold2-fast": "esmfold2_fast.miniworld.pt",
    "protenix1": "protenix1.miniworld.pt",
    "protenix2": "protenix2.miniworld.pt",
    "opendde": "opendde.miniworld.pt",
}

#: Buffers the graph registers at load time (not in a freshly built model's ``state_dict``).
LOOSE = ("diffusion_head.fourier_embeddings.weight", "diffusion_head.fourier_embeddings.bias")


#: Released DiT-block tensors an engine block class may lack, droppable only when all zero (see ``transformers._DROPPABLE_ZERO``).
DROPPABLE_ZERO = ("attention.to_query.bias",)


class ConversionError(RuntimeError):
    """A released checkpoint that does not convert exactly (an unread key that is not a known discard, a missing / extra / mis-shaped key)."""


def is_native(path: Path) -> bool:
    """Whether ``path`` names a miniworld-format file (by suffix)."""
    return path.name.endswith(SUFFIX)


# ------------------------------------------------------------------------------------------------------------------------ loading
def read(path: Path) -> dict[str, Any]:
    """The checked contents of a miniworld-format file (``weights_only``; format tag and version verified)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        message = f"{path} is not a miniworld-format checkpoint (format tag {blob.get('format') if isinstance(blob, dict) else None!r})"
        raise ValueError(message)
    if blob.get("version") != VERSION:
        message = f"{path} is miniworld format version {blob.get('version')}; this code reads version {VERSION}. Convert the release again."
        raise ValueError(message)
    return blob


def load_into(model: nn.Module, path: Path, *, family: str) -> dict[str, Any]:
    """Strict load of a miniworld-format file into ``model`` (fp32 parameters; the Fourier buffers are registered). Returns a report."""
    blob = read(path)
    if blob["family"] != family:
        message = f"{path} holds family {blob['family']!r}, the model is {family!r}"
        raise ValueError(message)
    state = dict(blob["state_dict"])
    fourier = model.get_submodule("diffusion_head.fourier_embeddings")
    for name in LOOSE:  # registered first (their shapes are the file's), then loaded like every other entry
        fourier.register_buffer(name.rsplit(".", 1)[1], torch.zeros_like(state[name], dtype=torch.float32).reshape(-1))
        state[name] = state[name].to(torch.float32).reshape(-1)
    model.load_state_dict(state, strict=True)
    return {"format": FORMAT, "format_version": VERSION, "source": blob.get("source"), "mapped_records": len(state)}


# ---------------------------------------------------------------------------------------------------------------------- converting
def _converters(engine_model: nn.Module) -> list[tuple[str, Any, nn.Module]]:
    """``(prefix, function(tracked_state, prefix) -> {relative engine key: tensor}, module)`` of every module of ``engine_model`` that
    is an engine class (outermost only: parts of a converted module are converted with it)."""
    from miniworld_engine.modules import PairformerBlock, Transition

    from foldforge.models.checkpoints.legacy import released
    from foldforge.modules.engine.graph import EvoformerBlock
    from foldforge.modules.engine.transformers import AtomDiTStack, TokenDiTStack

    def blocks(convert: Any) -> Any:
        def run(state: Mapping[str, torch.Tensor], prefix: str, module: nn.Module) -> dict[str, torch.Tensor]:
            first = module.blocks[0]
            if isinstance(module, TokenDiTStack):
                listed = convert(state, n_head=first.attention.n_head, prefix=prefix)
            else:
                listed = convert(state, n_head=first.attention.n_head, kv_norm=first.kv_norm, prefix=prefix)
            if len(listed) != len(module.blocks):
                message = f"{prefix}: the released state has {len(listed)} blocks, the graph {len(module.blocks)}"
                raise ConversionError(message)
            result: dict[str, torch.Tensor] = {}
            for i, (block, tensors) in enumerate(zip(module.blocks, listed, strict=True)):
                own = block.state_dict().keys()
                for k, v in tensors.items():
                    if k in own:
                        result[f"blocks.{i}.{k}"] = v
                    elif k in DROPPABLE_ZERO and not bool(v.any()):  # RoseTTAFold3's bias-free query, released as an all-zero bias
                        continue
                    else:
                        message = f"{prefix}.blocks.{i}: released tensor {k} has no parameter in {type(block).__name__}" + (
                            " and is not all zero" if k in DROPPABLE_ZERO else ""
                        )
                        raise ConversionError(message)
            return result

        return run

    token = blocks(released.token_dit_blocks_from_released)
    atom = blocks(released.atom_dit_blocks_from_released)
    found: list[tuple[str, Any, nn.Module]] = []
    taken: list[str] = []
    for name, module in engine_model.named_modules():
        if not name or any(name.startswith(f"{p}.") for p in taken):
            continue
        if isinstance(module, PairformerBlock):
            fn = lambda s, p, _m: released.pairformer_block_from_released(s, prefix=p)  # noqa: E731
        elif isinstance(module, EvoformerBlock):
            fn = lambda s, p, _m: released.evoformer_block_from_released(s, prefix=p)  # noqa: E731
        elif isinstance(module, Transition) and re.fullmatch(r"diffusion_head\.(pair|single)_transition_\d", name):
            fn = lambda s, p, _m: released.diffusion_transition_from_released(s, prefix=p)  # noqa: E731
        elif isinstance(module, TokenDiTStack):
            fn = token
        elif isinstance(module, AtomDiTStack):
            fn = atom
        else:
            continue
        taken.append(name)
        found.append((name, fn, module))
    return found


def convert_state(released_state: Mapping[str, torch.Tensor], engine_model: nn.Module) -> dict[str, torch.Tensor]:
    """The engine graph's ``state_dict`` (fp32) from the ``state_dict`` of the DENSE released-layout graph of the same family.

    Raises :class:`ConversionError` unless: every released key under a converted module was read by its conversion or is a known discard
    (:func:`foldforge.models.checkpoints.legacy.released.is_discarded`), every other released key is a key of the engine graph, and the result has the
    engine graph's keys (the Fourier buffers aside) with their shapes."""
    from foldforge.models.checkpoints.legacy import released

    tracked = released.Tracked(released_state)
    out: dict[str, torch.Tensor] = {}
    prefixes: list[str] = []
    for prefix, convert, module in _converters(engine_model):
        for key, value in convert(tracked, prefix, module).items():
            out[f"{prefix}.{key}"] = value
        prefixes.append(prefix)

    unread = sorted(
        k for k in released_state if any(k.startswith(f"{p}.") for p in prefixes) and k not in tracked.used and not released.is_discarded(k)
    )
    if unread:
        message = f"released keys neither read by a conversion nor a known discard: {unread[:8]}{' ...' if len(unread) > 8 else ''}"
        raise ConversionError(message)
    target = engine_model.state_dict()
    for key, value in released_state.items():
        if any(key.startswith(f"{p}.") for p in prefixes):
            continue
        if key not in target and key not in LOOSE:
            message = f"released key {key!r} is not a parameter of the engine graph and no conversion covers it"
            raise ConversionError(message)
        out[key] = value

    missing = sorted(set(target) - set(out))
    extra = sorted(set(out) - set(target) - set(LOOSE))
    wrong = sorted(k for k in set(out) & set(target) if tuple(out[k].shape) != tuple(target[k].shape))
    if missing or extra or wrong:
        message = f"conversion does not match the engine graph: missing={missing[:6]}, extra={extra[:6]}, mis-shaped={wrong[:6]}"
        raise ConversionError(message)
    for key in LOOSE:
        if key not in out:
            message = f"the released graph has no {key}"
            raise ConversionError(message)
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in out.items()}


def convert_haiku(blob: Path, family: str, out: Path, *, mode: str = "fast") -> dict[str, Any]:
    """Convert a released Haiku blob (``af3.bin.zst`` layout) of ``family`` to a miniworld-format file at ``out``; returns a report."""
    from foldforge.models.architectures.af3 import AlphaFold3
    from foldforge.models.checkpoints.legacy import dense_model
    from foldforge.models.checkpoints.legacy.haiku import import_jax_weights_
    from foldforge.modules.dense.spec import spec_for
    from foldforge.modules.engine.install import skip_engine_init

    kwargs = {"num_recycles": 1, "num_samples": 1, "diffusion_steps": 1}
    spec = spec_for(family, mode)
    dense = dense_model(spec, **kwargs)
    mapped = import_jax_weights_(dense, blob)
    with skip_engine_init():
        engine = AlphaFold3(spec=spec, **kwargs)
    state = convert_state(dense.state_dict(), engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    torch.save({"format": FORMAT, "version": VERSION, "family": family, "source": blob.name, "state_dict": state}, tmp)
    tmp.replace(out)
    return {"out": str(out), "records": len(state), "source_records": mapped["source_records"], "format_version": VERSION}


def convert_model(model: str, *, blob: Path | None = None, out: Path | None = None, mode: str = "fast") -> dict[str, Any]:
    """Convert registered ``model``'s released blob (default: its checkpoint directory) to its miniworld-format file (default: next to the blob)."""
    from foldforge.models import entry
    from foldforge.models.checkpoints import DEFAULT_FILES, resolve

    blob = blob or resolve(model, DEFAULT_FILES[model])
    out = out or blob.resolve().parent / NATIVE_FILES[model]
    return convert_haiku(blob, entry(model).family or "alphafold3", out, mode=mode)


def resolve_native(model: str) -> Path:
    """The miniworld-format file of registered ``model`` in its checkpoint directory.

    Never converts: a missing file raises with the exact command that makes it (the conversion is an offline, one-time step)."""
    from foldforge.models.checkpoints import DEFAULT_FILES, resolve

    if model not in NATIVE_FILES:
        message = f"no miniworld-format checkpoint is defined for {model!r}; known: {', '.join(sorted(NATIVE_FILES))}"
        raise ValueError(message)
    try:
        return resolve(model, NATIVE_FILES[model])
    except FileNotFoundError as error:
        try:
            resolve(model, DEFAULT_FILES[model])
        except FileNotFoundError:
            step = f"first fetch the released blob: foldforge weights download {model}; then convert it: foldforge weights convert {model}"
        else:
            step = f"the released blob is present; convert it once: foldforge weights convert {model}"
        message = (
            f"no miniworld-format checkpoint ({NATIVE_FILES[model]}) for {model!r}: {step} "
            f"(same as: python -m foldforge.models.checkpoints.miniworld convert {model}). The loader never converts at run time.\n{error}"
        )
        raise FileNotFoundError(message) from error


def main(argv: list[str] | None = None) -> int:
    """``convert <model> [--blob PATH] [--out PATH]``: the released blob of a registered model to its miniworld-format file."""
    parser = argparse.ArgumentParser(prog="foldforge.models.checkpoints.miniworld", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Haiku blob -> miniworld format")
    convert.add_argument("model", choices=sorted(NATIVE_FILES))
    convert.add_argument("--blob", type=Path, default=None, help="the released blob (default: the model's checkpoint directory)")
    convert.add_argument("--out", type=Path, default=None, help="the file to write (default: next to the blob)")
    convert.add_argument("--mode", default="fast", choices=("fast", "exact"))
    arguments = parser.parse_args(argv)
    report = convert_model(arguments.model, blob=arguments.blob, out=arguments.out, mode=arguments.mode)
    print(report)  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
