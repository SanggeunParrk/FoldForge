"""Which engine ops does a fold actually enter, and did the residual ride along?

``implementation=MINIWORLD_ENGINE`` is a request, not a guarantee. An op can fall
back to a slower path for the running shape or arch, and nothing says so — the
only symptom is a wall-clock number that never improves. This wraps every callable
on ``miniworld_engine.ops`` with a counter and a synchronised timer, folds once,
and reports call counts and device time per op.

It also reports whether each call was given a ``residual``. The engine fuses the
residual into the op's epilogue, so a fold where the residual never reaches the op
is paying for a separate elementwise pass over (B,L,L,d_pair) — worth ~2/3 of that
add — while looking like it is using the fused path.

Usage (GPU node)::

    sbatch scripts/op_dispatch_audit.sbatch --target 4yx2
"""

import argparse
import collections
import functools
import json
import logging
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from team_gm.modules.exceptions import ImplementationType

from foldforge.data.ccd import CCDDatabase, default_path
from foldforge.models.architectures.esmfold2 import ESMFold2Model as Model
from foldforge.models.checkpoints import esmfold2 as convert
from foldforge.models.config.esmfold2 import ESMFold2Config
from foldforge.modules.sequence.features import (
    ESMFold2InputBuilder,
    build_input,
    model_kwargs,
)

logger = logging.getLogger("op_dispatch_audit")


def instrument() -> tuple[
    collections.Counter, collections.Counter, collections.Counter
]:
    """Wrap every engine kernel with a call counter, a timer and a residual counter.

    ``miniworld_engine.ops`` is the WRONG surface: it is the whole-op API for
    external callers, and team-gm's blocks now consume ``miniworld_engine.modules``
    instead, whose nn.Modules call ``kernels.*`` directly. Instrumenting ``ops``
    reports zero calls on a fold that is demonstrably running fused kernels — the
    instrument being in the wrong place, not the kernels being absent.
    """
    from miniworld_engine import kernels as ops

    calls: collections.Counter = collections.Counter()
    seconds: collections.Counter = collections.Counter()
    with_residual: collections.Counter = collections.Counter()

    for name in getattr(ops, "__all__", ()) or dir(ops):
        if name.startswith("_"):
            continue
        original = getattr(ops, name, None)
        if not callable(original):
            continue

        @functools.wraps(original)
        def wrapper(  # noqa: ANN202
            *args: object,
            _original: object = original,
            _name: str = name,
            **kwargs: object,
        ):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                seconds[_name] += time.perf_counter() - start
                calls[_name] += 1
                if kwargs.get("residual") is not None:
                    with_residual[_name] += 1

        setattr(ops, name, wrapper)
    return calls, seconds, with_residual


def census(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Which backend each engine module actually resolved to.

    This is the direct answer to "is the wiring right": ``implementation`` is a
    request, and every engine module stores the concrete backend it resolved for
    the running shape and arch in ``_backend``. A module that quietly fell back to
    PYTORCH shows up here as a count, where wall clock would only show a number
    that never improves.
    """
    counts: dict[str, dict[str, int]] = {}
    for module in model.modules():
        backend = getattr(module, "_backend", None)
        if backend is None:
            continue
        name = type(module).__name__
        key = getattr(backend, "name", str(backend))
        counts.setdefault(name, {}).setdefault(key, 0)
        counts[name][key] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccd-db", type=Path, default=default_path())
    parser.add_argument("--target", default="4yx2")
    parser.add_argument("--data-root", default="validation/inputs/data")
    parser.add_argument("--out", default="benchmark/esmfold2/op_dispatch")
    parser.add_argument("--checkpoint", default="model_checkpoints/esmfold2")
    parser.add_argument("--msa-depth", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    checkpoint = Path(args.checkpoint)
    sample = Path(args.data_root) / args.target
    builder = ESMFold2InputBuilder(ccd_db=CCDDatabase(args.ccd_db))
    features, _ = builder.prepare_input(
        build_input(sample, args.msa_depth), seed=0, device=device
    )
    lm_hidden = torch.load(
        sample / "cache" / "esmc_hidden_states.pt", map_location=device
    ).to(dtype)

    config = ESMFold2Config.from_json(checkpoint)
    model = (
        Model(config, ImplementationType.MINIWORLD_ENGINE)
        .to(device=device, dtype=dtype)
        .eval()
    )
    model.load_state_dict(
        convert.convert_model(load_file(checkpoint / "model.safetensors"), config)
    )
    kwargs = model_kwargs(features, dtype)

    def fold() -> object:
        return model(
            **kwargs,
            lm_hidden_states=lm_hidden,
            num_diffusion_samples=1,
            generator=torch.Generator(device=device).manual_seed(0),
        )

    # Warm first: a cold fold measures Triton autotuning, and the tuner's own
    # launches would be charged to the ops it is tuning.
    with torch.no_grad():
        fold()
    backends = census(model)
    calls, seconds, with_residual = instrument()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        fold()
    torch.cuda.synchronize()
    wall = time.perf_counter() - start

    accounted = sum(seconds.values())
    report = {
        "target": args.target,
        "n_tokens": int(features["token_attention_mask"].shape[1]),
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "fold_seconds": round(wall, 4),
        # Sum of the timed op regions. The gap to fold_seconds is everything that
        # is NOT an engine op: python, the atom track, the solver, the heads.
        "accounted_seconds": round(accounted, 4),
        "resolved_backends": backends,
        "ops": {
            name: {
                "calls": calls[name],
                "calls_with_residual": with_residual[name],
                "seconds": round(seconds[name], 4),
                "share_of_fold": round(seconds[name] / wall, 4) if wall else None,
                "ms_per_call": round(1e3 * seconds[name] / calls[name], 4),
            }
            for name in sorted(calls, key=lambda n: -seconds[n])
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.target}_{args.dtype}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info("%s", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
