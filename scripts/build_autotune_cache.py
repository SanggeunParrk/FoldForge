"""Build the engine's Triton autotune cache for THIS GPU from real folds.

The engine ships caches keyed by `(op, dtype, shape-bucket)`. On an A6000 most
trunk ops have no entry at all (`layernorm_main_fwd`, `trimul_gate_elem_mul`,
`trimul_back*` carry H100 only) and the ones that do are tuned for L 314-384 —
so every validation target misses, the autotuner falls back to the full grid per
process, and the chosen config "may be suboptimal" for 78% of a fold.

The capture hook patches `Autotuner._bench`, so it records whatever fires during a
real forward pass. That makes the WORKLOAD the sweep, which for an inference-only
cache beats a synthetic one:

* the shapes are exactly the ones we run, not a guess at which L matters
* ops the synthetic sweep never touches (atom track, LM encoder, confidence head)
  are covered for free
* inference-only by construction — the model's forward is `@torch.no_grad`

One process, every target, one flush: the per-target buckets accumulate in the
same `_CAPTURE` dict, and flushing once avoids two runs racing on one JSON.

Known limits, worth reading before trusting the result:

* **Bucket coarseness is not fixed by this.** `get_seq_group` puts every L >= 385
  in one bucket, so a config tuned at 594 also serves L=2048. Fine while our
  workload sits in that range; a much larger target needs a re-capture.
* The cache records `built_utc/torch/triton` but NOT the L it was measured at, so
  a bucket-vs-actual-shape mismatch is invisible after the fact.
* Capture runs the FULL grid per kernel, so a fold takes far longer than normal.

Usage (GPU node, hours)::

    sbatch scripts/build_autotune_cache.sbatch --targets 1ubq,1a1k,3ptb,4yx2
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from team_gm.modules.exceptions import ImplementationType

from foldforge.models.esmfold2 import ESMFold2Config, convert
from foldforge.models.esmfold2 import ESMFold2Model as Model
from foldforge.models.esmfold2.features import (
    ESMFold2InputBuilder,
    build_input,
    model_kwargs,
)

logger = logging.getLogger("build_autotune_cache")


def fold_once(target: str, args: argparse.Namespace, model_cache: dict) -> float:
    """One forward pass for ``target``; returns wall seconds."""
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    checkpoint = Path(args.checkpoint)
    sample = Path(args.data_root) / target

    builder = ESMFold2InputBuilder(ccd_cache=checkpoint)
    features, _ = builder.prepare_input(
        build_input(sample, args.msa_depth), seed=0, device=device
    )
    lm_hidden = torch.load(
        sample / "cache" / "esmc_hidden_states.pt", map_location=device
    ).to(dtype)

    # One model reused across targets: rebuilding it per target would re-pay the
    # weight load without changing a single autotune key.
    if "model" not in model_cache:
        config = ESMFold2Config.from_json(checkpoint)
        model = (
            Model(config, ImplementationType.MINIWORLD_ENGINE)
            .to(device=device, dtype=dtype)
            .eval()
        )
        model.load_state_dict(
            convert.convert_model(load_file(checkpoint / "model.safetensors"), config)
        )
        model_cache["model"] = model
    model = model_cache["model"]

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(
            **model_kwargs(features, dtype),
            lm_hidden_states=lm_hidden,
            num_diffusion_samples=1,
            generator=torch.Generator(device=device).manual_seed(0),
        )
    torch.cuda.synchronize()
    return time.perf_counter() - start


def synthetic_once(length: int, args: argparse.Namespace) -> float:
    """Tune the trunk block at ``length`` without a real target of that size.

    The row buckets above our largest validation target (594) have no entry, and a
    protein of that size is not something we have on disk. One `PairUpdateBlock` is
    the two triangle multiplications plus the transition — 77% of a fold's time and
    every row-bucketed kernel that matters — so driving it directly covers the
    bucket without inventing a sequence and an MSA to go with it.
    """
    from foldforge.models.esmfold2.trunk import PairUpdateBlock

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    config = ESMFold2Config.from_json(Path(args.checkpoint))
    block = (
        PairUpdateBlock(
            d_pair=config.d_pair, implementation=ImplementationType.MINIWORLD_ENGINE
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    pair = torch.randn(1, length, length, config.d_pair, device=device, dtype=dtype)
    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        block(pair, mask)
    torch.cuda.synchronize()
    del pair, block
    torch.cuda.empty_cache()
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="1ubq,1a1k,3ptb,4yx2")
    parser.add_argument("--data-root", default="validation/data")
    parser.add_argument("--checkpoint", default="model_checkpoints/esmfold2")
    parser.add_argument("--msa-depth", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="benchmark/esmfold2/autotune")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "write the captured cache here instead of in-repo. Parallel jobs on one "
            "GPU MUST each use their own, or they lose each other's updates racing "
            "on shared op files; merge afterwards with submits/_merge_caches.py"
        ),
    )
    parser.add_argument(
        "--synthetic",
        default="",
        help=(
            "comma-separated L values to tune with a synthetic trunk block instead "
            "of a fold. For sizes we have no target at (1024+) — the row buckets "
            "still need entries, and the trunk block is 77%% of a fold"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True

    from miniworld_engine.autotune import cache, capture

    seconds = {}
    model_cache: dict = {}
    # capturing() installs, flushes and uninstalls as a unit, so an exception in the
    # middle of a fold cannot leave the Autotuner patched for the rest of the process.
    root = Path(args.cache_dir).expanduser() if args.cache_dir else None
    with capture.capturing(top_k=args.top_k, root=root) as written:
        for target in filter(None, args.targets.split(",")):
            elapsed = fold_once(target, args, model_cache)
            seconds[target] = round(elapsed, 1)
            logger.info("captured %s in %.1f s", target, elapsed)
        for length in filter(None, args.synthetic.split(",")):
            elapsed = synthetic_once(int(length), args)
            seconds[f"synthetic_L{length}"] = round(elapsed, 1)
            logger.info("captured synthetic L=%s in %.1f s", length, elapsed)

    report = {
        "gpu": cache.gpu_key(),
        "dtype": args.dtype,
        "targets": args.targets.split(","),
        "fold_seconds": seconds,
        "entries_written": len(written),
        "ops": sorted({op for op, *_ in written}),
        "by_entry": [
            {"op": op, "dtype": dt, "bucket": b, "n_configs": n}
            for op, dt, b, n, _ in written
        ],
    }
    # Name the report after what this job captured. The cache dirs are isolated per
    # job but a shared report path is not: six parallel jobs each wrote
    # "capture.json" and only the last one survived, so the record of which job
    # produced which buckets was lost.
    tag = "_".join(filter(None, [*args.targets.split(","), *(
        f"L{n}" for n in filter(None, args.synthetic.split(","))
    )])) or "capture"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info(
        "%s",
        json.dumps(
            {k: report[k] for k in ("gpu", "entries_written", "ops", "fold_seconds")},
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
