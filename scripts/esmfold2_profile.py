"""Where does a fold's time actually go, and what did torch.compile capture?

Answers two questions the optimization sweep could only guess at:

1. **Stage breakdown.** CUDA-synchronised timers around each of the four stages
   (inputs embedder, pair trunk, structure head, confidence head), plus a timer
   inside the structure head separating cache construction from the solver loop.
   This is what decides whether a diffusion-loop optimization *can* matter.
2. **Compile coverage.** ``torch._dynamo.explain`` on one block, reporting the
   graph and graph-break counts. A compiled module that is all breaks is
   compiled in name only.

Usage (GPU node)::

    sbatch scripts/esmfold2_profile.sbatch --target 3ptb
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file
from team_gm.modules.exceptions import ImplementationType

from foldforge.data.ccd import CCDDatabase, default_path
from foldforge.models.architectures.esmfold2 import ESMFold2Model as TeamGMModel
from foldforge.models.checkpoints import esmfold2 as convert
from foldforge.models.config.esmfold2 import ESMFold2Config
from foldforge.models.sampling import bind_sampling_seed
from foldforge.modules.sequence.features import ESMFold2InputBuilder, build_input
from foldforge.modules.sequence.features import model_kwargs as team_gm_kwargs

logger = logging.getLogger("esmfold2_profile")

TIMINGS: dict[str, float] = {}


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Accumulate synchronised wall time under ``name``."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        TIMINGS[name] = TIMINGS.get(name, 0.0) + time.perf_counter() - start


def instrument(model: TeamGMModel) -> None:
    """Wrap the four stages, and split the structure head into cache vs solver."""
    for name in ("inputs_embedder", "pair_trunk", "structure_head", "confidence_head"):
        module = getattr(model, name)
        original = module.forward if name != "structure_head" else module.sample
        label = name

        def wrapper(  # noqa: ANN202
            *args: object,
            _original: object = original,
            _label: str = label,
            **kwargs: object,
        ):
            with stage(_label):
                return _original(*args, **kwargs)

        if name == "structure_head":
            module.sample = wrapper
        else:
            module.forward = wrapper

    diffusion = model.structure_head.diffusion_module
    build_cache, denoise = diffusion.build_cache, diffusion.denoise

    def timed_build_cache(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with stage("  build_cache"):
            return build_cache(*args, **kwargs)

    def timed_denoise(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        with stage("  denoise_calls"):
            TIMINGS["  denoise_count"] = TIMINGS.get("  denoise_count", 0) + 1
            return denoise(*args, **kwargs)

    diffusion.build_cache = timed_build_cache
    diffusion.denoise = timed_denoise
    instrument_pair_trunk(model.pair_trunk)


def instrument_pair_trunk(trunk: torch.nn.Module) -> None:
    """Break the pair trunk down — it is over 90% of a large fold.

    Two levels: the recurrence's three parts (the hoisted injection base, the
    per-pass LM encoder, the 48-block folding trunk), then the three ops every
    ``PairUpdateBlock`` runs, summed over all blocks and all passes. The per-op
    wrappers add a synchronise each, but at 594 tokens a single op is milliseconds
    against a ~10 us sync, so the attribution is not paying for itself in noise.
    """
    for name in ("injection_base", "lm_encoder", "folding_trunk"):
        target = getattr(trunk, name, None)
        if target is None:
            continue
        original = (
            target
            if callable(target) and not isinstance(target, torch.nn.Module)
            else target.forward
        )

        def wrapper(  # noqa: ANN202
            *args: object,
            _original: object = original,
            _label: str = f"  {name}",
            **kwargs: object,
        ):
            with stage(_label):
                return _original(*args, **kwargs)

        if isinstance(target, torch.nn.Module):
            target.forward = wrapper
        else:
            setattr(trunk, name, wrapper)

    for block in trunk.folding_trunk.blocks:
        for op in ("tri_mul_out", "tri_mul_in", "pair_transition"):
            module = getattr(block, op)

            def op_wrapper(  # noqa: ANN202
                *args: object,
                _original: object = module.forward,
                _label: str = f"    {op}",
                **kwargs: object,
            ):
                with stage(_label):
                    return _original(*args, **kwargs)

            module.forward = op_wrapper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccd-db", type=Path, default=default_path())
    parser.add_argument("--target", default="3ptb")
    parser.add_argument("--data-root", default="validation/inputs/data")
    parser.add_argument("--out", default="benchmark/esmfold2/profile")
    parser.add_argument("--checkpoint", default="model_checkpoints/esmfold2")
    parser.add_argument("--msa-depth", type=int, default=512)
    parser.add_argument("--trunk-seed", type=int, default=0)
    parser.add_argument("--diffusion-seed", type=int, default=0)
    parser.add_argument(
        "--implementation", default=ImplementationType.MINIWORLD_ENGINE.value
    )
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
        build_input(sample, args.msa_depth), seed=args.trunk_seed, device=device
    )
    lm_hidden = torch.load(
        sample / "cache" / "esmc_hidden_states.pt", map_location=device
    ).to(dtype)

    config = ESMFold2Config.from_json(checkpoint)
    backend = ImplementationType(args.implementation)
    model = TeamGMModel(config, backend).to(device=device, dtype=dtype).eval()
    model.load_state_dict(
        convert.convert_model(load_file(checkpoint / "model.safetensors"), config)
    )
    bind_sampling_seed(model, "esmfold2", args.diffusion_seed)
    kwargs = team_gm_kwargs(features, dtype)

    def fold() -> torch.Tensor:
        return model(
            **kwargs,
            lm_hidden_states=lm_hidden,
            num_diffusion_samples=1,
            generator=torch.Generator(device=device).manual_seed(args.trunk_seed),
        ).coords

    # Warm-up fold first: otherwise the breakdown is a map of Triton autotuning.
    with torch.no_grad():
        fold()
    instrument(model)
    TIMINGS.clear()
    with stage("total"), torch.no_grad():
        fold()

    total = TIMINGS.get("total", 0.0)
    report = {
        "target": args.target,
        "n_tokens": int(features["token_attention_mask"].shape[1]),
        "device": torch.cuda.get_device_name(device),
        "backend": backend.value,
        "stages": {
            name: {
                "seconds": round(value, 4),
                "share": round(value / total, 4)
                if total and "count" not in name
                else None,
            }
            for name, value in TIMINGS.items()
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.target}.json").write_text(json.dumps(report, indent=2) + "\n")

    logger.info("total %.3f s over %d tokens", total, report["n_tokens"])
    for name, value in TIMINGS.items():
        if "count" in name:
            logger.info("%-18s %d", name, int(value))
        else:
            logger.info("%-18s %8.4f s  %5.1f%%", name, value, 100 * value / total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
