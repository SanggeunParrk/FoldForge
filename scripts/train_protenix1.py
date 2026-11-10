"""Protenix v1 finetuning on the FoldForge graph (MiniWorld recipe). Phase 0/1: eager steps, one GPU, gradient accumulation.

  # smoke: one fixed example (FoldBench target) overfit + gradient coverage + kernel table + checkpoint round trip
  python scripts/train_protenix1.py --smoke --target 8ayg-assembly1 --steps 50 --warmup-steps 0 --run-dir RUN

  # training from a data source (MiniWorld provides the adapter; FoldForge does not import MiniWorld)
  python scripts/train_protenix1.py --batch-source my_pkg.adapter:batches --steps 100000 --run-dir RUN [--ckpt X.miniworld.pt]

The recipe (loss weights, Adam, clip, EMA, recycles, augmentations) is :class:`foldforge.train.config.TrainConfig`; read the lr warning there.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

import foldforge as ff
from foldforge.train import (
    TrainBatch,
    TrainConfig,
    checkpoint,
    data,
    optim,
    prepare,
    step,
)

SPECS = Path("/home/psk6950/foldforge-scratch/foldbench/specs_all")


def smoke_batches(pred, target: str):
    """One fixed example: FoldForge's own featurisation of a FoldBench target; the ground truth is the model's own pre-training prediction of it
    (a STAND-IN: the real ground-truth atoms are not mapped to the dense layout in phase 0/1)."""
    net = pred._network
    captured: dict = {}
    original = net.run_trunk

    def wrapped(batch):
        captured["batch"] = batch
        out = original(batch)
        captured["trunk"] = out
        return out

    net.run_trunk = wrapped
    try:
        state = pred.run_trunk(ff.load_inputs(SPECS / f"{target}.yaml"))
    finally:
        net.run_trunk = original
    with torch.no_grad(), pred._scope():
        samples = net.sample_diffusion(captured["trunk"])["diffusion_samples"][
            "atom_positions"
        ]
    state.close()
    features = {
        k: v.clone() for k, v in captured["batch"].items()
    }  # leave inference mode: autograd needs ordinary tensors
    example = TrainBatch(
        features=features, atom_positions=samples[0].clone().float(), name=target
    )
    while True:
        yield example


def grad_coverage(model: torch.nn.Module) -> dict:
    """Which parameters got a gradient from one step: groups by the first three name components; ``confidence_head`` has no loss (expected)."""
    groups: dict[str, list[int]] = defaultdict(
        lambda: [0, 0, 0, 0, 0]
    )  # total, none, zero (live weight), nonfinite, zero (dead weight < 1e-30)
    for name, param in model.named_parameters():
        key = ".".join(name.split(".")[:3])
        row = groups[key]
        row[0] += 1
        if param.grad is None:
            row[1] += 1
        else:
            if not bool(param.grad.any()):
                # an exactly-zero gradient on a weight that is itself ~1e-37 (a decayed module): the bf16 gradient underflows
                row[4 if float(param.detach().abs().max()) < 1e-30 else 2] += 1
            if not bool(torch.isfinite(param.grad).all()):
                row[3] += 1
    return dict(groups)


def kernel_table(model, example, cfg, top: int = 30) -> list[tuple[str, float, int]]:
    """CUDA kernels of one forward + backward by total GPU time: engine kernels versus plain aten GEMMs / elementwise (= the pytorch path)."""
    from torch.profiler import ProfilerActivity, profile

    model.zero_grad(set_to_none=True)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step.micro_step(model, example, cfg, passes=1)
        torch.cuda.synchronize()
    rows = [
        (e.key, e.device_time_total / 1000.0, e.count)
        for e in prof.key_averages()
        if e.device_time_total > 0
    ]
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", default="8ayg-assembly1")
    ap.add_argument("--model", default="protenix1")
    ap.add_argument(
        "--batch-source",
        default=None,
        help="module:callable(cfg, device) -> iterator of TrainBatch",
    )
    ap.add_argument("--steps", type=int, default=50, help="optimizer steps")
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--num-augment", type=int, default=None)
    ap.add_argument("--max-lr", type=float, default=None)
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--trunk-lr-scale", type=float, default=None)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument(
        "--ckpt",
        default=None,
        help="start from this .miniworld.pt (default: the registered model's checkpoint)",
    )
    ap.add_argument("--run-dir", default="runs/train_protenix1")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig()
    changes = {"grad_accum_steps": 1 if args.smoke else cfg.grad_accum_steps}
    for key, value in (
        ("grad_accum_steps", args.grad_accum),
        ("num_augment", args.num_augment),
        ("max_lr", args.max_lr),
        ("min_lr", args.min_lr),
        ("warmup_steps", args.warmup_steps),
        ("trunk_lr_scale", args.trunk_lr_scale),
    ):
        if value is not None:
            changes[key] = value
    if args.freeze_trunk:
        changes["freeze_trunk"] = True
    cfg = replace(cfg, **changes)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = prepare.seed_everything(cfg.seed)

    # dead modules must run in training (their parameters train): do not flag them at load
    pred = ff.load_model(
        args.model,
        config={"engine": {"skip_dead_modules": False}},
        checkpoint=args.ckpt,
    )
    model = pred._network
    device = pred.device
    print(
        "CONFIG",
        json.dumps(
            {
                k: (v if not hasattr(v, "__dataclass_fields__") else v.__dict__)
                for k, v in cfg.__dict__.items()
            },
            default=str,
        ),
        flush=True,
    )

    if args.smoke:
        source = smoke_batches(pred, args.target)
    elif args.batch_source:
        source = iter(data.load_source(args.batch_source)(cfg, device))
    else:
        raise SystemExit("give --smoke or --batch-source module:callable")

    first = next(source)
    print(
        "INFO",
        json.dumps(
            {
                "num_res": int(first.atom_positions.shape[0]),
                "atoms_per_token": int(first.atom_positions.shape[1]),
                "target": first.name,
            }
        ),
        flush=True,
    )
    info = prepare.enable_training(model, cfg)
    trunk, diffusion, no_loss = optim.split_parameters(model)
    print(
        "PREPARE",
        json.dumps(
            {
                **info,
                "params_trunk": sum(p.numel() for _, p in trunk),
                "params_diffusion": sum(p.numel() for _, p in diffusion),
                "params_no_loss": sum(p.numel() for _, p in no_loss),
            }
        ),
        flush=True,
    )

    # -- phase 0: one micro step -> gradient coverage (+ kernels)
    model.zero_grad(set_to_none=True)
    parts = step.micro_step(model, first, cfg, passes=cfg.recycles_max)
    print("FIRST_STEP", json.dumps(parts), flush=True)
    cover = grad_coverage(model)
    bad = {
        k: v
        for k, v in cover.items()
        if (v[1] or v[2] or v[3]) and not k.startswith("confidence_head")
    }
    print(
        "GRAD_COVERAGE groups=%d  groups_with_missing_or_zero_or_nonfinite(excl confidence_head)=%d"
        % (len(cover), len(bad)),
        flush=True,
    )
    sums = [0, 0, 0, 0, 0]
    for key, row in sorted(cover.items()):
        total, none, zero, nonfinite, zero_dead = row
        if not key.startswith("confidence_head"):
            sums = [a + b for a, b in zip(sums, row)]
        flag = (
            ""
            if not (none or zero or nonfinite)
            else (
                "  <-- expected (no loss)"
                if key.startswith("confidence_head")
                else "  <-- CHECK"
            )
        )
        print(
            f"GRAD {key:60s} total={total:4d} none={none:4d} zero_live={zero:4d} zero_deadweight={zero_dead:4d} nonfinite={nonfinite:4d}{flag}",
            flush=True,
        )
    print(
        "GRAD_SUM (excl confidence_head) params=%d none=%d zero_live=%d nonfinite=%d zero_deadweight=%d"
        % (sums[0], sums[1], sums[2], sums[3], sums[4]),
        flush=True,
    )
    if args.smoke and not args.no_profile:
        for name, ms, count in kernel_table(model, first, cfg):
            print(f"KERNEL {ms:10.2f} ms  x{count:<6d} {name[:150]}", flush=True)

    # -- training loop
    optimizer = optim.build_optimizer(model, cfg)
    ema = optim.EMA(model, cfg.ema_decay) if cfg.use_ema else None
    params = [p for g in optimizer.param_groups for p in g["params"]]
    history: list[float] = []
    t0 = time.time()
    for it in range(args.steps):
        lr = optim.set_lr(optimizer, it, cfg)
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        for micro in range(cfg.grad_accum_steps):
            example = first if micro == 0 and it == 0 else next(source)
            out = step.micro_step(
                model,
                example,
                cfg,
                passes=prepare.draw_passes(cfg, rng),
                scale=1.0 / cfg.grad_accum_steps,
            )
            for k, v in out.items():
                totals[k] += v / cfg.grad_accum_steps
        norm = float(optim.clip_grad_norm(params, cfg.grad_clip_max_norm))
        optimizer.step()
        if ema is not None:
            ema.update(model)
        history.append(totals["loss"])
        print(
            "STEP %4d lr=%.2e loss=%.4f diffusion=%.4f bond=%.4f distogram=%.4f gnorm=%.3f t=%.1fs"
            % (
                it,
                lr,
                totals["loss"],
                totals["diffusion"],
                totals["bond"],
                totals["distogram"],
                norm,
                time.time() - t0,
            ),
            flush=True,
        )
        if not (totals["loss"] == totals["loss"]) or abs(totals["loss"]) > 1e6:
            print("DIVERGED at step", it, flush=True)
            break
        if args.save_every and (it + 1) % args.save_every == 0:
            checkpoint.save(
                model,
                run_dir / f"step{it + 1}.miniworld.pt",
                family=args.model,
                source="train",
                ema=ema,
                optimizer=optimizer,
                step=it + 1,
            )

    if history:
        n = max(1, min(10, len(history) // 2))
        print(
            "CURVE first%d_mean=%.4f last%d_mean=%.4f min=%.4f"
            % (n, sum(history[:n]) / n, n, sum(history[-n:]) / n, min(history)),
            flush=True,
        )

    # -- checkpoint round trip
    path = checkpoint.save(
        model,
        run_dir / "final.miniworld.pt",
        family=args.model,
        source="train",
        ema=ema,
        optimizer=optimizer,
        step=len(history),
    )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = model.state_dict()
    worst = max(
        float(
            (blob["state_dict"][k].to(state[k].device).float() - state[k].float())
            .abs()
            .max()
        )
        for k in list(state)[:50]
    )
    print(
        "CHECKPOINT",
        path,
        "keys",
        len(blob["state_dict"]),
        "format",
        blob["format"],
        "max_abs_diff(first50)=%.3g" % worst,
        flush=True,
    )


if __name__ == "__main__":
    main()
