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
from foldforge.modules.engine.configure import activation_autocast
from foldforge.train.profile import PhaseTimer, tensor_bytes

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


def split_sizes(model, example, cfg) -> dict:
    """Bytes a 2-GPU split of the step would move (the mini-rollout on a second GPU): the diffusion-head parameters (re-synced after every
    optimizer step), the trunk outputs the rollout reads (single / pair / target features, per crop), the predicted coordinates back."""
    features = {
        k: (v.float() if v.dtype == torch.float64 else v) for k, v in example.features.items()
    }
    with torch.no_grad(), activation_autocast(model):
        batch_data = model._prepare_batch(features)
        emb = step.run_trunk_train(model, batch_data, 1)
    head = model.diffusion_head
    res, atoms, _ = example.atom_positions.shape
    return {
        "num_res": int(res),
        "atoms_per_token": int(atoms),
        "diffusion_head_params": sum(p.numel() for p in head.parameters()),
        "diffusion_head_param_bytes_fp32": sum(p.numel() * p.element_size() for p in head.parameters()),
        "single_bytes": tensor_bytes(emb["single"]),
        "pair_bytes": tensor_bytes(emb["pair"]),
        "target_feat_bytes": tensor_bytes(emb["target_feat"]),
        "rollout_inputs_bytes": tensor_bytes(emb),
        "x_mini_bytes_fp32": res * atoms * 3 * 4,
        "pair_dtype": str(emb["pair"].dtype),
        "single_dtype": str(emb["single"].dtype),
    }


def grad_coverage(model: torch.nn.Module) -> dict:
    """Which parameters got a gradient from one step: groups by the first three name components; the confidence head is in the list (single-stage training; only ``--no-confidence`` leaves it without a loss)."""
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
    ap.add_argument(
        "--no-confidence",
        action="store_true",
        help="DEBUG ONLY: skip the confidence head (the finetuning is one stage; every parameter trains)",
    )
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument(
        "--profile-phases",
        action="store_true",
        help="per-phase GPU timing (CUDA events) of every step: trunk fwd per recycle, diffusion fwd / bwd, trunk bwd, mini-rollout, confidence fwd / bwd, optimizer + EMA",
    )
    ap.add_argument(
        "--rollout-order",
        choices=("after", "before", "side"),
        default=None,
        help="where the mini-rollout runs: after the trunk backward (default, least memory), before the diffusion forward, or on a second CUDA stream (side)",
    )
    ap.add_argument("--augment-chunk", type=int, default=None, help="diffusion augmentations per forward/backward chunk (0 = all)")
    ap.add_argument("--recycles", type=int, default=None, help="fix the trunk passes (min = max)")
    ap.add_argument("--only", choices=("trunk", "diffusion", "rollout", "confidence"), default=None, help="MEASUREMENT: run one part of the step alone")
    ap.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="recompute the pair-stack blocks in the backward instead of storing their activations",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="start from this .miniworld.pt (default: the registered model's checkpoint)",
    )
    ap.add_argument("--run-dir", default="runs/train_protenix1")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="skip the final checkpoint (measurement runs)")
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
        ("rollout_steps", args.rollout_steps),
        ("rollout", args.rollout_order),
        ("augment_chunk", args.augment_chunk),
        ("only", args.only),
        ("recycles_min", args.recycles),
        ("recycles_max", args.recycles),
    ):
        if value is not None:
            changes[key] = value
    if args.no_confidence:
        changes["confidence"] = False
    cfg = replace(cfg, **changes)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = prepare.seed_everything(cfg.seed)

    # dead modules must run in training (their parameters train): do not flag them at load
    pred = ff.load_model(
        args.model,
        # `atom_blocks` on pytorch for training: engine 2.5.1's sm90 atom-DiT backward (`kernels/integrations/local_dit_h100.py::_bwd`, kernel
        # `finalize`) raises CUDA_ERROR_ILLEGAL_ADDRESS on this model's blocks (engine 2.5.0 refused training on that backend and ran pytorch).
        config={"engine": {"skip_dead_modules": False, "ops": {"atom_blocks": "pytorch"}}},
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
    if args.activation_checkpointing:
        print("ACT_CKPT blocks=%d" % prepare.enable_activation_checkpointing(model), flush=True)
    trunk, heads = optim.split_parameters(model)
    print(
        "PREPARE",
        json.dumps(
            {
                **info,
                "params_trunk": sum(p.numel() for _, p in trunk),
                "params_heads": sum(p.numel() for _, p in heads),
            }
        ),
        flush=True,
    )

    if args.profile_phases:
        print("SIZES", json.dumps(split_sizes(model, first, cfg)), flush=True)

    # -- phase 0: one micro step -> gradient coverage (+ kernels)
    model.zero_grad(set_to_none=True)
    parts = step.micro_step(model, first, cfg, passes=cfg.recycles_max)
    print("FIRST_STEP", json.dumps(parts), flush=True)
    cover = grad_coverage(model)
    bad = {
        k: v
        for k, v in cover.items()
        if (v[1] or v[2] or v[3]) and (cfg.confidence or not k.startswith("confidence_head"))
    }
    print(
        "GRAD_COVERAGE groups=%d  groups_with_missing_or_zero_or_nonfinite=%d"
        % (len(cover), len(bad)),
        flush=True,
    )
    sums = [0, 0, 0, 0, 0]
    for key, row in sorted(cover.items()):
        total, none, zero, nonfinite, zero_dead = row
        if cfg.confidence or not key.startswith("confidence_head"):
            sums = [a + b for a, b in zip(sums, row)]
        flag = (
            ""
            if not (none or zero or nonfinite)
            else (
                "  <-- expected (no loss: --no-confidence)"
                if (not cfg.confidence and key.startswith("confidence_head"))
                else "  <-- CHECK"
            )
        )
        print(
            f"GRAD {key:60s} total={total:4d} none={none:4d} zero_live={zero:4d} zero_deadweight={zero_dead:4d} nonfinite={nonfinite:4d}{flag}",
            flush=True,
        )
    print(
        "GRAD_SUM params=%d none=%d zero_live=%d nonfinite=%d zero_deadweight=%d"
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
        timer = PhaseTimer(args.profile_phases)
        if args.profile_phases:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t_step = time.time()
        for micro in range(cfg.grad_accum_steps):
            example = first if micro == 0 and it == 0 else next(source)
            out = step.micro_step(
                model,
                example,
                cfg,
                passes=prepare.draw_passes(cfg, rng),
                scale=1.0 / cfg.grad_accum_steps,
                timer=timer,
            )
            for k, v in out.items():
                totals[k] += v / cfg.grad_accum_steps
        with timer.phase("optimizer_ema"):
            norm_t = optim.clip_grad_norm(params, cfg.grad_clip_max_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        norm = float(norm_t)
        if args.profile_phases:
            torch.cuda.synchronize()
            phases = timer.result()
            phases["step_wall"] = (time.time() - t_step) * 1000.0
            peaks = [v for k, v in phases.items() if k.endswith(":peak_gib")]
            phases["step_peak_gib"] = max(peaks) if peaks else 0.0
            phases["persist_after_step_gib"] = torch.cuda.memory_allocated() / 2**30
            phases["reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            print("PHASES %4d %s" % (it, json.dumps({k: round(v, 2) for k, v in phases.items()})), flush=True)
            if it == 0:
                gib = 2**30
                print(
                    "PERSISTENT",
                    json.dumps(
                        {
                            "params_fp32_gib": round(sum(p.numel() * p.element_size() for p in model.parameters()) / gib, 2),
                            "grads_gib": round(sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None) / gib, 2),
                            "adam_states_gib": round(tensor_bytes([list(s.values()) for s in optimizer.state.values()]) / gib, 2),
                            "ema_gib": round(tensor_bytes(list(ema.shadow.values())) / gib, 2) if ema is not None else 0.0,
                        }
                    ),
                    flush=True,
                )
        history.append(totals["loss"])
        print(
            "STEP %4d lr=%.2e loss=%.4f diffusion=%.4f bond=%.4f distogram=%.4f plddt=%.4f pde=%.4f pae=%.4f gnorm=%.3f t=%.1fs"
            % (
                it,
                lr,
                totals["loss"],
                totals["diffusion"],
                totals["bond"],
                totals["distogram"],
                totals["plddt"],
                totals["pde"],
                totals["pae"],
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

    if args.no_save:
        return
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
    try:
        main()
    except torch.OutOfMemoryError as error:
        print("OOM", str(error)[:240].replace("\n", " "), "max_allocated_gib=%.1f" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
