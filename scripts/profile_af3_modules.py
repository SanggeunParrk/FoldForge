"""Coarse per-module wall time for one released AF3 checkpoint forward.

Eager execution with a CUDA synchronize around every hooked module, so the
numbers are inclusive wall times per stage, not a kernel profile. Synchronizing
serializes host and device work; the totals are therefore an upper bound on the
same stages inside a compiled or graph-replayed forward, whose end-to-end
latency belongs to the benchmark script. Rows record stage, calls per forward
and seconds per forward for the cold run and each warm repeat.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from foldforge.models.io import runtime as runtime_module
from foldforge.models.io.paths import run_directory

STAGES = (
    "evoformer_conditioning",
    "evoformer",
    "evoformer.template_embedding",
    "evoformer.msa_stack",
    "evoformer.trunk_pairformer",
    "diffusion_head",
    "diffusion_head.atom_cross_att_encoder",
    "diffusion_head.transformer",
    "diffusion_head.atom_cross_att_decoder",
    "confidence_head",
    "distogram_head",
)


class StageTimer:
    """Accumulate synchronized wall time per stage for each complete forward."""

    def __init__(self) -> None:
        self.forwards: list[dict[str, Any]] = []
        self._reset()

    def _reset(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._starts: dict[str, list[float]] = defaultdict(list)

    def _begin(self, name: str) -> None:
        torch.cuda.synchronize()
        self._starts[name].append(time.perf_counter())

    def _end(self, name: str) -> None:
        torch.cuda.synchronize()
        self.seconds[name] += time.perf_counter() - self._starts[name].pop()
        self.calls[name] += 1

    def attach(self, model: torch.nn.Module) -> list[str]:
        """Hook every present stage; a ModuleList aggregates its layers."""
        hooked = []
        for name in STAGES:
            try:
                module = model.get_submodule(name)
            except AttributeError:
                continue
            layers = (
                list(module) if isinstance(module, torch.nn.ModuleList) else [module]
            )
            for layer in layers:
                layer.register_forward_pre_hook(lambda *_, n=name: self._begin(n))
                layer.register_forward_hook(lambda *_, n=name: self._end(n))
            hooked.append(name)

        # The sampler wrapper (seed binding) is an instance attribute, so time
        # the complete sampling loop around it: noise, augmentation and denoising.
        sample = model._sample_diffusion  # noqa: SLF001 - profiling hook

        def timed_sample(*args: Any, **kwargs: Any) -> Any:
            self._begin("sampling_loop")
            try:
                return sample(*args, **kwargs)
            finally:
                self._end("sampling_loop")

        model._sample_diffusion = timed_sample  # noqa: SLF001 - profiling hook

        def begin_forward(*_: Any) -> None:
            self._reset()
            self._begin("forward")

        def end_forward(*_: Any) -> None:
            self._end("forward")
            self.forwards.append(
                {
                    "seconds": dict(self.seconds),
                    "calls": dict(self.calls),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )

        model.register_forward_pre_hook(begin_forward)
        model.register_forward_hook(end_forward)
        return hooked


def prepare_spec(target: str, out: Path, *, samples: int, templates: bool) -> Path:
    """Resolve the qualification input like the benchmark driver does."""
    source = Path("validation/inputs/qualification") / target / "input.yaml"
    spec = yaml.safe_load(source.read_text())
    for key in ("fasta", "a3m", "msa_db"):
        spec[key] = {
            k: str((source.parent / Path(v)).resolve())
            for k, v in spec.get(key, {}).items()
        }
    for key in ("ccd_db", "template_db", "cif_db"):
        if spec.get(key):
            spec[key] = str((source.parent / Path(spec[key])).resolve())
    if not templates:
        spec["template"] = {}
    spec.update(n_diffusion_samples=samples, diffusion_batch_size=samples)
    path = out / "inputs" / f"{target}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec))
    return path


def render(rows: list[dict[str, Any]], hooked: list[str]) -> str:
    """One table: stage, calls per forward, cold seconds, warm seconds."""
    order = ["forward", *hooked, "sampling_loop"]
    cold, warm = rows[0], rows[1:]
    lines = ["| stage | calls / forward | cold (s) | warm median (s) | warm share |"]
    lines.append("|---|---:|---:|---:|---:|")
    warm_total = (
        float(torch.tensor([r["seconds"]["forward"] for r in warm]).median())
        if warm
        else None
    )
    for name in order:
        if name not in cold["seconds"]:
            continue
        values = [r["seconds"][name] for r in warm if name in r["seconds"]]
        median = float(torch.tensor(values).median()) if values else None
        share = (
            ""
            if median is None or not warm_total
            else f"{100 * median / warm_total:.1f}%"
        )
        lines.append(
            f"| {name} | {cold['calls'][name]} | {cold['seconds'][name]:.3f} | "
            f"{'' if median is None else f'{median:.3f}'} | {share} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="4yx2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", default="miniworld")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--msa-depth", type=int, default=2048)
    parser.add_argument("--recycles", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--repeats", type=int, default=2, help="Warm forwards")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    out = run_directory(args.out, model="af3")
    out.mkdir(parents=True, exist_ok=True)
    spec = prepare_spec(
        args.target, out, samples=args.samples, templates=args.templates
    )
    config = {
        "backend": args.backend,
        "precision": args.precision,
        "trunk_seed": 0,
        "diffusion_seed": 0,
        "trunk": {"recycles": args.recycles, "msa_depth": args.msa_depth},
        "diffusion": {"steps": args.steps},
        "execution": {
            "compile": args.compile,
            "cuda_graph": args.cuda_graph,
            "bucketing": True,
            "scope": "denoiser",
            "benchmark_repeats": args.repeats,
        },
    }
    config_path = out / "profile-config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    timer = StageTimer()
    hooked: list[str] = []
    original_bind = runtime_module.Runtime.bind

    def bind(self: Any, model: torch.nn.Module) -> Any:
        execution = original_bind(self, model)
        hooked.extend(timer.attach(model))
        return execution

    runtime_module.Runtime.bind = bind  # type: ignore[method-assign]
    from foldforge.models.inference import run

    status = run(
        "af3",
        ["--spec", str(spec), "--config", str(config_path), "--out", str(out / "e2e")],
    )
    if status:
        return status
    table = render(timer.forwards, hooked)
    summary = {
        "target": args.target,
        "gpu": torch.cuda.get_device_name(),
        "config": config,
        "hooked": hooked,
        "forwards": timer.forwards,
        "scope": (
            "eager stage wall time with CUDA synchronize per hooked module; "
            "inclusive of nested stages; not a kernel profile"
        ),
    }
    (out / "profile.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "profile.md").write_text(table + "\n")
    print(table, flush=True)  # noqa: T201 - CLI output contract
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
