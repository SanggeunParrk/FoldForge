# ruff: noqa: T201, S603, SLF001 - CLI diagnostics, argv execution, pinned engine bridge
"""Fill cache gaps through the ordinary released-model inference entry point.

Run on an allocated GPU. Each invocation owns a subprocess, timeout, log and shard.
A successful unit merges into a private cache directory, seeded from shipped data.
Use --engine-cache-dir with ordinary inference to consume that directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def supervise(command: list[str], log: Path, seconds: float) -> int:
    """Bound the whole unit including descendants, retaining its log on failure."""
    start = time.monotonic()
    with log.open("w") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                remaining = seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, seconds)  # noqa: TRY301 - one cleanup boundary
                try:
                    return process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    print(
                        f"unit elapsed={time.monotonic() - start:.0f}s log={log}",
                        flush=True,
                    )
        except BaseException:
            # Compiler workers create their own sessions. Reuse the engine's
            # ancestry/start-time checked cleanup; killpg alone misses them.
            from miniworld_engine.autotune.builder import _kill_unit_tree

            _kill_unit_tree(process.pid)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def validate_backend(fold_args: list[str]) -> None:
    """A PyTorch baseline must never be reported as an engine cache build."""
    import yaml
    from foldforge.models.config import Config

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    options, _ = parser.parse_known_args(fold_args)
    config = (
        Config.model_validate(yaml.safe_load(options.config.read_text()) or {})
        if options.config
        else Config()
    )
    if config.backend != "miniworld":
        message = "Cache building requires backend=miniworld in the model config"
        raise ValueError(message)


def worker(args: argparse.Namespace, fold_args: list[str]) -> int:
    from miniworld_engine import settings
    from miniworld_engine.autotune import cache, capture

    cache._CACHE_ROOT = args.cache_dir.resolve()
    settings.configure(
        autotune_on_miss_shards=str(args.out / "partial"),
        compile_jobs=args.compile_jobs,
    )
    capture.set_round_cache(str(args.cache_dir.with_suffix(".rounds")))
    capture.set_incremental(True)
    capture.install()
    os.environ["MINIWORLD_SMEM_LOG"] = str(args.out / "compile.smem")

    def terminate(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    from foldforge.models.inference import run

    success = False
    try:
        result = run(args.model, fold_args)
        if capture.record_errors():
            message = f"capture errors: {capture.record_errors()}"
            raise RuntimeError(message)
        success = result in (None, 0)
        return result or 0
    finally:
        capture.dump_shard(str(args.out / "unit.json"), unit_complete=success)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("af3", "protenix", "opendde", "esmfold2")
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--compile-jobs", type=int, default=8)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args, fold_args = parser.parse_known_args()
    if fold_args[:1] == ["--"]:
        fold_args = fold_args[1:]
    if args.timeout <= 0 or args.compile_jobs < 1:
        parser.error("timeout and compile-jobs must be positive")
    args.out = args.out.resolve()
    args.cache_dir = args.cache_dir.resolve()
    validate_backend(fold_args)
    if args.worker:
        return worker(args, fold_args)
    import fcntl

    from miniworld_engine.autotune import cache, capture

    if args.cache_dir == cache._CACHE_ROOT.resolve():
        parser.error("use a separate cache directory; shipped data is preserved")
    args.cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with args.cache_dir.with_suffix(".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.cache_dir.exists():
            seed = args.cache_dir.with_name(
                args.cache_dir.name + f".seed-{os.getpid()}"
            )
            shutil.copytree(cache._CACHE_ROOT, seed)
            seed.rename(args.cache_dir)
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / "unit.json").exists():
            parser.error("output already contains a shard; choose another --out")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--model",
            args.model,
            "--cache-dir",
            str(args.cache_dir),
            "--out",
            str(args.out),
            "--compile-jobs",
            str(args.compile_jobs),
            "--",
            *fold_args,
            "--engine-cache-dir",
            str(args.cache_dir),
            "--out",
            str(args.out / "prediction"),
        ]
        start = time.monotonic()
        try:
            code = supervise(command, args.out / "worker.log", args.timeout)
            status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            code, status = 124, "timeout"
        report = {
            "model": args.model,
            "status": status,
            "returncode": code,
            "seconds": time.monotonic() - start,
            "arguments": fold_args,
            "cache_dir": str(args.cache_dir),
            "merged_entries": 0,
        }
        if code == 0:
            shard = args.out / "unit.json"
            if not json.loads(shard.read_text()).get("_unit_complete"):
                message = "worker did not finish its unit"
                raise RuntimeError(message)
            cache._CACHE_ROOT = args.cache_dir
            written = capture.merge_shards([shard])
            if capture._MERGE_SKIPPED:
                message = f"merge rejected: {capture._MERGE_SKIPPED}"
                raise RuntimeError(message)
            report["merged_entries"] = len(written)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
