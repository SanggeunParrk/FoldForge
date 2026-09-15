"""Cache build supervision must retain failures and terminate whole process groups."""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest
from support import REPOSITORY_ROOT


@pytest.fixture
def builder():
    path = REPOSITORY_ROOT / "scripts" / "build_autotune_cache.py"
    spec = importlib.util.spec_from_file_location("cache_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("code", [0, 7])
def test_exit_status_and_log(builder, tmp_path, code):
    log = tmp_path / "worker.log"
    result = builder.supervise(
        [sys.executable, "-c", f'print("retained"); raise SystemExit({code})'], log, 10
    )
    assert result == code
    assert "retained" in log.read_text()


@pytest.mark.parametrize("detached", [False, True])
def test_timeout_kills_descendants(builder, tmp_path, detached):
    log = tmp_path / "worker.log"
    code = (
        "import subprocess,sys,time; "
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"],'
        "start_new_session=DETACHED); "
        "print(p.pid,flush=True); time.sleep(120)"
    )
    code = code.replace("DETACHED", str(detached))
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        builder.supervise([sys.executable, "-c", code], log, 2)
    assert time.monotonic() - start < 30
    pid = int(log.read_text().strip())
    stat = Path(f"/proc/{pid}/stat")
    # SIGKILL delivery and descendant reaping are asynchronous. The supervisor
    # waits for its direct child; a grandchild may briefly remain runnable.
    deadline = time.monotonic() + 3
    while True:
        try:
            state = stat.read_text().rsplit(") ", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            state = "gone"
        if state in ("X", "Z", "gone") or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert state in ("X", "Z", "gone")


def test_pytorch_baseline_is_not_a_cache_build(builder, tmp_path):
    config = tmp_path / "baseline.yaml"
    config.write_text("backend: pytorch\n")
    with pytest.raises(ValueError, match="backend=miniworld"):
        builder.validate_backend(["--config", str(config)])
    builder.validate_backend([])


def test_partial_search_is_never_promoted(builder):
    a = {"kwargs": {"TILE": 16}, "num_warps": 4, "num_stages": 1}
    b = {"kwargs": {"TILE": 32}, "num_warps": 4, "num_stages": 1}
    c = {"kwargs": {"TILE": 64}, "num_warps": 4, "num_stages": 1}
    good = {"searched": [a, b], "entries": [{**a, "ms": 0.1}], "workload": {}}
    profiles = {
        "complete": {"work": good},
        "partial": {"work": {**good, "searched": [a]}},
        "wrong_space": {"work": {**good, "searched": [a, c]}},
        "failed": {"work": {**good, "entries": [{**a, "ms": float("inf")}]}},
    }
    shard = {
        "_unit_complete": False,
        "_provenance": {"sentinel": "kept"},
        "op": {"grid": [a, b], "measurements": profiles, "op_id": "identity"},
    }
    result = builder.completed_subset(shard)
    assert result["_has_entries"]
    assert not result["_unit_complete"]
    assert result["_provenance"] == shard["_provenance"]
    assert set(result["op"]["entries"]) == {"complete"}
    assert set(result["op"]["measurements"]) == {"complete"}
    assert len(shard["op"]["measurements"]) == 4


def test_legacy_shard_without_search_evidence_is_not_recovered(builder):
    result = builder.completed_subset(
        {"_unit_complete": False, "op": {"entries": {"key": []}}}
    )
    assert not result["_has_entries"]
    assert "op" not in result


def test_worker_output_uses_shared_runs_policy(builder, tmp_path, monkeypatch):
    from foldforge.models.io import paths

    root = tmp_path / "runs"
    monkeypatch.setattr(paths, "RUNS_ROOT", root)
    captured = []
    monkeypatch.setattr(
        builder, "worker", lambda args, _fold_args: captured.append(args.out) or 0
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_autotune_cache.py",
            "--worker",
            "--model",
            "af3",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            "cache-unit",
        ],
    )
    assert builder.main() == 0
    assert captured == [root / "cache-unit"]
    assert not root.exists()
