"""Cache build supervision must retain failures and terminate whole process groups."""

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "scripts" / "build_autotune_cache.py"
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
    try:
        state = stat.read_text().rsplit(") ", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        state = "gone"
    assert state in ("Z", "gone")


def test_pytorch_baseline_is_not_a_cache_build(builder, tmp_path):
    config = tmp_path / "baseline.yaml"
    config.write_text("backend: pytorch\n")
    with pytest.raises(ValueError, match="backend=miniworld"):
        builder.validate_backend(["--config", str(config)])
    builder.validate_backend([])
