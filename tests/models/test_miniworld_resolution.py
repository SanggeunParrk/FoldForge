"""Default checkpoint resolution (miniworld files, never a silent conversion) and the converter-only isolation of the legacy graph."""

from __future__ import annotations

import subprocess
import sys

import pytest

from foldforge.models import checkpoints
from foldforge.models.checkpoints import miniworld


@pytest.fixture
def empty_root(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(checkpoints, "REGISTRY", tmp_path / "registry.yaml")
    return tmp_path


def test_resolve_native_finds_the_miniworld_file(empty_root):
    target = empty_root / "af3" / "af3.miniworld.pt"
    target.parent.mkdir()
    target.write_bytes(b"x")
    assert miniworld.resolve_native("af3") == target


def test_missing_file_without_blob_says_download_then_convert(empty_root):
    with pytest.raises(FileNotFoundError) as error:
        miniworld.resolve_native("af3")
    text = str(error.value)
    assert "foldforge weights download af3" in text
    assert "foldforge weights convert af3" in text
    assert "never converts" in text


def test_missing_file_with_blob_names_the_exact_convert_command(empty_root):
    blob = empty_root / "boltz2" / "boltz2.bin.zst"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match=r"convert it once: foldforge weights convert boltz2"):
        miniworld.resolve_native("boltz2")
    assert not (blob.parent / "boltz2.miniworld.pt").exists()  # nothing was converted behind the caller's back


def test_every_registered_model_has_a_native_file_name():
    assert set(miniworld.NATIVE_FILES) == set(checkpoints.DEFAULT_FILES)


def test_load_model_without_a_converted_file_fails_with_the_command(empty_root):
    import foldforge

    with pytest.raises(FileNotFoundError, match="foldforge weights convert af3"):
        foldforge.load_model("af3", device="cpu")


def test_the_loader_refuses_a_released_blob(tmp_path):
    from foldforge.models.loading import load_checkpoint

    blob = tmp_path / "weights.bin.zst"  # not a locked blob name: the size gate is not the subject
    blob.write_bytes(b"x")
    with pytest.raises(ValueError, match="foldforge weights convert af3"):
        load_checkpoint("af3", blob, device="cpu")


def test_runtime_imports_nothing_of_the_legacy_released_graph():
    """Load / predict / training paths never import the converter's dense graph, the Haiku codec or the released-layout modules."""
    code = (
        "import sys\n"
        "import foldforge, foldforge.api, foldforge.models.loading, foldforge.models.execution\n"
        "import foldforge.models.architectures.af3, foldforge.models.checkpoints.miniworld\n"
        "bad = sorted(m for m in sys.modules if m.startswith('foldforge.models.checkpoints.legacy') or m.endswith('checkpoints.haiku'))\n"
        "assert not bad, bad\n"
        "assert 'foldforge.modules.dense.pairformer' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]


def test_weights_convert_command_arguments(empty_root):
    from click.testing import CliRunner

    from foldforge.models.checkpoints.commands import weights

    runner = CliRunner()
    assert runner.invoke(weights, ["convert"]).exit_code != 0  # models or --all
    unknown = runner.invoke(weights, ["convert", "nonesuch"])
    assert unknown.exit_code != 0 and "Unknown models" in unknown.output
    both = runner.invoke(weights, ["convert", "af3", "--all"])
    assert both.exit_code != 0
    skipped = runner.invoke(weights, ["convert", "--all"])  # no blob installed anywhere: every model is skipped, nothing converted
    assert skipped.exit_code == 0, skipped.output
    assert skipped.output.count("SKIP") == len(miniworld.NATIVE_FILES)
    assert "foldforge weights download af3" in skipped.output
