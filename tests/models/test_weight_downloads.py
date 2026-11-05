"""Weight provisioning exercises real files and HTTP without multi-GB fixtures."""

# ruff: noqa: SLF001 - exercise provisioning boundaries and failure paths

from __future__ import annotations

import hashlib
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from http.client import HTTPResponse

import pytest
from click.testing import CliRunner

from foldforge.cli import app
from foldforge.models.checkpoints import download as weights
from foldforge.models.checkpoints.lock import CheckpointMismatchError, Pin


def artifact(name, content, **kwargs: Any):
    return {
        "name": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        **kwargs,
    }


@pytest.fixture
def tiny_catalog(monkeypatch):
    blob = b"validated converted weights"
    lm = b"language model"
    sidecar = b"model-specific adapter"
    raw = b"published checkpoint"
    data = {
        "models": {
            "boltz2": {
                "reason": None,
                "converter": "boltz2",
                "checkpoint": "raw.pt",
                "files": [artifact("raw.pt", raw, url="https://example.test/raw.pt")],
                "aux": ["tower"],
                "extras": [artifact("adapter.npz", sidecar)],
            },
            "af3": {
                "reason": "Request authorized weights",
                "converter": None,
                "checkpoint": ".",
                "files": [],
                "aux": [],
            },
        },
        "aux": {"tower": [artifact("model.bin", lm, url="https://example.test/tower")]},
    }
    pins = {
        name: Pin(hashlib.sha256(blob).hexdigest(), len(blob), f"{name}/{name}.bin.zst")
        for name in data["models"]
    }
    monkeypatch.setattr(weights, "catalog", lambda: data)
    monkeypatch.setattr(weights, "model_pin", pins.__getitem__)
    return data, pins, blob, lm, sidecar, raw


def populate_source(root, fixture):
    _, pins, blob, lm, sidecar, _ = fixture
    for pin in pins.values():
        path = root / pin.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    (root / "tower").mkdir()
    (root / "tower/model.bin").write_bytes(lm)
    (root / "boltz2/adapter.npz").write_bytes(sidecar)


def test_offline_import_verifies_every_dependency_and_is_repeatable(
    tmp_path, tiny_catalog, monkeypatch
):
    source, target = tmp_path / "source", tmp_path / "installed"
    populate_source(source, tiny_catalog)
    monkeypatch.setattr(
        weights, "_fetch", lambda *_: pytest.fail("offline import used network")
    )
    results = weights.download_weights(
        ["boltz2", "af3", "boltz2"], root=target, source_dir=source
    )
    assert [r.model for r in results] == ["boltz2", "af3"]
    assert all(r.installed for r in results)
    stamp = (target / "boltz2/boltz2.bin.zst").stat().st_mtime_ns
    assert weights.download_weights(["boltz2"], root=target, source_dir=source)[
        0
    ].installed
    assert (target / "boltz2/boltz2.bin.zst").stat().st_mtime_ns == stamp
    assert (target / "boltz2/adapter.npz").read_bytes() == tiny_catalog[4]
    assert (target / "tower/model.bin").read_bytes() == tiny_catalog[3]


@pytest.mark.parametrize(
    "damaged", ["boltz2/boltz2.bin.zst", "boltz2/adapter.npz", "tower/model.bin"]
)
def test_corrupted_import_is_never_success(tmp_path, tiny_catalog, damaged):
    source, target = tmp_path / "source", tmp_path / "installed"
    populate_source(source, tiny_catalog)
    (source / damaged).write_bytes(b"corrupted")
    result = weights.download_weights(["boltz2"], root=target, source_dir=source)[0]
    assert not result.installed
    assert "Checksum mismatch" in result.message
    assert not list(target.rglob(".weights-*"))


def test_existing_unknown_blob_is_not_replaced(tmp_path, tiny_catalog):
    source, target = tmp_path / "source", tmp_path / "installed"
    populate_source(source, tiny_catalog)
    target_file = target / "boltz2/boltz2.bin.zst"
    target_file.parent.mkdir(parents=True)
    target_file.write_bytes(b"someone else's weights")
    assert not weights.download_weights(["boltz2"], root=target, source_dir=source)[
        0
    ].installed
    assert target_file.read_bytes() == b"someone else's weights"


def test_manual_failure_does_not_stop_other_models(tmp_path, tiny_catalog, monkeypatch):
    _, _, blob, lm, sidecar, raw = tiny_catalog

    def fetch(item, cache, _report) -> Path:
        cache.mkdir(parents=True, exist_ok=True)
        result = cache / Path(item.name).name
        result.write_bytes(raw if item.name == "raw.pt" else lm)
        return result

    def convert(_model, source, output, _cache) -> None:
        assert (source).read_bytes() == raw
        (output / "boltz2.bin.zst").write_bytes(blob)
        (output / "adapter.npz").write_bytes(sidecar)

    monkeypatch.setattr(weights, "_fetch", fetch)
    monkeypatch.setattr(weights, "_convert", convert)
    results = weights.download_weights(["af3", "boltz2"], root=tmp_path)
    assert not results[0].installed
    assert "Request authorized" in results[0].message
    assert results[1].installed


def test_invalid_conversion_stays_out_of_inference_tree(
    tmp_path, tiny_catalog, monkeypatch
):
    source = tmp_path / "download"
    source.write_bytes(tiny_catalog[-1])
    monkeypatch.setattr(weights, "_fetch", lambda *_: source)

    def convert(_model, _source, output, _cache) -> None:
        (output / "boltz2.bin.zst").write_bytes(b"wrong converter output")

    monkeypatch.setattr(weights, "_convert", convert)
    result = weights.download_weights(["boltz2"], root=tmp_path / "root")[0]
    assert not result.installed
    assert not (tmp_path / "root/boltz2/boltz2.bin.zst").exists()
    assert (tmp_path / "root/.weights/converted/boltz2/boltz2.bin.zst").exists()


def test_unknown_model_has_no_filesystem_side_effect(tmp_path, tiny_catalog):
    assert tiny_catalog
    target = tmp_path / "unused"
    with pytest.raises(ValueError, match="Unknown"):
        weights.download_weights(["typo"], root=target)
    assert not target.exists()


def test_http_transfer_checks_hash_retries_and_reuses_cache(tmp_path, monkeypatch):
    content = b"raw HTTP fixture"
    state = {"content": b"truncated", "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state["requests"] += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(state["content"])

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = urllib.request.urlopen

    def routed(request, **kwargs: Any) -> HTTPResponse:
        assert request.full_url == "https://example.test/weights.pt"
        assert not request.has_header("Authorization")
        assert request.get_header("User-agent") == "foldforge/0.1.0"
        return original(f"http://127.0.0.1:{server.server_port}/weights.pt", **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", routed)
    item = weights.Artifact(
        **artifact("weights.pt", content, url="https://example.test/weights.pt")
    )
    try:
        with pytest.raises(CheckpointMismatchError):
            weights._fetch(item, tmp_path, lambda _: None)
        assert not (tmp_path / "downloads" / item.sha256 / item.name).exists()
        state["content"] = content
        result = weights._fetch(item, tmp_path, lambda _: None)
        assert result.read_bytes() == content
        assert weights._fetch(item, tmp_path, lambda _: None) == result
        assert state["requests"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_huggingface_download_passes_immutable_revision(tmp_path, monkeypatch):
    import types

    content = b"HF source fixture"
    source = tmp_path / "hub-file"
    source.write_bytes(content)
    calls = []

    def download(repo, name, **kwargs: Any) -> str:
        calls.append((repo, name, kwargs))
        return str(source)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download)
    )
    item = weights.Artifact(
        **artifact("weights.pt", content, repo="owner/repo", revision="a" * 40)
    )
    result = weights._fetch(item, tmp_path / "cache", lambda _: None)
    assert result.read_bytes() == content
    assert calls[0][2]["revision"] == "a" * 40


@pytest.mark.parametrize("args", [["--all", "boltz2"], [], ["typo"]])
def test_cli_rejects_ambiguous_or_unknown_models(args, tmp_path, tiny_catalog):
    assert tiny_catalog
    target = tmp_path / "unused"
    result = CliRunner().invoke(
        app, ["weights", "download", *args, "--root", str(target)]
    )
    assert result.exit_code == 2
    assert not target.exists()


def test_cli_dry_run_does_not_access_network_or_create_root(
    tmp_path, tiny_catalog, monkeypatch
):
    assert tiny_catalog
    monkeypatch.setattr(
        weights,
        "download_weights",
        lambda *_args, **_kwargs: pytest.fail("dry run executed"),
    )
    result = CliRunner().invoke(
        app,
        [
            "weights",
            "download",
            "--all",
            "--dry-run",
            "--root",
            str(tmp_path / "unused"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "BLOCKED" in result.output
    assert "tower" in result.output
    assert not (tmp_path / "unused").exists()


def test_cli_reports_partial_failure_and_nonzero_exit(tmp_path, tiny_catalog):
    source = tmp_path / "source"
    populate_source(source, tiny_catalog)
    (source / "af3/af3.bin.zst").unlink()
    result = CliRunner().invoke(
        app,
        [
            "weights",
            "download",
            "--all",
            "--from-dir",
            str(source),
            "--root",
            str(tmp_path / "target"),
        ],
    )
    assert result.exit_code == 1
    assert "OK boltz2" in result.output
    assert "FAILED af3" in result.output


def test_real_catalog_covers_lock_and_pins_all_remote_assets():
    data = weights.catalog()
    assert set(data["models"]) == weights.DEFAULT_FILES.keys()
    assert sum(not item["reason"] for item in data["models"].values()) == 11
    assert len({weights.model_pin(name).path for name in data["models"]}) == 12
    for item in [
        *(f for m in data["models"].values() for f in m["files"]),
        *(f for group in data["aux"].values() for f in group),
    ]:
        assert len(item["sha256"]) == 64
        assert item["size"] > 0
        assert item["url"].startswith("https://")
        if "repo" in item:
            assert len(item["revision"]) == 40
    for name in ("esmfold2", "esmfold2-fast", "chai1"):
        assert data["models"][name]["extras"]


def test_all_public_selects_only_automatic_releases(tmp_path, tiny_catalog):
    assert tiny_catalog
    result = CliRunner().invoke(
        app,
        [
            "weights",
            "download",
            "--all-public",
            "--dry-run",
            "--root",
            str(tmp_path / "unused"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "boltz2:" in result.output
    assert "af3:" not in result.output
    assert not (tmp_path / "unused").exists()


def test_fresh_offline_import_accepts_an_original_qualified_release(
    tmp_path, tiny_catalog, monkeypatch
):
    source, target = tmp_path / "source", tmp_path / "target"
    populate_source(source, tiny_catalog)
    original = tiny_catalog[1]["boltz2"]
    newer = weights.Pin(
        hashlib.sha256(b"new release").hexdigest(), len(b"new release"), original.path
    )
    monkeypatch.setattr(
        weights,
        "model_pin",
        lambda name: newer if name == "boltz2" else tiny_catalog[1][name],
    )
    monkeypatch.setattr(
        weights,
        "accepted_pins",
        lambda filename: (
            (original, newer) if filename == Path(original.path).name else ()
        ),
    )
    result = weights.download_weights(["boltz2"], root=target, source_dir=source)[0]
    assert result.installed, result.message
    assert (target / original.path).read_bytes() == tiny_catalog[2]


@pytest.mark.parametrize("corrupt", [False, True])
def test_ready_release_is_verified_and_installed_without_conversion(
    tmp_path, tiny_catalog, monkeypatch, corrupt
):
    data, pins, blob, *_ = tiny_catalog
    spec = data["models"]["af3"]
    spec.update(
        reason=None,
        converter=None,
        format="converted",
        checkpoint="af3.bin.zst",
        files=[artifact("af3.bin.zst", blob, url="https://example.test/af3.bin.zst")],
    )
    source = tmp_path / "upstream"
    source.write_bytes(b"corrupt" if corrupt else blob)
    monkeypatch.setattr(weights, "_fetch", lambda *_: source)
    monkeypatch.setattr(
        weights,
        "_convert",
        lambda *_: pytest.fail("converted an already-ready release"),
    )
    target = tmp_path / "installed"
    result = weights.download_weights(["af3"], root=target)[0]
    assert result.installed is not corrupt
    if corrupt:
        assert not (target / pins["af3"].path).exists()
    else:
        assert (target / pins["af3"].path).read_bytes() == blob


def test_missing_external_language_model_is_repaired_without_reconversion(
    tmp_path, tiny_catalog, monkeypatch
):
    data, pins, blob, lm, adapter, _ = tiny_catalog
    data["models"]["boltz2"]["extras"][0]["url"] = "https://example.test/adapter.npz"
    target = tmp_path / "installed"
    weight = target / pins["boltz2"].path
    weight.parent.mkdir(parents=True)
    weight.write_bytes(blob)

    def fetch(item, cache, _report) -> Path:
        path = cache / Path(item.name).name
        path.write_bytes(adapter if item.name == "adapter.npz" else lm)
        return path

    monkeypatch.setattr(weights, "_fetch", fetch)
    monkeypatch.setattr(
        weights, "_convert", lambda *_: pytest.fail("reconverted a verified checkpoint")
    )
    result = weights.download_weights(["boltz2"], root=target)[0]
    assert result.installed, result.message
    assert (weight.parent / "adapter.npz").read_bytes() == adapter
    assert (target / "tower/model.bin").read_bytes() == lm
