"""The code refuses a converted blob it was not validated with."""

from __future__ import annotations

import pytest

from foldforge.models.checkpoints import lock


def test_every_default_checkpoint_is_locked():
    from foldforge.models.checkpoints import DEFAULT_FILES

    pinned = lock.pins()
    for name, filename in DEFAULT_FILES.items():
        assert filename in pinned, name


def test_a_resized_blob_is_refused_by_name(tmp_path, monkeypatch):
    lock_file = tmp_path / "checkpoints.lock"
    lock_file.write_text("# comment\n" + "ab" * 32 + " 4 chai1/chai1.bin.zst\n")
    monkeypatch.setattr(lock, "LOCK", lock_file)
    blob = tmp_path / "chai1.bin.zst"
    blob.write_bytes(b"1234")
    lock.check_size(blob)  # the locked size passes
    blob.write_bytes(b"12345")
    with pytest.raises(lock.CheckpointMismatchError, match="validated with"):
        lock.check_size(blob)
    alias = tmp_path / "alias.bin.zst"
    alias.symlink_to(blob)
    with pytest.raises(lock.CheckpointMismatchError):
        lock.check_size(alias)
    # An unlocked file is not this module's business.
    other = tmp_path / "custom.bin.zst"
    other.write_bytes(b"x")
    lock.check_size(other)


def test_verify_reports_missing_and_resized(tmp_path, monkeypatch):
    lock_file = tmp_path / "checkpoints.lock"
    lock_file.write_text("ab" * 32 + " 3 a/one.bin\n" + "cd" * 32 + " 2 b/two.bin\n")
    monkeypatch.setattr(lock, "LOCK", lock_file)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"12")
    problems = lock.verify(tmp_path, quick=True)
    assert problems == ["size     a/one.bin", "missing  b/two.bin"]
