"""Transport refactors must preserve submission and cache failure semantics."""

import io
import tarfile
from unittest.mock import Mock

import pytest
import requests

from foldforge.data.msa import service


def test_submission_retries_and_preserves_protocol(monkeypatch):
    response = Mock()
    response.json.return_value = {"id": "ticket", "status": "PENDING"}
    request = Mock(side_effect=[requests.Timeout("transient"), response])
    sleep = Mock()
    monkeypatch.setattr(service.requests, "request", request)
    monkeypatch.setattr(service.time, "sleep", sleep)
    client = service._MSAService(  # noqa: SLF001 - exercise the shared transport boundary
        "https://example.invalid", "ticket/msa", {"User-Agent": "test"}, "a@b.c"
    )
    assert client.submit(["AAA", "GG"], "env") == response.json.return_value
    assert request.call_count == 2
    assert request.call_args.kwargs["data"] == {
        "q": ">101\nAAA\n>102\nGG\n",
        "mode": "env",
        "email": "a@b.c",
    }
    response.raise_for_status.assert_called_once()
    sleep.assert_called_once_with(2)


@pytest.mark.parametrize("valid", [False, True])
def test_download_replaces_cache_only_after_archive_validation(
    tmp_path, monkeypatch, *, valid: bool
):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        record = tarfile.TarInfo("result.a3m")
        record.size = 4
        archive.addfile(record, io.BytesIO(b">1\nA"))
    payload = stream.getvalue() if valid else b"not an archive"
    response = Mock(content=payload)
    client = service._MSAService(  # noqa: SLF001 - exercise the shared transport boundary
        "https://example.invalid", "ticket/msa", {}, ""
    )
    monkeypatch.setattr(client, "request_with_retries", Mock(return_value=response))
    target = tmp_path / "result.tar.gz"
    target.write_bytes(b"previous cache")
    if valid:
        client.download("ticket", str(target))
        assert target.read_bytes() == payload
    else:
        with pytest.raises(tarfile.ReadError, match="file could not be opened"):
            client.download("ticket", str(target))
        assert target.read_bytes() == b"previous cache"
    assert not target.with_name(target.name + ".tmp").exists()
