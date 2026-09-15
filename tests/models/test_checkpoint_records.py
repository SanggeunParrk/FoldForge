"""Checkpoint serialization and concatenated binary stream contracts."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch
import zstandard

from foldforge.models.checkpoints.haiku import (
    RecordError,
    encode_record,
    open_for_reading,
    read_records,
    select_model_files,
)


@pytest.mark.parametrize("compressed", [False, True])
def test_records_cross_empty_and_nonempty_shards(tmp_path: Path, *, compressed: bool):
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    payload = encode_record("scope/one", "weights", expected)
    payload += encode_record("scope/two", "bias", expected[0, 0])
    if compressed:
        payload = zstandard.ZstdCompressor().compress(payload)
    chunks = [b"", payload[:7], b"", payload[7:23], payload[23:], b""]
    paths = []
    for index, chunk in enumerate(chunks):
        path = tmp_path / f"shard{index}.bin"
        path.write_bytes(chunk)
        paths.append(path)
    with open_for_reading(paths, is_compressed=compressed) as stream:
        records = list(read_records(stream))
    assert [(scope, name) for scope, name, _ in records] == [
        ("scope/one", "weights"),
        ("scope/two", "bias"),
    ]
    torch.testing.assert_close(records[0][2], torch.from_numpy(expected))
    torch.testing.assert_close(records[1][2], torch.from_numpy(expected[0, 0]))


def test_seek_end_and_eof_across_shards(tmp_path: Path):
    # Obtain the actual stream through its public context manager.
    paths = [tmp_path / str(index) for index in range(3)]
    for path, value in zip(paths, [b"abc", b"", b"def"], strict=True):
        path.write_bytes(value)
    with open_for_reading(paths, is_compressed=False) as stream:
        assert isinstance(stream, io.BufferedReader)
        assert stream.seek(-2, io.SEEK_END) == 4
        assert stream.read() == b"ef"
        assert stream.read(2) == b""
        assert stream.seek(1) == 1
        assert stream.read(4) == b"bcde"
        assert stream.seek(2, io.SEEK_CUR) == 7
        assert stream.read() == b""
        with pytest.raises(ValueError, match="negative"):
            stream.seek(-1)
    assert stream.closed
    with pytest.raises(ValueError, match="closed"):
        stream.read(1)


@pytest.mark.parametrize("cut", [1, 19, 21, -1])
def test_truncated_record_is_rejected(cut: int):
    payload = encode_record("s", "w", np.ones(3, dtype=np.float32))
    with pytest.raises(RecordError, match="Incomplete"):
        list(read_records(io.BytesIO(payload[:cut])))


def test_numeric_shard_selection(tmp_path: Path):
    for index in (10, 2, 1):
        (tmp_path / f"model.bin.{index}").touch()
    paths, compressed = select_model_files(tmp_path, model_name="model")
    assert [path.name for path in paths] == [
        "model.bin.1",
        "model.bin.2",
        "model.bin.10",
    ]
    assert not compressed


@pytest.mark.parametrize("shape", [(3, 2), (4, 3, 2), (2, 2, 3, 2)])
def test_assign_transposes_single_and_stacked_weights(shape):
    from foldforge.models.checkpoints.haiku import Param, ParamType, assign

    source = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    count = int(np.prod(shape[:-2]))
    targets = [torch.empty(2, 3) for _ in range(count)]
    parameter = Param(
        param=targets[0] if len(shape) == 2 else targets,
        param_type=ParamType.linear_weight,
        stacked=len(shape) > 2,
    )
    assign({"projection": parameter}, {"projection": source})
    expected = source.reshape(-1, 3, 2).transpose(-1, -2)
    for actual, weight in zip(targets, expected, strict=True):
        torch.testing.assert_close(actual, weight, atol=0, rtol=0)


def test_assign_rejects_wrong_stack_count_and_shape():
    from foldforge.models.checkpoints.haiku import Param, assign

    parameter = Param(param=[torch.zeros(3), torch.zeros(3)], stacked=True)
    with pytest.raises(ValueError, match=r"projection.*count mismatch"):
        assign({"projection": parameter}, {"projection": torch.ones(3, 3)})
    with pytest.raises(ValueError, match=r"projection.*!="):
        assign({"projection": Param(torch.zeros(3))}, {"projection": torch.ones(4)})
