"""Shared utility contracts that differed between the original consumers."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any
from unittest.mock import patch

import lmdb
import numpy as np
import pytest
import torch

from foldforge.data import io
from foldforge.utils import distributed, geometry, tensor
from foldforge.utils.seed import seed_everything


@pytest.mark.parametrize("before_rotation", [False, True])
@pytest.mark.parametrize("centralize", [False, True])
def test_coordinate_transform_preserves_translation_frame(before_rotation, centralize):
    points = np.array([[2.0, 0.0, 0.0], [4.0, 2.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([1.0, 2.0, 3.0])

    class FixedRotation:
        @staticmethod
        def random() -> Any:
            class R:
                def as_matrix(self) -> np.ndarray:
                    return rotation

            return R()

    original = points.copy()
    with (
        patch.object(geometry, "Rotation", FixedRotation),
        patch.object(geometry.np.random, "uniform", return_value=translation),
    ):
        result = geometry.random_transform(
            points,
            apply_augmentation=True,
            centralize=centralize,
            translation_before_rotation=before_rotation,
        )
    centered = points - points.mean(0) if centralize else points
    expected = (
        (centered + translation) @ rotation.T
        if before_rotation
        else centered @ rotation.T + translation
    )
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_angle_regularization_and_degenerate_convention():
    zero = np.zeros(3)
    assert geometry.angle_3p(zero, zero, zero) == 0
    assert geometry.angle_3p(zero, zero, zero, eps=1e-4) == 90
    assert geometry.angle_3p([0, 0, 0], [1, 0, 0], [2, 0, 0]) == 0
    expected = np.degrees(np.arccos(1 / 1.0001))
    assert geometry.angle_3p([0, 0, 0], [1, 0, 0], [2, 0, 0], eps=1e-4) == expected


def test_distance_modes_are_explicit():
    points = torch.tensor([[10000.0, 10000.0], [10000.25, 10000.0]])
    with patch.object(torch, "cdist", wraps=torch.cdist) as called:
        result = tensor.cdist(points, compute_mode="donot_use_mm_for_euclid_dist")
        assert called.call_args.kwargs["compute_mode"] == "donot_use_mm_for_euclid_dist"
        torch.testing.assert_close(result, torch.tensor([[0.0, 0.25], [0.25, 0.0]]))
        tensor.cdist(points)
        assert (
            called.call_args.kwargs["compute_mode"]
            == "use_mm_for_euclid_dist_if_necessary"
        )


def test_export_and_device_transfer_do_not_mutate_containers(tmp_path):
    values = {"nested": {"x": torch.tensor([1.5], dtype=torch.bfloat16)}, "name": "x"}
    original = copy.deepcopy(values)
    moved = tensor.to_device(values, "cpu")
    converted = tensor.map_values_to_list(values)
    assert moved is not values
    assert moved["nested"] is not values["nested"]
    assert converted == {"nested": {"x": [1.5]}, "name": "x"}
    io.save_json(values, tmp_path / "values.json")
    io.save_tensor(values, tmp_path / "values.pt")
    assert isinstance(values["nested"]["x"], torch.Tensor)
    torch.testing.assert_close(values["nested"]["x"], original["nested"]["x"])
    assert json.loads((tmp_path / "values.json").read_text()) == converted


def test_shared_json_pickle_and_hashed_lmdb_readers(tmp_path):
    path = tmp_path / "rows.lmdb"
    with (
        lmdb.open(str(path), subdir=False, map_size=1048576) as env,
        env.begin(write=True) as txn,
    ):
        txn.put(
            hashlib.sha1(b"sequence", usedforsecurity=False).hexdigest().encode(),
            b"ACDE",
        )
    reader = io.load_json_cached(path)
    try:
        assert reader["sequence"] == "ACDE"
        assert "sequence" in reader
        assert reader.get("missing", "absent") == "absent"
    finally:
        reader.close()
    payload = {"nested": [1, 2, 3]}
    io.dump_gzip_pickle(payload, tmp_path / "data.pkl.gz")
    assert io.load_gzip_pickle(tmp_path / "data.pkl.gz") == payload
    (tmp_path / "data.json").write_text(json.dumps(payload))
    assert io.load_json_cached(tmp_path / "data.json") == payload


def test_seed_reproducibility_and_determinism_reset():
    prior = torch.are_deterministic_algorithms_enabled()
    prior_cudnn = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    import os

    config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        seed_everything(5, deterministic=True)
        first = (random.random(), np.random.random(), torch.rand(3))
        seed_everything(5, deterministic=False)
        second = (random.random(), np.random.random(), torch.rand(3))
        assert first[:2] == second[:2]
        torch.testing.assert_close(first[2], second[2])
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(prior)
        torch.backends.cudnn.deterministic = prior_cudnn
        torch.backends.cudnn.benchmark = benchmark
        if config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = config


def test_distributed_gather_uses_initialized_group_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    wrapper = distributed.DistWrapper()

    def gather(dest, obj, group=None) -> list[dict[str, Any]]:  # noqa: ARG001 - shared callback or fixture signature
        assert len(dest) == 2
        dest[:] = [obj, {"score": 2}]

    with (
        patch.object(distributed, "distributed_available", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
    ):
        assert wrapper.all_gather_object({"score": 1}) == [{"score": 1}, {"score": 2}]
