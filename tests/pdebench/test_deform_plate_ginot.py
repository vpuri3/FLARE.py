from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from pdebench.dataset.ginot.deform_plate import (
    FEAT_DIM,
    FORBIDDEN_CACHE_FIELDS,
    GINOT_CACHE_SAMPLE_FIELDS,
    GINOT_INDEXABLE_COUNT,
    GINOT_TRAIN_COUNT,
    HANDLE,
    INITIAL_WORLD_STEP,
    NORMAL,
    OBSTACLE,
    ONEHOT_DIM,
    TARGET_STEP,
    TRAJECTORY_LENGTH,
    audit_deform_plate_trajectory,
    build_deform_plate_cache,
    build_node_feature_matrix,
    compute_deform_plate_normalizers,
    encode_deform_plate_feats,
    equilibrium_arrays_for_cache,
    one_hot_node_type,
    parse_deform_plate_trajectory,
    resolve_deform_plate_splits,
    resolve_target_step,
    scatter_plate_laplacian_eigenvectors,
)
from pdebench.dataset.ginot.features import active_feats_dim, attach_sample_feats
from pdebench.dataset.ginot.forward import (
    ginot_apply_deform_plate_boundary_conditions,
    ginot_per_graph_free_node_rel_l2,
)
from pdebench.dataset.ginot.graph_cache import ensure_graph_cache, load_graph_cache_sample
from pdebench.dataset.ginot.sample import build_graph_sample_dict
from pdebench.dataset.ginot.types import GINOT_DATASETS, GinotRawDataset, StandardNormalizer


def _make_tet_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    node_type = np.array([NORMAL, NORMAL, HANDLE, NORMAL, OBSTACLE, OBSTACLE, OBSTACLE, OBSTACLE], dtype=np.int64)
    cells = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=np.int64,
    )
    return mesh_pos, node_type, cells


def _make_trajectory(
    *,
    num_states: int = TRAJECTORY_LENGTH,
    plate_shift: float = 0.2,
    actuator_shift: float = 1.0,
) -> dict[str, np.ndarray]:
    mesh_pos, node_type, cells = _make_tet_mesh()
    world = np.repeat(mesh_pos[None, ...], num_states, axis=0)
    for t in range(num_states):
        alpha = float(t) / float(max(num_states - 1, 1))
        world[t, node_type == NORMAL] += np.array([plate_shift * alpha, 0.0, 0.0], dtype=np.float32)
        world[t, node_type == HANDLE] += np.array([plate_shift * alpha, 0.0, 0.0], dtype=np.float32)
        world[t, node_type == OBSTACLE] += np.array([actuator_shift * alpha, 0.0, 0.0], dtype=np.float32)
    return {
        "cells": np.repeat(cells[None, ...], num_states, axis=0),
        "mesh_pos": np.repeat(mesh_pos[None, ...], num_states, axis=0),
        "node_type": np.repeat(node_type[None, :, None], num_states, axis=0),
        "world_pos": world,
        "stress": np.zeros((num_states, mesh_pos.shape[0], 1), dtype=np.float32),
    }


def test_deform_plate_registered_in_ginot_datasets() -> None:
    assert "deform_plate" in GINOT_DATASETS


def test_parse_deform_plate_trajectory_builds_equilibrium_sample() -> None:
    parsed = parse_deform_plate_trajectory(_make_trajectory())
    assert parsed["x0"].shape == (8, 3)
    assert parsed["u_target"].shape == (8, 3)
    assert parsed["u_prescribed"][parsed["free_mask"]].sum() == 0.0
    assert np.allclose(parsed["u_prescribed"][parsed["bc_mask"]], parsed["u_target"][parsed["bc_mask"]])
    assert parsed["plate_cells"].shape == (1, 4)
    assert parsed["plate_nodes"].shape == (4,)
    audit = audit_deform_plate_trajectory(parsed)
    assert audit["num_components"] == 2
    assert audit["normal_count"] == 3


def test_resolve_target_step_defaults_to_360() -> None:
    trajectory = _make_trajectory(plate_shift=0.25, actuator_shift=0.5)
    world_pos = trajectory["world_pos"]
    node_type = trajectory["node_type"][0, :, 0]
    assert resolve_target_step(world_pos, node_type) == TARGET_STEP


def test_u_target_is_world_pos_delta_at_loaded_quasi_static_step() -> None:
    trajectory = _make_trajectory(plate_shift=0.25, actuator_shift=0.5)
    world_pos = trajectory["world_pos"]
    parsed = parse_deform_plate_trajectory(trajectory)
    loaded_step = TARGET_STEP
    expected = world_pos[loaded_step] - world_pos[INITIAL_WORLD_STEP]
    assert np.allclose(parsed["u_target"], expected)
    assert int(parsed["target_step"]) == loaded_step
    assert np.allclose(parsed["x0"], world_pos[INITIAL_WORLD_STEP])
    # mesh_pos is static reference; must not be used for displacement target
    mesh_pos = trajectory["mesh_pos"][0]
    assert not np.allclose(parsed["u_target"], mesh_pos - world_pos[INITIAL_WORLD_STEP])


def test_parse_deform_plate_rejects_mixed_actuator_plate_tets() -> None:
    trajectory = _make_trajectory()
    trajectory["cells"] = np.repeat(
        np.array([[0, 1, 2, 4]], dtype=np.int64)[None, ...],
        TRAJECTORY_LENGTH,
        axis=0,
    )
    with pytest.raises(ValueError, match="connecting the actuator and plate"):
        parse_deform_plate_trajectory(trajectory)


def test_one_hot_node_type_remapped_channels() -> None:
    node_type = np.array([NORMAL, OBSTACLE, HANDLE], dtype=np.int64)
    onehot = one_hot_node_type(node_type)
    assert onehot.shape == (3, 3)
    assert onehot[0, 0] == 1.0
    assert onehot[1, 1] == 1.0
    assert onehot[2, 2] == 1.0


def test_build_node_feature_matrix_shape() -> None:
    node_type = np.array([NORMAL, OBSTACLE], dtype=np.int64)
    u_prescribed = np.zeros((2, 3), dtype=np.float32)
    feats = build_node_feature_matrix(node_type, u_prescribed)
    assert feats.shape == (2, FEAT_DIM)
    assert feats[0, 0] == 1.0
    assert feats[1, 1] == 1.0
    assert np.all(feats[:, 3:] == 0.0)


def test_node_features_exclude_bc_mask() -> None:
    node_type = np.array([NORMAL, OBSTACLE, HANDLE], dtype=np.int64)
    u_prescribed = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    feats = build_node_feature_matrix(node_type, u_prescribed)
    assert feats.shape == (3, FEAT_DIM)
    assert feats.shape[1] == ONEHOT_DIM + 3


def test_masks_recoverable_from_node_type() -> None:
    parsed = parse_deform_plate_trajectory(_make_trajectory())
    node_type = parsed["node_type"]
    free_mask = node_type == NORMAL
    actuator_mask = node_type == OBSTACLE
    handle_mask = node_type == HANDLE
    assert np.array_equal(parsed["free_mask"], free_mask)
    assert np.array_equal(parsed["actuator_mask"], actuator_mask)
    assert np.array_equal(parsed["handle_mask"], handle_mask)
    assert np.array_equal(parsed["bc_mask"], actuator_mask | handle_mask)
    assert np.array_equal(parsed["plate_mask"], free_mask | handle_mask)


def test_u_prescribed_matches_world_pos_delta_on_bc_nodes() -> None:
    trajectory = _make_trajectory(plate_shift=0.25, actuator_shift=0.5)
    world_pos = trajectory["world_pos"]
    parsed = parse_deform_plate_trajectory(trajectory)
    x0 = world_pos[INITIAL_WORLD_STEP]
    loaded_step = int(parsed["target_step"])
    x_loaded = world_pos[loaded_step]
    for mask_name in ("actuator_mask", "handle_mask", "bc_mask"):
        mask = parsed[mask_name]
        np.testing.assert_allclose(
            x_loaded[mask],
            x0[mask] + parsed["u_prescribed"][mask],
        )


def test_resolve_deform_plate_splits_official_train_valid() -> None:
    train, valid = resolve_deform_plate_splits("data", 0)
    assert len(train) == GINOT_TRAIN_COUNT
    assert len(valid) == GINOT_INDEXABLE_COUNT - GINOT_TRAIN_COUNT
    assert train[-1] == GINOT_TRAIN_COUNT - 1
    assert valid[0] == GINOT_TRAIN_COUNT


def test_scatter_plate_laplacian_eigenvectors() -> None:
    plate_nodes = np.array([1, 3, 5], dtype=np.int64)
    plate_eig = torch.ones((3, 4))
    full = scatter_plate_laplacian_eigenvectors(plate_nodes, plate_eig, num_nodes=7)
    assert full.shape == (7, 4)
    assert torch.all(full[plate_nodes] == 1.0)
    assert torch.all(full[[0, 2, 4, 6]] == 0.0)


def test_ginot_per_graph_free_node_rel_l2_ignores_bc_nodes() -> None:
    yh = torch.tensor([[2.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0], [5.0, 5.0], [1.0, 0.0]], dtype=torch.float32)
    batch_index = torch.tensor([0, 0, 1], dtype=torch.long)
    free_mask = torch.tensor([True, False, True], dtype=torch.bool)
    rel = ginot_per_graph_free_node_rel_l2(yh, y, batch_index, num_graphs=2, free_mask=free_mask)
    assert rel.numel() == 2
    assert rel[0].item() == pytest.approx(1.0)
    assert rel[1].item() == pytest.approx(1.0)


def test_ginot_apply_deform_plate_boundary_conditions() -> None:
    yh = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    batch = {
        "flat_bc_mask": torch.tensor([False, True], dtype=torch.bool),
        "flat_u_prescribed": torch.tensor([[0.0, 0.0], [5.0, 6.0]], dtype=torch.float32),
    }
    normalizer = StandardNormalizer(mean=torch.zeros(1, 2), std=torch.ones(1, 2))
    out = ginot_apply_deform_plate_boundary_conditions(yh, batch, y_normalizer=normalizer)
    assert torch.allclose(out[0], yh[0])
    assert torch.allclose(out[1], batch["flat_u_prescribed"][1])


def _make_cached_raw(tmp_path, monkeypatch) -> GinotRawDataset:
    data_root = tmp_path / "data"
    dataset_dir = data_root / "deforming_plate"
    dataset_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "pdebench.dataset.ginot.deform_plate_core.SPLIT_COUNTS",
        {"train": 2, "valid": 1, "test": 1},
    )
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate.SPLIT_COUNTS", {"train": 2, "valid": 1, "test": 1})
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate_core.GINOT_TRAIN_COUNT", 2)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate.GINOT_TRAIN_COUNT", 2)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate_core.GINOT_VALID_OFFSET", 2)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate.GINOT_VALID_OFFSET", 2)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate_core.GINOT_VALID_COUNT", 1)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate.GINOT_VALID_COUNT", 1)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate_core.GINOT_INDEXABLE_COUNT", 3)
    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate.GINOT_INDEXABLE_COUNT", 3)
    meta = {
        "field_names": ["cells", "mesh_pos", "node_type", "world_pos", "stress"],
        "features": {
            "cells": {"dtype": "int32", "shape": [TRAJECTORY_LENGTH, 1, 4]},
            "mesh_pos": {"dtype": "float32", "shape": [TRAJECTORY_LENGTH, 8, 3]},
            "node_type": {"dtype": "int32", "shape": [TRAJECTORY_LENGTH, 8, 1]},
            "world_pos": {"dtype": "float32", "shape": [TRAJECTORY_LENGTH, 8, 3]},
            "stress": {"dtype": "float32", "shape": [TRAJECTORY_LENGTH, 8, 1]},
        },
        "trajectory_length": TRAJECTORY_LENGTH,
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for split_name in ("train", "valid", "test"):
        (dataset_dir / f"{split_name}.tfrecord").write_bytes(b"")

    def _fake_load_tfrecord(path, _meta):
        split_name = str(path).split("/")[-1].replace(".tfrecord", "")
        from pdebench.dataset.ginot import deform_plate_core as deform_plate_mod

        count = deform_plate_mod.SPLIT_COUNTS[split_name]
        return [_make_trajectory() for _ in range(count)]

    monkeypatch.setattr("pdebench.dataset.ginot.deform_plate_core._load_tfrecord", _fake_load_tfrecord)
    from pdebench.dataset.ginot.deform_plate import _cached_store

    _cached_store.cache_clear()
    build_deform_plate_cache(str(data_root))
    from pdebench.dataset.ginot.deform_plate import load_deform_plate

    return load_deform_plate(str(data_root))


def test_equilibrium_cache_excludes_trajectory_fields() -> None:
    parsed = parse_deform_plate_trajectory(_make_trajectory())
    arrays = equilibrium_arrays_for_cache(parsed)
    assert set(arrays) == set(GINOT_CACHE_SAMPLE_FIELDS)
    assert not FORBIDDEN_CACHE_FIELDS.intersection(arrays)
    assert "world_pos" not in arrays
    assert "stress" not in arrays
    for value in arrays.values():
        arr = np.asarray(value)
        assert not (arr.ndim == 3 and arr.shape[0] == TRAJECTORY_LENGTH)


def test_build_deform_plate_cache_and_load(tmp_path, monkeypatch) -> None:
    raw = _make_cached_raw(tmp_path, monkeypatch)
    assert len(raw.query_points) == 3
    assert active_feats_dim(raw) == FEAT_DIM
    pos_normalizer, boundary_pos_normalizer, y_normalizer, scale = compute_deform_plate_normalizers(
        raw, list(range(3))
    )
    assert scale > 0.0
    assert pos_normalizer.std.reshape(-1)[0].item() == pytest.approx(scale)
    sample = {
        "pos": torch.zeros((8, 3)),
    }
    sample = attach_sample_feats(sample, raw, 0, None, y_normalizer=y_normalizer)
    assert sample["feats"].shape == (8, FEAT_DIM)
    assert sample["free_mask"].dtype == torch.bool
    feats = encode_deform_plate_feats(raw, 0, y_normalizer, 8)
    assert feats.shape == (8, FEAT_DIM)
    parsed = raw.deform_plate_store._load(0)
    assert np.allclose(parsed["u_prescribed"][parsed["free_mask"]], 0.0)


def test_attach_sample_feats_skips_cached_deform_plate_boundary_fields(tmp_path, monkeypatch) -> None:
    raw = _make_cached_raw(tmp_path, monkeypatch)
    _, _, y_normalizer, _ = compute_deform_plate_normalizers(raw, [0])
    cached_feats = torch.zeros((8, FEAT_DIM))
    cached = {
        "pos": torch.zeros((8, 3)),
        "feats": cached_feats.clone(),
        "free_mask": torch.tensor([True, False, True, False, True, False, True, False]),
        "bc_mask": torch.tensor([False, True, False, True, False, True, False, True]),
        "u_prescribed": torch.ones((8, 3)),
    }

    def _fail_load(_idx: int) -> None:
        raise AssertionError("NPZ load should be skipped when boundary fields are cached")

    monkeypatch.setattr(raw.deform_plate_store, "_load", _fail_load)
    out = attach_sample_feats(dict(cached), raw, 0, None, y_normalizer=y_normalizer)
    assert torch.equal(out["feats"], cached_feats)
    assert torch.equal(out["free_mask"], cached["free_mask"])
    assert torch.equal(out["bc_mask"], cached["bc_mask"])
    assert torch.equal(out["u_prescribed"], cached["u_prescribed"])


def test_build_graph_sample_dict_bakes_deform_plate_boundary_fields(tmp_path, monkeypatch) -> None:
    raw = _make_cached_raw(tmp_path, monkeypatch)
    pos_norm, boundary_norm, y_norm, _ = compute_deform_plate_normalizers(raw, [0])
    sample = build_graph_sample_dict(
        raw,
        0,
        pos_norm,
        boundary_norm,
        y_norm,
        dataset_name="deform_plate",
        to_cpu=True,
    )
    parsed = raw.deform_plate_store._load(0)
    assert torch.equal(sample["free_mask"], torch.from_numpy(parsed["free_mask"]).bool())
    assert torch.equal(sample["bc_mask"], torch.from_numpy(parsed["bc_mask"]).bool())
    assert torch.equal(sample["u_prescribed"], torch.from_numpy(parsed["u_prescribed"]).float())


def test_graph_cache_bakes_deform_plate_boundary_fields(tmp_path, monkeypatch) -> None:
    raw = _make_cached_raw(tmp_path, monkeypatch)
    pos_norm, boundary_norm, y_norm, _ = compute_deform_plate_normalizers(raw, list(range(3)))
    cache_dir = ensure_graph_cache(
        raw=raw,
        dataset_name="deform_plate",
        split_name="train",
        indices=[0, 1, 2],
        split_seed=0,
        dataset_split="official_t360",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=boundary_norm,
        y_normalizer=y_norm,
        feats_normalizer=None,
    )
    npz_loads = 0

    def _count_load(_idx: int) -> dict:
        nonlocal npz_loads
        npz_loads += 1
        raise AssertionError("attach_sample_feats should not re-read NPZ when graph cache has boundary fields")

    monkeypatch.setattr(raw.deform_plate_store, "_load", _count_load)
    for idx in range(3):
        loaded = load_graph_cache_sample(cache_dir, idx)
        assert "free_mask" in loaded
        assert "bc_mask" in loaded
        assert "u_prescribed" in loaded
        out = attach_sample_feats(loaded, raw, idx, None, y_normalizer=y_norm)
        assert out["feats"].shape == (8, FEAT_DIM)
    assert npz_loads == 0
