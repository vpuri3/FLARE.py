from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch

from pdebench.dataset.ginot.bumper_beam import (
    REQUIRED_TIMES,
    TARGET_FIELDS,
    BumperBeamRun,
    BumperBeamSequence,
    BumperBeamStore,
    discover_bumper_beam_files,
    read_bumper_beam_run,
)
from pdebench.dataset.ginot.types import GinotRawDataset


def _make_run_mesh(*, num_points: int = 4, offset: float = 0.0, mixed_faces: bool = False) -> pv.PolyData:
    if mixed_faces:
        points = np.array(
            [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [3, 0.5, 0]],
            dtype=np.float32,
        ) + np.float32(offset)
        faces = np.array([4, 0, 1, 2, 3, 3, 1, 4, 2], dtype=np.int64)
    else:
        points = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=np.float32)[:num_points] + np.float32(offset)
        faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    mesh = pv.PolyData(points, faces)
    mesh.point_data["thickness"] = np.arange(1, points.shape[0] + 1, dtype=np.float32)
    return mesh


def _attach_time_arrays(mesh: pv.PolyData, *, extra_t110: bool = False) -> None:
    times = [*REQUIRED_TIMES, *([110] if extra_t110 else [])]
    cell_count = mesh.n_cells
    node_count = mesh.n_points
    for time in times:
        suffix = f"t{time}.000"
        mesh.point_data[f"displacement_{suffix}"] = np.full((node_count, 3), time / 10.0, dtype=np.float32)
        mesh.cell_data[f"cell_effective_plastic_strain_{suffix}"] = np.full(cell_count, time + 0.25, dtype=np.float32)
        mesh.cell_data[f"cell_stress_vm_{suffix}"] = np.full(cell_count, time + 0.75, dtype=np.float32)


def _write_run(root: Path, run_id: int, *, extra_t110: bool = False, offset: float = 0.0) -> Path:
    split_dir = root / "CURATED_DATA_VTP" / ("TRAINING_DATA" if run_id % 2 else "VALIDATION_DATA")
    split_dir.mkdir(parents=True, exist_ok=True)
    mesh = _make_run_mesh(offset=offset)
    _attach_time_arrays(mesh, extra_t110=extra_t110)
    path = split_dir / f"Run{run_id}.vtp"
    mesh.save(path)
    return path


@pytest.fixture
def bumper_root(tmp_path: Path) -> Path:
    for run_id in (2, 10, 1):
        _write_run(tmp_path, run_id, extra_t110=run_id == 2, offset=float(run_id))
    globals_by_run = {
        f"Run{run_id}": {"velocity_x": -5.0, "thickness_scale": 1.0 + run_id, "rwall_origin_y": 10.0 * run_id}
        for run_id in (1, 2, 10)
    }
    metadata = tmp_path / "CURATED_DATA_VTP" / "GLOBAL_FEATURES.json"
    metadata.write_text(json.dumps(globals_by_run), encoding="utf-8")
    return tmp_path


def _raw_bumper_beam(bumper_root: Path) -> GinotRawDataset:
    metadata_path, paths = discover_bumper_beam_files(str(bumper_root))
    store = BumperBeamStore(paths, json.loads(metadata_path.read_text(encoding="utf-8")))
    return GinotRawDataset(
        query_points=BumperBeamSequence(store, "reference_points"),
        point_clouds=BumperBeamSequence(store, "reference_points"),
        targets=BumperBeamSequence(store, "targets"),
        cells=BumperBeamSequence(store, "cells"),
        input_params=None,
        target_fields=TARGET_FIELDS,
        space_dim=3,
        normalize_targets=False,
        bumper_beam_store=store,
    )


def test_discovery_naturally_sorts_combined_upstream_folders(bumper_root: Path) -> None:
    metadata_path, paths = discover_bumper_beam_files(str(bumper_root))
    assert metadata_path.name == "GLOBAL_FEATURES.json"
    assert [path.stem for path in paths] == ["Run1", "Run2", "Run10"]


def test_vtp_parser_builds_absolute_positions_and_ignores_t110(bumper_root: Path) -> None:
    run = read_bumper_beam_run(bumper_root / "CURATED_DATA_VTP" / "VALIDATION_DATA" / "Run2.vtp")
    assert run.positions.shape == (11, 4, 3)
    assert run.targets.shape == (4, 50)
    np.testing.assert_allclose(run.positions[0], run.reference_points)
    np.testing.assert_allclose(run.positions[1], run.reference_points + 1.0)
    assert run.cells.shape == (1, 4)
    assert run.thickness.shape == (4, 1)
    assert len(TARGET_FIELDS) == 50
    np.testing.assert_allclose(run.targets[:, 3], 10.25)
    np.testing.assert_allclose(run.targets[:, 4], 10.75)


def test_vtp_parser_supports_mixed_polygon_widths_without_clique_diagonals(tmp_path: Path) -> None:
    from pdebench.dataset.ginot.sample import build_sample_edge_tensors

    split_dir = tmp_path / "CURATED_DATA_VTP" / "TRAINING_DATA"
    split_dir.mkdir(parents=True, exist_ok=True)
    mesh = _make_run_mesh(mixed_faces=True)
    _attach_time_arrays(mesh)
    path = split_dir / "Run7.vtp"
    mesh.save(path)

    run = read_bumper_beam_run(path)
    assert run.cells.shape == (2, 4)
    np.testing.assert_array_equal(run.cells[0], np.array([0, 1, 2, 3], dtype=np.int64))
    np.testing.assert_array_equal(run.cells[1], np.array([1, 4, 2, -1], dtype=np.int64))

    raw = GinotRawDataset(
        query_points=[run.reference_points],
        point_clouds=[run.reference_points],
        targets=[run.targets],
        cells=[run.cells],
        input_params=None,
        target_fields=TARGET_FIELDS,
        space_dim=3,
        normalize_targets=False,
        bumper_beam_store=object(),
    )
    edge_index, edge_attr, cells = build_sample_edge_tensors(
        raw,
        0,
        torch.from_numpy(run.reference_points).float(),
        dataset_name="bumper_beam",
    )
    undirected = {tuple(sorted(edge)) for edge in edge_index.t().tolist()}
    assert undirected == {(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (2, 4)}
    assert (0, 2) not in undirected
    assert (1, 3) not in undirected
    assert edge_attr.shape == (12, 4)
    np.testing.assert_array_equal(cells, run.cells)


def test_vtp_parser_rejects_non_finite_required_arrays(tmp_path: Path) -> None:
    path = _write_run(tmp_path, 3)
    mesh = pv.read(path)
    thickness = np.asarray(mesh.point_data["thickness"], dtype=np.float32).copy()
    thickness[0] = np.nan
    mesh.point_data["thickness"] = thickness
    mesh.save(path)

    with pytest.raises(ValueError, match=r"Run3\.vtp: non-finite values in required bumper-beam arrays"):
        read_bumper_beam_run(path)


def test_store_loads_runs_lazily_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pdebench.dataset.ginot.bumper_beam as bumper_beam

    path = tmp_path / "Run1.vtp"
    path.touch()
    calls: list[Path] = []
    fake_run = BumperBeamRun(
        reference_points=np.zeros((1, 3), dtype=np.float32),
        positions=np.zeros((len(REQUIRED_TIMES), 1, 3), dtype=np.float32),
        thickness=np.ones((1, 1), dtype=np.float32),
        global_features=None,
        cells=np.zeros((1, 4), dtype=np.int64),
        targets=np.zeros((1, len(TARGET_FIELDS)), dtype=np.float32),
    )

    def _fake_read(arg: Path) -> BumperBeamRun:
        calls.append(arg)
        return fake_run

    monkeypatch.setattr(bumper_beam, "read_bumper_beam_run", _fake_read)
    store = BumperBeamStore(
        [path],
        {"Run1": {"velocity_x": -5.0, "thickness_scale": 2.0, "rwall_origin_y": 10.0}},
    )

    targets_seq = BumperBeamSequence(store, "targets")
    points_seq = BumperBeamSequence(store, "reference_points")

    np.testing.assert_allclose(targets_seq[0], fake_run.targets)
    np.testing.assert_allclose(points_seq[0], fake_run.reference_points)
    np.testing.assert_allclose(store.load(0).global_features, np.array([-5.0, 2.0, 10.0], dtype=np.float32))
    assert calls == [path]


def test_store_caches_decoded_runs_across_indices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Training shuffles all runs; a tiny LRU would re-parse VTPs every step."""
    import pdebench.dataset.ginot.bumper_beam as bumper_beam

    paths = [tmp_path / f"Run{i}.vtp" for i in (1, 2, 3)]
    fake_run = BumperBeamRun(
        reference_points=np.zeros((1, 3), dtype=np.float32),
        positions=np.zeros((1, 1, 3), dtype=np.float32),
        thickness=np.zeros((1, 1), dtype=np.float32),
        global_features=None,
        cells=np.zeros((1, 4), dtype=np.int64),
        targets=np.zeros((1, len(TARGET_FIELDS)), dtype=np.float32),
    )
    calls: list[Path] = []

    def _fake_read(arg: Path) -> BumperBeamRun:
        calls.append(arg)
        return fake_run

    monkeypatch.setattr(bumper_beam, "read_bumper_beam_run", _fake_read)
    meta = {
        path.stem: {"velocity_x": -5.0, "thickness_scale": 2.0, "rwall_origin_y": 10.0} for path in paths
    }
    store = BumperBeamStore(paths, meta)
    for idx in (0, 1, 2, 0, 1, 2, 1):
        store.load(idx)
    assert calls == paths
    store.preload()
    assert calls == paths


def test_preload_bumper_beam_store_is_noop_without_store() -> None:
    from pdebench.dataset.ginot.bumper_beam import preload_bumper_beam_store
    from pdebench.dataset.ginot.types import GinotRawDataset

    preload_bumper_beam_store(None)
    raw = GinotRawDataset(
        dataset_dir="/tmp",
        query_points=[np.zeros((1, 3), dtype=np.float32)],
        point_clouds=[np.zeros((1, 3), dtype=np.float32)],
        targets=[np.zeros((1, 1), dtype=np.float32)],
        cells=None,
        input_params=None,
        target_fields=("y",),
        space_dim=3,
        bumper_beam_store=None,
    )
    preload_bumper_beam_store(raw)


def test_loader_preloads_bumper_store_only_for_raw_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No-edge training needs a warm VTP cache; warm edge LMDB startups must not."""
    from pdebench.dataset.ginot import bumper_beam
    from pdebench.dataset.ginot import loader as ginot_loader
    from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer

    calls: list[str] = []

    class _Store:
        def preload(self) -> None:
            calls.append("preload")

    def _fake_raw(_name: str, _root: str) -> GinotRawDataset:
        return GinotRawDataset(
            dataset_dir=str(tmp_path),
            query_points=[np.zeros((4, 3), dtype=np.float32) for _ in range(4)],
            point_clouds=[np.zeros((4, 3), dtype=np.float32) for _ in range(4)],
            targets=[np.zeros((4, 50), dtype=np.float32) for _ in range(4)],
            cells=None,
            input_params=None,
            target_fields=TARGET_FIELDS,
            space_dim=3,
            normalize_targets=False,
            bumper_beam_store=_Store(),
        )

    identity = StandardNormalizer(torch.zeros(1, 3), torch.ones(1, 3))
    y_norm = StandardNormalizer(torch.zeros(1, 50), torch.ones(1, 50))
    feats_norm = StandardNormalizer(torch.zeros(1, 4), torch.ones(1, 4))

    def _preload(raw) -> None:
        calls.append("loader")
        bumper_beam.preload_bumper_beam_store(raw)

    monkeypatch.setattr(bumper_beam, "EXPECTED_RUNS", 4)
    monkeypatch.setattr("pdebench.dataset.ginot._load_raw_dataset", _fake_raw)
    monkeypatch.setattr(ginot_loader, "_split_train_test", lambda *_a, **_k: ([0, 1, 2], [3]))
    monkeypatch.setattr(
        ginot_loader,
        "_resolve_ginot_normalizers",
        lambda **_k: (identity, identity, y_norm, feats_norm),
    )
    monkeypatch.setattr(ginot_loader, "preload_bumper_beam_store", _preload)

    ginot_loader.load_ginot_dataset("bumper_beam", str(tmp_path), include_edges=False, build_collate_metadata=False)
    assert calls == ["loader", "preload"]

    calls.clear()
    monkeypatch.setattr(
        ginot_loader,
        "open_graph_cache_split",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop-before-lmdb")),
    )
    monkeypatch.setattr(
        ginot_loader,
        "_resolve_graph_cache_dir",
        lambda **_k: str(tmp_path / "graph_cache"),
    )
    with pytest.raises(RuntimeError, match="stop-before-lmdb"):
        ginot_loader.load_ginot_dataset("bumper_beam", str(tmp_path), include_edges=True, build_collate_metadata=False)
    assert calls == []


def test_graph_cache_build_preloads_bumper_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pdebench.dataset.ginot import graph_cache
    from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer

    calls: list[str] = []

    class _Store:
        def preload(self) -> None:
            calls.append("preload")

    raw = GinotRawDataset(
        dataset_dir=str(tmp_path),
        query_points=[np.zeros((2, 3), dtype=np.float32)],
        point_clouds=[np.zeros((2, 3), dtype=np.float32)],
        targets=[np.zeros((2, 1), dtype=np.float32)],
        cells=None,
        input_params=None,
        target_fields=("y",),
        space_dim=3,
        bumper_beam_store=_Store(),
    )
    identity = StandardNormalizer(torch.zeros(1, 3), torch.ones(1, 3))
    y_norm = StandardNormalizer(torch.zeros(1, 1), torch.ones(1, 1))

    def _warm(raw_arg) -> None:
        calls.append("warm")
        raw_arg.bumper_beam_store.preload()

    monkeypatch.setattr(graph_cache, "distributed_rank", lambda: 0)
    monkeypatch.setattr(graph_cache, "distributed_barrier", lambda: None)
    monkeypatch.setattr(graph_cache, "graph_cache_dir", lambda **_k: str(tmp_path / "cache"))
    monkeypatch.setattr(graph_cache, "graph_cache_num_shards", lambda *_a, **_k: 1)
    monkeypatch.setattr(graph_cache, "_load_sample_ids", lambda *_a, **_k: set())
    monkeypatch.setattr(graph_cache, "preload_bumper_beam_store", _warm)
    monkeypatch.setattr(
        graph_cache,
        "_write_sharded_lmdb_graph_cache",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("wrote")),
    )

    with pytest.raises(RuntimeError, match="wrote"):
        graph_cache.ensure_graph_cache(
            raw=raw,
            dataset_name="bumper_beam",
            split_name="train",
            indices=[0],
            split_seed=0,
            dataset_split="tag",
            pos_normalizer=identity,
            boundary_pos_normalizer=identity,
            y_normalizer=y_norm,
        )
    assert calls == ["warm", "preload"]


def test_store_qualifies_missing_run_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pdebench.dataset.ginot.bumper_beam as bumper_beam

    path = tmp_path / "Run21.vtp"
    monkeypatch.setattr(bumper_beam, "read_bumper_beam_run", lambda _: object())
    store = BumperBeamStore([path], {})

    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}: missing global metadata key 'Run21'"):
        store.load(0)


def test_store_qualifies_missing_global_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pdebench.dataset.ginot.bumper_beam as bumper_beam

    path = tmp_path / "Run22.vtp"
    monkeypatch.setattr(bumper_beam, "read_bumper_beam_run", lambda _: object())
    metadata = {"Run22": {"velocity_x": -5.0, "rwall_origin_y": 10.0}}
    store = BumperBeamStore([path], metadata)

    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}: missing global metadata key 'thickness_scale'"):
        store.load(0)


def test_bumper_normalization_uses_train_states_and_preserves_raw_channels(bumper_root: Path) -> None:
    from pdebench.dataset.ginot.bumper_beam import (
        compute_bumper_beam_normalizers,
        encode_bumper_beam_feats,
        encode_bumper_beam_target,
    )
    from pdebench.dataset.ginot.features import active_feats_dim, encode_sample_feats
    from pdebench.dataset.ginot.sample import encode_raw_sample

    raw = _raw_bumper_beam(bumper_root)
    pos_norm, boundary_norm, y_norm, feats_norm = compute_bumper_beam_normalizers(raw, [0, 1])
    assert boundary_norm is pos_norm
    assert y_norm.mean.shape == (1, 50)
    train_positions = np.concatenate([raw.bumper_beam_store.load(idx).positions for idx in (0, 1)], axis=1)
    train_min = train_positions.min(axis=(0, 1), keepdims=False).reshape(1, 3)
    train_max = train_positions.max(axis=(0, 1), keepdims=False).reshape(1, 3)
    torch.testing.assert_close(pos_norm.mean, torch.from_numpy(train_min))
    torch.testing.assert_close(
        pos_norm.std,
        torch.from_numpy(train_max - train_min),
    )

    pos, _, y = encode_raw_sample(raw, 0, pos_norm, boundary_norm, y_norm)
    assert pos.shape == (4, 3)
    assert y.shape == (4, 50)
    torch.testing.assert_close(y, encode_bumper_beam_target(raw, 0, pos_norm))
    raw_y = raw.bumper_beam_store.load(0).targets
    for base in range(0, 50, 5):
        assert torch.all(y[:, base : base + 3] >= 0)
        assert torch.all(y[:, base : base + 3] <= 1)
        torch.testing.assert_close(y[:, base + 3 : base + 5], torch.from_numpy(raw_y[:, base + 3 : base + 5]))

    feats = encode_bumper_beam_feats(raw, 0, feats_norm, 4)
    assert active_feats_dim(raw) == 4
    torch.testing.assert_close(encode_sample_feats(raw, 0, feats_norm, 4), feats)
    assert feats.shape == (4, 4)
    assert torch.isfinite(feats).all()
    run = raw.bumper_beam_store.load(0)
    expected = torch.from_numpy(
        np.concatenate((run.thickness, np.broadcast_to(run.global_features.reshape(1, 3), (4, 3))), axis=1).copy()
    )
    torch.testing.assert_close(feats_norm.decode(feats), expected)
    assert torch.all(feats[:, 1] == 0), "constant velocity_x must normalize to zero"


def test_bumper_target_coordinates_are_not_clipped_to_training_range(bumper_root: Path) -> None:
    from pdebench.dataset.ginot.bumper_beam import compute_bumper_beam_normalizers, encode_bumper_beam_target

    raw = _raw_bumper_beam(bumper_root)
    pos_norm, _, _, _ = compute_bumper_beam_normalizers(raw, [0])
    encoded = encode_bumper_beam_target(raw, 2, pos_norm)
    assert torch.any(encoded[:, 0::5] > 1.0)


def test_decode_bumper_beam_target_restores_physical_positions(bumper_root: Path) -> None:
    from pdebench.dataset.ginot.bumper_beam import (
        compute_bumper_beam_normalizers,
        decode_bumper_beam_target,
        encode_bumper_beam_target,
    )

    raw = _raw_bumper_beam(bumper_root)
    pos_norm, _, _, _ = compute_bumper_beam_normalizers(raw, [0, 1])
    physical = torch.from_numpy(raw.bumper_beam_store.load(1).targets.copy()).float()
    encoded = encode_bumper_beam_target(raw, 1, pos_norm)
    decoded = decode_bumper_beam_target(encoded, pos_norm)
    torch.testing.assert_close(decoded[:, 0:3], physical[:, 0:3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded[:, 3:5], physical[:, 3:5], atol=0.0, rtol=0.0)


def test_bumper_quad_uses_perimeter_edges_without_diagonals(bumper_root: Path) -> None:
    from pdebench.dataset.ginot.bumper_beam import compute_bumper_beam_normalizers
    from pdebench.dataset.ginot.sample import build_sample_edge_tensors

    raw = _raw_bumper_beam(bumper_root)
    pos_norm, _, _, _ = compute_bumper_beam_normalizers(raw, [0, 1])
    pos = pos_norm.encode(torch.from_numpy(raw.query_points[0]).float())
    edge_index, edge_attr, cells = build_sample_edge_tensors(raw, 0, pos, dataset_name="bumper_beam")
    undirected = {tuple(sorted(edge)) for edge in edge_index.t().tolist()}
    assert undirected == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert edge_index.shape == (2, 8)
    assert edge_attr.shape == (8, 4)
    assert cells.shape == (1, 4)


def test_public_loader_exposes_n7_input_and_n50_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pdebench.dataset.ginot import bumper_beam
    from pdebench.dataset.ginot.loader import load_ginot_dataset

    global_features = {}
    for run_id in range(1, 11):
        _write_run(tmp_path, run_id, offset=float(run_id))
        global_features[f"Run{run_id}"] = {
            "velocity_x": -5.0,
            "thickness_scale": 1.0 + run_id,
            "rwall_origin_y": 10.0 * run_id,
        }
    metadata_path = tmp_path / "CURATED_DATA_VTP" / "GLOBAL_FEATURES.json"
    metadata_path.write_text(json.dumps(global_features), encoding="utf-8")
    monkeypatch.setattr(bumper_beam, "EXPECTED_RUNS", 10)

    train, test, metadata = load_ginot_dataset(
        "BUMPER_BEAM", str(tmp_path), split_seed=0, include_edges=False
    )

    assert (len(train), len(test)) == (8, 2)
    assert metadata["dataset"] == "bumper_beam"
    assert {key: metadata[key] for key in ("c_in", "c_out", "pos_dim", "feats_dim")} == {
        "c_in": 7,
        "c_out": 50,
        "pos_dim": 3,
        "feats_dim": 4,
    }
    assert metadata["mesh_split_note"] == (
        "bumper_beam combines all 10 curated runs and uses an 80/20 split "
        "with seed=0: 8 train samples and 2 test samples."
    )
    assert metadata["ginot_train_mean_nodes_per_run"] is None
    assert metadata["ginot_train_mean_nodes_per_batch_bs1"] is None
    sample = train[0]
    assert sample["pos"].shape == (4, 3)
    assert sample["feats"].shape == (4, 4)
    assert sample["y"].shape == (4, 50)


def test_graph_cache_loader_reports_bumper_beam_train_mean_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pdebench.dataset.ginot import bumper_beam
    from pdebench.dataset.ginot.loader import load_ginot_dataset

    global_features = {}
    for run_id in range(1, 11):
        _write_run(tmp_path, run_id, offset=float(run_id))
        global_features[f"Run{run_id}"] = {
            "velocity_x": -5.0,
            "thickness_scale": 1.0 + run_id,
            "rwall_origin_y": 10.0 * run_id,
        }
    metadata_path = tmp_path / "CURATED_DATA_VTP" / "GLOBAL_FEATURES.json"
    metadata_path.write_text(json.dumps(global_features), encoding="utf-8")
    monkeypatch.setattr(bumper_beam, "EXPECTED_RUNS", 10)

    train, test, metadata = load_ginot_dataset(
        "bumper_beam", str(tmp_path), split_seed=0, include_edges=True
    )

    assert (len(train), len(test)) == (8, 2)
    assert metadata["ginot_train_mean_nodes_per_run"] == pytest.approx(4.0)
    assert metadata["ginot_train_mean_nodes_per_batch_bs1"] == pytest.approx(4.0)
    assert "bumper_beam_curated_vtp_v1_n7_y50_80_20_seed0" in metadata["ginot_graph_cache_dir"]
    sample = train[0]
    assert sample["pos"].shape == (4, 3)
    assert sample["feats"].shape == (4, 4)
    assert sample["y"].shape == (4, 50)
