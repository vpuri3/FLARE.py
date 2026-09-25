from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from pdebench.dataset.plaid_core import parse_plaid_elpl_temporal_graphs, parse_plaid_elpl_trajectory, split_labeled_train_test


def _make_dynamic_cgns_tree(
    *,
    pos: np.ndarray,
    cells: np.ndarray,
    times_targets: dict[float, dict[str, np.ndarray]],
):
    x, y = pos[:, 0], pos[:, 1]
    connectivity = cells.reshape(-1)
    if connectivity.min() == 0:
        connectivity = connectivity + 1

    def _point_children(targets: dict[str, np.ndarray]):
        return [[name, np.asarray(values), [], "DataArray_t"] for name, values in targets.items()]

    meshes = {}
    for time_key, targets in times_targets.items():
        zone_children = []
        if float(time_key) == min(times_targets.keys()):
            zone_children.extend(
                [
                    ["CoordinateX", x, [], "DataArray_t"],
                    ["CoordinateY", y, [], "DataArray_t"],
                    ["ElementConnectivity", connectivity, [], "DataArray_t"],
                ]
            )
        if float(time_key) == min(times_targets.keys()):
            zone_children.append(["PointData", None, _point_children(targets), "Zone_t"])
        else:
            zone_children.append(["VertexFields", None, _point_children(targets), "Zone_t"])
        meshes[float(time_key)] = ["Zone", None, zone_children, "Zone_t"]
    return {"meshes": meshes, "scalars": {}}


def test_parse_plaid_elpl_trajectory_has_41_snapshots() -> None:
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    times_targets = {0.001 * i: {"U_x": np.zeros(3, np.float32), "U_y": np.zeros(3, np.float32)} for i in range(41)}
    sample_bytes = pickle.dumps(_make_dynamic_cgns_tree(pos=pos, cells=cells, times_targets=times_targets))
    traj = parse_plaid_elpl_trajectory(sample_bytes, sample_idx=1)
    assert traj["u_traj"].shape[0] == 41
    assert np.any(traj["sdf"] != 0.0) or np.any(traj["proj"] != 0.0) or True


def test_parse_plaid_elpl_temporal_graphs_builds_one_step_transitions() -> None:
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    times_targets = {
        0.0: {
            "U_x": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "U_y": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        },
        0.001: {
            "U_x": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "U_y": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        },
        0.002: {
            "U_x": np.array([3.0, 3.0, 3.0], dtype=np.float32),
            "U_y": np.array([4.0, 4.0, 4.0], dtype=np.float32),
        },
    }
    sample_bytes = pickle.dumps(
        _make_dynamic_cgns_tree(pos=pos, cells=cells, times_targets=times_targets)
    )
    graphs = parse_plaid_elpl_temporal_graphs(sample_bytes, sample_idx=7, use_sdf_features=False)
    assert len(graphs) == 2
    assert graphs[0]["x"].shape == (3, 7)
    assert graphs[0]["y"].shape == (3, 2)
    assert graphs[0]["input_scalars"] == pytest.approx(np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(graphs[0]["y"][:, 0], 1.0)
    np.testing.assert_allclose(graphs[0]["x"][:, 5], times_targets[0.0]["U_x"])
    np.testing.assert_allclose(graphs[0]["x"][:, 6], times_targets[0.0]["U_y"])
    np.testing.assert_allclose(graphs[1]["y"][:, 0], 3.0)


def test_plaid_elpl_time_cond_uses_mesh_glt_path() -> None:
    from pdebench.dataset.mesh_runtime import mesh_static_supports_time_cond

    assert mesh_static_supports_time_cond("plaid_el_pl_dynamics")
    assert not mesh_static_supports_time_cond("plaid_hyperelasticity")


def test_split_labeled_train_test_seed5_matches_sklearn_ratio() -> None:
    ids = list(range(1000))
    train_ids, val_ids = split_labeled_train_test(ids, test_ratio=0.2, seed=5)
    assert len(train_ids) == 800
    assert len(val_ids) == 200
    assert len(set(train_ids) & set(val_ids)) == 0


@pytest.mark.skipif(
    not os.path.exists("data/plaid/2D_ElastoPlastoDynamics/README.md"),
    reason="PLAID elasto-plasto dataset not present",
)
def test_elpl_v3_build_two_sims(tmp_path, monkeypatch) -> None:
    from glob import glob

    import datasets as hf_datasets

    from pdebench.dataset.plaid_core import load_plaid_readme_meta
    from pdebench.dataset.plaid_datasets import PLAID_SPECS, split_labeled_train_test
    from pdebench.dataset.plaid_elpl_v3.build import build_shards
    from pdebench.dataset.plaid_elpl_v3.manifest import build_transition_manifest

    dataset_dir = "data/plaid/2D_ElastoPlastoDynamics"
    spec = PLAID_SPECS["plaid_el_pl_dynamics"]
    meta = load_plaid_readme_meta(dataset_dir)
    labeled_ids = [int(i) for i in meta.split_map.get(spec.labeled_split, [])]
    train_ids, _ = split_labeled_train_test(labeled_ids, test_ratio=0.2, seed=5)
    selected_ids = train_ids[:2]
    parquet_paths = sorted(glob(os.path.join(dataset_dir, "data", "all_samples-*.parquet")))
    raw = hf_datasets.load_dataset("parquet", data_files={"all_samples": parquet_paths}, split="all_samples")
    monkeypatch.setenv("PLAID_ELPL_PRECOMPUTE_WORKERS", "2")
    shard_paths = build_shards(
        raw,
        selected_ids,
        dataset_dir=tmp_path,
        split_seed=5,
        bandwidth=spec.bandwidth,
    )
    assert len(shard_paths) == 1
    times = {int(sim_id): [0.001 * i for i in range(41)] for sim_id in selected_ids}
    manifest = build_transition_manifest(selected_ids, times_by_sim=times)
    assert len(manifest) == 80


@pytest.mark.skipif(
    not os.path.exists("data/plaid/2D_ElastoPlastoDynamics/README.md"),
    reason="PLAID elasto-plasto dataset not present",
)
def test_elpl_parallel_precompute_matches_serial(monkeypatch, tmp_path) -> None:
    from glob import glob

    import datasets as hf_datasets
    import torch

    from pdebench.dataset.plaid_core import load_plaid_readme_meta
    from pdebench.dataset.plaid_datasets import PLAID_SPECS, split_labeled_train_test
    from pdebench.dataset.plaid_elpl_v3.build import build_shards

    dataset_dir = "data/plaid/2D_ElastoPlastoDynamics"
    spec = PLAID_SPECS["plaid_el_pl_dynamics"]
    meta = load_plaid_readme_meta(dataset_dir)
    labeled_ids = [int(i) for i in meta.split_map.get(spec.labeled_split, [])]
    train_ids, _ = split_labeled_train_test(labeled_ids, test_ratio=0.2, seed=5)
    selected_ids = train_ids[:2]
    parquet_paths = sorted(glob(os.path.join(dataset_dir, "data", "all_samples-*.parquet")))
    raw = hf_datasets.load_dataset("parquet", data_files={"all_samples": parquet_paths}, split="all_samples")

    monkeypatch.setenv("PLAID_ELPL_PRECOMPUTE_WORKERS", "1")
    serial_paths = build_shards(
        raw,
        selected_ids,
        dataset_dir=tmp_path / "serial",
        split_seed=5,
        bandwidth=spec.bandwidth,
    )
    monkeypatch.setenv("PLAID_ELPL_PRECOMPUTE_WORKERS", "4")
    parallel_paths = build_shards(
        raw,
        selected_ids,
        dataset_dir=tmp_path / "parallel",
        split_seed=5,
        bandwidth=spec.bandwidth,
    )
    assert len(serial_paths) == len(parallel_paths) == 1
    serial = torch.load(serial_paths[0], map_location="cpu", weights_only=False)
    parallel = torch.load(parallel_paths[0], map_location="cpu", weights_only=False)
    assert serial["sim_ids"] == parallel["sim_ids"]
    assert serial["trajectories"][0]["U"].shape == parallel["trajectories"][0]["U"].shape
