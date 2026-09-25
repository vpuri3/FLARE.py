from __future__ import annotations

import numpy as np
import torch

from pdebench.dataset.ginot.types import GinotRawDataset
from pdebench.dataset.laplacian.precompute import (
    compute_or_reuse_graph_payload,
    group_sample_ids_by_topology,
)


def _mock_poisson_structured_raw(num_samples: int = 5) -> GinotRawDataset:
    cells0 = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    cells = [cells0.copy() for _ in range(num_samples)]
    query_points = [
        np.random.default_rng(i).random((4, 2), dtype=np.float32) for i in range(num_samples)
    ]
    return GinotRawDataset(
        dataset_dir="/tmp/mock_poisson_structured",
        query_points=query_points,
        point_clouds=query_points,
        targets=[np.zeros((4, 1), dtype=np.float32) for _ in range(num_samples)],
        cells=cells,
        input_params=None,
        target_fields=("u",),
        space_dim=2,
    )


def _mock_micro_puc_raw(num_samples: int, num_meshes: int = 3) -> GinotRawDataset:
    mesh_idx = np.array([i % num_meshes for i in range(num_samples)], dtype=np.int32)
    cells = []
    for mesh_id in range(num_meshes):
        n = 4 + mesh_id
        cells.append(
            np.asarray(
                [[0, 1, 2], [0, 2, 3]] + ([[1, 2, 3]] if n > 4 else []),
                dtype=np.int64,
            )[: max(1, mesh_id + 1)]
        )
    query_points = [
        np.random.default_rng(mesh_idx[i]).random((4 + mesh_idx[i] % 3, 2), dtype=np.float32)
        for i in range(num_samples)
    ]
    return GinotRawDataset(
        dataset_dir="/tmp/mock_micro_puc",
        query_points=query_points,
        point_clouds=query_points,
        targets=[np.zeros((qp.shape[0], 1), dtype=np.float32) for qp in query_points],
        cells=cells,
        input_params=None,
        target_fields=("y",),
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
        micro_puc_mesh_idx=mesh_idx,
    )


def test_group_sample_ids_by_topology_poisson_structured() -> None:
    raw = _mock_poisson_structured_raw(6)
    groups = group_sample_ids_by_topology(raw, "poisson_structured", [5, 3, 1, 4, 0, 2])
    assert groups == {0: [0, 1, 2, 3, 4, 5]}


def test_group_sample_ids_by_topology_micro_puc_groups_by_mesh() -> None:
    raw = _mock_micro_puc_raw(9, num_meshes=3)
    groups = group_sample_ids_by_topology(raw, "micro_puc", list(range(9)))
    assert set(groups.keys()) == {0, 1, 2}
    assert groups[0] == [0, 3, 6]
    assert groups[1] == [1, 4, 7]
    assert groups[2] == [2, 5, 8]


def test_group_sample_ids_by_topology_none_are_singletons() -> None:
    groups = group_sample_ids_by_topology(None, "poisson_unstructured", [10, 11, 12])
    assert len(groups) == 3
    members = sorted(v[0] for v in groups.values())
    assert members == [10, 11, 12]
    for members_list in groups.values():
        assert len(members_list) == 1
    # Singleton groups use distinct negative keys, never collapsed together.
    assert len(set(groups.keys())) == 3
    assert all(key < 0 for key in groups.keys())


def test_group_sample_ids_by_topology_missing_raw_falls_back_to_singletons() -> None:
    groups = group_sample_ids_by_topology(None, "poisson_structured", [1, 2, 3])
    assert len(groups) == 3
    assert all(len(v) == 1 for v in groups.values())


def test_compute_or_reuse_graph_payload_computes_once_for_shared_topology() -> None:
    memo: dict[int, tuple[bytes, torch.Tensor, torch.Tensor]] = {}
    calls = []

    def compute_fn() -> tuple[bytes, torch.Tensor, torch.Tensor]:
        calls.append(1)
        return b"payload-for-topo-0", torch.zeros(2), torch.ones(3, 2)

    first = compute_or_reuse_graph_payload(memo, 0, compute_fn)
    second = compute_or_reuse_graph_payload(memo, 0, compute_fn)
    third = compute_or_reuse_graph_payload(memo, 0, compute_fn)

    assert len(calls) == 1
    assert first == second == third
    payload, eigenvalues, eigenvectors = first
    assert payload == b"payload-for-topo-0"
    assert torch.equal(eigenvalues, torch.zeros(2))
    assert torch.equal(eigenvectors, torch.ones(3, 2))
    assert memo == {0: first}


def test_compute_or_reuse_graph_payload_recomputes_per_distinct_topology() -> None:
    memo: dict[int, tuple[bytes, torch.Tensor, torch.Tensor]] = {}
    call_count = {"n": 0}

    def compute_fn() -> tuple[bytes, torch.Tensor, torch.Tensor]:
        call_count["n"] += 1
        payload = f"payload-{call_count['n']}".encode()
        return payload, torch.tensor([float(call_count["n"])]), torch.full((2, 1), float(call_count["n"]))

    result_a = compute_or_reuse_graph_payload(memo, 0, compute_fn)
    result_b = compute_or_reuse_graph_payload(memo, 1, compute_fn)

    assert call_count["n"] == 2
    assert result_a[0] != result_b[0]
    assert memo == {0: result_a, 1: result_b}


def test_compute_or_reuse_graph_payload_none_key_never_caches() -> None:
    memo: dict[int, tuple[bytes, torch.Tensor, torch.Tensor]] = {}
    call_count = {"n": 0}

    def compute_fn() -> tuple[bytes, torch.Tensor, torch.Tensor]:
        call_count["n"] += 1
        payload = f"payload-{call_count['n']}".encode()
        return payload, torch.tensor([float(call_count["n"])]), torch.full((2, 1), float(call_count["n"]))

    result_a = compute_or_reuse_graph_payload(memo, None, compute_fn)
    result_b = compute_or_reuse_graph_payload(memo, None, compute_fn)

    assert call_count["n"] == 2
    assert result_a[0] != result_b[0]
    assert memo == {}


def test_compute_or_reuse_graph_payload_hit_restores_eigenvectors() -> None:
    """A cache hit must also return eigenvectors so callers can warm-start graph_start."""
    memo: dict[int, tuple[bytes, torch.Tensor, torch.Tensor]] = {}

    def compute_fn() -> tuple[bytes, torch.Tensor, torch.Tensor]:
        return b"payload", torch.tensor([1.0, 2.0]), torch.eye(2)

    _, _, eigenvectors_miss = compute_or_reuse_graph_payload(memo, 5, compute_fn)

    def _fail_compute_fn() -> tuple[bytes, torch.Tensor, torch.Tensor]:
        raise AssertionError("compute_fn must not run again on a cache hit")

    _, _, eigenvectors_hit = compute_or_reuse_graph_payload(memo, 5, _fail_compute_fn)
    assert torch.equal(eigenvectors_hit, eigenvectors_miss)
