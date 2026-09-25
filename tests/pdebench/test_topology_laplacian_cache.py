from __future__ import annotations

import numpy as np
import torch

import pdebench.dataset.ginot.dataset as ginot_dataset
from pdebench.dataset.ginot.dataset import GraphCacheDataset, GraphCacheSplit
from pdebench.dataset.ginot.mesh import (
    topology_key_for_row,
    unique_topology_representatives,
)
from pdebench.dataset.ginot.sample import build_graph_sample_dict
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer
from pdebench.dataset.laplacian import (
    DEFAULT_LAPLACIAN_SPECS,
    compute_laplacian_eigendecomp_part,
    load_laplacian_cache,
    parse_laplacian_spec,
    save_split_laplacian_lmdb,
    split_laplacian_lmdb_has,
)


def _identity_normalizer(dim: int) -> StandardNormalizer:
    return StandardNormalizer(mean=torch.zeros(dim), std=torch.ones(dim))


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


def _mock_micro_puc_fixed_raw(num_samples: int, num_geometries: int = 3) -> GinotRawDataset:
    source_ids = np.array([i % num_geometries for i in range(num_samples)], dtype=np.int64)
    cells_per_geom = []
    for geom_id in range(num_geometries):
        cells_per_geom.append(
            np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
    cells = [cells_per_geom[source_ids[i]] for i in range(num_samples)]
    query_points = [np.random.default_rng(int(source_ids[i])).random((4, 2), dtype=np.float32) for i in range(num_samples)]
    return GinotRawDataset(
        dataset_dir="/tmp/mock_micro_puc_fixed",
        query_points=query_points,
        point_clouds=query_points,
        targets=[np.zeros((4, 1), dtype=np.float32) for _ in range(num_samples)],
        cells=cells,
        input_params=None,
        target_fields=("y",),
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
        micro_puc_source_geometry_ids=source_ids,
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


def test_topology_key_poisson_structured_is_always_zero() -> None:
    raw = _mock_poisson_structured_raw(8)
    for row in range(8):
        assert topology_key_for_row(raw, "poisson_structured", row) == 0


def test_unique_topology_representatives_poisson_structured_single_entry() -> None:
    raw = _mock_poisson_structured_raw(10)
    reps = unique_topology_representatives(raw, "poisson_structured", 10)
    assert reps == {0: 0}


def test_topology_key_for_micro_puc_matches_mesh_idx() -> None:
    raw = _mock_micro_puc_raw(8)
    for row in range(8):
        assert topology_key_for_row(raw, "micro_puc", row) == int(raw.micro_puc_mesh_idx[row])


def test_topology_key_for_micro_puc_fixed_matches_source_geometry_ids() -> None:
    raw = _mock_micro_puc_fixed_raw(8)
    for row in range(8):
        assert topology_key_for_row(raw, "micro_puc_fixed", row) == int(raw.micro_puc_source_geometry_ids[row])


def test_unique_topology_representatives_picks_first_row() -> None:
    raw = _mock_micro_puc_raw(10, num_meshes=3)
    reps = unique_topology_representatives(raw, "micro_puc", 10)
    assert set(reps.keys()) == {0, 1, 2}
    assert reps[0] == 0
    assert reps[1] == 1
    assert reps[2] == 2


def test_rows_sharing_topology_key_have_identical_edge_index_micro_puc() -> None:
    raw = _mock_micro_puc_raw(12, num_meshes=3)
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    by_key: dict[int, torch.Tensor] = {}
    for row in range(12):
        key = topology_key_for_row(raw, "micro_puc", row)
        sample = build_graph_sample_dict(raw, row, pos_norm, pos_norm, y_norm, to_cpu=True)
        edge_index = sample["edge_index"]
        if key in by_key:
            assert torch.equal(by_key[key], edge_index)
        else:
            by_key[key] = edge_index


def test_rows_sharing_topology_key_have_identical_edge_index_micro_puc_fixed() -> None:
    raw = _mock_micro_puc_fixed_raw(12, num_geometries=3)
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    by_key: dict[int, torch.Tensor] = {}
    for row in range(12):
        key = topology_key_for_row(raw, "micro_puc_fixed", row)
        sample = build_graph_sample_dict(raw, row, pos_norm, pos_norm, y_norm, to_cpu=True)
        edge_index = sample["edge_index"]
        if key in by_key:
            assert torch.equal(by_key[key], edge_index)
        else:
            by_key[key] = edge_index


def _write_minimal_graph_cache_index(graph_cache_dir, sample_ids: list[int]) -> None:
    ids = torch.tensor(sample_ids, dtype=torch.long)
    torch.save(
        {
            "sample_ids": ids,
            "shard_id": torch.zeros_like(ids),
            "node_lengths": torch.full_like(ids, 20),
            "boundary_lengths": torch.full_like(ids, 20),
        },
        graph_cache_dir / "index.pt",
    )


def test_graph_cache_dataset_loads_laplacian_through_service(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str, int, str, int]] = []
    eigvals = torch.tensor([0.5, 1.5])
    eigvecs = torch.ones(3, 2)
    cache_dir = tmp_path / "train"

    class FakeBackend:
        def __init__(self, graph_cache_dirs):
            assert graph_cache_dirs == {("poisson_unstructured", "train"): cache_dir}

    class FakeService:
        def __init__(self, backend):
            assert isinstance(backend, FakeBackend)

        def attach(self, sample, *, canonical_dataset, split_group, spec, K):
            calls.append((canonical_dataset, split_group, int(sample.sample_id), spec, K))
            sample.laplacian_eig = eigvecs
            sample.laplacian_eigvals = eigvals
            return sample

    monkeypatch.setattr(ginot_dataset, "LmdbLaplacianBackend", FakeBackend)
    monkeypatch.setattr(ginot_dataset, "LaplacianService", FakeService)
    monkeypatch.setattr(
        ginot_dataset,
        "load_graph_cache_sample",
        lambda cache_dir, idx: {
            "pos": torch.randn(3, 2),
            "y": torch.zeros(3, 1),
            "edge_index": torch.tensor([[0], [1]]),
            "sample_id": torch.tensor(int(idx), dtype=torch.long),
        },
    )
    monkeypatch.setattr(ginot_dataset, "attach_sample_feats", lambda sample, *args, **kwargs: sample)
    split = GraphCacheSplit(
        cache_dir=str(cache_dir),
        indices=[7],
        node_lengths=[3],
        shard_ids=[0],
        max_node_length=3,
        max_boundary_length=3,
    )

    dataset = GraphCacheDataset(
        split,
        dataset_name="poisson_unstructured",
        raw=None,
        laplacian_eig_dim=2,
        laplacian_spec="graph",
    )
    sample = dataset[0]

    assert calls == [("poisson_unstructured", "train", 7, "graph", 2)]
    assert torch.equal(sample["laplacian_eig"], eigvecs)
    assert torch.equal(sample["laplacian_eigvals"], eigvals)


def test_default_laplacian_specs_are_graph_only() -> None:
    assert DEFAULT_LAPLACIAN_SPECS == "graph:64"


def test_dataset_laplacian_specs_keep_graph_64_except_bumper_beam_override() -> None:
    from pdebench.dataset.laplacian import DATASET_LAPLACIAN_SPECS

    for dataset, spec in DATASET_LAPLACIAN_SPECS.items():
        if dataset == "bumper_beam":
            continue
        assert spec == DEFAULT_LAPLACIAN_SPECS, f"{dataset!r} has {spec!r}, expected graph:64"

    assert DATASET_LAPLACIAN_SPECS["bumper_beam"] == "graph:32"


def test_ambiguous_fem_laplacian_spec_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported Laplacian cache operator"):
        parse_laplacian_spec("fem:64", 64)


def test_graph_reordered_laplacian_spec_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="Unsupported Laplacian cache operator"):
        parse_laplacian_spec("graph_reordered:32", 64)
    with pytest.raises(ValueError, match="Unsupported Laplacian cache operator"):
        parse_laplacian_spec("graph-reordered", 16)


def test_fem_v_infers_2d_and_differs_from_u() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    _, vecs_v = compute_laplacian_eigendecomp_part(edge_index, pos, cells, "fem_v", 2)
    _, vecs_u = compute_laplacian_eigendecomp_part(edge_index, pos, cells, "fem_u", 2)

    gram_v = vecs_v.T @ vecs_v
    assert torch.allclose(gram_v, torch.eye(2), atol=1e-4)
    assert not torch.allclose(vecs_v.abs(), vecs_u.abs(), atol=1e-4)


def test_fem_v_infers_3d_volume_tetrahedra() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    vals, vecs = compute_laplacian_eigendecomp_part(edge_index, pos, cells, "fem_v", 2)

    assert vals.shape == (2,)
    assert vecs.shape == (5, 2)
    assert torch.all(vals >= -1e-5)
    assert torch.allclose(vecs.T @ vecs, torch.eye(2), atol=1e-4)


def test_fem_u_and_v_share_one_cache_payload(tmp_path) -> None:
    graph_cache_dir = tmp_path / "static_cache" / "graph_lmdb" / "poisson_unstructured_default_seed0" / "train"
    graph_cache_dir.mkdir(parents=True)
    _write_minimal_graph_cache_index(graph_cache_dir, [3])

    eigenvalues = torch.tensor([0.5, 1.5], dtype=torch.float32)
    eigenvectors_u = torch.full((5, 2), 2.0)
    eigenvectors_v = torch.full((5, 2), 3.0)
    save_split_laplacian_lmdb(
        graph_cache_dir,
        sample_id=3,
        num_eigenvectors=2,
        laplacian_spec="fem-v:2",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors_v,
        eigenvectors_u=eigenvectors_u,
        eigenvectors_v=eigenvectors_v,
    )

    vals_u, vecs_u = load_laplacian_cache(graph_cache_dir, 3, 2, "fem-u:2")
    vals_v, vecs_v = load_laplacian_cache(graph_cache_dir, 3, 2, "fem-v:2")

    assert torch.allclose(vals_u, eigenvalues)
    assert torch.allclose(vals_v, eigenvalues)
    assert torch.allclose(vecs_u, eigenvectors_u.to(torch.float16).float())
    assert torch.allclose(vecs_v, eigenvectors_v.to(torch.float16).float())
    assert split_laplacian_lmdb_has(graph_cache_dir, 3, 2, "fem-u:2")


def test_composite_fem_u_and_v_cache_concatenates_values_and_vectors(tmp_path) -> None:
    graph_cache_dir = tmp_path / "static_cache" / "graph_lmdb" / "poisson_unstructured_default_seed0" / "train"
    graph_cache_dir.mkdir(parents=True)
    _write_minimal_graph_cache_index(graph_cache_dir, [3])

    eigenvalues = torch.tensor([0.5, 1.5], dtype=torch.float32)
    eigenvectors_u = torch.full((5, 2), 2.0)
    eigenvectors_v = torch.full((5, 2), 3.0)
    save_split_laplacian_lmdb(
        graph_cache_dir,
        sample_id=3,
        num_eigenvectors=2,
        laplacian_spec="fem-v:2",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors_v,
        eigenvectors_u=eigenvectors_u,
        eigenvectors_v=eigenvectors_v,
    )

    vals, vecs = load_laplacian_cache(graph_cache_dir, 3, 2, "fem-u:2,fem-v:2")

    assert vals is not None
    assert torch.allclose(vals, torch.cat([eigenvalues, eigenvalues], dim=0))
    assert torch.allclose(
        vecs,
        torch.cat(
            [eigenvectors_u.to(torch.float16).float(), eigenvectors_v.to(torch.float16).float()],
            dim=-1,
        ),
    )


def test_load_laplacian_cache_reads_split_lmdb(tmp_path) -> None:
    graph_cache_dir = tmp_path / "static_cache" / "graph_lmdb" / "micro_puc_full73879_seed0" / "train"
    graph_cache_dir.mkdir(parents=True)
    _write_minimal_graph_cache_index(graph_cache_dir, [42])
    eigenvalues = torch.linspace(0.2, 2.0, 64)
    eigenvectors = torch.randn(20, 64)
    save_split_laplacian_lmdb(
        graph_cache_dir,
        sample_id=42,
        num_eigenvectors=64,
        laplacian_spec="graph:64",
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )
    assert split_laplacian_lmdb_has(graph_cache_dir, 42, 64, "graph:64")

    vals, vecs = load_laplacian_cache(
        graph_cache_dir,
        sample_id=42,
        num_eigenvectors=64,
        laplacian_spec="graph:64",
    )
    assert torch.allclose(vals, eigenvalues)
    assert torch.allclose(vecs, eigenvectors.to(torch.float16).float(), atol=1e-3)


def test_distinct_samples_get_distinct_split_caches_even_with_shared_topology(tmp_path) -> None:
    graph_cache_dir = tmp_path / "graph_cache" / "train"
    graph_cache_dir.mkdir(parents=True)
    _write_minimal_graph_cache_index(graph_cache_dir, [0, 1])
    save_split_laplacian_lmdb(
        graph_cache_dir,
        sample_id=0,
        num_eigenvectors=64,
        laplacian_spec="graph:64",
        eigenvalues=torch.zeros(64),
        eigenvectors=torch.ones(8, 64),
    )
    save_split_laplacian_lmdb(
        graph_cache_dir,
        sample_id=1,
        num_eigenvectors=64,
        laplacian_spec="graph:64",
        eigenvalues=torch.ones(64),
        eigenvectors=torch.zeros(8, 64),
    )
    _, vecs0 = load_laplacian_cache(graph_cache_dir, 0, 64, "graph:64")
    _, vecs1 = load_laplacian_cache(graph_cache_dir, 1, 64, "graph:64")
    assert not torch.equal(vecs0, vecs1)
