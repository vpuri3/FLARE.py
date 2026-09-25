from __future__ import annotations

import pytest
import torch

from pdebench.dataset.laplacian import LaplacianCacheKey, LaplacianService, LmdbLaplacianBackend
from pdebench.dataset.laplacian.backends import InMemoryLaplacianBackend
from pdebench.dataset.sample import Sample, SampleKind


def _write_minimal_graph_cache_index(graph_cache_dir, sample_ids: list[int]) -> None:
    ids = torch.tensor(sample_ids, dtype=torch.long)
    torch.save(
        {
            "sample_ids": ids,
            "shard_id": torch.zeros_like(ids),
            "node_lengths": torch.full_like(ids, 4),
            "boundary_lengths": torch.full_like(ids, 4),
        },
        graph_cache_dir / "index.pt",
    )


def _sample(sample_id: str = "0") -> Sample:
    return Sample(
        pos=torch.randn(4, 2),
        y=torch.zeros(4, 1),
        sample_id=sample_id,
        kind=SampleKind.STATIC,
    )


def test_is_complete_false_when_cache_empty() -> None:
    backend = InMemoryLaplacianBackend()
    service = LaplacianService(backend)
    assert not service.is_complete(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0", "1"],
        spec="graph",
        K=4,
    )


def test_ensure_computes_and_marks_complete() -> None:
    backend = InMemoryLaplacianBackend()
    calls: list[str] = []

    def compute(key: LaplacianCacheKey, sample: Sample):
        del sample
        calls.append(key.sample_id)
        k = int(key.K)
        eigvals = torch.arange(k, dtype=torch.float32)
        eigvecs = torch.eye(4, k)
        return eigvals, eigvecs

    service = LaplacianService(backend, compute_fn=compute)
    samples = {"0": _sample("0"), "1": _sample("1")}
    service.ensure(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0", "1"],
        spec="graph",
        K=4,
        samples=samples,
    )
    assert service.is_complete(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0", "1"],
        spec="graph",
        K=4,
    )
    assert sorted(calls) == ["0", "1"]


def test_attach_sets_laplacian_fields() -> None:
    backend = InMemoryLaplacianBackend()
    key = LaplacianCacheKey("poisson_unstructured", "seed0", "0", "graph", 3)
    eigvals = torch.tensor([0.1, 0.2, 0.3])
    eigvecs = torch.randn(4, 3)
    backend.save(key, eigvals, eigvecs)

    service = LaplacianService(backend)
    sample = _sample("0")
    out = service.attach(
        sample,
        canonical_dataset="poisson_unstructured",
        split_group="seed0",
        spec="graph",
        K=3,
    )
    assert out is sample
    assert out.laplacian_eig is not None
    assert out.laplacian_eigvals is not None
    assert torch.allclose(out.laplacian_eig, eigvecs)
    assert torch.allclose(out.laplacian_eigvals, eigvals)


def test_attach_incomplete_cache_raises_precompute_hint() -> None:
    service = LaplacianService(InMemoryLaplacianBackend())
    sample = _sample("7")
    with pytest.raises(FileNotFoundError, match=r"precompute") as exc_info:
        service.attach(
            sample,
            canonical_dataset="poisson_unstructured",
            split_group="seed0",
            spec="graph",
            K=8,
        )
    msg = str(exc_info.value)
    assert "poisson_unstructured" in msg
    assert "seed0" in msg
    assert sample.laplacian_eig is None


def test_load_returns_cached_eigenpairs() -> None:
    backend = InMemoryLaplacianBackend()
    key = LaplacianCacheKey("micro_puc", "full_seed0", "42", "graph", 2)
    eigvals = torch.tensor([1.0, 2.0])
    eigvecs = torch.ones(5, 2)
    backend.save(key, eigvals, eigvecs)

    service = LaplacianService(backend)
    got_vals, got_vecs = service.load(
        "micro_puc",
        "full_seed0",
        "42",
        spec="graph",
        K=2,
    )
    assert torch.allclose(got_vals, eigvals)
    assert torch.allclose(got_vecs, eigvecs)


def test_load_slices_prefix_k_from_larger_cache() -> None:
    """K=32 requests must reuse a K=64 graph cache (first 32 eigenpairs)."""
    backend = InMemoryLaplacianBackend()
    eigvals64 = torch.arange(64, dtype=torch.float32)
    eigvecs64 = torch.randn(8, 64)
    backend.save(
        LaplacianCacheKey("poisson_unstructured", "seed0", "0", "graph", 64),
        eigvals64,
        eigvecs64,
    )
    service = LaplacianService(backend)

    assert service.is_complete(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0"],
        spec="graph",
        K=32,
    )
    got_vals, got_vecs = service.load(
        "poisson_unstructured",
        "seed0",
        "0",
        spec="graph",
        K=32,
    )
    assert got_vals is not None
    assert got_vals.shape == (32,)
    assert got_vecs.shape == (8, 32)
    assert torch.allclose(got_vals, eigvals64[:32])
    assert torch.allclose(got_vecs, eigvecs64[:, :32])

    sample = _sample("0")
    service.attach(
        sample,
        canonical_dataset="poisson_unstructured",
        split_group="seed0",
        spec="graph",
        K=32,
    )
    assert sample.laplacian_eig is not None
    assert sample.laplacian_eig.shape[-1] == 32


def test_load_prefix_k_does_not_use_smaller_cache() -> None:
    backend = InMemoryLaplacianBackend()
    backend.save(
        LaplacianCacheKey("poisson_unstructured", "seed0", "0", "graph", 16),
        torch.arange(16, dtype=torch.float32),
        torch.randn(4, 16),
    )
    service = LaplacianService(backend)
    assert not service.is_complete(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0"],
        spec="graph",
        K=32,
    )
    with pytest.raises(FileNotFoundError, match=r"precompute"):
        service.load("poisson_unstructured", "seed0", "0", spec="graph", K=32)


def test_attach_k_zero_is_noop() -> None:
    service = LaplacianService(InMemoryLaplacianBackend())
    sample = _sample("0")
    out = service.attach(
        sample,
        canonical_dataset="poisson_unstructured",
        split_group="seed0",
        spec="graph",
        K=0,
    )
    assert out.laplacian_eig is None
    assert out.laplacian_eigvals is None


def test_ensure_skips_already_cached() -> None:
    backend = InMemoryLaplacianBackend()
    key = LaplacianCacheKey("poisson_unstructured", "seed0", "0", "graph", 2)
    backend.save(key, torch.zeros(2), torch.zeros(4, 2))
    calls: list[str] = []

    def compute(key: LaplacianCacheKey, sample: Sample):
        del sample
        calls.append(key.sample_id)
        return torch.ones(2), torch.ones(4, 2)

    service = LaplacianService(backend, compute_fn=compute)
    service.ensure(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0", "1"],
        spec="graph",
        K=2,
        samples={"0": _sample("0"), "1": _sample("1")},
    )
    assert calls == ["1"]


def test_lmdb_backend_roundtrip_via_service(tmp_path) -> None:
    graph_cache_dir = (
        tmp_path / "static_cache" / "graph_lmdb" / "poisson_unstructured_default_seed0" / "train"
    )
    graph_cache_dir.mkdir(parents=True)
    _write_minimal_graph_cache_index(graph_cache_dir, [0])

    backend = LmdbLaplacianBackend(
        {("poisson_unstructured", "seed0"): graph_cache_dir},
    )
    service = LaplacianService(backend)
    eigvals = torch.tensor([0.5, 1.5], dtype=torch.float32)
    eigvecs = torch.randn(4, 2)
    key = LaplacianCacheKey("poisson_unstructured", "seed0", "0", "graph:2", 2)
    backend.save(key, eigvals, eigvecs)

    assert service.is_complete(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0"],
        spec="graph:2",
        K=2,
    )
    sample = _sample("0")
    service.attach(
        sample,
        canonical_dataset="poisson_unstructured",
        split_group="seed0",
        spec="graph:2",
        K=2,
    )
    assert sample.laplacian_eig is not None
    assert sample.laplacian_eigvals is not None
    assert torch.allclose(sample.laplacian_eigvals, eigvals)
    assert sample.laplacian_eig.shape == (4, 2)


def test_lmdb_backend_prefix_k_slice_from_larger_cache(tmp_path) -> None:
    graph_cache_dir = (
        tmp_path / "static_cache" / "graph_lmdb" / "poisson_unstructured_default_seed0" / "train"
    )
    graph_cache_dir.mkdir(parents=True)
    _write_minimal_graph_cache_index(graph_cache_dir, [0])

    backend = LmdbLaplacianBackend({("poisson_unstructured", "seed0"): graph_cache_dir})
    service = LaplacianService(backend)
    eigvals = torch.arange(4, dtype=torch.float32)
    eigvecs = torch.randn(5, 4)
    backend.save(
        LaplacianCacheKey("poisson_unstructured", "seed0", "0", "graph:4", 4),
        eigvals,
        eigvecs,
    )

    assert service.is_complete(
        "poisson_unstructured",
        "seed0",
        sample_ids=["0"],
        spec="graph",
        K=2,
    )
    full_vals, full_vecs = service.load(
        "poisson_unstructured",
        "seed0",
        "0",
        spec="graph",
        K=4,
    )
    got_vals, got_vecs = service.load(
        "poisson_unstructured",
        "seed0",
        "0",
        spec="graph",
        K=2,
    )
    assert full_vals is not None and got_vals is not None
    assert got_vecs.shape == (5, 2)
    assert torch.allclose(got_vals, full_vals[:2])
    assert torch.allclose(got_vecs, full_vecs[:, :2])


def test_torch_file_backend_roundtrip(tmp_path) -> None:
    from pdebench.dataset.laplacian import TorchFileLaplacianBackend

    backend = TorchFileLaplacianBackend(tmp_path)
    service = LaplacianService(backend)
    key = LaplacianCacheKey("plaid_tensile2d", "graph_K4", "7", "graph", 4)
    eigvals = torch.arange(4, dtype=torch.float32)
    eigvecs = torch.randn(5, 4)
    backend.save(key, eigvals, eigvecs)
    assert backend.has(key)
    got_vals, got_vecs = service.load("plaid_tensile2d", "graph_K4", 7, "graph", 4)
    assert torch.allclose(got_vals, eigvals)
    assert torch.allclose(got_vecs, eigvecs)


def test_torch_file_backend_uses_unique_same_directory_temp_paths(tmp_path, monkeypatch) -> None:
    from pdebench.dataset.laplacian import TorchFileLaplacianBackend

    backend = TorchFileLaplacianBackend(tmp_path)
    key = LaplacianCacheKey("plaid_tensile2d", "graph_K4", "7", "graph", 4)
    temp_paths = []

    def fake_save(payload, path) -> None:
        del payload
        temp_paths.append(path)
        path.write_bytes(b"cache")

    monkeypatch.setattr(torch, "save", fake_save)
    backend.save(key, torch.arange(4), torch.eye(5, 4))
    backend.save(key, torch.arange(4), torch.eye(5, 4))

    cache_file = backend._cache_file(key)
    assert len(set(temp_paths)) == 2
    assert all(path.parent == cache_file.parent for path in temp_paths)
    assert all(path.name.startswith(f".{cache_file.name}.") for path in temp_paths)


def test_plaid_attach_fail_loud_without_cache(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pdebench.dataset.plaid_datasets as plaid_datasets
    import pdebench.dataset.plaid_laplacian as plaid_laplacian
    from pdebench.dataset.laplacian import TorchFileLaplacianBackend

    calls: list[str] = []

    def boom(key, sample):
        del key, sample
        calls.append("compute")
        raise AssertionError("attach must not compute on miss")

    monkeypatch.setattr(
        plaid_laplacian,
        "_plaid_laplacian_service",
        lambda dataset_dir: LaplacianService(
            TorchFileLaplacianBackend(dataset_dir),
            compute_fn=boom,
        ),
    )
    graph = SimpleNamespace(
        pos=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
    )
    with pytest.raises(FileNotFoundError, match="Incomplete Laplacian cache"):
        plaid_datasets._attach_laplacian_features(
            graph,
            dataset_dir=str(tmp_path),
            dataset_name="plaid_tensile2d",
            sample_idx=0,
            laplacian_eig_dim=4,
            laplacian_spec="graph",
        )
    assert calls == []


def test_plaid_attach_via_service_matches_direct_attach(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import pdebench.dataset.plaid_laplacian as plaid_laplacian

    backend = InMemoryLaplacianBackend()
    split_group = plaid_laplacian._plaid_laplacian_split_group("graph", 3)
    key = LaplacianCacheKey("plaid_tensile2d", split_group, "3", "graph", 3)
    eigvals = torch.tensor([0.1, 0.2, 0.3])
    eigvecs = torch.eye(4, 3)
    backend.save(key, eigvals, eigvecs)

    service = LaplacianService(backend)
    monkeypatch.setattr(
        plaid_laplacian,
        "_plaid_laplacian_service",
        lambda dataset_dir: service,
    )

    pos = torch.randn(4, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    graph = SimpleNamespace(pos=pos, edge_index=edge_index)

    direct = Sample(
        pos=pos,
        y=torch.zeros(4, 1),
        edge_index=edge_index,
        sample_id="3",
        kind=SampleKind.STATIC,
    )
    service.attach(
        direct,
        canonical_dataset="plaid_tensile2d",
        split_group=split_group,
        spec="graph",
        K=3,
    )

    plaid_laplacian._attach_laplacian_features(
        graph,
        dataset_dir=str(tmp_path),
        dataset_name="plaid_tensile2d",
        sample_idx=3,
        laplacian_eig_dim=3,
        laplacian_spec="graph",
    )

    assert torch.allclose(graph.laplacian_eig, direct.laplacian_eig)
    assert torch.allclose(graph.laplacian_eigvals.reshape(-1), direct.laplacian_eigvals)
    assert graph.laplacian_eigvals.shape == (1, 3)


def test_ginot_graph_cache_attach_via_service(monkeypatch, tmp_path) -> None:
    import pdebench.dataset.ginot.dataset as ginot_dataset
    from pdebench.dataset.ginot.dataset import GraphCacheDataset, GraphCacheSplit

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


def test_ginot_graph_cache_attach_fail_loud_without_cache(monkeypatch, tmp_path) -> None:
    import pdebench.dataset.ginot.dataset as ginot_dataset
    from pdebench.dataset.ginot.dataset import GraphCacheDataset, GraphCacheSplit

    cache_dir = tmp_path / "train"

    def boom(*args, **kwargs):
        raise AssertionError("attach must not compute on miss")

    monkeypatch.setattr(ginot_dataset, "LmdbLaplacianBackend", InMemoryLaplacianBackend)
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
        indices=[0],
        node_lengths=[3],
        shard_ids=[0],
        max_node_length=3,
        max_boundary_length=3,
    )
    dataset = GraphCacheDataset(
        split,
        dataset_name="poisson_unstructured",
        raw=None,
        laplacian_eig_dim=4,
        laplacian_spec="graph",
    )
    dataset._laplacian_service._compute_fn = boom  # type: ignore[attr-defined]

    with pytest.raises(FileNotFoundError, match="Incomplete Laplacian cache"):
        dataset[0]


def test_plaid_attach_loads_cached_eigenpairs(tmp_path) -> None:
    from types import SimpleNamespace

    import pdebench.dataset.plaid_datasets as plaid_datasets
    import pdebench.dataset.plaid_laplacian as plaid_laplacian
    from pdebench.dataset.laplacian import TorchFileLaplacianBackend

    backend = TorchFileLaplacianBackend(tmp_path)
    split_group = plaid_laplacian._plaid_laplacian_split_group("graph", 3)
    key = LaplacianCacheKey("plaid_tensile2d", split_group, "3", "graph", 3)
    eigvals = torch.tensor([0.1, 0.2, 0.3])
    eigvecs = torch.eye(4, 3)
    backend.save(key, eigvals, eigvecs)

    graph = SimpleNamespace(
        pos=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
    )
    plaid_datasets._attach_laplacian_features(
        graph,
        dataset_dir=str(tmp_path),
        dataset_name="plaid_tensile2d",
        sample_idx=3,
        laplacian_eig_dim=3,
        laplacian_spec="graph",
    )
    assert torch.allclose(graph.laplacian_eig, eigvecs)
    assert torch.allclose(graph.laplacian_eigvals.reshape(-1), eigvals)
