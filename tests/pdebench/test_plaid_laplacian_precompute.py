from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch


def _module():
    try:
        return importlib.import_module("pdebench.dataset.plaid_laplacian_precompute")
    except ModuleNotFoundError:
        pytest.fail("pdebench.dataset.plaid_laplacian_precompute is missing")


def _graph(sample_id: int):
    return SimpleNamespace(
        sample_id=sample_id,
        pos=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        cells=torch.tensor([[0, 1, 2]], dtype=torch.long),
    )


def test_partition_sample_ids_is_balanced_disjoint_and_round_robin() -> None:
    chunks = _module()._partition_sample_ids([9, 2, 7, 1, 4, 3, 8], 3)

    assert chunks == [[1, 4, 9], [2, 7], [3, 8]]
    assert max(map(len, chunks)) - min(map(len, chunks)) <= 1
    assert len({sample_id for chunk in chunks for sample_id in chunk}) == 7


def test_index_k0_payload_includes_train_and_validation_graphs() -> None:
    train = [_graph(7), _graph(2)]
    validation = [_graph(11)]

    index = _module()._index_k0_payload(
        {"train_graphs": train, "val_graphs": validation}
    )

    assert index == {2: train[1], 7: train[0], 11: validation[0]}


def test_index_k0_payload_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate K0 sample ID 3"):
        _module()._index_k0_payload(
            {"train_graphs": [_graph(3)], "val_graphs": [_graph(3)]}
        )


def test_index_k0_payload_rejects_missing_split_lists_as_invalid_schema() -> None:
    with pytest.raises(ValueError, match="Invalid static PLAID K0 cache payload"):
        _module()._index_k0_payload({"train_graphs": [_graph(3)]})


def test_filter_missing_ids_skips_exact_or_larger_covering_cache() -> None:
    class Service:
        def is_complete(self, dataset, split_group, sample_ids, spec, K):
            del dataset, split_group, spec, K
            return int(sample_ids[0]) in {2, 9}

    assert _module()._filter_missing_ids(
        [9, 2, 7],
        service=Service(),
        dataset_name="plaid_tensile2d",
        laplacian_eig_dim=64,
        laplacian_spec="graph",
    ) == [7]


def test_worker_forwards_explicit_cuda_device(monkeypatch, tmp_path) -> None:
    module = _module()
    graphs = {5: _graph(5)}
    calls = []
    monkeypatch.setattr(module, "_load_k0_index", lambda path: graphs)
    monkeypatch.setattr(module.torch.cuda, "set_device", lambda gpu_id: None)
    monkeypatch.setattr(module, "_ensure_plaid_laplacian", lambda **kwargs: calls.append(kwargs))

    module._worker_entry(
        gpu_id=2,
        sample_ids=[5],
        k0_cache_file=str(tmp_path / "graphs.pt"),
        dataset_dir=str(tmp_path),
        dataset_name="plaid_tensile2d",
        laplacian_eig_dim=64,
        laplacian_spec="graph",
    )

    assert calls[0]["device"] == "cuda:2"


def test_process_cleanup_after_partial_start_failure() -> None:
    module = _module()

    class Process:
        def __init__(self, *, start_error=None):
            self.start_error = start_error
            self.started = False
            self.terminated = False
            self.join_count = 0

        def start(self) -> None:
            if self.start_error is not None:
                raise self.start_error
            self.started = True

        def is_alive(self) -> bool:
            return self.started and not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def join(self) -> None:
            self.join_count += 1

    first = Process()
    failed = Process(start_error=RuntimeError("start failed"))
    never_started = Process()

    with pytest.raises(RuntimeError, match="start failed"):
        module._start_and_join_processes([first, failed, never_started])

    assert first.terminated and first.join_count == 1
    assert not failed.terminated and failed.join_count == 0
    assert not never_started.started and never_started.join_count == 0


def test_process_cleanup_after_join_interruption() -> None:
    module = _module()

    class Process:
        def __init__(self, *, interrupt_first_join=False):
            self.interrupt_first_join = interrupt_first_join
            self.started = False
            self.terminated = False
            self.join_count = 0

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started and not self.terminated

        def terminate(self) -> None:
            self.terminated = True

        def join(self) -> None:
            self.join_count += 1
            if self.interrupt_first_join and self.join_count == 1:
                raise KeyboardInterrupt

    interrupted = Process(interrupt_first_join=True)
    waiting = Process()

    with pytest.raises(KeyboardInterrupt):
        module._start_and_join_processes([interrupted, waiting])

    assert interrupted.terminated and interrupted.join_count == 2
    assert waiting.terminated and waiting.join_count == 1


def test_plaid_k0_graph_cache_uses_unique_same_directory_temp_paths(
    monkeypatch, tmp_path
) -> None:
    import pdebench.dataset.plaid_datasets as plaid_datasets

    cache_file = tmp_path / "cache" / "graphs.pt"
    temp_paths = []

    def fake_save(payload, path) -> None:
        del payload
        temp_paths.append(path)
        path.write_bytes(b"cache")

    monkeypatch.setattr(torch, "save", fake_save)
    for _ in range(2):
        plaid_datasets._save_plaid_static_graph_cache(
            cache_file,
            train_graphs=[],
            val_graphs=[],
            test_graphs=[],
            metadata={},
        )

    assert len(set(temp_paths)) == 2
    assert all(path.parent == cache_file.parent for path in temp_paths)
    assert all(path.name.startswith(f".{cache_file.name}.") for path in temp_paths)


def test_precompute_rejects_unsupported_dataset_before_cuda_checks(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported static PLAID dataset"):
        _module().precompute_static_plaid_parallel(
            datasets=["plaid_el_pl_dynamics"],
            data_root=str(tmp_path),
            split_seed=0,
            laplacian_eig_dim=64,
            laplacian_spec="graph",
            num_gpus=4,
        )


def test_precompute_existing_valid_k0_cache_bypasses_materializing_loader(
    capsys, monkeypatch, tmp_path
) -> None:
    module = _module()
    cache_file = tmp_path / "graphs.pt"
    torch.save(
        {
            "version": 1,
            "train_graphs": [_graph(2)],
            "val_graphs": [_graph(7)],
        },
        cache_file,
    )

    class Service:
        def is_complete(self, dataset, split_group, sample_ids, spec, K):
            del dataset, split_group, sample_ids, spec, K
            return True

    class Process:
        exitcode = 0

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class Context:
        def Process(self, **kwargs):
            del kwargs
            return Process()

    def materializing_loader(**kwargs):
        del kwargs
        raise AssertionError("existing K0 cache must bypass materializing loader")

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.mp, "get_context", lambda method: Context())
    monkeypatch.setattr(module, "_plaid_static_graph_cache_file", lambda **kwargs: cache_file)
    monkeypatch.setattr(module, "_plaid_laplacian_service", lambda dataset_dir: Service())
    monkeypatch.setattr(module, "load_mesh_static_dataset", materializing_loader)

    module.precompute_static_plaid_parallel(
        datasets=["plaid_tensile2d"],
        data_root=str(tmp_path),
        split_seed=0,
        laplacian_eig_dim=64,
        laplacian_spec="graph",
        num_gpus=1,
    )

    output = capsys.readouterr().out
    assert "scheduled=0" in output
    assert "built=" not in output


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "train_graphs": [_graph(3)]},
        {
            "version": 1,
            "train_graphs": [_graph(3)],
            "val_graphs": [_graph(3)],
        },
    ],
    ids=["missing-split", "duplicate-id"],
)
def test_precompute_quarantines_malformed_k0_before_loader_rebuild(
    payload, monkeypatch, tmp_path
) -> None:
    module = _module()
    cache_file = tmp_path / "graphs.pt"
    torch.save(payload, cache_file)
    loader_saw_cache = []

    class Service:
        def is_complete(self, dataset, split_group, sample_ids, spec, K):
            del dataset, split_group, sample_ids, spec, K
            return True

    class Process:
        exitcode = 0

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class Context:
        def Process(self, **kwargs):
            del kwargs
            return Process()

    def rebuilding_loader(**kwargs):
        del kwargs
        loader_saw_cache.append(cache_file.exists())
        torch.save(
            {
                "version": 1,
                "train_graphs": [_graph(2)],
                "val_graphs": [_graph(7)],
            },
            cache_file,
        )

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.mp, "get_context", lambda method: Context())
    monkeypatch.setattr(module, "_plaid_static_graph_cache_file", lambda **kwargs: cache_file)
    monkeypatch.setattr(module, "_plaid_laplacian_service", lambda dataset_dir: Service())
    monkeypatch.setattr(module, "load_mesh_static_dataset", rebuilding_loader)

    module.precompute_static_plaid_parallel(
        datasets=["plaid_tensile2d"],
        data_root=str(tmp_path),
        split_seed=0,
        laplacian_eig_dim=64,
        laplacian_spec="graph",
        num_gpus=1,
    )

    assert loader_saw_cache == [False]
    assert list(module._load_k0_index(cache_file)) == [2, 7]
    assert len(list(tmp_path.glob(f".{cache_file.name}.invalid.*"))) == 1
