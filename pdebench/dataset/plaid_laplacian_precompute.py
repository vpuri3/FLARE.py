"""Multi-GPU Laplacian precompute for static PLAID datasets."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from pdebench.dataset.plaid_datasets import (
    PLAID_SPECS,
    _plaid_static_graph_cache_file,
    load_mesh_static_dataset,
)
from pdebench.dataset.plaid_laplacian import (
    _ensure_plaid_laplacian,
    _plaid_laplacian_service,
    _plaid_laplacian_split_group,
)

SUPPORTED_DATASETS = frozenset({"plaid_tensile2d", "plaid_hyperelasticity"})


def _partition_sample_ids(sample_ids: list[int], num_workers: int) -> list[list[int]]:
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive")
    chunks = [[] for _ in range(int(num_workers))]
    for offset, sample_id in enumerate(sorted(int(item) for item in sample_ids)):
        chunks[offset % int(num_workers)].append(sample_id)
    return chunks


def _index_k0_payload(payload: dict[str, Any]) -> dict[int, Any]:
    split_names = ("train_graphs", "val_graphs")
    if any(not isinstance(payload.get(split_name), list) for split_name in split_names):
        raise ValueError("Invalid static PLAID K0 cache payload: missing graph split lists")
    index: dict[int, Any] = {}
    for split_name in split_names:
        for graph in payload[split_name]:
            sample_id = int(graph.sample_id)
            if sample_id in index:
                raise ValueError(f"Duplicate K0 sample ID {sample_id}")
            index[sample_id] = graph
    return dict(sorted(index.items()))


def _load_k0_index(cache_file: str | Path) -> dict[int, Any]:
    payload = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or int(payload.get("version", -1)) != 1:
        raise ValueError(f"Invalid static PLAID K0 cache: {cache_file}")
    return _index_k0_payload(payload)


def _quarantine_invalid_k0_cache(cache_file: str | Path) -> Path | None:
    cache_file = Path(cache_file)
    quarantine = cache_file.parent / (
        f".{cache_file.name}.invalid.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        os.replace(cache_file, quarantine)
    except FileNotFoundError:
        return None
    return quarantine


def _filter_missing_ids(
    sample_ids: list[int],
    *,
    service: Any,
    dataset_name: str,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> list[int]:
    split_group = _plaid_laplacian_split_group(laplacian_spec, laplacian_eig_dim)
    return [
        sample_id
        for sample_id in sorted(int(item) for item in sample_ids)
        if not service.is_complete(
            dataset_name,
            split_group,
            sample_ids=[sample_id],
            spec=laplacian_spec,
            K=int(laplacian_eig_dim),
        )
    ]


def _worker_entry(
    *,
    gpu_id: int,
    sample_ids: list[int],
    k0_cache_file: str,
    dataset_dir: str,
    dataset_name: str,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> None:
    torch.cuda.set_device(int(gpu_id))
    graphs = _load_k0_index(k0_cache_file)
    device = f"cuda:{int(gpu_id)}"
    for sample_id in sample_ids:
        graph = graphs[int(sample_id)]
        _ensure_plaid_laplacian(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            sample_idx=int(sample_id),
            edge_index=graph.edge_index,
            pos=graph.pos,
            cells=graph.cells,
            num_eigenvectors=int(laplacian_eig_dim),
            laplacian_spec=laplacian_spec,
            device=device,
        )


def _start_and_join_processes(processes: list[Any]) -> None:
    started: list[Any] = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        for process in started:
            process.join()
    except BaseException:
        live_started = [process for process in started if process.is_alive()]
        for process in live_started:
            try:
                process.terminate()
            except BaseException:
                pass
        for process in live_started:
            try:
                process.join()
            except BaseException:
                pass
        raise


def _validate_request(datasets: list[str], laplacian_eig_dim: int, num_gpus: int) -> None:
    unsupported = sorted(set(datasets) - SUPPORTED_DATASETS)
    if unsupported:
        raise ValueError(f"Unsupported static PLAID dataset(s): {', '.join(unsupported)}")
    if not datasets:
        raise ValueError("At least one static PLAID dataset is required")
    if int(laplacian_eig_dim) <= 0:
        raise ValueError("laplacian_eig_dim must be positive")
    if int(num_gpus) <= 0:
        raise ValueError("num_gpus must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Static PLAID Laplacian precompute requires CUDA")
    available = int(torch.cuda.device_count())
    if int(num_gpus) > available:
        raise ValueError(f"Requested {num_gpus} GPUs, but only {available} are visible")


def precompute_static_plaid_parallel(
    datasets: list[str],
    data_root: str,
    split_seed: int,
    laplacian_eig_dim: int,
    laplacian_spec: str,
    num_gpus: int,
) -> None:
    _validate_request(datasets, laplacian_eig_dim, num_gpus)
    context = mp.get_context("spawn")

    for dataset_name in datasets:
        spec = PLAID_SPECS[dataset_name]
        dataset_dir = Path(data_root) / spec.folder
        k0_cache_file = _plaid_static_graph_cache_file(
            dataset_dir=str(dataset_dir),
            dataset_name=dataset_name,
            split_seed=int(split_seed),
            graph_backend="pyg",
            use_sdf_features=True,
            max_samples=0,
            load_public_test=False,
            laplacian_eig_dim=0,
            laplacian_spec=laplacian_spec,
        )
        try:
            k0_index = _load_k0_index(k0_cache_file)
        except (FileNotFoundError, ValueError) as error:
            quarantine = (
                _quarantine_invalid_k0_cache(k0_cache_file)
                if isinstance(error, ValueError)
                else None
            )
            try:
                load_mesh_static_dataset(
                    dataset_name=dataset_name,
                    data_root=data_root,
                    split_seed=int(split_seed),
                    graph_backend="pyg",
                    use_sdf_features=True,
                    max_samples=0,
                    load_public_test=False,
                    laplacian_eig_dim=0,
                    laplacian_spec=laplacian_spec,
                )
            except BaseException:
                if quarantine is not None and not k0_cache_file.exists():
                    os.replace(quarantine, k0_cache_file)
                raise
            k0_index = _load_k0_index(k0_cache_file)
        sample_ids = list(k0_index)
        missing_ids = _filter_missing_ids(
            sample_ids,
            service=_plaid_laplacian_service(str(dataset_dir)),
            dataset_name=dataset_name,
            laplacian_eig_dim=int(laplacian_eig_dim),
            laplacian_spec=laplacian_spec,
        )
        chunks = _partition_sample_ids(missing_ids, int(num_gpus))
        processes = [
            context.Process(
                target=_worker_entry,
                kwargs={
                    "gpu_id": gpu_id,
                    "sample_ids": chunk,
                    "k0_cache_file": str(k0_cache_file),
                    "dataset_dir": str(dataset_dir),
                    "dataset_name": dataset_name,
                    "laplacian_eig_dim": int(laplacian_eig_dim),
                    "laplacian_spec": laplacian_spec,
                },
            )
            for gpu_id, chunk in enumerate(chunks)
        ]
        _start_and_join_processes(processes)
        failures = [process.exitcode for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"Static PLAID worker failure(s) for {dataset_name}: {failures}")
        print(
            f"{dataset_name}: already cached={len(sample_ids) - len(missing_ids)}, "
            f"scheduled={len(missing_ids)}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(SUPPORTED_DATASETS),
        choices=sorted(SUPPORTED_DATASETS),
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--laplacian-eig-dim", type=int, default=64)
    parser.add_argument("--laplacian-spec", default="graph")
    parser.add_argument("--num-gpus", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    precompute_static_plaid_parallel(
        datasets=list(args.datasets),
        data_root=args.data_root,
        split_seed=args.split_seed,
        laplacian_eig_dim=args.laplacian_eig_dim,
        laplacian_spec=args.laplacian_spec,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
