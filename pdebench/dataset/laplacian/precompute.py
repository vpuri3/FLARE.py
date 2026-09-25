"""Batch GPU precompute for GINOT graph and Laplacian spectral caches.

Graph caches are built via ``load_ginot_precompute_context`` (same as training).
Spectral precompute uses a **compute/write split**:

- GPU workers: LOBPCG only, batched handoff to writers (no LMDB writes).
- CPU writers: one persistent LMDB env per (spec, shard) lane; batched puts, sync on close.

Run from the repo root::

    python -m pdebench.dataset.laplacian.precompute --datasets poisson_unstructured

Defaults (override via CLI or out/pdebench/ginot_precompute_env.sh):

- ``--direct-writes`` on (``GINOT_PRECOMPUTE_DIRECT_WRITES``); ``--no-direct-writes`` for profiling.
- ``--staged-specs``: one Laplacian operator per pass (graph, then edge).
- ``--write-batch-size`` 64; per-dataset specs in ``laplacian.DATASET_LAPLACIAN_SPECS``.
- Profile caps stratify samples across graph shards (multi-GPU representative).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import torch
from tqdm import tqdm

from pdebench.dataset.ginot.deform_plate import compute_deform_plate_plate_laplacian_eigendecomp
from pdebench.dataset.ginot.graph_cache import get_sharded_lmdb_graph_cache, load_graph_cache_sample
from pdebench.dataset.ginot.io import load_raw_dataset
from pdebench.dataset.ginot.loader import load_ginot_precompute_context
from pdebench.dataset.ginot.mesh import normalize_cells, select_cells, topology_key_for_row
from pdebench.dataset.ginot.types import GINOT_DATASETS, GinotRawDataset
from pdebench.dataset.laplacian import (
    DEFAULT_LAPLACIAN_EIGENVECTORS,
    LmdbLaplacianBackend,
    compute_laplacian_eigendecomp_part,
    compute_laplacian_fem_eigendecomp_both,
)
from pdebench.dataset.laplacian.lmdb import (
    laplacian_eigen_payload_bytes,
    list_split_laplacian_cached_sample_ids,
    split_laplacian_lmdb_dir,
)
from pdebench.dataset.laplacian.spec import (
    laplacian_cache_spec_entry,
    resolve_laplacian_specs_for_dataset,
    single_laplacian_spec_part,
)

# Amortize GPU flush/sync; 64 helped when compute >> load (micro_puc / poisson graph spec).
DEFAULT_WRITE_BATCH_SIZE = 64
DEFAULT_MAX_CPU_WORKERS = 104
# Unbounded so GPU workers never block on enqueue while writers flush large batches.
WRITE_QUEUE_MAXSIZE = 0

# Datasets where topology_key_for_row can return a non-None key (shared-mesh reuse).
TOPOLOGY_KEY_DATASETS = frozenset({"poisson_structured", "micro_puc", "micro_puc_fixed"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute GINOT Laplacian spectral caches.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--datasets",
        default="poisson_unstructured,poisson_structured,bracket_lug,micro_puc,micro_puc_fixed,deform_plate,bumper_beam",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap train/test splits before graph and Laplacian precompute (0 = full split).",
    )
    parser.add_argument("--laplacian-dim", type=int, default=DEFAULT_LAPLACIAN_EIGENVECTORS)
    parser.add_argument(
        "--laplacian-specs",
        default=None,
        help="Comma-separated specs (e.g. graph:64,edge:64). Default: per-dataset (see laplacian.DATASET_LAPLACIAN_SPECS).",
    )
    parser.add_argument("--num-gpus", type=int, default=0, help="GPU compute workers. 0 => all visible GPUs.")
    parser.add_argument(
        "--num-writers",
        type=int,
        default=0,
        help="CPU LMDB writer processes during spectral precompute. 0 => min(lanes, 8*num_gpus). "
        "Do not match graph-cache worker counts; too many writers starve GPU compute.",
    )
    parser.add_argument(
        "--num-cpu-workers",
        type=int,
        default=0,
        help="CPU worker budget (writers + loaders). 0 => SLURM_CPUS_PER_TASK*num_gpus or 104 cap.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=DEFAULT_WRITE_BATCH_SIZE,
        help="Samples per GPU flush before enqueueing shard write batches.",
    )
    parser.add_argument(
        "--direct-writes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "GPU workers write LMDB directly (shard-partitioned). Default on; use --no-direct-writes "
            "for queue+CPU writers (better for small --profile-samples on one shard). "
            "Override default with GINOT_PRECOMPUTE_DIRECT_WRITES=0|1."
        ),
    )
    parser.add_argument(
        "--staged-specs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run one Laplacian spec per pass (graph then edge, etc.). Easier resume and balances GPUs "
            "on profile caps. Use --no-staged-specs to fuse all specs in one pass per sample."
        ),
    )
    parser.add_argument(
        "--skip-cache-scan",
        action="store_true",
        help="Do not scan existing Laplacian LMDB entries; recompute selected split samples for requested specs.",
    )
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=0,
        help="If >0, run a bounded spectral pass on at most this many missing samples per split then exit.",
    )
    parser.add_argument(
        "--profile-split",
        choices=("train", "test", "both"),
        default="train",
        help="Which split(s) to include when --profile-samples is set.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run GPU spectral compute without persisting Laplacian LMDB writes (for profiling only).",
    )
    return parser.parse_args()


def _resolve_direct_writes(requested: bool | None) -> bool:
    if requested is not None:
        return bool(requested)
    env = os.environ.get("GINOT_PRECOMPUTE_DIRECT_WRITES", "1").strip().lower()
    return env not in ("0", "false", "no")


def _stratified_profile_sample_ids(
    graph_cache_dir: str,
    sample_ids: list[int],
    cap: int,
    num_shards: int,
) -> list[int]:
    """Spread profile samples across graph shards so multi-GPU / direct_writes is representative."""
    from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id

    if cap <= 0 or len(sample_ids) <= cap:
        return list(sample_ids)
    by_shard: dict[int, list[int]] = {}
    for idx in sample_ids:
        shard = int(graph_cache_shard_id(graph_cache_dir, int(idx)))
        by_shard.setdefault(shard, []).append(int(idx))
    for shard in by_shard:
        by_shard[shard].sort()
    shard_order = sorted(by_shard)
    picked: list[int] = []
    while len(picked) < cap and shard_order:
        next_shards = []
        for shard in shard_order:
            bucket = by_shard[shard]
            if bucket:
                picked.append(bucket.pop(0))
                if len(picked) >= cap:
                    break
            if bucket:
                next_shards.append(shard)
        shard_order = next_shards
    return picked


def _resolve_num_cpu_workers(requested: int, num_gpus: int) -> int:
    if requested > 0:
        return int(requested)
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or "0")
    if slurm_cpus > 0:
        return min(DEFAULT_MAX_CPU_WORKERS, slurm_cpus * int(num_gpus))
    return min(DEFAULT_MAX_CPU_WORKERS, int(os.cpu_count() or 1))


def _graph_cache_num_shards(graph_cache_dir: str) -> int:
    shard_ids = get_sharded_lmdb_graph_cache(graph_cache_dir)._shard_by_sample.values()
    return max(1, len(set(int(s) for s in shard_ids)))


def _write_lanes(laplacian_specs: list[str], num_shards: int, laplacian_dim: int) -> list[tuple[str, int]]:
    lanes = {
        (laplacian_cache_spec_entry(spec, laplacian_dim), int(shard_id))
        for spec in laplacian_specs
        for shard_id in range(int(num_shards))
    }
    return sorted(lanes)


def _resolve_num_writers(
    requested: int,
    num_cpu_workers: int,
    num_lanes: int,
    *,
    num_gpus: int = 1,
) -> int:
    lanes = max(1, int(num_lanes))
    # LMDB writers are not GPU compute workers; ~8 per GPU is enough for shard lanes.
    default = min(lanes, max(8, int(num_gpus) * 8))
    if requested > 0:
        return max(1, min(int(requested), int(num_cpu_workers), lanes))
    return default


def _partition_lanes(lanes: list[tuple[str, int]], num_writers: int) -> list[list[tuple[str, int]]]:
    buckets = [[] for _ in range(max(1, int(num_writers)))]
    for lane_index, lane in enumerate(lanes):
        buckets[lane_index % len(buckets)].append(lane)
    return buckets


def _lane_queue_index(lanes: list[tuple[str, int]]) -> dict[tuple[str, int], int]:
    return {lane: index for index, lane in enumerate(lanes)}


def _partition_sample_ids(sample_ids: list[int], num_parts: int) -> list[list[int]]:
    buckets = [[] for _ in range(max(1, int(num_parts)))]
    for i, sample_id in enumerate(sample_ids):
        buckets[i % len(buckets)].append(int(sample_id))
    return buckets


def _partition_sample_ids_by_graph_shard(graph_cache_dir: str, sample_ids: list[int], num_parts: int) -> list[list[int]]:
    from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id

    buckets = [[] for _ in range(max(1, int(num_parts)))]
    for sample_id in sample_ids:
        shard_id = graph_cache_shard_id(graph_cache_dir, int(sample_id))
        buckets[int(shard_id) % len(buckets)].append(int(sample_id))
    return buckets


def compute_or_reuse_graph_payload(
    memo: dict[int, tuple[bytes, torch.Tensor, torch.Tensor]],
    topo_key: int | None,
    compute_fn: Callable[[], tuple[bytes, torch.Tensor, torch.Tensor]],
) -> tuple[bytes, torch.Tensor, torch.Tensor]:
    """Compute a graph Laplacian payload + eigenpair once per topology key, reusing it for repeats.

    ``compute_fn`` runs only on a cache miss and must return
    ``(payload_bytes, eigenvalues, eigenvectors)``. When ``topo_key`` is ``None`` (no
    shared topology, e.g. edge/fem specs or datasets without a topology notion) the
    result is always recomputed and never cached.

    The eigenvalues/eigenvectors are memoized alongside the payload (not just returned
    on a miss) so that cache *hits* also restore them -- this lets callers set
    ``graph_start`` from the returned eigenvectors on every call, hit or miss, which
    matters for fused multi-spec passes where a later edge/fem spec on the same sample
    warm-starts from the graph spec's eigenvectors.
    """
    if topo_key is not None:
        cached = memo.get(topo_key)
        if cached is not None:
            return cached
    result = compute_fn()
    if topo_key is not None:
        memo[topo_key] = result
    return result


def group_sample_ids_by_topology(
    raw: GinotRawDataset | None,
    dataset_name: str,
    sample_ids: list[int],
) -> dict[int, list[int]]:
    """Group sample ids that share a graph Laplacian topology key.

    Rows without a topology key (``topology_key_for_row`` returns ``None``,
    e.g. ``raw`` is unavailable or the dataset has no shared-mesh notion)
    each get their own singleton group under a distinct negative key so they
    are never merged with an unrelated row.

    Not called by the GPU compute worker (which memoizes per-sample via
    ``compute_or_reuse_graph_payload`` instead of pre-grouping). This is the
    group/representative API used by tests and reserved for future graph-only
    passes that want to iterate one representative per topology key.
    """
    groups: dict[int, list[int]] = {}
    next_singleton_key = -1
    for sample_id in sample_ids:
        idx = int(sample_id)
        key = topology_key_for_row(raw, dataset_name, idx) if raw is not None else None
        if key is None:
            groups[next_singleton_key] = [idx]
            next_singleton_key -= 1
        else:
            groups.setdefault(int(key), []).append(idx)
    for members in groups.values():
        members.sort()
    return groups


def _indices_missing_specs(
    graph_cache_dir: str,
    indices: list[int],
    specs: list[str],
    laplacian_dim: int,
) -> dict[str, list[int]]:
    index_set = {int(idx) for idx in indices}
    return {
        spec: sorted(index_set - list_split_laplacian_cached_sample_ids(graph_cache_dir, laplacian_dim, spec))
        for spec in specs
    }


def _flush_write_buffer(
    write_buffer: dict[tuple[str, int], list[tuple[int, bytes]]],
    lane_queues: list[mp.Queue],
    lane_queue_index: dict[tuple[str, int], int],
) -> int:
    """Enqueue shard write batches to per-lane queues."""
    flushed = 0
    for (spec, shard_id), items in list(write_buffer.items()):
        if not items:
            continue
        lane_queues[lane_queue_index[(str(spec), int(shard_id))]].put(items)
        flushed += len(items)
    write_buffer.clear()
    return flushed


def _load_graph_sample_cpu(
    graph_cache_dir: str,
    idx: int,
    raw: GinotRawDataset | None,
    needs_cells: bool,
) -> tuple[dict[str, Any], float, Any | None]:
    t0 = time.perf_counter()
    graph_sample = load_graph_cache_sample(graph_cache_dir, idx)
    cells = None
    if needs_cells and raw is not None and raw.cells is not None:
        selected_cells = select_cells(raw, idx)
        if selected_cells is not None:
            cells_np = normalize_cells(selected_cells, num_nodes=int(graph_sample["pos"].shape[0]))
            if cells_np is not None:
                cells = cells_np
    return graph_sample, time.perf_counter() - t0, cells


def _transfer_graph_sample_to_device(
    graph_sample: dict[str, Any],
    cells_np: Any | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pos = graph_sample["pos"].to(device=device, dtype=torch.float32, non_blocking=True)
    edge_index = graph_sample["edge_index"].to(device=device, non_blocking=True)
    cells = None
    if cells_np is not None:
        cells = torch.as_tensor(cells_np, dtype=torch.long, device=device)
    return pos, edge_index, cells


def _summarize_pass_metrics(
    totals: dict[str, float],
    *,
    num_gpus: int,
    write_batch_size: int,
    no_write: bool,
) -> None:
    samples = max(int(totals.get("samples", 0)), 1)
    elapsed = max(float(totals.get("elapsed", 0.0)), 1e-9)
    load = float(totals.get("load", 0.0))
    compute = float(totals.get("compute", 0.0))
    write = float(totals.get("write", 0.0))
    gpu_seconds = load + compute
    gpu_util = gpu_seconds / (elapsed * max(int(num_gpus), 1))
    print(
        f"  profile: gpu_util≈{gpu_util:.2f} (ideal=1.0 per GPU) "
        f"load={load / samples:.3f}s compute={compute / samples:.3f}s "
        f"write={write:.3f}s total no_write={no_write}",
        flush=True,
    )
    if compute > 0 and load > 0:
        ratio = compute / load
        if ratio >= 8.0 and int(write_batch_size) < 64:
            suggested = min(128, max(int(write_batch_size) * 2, 64))
            print(
                f"  hint: compute dominates load ({ratio:.1f}x); try --write-batch-size {suggested} "
                f"to amortize flush/sync overhead.",
                flush=True,
            )
        elif load >= compute * 2.0:
            print(
                "  hint: load dominates compute; graph-cache reads are the bottleneck "
                "(shard-local batches help training, not precompute).",
                flush=True,
            )


def _flush_direct_write_buffer(
    write_buffer: dict[tuple[str, int], list[tuple[int, bytes]]],
    writer_session: Any,
) -> tuple[int, float]:
    flushed = 0
    write_time = 0.0
    for (spec, shard_id), items in list(write_buffer.items()):
        if not items:
            continue
        t0 = time.perf_counter()
        flushed += int(writer_session.write_batch(str(spec), int(shard_id), items))
        write_time += time.perf_counter() - t0
    write_buffer.clear()
    return flushed, write_time


def _gpu_compute_worker(
    dataset_name: str,
    data_root: str,
    graph_cache_dir: str,
    laplacian_specs: list[str],
    laplacian_dim: int,
    device_idx: int,
    sample_ids: list[int],
    write_batch_size: int,
    lane_queues: list[mp.Queue],
    lane_queue_index: dict[tuple[str, int], int],
    progress_queue: mp.Queue,
    direct_writes: bool,
    no_write: bool,
) -> None:
    needs_cells = any(single_laplacian_spec_part(spec, laplacian_dim)[0].startswith("fem") for spec in laplacian_specs)
    needs_raw = needs_cells or dataset_name == "deform_plate" or dataset_name in TOPOLOGY_KEY_DATASETS
    raw = load_raw_dataset(dataset_name, data_root) if needs_raw else None
    device = torch.device(f"cuda:{int(device_idx)}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id

    print(
        f"gpu_compute_start device={device} samples={len(sample_ids)} "
        f"specs={','.join(laplacian_specs)} write_batch_size={write_batch_size} "
        f"direct_writes={direct_writes} no_write={no_write}",
        flush=True,
    )
    write_buffer: dict[tuple[str, int], list[tuple[int, bytes]]] = {}
    samples_since_flush = 0
    direct_session = None
    if direct_writes and not no_write:
        assigned_shards = sorted({graph_cache_shard_id(graph_cache_dir, int(idx)) for idx in sample_ids})
        assigned_lanes = sorted({
            (laplacian_cache_spec_entry(spec, laplacian_dim), int(shard_id))
            for spec in laplacian_specs
            for shard_id in assigned_shards
        })
        direct_session = LmdbLaplacianBackend.open_bulk_writer_at(
            graph_cache_dir, laplacian_dim, assigned_lanes
        )

    def _flush_writes() -> None:
        nonlocal samples_since_flush
        if no_write:
            write_buffer.clear()
            samples_since_flush = 0
            return
        if direct_session is not None:
            n, write_time = _flush_direct_write_buffer(write_buffer, direct_session)
            progress_queue.put(("written", n, write_time))
        else:
            _flush_write_buffer(write_buffer, lane_queues, lane_queue_index)
        samples_since_flush = 0

    from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id as _shard_id

    sample_ids = sorted(sample_ids, key=lambda i: (_shard_id(graph_cache_dir, int(i)), int(i)))

    # graph Laplacians are shared across sample_ids with the same topology key (see
    # topology_key_for_row); compute once per key and replicate the payload. edge/fem
    # (and deform_plate, which routes through its own per-sample mesh path) stay per-sample.
    # This memo is per-worker-process (one dict per GPU compute worker), not shared
    # across GPUs; each worker only benefits from topology repeats within its own
    # sample_ids partition, which is intentional (no cross-process/cross-GPU sharing).
    graph_payload_by_topo: dict[int, tuple[bytes, torch.Tensor, torch.Tensor]] = {}

    try:
        with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
            pending = None
            if sample_ids:
                pending = prefetch_pool.submit(
                    _load_graph_sample_cpu,
                    graph_cache_dir,
                    int(sample_ids[0]),
                    raw,
                    needs_cells,
                )
            for sample_pos, idx in enumerate(sample_ids):
                assert pending is not None
                graph_sample, load_time, cells_np = pending.result()
                if sample_pos + 1 < len(sample_ids):
                    pending = prefetch_pool.submit(
                        _load_graph_sample_cpu,
                        graph_cache_dir,
                        int(sample_ids[sample_pos + 1]),
                        raw,
                        needs_cells,
                    )
                else:
                    pending = None

                pos, edge_index, cells = _transfer_graph_sample_to_device(graph_sample, cells_np, device)
                torch.cuda.current_stream(device).synchronize()

                graph_start = None
                fem_by_count: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                compute_time = 0.0
                for spec in laplacian_specs:
                    name, count = single_laplacian_spec_part(spec, laplacian_dim)
                    init = None if name == "graph" else graph_start

                    # graph Laplacians are the same for every sample_id sharing a topology
                    # key; edge/fem stay per-sample below regardless of topology. Only
                    # datasets in TOPOLOGY_KEY_DATASETS can return a non-None topology key
                    # (deform_plate is excluded by construction, not just by name check).
                    is_shareable_graph = (
                        name == "graph" and dataset_name in TOPOLOGY_KEY_DATASETS and raw is not None
                    )
                    topo_key = topology_key_for_row(raw, dataset_name, int(idx)) if is_shareable_graph else None

                    t1 = time.perf_counter()
                    payload = None
                    if is_shareable_graph and not no_write:

                        def _compute_graph_payload(
                            _name: str = name, _count: int = count, _init: torch.Tensor | None = init
                        ) -> tuple[bytes, torch.Tensor, torch.Tensor]:
                            eigvals, eigvecs = compute_laplacian_eigendecomp_part(
                                edge_index, pos, cells, _name, _count, init_eigenvectors=_init
                            )
                            payload_bytes = laplacian_eigen_payload_bytes(eigvals, eigvecs)
                            return payload_bytes, eigvals.detach().cpu(), eigvecs.detach().cpu()

                        payload, eigenvalues_cpu, eigenvectors_cpu = compute_or_reuse_graph_payload(
                            graph_payload_by_topo, topo_key, _compute_graph_payload
                        )
                        # Restore tensors on both hit and miss so graph_start below stays
                        # populated for later specs in a fused (non-staged) multi-spec pass.
                        eigenvalues = eigenvalues_cpu.to(device=device)
                        eigenvectors = eigenvectors_cpu.to(device=device)
                        eigenvectors_u = None
                        eigenvectors_v = None
                    elif dataset_name == "deform_plate":
                        eigenvalues, eigenvectors = compute_deform_plate_plate_laplacian_eigendecomp(
                            raw,
                            int(idx),
                            pos,
                            name,
                            count,
                            init_eigenvectors=init,
                        )
                        eigenvectors_u = None
                        eigenvectors_v = None
                    elif name in {"fem_u", "fem_v"}:
                        cached_fem = fem_by_count.get(int(count))
                        if cached_fem is None:
                            cached_fem = compute_laplacian_fem_eigendecomp_both(
                                pos,
                                cells,
                                count,
                                init_eigenvectors=init,
                            )
                            fem_by_count[int(count)] = cached_fem
                        eigenvalues, eigenvectors_u, eigenvectors_v = cached_fem
                        eigenvectors = eigenvectors_u if name == "fem_u" else eigenvectors_v
                    else:
                        eigenvalues, eigenvectors = compute_laplacian_eigendecomp_part(
                            edge_index,
                            pos,
                            cells,
                            name,
                            count,
                            init_eigenvectors=init,
                        )
                        eigenvectors_u = None
                        eigenvectors_v = None
                    compute_time += time.perf_counter() - t1
                    if name == "graph" and eigenvectors is not None:
                        graph_start = eigenvectors
                    if not no_write:
                        if payload is None:
                            payload = laplacian_eigen_payload_bytes(
                                eigenvalues,
                                eigenvectors,
                                eigenvectors_u=eigenvectors_u,
                                eigenvectors_v=eigenvectors_v,
                            )
                        shard_id = graph_cache_shard_id(graph_cache_dir, int(idx))
                        key = (laplacian_cache_spec_entry(spec, laplacian_dim), int(shard_id))
                        write_buffer.setdefault(key, []).append((int(idx), payload))

                samples_since_flush += 1
                if samples_since_flush >= int(write_batch_size):
                    _flush_writes()

                progress_queue.put(("computed", device_idx, int(idx), load_time, compute_time))

        _flush_writes()
        if not direct_writes and not no_write:
            for lane_queue in lane_queues:
                lane_queue.close()
                lane_queue.join_thread()
        progress_queue.put(("done", device_idx))
    except Exception:
        progress_queue.put(("error", device_idx, int(idx) if "idx" in locals() else -1, traceback.format_exc()))
        return
    finally:
        if direct_session is not None:
            direct_session.close()


def _cpu_lane_write_worker(
    graph_cache_dir: str,
    laplacian_dim: int,
    laplacian_spec: str,
    shard_id: int,
    lane_queue: mp.Queue,
    progress_queue: mp.Queue,
) -> None:
    session = LmdbLaplacianBackend.open_bulk_writer_at(
        graph_cache_dir, laplacian_dim, [(str(laplacian_spec), int(shard_id))]
    )
    try:
        while True:
            items = lane_queue.get()
            if items is None:
                break
            t0 = time.perf_counter()
            n = session.write_batch(laplacian_spec, int(shard_id), items)
            progress_queue.put(("written", n, time.perf_counter() - t0))
    except Exception as exc:
        progress_queue.put(("write_error", repr(exc)))
        return
    finally:
        session.close()
    progress_queue.put(("writer_done",))


def _cpu_multi_lane_write_worker(
    graph_cache_dir: str,
    laplacian_dim: int,
    assigned_lanes: list[tuple[str, int]],
    lane_queues: list[mp.Queue],
    lane_queue_index: dict[tuple[str, int], int],
    progress_queue: mp.Queue,
) -> None:
    session = LmdbLaplacianBackend.open_bulk_writer_at(graph_cache_dir, laplacian_dim, assigned_lanes)
    pending = {lane_queue_index[lane] for lane in assigned_lanes}
    try:
        while pending:
            for queue_idx in list(pending):
                try:
                    items = lane_queues[queue_idx].get(timeout=0.05)
                except queue.Empty:
                    continue
                if items is None:
                    pending.discard(queue_idx)
                    continue
                lane = next(lane for lane in assigned_lanes if lane_queue_index[lane] == queue_idx)
                laplacian_spec, shard_id = lane
                t0 = time.perf_counter()
                n = session.write_batch(laplacian_spec, int(shard_id), items)
                progress_queue.put(("written", n, time.perf_counter() - t0))
    except Exception as exc:
        progress_queue.put(("write_error", repr(exc)))
        return
    finally:
        session.close()
    progress_queue.put(("writer_done",))


def _run_compute_write_pass(
    dataset_name: str,
    data_root: str,
    graph_cache_dir: str,
    split_name: str,
    laplacian_specs: list[str],
    laplacian_dim: int,
    sample_ids: list[int],
    num_gpus: int,
    num_writers: int,
    write_batch_size: int,
    num_shards: int,
    direct_writes: bool,
    *,
    no_write: bool = False,
) -> dict[str, float]:
    if not sample_ids:
        print(f"  pass={','.join(laplacian_specs)} nothing missing")
        return {}

    ctx = mp.get_context("spawn")
    lanes = _write_lanes(laplacian_specs, num_shards, laplacian_dim)
    lane_queue_index = _lane_queue_index(lanes)
    lane_queues = [ctx.Queue(maxsize=WRITE_QUEUE_MAXSIZE) for _ in lanes]
    progress_queue: mp.Queue = ctx.Queue()

    partitions = (
        _partition_sample_ids_by_graph_shard(graph_cache_dir, sample_ids, num_gpus)
        if direct_writes
        else _partition_sample_ids(sample_ids, num_gpus)
    )
    gpu_workers = [
        ctx.Process(
            target=_gpu_compute_worker,
            args=(
                dataset_name,
                data_root,
                graph_cache_dir,
                laplacian_specs,
                laplacian_dim,
                device_idx,
                partition,
                write_batch_size,
                lane_queues,
                lane_queue_index,
                progress_queue,
                direct_writes,
                no_write,
            ),
        )
        for device_idx, partition in enumerate(partitions)
        if partition
    ]

    writer_workers: list[mp.Process] = []
    writer_partitions = _partition_lanes(lanes, num_writers)
    if not direct_writes and not no_write:
        for assigned_lanes in writer_partitions:
            if not assigned_lanes:
                continue
            if len(assigned_lanes) == 1:
                spec, shard_id = assigned_lanes[0]
                writer_workers.append(
                    ctx.Process(
                        target=_cpu_lane_write_worker,
                        args=(
                            graph_cache_dir,
                            laplacian_dim,
                            spec,
                            shard_id,
                            lane_queues[lane_queue_index[(spec, shard_id)]],
                            progress_queue,
                        ),
                    )
                )
            else:
                writer_workers.append(
                    ctx.Process(
                        target=_cpu_multi_lane_write_worker,
                        args=(
                            graph_cache_dir,
                            laplacian_dim,
                            assigned_lanes,
                            lane_queues,
                            lane_queue_index,
                            progress_queue,
                        ),
                    )
                )

    for worker in writer_workers + gpu_workers:
        worker.start()

    print(
        f"  compute/write: {len(gpu_workers)} GPU worker(s), {len(writer_workers)} CPU writer(s), "
        f"{len(lanes)} shard lane(s), write_batch_size={write_batch_size}, "
        f"direct_writes={direct_writes}, no_write={no_write}",
        flush=True,
    )

    totals: dict[str, float] = {"load": 0.0, "compute": 0.0, "write": 0.0}
    computed = 0
    written_entries = 0
    gpu_done = 0
    writers_done = 0
    start = time.perf_counter()

    def _handle_progress(msg: tuple[Any, ...]) -> bool:
        nonlocal computed, gpu_done, writers_done, written_entries
        tag = msg[0]
        if tag == "error":
            _, device_idx, idx, error = msg
            for worker in gpu_workers + writer_workers:
                worker.terminate()
            raise RuntimeError(f"GPU compute failed on cuda:{device_idx} sample={idx}: {error}")
        if tag == "write_error":
            for worker in gpu_workers + writer_workers:
                worker.terminate()
            raise RuntimeError(f"LMDB writer failed: {msg[1]}")
        if tag == "computed":
            _, _, _, load_time, compute_time = msg
            totals["load"] += float(load_time)
            totals["compute"] += float(compute_time)
            computed += 1
            pbar.update(1)
            if computed % 50 == 0:
                elapsed = max(time.perf_counter() - start, 1e-9)
                print(
                    f"  compute {computed}/{len(sample_ids)} {computed / elapsed:.2f} samples/s "
                    f"avg_load={totals['load'] / computed:.3f}s "
                    f"avg_compute={totals['compute'] / computed:.3f}s",
                    flush=True,
                )
        elif tag == "written":
            _, n, write_time = msg
            written_entries += int(n)
            totals["write"] += float(write_time)
        elif tag == "done":
            gpu_done += 1
        elif tag == "writer_done":
            writers_done += 1
        return True

    with tqdm(total=len(sample_ids), desc=f"{split_name} compute", ncols=110) as pbar:
        while gpu_done < len(gpu_workers):
            _handle_progress(progress_queue.get())

    if not direct_writes and not no_write:
        for lane_queue in lane_queues:
            lane_queue.put(None)

    while writers_done < len(writer_workers):
        _handle_progress(progress_queue.get())

    for worker in gpu_workers + writer_workers:
        worker.join(timeout=300)
        terminated = False
        if worker.is_alive():
            print(f"  terminating stuck worker pid={worker.pid}", flush=True)
            worker.terminate()
            worker.join(timeout=30)
            terminated = True
        if worker.exitcode not in (0, None) and not terminated:
            role = "gpu" if worker in gpu_workers else "writer"
            raise RuntimeError(f"{role} worker pid={worker.pid} exited with code {worker.exitcode}")

    elapsed = max(time.perf_counter() - start, 1e-9)
    totals["elapsed"] = elapsed
    totals["samples"] = float(len(sample_ids))
    totals["written_entries"] = float(written_entries)
    print(
        f"  split done: {len(sample_ids)} samples in {elapsed:.1f}s ({len(sample_ids) / elapsed:.2f} samples/s) "
        f"avg_load={totals['load'] / len(sample_ids):.3f}s "
        f"avg_compute={totals['compute'] / len(sample_ids):.3f}s "
        f"write_batches={written_entries} total_write_time={totals['write']:.1f}s",
        flush=True,
    )
    _summarize_pass_metrics(
        totals,
        num_gpus=len(gpu_workers),
        write_batch_size=write_batch_size,
        no_write=no_write,
    )
    return totals


def _precompute_split(
    dataset_name: str,
    data_root: str,
    dataset,
    split_name: str,
    laplacian_specs: list[str],
    laplacian_dim: int,
    num_gpus: int,
    num_writers: int,
    write_batch_size: int,
    direct_writes: bool,
    skip_cache_scan: bool,
    *,
    sample_cap: int | None = None,
    no_write: bool = False,
) -> dict[str, float] | None:
    if dataset.graph_cache_dir is None:
        raise RuntimeError(f"{split_name} graph cache directory is missing.")

    split_indices = list(dataset.indices)
    print(f"\nPhase 1/3: spectral cache scan for {dataset_name} split={split_name}")
    print(f"  graph cache dir: {dataset.graph_cache_dir}")
    print(f"  samples in split: {len(split_indices)}")
    if skip_cache_scan:
        missing_by_spec = {spec: list(split_indices) for spec in laplacian_specs}
        print("  skipping existing Laplacian cache scan; selected samples will be recomputed", flush=True)
    else:
        missing_by_spec = _indices_missing_specs(dataset.graph_cache_dir, split_indices, laplacian_specs, laplacian_dim)
    for spec, missing in missing_by_spec.items():
        cache_dir = split_laplacian_lmdb_dir(dataset.graph_cache_dir, laplacian_dim, spec)
        print(f"  spec={spec} missing={len(missing)} cache_dir={cache_dir}")

    if not any(missing_by_spec.values()):
        print(f"  phase 1 result: all spectral files already present for {split_name}")
        return None

    pass_missing = sorted({idx for missing in missing_by_spec.values() for idx in missing})
    if sample_cap is not None and int(sample_cap) > 0:
        pass_missing = _stratified_profile_sample_ids(
            dataset.graph_cache_dir,
            pass_missing,
            int(sample_cap),
            num_shards=_graph_cache_num_shards(dataset.graph_cache_dir),
        )
        print(
            f"  phase 1 result: profiling {len(pass_missing)} samples (cap={int(sample_cap)}, "
            f"stratified across graph shards)",
            flush=True,
        )
    else:
        print(f"  phase 1 result: {len(pass_missing)} samples need spectral features")
    print("Phase 2/3: GPU LOBPCG compute (partitioned across GPUs)")
    print("Phase 3/3: batched CPU LMDB shard writes")
    num_shards = _graph_cache_num_shards(dataset.graph_cache_dir)
    return _run_compute_write_pass(
        dataset_name,
        data_root,
        dataset.graph_cache_dir,
        split_name,
        laplacian_specs,
        laplacian_dim,
        pass_missing,
        num_gpus,
        num_writers,
        write_batch_size,
        num_shards,
        direct_writes,
        no_write=no_write,
    )


def _estimate_dataset_seconds(
    dataset_name: str,
    train_stats: dict[str, float] | None,
    test_stats: dict[str, float] | None,
    full_train: int,
    full_test: int,
    *,
    num_gpus: int,
    num_writers: int,
) -> float | None:
    profiles: list[tuple[str, dict[str, float], int]] = []
    if train_stats is not None and train_stats.get("samples", 0) > 0:
        profiles.append(("train", train_stats, full_train))
    if test_stats is not None and test_stats.get("samples", 0) > 0:
        profiles.append(("test", test_stats, full_test))
    if not profiles:
        return None

    best_name, best_stats, best_full = max(
        profiles,
        key=lambda item: float(item[1]["samples"]) / max(float(item[1]["elapsed"]), 1e-9),
    )
    rate = float(best_stats["samples"]) / max(float(best_stats["elapsed"]), 1e-9)
    if rate <= 0:
        return None
    total_samples = int(full_train) + int(full_test)
    est_total = total_samples / rate
    print(
        f"\n=== Estimate for {dataset_name} (full train+test, all specs) ===\n"
        f"  profile basis: {best_name} {best_stats['samples']:.0f} samples in {best_stats['elapsed']:.1f}s "
        f"({rate:.2f} samples/s, {int(num_gpus)} GPUs, {int(num_writers)} writers)\n"
        f"  avg_load={best_stats['load'] / best_stats['samples']:.3f}s "
        f"avg_compute={best_stats['compute'] / best_stats['samples']:.3f}s "
        f"write_time={best_stats.get('write', 0.0):.1f}s total\n"
        f"  extrapolated {total_samples} samples: {est_total / 60:.1f} min ({est_total / 3600:.2f} h)",
        flush=True,
    )
    if float(best_stats["samples"]) < 20:
        print("  note: profile used fewer than 20 samples; treat estimate as rough.", flush=True)
    return est_total


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Laplacian precompute requires CUDA.")

    available_gpus = torch.cuda.device_count()
    num_gpus = available_gpus if args.num_gpus <= 0 else min(args.num_gpus, available_gpus)
    if num_gpus <= 0:
        raise RuntimeError("Laplacian precompute requires at least one GPU (--num-gpus > 0).")

    num_cpu_workers = _resolve_num_cpu_workers(args.num_cpu_workers, num_gpus)
    direct_writes = _resolve_direct_writes(args.direct_writes)
    write_batch_size = max(1, int(args.write_batch_size))
    profile_samples = max(0, int(args.profile_samples))
    no_write = bool(args.no_write)
    profile_mode = profile_samples > 0
    staged_specs = bool(args.staged_specs)
    if profile_mode:
        print(
            f"Profile mode: up to {profile_samples} sample(s) per selected split "
            f"(split={args.profile_split}, no_write={no_write})",
            flush=True,
        )

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    for dataset_name in datasets:
        if dataset_name not in GINOT_DATASETS and dataset_name != "lpbf":
            raise ValueError(f"Unsupported dataset {dataset_name!r}")

        num_shards = None
        num_writers = args.num_writers

        print(f"\n=== Phase 0/3: loading {dataset_name} precompute context ===", flush=True)
        train_data, test_data, raw = load_ginot_precompute_context(
            dataset_name=dataset_name,
            data_root=args.data_root,
            split_seed=args.split_seed,
            max_samples=int(args.max_samples),
        )
        full_train = len(train_data.indices)
        full_test = len(test_data.indices)
        del raw

        laplacian_specs = resolve_laplacian_specs_for_dataset(
            dataset_name,
            args.laplacian_specs,
            args.laplacian_dim,
        )
        if not laplacian_specs:
            raise ValueError(f"No Laplacian operators resolved for dataset {dataset_name!r}.")
        spec_passes = [[spec] for spec in laplacian_specs] if staged_specs else [laplacian_specs]

        if train_data.graph_cache_dir is not None:
            num_shards = _graph_cache_num_shards(train_data.graph_cache_dir)
            num_lanes = len(_write_lanes(laplacian_specs, num_shards, args.laplacian_dim))
            num_writers = _resolve_num_writers(
                args.num_writers,
                num_cpu_workers,
                num_lanes,
                num_gpus=num_gpus,
            )

        print(
            f"=== Phase 0/3 complete: graph cache ready for {dataset_name} "
            f"(train={full_train}, test={full_test}) "
            f"gpus={num_gpus} writers={num_writers} cpu_budget={num_cpu_workers} "
            f"graph_shards={num_shards} write_batch_size={write_batch_size} "
            f"direct_writes={direct_writes} staged_specs={staged_specs} "
            f"specs={','.join(laplacian_specs)} ===",
            flush=True,
        )

        sample_cap = profile_samples if profile_mode else None
        run_train = (not profile_mode) or args.profile_split in ("train", "both")
        run_test = (not profile_mode) or args.profile_split in ("test", "both")

        train_stats = None
        test_stats = None
        for pass_specs in spec_passes:
            if len(spec_passes) > 1:
                print(f"\n--- Spectral pass: {','.join(pass_specs)} ---", flush=True)
            if run_train:
                train_stats = _precompute_split(
                    dataset_name,
                    args.data_root,
                    train_data,
                    "train",
                    pass_specs,
                    args.laplacian_dim,
                    num_gpus,
                    num_writers,
                    write_batch_size,
                    direct_writes,
                    args.skip_cache_scan,
                    sample_cap=sample_cap,
                    no_write=no_write,
                )
            if run_test:
                test_stats = _precompute_split(
                    dataset_name,
                    args.data_root,
                    test_data,
                    "test",
                    pass_specs,
                    args.laplacian_dim,
                    num_gpus,
                    num_writers,
                    write_batch_size,
                    direct_writes,
                    args.skip_cache_scan,
                    sample_cap=sample_cap,
                    no_write=no_write,
                )

        if not profile_mode:
            _estimate_dataset_seconds(
                dataset_name,
                train_stats,
                test_stats,
                full_train,
                full_test,
                num_gpus=num_gpus,
                num_writers=num_writers,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
