#!/usr/bin/env python3
"""Detailed bottleneck profile for GINOT spectral precompute (single-GPU + pipeline)."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from collections import defaultdict

import torch

from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id, load_graph_cache_sample
from pdebench.dataset.ginot.io import load_raw_dataset
from pdebench.dataset.ginot.loader import load_ginot_precompute_context
from pdebench.dataset.ginot.mesh import normalize_cells, select_cells
from pdebench.dataset.laplacian import (
    DEFAULT_LAPLACIAN_SPECS,
    compute_laplacian_eigendecomp_part,
    laplacian_eigen_payload_bytes,
    laplacian_spec_entries,
    list_split_laplacian_cached_sample_ids,
    single_laplacian_spec_part,
    write_split_laplacian_shard_batch,
)


def _pick_missing_sample_ids(graph_cache_dir: str, indices: list[int], specs: list[str], dim: int, n: int) -> list[int]:
    missing_any: set[int] = set()
    index_set = set(int(i) for i in indices)
    for spec in specs:
        missing_any |= index_set - list_split_laplacian_cached_sample_ids(graph_cache_dir, dim, spec)
    ordered = [int(i) for i in indices if int(i) in missing_any]
    return ordered[: int(n)]


def profile_single_gpu(
    graph_cache_dir: str,
    dataset_name: str,
    data_root: str,
    sample_ids: list[int],
    specs: list[str],
    dim: int,
    device: torch.device,
) -> dict[str, float]:
    raw = load_raw_dataset(dataset_name, data_root)
    totals = defaultdict(float)
    per_spec = defaultdict(float)
    counts = defaultdict(int)

    for idx in sample_ids:
        t0 = time.perf_counter()
        gs = load_graph_cache_sample(graph_cache_dir, idx)
        pos = gs["pos"].to(device=device, dtype=torch.float32)
        ei = gs["edge_index"].to(device=device)
        cells = None
        sc = select_cells(raw, idx)
        if sc is not None:
            cn = normalize_cells(sc, num_nodes=int(pos.shape[0]))
            if cn is not None:
                cells = torch.as_tensor(cn, dtype=torch.long, device=device)
        torch.cuda.synchronize()
        totals["load"] += time.perf_counter() - t0

        graph_start = None
        batch: dict[tuple[str, int], list[tuple[int, bytes]]] = defaultdict(list)
        for spec in specs:
            name, count = single_laplacian_spec_part(spec, dim)
            init = None if name == "graph" else graph_start
            t1 = time.perf_counter()
            ev, evec = compute_laplacian_eigendecomp_part(ei, pos, cells, name, count, init_eigenvectors=init)
            torch.cuda.synchronize()
            per_spec[name] += time.perf_counter() - t1
            counts[name] += 1
            if name == "graph":
                graph_start = evec
            t2 = time.perf_counter()
            payload = laplacian_eigen_payload_bytes(ev, evec)
            torch.cuda.synchronize()
            totals["serialize"] += time.perf_counter() - t2
            shard_id = graph_cache_shard_id(graph_cache_dir, int(idx))
            batch[(spec, shard_id)].append((int(idx), payload))

        t3 = time.perf_counter()
        for (spec, shard_id), items in batch.items():
            write_split_laplacian_shard_batch(graph_cache_dir, shard_id, dim, spec, items)
        totals["write"] += time.perf_counter() - t3
        counts["samples"] += 1

    out = {k: float(v) for k, v in totals.items()}
    out["samples"] = float(counts["samples"])
    for name in ("graph", "edge", "fem_v"):
        if counts[name]:
            out[f"compute_{name}"] = per_spec[name] / counts[name]
    return out


def profile_cprofile_one_sample(
    graph_cache_dir: str,
    dataset_name: str,
    data_root: str,
    sample_id: int,
    specs: list[str],
    dim: int,
    device: torch.device,
) -> str:
    raw = load_raw_dataset(dataset_name, data_root)

    def _run() -> None:
        gs = load_graph_cache_sample(graph_cache_dir, sample_id)
        pos = gs["pos"].to(device=device, dtype=torch.float32)
        ei = gs["edge_index"].to(device=device)
        cells = None
        sc = select_cells(raw, sample_id)
        if sc is not None:
            cn = normalize_cells(sc, num_nodes=int(pos.shape[0]))
            if cn is not None:
                cells = torch.as_tensor(cn, dtype=torch.long, device=device)
        graph_start = None
        for spec in specs:
            name, count = single_laplacian_spec_part(spec, dim)
            init = None if name == "graph" else graph_start
            ev, evec = compute_laplacian_eigendecomp_part(ei, pos, cells, name, count, init_eigenvectors=init)
            if name == "graph":
                graph_start = evec
            laplacian_eigen_payload_bytes(ev, evec)
        torch.cuda.synchronize()

    prof = cProfile.Profile()
    prof.enable()
    _run()
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(30)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="poisson_unstructured")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cprofile-sample", type=int, default=0, help="If set, run cProfile on this sample id.")
    args = parser.parse_args()

    train, _, _ = load_ginot_precompute_context(args.dataset, args.data_root)
    assert train.graph_cache_dir is not None
    specs = laplacian_spec_entries(DEFAULT_LAPLACIAN_SPECS, 64)
    sample_ids = _pick_missing_sample_ids(train.graph_cache_dir, list(train.indices), specs, 64, args.num_samples)
    if not sample_ids:
        print("No missing samples in train split for profiling.")
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"Profiling {len(sample_ids)} missing train samples on {device}: {sample_ids[:5]}...")
    t0 = time.perf_counter()
    stats = profile_single_gpu(train.graph_cache_dir, args.dataset, args.data_root, sample_ids, specs, 64, device)
    elapsed = time.perf_counter() - t0
    n = max(int(stats["samples"]), 1)

    print("\n=== Single-GPU sequential breakdown (per sample averages) ===")
    print(f"  samples: {n}")
    print(f"  wall: {elapsed:.2f}s ({n / elapsed:.2f} samples/s)")
    print(f"  load:       {stats.get('load', 0) / n:.3f}s")
    print(f"  graph:      {stats.get('compute_graph', 0):.3f}s")
    print(f"  edge:       {stats.get('compute_edge', 0):.3f}s")
    print(f"  fem_v:      {stats.get('compute_fem_v', 0):.3f}s")
    compute_total = stats.get("compute_graph", 0) + stats.get("compute_edge", 0) + stats.get("compute_fem_v", 0)
    print(f"  compute Σ:  {compute_total:.3f}s")
    print(f"  serialize:  {stats.get('serialize', 0) / n:.3f}s")
    print(f"  write:      {stats.get('write', 0) / n:.3f}s")
    non_compute = stats.get("load", 0) / n + stats.get("serialize", 0) / n + stats.get("write", 0) / n
    print(f"  non-compute:{non_compute:.3f}s")
    print(f"  bottleneck: {'LOBPCG/FEM' if compute_total >= max(non_compute, 1e-6) else 'I/O (load/write/serialize)'}")

    if args.cprofile_sample or sample_ids:
        sid = int(args.cprofile_sample or sample_ids[0])
        print(f"\n=== cProfile cumulative top (sample {sid}) ===")
        print(profile_cprofile_one_sample(train.graph_cache_dir, args.dataset, args.data_root, sid, specs, 64, device))


if __name__ == "__main__":
    main()
