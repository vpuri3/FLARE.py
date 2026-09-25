#!/usr/bin/env python3
"""Profile the 4-GPU compute + batched-write precompute pipeline."""

from __future__ import annotations

import argparse
import time

from pdebench.dataset.ginot.loader import load_ginot_precompute_context
from pdebench.dataset.laplacian.precompute import _graph_cache_num_shards, _run_compute_write_pass
from pdebench.dataset.laplacian import DEFAULT_LAPLACIAN_SPECS, laplacian_spec_entries, list_split_laplacian_cached_sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="poisson_unstructured")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-writers", type=int, default=6)
    parser.add_argument("--write-batch-size", type=int, default=32)
    parser.add_argument(
        "--direct-writes",
        action="store_true",
        help="Write Laplacian LMDB from GPU workers (same as precompute --direct-writes).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Profile GPU compute only; skip LMDB persistence.",
    )
    args = parser.parse_args()

    train, _, _ = load_ginot_precompute_context(args.dataset, args.data_root)
    assert train.graph_cache_dir is not None
    specs = laplacian_spec_entries(DEFAULT_LAPLACIAN_SPECS, 64)
    index_set = {int(i) for i in train.indices}
    missing = sorted(index_set - list_split_laplacian_cached_sample_ids(train.graph_cache_dir, 64, specs[0]))[
        : int(args.num_samples)
    ]
    print(f"Pipeline profile: {len(missing)} missing train samples, {args.num_gpus} GPUs")
    t0 = time.perf_counter()
    num_shards = _graph_cache_num_shards(train.graph_cache_dir)
    num_writers = args.num_writers if args.num_writers > 0 else len(specs) * num_shards
    stats = _run_compute_write_pass(
        args.dataset,
        args.data_root,
        train.graph_cache_dir,
        "train",
        specs,
        64,
        missing,
        args.num_gpus,
        num_writers,
        args.write_batch_size,
        num_shards,
        args.direct_writes,
        no_write=args.no_write,
    )
    wall = time.perf_counter() - t0
    n = max(len(missing), 1)
    gpu_sec = stats["load"] + stats["compute"]
    print("\n=== Multi-GPU pipeline ===")
    print(f"  wall: {wall:.1f}s ({n / wall:.2f} samples/s)")
    print(f"  summed avg_load: {stats['load'] / n:.3f}s  summed avg_compute: {stats['compute'] / n:.3f}s")
    print(f"  write_time (parallel): {stats['write']:.1f}s")
    print(f"  GPU-seconds per wall second: {gpu_sec / wall:.2f} (ideal={args.num_gpus})")
    print(f"  compute is {100 * stats['compute'] / gpu_sec:.0f}% of summed GPU work")


if __name__ == "__main__":
    main()
