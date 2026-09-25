#!/usr/bin/env python3
"""Fill missing per-sample Laplacian LMDB entries (sequential GPU compute + direct writes)."""

from __future__ import annotations

import argparse

import torch

from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id, load_graph_cache_sample
from pdebench.dataset.ginot.io import load_raw_dataset
from pdebench.dataset.ginot.loader import load_ginot_precompute_context
from pdebench.dataset.ginot.mesh import normalize_cells, select_cells
from pdebench.dataset.laplacian import (
    compute_laplacian_eigendecomp_part,
    compute_laplacian_fem_eigendecomp_both,
    laplacian_eigen_payload_bytes,
    single_laplacian_spec_part,
    split_laplacian_lmdb_has,
    write_split_laplacian_shard_batch,
)


def _missing_ids(graph_cache_dir: str, indices: list[int], k: int, spec: str) -> list[int]:
    return [int(i) for i in indices if not split_laplacian_lmdb_has(graph_cache_dir, int(i), k, spec)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bracket_lug")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--laplacian-spec", default="graph:64")
    parser.add_argument("--laplacian-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    name, count = single_laplacian_spec_part(args.laplacian_spec, args.laplacian_dim)
    device = torch.device(args.device)

    train, test, _ = load_ginot_precompute_context(args.dataset, args.data_root, split_seed=args.split_seed)
    raw = load_raw_dataset(args.dataset, args.data_root) if name.startswith("fem") else None
    todo: list[tuple[str, str, list[int]]] = [
        ("train", train.graph_cache_dir, list(train.indices)),
        ("test", test.graph_cache_dir, list(test.indices)),
    ]

    for split_name, graph_cache_dir, indices in todo:
        missing = _missing_ids(graph_cache_dir, indices, args.laplacian_dim, args.laplacian_spec)
        if not missing:
            print(f"{split_name}: complete ({len(indices)} samples)")
            continue
        print(f"{split_name}: filling {len(missing)} missing samples on {device} ...", flush=True)
        for idx in missing:
            sample = load_graph_cache_sample(graph_cache_dir, int(idx))
            pos = sample["pos"].to(device=device, dtype=torch.float32)
            edge_index = sample["edge_index"].to(device=device)
            cells = None
            if raw is not None and raw.cells is not None:
                selected_cells = select_cells(raw, int(idx))
                if selected_cells is not None:
                    cells_np = normalize_cells(selected_cells, num_nodes=int(pos.shape[0]))
                    if cells_np is not None:
                        cells = torch.as_tensor(cells_np, dtype=torch.long, device=device)
            if name in {"fem_u", "fem_v"}:
                eigenvalues, eigenvectors_u, eigenvectors_v = compute_laplacian_fem_eigendecomp_both(pos, cells, count)
                eigenvectors = eigenvectors_u if name == "fem_u" else eigenvectors_v
            else:
                eigenvalues, eigenvectors = compute_laplacian_eigendecomp_part(
                    edge_index, pos, cells, name, count
                )
                eigenvectors_u = None
                eigenvectors_v = None
            shard_id = graph_cache_shard_id(graph_cache_dir, int(idx))
            payload = laplacian_eigen_payload_bytes(
                eigenvalues,
                eigenvectors,
                eigenvectors_u=eigenvectors_u,
                eigenvectors_v=eigenvectors_v,
            )
            n = write_split_laplacian_shard_batch(
                graph_cache_dir,
                shard_id,
                args.laplacian_dim,
                args.laplacian_spec,
                [(int(idx), payload)],
            )
            print(f"  sample_id={idx} shard={shard_id} wrote={n}", flush=True)

        still = _missing_ids(graph_cache_dir, indices, args.laplacian_dim, args.laplacian_spec)
        if still:
            raise RuntimeError(f"{split_name} still missing {len(still)} samples: {still[:10]}")
        print(f"{split_name}: done", flush=True)


if __name__ == "__main__":
    main()
