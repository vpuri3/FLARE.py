#!/usr/bin/env python3
"""Profile one GINOT spectral precompute sample (load / compute / save)."""

from __future__ import annotations

import argparse
import time

import torch

from pdebench.dataset.ginot.graph_cache import load_graph_cache_sample
from pdebench.dataset.ginot.io import load_raw_dataset
from pdebench.dataset.ginot.mesh import normalize_cells, select_cells
from pdebench.dataset.laplacian import (
    DEFAULT_LAPLACIAN_SPECS,
    SplitLaplacianWriteSession,
    compute_laplacian_eigendecomp_part,
    laplacian_spec_entries,
    save_split_laplacian_lmdb,
    single_laplacian_spec_part,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-cache-dir", required=True)
    parser.add_argument("--dataset", default="poisson_unstructured")
    parser.add_argument("--sample-id", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--session-save", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    specs = laplacian_spec_entries(DEFAULT_LAPLACIAN_SPECS, 64)
    raw = load_raw_dataset(args.dataset, "data")

    gs = load_graph_cache_sample(args.graph_cache_dir, args.sample_id)
    pos = gs["pos"].to(device=device, dtype=torch.float32)
    ei = gs["edge_index"].to(device=device)
    cells = None
    sc = select_cells(raw, args.sample_id)
    if sc is not None:
        cn = normalize_cells(sc, num_nodes=int(pos.shape[0]))
        if cn is not None:
            cells = torch.as_tensor(cn, dtype=torch.long, device=device)

    writer = SplitLaplacianWriteSession(args.graph_cache_dir, 64) if args.session_save else None
    graph_start = None
    for spec in specs:
        name, count = single_laplacian_spec_part(spec, 64)
        init = None if name == "graph" else graph_start
        t0 = time.perf_counter()
        ev, evec = compute_laplacian_eigendecomp_part(ei, pos, cells, name, count, init_eigenvectors=init)
        torch.cuda.synchronize()
        comp = time.perf_counter() - t0
        t1 = time.perf_counter()
        if writer is not None:
            writer.save(args.sample_id, spec, ev, evec)
        else:
            save_split_laplacian_lmdb(args.graph_cache_dir, args.sample_id, 64, spec, ev, evec)
        save_t = time.perf_counter() - t1
        print(f"{name}: compute={comp:.3f}s save={save_t:.3f}s")
        if name == "graph":
            graph_start = evec
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
