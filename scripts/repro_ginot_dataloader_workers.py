#!/usr/bin/env python3
"""Reproduce GINOT DataLoader worker failures (run as: python scripts/repro_ginot_dataloader_workers.py)."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from pdebench.dataset.ginot.loader import load_ginot_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bracket_lug")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--laplacian-spec", default="edge")
    parser.add_argument("--laplacian-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    train, _test, meta = load_ginot_dataset(
        args.dataset,
        args.data_root,
        split_seed=0,
        laplacian_eig_dim=args.laplacian_dim,
        laplacian_spec=args.laplacian_spec,
        include_edges=True,
    )
    print(f"dataset={type(train).__name__} len={len(train)}", flush=True)

    bs = BatchSampler(RandomSampler(train), batch_size=args.batch_size, drop_last=False)
    kw: dict = {
        "batch_sampler": bs,
        "num_workers": args.num_workers,
        "collate_fn": meta["train_collate_fn"],
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kw["prefetch_factor"] = 4
        kw["persistent_workers"] = True
        kw["multiprocessing_context"] = mp.get_context("spawn")

    print(f"creating DataLoader num_workers={args.num_workers} mp={kw.get('multiprocessing_context')}", flush=True)
    dl = DataLoader(train, **kw)
    print("fetching first batch...", flush=True)
    batch = next(iter(dl))
    print("ok", {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in batch.items()}, flush=True)


if __name__ == "__main__":
    main()
