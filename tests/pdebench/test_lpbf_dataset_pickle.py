"""LPBF FLARE dataset must be picklable for DataLoader num_workers > 0."""

from __future__ import annotations

import pickle

import torch
from torch.utils.data import DataLoader

from pdebench.dataset.lpbf import LPBFDataset, create_lpbf_dataset


def test_lpbf_dataset_class_is_module_level() -> None:
    assert LPBFDataset.__module__ == "pdebench.dataset.lpbf"
    assert "<locals>" not in LPBFDataset.__qualname__


def test_create_lpbf_dataset_roundtrips_pickle() -> None:
    ds = create_lpbf_dataset(split="train")
    assert isinstance(ds, LPBFDataset)
    blob = pickle.dumps(ds)
    restored = pickle.loads(blob)
    assert len(restored) == len(ds)
    g0 = restored.get(0)
    assert g0.pos.ndim == 2
    assert g0.y.ndim == 1


def test_lpbf_dataloader_with_workers_fetches_batch() -> None:
    ds = create_lpbf_dataset(split="train")
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=lambda xs: xs[0])
    graph = next(iter(loader))
    assert isinstance(graph.pos, torch.Tensor)
    assert graph.pos.shape[0] == graph.y.shape[0]
