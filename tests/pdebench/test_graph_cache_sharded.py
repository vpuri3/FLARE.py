from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pdebench.dataset.ginot.features import attach_sample_feats
from pdebench.dataset.ginot.graph_cache import (
    ensure_graph_cache,
    ensure_graph_cache_length_index,
    graph_cache_num_shards,
    graph_cache_shard_id,
    is_lmdb_graph_cache,
    load_graph_cache_sample,
    max_padding_lengths_for_indices,
    node_lengths_for_indices,
)
from pdebench.dataset.ginot.sample import build_graph_sample_dict
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer


def _identity_normalizer(dim: int) -> StandardNormalizer:
    return StandardNormalizer(mean=torch.zeros(dim), std=torch.ones(dim))


def _mock_raw(num_samples: int) -> GinotRawDataset:
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    query_points = [
        np.random.default_rng(i).random((4, 2), dtype=np.float32) for i in range(num_samples)
    ]
    return GinotRawDataset(
        dataset_dir="/tmp/mock_graph_cache",
        query_points=query_points,
        point_clouds=query_points,
        targets=[np.zeros((4, 1), dtype=np.float32) for _ in range(num_samples)],
        cells=[cells for _ in range(num_samples)],
        input_params=None,
        target_fields=("y",),
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )


def test_ensure_graph_cache_writes_lmdb_layout(tmp_path) -> None:
    raw = _mock_raw(5)
    raw = GinotRawDataset(
        dataset_dir=str(tmp_path / "dataset"),
        query_points=raw.query_points,
        point_clouds=raw.point_clouds,
        targets=raw.targets,
        cells=raw.cells,
        input_params=None,
        target_fields=raw.target_fields,
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)

    cache_dir = ensure_graph_cache(
        raw=raw,
        dataset_name="poisson_unstructured",
        split_name="train",
        indices=[0, 1, 2, 3, 4],
        split_seed=0,
        dataset_split="default",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=pos_norm,
        y_normalizer=y_norm,
    )

    assert is_lmdb_graph_cache(cache_dir)
    assert (tmp_path / "dataset" / "static_cache" / "graph_lmdb").exists()
    shard_mdbs = list((Path(cache_dir) / "shards").rglob("data.mdb"))
    assert shard_mdbs
    assert (Path(cache_dir) / "index.pt").is_file()

    for idx in range(5):
        loaded = load_graph_cache_sample(cache_dir, idx)
        expected = build_graph_sample_dict(raw, idx, pos_norm, pos_norm, y_norm, to_cpu=True)
        assert torch.equal(loaded["edge_index"], expected["edge_index"])
        assert loaded["pos"].shape == expected["pos"].shape


def test_graph_cache_length_index_uses_lmdb_metadata(tmp_path) -> None:
    raw = _mock_raw(5)
    raw = GinotRawDataset(
        dataset_dir=str(tmp_path / "dataset"),
        query_points=raw.query_points,
        point_clouds=raw.point_clouds,
        targets=raw.targets,
        cells=raw.cells,
        input_params=None,
        target_fields=raw.target_fields,
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    cache_dir = ensure_graph_cache(
        raw=raw,
        dataset_name="poisson_unstructured",
        split_name="train",
        indices=[0, 1, 2, 3, 4],
        split_seed=0,
        dataset_split="default",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=pos_norm,
        y_normalizer=y_norm,
    )
    ensure_graph_cache_length_index(cache_dir)
    assert node_lengths_for_indices(cache_dir, [0, 2, 4]) == [4, 4, 4]
    pad_nodes, pad_boundary = max_padding_lengths_for_indices(cache_dir, [0, 1, 2, 3, 4])
    assert pad_nodes == 4
    assert pad_boundary == 4


def test_graph_cache_bakes_normalized_feats(tmp_path) -> None:
    rng = np.random.default_rng(0)
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    query_points = [rng.random((4, 2), dtype=np.float32) for _ in range(3)]
    input_params = rng.random((3, 3), dtype=np.float32)
    raw = GinotRawDataset(
        dataset_dir=str(tmp_path / "dataset"),
        query_points=query_points,
        point_clouds=query_points,
        targets=[np.zeros((4, 1), dtype=np.float32) for _ in range(3)],
        cells=[cells for _ in range(3)],
        input_params=input_params,
        target_fields=("y",),
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    feats_norm = StandardNormalizer(
        mean=torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        std=torch.tensor([[0.25, 0.25, 0.25]], dtype=torch.float32),
    )
    cache_dir = ensure_graph_cache(
        raw=raw,
        dataset_name="bracket_lug",
        split_name="train",
        indices=[0, 1, 2],
        split_seed=0,
        dataset_split="default",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=pos_norm,
        y_normalizer=y_norm,
        feats_normalizer=feats_norm,
    )

    for idx in range(3):
        loaded = load_graph_cache_sample(cache_dir, idx)
        expected = build_graph_sample_dict(
            raw,
            idx,
            pos_norm,
            pos_norm,
            y_norm,
            feats_normalizer=feats_norm,
            to_cpu=True,
        )
        assert "feats" in loaded
        assert torch.allclose(loaded["feats"], expected["feats"])


def test_attach_sample_feats_skips_cached_feats() -> None:
    cached_feats = torch.ones(4, 3)
    sample = {"pos": torch.zeros(4, 2), "feats": cached_feats.clone()}
    raw = GinotRawDataset(
        dataset_dir="/tmp/mock_graph_cache",
        query_points=[np.zeros((4, 2), dtype=np.float32)],
        point_clouds=[np.zeros((4, 2), dtype=np.float32)],
        targets=[np.zeros((4, 1), dtype=np.float32)],
        cells=None,
        input_params=np.zeros((1, 3), dtype=np.float32),
        target_fields=("y",),
        space_dim=2,
    )
    feats_norm = StandardNormalizer(
        mean=torch.zeros(1, 3),
        std=torch.ones(1, 3),
    )
    out = attach_sample_feats(sample, raw, 0, feats_norm)
    assert torch.equal(out["feats"], cached_feats)


def test_graph_cache_rejects_preprocessing_manifest_mismatch(tmp_path) -> None:
    raw = _mock_raw(5)
    raw = GinotRawDataset(
        dataset_dir=str(tmp_path / "dataset"),
        query_points=raw.query_points,
        point_clouds=raw.point_clouds,
        targets=raw.targets,
        cells=raw.cells,
        input_params=None,
        target_fields=raw.target_fields,
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    indices = [0, 1, 2, 3, 4]
    ensure_graph_cache(
        raw=raw,
        dataset_name="poisson_unstructured",
        split_name="train",
        indices=indices,
        split_seed=0,
        dataset_split="default",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=pos_norm,
        y_normalizer=y_norm,
    )

    shifted_y_norm = StandardNormalizer(mean=torch.ones(1), std=torch.ones(1))
    with pytest.raises(RuntimeError, match="different preprocessing metadata"):
        ensure_graph_cache(
            raw=raw,
            dataset_name="poisson_unstructured",
            split_name="train",
            indices=indices,
            split_seed=0,
            dataset_split="default",
            pos_normalizer=pos_norm,
            boundary_pos_normalizer=pos_norm,
            y_normalizer=shifted_y_norm,
        )


def test_graph_cache_rejects_feats_normalizer_mismatch(tmp_path) -> None:
    rng = np.random.default_rng(1)
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    query_points = [rng.random((4, 2), dtype=np.float32) for _ in range(2)]
    raw = GinotRawDataset(
        dataset_dir=str(tmp_path / "dataset"),
        query_points=query_points,
        point_clouds=query_points,
        targets=[np.zeros((4, 1), dtype=np.float32) for _ in range(2)],
        cells=[cells for _ in range(2)],
        input_params=rng.random((2, 3), dtype=np.float32),
        target_fields=("y",),
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    feats_norm = StandardNormalizer(mean=torch.zeros(1, 3), std=torch.ones(1, 3))
    indices = [0, 1]
    ensure_graph_cache(
        raw=raw,
        dataset_name="bracket_lug",
        split_name="train",
        indices=indices,
        split_seed=0,
        dataset_split="default",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=pos_norm,
        y_normalizer=y_norm,
        feats_normalizer=feats_norm,
    )
    shifted_feats = StandardNormalizer(mean=torch.ones(1, 3), std=torch.ones(1, 3))
    with pytest.raises(RuntimeError, match="different preprocessing metadata"):
        ensure_graph_cache(
            raw=raw,
            dataset_name="bracket_lug",
            split_name="train",
            indices=indices,
            split_seed=0,
            dataset_split="default",
            pos_normalizer=pos_norm,
            boundary_pos_normalizer=pos_norm,
            y_normalizer=y_norm,
            feats_normalizer=shifted_feats,
        )


def test_graph_cache_num_shards_prefers_loader_latency() -> None:
    assert graph_cache_num_shards("micro_puc", 70_000) == 48
    assert graph_cache_num_shards("poisson_unstructured", 10_000) == 2
    assert graph_cache_num_shards("poisson_structured", 6001) == 2
    assert graph_cache_num_shards("bracket_lug", 500) == 2
    assert graph_cache_num_shards("unknown_dataset", 3) == 3


def test_micro_puc_graph_cache_uses_multiple_shards(tmp_path) -> None:
    raw = _mock_raw(4)
    raw = GinotRawDataset(
        dataset_dir=str(tmp_path / "dataset"),
        query_points=raw.query_points,
        point_clouds=raw.point_clouds,
        targets=raw.targets,
        cells=raw.cells,
        input_params=None,
        target_fields=raw.target_fields,
        space_dim=2,
        normalize_pos=False,
        normalize_boundary_pos=False,
    )
    pos_norm = _identity_normalizer(2)
    y_norm = _identity_normalizer(1)
    cache_dir = ensure_graph_cache(
        raw=raw,
        dataset_name="micro_puc",
        split_name="train",
        indices=[0, 1, 2, 3],
        split_seed=0,
        dataset_split="default",
        pos_normalizer=pos_norm,
        boundary_pos_normalizer=pos_norm,
        y_normalizer=y_norm,
    )
    index = torch.load(Path(cache_dir) / "index.pt", weights_only=True)
    num_shards = int(index["shard_id"].max().item()) + 1
    assert num_shards == graph_cache_num_shards("micro_puc", 4)
    assert num_shards == 4
    for idx in range(4):
        assert graph_cache_shard_id(cache_dir, idx) == int(index["shard_id"][index["sample_ids"] == idx].item())
