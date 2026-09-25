from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset

from pdebench.dataset.ginot.features import attach_sample_feats
from pdebench.dataset.ginot.graph_cache import (
    graph_cache_shard_id,
    is_lmdb_graph_cache,
    load_graph_cache_sample,
    node_lengths_for_indices,
    read_graph_cache_meta,
)
from pdebench.dataset.ginot.sample import build_sample_edge_tensors, encode_raw_sample
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer
from pdebench.dataset.ginot.utils import as_float_array
from pdebench.dataset.laplacian import LaplacianService, LmdbLaplacianBackend
from pdebench.dataset.laplacian.spec import laplacian_spec_dim
from pdebench.dataset.sample import SampleKind
from pdebench.dataset.sample_bridge import ginot_dict_to_sample


def resolve_eval_batch_sampler(dataset: Dataset, batch_size: int, *, seed: int = 0):
    """Return a shard/length-aware batch sampler for eval loaders when supported."""
    make_batch_sampler = getattr(dataset, "make_batch_sampler", None)
    if make_batch_sampler is None:
        return None
    return make_batch_sampler(batch_size=int(batch_size), drop_last=False, seed=int(seed))


class GinotDataset(Dataset):
    """Live GINOT dataset view over raw arrays.

    Static graph-cache reads are handled by GraphCacheDataset. Keeping this class
    raw-only avoids two subtly different cache paths.
    """

    def __init__(
        self,
        dataset_name: str,
        raw: GinotRawDataset,
        indices: list[int],
        pos_normalizer: StandardNormalizer,
        boundary_pos_normalizer: StandardNormalizer,
        y_normalizer: StandardNormalizer,
        feats_normalizer: StandardNormalizer | None = None,
        include_edges: bool = False,
        bucketed_batches: bool = False,
    ):
        self.dataset_name = dataset_name
        self.raw = raw
        self.indices = indices
        self.pos_normalizer = pos_normalizer
        self.boundary_pos_normalizer = boundary_pos_normalizer
        self.y_normalizer = y_normalizer
        self.feats_normalizer = feats_normalizer
        self.include_edges = bool(include_edges)
        self.bucketed_batches = bool(bucketed_batches)
        self._node_lengths = None
        if self.bucketed_batches:
            self._node_lengths = [
                int(as_float_array(raw.query_points[idx], dims=raw.space_dim).shape[0]) for idx in indices
            ]

    def __len__(self):
        return len(self.indices)

    def make_batch_sampler(self, batch_size: int, drop_last: bool, seed: int = 0):
        if not self.bucketed_batches or self._node_lengths is None:
            return None
        return LengthBucketBatchSampler(
            lengths=self._node_lengths,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
        )

    def __getitem__(self, item):
        idx = self.indices[item]
        pos, boundary_pos, y = encode_raw_sample(
            self.raw,
            idx,
            self.pos_normalizer,
            self.boundary_pos_normalizer,
            self.y_normalizer,
        )

        if self.include_edges:
            edge_index, edge_attr, cells_np = build_sample_edge_tensors(
                self.raw,
                idx,
                pos,
                dataset_name=self.dataset_name,
            )
            cells = torch.from_numpy(cells_np).long() if cells_np is not None else None
            edge_cache_key = self._format_edge_cache_key(int(idx), edge_index)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, pos.shape[-1] + 1), dtype=pos.dtype)
            cells = None
            edge_cache_key = None

        sample = {
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "boundary_pos": boundary_pos,
            "y": y,
            "sample_id": torch.tensor(int(idx), dtype=torch.long),
        }
        if edge_cache_key is not None:
            sample["edge_cache_key"] = edge_cache_key
        if cells is not None:
            sample["cells"] = cells
        return attach_sample_feats(
            sample,
            self.raw,
            int(idx),
            self.feats_normalizer,
            y_normalizer=self.y_normalizer,
        )

    @staticmethod
    def _format_edge_cache_key(sample_id: int, edge_index: torch.Tensor) -> str:
        return f"{int(sample_id)}:{int(edge_index.shape[1])}"


class LengthBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        drop_last: bool,
        seed: int = 0,
        bucket_size_multiplier: int = 16,
    ):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))
        sorted_indices = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        self.buckets = [sorted_indices[i:i + bucket_size] for i in range(0, len(sorted_indices), bucket_size)]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        bucket_order = rng.permutation(len(self.buckets)).tolist()
        for bucket_idx in bucket_order:
            bucket = list(self.buckets[bucket_idx])
            rng.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch
        self.epoch += 1

    def __len__(self):
        total = len(self.lengths)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


class ShardLengthBucketBatchSampler(BatchSampler):
    """Batch sampler that keeps LMDB graph-cache batches shard-local.

    Random batches over many LMDB shards churn both open environments and the OS
    page cache. This sampler still shuffles each epoch, but builds length buckets
    independently per shard so each yielded batch usually reads one shard.
    """

    def __init__(
        self,
        lengths: list[int],
        shard_ids: list[int],
        batch_size: int,
        drop_last: bool,
        seed: int = 0,
        bucket_size_multiplier: int = 16,
    ):
        if len(lengths) != len(shard_ids):
            raise ValueError(f"lengths and shard_ids must have equal length, got {len(lengths)} and {len(shard_ids)}.")
        self.lengths = [int(length) for length in lengths]
        self.shard_ids = [int(shard_id) for shard_id in shard_ids]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))

        by_shard: dict[int, list[int]] = {}
        for position, shard_id in enumerate(self.shard_ids):
            by_shard.setdefault(shard_id, []).append(position)
        self.buckets: list[list[int]] = []
        for positions in by_shard.values():
            ordered = sorted(positions, key=self.lengths.__getitem__)
            self.buckets.extend(ordered[i:i + bucket_size] for i in range(0, len(ordered), bucket_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        bucket_order = rng.permutation(len(self.buckets)).tolist()
        for bucket_idx in bucket_order:
            bucket = list(self.buckets[bucket_idx])
            rng.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch
        self.epoch += 1

    def __len__(self):
        if self.drop_last:
            return sum(len(bucket) // self.batch_size for bucket in self.buckets)
        return sum((len(bucket) + self.batch_size - 1) // self.batch_size for bucket in self.buckets)


@dataclass(frozen=True)
class GraphCacheSplit:
    """Resolved train/test view over a prebuilt sharded graph LMDB cache."""

    cache_dir: str
    indices: list[int]
    node_lengths: list[int]
    shard_ids: list[int]
    max_node_length: int
    max_boundary_length: int


def resolve_graph_cache_split(cache_dir: str, indices: list[int]) -> GraphCacheSplit:
    if not is_lmdb_graph_cache(cache_dir):
        raise FileNotFoundError(
            f"Expected sharded LMDB graph cache at {cache_dir} (missing index.pt and shards/*/data.mdb). "
            "Build it first with ensure_graph_cache."
        )
    meta = read_graph_cache_meta(cache_dir)
    node_lengths = node_lengths_for_indices(cache_dir, indices)
    return GraphCacheSplit(
        cache_dir=str(cache_dir),
        indices=[int(idx) for idx in indices],
        node_lengths=node_lengths,
        shard_ids=[graph_cache_shard_id(cache_dir, int(idx)) for idx in indices],
        max_node_length=int(meta["max_node_length"]),
        max_boundary_length=int(meta["max_boundary_length"]),
    )


class GraphCacheDataset(Dataset):
    """Dataset view over a prebuilt sharded graph LMDB cache (optional Laplacian LMDB)."""

    def __init__(
        self,
        split: GraphCacheSplit,
        *,
        dataset_name: str,
        raw: GinotRawDataset,
        feats_normalizer: StandardNormalizer | None = None,
        y_normalizer: StandardNormalizer | None = None,
        laplacian_eig_dim: int = 0,
        laplacian_spec: str = "graph",
        bucketed_batches: bool = False,
    ):
        self.split = split
        self.dataset_name = dataset_name
        self.raw = raw
        self.feats_normalizer = feats_normalizer
        self.y_normalizer = y_normalizer
        self.indices = split.indices
        self.graph_cache_dir = split.cache_dir
        self.laplacian_spec = str(laplacian_spec)
        self.laplacian_default_dim = int(laplacian_eig_dim)
        self.laplacian_eig_dim = (
            laplacian_spec_dim(self.laplacian_spec, self.laplacian_default_dim)
            if self.laplacian_default_dim > 0
            else 0
        )
        self._laplacian_split_group = Path(self.graph_cache_dir).name
        self._laplacian_service = (
            LaplacianService(
                LmdbLaplacianBackend(
                    {
                        (self.dataset_name, self._laplacian_split_group): Path(self.graph_cache_dir),
                    }
                )
            )
            if self.laplacian_eig_dim > 0
            else None
        )
        self.bucketed_batches = bool(bucketed_batches)
        self._node_lengths = list(split.node_lengths) if bucketed_batches else None
        self._shard_ids = list(split.shard_ids) if bucketed_batches else None

    def __len__(self) -> int:
        return len(self.indices)

    def make_batch_sampler(self, batch_size: int, drop_last: bool, seed: int = 0):
        if not self.bucketed_batches or self._node_lengths is None:
            return None
        if self._shard_ids is not None:
            return ShardLengthBucketBatchSampler(
                lengths=self._node_lengths,
                shard_ids=self._shard_ids,
                batch_size=batch_size,
                drop_last=drop_last,
                seed=seed,
            )
        return LengthBucketBatchSampler(
            lengths=self._node_lengths,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
        )

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        sample = load_graph_cache_sample(self.graph_cache_dir, idx)
        sample["edge_cache_key"] = GinotDataset._format_edge_cache_key(idx, sample["edge_index"])
        sample["graph_cache_dir"] = self.graph_cache_dir
        if self._laplacian_service is not None:
            bridge_sample = ginot_dict_to_sample(sample, kind=SampleKind.STATIC)
            self._laplacian_service.attach(
                bridge_sample,
                canonical_dataset=self.dataset_name,
                split_group=self._laplacian_split_group,
                spec=self.laplacian_spec,
                K=self.laplacian_default_dim,
            )
            sample["laplacian_eig"] = bridge_sample.laplacian_eig
            if bridge_sample.laplacian_eigvals is not None:
                sample["laplacian_eigvals"] = bridge_sample.laplacian_eigvals
        return attach_sample_feats(
            sample,
            self.raw,
            idx,
            self.feats_normalizer,
            y_normalizer=self.y_normalizer,
        )
