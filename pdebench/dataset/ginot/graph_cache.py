from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import lmdb
import torch
from tqdm import tqdm

from pdebench.dataset.distributed import distributed_barrier, distributed_rank
from pdebench.dataset.ginot.bumper_beam import preload_bumper_beam_store
from pdebench.dataset.ginot.mesh import build_edge_index_from_cells, cells_are_sample_indexed, topology_key_for_row
from pdebench.dataset.ginot.sample import build_graph_sample_dict
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer
from pdebench.dataset.ginot.utils import as_float_array

GRAPH_CACHE_FORMAT = "lmdb_pyg_tensor_dict_sharded"
GRAPH_CACHE_SCHEMA_VERSION = 5
DEFAULT_GRAPH_CACHE_WORKERS = 32
DEFAULT_LMDB_COMMIT_INTERVAL = 1024
DEFAULT_LMDB_MAP_SIZE = 1 << 38
# Profiling: 128 open shards beats 64 on poisson_unstructured / micro_puc_fixed (48 shards).
DEFAULT_OPEN_SHARD_LRU = 128

# Tuned for dataloader latency (more shards on large splits).
_GRAPH_CACHE_NUM_SHARDS: dict[str, int] = {
    "micro_puc": 48,
    "micro_puc_fixed": 48,
    "bracket_lug": 2,
    "poisson_unstructured": 2,
    "poisson_structured": 2,
    "bumper_beam": 4,
    "deform_plate": 16,
}
_DEFAULT_GRAPH_CACHE_NUM_SHARDS = 4

_GRAPH_CACHE_WORKER_STATE: dict[str, Any] = {}
_SHARDED_LMDB_GRAPH_CACHE_READERS: dict[str, Any] = {}


def graph_cache_num_shards(dataset_name: str, num_samples: int) -> int:
    name = str(dataset_name).lower()
    target = _GRAPH_CACHE_NUM_SHARDS.get(name, _DEFAULT_GRAPH_CACHE_NUM_SHARDS)
    return max(1, min(int(target), int(num_samples)))


def _serialize_torch(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _deserialize_torch(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _lmdb_map_size() -> int:
    return int(os.environ.get("GINOT_LMDB_MAP_SIZE", str(DEFAULT_LMDB_MAP_SIZE)))


def _lmdb_commit_interval() -> int:
    return max(1, int(os.environ.get("GINOT_LMDB_COMMIT_INTERVAL", str(DEFAULT_LMDB_COMMIT_INTERVAL))))


def _open_shard_lru_size() -> int:
    return max(1, int(os.environ.get("GINOT_LMDB_OPEN_SHARD_LRU", str(DEFAULT_OPEN_SHARD_LRU))))


def _lmdb_readahead() -> bool:
    return os.environ.get("GINOT_LMDB_READAHEAD", "1").strip().lower() not in ("0", "false", "no")


def _sample_key(sample_id: int) -> bytes:
    return f"sample_{int(sample_id):08d}".encode("ascii")


def _shard_name(shard_id: int) -> str:
    return f"shard_{int(shard_id):06d}"


def _shard_dir(cache_dir: str | os.PathLike[str], shard_id: int) -> Path:
    return Path(cache_dir) / "shards" / _shard_name(shard_id)


def _partition_indices(indices: list[int], num_shards: int) -> list[list[int]]:
    if num_shards <= 1:
        return [list(indices)]
    ordered = sorted(int(idx) for idx in indices)
    chunk = max(1, math.ceil(len(ordered) / int(num_shards)))
    return [ordered[i : i + chunk] for i in range(0, len(ordered), chunk)]


def _set_graph_cache_thread_limits() -> None:
    from pdebench.dataset.thread_limits import set_compute_thread_limits

    set_compute_thread_limits()


def _load_graph_cache_index(cache_dir: str | os.PathLike[str]) -> dict[str, torch.Tensor]:
    index_path = Path(cache_dir) / "index.pt"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing graph cache index at {index_path}")
    return torch.load(index_path, map_location="cpu", weights_only=True)


class ShardedLmdbGraphCache:
    """Reader for graph_lmdb caches split into multiple LMDB shard directories."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        index = _load_graph_cache_index(self.cache_dir)
        sample_ids = index["sample_ids"].tolist()
        shard_ids = index["shard_id"].tolist()
        self._shard_by_sample = {int(sid): int(shard) for sid, shard in zip(sample_ids, shard_ids)}
        self._envs: OrderedDict[int, lmdb.Environment] = OrderedDict()
        self._max_open = _open_shard_lru_size()

    def _env_for_shard(self, shard_id: int) -> lmdb.Environment:
        key = int(shard_id)
        env = self._envs.get(key)
        if env is not None:
            self._envs.move_to_end(key)
            return env
        shard_path = _shard_dir(self.cache_dir, key)
        env = lmdb.open(
            str(shard_path),
            readonly=True,
            lock=False,
            readahead=_lmdb_readahead(),
            subdir=True,
            max_readers=4096,
        )
        self._envs[key] = env
        self._envs.move_to_end(key)
        while len(self._envs) > self._max_open:
            self._envs.popitem(last=False)
        return env

    def get(self, sample_id: int) -> dict[str, torch.Tensor]:
        sample_id = int(sample_id)
        shard_id = self._shard_by_sample.get(sample_id)
        if shard_id is None:
            raise KeyError(f"Sample {sample_id} is not present in sharded graph cache at {self.cache_dir}.")
        env = self._env_for_shard(shard_id)
        with env.begin(write=False) as txn:
            data = txn.get(_sample_key(sample_id))
        if data is None:
            raise KeyError(
                f"Sample {sample_id} missing from shard {shard_id} of graph cache at {self.cache_dir}."
            )
        return _deserialize_torch(data)


def graph_cache_root(raw: GinotRawDataset) -> str:
    return os.path.join(raw.dataset_dir, "static_cache", "graph_lmdb")


def graph_cache_dir(
    raw: GinotRawDataset,
    dataset_name: str,
    split_name: str,
    split_seed: int,
    dataset_split: str,
) -> str:
    return os.path.join(
        graph_cache_root(raw),
        f"{dataset_name}_{dataset_split}_seed{int(split_seed)}",
        split_name,
    )


def is_sharded_lmdb_graph_cache(cache_dir: str | os.PathLike[str]) -> bool:
    cache_path = Path(cache_dir)
    if not (cache_path / "index.pt").is_file():
        return False
    shards_dir = cache_path / "shards"
    if not shards_dir.is_dir():
        return False
    return any((shard / "data.mdb").is_file() for shard in shards_dir.iterdir() if shard.is_dir())


is_lmdb_graph_cache = is_sharded_lmdb_graph_cache


def get_sharded_lmdb_graph_cache(cache_dir: str | os.PathLike[str]) -> ShardedLmdbGraphCache:
    key = str(Path(cache_dir).resolve())
    reader = _SHARDED_LMDB_GRAPH_CACHE_READERS.get(key)
    if reader is None:
        reader = ShardedLmdbGraphCache(key)
        _SHARDED_LMDB_GRAPH_CACHE_READERS[key] = reader
    return reader


def graph_cache_shard_id(cache_dir: str | os.PathLike[str], sample_id: int) -> int:
    return int(get_sharded_lmdb_graph_cache(cache_dir)._shard_by_sample[int(sample_id)])


def load_graph_cache_sample(cache_dir: str | os.PathLike[str], idx: int) -> dict[str, torch.Tensor]:
    cache_dir = str(cache_dir)
    if not is_sharded_lmdb_graph_cache(cache_dir):
        raise FileNotFoundError(
            f"Expected sharded LMDB graph cache at {cache_dir} (missing index.pt and shards/*/data.mdb). "
            "Build it with ensure_graph_cache."
        )
    return get_sharded_lmdb_graph_cache(cache_dir).get(int(idx))


def _load_sample_ids(cache_dir: str | os.PathLike[str]) -> set[int]:
    index_path = Path(cache_dir) / "index.pt"
    if not index_path.is_file():
        return set()
    index = _load_graph_cache_index(cache_dir)
    return {int(sid) for sid in index["sample_ids"].tolist()}


def _normalizer_manifest(normalizer) -> dict[str, list]:
    return {
        "mean": normalizer.mean.detach().cpu().float().tolist(),
        "std": normalizer.std.detach().cpu().float().tolist(),
    }


def _normalizer_from_manifest(entry: dict[str, list]) -> StandardNormalizer:
    return StandardNormalizer(
        mean=torch.tensor(entry["mean"], dtype=torch.float32),
        std=torch.tensor(entry["std"], dtype=torch.float32),
    )


def load_graph_cache_normalizers(
    cache_dir: str | os.PathLike[str],
) -> tuple[StandardNormalizer, StandardNormalizer, StandardNormalizer, StandardNormalizer | None]:
    manifest_path = Path(cache_dir) / "manifest.pt"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Static graph cache at {cache_dir} is missing manifest.pt; rebuild the cache before loading."
        )
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Static graph cache manifest at {manifest_path} is invalid.")
    normalizers = manifest.get("normalizers")
    if not isinstance(normalizers, dict):
        raise RuntimeError(f"Static graph cache manifest at {manifest_path} is missing normalizers.")
    feats_entry = normalizers.get("feats")
    feats_normalizer = (
        _normalizer_from_manifest(feats_entry) if isinstance(feats_entry, dict) else None
    )
    return (
        _normalizer_from_manifest(normalizers["pos"]),
        _normalizer_from_manifest(normalizers["boundary_pos"]),
        _normalizer_from_manifest(normalizers["y"]),
        feats_normalizer,
    )


def _optional_normalizer_manifest(normalizer: StandardNormalizer | None) -> dict[str, list] | None:
    if normalizer is None:
        return None
    return _normalizer_manifest(normalizer)


def _graph_cache_manifest(
    *,
    dataset_name: str,
    split_name: str,
    indices: list[int],
    split_seed: int,
    dataset_split: str,
    num_shards: int,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    feats_normalizer: StandardNormalizer | None = None,
    normalize_targets: bool = True,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "format": GRAPH_CACHE_FORMAT,
        "dataset_name": str(dataset_name),
        "dataset_split": str(dataset_split),
        "split_name": str(split_name),
        "split_seed": int(split_seed),
        "num_samples": len(indices),
        "sample_ids": torch.tensor(sorted(int(idx) for idx in indices), dtype=torch.long),
        "num_shards": int(num_shards),
        "normalize_targets": bool(normalize_targets),
        "normalizers": {
            "pos": _normalizer_manifest(pos_normalizer),
            "boundary_pos": _normalizer_manifest(boundary_pos_normalizer),
            "y": _normalizer_manifest(y_normalizer),
            "feats": _optional_normalizer_manifest(feats_normalizer),
        },
    }
    if str(dataset_name) == "deform_plate":
        manifest["deform_plate_boundary_fields"] = True
    return manifest


def _manifest_without_sample_ids(manifest: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in manifest.items() if key not in {"num_samples", "sample_ids", "num_shards"}}


def _manifest_for_cache_validation(manifest: dict[str, object]) -> dict[str, object]:
    """Compare preprocessing that affects cached tensors.

    When normalize_targets=False, y is stored raw and y_normalizer is resolved at
    load time — omit y from manifest equality checks.
    """
    core = _manifest_without_sample_ids(manifest)
    if core.get("normalize_targets"):
        return core
    normalizers = core.get("normalizers")
    if not isinstance(normalizers, dict):
        return core
    core = dict(core)
    core["normalizers"] = {key: value for key, value in normalizers.items() if key != "y"}
    return core


def _validate_graph_cache_manifest(cache_dir: str | os.PathLike[str], expected: dict[str, object], required: set[int]) -> None:
    manifest_path = Path(cache_dir) / "manifest.pt"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Static graph cache at {cache_dir} is missing manifest.pt with preprocessing metadata. "
            "Delete and rebuild the cache so cached pos/y/edge_attr match the current normalizers."
        )
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Static graph cache manifest at {manifest_path} is invalid; delete and rebuild the cache.")

    actual_core = _manifest_for_cache_validation(manifest)
    expected_core = _manifest_for_cache_validation(expected)
    if actual_core != expected_core:
        raise RuntimeError(
            f"Static graph cache at {cache_dir} was built with different preprocessing metadata. "
            "Delete and rebuild it before training/evaluating. "
            f"Expected {expected_core!r}, found {actual_core!r}."
        )

    cached_ids = manifest.get("sample_ids")
    if not torch.is_tensor(cached_ids):
        raise RuntimeError(
            f"Static graph cache manifest at {manifest_path} is missing sample_ids; delete and rebuild the cache."
        )
    missing = sorted(required - {int(idx) for idx in cached_ids.tolist()})
    if missing:
        raise RuntimeError(
            f"Static graph cache manifest at {manifest_path} is missing {len(missing)} requested samples. "
            f"First missing sample id: {missing[0]}. Delete and rebuild the cache."
        )


def _sample_num_nodes(sample: dict[str, torch.Tensor]) -> int:
    return int(sample["pos"].shape[0])


def _sample_boundary_num_nodes(sample: dict[str, torch.Tensor]) -> int:
    boundary = sample.get("boundary_pos")
    if boundary is not None:
        return int(boundary.shape[0])
    return _sample_num_nodes(sample)


def ensure_graph_cache_length_index(cache_dir: str | os.PathLike[str]) -> None:
    cache_path = Path(cache_dir)
    index = _load_graph_cache_index(cache_path)
    sample_ids = index["sample_ids"]
    if (
        "node_lengths" in index
        and "boundary_lengths" in index
        and "shard_id" in index
        and int(index["node_lengths"].numel()) == int(sample_ids.numel())
    ):
        return
    raise RuntimeError(f"Sharded LMDB graph cache at {cache_path} is missing length metadata; rebuild it.")


def node_lengths_for_indices(cache_dir: str | os.PathLike[str], indices: list[int]) -> list[int]:
    ensure_graph_cache_length_index(cache_dir)
    index = _load_graph_cache_index(cache_dir)
    length_map = {
        int(sid): int(length)
        for sid, length in zip(index["sample_ids"].tolist(), index["node_lengths"].tolist())
    }
    return [length_map[int(idx)] for idx in indices]


def read_graph_cache_meta(cache_dir: str | os.PathLike[str]) -> dict[str, int | str]:
    cache_path = Path(cache_dir)
    ensure_graph_cache_length_index(cache_path)
    index = _load_graph_cache_index(cache_path)
    max_node_length = int(index["node_lengths"].max().item())
    max_boundary_length = int(index["boundary_lengths"].max().item())
    meta_path = cache_path / "meta.json"
    meta: dict[str, int | str] = {}
    if meta_path.is_file():
        meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    meta["max_node_length"] = max_node_length
    meta["max_boundary_length"] = max_boundary_length
    meta["num_shards"] = int(index["shard_id"].max().item()) + 1
    tmp_path = cache_path / f"meta.json.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_path, meta_path)
    return meta


def validate_graph_cache_complete(cache_dir: str | os.PathLike[str], indices: list[int]) -> None:
    missing = sorted({int(idx) for idx in indices} - _load_sample_ids(cache_dir))
    if missing:
        preview = missing[:8]
        suffix = "..." if len(missing) > len(preview) else ""
        raise FileNotFoundError(
            f"Graph cache at {cache_dir} is missing {len(missing)} samples: {preview}{suffix}. "
            "Build or rebuild the static cache before training."
        )


def open_graph_cache_split(cache_dir: str | os.PathLike[str], indices: list[int]):
    from pdebench.dataset.ginot.dataset import resolve_graph_cache_split

    validate_graph_cache_complete(cache_dir, indices)
    return resolve_graph_cache_split(str(cache_dir), indices)


def max_padding_lengths_for_indices(cache_dir: str | os.PathLike[str], indices: list[int]) -> tuple[int, int]:
    del indices
    meta = read_graph_cache_meta(cache_dir)
    return int(meta["max_node_length"]), int(meta["max_boundary_length"])


def _init_graph_cache_worker(
    raw: GinotRawDataset,
    dataset_name: str,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    cache_dir: str,
    shared_edge_index: torch.Tensor | None = None,
    feats_normalizer: StandardNormalizer | None = None,
) -> None:
    _set_graph_cache_thread_limits()
    # Parent usually preloads before fork; this is idempotent and covers spawn.
    preload_bumper_beam_store(raw)
    _GRAPH_CACHE_WORKER_STATE.clear()
    _GRAPH_CACHE_WORKER_STATE.update(
        raw=raw,
        dataset_name=dataset_name,
        pos_normalizer=pos_normalizer,
        boundary_pos_normalizer=boundary_pos_normalizer,
        y_normalizer=y_normalizer,
        feats_normalizer=feats_normalizer,
        cache_dir=cache_dir,
        shared_edge_index=shared_edge_index,
        edge_index_cache={},
    )


def _build_graph_sample_with_topology_cache(idx: int) -> dict[str, torch.Tensor]:
    raw = _GRAPH_CACHE_WORKER_STATE["raw"]
    dataset_name = _GRAPH_CACHE_WORKER_STATE["dataset_name"]
    pos_normalizer = _GRAPH_CACHE_WORKER_STATE["pos_normalizer"]
    boundary_pos_normalizer = _GRAPH_CACHE_WORKER_STATE["boundary_pos_normalizer"]
    y_normalizer = _GRAPH_CACHE_WORKER_STATE["y_normalizer"]
    feats_normalizer = _GRAPH_CACHE_WORKER_STATE.get("feats_normalizer")
    shared_edge_index = _GRAPH_CACHE_WORKER_STATE.get("shared_edge_index")
    edge_index_cache: dict[int, torch.Tensor] = _GRAPH_CACHE_WORKER_STATE["edge_index_cache"]

    topology_key = topology_key_for_row(raw, dataset_name, int(idx))
    if topology_key is not None and topology_key in edge_index_cache:
        shared_edge_index = edge_index_cache[topology_key]

    sample = build_graph_sample_dict(
        raw=raw,
        idx=int(idx),
        pos_normalizer=pos_normalizer,
        boundary_pos_normalizer=boundary_pos_normalizer,
        y_normalizer=y_normalizer,
        feats_normalizer=feats_normalizer,
        dataset_name=dataset_name,
        shared_edge_index=shared_edge_index,
        to_cpu=True,
    )
    if topology_key is not None and topology_key not in edge_index_cache:
        edge_index_cache[topology_key] = sample["edge_index"].cpu()
    return sample


def _build_graph_cache_sample_worker(idx: int) -> tuple[int, dict[str, torch.Tensor]]:
    return int(idx), _build_graph_sample_with_topology_cache(int(idx))


def _write_shard_lmdb(
    shard_dir: Path,
    indices: list[int],
    samples: dict[int, dict[str, torch.Tensor]],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(shard_dir),
        map_size=_lmdb_map_size(),
        subdir=True,
        map_async=True,
        sync=False,
        metasync=False,
        max_dbs=1,
    )
    commit_interval = _lmdb_commit_interval()
    txn = env.begin(write=True)
    written = 0
    try:
        for sid in indices:
            sample = samples[int(sid)]
            txn.put(_sample_key(int(sid)), _serialize_torch(sample))
            written += 1
            if written % commit_interval == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        raise
    finally:
        env.sync()
        env.close()


def _write_sharded_lmdb_graph_cache(
    *,
    raw: GinotRawDataset,
    dataset_name: str,
    indices: list[int],
    cache_dir: str,
    num_shards: int,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    shared_edge_index: torch.Tensor | None,
    feats_normalizer: StandardNormalizer | None = None,
) -> None:
    cache_path = Path(cache_dir)
    shards_root = cache_path / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    _set_graph_cache_thread_limits()

    shard_partitions = _partition_indices(indices, num_shards)
    num_shards = len(shard_partitions)
    worker_cap = int(os.environ.get("GINOT_GRAPH_CACHE_WORKERS", str(DEFAULT_GRAPH_CACHE_WORKERS)))
    chunksize = max(1, int(os.environ.get("GINOT_GRAPH_CACHE_CHUNKSIZE", "1")))

    all_ids = [int(idx) for part in shard_partitions for idx in part]
    print(
        f"Using {min(worker_cap, os.cpu_count() or 1, len(all_ids))} LMDB graph cache worker(s); "
        f"chunksize={chunksize}; num_shards={num_shards}."
    )

    num_workers = max(1, min(worker_cap, os.cpu_count() or 1, len(all_ids)))
    ctx = mp.get_context("fork")
    sample_ids: list[int] = []
    shard_ids: list[int] = []
    node_lengths: list[int] = []
    boundary_lengths: list[int] = []

    with ctx.Pool(
        processes=num_workers,
        initializer=_init_graph_cache_worker,
        initargs=(
            raw,
            dataset_name,
            pos_normalizer,
            boundary_pos_normalizer,
            y_normalizer,
            cache_dir,
            shared_edge_index,
            feats_normalizer,
        ),
    ) as pool:
        for shard_id, shard_indices in enumerate(shard_partitions):
            if not shard_indices:
                continue
            shard_samples: dict[int, dict[str, torch.Tensor]] = {}
            for sid, sample in tqdm(
                pool.imap_unordered(_build_graph_cache_sample_worker, shard_indices, chunksize=chunksize),
                total=len(shard_indices),
                desc=f"Graph LMDB shard {shard_id + 1}/{num_shards}",
                ncols=90,
            ):
                shard_samples[int(sid)] = sample
            _write_shard_lmdb(_shard_dir(cache_path, shard_id), shard_indices, shard_samples)
            for sid in shard_indices:
                sample = shard_samples[int(sid)]
                sample_ids.append(int(sid))
                shard_ids.append(int(shard_id))
                node_lengths.append(_sample_num_nodes(sample))
                boundary_lengths.append(_sample_boundary_num_nodes(sample))
            shard_samples.clear()

    order = sorted(range(len(sample_ids)), key=lambda i: sample_ids[i])
    index = {
        "sample_ids": torch.tensor([sample_ids[i] for i in order], dtype=torch.long),
        "shard_id": torch.tensor([shard_ids[i] for i in order], dtype=torch.long),
        "node_lengths": torch.tensor([node_lengths[i] for i in order], dtype=torch.long),
        "boundary_lengths": torch.tensor([boundary_lengths[i] for i in order], dtype=torch.long),
    }
    tmp_path = cache_path / f"index.pt.tmp.{os.getpid()}"
    torch.save(index, tmp_path)
    os.replace(tmp_path, cache_path / "index.pt")


def ensure_graph_cache(
    raw: GinotRawDataset,
    dataset_name: str,
    split_name: str,
    indices: list[int],
    split_seed: int,
    dataset_split: str,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    feats_normalizer: StandardNormalizer | None = None,
) -> str:
    cache_dir = graph_cache_dir(
        raw=raw,
        dataset_name=dataset_name,
        split_name=split_name,
        split_seed=split_seed,
        dataset_split=dataset_split,
    )
    required = {int(idx) for idx in indices}
    rank = distributed_rank()
    num_shards = graph_cache_num_shards(dataset_name, len(required))
    expected_manifest = _graph_cache_manifest(
        dataset_name=dataset_name,
        split_name=split_name,
        indices=sorted(required),
        split_seed=split_seed,
        dataset_split=dataset_split,
        num_shards=num_shards,
        pos_normalizer=pos_normalizer,
        boundary_pos_normalizer=boundary_pos_normalizer,
        y_normalizer=y_normalizer,
        feats_normalizer=feats_normalizer,
        normalize_targets=bool(raw.normalize_targets),
    )

    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        existing = _load_sample_ids(cache_dir)
        missing = sorted(required - existing)
        if existing:
            _validate_graph_cache_manifest(cache_dir, expected_manifest, required)

        if missing:
            print()
            print(f"Building static GINOT graph cache for {dataset_name} split={split_name}.")
            print(f"Cache path: {cache_dir}")
            print(
                "This one-time preprocessing step stores PyG-style graph tensors "
                f"(pos, y, boundary_pos, edge_index, edge_attr"
                f"{', feats' if feats_normalizer is not None else ''}) "
                f"in {num_shards} sharded LMDB files."
            )
            # Decode bumper VTPs once in the parent so fork workers inherit the
            # store cache (edge models only pay this when the LMDB is missing).
            preload_bumper_beam_store(raw)
            shared_edge_index = None
            if raw.cells is not None and not cells_are_sample_indexed(raw):
                num_nodes = int(as_float_array(raw.query_points[indices[0]], dims=raw.space_dim).shape[0])
                print("Precomputing shared mesh edge_index once for this split.")
                shared_edge_index = build_edge_index_from_cells(raw.cells, num_nodes=num_nodes).cpu()
                print(f"Shared edge_index edges={int(shared_edge_index.shape[1])}.")

            if existing:
                raise RuntimeError(
                    f"Partial sharded LMDB cache at {cache_dir} has {len(existing)} samples but is missing {len(missing)}. "
                    "Delete it and rebuild to avoid mixed cache contents."
                )
            _write_sharded_lmdb_graph_cache(
                raw=raw,
                dataset_name=dataset_name,
                indices=sorted(required),
                cache_dir=cache_dir,
                num_shards=num_shards,
                pos_normalizer=pos_normalizer,
                boundary_pos_normalizer=boundary_pos_normalizer,
                y_normalizer=y_normalizer,
                shared_edge_index=shared_edge_index,
                feats_normalizer=feats_normalizer,
            )
            print(f"Finished static GINOT graph cache for {dataset_name} split={split_name}.")

        if missing or not existing:
            manifest_path = os.path.join(cache_dir, "manifest.pt")
            torch.save(expected_manifest, manifest_path)

    distributed_barrier()
    missing_after = sorted(required - _load_sample_ids(cache_dir))
    if missing_after:
        raise RuntimeError(
            f"Static graph cache is incomplete at {cache_dir}; missing {len(missing_after)} samples. "
            f"First missing sample id: {missing_after[0]}."
        )
    if rank == 0:
        ensure_graph_cache_length_index(cache_dir)
        read_graph_cache_meta(cache_dir)
    distributed_barrier()
    return cache_dir
