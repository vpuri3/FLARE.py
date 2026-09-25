"""Sharded LMDB Laplacian cache I/O."""

from __future__ import annotations

import fcntl
import io
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import lmdb
import torch

from pdebench.dataset.laplacian.spec import (
    _cache_operator_name,
    _cache_spec_for_part,
    laplacian_spec_dim,
    laplacian_spec_slug,
    parse_laplacian_spec,
)

DEFAULT_LMDB_MAP_SIZE = 1 << 38

def _serialize_torch(obj) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _deserialize_torch(data: bytes):
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _lmdb_map_size() -> int:
    return int(os.environ.get("GINOT_LMDB_MAP_SIZE", str(DEFAULT_LMDB_MAP_SIZE)))


def _lmdb_commit_interval() -> int:
    return max(1, int(os.environ.get("GINOT_LMDB_COMMIT_INTERVAL", "1024")))

def laplacian_eigen_payload(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    eigenvectors_u: torch.Tensor | None = None,
    eigenvectors_v: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    payload = {
        "eigenvalues": eigenvalues.to(torch.float32).cpu(),
        "eigenvectors": eigenvectors.to(torch.float16).cpu(),
    }
    if eigenvectors_u is not None:
        payload["eigenvectors_u"] = eigenvectors_u.to(torch.float16).cpu()
    if eigenvectors_v is not None:
        payload["eigenvectors_v"] = eigenvectors_v.to(torch.float16).cpu()
    return payload


def laplacian_eigen_payload_bytes(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    eigenvectors_u: torch.Tensor | None = None,
    eigenvectors_v: torch.Tensor | None = None,
) -> bytes:
    return _serialize_torch(
        laplacian_eigen_payload(
            eigenvalues,
            eigenvectors,
            eigenvectors_u=eigenvectors_u,
            eigenvectors_v=eigenvectors_v,
        )
    )


def _sample_lmdb_key(sample_id: int) -> bytes:
    return f"sample_{int(sample_id):08d}".encode("ascii")


_LMDB_SPLIT_LAPLACIAN_READERS: dict[str, "ShardedLmdbSplitLaplacianCache"] = {}
DEFAULT_OPEN_SHARD_LRU = 64


def _evict_split_laplacian_reader(split_root_key: str) -> None:
    reader = _LMDB_SPLIT_LAPLACIAN_READERS.pop(split_root_key, None)
    if reader is None:
        return
    for env in list(reader._envs.values()):
        env.close()
    reader._envs.clear()


def _open_shard_lru_size() -> int:
    return max(1, int(os.environ.get("GINOT_LMDB_OPEN_SHARD_LRU", str(DEFAULT_OPEN_SHARD_LRU))))


def _split_laplacian_shard_name(shard_id: int) -> str:
    return f"shard_{int(shard_id):06d}"


def split_laplacian_lmdb_dir(
    graph_cache_dir: str | os.PathLike[str],
    num_eigenvectors: int = 64,
    laplacian_spec: str | None = "graph",
) -> Path:
    graph_dir = Path(graph_cache_dir)
    static_cache = graph_dir.parent.parent.parent
    split_group = graph_dir.parent.name
    split_name = graph_dir.name
    spec_slug = laplacian_spec_slug(laplacian_spec, int(num_eigenvectors))
    total_dim = laplacian_spec_dim(laplacian_spec, int(num_eigenvectors))
    return static_cache / "laplacian_lmdb" / spec_slug / f"K{int(total_dim)}" / split_group / split_name


def split_laplacian_lmdb_shard_dir(
    graph_cache_dir: str | os.PathLike[str],
    shard_id: int,
    num_eigenvectors: int = 64,
    laplacian_spec: str | None = "graph",
) -> Path:
    return (
        split_laplacian_lmdb_dir(graph_cache_dir, num_eigenvectors, laplacian_spec)
        / "shards"
        / _split_laplacian_shard_name(shard_id)
    )


def is_sharded_split_laplacian_cache(
    graph_cache_dir: str | os.PathLike[str],
    num_eigenvectors: int,
    laplacian_spec: str | None,
) -> bool:
    shards_dir = split_laplacian_lmdb_dir(graph_cache_dir, num_eigenvectors, laplacian_spec) / "shards"
    if not shards_dir.is_dir():
        return False
    return any((shard / "data.mdb").is_file() for shard in shards_dir.iterdir() if shard.is_dir())


class ShardedLmdbSplitLaplacianCache:
    """Per-sample Laplacian LMDB reader sharded to match the graph cache layout."""

    def __init__(self, split_root: str | Path, graph_cache_dir: str | Path):
        from pdebench.dataset.ginot.graph_cache import get_sharded_lmdb_graph_cache

        self.split_root = Path(split_root)
        self.graph_cache_dir = Path(graph_cache_dir)
        self._shard_by_sample = get_sharded_lmdb_graph_cache(self.graph_cache_dir)._shard_by_sample
        self._envs: OrderedDict[int, lmdb.Environment] = OrderedDict()
        self._max_open = _open_shard_lru_size()

    def _shard_path(self, shard_id: int) -> Path:
        return self.split_root / "shards" / _split_laplacian_shard_name(int(shard_id))

    def _shard_exists(self, shard_id: int) -> bool:
        return (self._shard_path(shard_id) / "data.mdb").is_file()

    def _env_for_shard(self, shard_id: int) -> lmdb.Environment:
        key = int(shard_id)
        env = self._envs.get(key)
        if env is not None:
            self._envs.move_to_end(key)
            return env
        shard_path = self._shard_path(key)
        if not (shard_path / "data.mdb").is_file():
            raise FileNotFoundError(f"Laplacian LMDB shard missing at {shard_path}")
        env = lmdb.open(
            str(shard_path),
            readonly=True,
            lock=False,
            readahead=False,
            subdir=True,
            max_readers=4096,
        )
        self._envs[key] = env
        self._envs.move_to_end(key)
        while len(self._envs) > self._max_open:
            self._envs.popitem(last=False)
        return env

    def has(self, sample_id: int) -> bool:
        sample_id = int(sample_id)
        shard_id = self._shard_by_sample.get(sample_id)
        if shard_id is None or not self._shard_exists(shard_id):
            return False
        env = self._env_for_shard(shard_id)
        with env.begin(write=False) as txn:
            return txn.get(_sample_lmdb_key(sample_id)) is not None

    def get_unpacked(
        self,
        sample_id: int,
        feature_dim: int,
        *,
        operator_name: str | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor] | None:
        sample_id = int(sample_id)
        shard_id = self._shard_by_sample.get(sample_id)
        if shard_id is None or not self._shard_exists(shard_id):
            return None
        env = self._env_for_shard(shard_id)
        with env.begin(write=False) as txn:
            data = txn.get(_sample_lmdb_key(sample_id))
        if data is None:
            return None
        return _unpack_laplacian_cache(_deserialize_torch(data), int(feature_dim), operator_name=operator_name)


def get_split_laplacian_lmdb_reader(
    graph_cache_dir: str | os.PathLike[str],
    num_eigenvectors: int,
    laplacian_spec: str | None,
) -> ShardedLmdbSplitLaplacianCache | None:
    if not is_sharded_split_laplacian_cache(graph_cache_dir, num_eigenvectors, laplacian_spec):
        return None
    split_root = split_laplacian_lmdb_dir(graph_cache_dir, num_eigenvectors, laplacian_spec)
    key = str(split_root.resolve())
    reader = _LMDB_SPLIT_LAPLACIAN_READERS.get(key)
    if reader is None:
        reader = ShardedLmdbSplitLaplacianCache(split_root, graph_cache_dir)
        _LMDB_SPLIT_LAPLACIAN_READERS[key] = reader
    return reader


def list_split_laplacian_cached_sample_ids(
    graph_cache_dir: str | os.PathLike[str],
    num_eigenvectors: int,
    laplacian_spec: str | None,
) -> set[int]:
    """Scan all LMDB shards once and return cached sample ids for a spec."""
    shards_dir = split_laplacian_lmdb_dir(graph_cache_dir, num_eigenvectors, laplacian_spec) / "shards"
    if not shards_dir.is_dir():
        return set()
    prefix = b"sample_"
    cached: set[int] = set()
    for shard_dir in sorted(shards_dir.iterdir()):
        if not shard_dir.is_dir() or not (shard_dir / "data.mdb").is_file():
            continue
        env = lmdb.open(str(shard_dir), readonly=True, lock=False, readahead=False, subdir=True)
        try:
            with env.begin(write=False) as txn:
                for key, _ in txn.cursor():
                    if key.startswith(prefix):
                        cached.add(int(key[len(prefix) :].decode("ascii")))
        finally:
            env.close()
    return cached


def split_laplacian_lmdb_has(
    graph_cache_dir: str | os.PathLike[str],
    sample_id: int,
    num_eigenvectors: int,
    laplacian_spec: str | None,
) -> bool:
    reader = get_split_laplacian_lmdb_reader(graph_cache_dir, num_eigenvectors, laplacian_spec)
    if reader is None:
        return False
    return reader.has(int(sample_id))


@dataclass
class _ResolvedLaplacianPart:
    name: str
    requested_dim: int
    cache_dim: int
    cache_spec: str
    reader: ShardedLmdbSplitLaplacianCache


class ResolvedLaplacianCachePlan:
    """Pre-resolved Laplacian LMDB readers for a dataset split.

    The training dataloader calls this once per sample, so directory discovery and
    reader selection must happen at dataset construction time instead of inside
    every ``__getitem__``.
    """

    def __init__(
        self,
        graph_cache_dir: str | os.PathLike[str],
        num_eigenvectors: int,
        laplacian_spec: str | None,
        *,
        feature_dim: int | None = None,
    ):
        self.graph_cache_dir = graph_cache_dir
        self.num_eigenvectors = int(num_eigenvectors)
        self.laplacian_spec = laplacian_spec
        self.feature_dim = (
            laplacian_spec_dim(laplacian_spec, self.num_eigenvectors)
            if feature_dim is None
            else int(feature_dim)
        )
        self.parts = [
            self._resolve_part(name, count)
            for name, count in parse_laplacian_spec(laplacian_spec, self.num_eigenvectors)
        ]
        if not self.parts:
            raise ValueError(f"No Laplacian operators in spec {laplacian_spec!r}.")

    def _resolve_part(self, name: str, requested_dim: int) -> _ResolvedLaplacianPart:
        cache_name = _cache_operator_name(name)
        graph_dir = Path(self.graph_cache_dir)
        static_cache = graph_dir.parent.parent.parent
        root = static_cache / "laplacian_lmdb"
        candidates: list[tuple[int, str, ShardedLmdbSplitLaplacianCache]] = []
        if root.is_dir():
            for spec_dir in root.iterdir():
                if not spec_dir.is_dir():
                    continue
                spec_slug = spec_dir.name
                if not spec_slug.startswith(cache_name):
                    continue
                suffix = spec_slug[len(cache_name) :]
                if not suffix.isdigit():
                    continue
                candidate_dim = int(suffix)
                if candidate_dim < int(requested_dim):
                    continue
                candidate_spec = _cache_spec_for_part(name, candidate_dim)
                reader = get_split_laplacian_lmdb_reader(self.graph_cache_dir, candidate_dim, candidate_spec)
                if reader is not None:
                    candidates.append((candidate_dim, candidate_spec, reader))
        if not candidates:
            candidate_spec = _cache_spec_for_part(name, int(requested_dim))
            reader = get_split_laplacian_lmdb_reader(self.graph_cache_dir, int(requested_dim), candidate_spec)
            if reader is None:
                raise FileNotFoundError(
                    f"Missing per-sample Laplacian LMDB cache for spec={candidate_spec!r} at "
                    f"{split_laplacian_lmdb_dir(self.graph_cache_dir, int(requested_dim), candidate_spec)}. "
                    "Run python -m pdebench.dataset.laplacian.precompute first."
                )
            candidates.append((int(requested_dim), candidate_spec, reader))
        cache_dim, cache_spec, reader = min(candidates, key=lambda item: item[0])
        return _ResolvedLaplacianPart(
            name=name,
            requested_dim=int(requested_dim),
            cache_dim=int(cache_dim),
            cache_spec=str(cache_spec),
            reader=reader,
        )

    def load(self, sample_id: int) -> tuple[torch.Tensor | None, torch.Tensor]:
        values = []
        vectors = []
        for part in self.parts:
            unpacked = part.reader.get_unpacked(
                int(sample_id),
                int(part.requested_dim),
                operator_name=part.name,
            )
            if unpacked is None:
                raise FileNotFoundError(
                    f"Missing Laplacian LMDB cache for sample_id={int(sample_id)} "
                    f"spec={part.cache_spec!r} at "
                    f"{split_laplacian_lmdb_dir(self.graph_cache_dir, part.cache_dim, part.cache_spec)}."
                )
            part_values, part_vectors = unpacked
            if part_values is not None:
                values.append(part_values)
            vectors.append(part_vectors)
        if len(self.parts) == 1:
            cat_values = values[0][:self.feature_dim] if values else None
            return cat_values, vectors[0][:, :self.feature_dim]
        cat_values = torch.cat(values, dim=-1)[:self.feature_dim] if values else None
        return cat_values, torch.cat(vectors, dim=-1)[:, :self.feature_dim]


def prepare_laplacian_cache_plan(
    graph_cache_dir: str | os.PathLike[str],
    num_eigenvectors: int,
    laplacian_spec: str | None,
    *,
    feature_dim: int | None = None,
) -> ResolvedLaplacianCachePlan:
    return ResolvedLaplacianCachePlan(
        graph_cache_dir,
        int(num_eigenvectors),
        laplacian_spec,
        feature_dim=feature_dim,
    )


class _ShardLaplacianWriteState:
    def __init__(self, shard_dir: Path, *, exclusive: bool = False):
        self.shard_dir = shard_dir
        self._exclusive = bool(exclusive)
        self._env: lmdb.Environment | None = None
        self._txn: lmdb.Transaction | None = None
        self._lock_file = None
        self._pending = 0
        self._commit_interval = _lmdb_commit_interval()

    def _ensure_open(self) -> None:
        if self._env is not None:
            return
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        if not self._exclusive:
            self._lock_file = open(self.shard_dir / ".write.lock", "w", encoding="ascii")
        self._env = lmdb.open(
            str(self.shard_dir),
            map_size=_lmdb_map_size(),
            subdir=True,
            map_async=True,
            sync=False,
            metasync=False,
            max_dbs=1,
        )
        self._txn = self._env.begin(write=True)

    def put(self, sample_id: int, payload: bytes) -> None:
        self._ensure_open()
        assert self._txn is not None
        if self._exclusive:
            self._txn.put(_sample_lmdb_key(int(sample_id)), payload)
            self._pending += 1
            if self._pending >= self._commit_interval:
                self._commit()
            return
        assert self._lock_file is not None
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            self._txn.put(_sample_lmdb_key(int(sample_id)), payload)
            self._pending += 1
            if self._pending >= self._commit_interval:
                self._commit()
        finally:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)

    def put_many(self, items: list[tuple[int, bytes]]) -> int:
        for sample_id, payload in items:
            self.put(int(sample_id), payload)
        return len(items)

    def _commit(self) -> None:
        if self._txn is None or self._env is None:
            return
        self._txn.commit()
        self._txn = self._env.begin(write=True)
        self._pending = 0

    def close(self) -> None:
        if self._txn is not None:
            self._txn.commit()
            self._txn = None
        if self._env is not None:
            self._env.sync()
            self._env.close()
            self._env = None
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None


class ShardAffineWriterSession:
    """Persistent LMDB writer(s) for precompute; one open env per assigned (spec, shard) lane."""

    def __init__(
        self,
        graph_cache_dir: str | os.PathLike[str],
        num_eigenvectors: int,
        lanes: list[tuple[str, int]],
    ):
        self.graph_cache_dir = graph_cache_dir
        self.num_eigenvectors = int(num_eigenvectors)
        self._states: dict[tuple[str, int], _ShardLaplacianWriteState] = {}
        for laplacian_spec, shard_id in lanes:
            key = (str(laplacian_spec), int(shard_id))
            if key in self._states:
                continue
            shard_dir = split_laplacian_lmdb_shard_dir(
                self.graph_cache_dir,
                int(shard_id),
                self.num_eigenvectors,
                laplacian_spec,
            )
            self._states[key] = _ShardLaplacianWriteState(shard_dir, exclusive=True)

    def write_batch(
        self,
        laplacian_spec: str | None,
        shard_id: int,
        items: list[tuple[int, bytes]],
    ) -> int:
        if not items:
            return 0
        state = self._states[(str(laplacian_spec), int(shard_id))]
        return state.put_many(items)

    def close(self) -> None:
        for state in self._states.values():
            state.close()
        self._states.clear()


def write_split_laplacian_shard_batch(
    graph_cache_dir: str | os.PathLike[str],
    shard_id: int,
    num_eigenvectors: int,
    laplacian_spec: str | None,
    items: list[tuple[int, bytes]],
) -> int:
    """Write many samples to one LMDB shard in a single open/sync cycle."""
    if not items:
        return 0
    shard_dir = split_laplacian_lmdb_shard_dir(graph_cache_dir, int(shard_id), num_eigenvectors, laplacian_spec)
    state = _ShardLaplacianWriteState(shard_dir, exclusive=True)
    try:
        return state.put_many(items)
    finally:
        state.close()


class SplitLaplacianWriteSession:
    """Reusable per-process LMDB writer for spectral precompute (avoids open/sync per sample)."""

    def __init__(self, graph_cache_dir: str | os.PathLike[str], num_eigenvectors: int):
        self.graph_cache_dir = graph_cache_dir
        self.num_eigenvectors = int(num_eigenvectors)
        self._shards: dict[str, _ShardLaplacianWriteState] = {}

    def save(
        self,
        sample_id: int,
        laplacian_spec: str | None,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        *,
        eigenvectors_u: torch.Tensor | None = None,
        eigenvectors_v: torch.Tensor | None = None,
    ) -> None:
        from pdebench.dataset.ginot.graph_cache import graph_cache_shard_id

        shard_id = graph_cache_shard_id(self.graph_cache_dir, int(sample_id))
        shard_dir = split_laplacian_lmdb_shard_dir(
            self.graph_cache_dir,
            shard_id,
            self.num_eigenvectors,
            laplacian_spec,
        )
        key = str(shard_dir.resolve())
        state = self._shards.get(key)
        if state is None:
            state = _ShardLaplacianWriteState(shard_dir)
            self._shards[key] = state
        state.put(
            int(sample_id),
            laplacian_eigen_payload_bytes(
                eigenvalues,
                eigenvectors,
                eigenvectors_u=eigenvectors_u,
                eigenvectors_v=eigenvectors_v,
            ),
        )

    def close(self) -> None:
        for state in self._shards.values():
            state.close()
        self._shards.clear()


def save_split_laplacian_lmdb(
    graph_cache_dir: str | os.PathLike[str],
    sample_id: int,
    num_eigenvectors: int,
    laplacian_spec: str | None,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    *,
    eigenvectors_u: torch.Tensor | None = None,
    eigenvectors_v: torch.Tensor | None = None,
) -> None:
    session = SplitLaplacianWriteSession(graph_cache_dir, num_eigenvectors)
    try:
        session.save(
            sample_id,
            laplacian_spec,
            eigenvalues,
            eigenvectors,
            eigenvectors_u=eigenvectors_u,
            eigenvectors_v=eigenvectors_v,
        )
    finally:
        session.close()


def resolve_split_laplacian_lmdb(
    graph_cache_dir: str | os.PathLike[str],
    sample_id: int,
    num_eigenvectors: int,
    laplacian_spec: str | None,
    *,
    feature_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor] | None:
    parts = parse_laplacian_spec(laplacian_spec, int(num_eigenvectors))
    if len(parts) != 1:
        return None
    name, requested_dim = parts[0]
    cache_name = _cache_operator_name(name)
    graph_dir = Path(graph_cache_dir)
    static_cache = graph_dir.parent.parent.parent
    root = static_cache / "laplacian_lmdb"
    candidates: list[tuple[int, ShardedLmdbSplitLaplacianCache]] = []
    if root.is_dir():
        for spec_dir in root.iterdir():
            if not spec_dir.is_dir():
                continue
            spec_slug = spec_dir.name
            if not spec_slug.startswith(cache_name):
                continue
            suffix = spec_slug[len(cache_name) :]
            if not suffix.isdigit():
                continue
            candidate_dim = int(suffix)
            if candidate_dim < requested_dim:
                continue
            candidate_spec = _cache_spec_for_part(name, candidate_dim)
            if not is_sharded_split_laplacian_cache(graph_cache_dir, candidate_dim, candidate_spec):
                continue
            reader = get_split_laplacian_lmdb_reader(graph_cache_dir, candidate_dim, candidate_spec)
            if reader is not None and reader.has(int(sample_id)):
                candidates.append((candidate_dim, reader))
    if not candidates:
        reader = get_split_laplacian_lmdb_reader(graph_cache_dir, num_eigenvectors, laplacian_spec)
        if reader is None or not reader.has(int(sample_id)):
            return None
        dim = laplacian_spec_dim(laplacian_spec, int(num_eigenvectors)) if feature_dim is None else int(feature_dim)
        return reader.get_unpacked(int(sample_id), dim, operator_name=name)
    candidate_dim, reader = min(candidates, key=lambda item: item[0])
    dim = int(feature_dim) if feature_dim is not None else int(candidate_dim)
    return reader.get_unpacked(int(sample_id), dim, operator_name=name)


def _unpack_laplacian_cache(
    obj,
    feature_dim: int,
    *,
    operator_name: str | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(obj, dict):
        vectors = None
        if operator_name == "fem_u":
            vectors = obj.get("eigenvectors_u")
        elif operator_name == "fem_v":
            vectors = obj.get("eigenvectors_v")
        if vectors is None:
            vectors = obj.get("eigenvectors", obj.get("eigvecs", obj.get("vectors")))
        values = obj.get("eigenvalues", obj.get("eigvals", obj.get("values")))
        if vectors is None:
            raise RuntimeError("Laplacian cache dict is missing eigenvectors.")
        vectors = vectors[:, :feature_dim].float()
        if values is None:
            return None, vectors
        return values[:feature_dim].float(), vectors
    return None, obj[:, :feature_dim].float()


def load_laplacian_cache(
    graph_cache_dir: str | os.PathLike[str],
    sample_id: int,
    num_eigenvectors: int,
    laplacian_spec: str | None,
    laplacian_eig_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    feature_dim = (
        laplacian_spec_dim(laplacian_spec, int(num_eigenvectors))
        if laplacian_eig_dim is None
        else int(laplacian_eig_dim)
    )
    return prepare_laplacian_cache_plan(
        graph_cache_dir,
        int(num_eigenvectors),
        laplacian_spec,
        feature_dim=feature_dim,
    ).load(int(sample_id))
