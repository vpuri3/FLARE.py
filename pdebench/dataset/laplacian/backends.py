"""Cache backends for LaplacianService."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import torch

from pdebench.dataset.laplacian.service import LaplacianCacheKey


class LaplacianBackend(Protocol):
    def has(self, key: LaplacianCacheKey) -> bool: ...

    def load(self, key: LaplacianCacheKey) -> tuple[torch.Tensor | None, torch.Tensor]: ...

    def save(
        self,
        key: LaplacianCacheKey,
        eigvals: torch.Tensor | None,
        eigvecs: torch.Tensor,
    ) -> None: ...

    def covering_key(self, key: LaplacianCacheKey) -> LaplacianCacheKey | None:
        """Exact key if present, else smallest same-operator cache with K' >= K."""
        ...


def _parts_cover_request(requested_key: LaplacianCacheKey, candidate: LaplacianCacheKey) -> bool:
    from pdebench.dataset.laplacian.service import _parts_cover, _spec_parts

    return _parts_cover(
        _spec_parts(requested_key.spec, requested_key.K),
        _spec_parts(candidate.spec, candidate.K),
    )


@dataclass
class InMemoryLaplacianBackend:
    """Dict-backed store for unit tests and synthetic fixtures."""

    _store: dict[LaplacianCacheKey, tuple[torch.Tensor | None, torch.Tensor]] | None = None

    def __post_init__(self) -> None:
        if self._store is None:
            self._store = {}

    def has(self, key: LaplacianCacheKey) -> bool:
        assert self._store is not None
        return key in self._store

    def covering_key(self, key: LaplacianCacheKey) -> LaplacianCacheKey | None:
        assert self._store is not None
        if key in self._store:
            return key
        best: LaplacianCacheKey | None = None
        for candidate in self._store:
            if not isinstance(candidate, LaplacianCacheKey):
                continue
            if (
                candidate.canonical_dataset != key.canonical_dataset
                or candidate.split_group != key.split_group
                or candidate.sample_id != key.sample_id
            ):
                continue
            if int(candidate.K) < int(key.K):
                continue
            if not _parts_cover_request(key, candidate):
                continue
            if best is None or int(candidate.K) < int(best.K):
                best = candidate
        return best

    def load(self, key: LaplacianCacheKey) -> tuple[torch.Tensor | None, torch.Tensor]:
        assert self._store is not None
        return self._store[key]

    def save(
        self,
        key: LaplacianCacheKey,
        eigvals: torch.Tensor | None,
        eigvecs: torch.Tensor,
    ) -> None:
        assert self._store is not None
        self._store[key] = (eigvals, eigvecs)


class LmdbLaplacianBackend:
    """Thin wrapper over ``laplacian.lmdb`` sharded LMDB I/O.

    Resolves ``(canonical_dataset, split_group)`` to a graph-cache directory via
    ``graph_cache_dirs`` (mapping) or ``resolve_graph_cache_dir`` (callable).
    Sample ids are coerced to ``int`` for the existing LMDB key format.

    Train path: ``LaplacianService`` uses ``has`` / ``load`` / ``save`` (per-sample).
    Bulk GINOT precompute: use ``open_bulk_writer`` (``ShardAffineWriterSession``) so
    multi-GPU lanes write the same on-disk layout without going through per-key ``save``.
    """

    def __init__(
        self,
        graph_cache_dirs: Mapping[tuple[str, str], str | Path] | None = None,
        *,
        resolve_graph_cache_dir: Callable[[str, str], str | Path] | None = None,
    ):
        if graph_cache_dirs is None and resolve_graph_cache_dir is None:
            raise ValueError("LmdbLaplacianBackend requires graph_cache_dirs or resolve_graph_cache_dir")
        self._dirs = dict(graph_cache_dirs) if graph_cache_dirs is not None else None
        self._resolve = resolve_graph_cache_dir

    def _graph_cache_dir(self, key: LaplacianCacheKey) -> Path:
        if self._resolve is not None:
            return Path(self._resolve(key.canonical_dataset, key.split_group))
        assert self._dirs is not None
        lookup = (key.canonical_dataset, key.split_group)
        if lookup not in self._dirs:
            raise KeyError(
                f"No graph_cache_dir registered for dataset={key.canonical_dataset!r} "
                f"split_group={key.split_group!r}"
            )
        return Path(self._dirs[lookup])

    def graph_cache_dir_for(self, canonical_dataset: str, split_group: str) -> Path:
        """Resolve the graph-cache directory for a dataset/split without a full cache key."""
        return self._graph_cache_dir(
            LaplacianCacheKey(
                canonical_dataset=str(canonical_dataset),
                split_group=str(split_group),
                sample_id="0",
                spec="graph",
                K=0,
            )
        )

    def open_bulk_writer(
        self,
        canonical_dataset: str,
        split_group: str,
        num_eigenvectors: int,
        assigned_lanes: list[tuple[str, int]],
    ):
        """Open a sharded LMDB writer session for multi-GPU / multi-lane precompute."""
        return self.open_bulk_writer_at(
            self.graph_cache_dir_for(canonical_dataset, split_group),
            int(num_eigenvectors),
            assigned_lanes,
        )

    @staticmethod
    def open_bulk_writer_at(
        graph_cache_dir: str | Path,
        num_eigenvectors: int,
        assigned_lanes: list[tuple[str, int]],
    ):
        """Open a bulk writer for an already-resolved graph-cache directory."""
        from pdebench.dataset.laplacian.lmdb import ShardAffineWriterSession

        return ShardAffineWriterSession(graph_cache_dir, int(num_eigenvectors), assigned_lanes)

    @staticmethod
    def _sample_id_int(sample_id: str) -> int:
        return int(sample_id)

    def has(self, key: LaplacianCacheKey) -> bool:
        from pdebench.dataset.laplacian.lmdb import split_laplacian_lmdb_has

        return split_laplacian_lmdb_has(
            self._graph_cache_dir(key),
            self._sample_id_int(key.sample_id),
            int(key.K),
            key.spec,
        )

    def covering_key(self, key: LaplacianCacheKey) -> LaplacianCacheKey | None:
        """Prefer exact K; else smallest on-disk same-operator cache with K' >= K."""
        from pdebench.dataset.laplacian.lmdb import is_sharded_split_laplacian_cache, split_laplacian_lmdb_has
        from pdebench.dataset.laplacian.spec import laplacian_spec_slug

        if self.has(key):
            return key

        graph_cache_dir = self._graph_cache_dir(key)
        static_cache = graph_cache_dir.parent.parent.parent
        split_group = graph_cache_dir.parent.name
        split_name = graph_cache_dir.name
        lap_root = static_cache / "laplacian_lmdb"
        if not lap_root.is_dir():
            return None

        sample_id = self._sample_id_int(key.sample_id)
        best: LaplacianCacheKey | None = None
        for spec_dir in lap_root.iterdir():
            if not spec_dir.is_dir():
                continue
            for k_dir in spec_dir.iterdir():
                if not k_dir.is_dir() or not k_dir.name.startswith("K"):
                    continue
                try:
                    cand_k = int(k_dir.name[1:])
                except ValueError:
                    continue
                if cand_k < int(key.K):
                    continue
                split_root = k_dir / split_group / split_name
                if not split_root.is_dir():
                    continue
                # Reconstruct a candidate spec string that yields this slug at cand_k.
                # Single-operator caches use slug "{op}{K}" (fem_* → fem).
                slug = spec_dir.name
                if slug.endswith(str(cand_k)):
                    op = slug[: -len(str(cand_k))]
                else:
                    continue
                if op == "fem":
                    cand_spec = f"fem_v:{cand_k}"
                else:
                    cand_spec = f"{op}:{cand_k}"
                if laplacian_spec_slug(cand_spec, cand_k) != slug:
                    continue
                candidate = LaplacianCacheKey(
                    canonical_dataset=key.canonical_dataset,
                    split_group=key.split_group,
                    sample_id=key.sample_id,
                    spec=cand_spec,
                    K=cand_k,
                )
                if not _parts_cover_request(key, candidate):
                    continue
                if not is_sharded_split_laplacian_cache(graph_cache_dir, cand_k, cand_spec):
                    continue
                if not split_laplacian_lmdb_has(graph_cache_dir, sample_id, cand_k, cand_spec):
                    continue
                if best is None or cand_k < int(best.K):
                    best = candidate
        return best

    def load(self, key: LaplacianCacheKey) -> tuple[torch.Tensor | None, torch.Tensor]:
        from pdebench.dataset.laplacian.lmdb import load_laplacian_cache

        return load_laplacian_cache(
            self._graph_cache_dir(key),
            self._sample_id_int(key.sample_id),
            int(key.K),
            key.spec,
        )

    def save(
        self,
        key: LaplacianCacheKey,
        eigvals: torch.Tensor | None,
        eigvecs: torch.Tensor,
    ) -> None:
        from pdebench.dataset.laplacian.lmdb import save_split_laplacian_lmdb

        if eigvals is None:
            eigvals = torch.zeros(int(key.K), dtype=torch.float32)
        save_split_laplacian_lmdb(
            self._graph_cache_dir(key),
            self._sample_id_int(key.sample_id),
            int(key.K),
            key.spec,
            eigvals,
            eigvecs,
        )


@dataclass
class TorchFileLaplacianBackend:
    """Per-sample ``.pt`` cache for PLAID static-mesh Laplacian features.

    Layout::

        {dataset_dir}/static_cache/laplacian_pt/{spec_slug}/K{K}/sample_{id:08d}.pt
    """

    dataset_dir: Path | str

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)

    def _cache_file(self, key: LaplacianCacheKey) -> Path:
        from pdebench.dataset.laplacian.paths import torchfile_laplacian_cache_path

        return torchfile_laplacian_cache_path(
            self.dataset_dir,
            key.sample_id,
            laplacian_spec=key.spec,
            num_eigenvectors=int(key.K),
        )

    def has(self, key: LaplacianCacheKey) -> bool:
        return self._cache_file(key).exists()

    def covering_key(self, key: LaplacianCacheKey) -> LaplacianCacheKey | None:
        if self.has(key):
            return key
        from pdebench.dataset.laplacian.spec import laplacian_spec_slug

        root = Path(self.dataset_dir) / "static_cache" / "laplacian_pt"
        if not root.is_dir():
            return None
        best: LaplacianCacheKey | None = None
        for spec_dir in root.iterdir():
            if not spec_dir.is_dir():
                continue
            for k_dir in spec_dir.iterdir():
                if not k_dir.is_dir() or not k_dir.name.startswith("K"):
                    continue
                try:
                    cand_k = int(k_dir.name[1:])
                except ValueError:
                    continue
                if cand_k < int(key.K):
                    continue
                slug = spec_dir.name
                if not slug.endswith(str(cand_k)):
                    continue
                op = slug[: -len(str(cand_k))]
                cand_spec = f"fem_v:{cand_k}" if op == "fem" else f"{op}:{cand_k}"
                if laplacian_spec_slug(cand_spec, cand_k) != slug:
                    continue
                candidate = LaplacianCacheKey(
                    canonical_dataset=key.canonical_dataset,
                    split_group=key.split_group,
                    sample_id=key.sample_id,
                    spec=cand_spec,
                    K=cand_k,
                )
                if not _parts_cover_request(key, candidate):
                    continue
                if not self._cache_file(candidate).exists():
                    continue
                if best is None or cand_k < int(best.K):
                    best = candidate
        return best

    def load(self, key: LaplacianCacheKey) -> tuple[torch.Tensor | None, torch.Tensor]:
        cache_file = self._cache_file(key)
        payload = torch.load(cache_file, map_location="cpu", weights_only=True, mmap=True)
        eigvals = payload.get("eigenvalues")
        eigvecs = payload["eigenvectors"]
        if eigvals is not None:
            eigvals = eigvals.float()
        return eigvals, eigvecs.float()

    def save(
        self,
        key: LaplacianCacheKey,
        eigvals: torch.Tensor | None,
        eigvecs: torch.Tensor,
    ) -> None:
        cache_file = self._cache_file(key)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eigenvalues": eigvals.detach().cpu().float() if eigvals is not None else None,
            "eigenvectors": eigvecs.detach().cpu().float(),
        }
        tmp = cache_file.parent / f".{cache_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, cache_file)
        finally:
            tmp.unlink(missing_ok=True)
