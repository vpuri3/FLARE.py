"""Unified Laplacian eigenfeature service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import torch

from pdebench.dataset.laplacian.compute import compute_laplacian_eigendecomp_part
from pdebench.dataset.laplacian.spec import laplacian_spec_dim, parse_laplacian_spec
from pdebench.dataset.sample import Sample

PRECOMPUTE_HINT = (
    "Run python -m pdebench.dataset.laplacian.precompute first "
    "(or LaplacianService.ensure / precompute)."
)


def _spec_parts(spec: str, K: int) -> list[tuple[str, int]]:
    return parse_laplacian_spec(spec, int(K))


def _parts_cover(requested: list[tuple[str, int]], covering: list[tuple[str, int]]) -> bool:
    if len(requested) != len(covering):
        return False
    return all(
        req_name == cov_name and int(cov_count) >= int(req_count)
        for (req_name, req_count), (cov_name, cov_count) in zip(requested, covering, strict=True)
    )


def slice_eigenpairs_to_spec(
    eigvals: torch.Tensor | None,
    eigvecs: torch.Tensor,
    *,
    requested_spec: str,
    requested_K: int,
    covering_spec: str,
    covering_K: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Take a prefix of each operator block from a larger cached eigenpair payload."""
    req_parts = _spec_parts(requested_spec, requested_K)
    cov_parts = _spec_parts(covering_spec, covering_K)
    if not _parts_cover(req_parts, cov_parts):
        raise ValueError(
            f"Cannot slice requested spec={requested_spec!r} K={requested_K} "
            f"from covering spec={covering_spec!r} K={covering_K}"
        )
    if (
        len(req_parts) == 1
        and len(cov_parts) == 1
        and int(req_parts[0][1]) == int(requested_K)
        and int(cov_parts[0][1]) == int(eigvecs.shape[-1])
    ):
        k = int(requested_K)
        sliced_vecs = eigvecs[:, :k].contiguous()
        sliced_vals = None if eigvals is None else eigvals[:k].contiguous()
        return sliced_vals, sliced_vecs

    vec_blocks: list[torch.Tensor] = []
    val_blocks: list[torch.Tensor] = []
    offset = 0
    for (_req_name, req_count), (_cov_name, cov_count) in zip(req_parts, cov_parts, strict=True):
        req_c = int(req_count)
        cov_c = int(cov_count)
        vec_blocks.append(eigvecs[:, offset : offset + req_c].contiguous())
        if eigvals is not None:
            val_blocks.append(eigvals[offset : offset + req_c].contiguous())
        offset += cov_c
    sliced_vecs = vec_blocks[0] if len(vec_blocks) == 1 else torch.cat(vec_blocks, dim=-1)
    sliced_vals = None
    if eigvals is not None:
        sliced_vals = val_blocks[0] if len(val_blocks) == 1 else torch.cat(val_blocks, dim=-1)
    return sliced_vals, sliced_vecs


@dataclass(frozen=True)
class LaplacianCacheKey:
    """Stable cache identity for one sample's spectral features."""

    canonical_dataset: str
    split_group: str
    sample_id: str
    spec: str
    K: int


class _Backend(Protocol):
    def has(self, key: LaplacianCacheKey) -> bool: ...

    def load(self, key: LaplacianCacheKey) -> tuple[torch.Tensor | None, torch.Tensor]: ...

    def save(
        self,
        key: LaplacianCacheKey,
        eigvals: torch.Tensor | None,
        eigvecs: torch.Tensor,
    ) -> None: ...


ComputeFn = Callable[[LaplacianCacheKey, Sample], tuple[torch.Tensor | None, torch.Tensor]]


def _default_compute(key: LaplacianCacheKey, sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse ginot LOBPCG/FEM operators — single compute path."""
    if sample.edge_index is None:
        raise ValueError(
            f"Sample {key.sample_id!r} needs edge_index to compute Laplacian eigenpairs "
            f"(spec={key.spec!r}, K={key.K})."
        )
    parts = parse_laplacian_spec(key.spec, int(key.K))
    if not parts:
        raise ValueError(f"Empty Laplacian spec {key.spec!r}")
    cells = sample.extras.get("cells") if sample.extras else None
    values: list[torch.Tensor] = []
    vectors: list[torch.Tensor] = []
    for name, count in parts:
        eigvals, eigvecs = compute_laplacian_eigendecomp_part(
            sample.edge_index,
            sample.pos,
            cells,
            name,
            int(count),
        )
        values.append(eigvals)
        vectors.append(eigvecs)
    if len(values) == 1:
        return values[0], vectors[0]
    return torch.cat(values, dim=-1), torch.cat(vectors, dim=-1)


class LaplacianService:
    """One workflow: ensure / is_complete / load / attach (no train compute-on-miss)."""

    def __init__(
        self,
        backend: _Backend,
        *,
        compute_fn: ComputeFn | None = None,
    ):
        self.backend = backend
        self._compute_fn: ComputeFn = compute_fn or _default_compute

    @staticmethod
    def make_key(
        canonical_dataset: str,
        split_group: str,
        sample_id: str | int,
        spec: str,
        K: int,
    ) -> LaplacianCacheKey:
        return LaplacianCacheKey(
            canonical_dataset=str(canonical_dataset),
            split_group=str(split_group),
            sample_id=str(sample_id),
            spec=str(spec),
            K=int(K),
        )

    def _covering_key(self, key: LaplacianCacheKey) -> LaplacianCacheKey | None:
        """Exact key if present, else a same-operator cache with K' >= K (prefix-reusable)."""
        finder = getattr(self.backend, "covering_key", None)
        if callable(finder):
            return finder(key)
        if self.backend.has(key):
            return key
        return None

    def is_complete(
        self,
        canonical_dataset: str,
        split_group: str,
        sample_ids: Sequence[str | int],
        spec: str,
        K: int,
    ) -> bool:
        if int(K) <= 0:
            return True
        return all(
            self._covering_key(self.make_key(canonical_dataset, split_group, sid, spec, K)) is not None
            for sid in sample_ids
        )

    def ensure(
        self,
        canonical_dataset: str,
        split_group: str,
        sample_ids: Sequence[str | int],
        spec: str,
        K: int,
        *,
        samples: Mapping[str, Sample] | None = None,
    ) -> None:
        """Compute and save missing eigenpairs (precompute path). Does not attach."""
        if int(K) <= 0:
            return
        sample_map = {str(k): v for k, v in (samples or {}).items()}
        for sid in sample_ids:
            key = self.make_key(canonical_dataset, split_group, sid, spec, K)
            if self._covering_key(key) is not None:
                continue
            sample = sample_map.get(key.sample_id)
            if sample is None:
                raise KeyError(
                    f"Missing Sample for sample_id={key.sample_id!r} required by ensure(); "
                    f"pass samples={{...}} or pre-populate the cache."
                )
            eigvals, eigvecs = self._compute_fn(key, sample)
            self.backend.save(key, eigvals, eigvecs)

    def precompute(
        self,
        canonical_dataset: str,
        split_group: str,
        sample_ids: Sequence[str | int],
        spec: str,
        K: int,
        *,
        samples: Mapping[str, Sample] | None = None,
    ) -> None:
        """Alias for :meth:`ensure` (design / CLI naming)."""
        self.ensure(
            canonical_dataset,
            split_group,
            sample_ids,
            spec,
            K,
            samples=samples,
        )

    def load(
        self,
        canonical_dataset: str,
        split_group: str,
        sample_id: str | int,
        spec: str,
        K: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if int(K) <= 0:
            raise ValueError("load() requires K > 0")
        key = self.make_key(canonical_dataset, split_group, sample_id, spec, K)
        covering = self._covering_key(key)
        if covering is None:
            raise FileNotFoundError(self.incomplete_cache_message(key))
        try:
            eigvals, eigvecs = self.backend.load(covering)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise FileNotFoundError(self.incomplete_cache_message(key)) from exc
        if covering.K != key.K or covering.spec != key.spec:
            try:
                eigvals, eigvecs = slice_eigenpairs_to_spec(
                    eigvals,
                    eigvecs,
                    requested_spec=key.spec,
                    requested_K=key.K,
                    covering_spec=covering.spec,
                    covering_K=covering.K,
                )
            except (TypeError, ValueError, IndexError, RuntimeError) as exc:
                raise FileNotFoundError(self.incomplete_cache_message(key)) from exc
        expected_dim = laplacian_spec_dim(key.spec, key.K)
        if (
            not torch.is_tensor(eigvecs)
            or eigvecs.ndim != 2
            or int(eigvecs.shape[-1]) != expected_dim
            or (
                eigvals is not None
                and (not torch.is_tensor(eigvals) or int(eigvals.numel()) != expected_dim)
            )
        ):
            raise FileNotFoundError(self.incomplete_cache_message(key))
        return eigvals, eigvecs

    def attach(
        self,
        sample: Sample,
        *,
        canonical_dataset: str,
        split_group: str,
        spec: str,
        K: int,
    ) -> Sample:
        """Load cached eigenpairs onto ``sample``. Never computes on miss."""
        if int(K) <= 0:
            return sample
        eigvals, eigvecs = self.load(
            canonical_dataset,
            split_group,
            sample.sample_id,
            spec,
            K,
        )
        sample.laplacian_eig = eigvecs
        sample.laplacian_eigvals = eigvals
        return sample

    @staticmethod
    def incomplete_cache_message(key: LaplacianCacheKey) -> str:
        return (
            f"Incomplete Laplacian cache for dataset={key.canonical_dataset!r} "
            f"split_group={key.split_group!r} sample_id={key.sample_id!r} "
            f"spec={key.spec!r} K={key.K}. {PRECOMPUTE_HINT}"
        )
