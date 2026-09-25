"""Unified Laplacian eigenfeature service and cache backends."""

from __future__ import annotations

from pdebench.dataset.laplacian.backends import InMemoryLaplacianBackend, LmdbLaplacianBackend, TorchFileLaplacianBackend
from pdebench.dataset.laplacian.compute import (
    _cells_to_tetrahedra_numpy,
    compute_laplacian_eigendecomp,
    compute_laplacian_eigendecomp_part,
    compute_laplacian_feature_part,
    compute_laplacian_features,
    compute_laplacian_fem_eigendecomp_both,
)
from pdebench.dataset.laplacian.lmdb import (
    ResolvedLaplacianCachePlan,
    ShardAffineWriterSession,
    SplitLaplacianWriteSession,
    laplacian_eigen_payload_bytes,
    list_split_laplacian_cached_sample_ids,
    load_laplacian_cache,
    prepare_laplacian_cache_plan,
    save_split_laplacian_lmdb,
    split_laplacian_lmdb_dir,
    split_laplacian_lmdb_has,
    write_split_laplacian_shard_batch,
)
from pdebench.dataset.laplacian.paths import torchfile_laplacian_cache_path
from pdebench.dataset.laplacian.service import LaplacianCacheKey, LaplacianService
from pdebench.dataset.laplacian.spec import (
    DATASET_LAPLACIAN_SPECS,
    DEFAULT_LAPLACIAN_EIGENVECTORS,
    DEFAULT_LAPLACIAN_SPECS,
    LAPLACIAN_OPERATORS,
    default_laplacian_specs_for_dataset,
    laplacian_cache_spec_entry,
    laplacian_spec_dim,
    laplacian_spec_entries,
    laplacian_spec_slug,
    parse_laplacian_spec,
    resolve_laplacian_specs_for_dataset,
    single_laplacian_spec_part,
)

__all__ = [
    "DATASET_LAPLACIAN_SPECS",
    "DEFAULT_LAPLACIAN_EIGENVECTORS",
    "DEFAULT_LAPLACIAN_SPECS",
    "InMemoryLaplacianBackend",
    "LAPLACIAN_OPERATORS",
    "LaplacianCacheKey",
    "LaplacianService",
    "LmdbLaplacianBackend",
    "ResolvedLaplacianCachePlan",
    "ShardAffineWriterSession",
    "SplitLaplacianWriteSession",
    "TorchFileLaplacianBackend",
    "_cells_to_tetrahedra_numpy",
    "compute_laplacian_eigendecomp",
    "compute_laplacian_eigendecomp_part",
    "compute_laplacian_feature_part",
    "compute_laplacian_features",
    "compute_laplacian_fem_eigendecomp_both",
    "default_laplacian_specs_for_dataset",
    "laplacian_cache_spec_entry",
    "laplacian_eigen_payload_bytes",
    "laplacian_spec_dim",
    "laplacian_spec_entries",
    "laplacian_spec_slug",
    "list_split_laplacian_cached_sample_ids",
    "load_laplacian_cache",
    "parse_laplacian_spec",
    "prepare_laplacian_cache_plan",
    "resolve_laplacian_specs_for_dataset",
    "save_split_laplacian_lmdb",
    "single_laplacian_spec_part",
    "split_laplacian_lmdb_dir",
    "split_laplacian_lmdb_has",
    "torchfile_laplacian_cache_path",
    "write_split_laplacian_shard_batch",
]
