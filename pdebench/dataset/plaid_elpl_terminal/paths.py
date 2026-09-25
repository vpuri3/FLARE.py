"""Filesystem layout for el-pl terminal manifests and norm stats."""

from __future__ import annotations

from pathlib import Path

from pdebench.dataset.plaid_elpl_terminal.constants import CACHE_TAG
from pdebench.dataset.plaid_elpl_v3.paths import cache_root, shard_dir
from pdebench.dataset.plaid_elpl_v3.paths import sim_to_shard_path as v3_sim_to_shard_path


def manifest_dir(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return cache_root(dataset_dir) / "manifest" / CACHE_TAG / f"split{int(split_seed)}"


def train_manifest_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return manifest_dir(dataset_dir, split_seed=split_seed) / "train.parquet"


def val_manifest_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return manifest_dir(dataset_dir, split_seed=split_seed) / "val.parquet"


def norm_stats_path(dataset_dir: str | Path, *, split_seed: int, use_sdf_features: bool) -> Path:
    """Frozen terminal cache stats (pos/geom + z-score y)."""
    sdf_tag = "sdf1" if bool(use_sdf_features) else "sdf0"
    return cache_root(dataset_dir) / "norm" / CACHE_TAG / f"split{int(split_seed)}_{sdf_tag}_norm_stats.pt"


def runtime_y_norm_stats_path(
    dataset_dir: str | Path,
    *,
    split_seed: int,
    use_sdf_features: bool,
    y_norm_mode: str,
) -> Path:
    """Optional per-mode runtime y stats (separate from frozen cache)."""
    sdf_tag = "sdf1" if bool(use_sdf_features) else "sdf0"
    mode = str(y_norm_mode).strip().lower()
    return (
        cache_root(dataset_dir)
        / "norm"
        / CACHE_TAG
        / "y_runtime"
        / mode
        / f"split{int(split_seed)}_{sdf_tag}_norm_stats.pt"
    )


def is_terminal_cache_complete(
    dataset_dir: str | Path,
    *,
    split_seed: int,
    use_sdf_features: bool,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
) -> bool:
    if not train_manifest_path(dataset_dir, split_seed=split_seed).is_file():
        return False
    if not val_manifest_path(dataset_dir, split_seed=split_seed).is_file():
        return False
    if not norm_stats_path(dataset_dir, split_seed=split_seed, use_sdf_features=use_sdf_features).is_file():
        return False
    shard_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    return bool(shard_root.is_dir()) and bool(list(shard_root.glob("shard_*.pt")))


def resolve_sim_to_shard_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    """Reuse the elpl_v3 sim_to_shard table (same shard layout)."""
    return v3_sim_to_shard_path(dataset_dir, split_seed=split_seed)
