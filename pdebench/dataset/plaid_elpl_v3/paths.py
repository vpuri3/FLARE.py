"""Filesystem layout for el-pl v3 cache."""

from __future__ import annotations

import json
from pathlib import Path

from pdebench.dataset.laplacian.spec import laplacian_spec_slug
from pdebench.dataset.plaid_elpl_v3.constants import CACHE_FORMAT, CACHE_SCHEMA_VERSION


def cache_root(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / "static_cache" / CACHE_FORMAT


def laplacian_shard_tag(laplacian_eig_dim: int, laplacian_spec: str) -> str:
    if int(laplacian_eig_dim) <= 0:
        return "K0"
    return f"graph_{laplacian_spec_slug(str(laplacian_spec), int(laplacian_eig_dim))}"


def shard_dir(
    dataset_dir: str | Path,
    *,
    split_seed: int,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
) -> Path:
    return (
        cache_root(dataset_dir)
        / "shards"
        / f"split{int(split_seed)}"
        / laplacian_shard_tag(laplacian_eig_dim, laplacian_spec)
    )


def shard_path(
    dataset_dir: str | Path,
    *,
    split_seed: int,
    shard_id: int,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
) -> Path:
    return shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    ) / f"shard_{int(shard_id):04d}.pt"


def manifest_dir(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return cache_root(dataset_dir) / "manifest" / f"split{int(split_seed)}"


def train_manifest_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return manifest_dir(dataset_dir, split_seed=split_seed) / "train.parquet"


def val_manifest_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return manifest_dir(dataset_dir, split_seed=split_seed) / "val.parquet"


def sim_to_shard_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return manifest_dir(dataset_dir, split_seed=split_seed) / "sim_to_shard.parquet"


def norm_stats_path(dataset_dir: str | Path, *, split_seed: int) -> Path:
    return cache_root(dataset_dir) / "norm" / f"split{int(split_seed)}_norm_stats.pt"


def meta_path(dataset_dir: str | Path) -> Path:
    return cache_root(dataset_dir) / "meta.json"


def write_meta(
    dataset_dir: str | Path,
    *,
    split_seed: int,
    bandwidth: float,
    target_fields: tuple[str, ...],
    laplacian_eig_dim: int,
    laplacian_spec: str,
    num_shards: int,
) -> None:
    path = meta_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CACHE_FORMAT,
        "schema_version": CACHE_SCHEMA_VERSION,
        "split_seed": int(split_seed),
        "bandwidth": float(bandwidth),
        "target_fields": list(target_fields),
        "laplacian_eig_dim": int(laplacian_eig_dim),
        "laplacian_spec": str(laplacian_spec),
        "num_shards": int(num_shards),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def is_cache_complete(
    dataset_dir: str | Path,
    *,
    split_seed: int,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
    expected_shards: int | None = None,
) -> bool:
    if not meta_path(dataset_dir).is_file():
        return False
    if not norm_stats_path(dataset_dir, split_seed=split_seed).is_file():
        return False
    if not train_manifest_path(dataset_dir, split_seed=split_seed).is_file():
        return False
    if not val_manifest_path(dataset_dir, split_seed=split_seed).is_file():
        return False
    if not sim_to_shard_path(dataset_dir, split_seed=split_seed).is_file():
        return False
    shard_root = shard_dir(
        dataset_dir,
        split_seed=split_seed,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    if not shard_root.is_dir():
        return False
    shard_files = sorted(shard_root.glob("shard_*.pt"))
    if not shard_files:
        return False
    if expected_shards is not None and len(shard_files) != int(expected_shards):
        return False
    return True
