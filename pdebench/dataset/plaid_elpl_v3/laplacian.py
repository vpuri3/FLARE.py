"""Laplacian packing for el-pl v3 shards."""

from __future__ import annotations

from pathlib import Path

import torch

from pdebench.dataset.laplacian import LaplacianService, TorchFileLaplacianBackend
from pdebench.dataset.laplacian.paths import torchfile_laplacian_cache_path
from pdebench.dataset.laplacian.spec import laplacian_spec_dim
from pdebench.dataset.plaid_elpl_v3.schema import LaplacianBundle, TrajectoryBundle


def _laplacian_cache_file(
    dataset_dir: str | Path,
    sample_idx: int,
    num_eigenvectors: int,
    laplacian_spec: str,
) -> Path:
    return torchfile_laplacian_cache_path(
        dataset_dir,
        sample_idx,
        laplacian_spec=laplacian_spec,
        num_eigenvectors=num_eigenvectors,
    )


def load_laplacian_bundle(
    *,
    dataset_dir: str | Path,
    sim_id: int,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> LaplacianBundle | None:
    if int(laplacian_eig_dim) <= 0:
        return None
    service = LaplacianService(TorchFileLaplacianBackend(dataset_dir=dataset_dir))
    eigenvalues, eigenvectors = service.load(
        "plaid_el_pl_dynamics",
        "laplacian_pt",
        sim_id,
        laplacian_spec,
        laplacian_eig_dim,
    )
    bundle = LaplacianBundle(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    return require_laplacian_bundle(
        bundle,
        canonical_dataset="plaid_el_pl_dynamics",
        split_group="laplacian_pt",
        sample_id=sim_id,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )


def require_laplacian_bundle(
    bundle: LaplacianBundle | None,
    *,
    canonical_dataset: str,
    split_group: str,
    sample_id: str | int,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> LaplacianBundle | None:
    """Validate a requested cached bundle, raising the unified precompute error on miss."""
    if int(laplacian_eig_dim) <= 0:
        return bundle
    key = LaplacianService.make_key(
        canonical_dataset,
        split_group,
        sample_id,
        laplacian_spec,
        laplacian_eig_dim,
    )
    expected_dim = laplacian_spec_dim(laplacian_spec, int(laplacian_eig_dim))
    if (
        bundle is None
        or not torch.is_tensor(bundle.eigenvalues)
        or int(bundle.eigenvalues.numel()) != expected_dim
        or not torch.is_tensor(bundle.eigenvectors)
        or bundle.eigenvectors.ndim != 2
        or int(bundle.eigenvectors.shape[-1]) != expected_dim
    ):
        raise FileNotFoundError(LaplacianService.incomplete_cache_message(key))
    return bundle


def pack_laplacian_for_trajectory(
    traj: TrajectoryBundle,
    *,
    dataset_dir: str | Path,
    laplacian_eig_dim: int,
    laplacian_spec: str,
) -> LaplacianBundle | None:
    return load_laplacian_bundle(
        dataset_dir=dataset_dir,
        sim_id=int(traj.sim_id),
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
