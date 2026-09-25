import os
from pathlib import Path

import torch

from pdebench.dataset.laplacian import LaplacianService, TorchFileLaplacianBackend
from pdebench.dataset.laplacian.paths import torchfile_laplacian_cache_path
from pdebench.dataset.laplacian.spec import laplacian_spec_slug
from pdebench.dataset.sample import Sample, SampleKind


def _plaid_laplacian_cache_file(
    dataset_dir: str,
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


def _plaid_laplacian_split_group(laplacian_spec: str, num_eigenvectors: int) -> str:
    return laplacian_spec_slug(str(laplacian_spec), int(num_eigenvectors))


def _plaid_laplacian_service(dataset_dir: str) -> LaplacianService:
    return LaplacianService(TorchFileLaplacianBackend(dataset_dir=dataset_dir))


def _ensure_plaid_laplacian(
    *,
    dataset_dir: str,
    dataset_name: str,
    sample_idx: int,
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    cells: torch.Tensor,
    num_eigenvectors: int,
    laplacian_spec: str,
    device: str | torch.device | None = None,
) -> None:
    """Compute+save missing PLAID Laplacian eigenpairs (precompute / cache-build only)."""
    if int(num_eigenvectors) <= 0:
        return
    service = _plaid_laplacian_service(dataset_dir)
    split_group = _plaid_laplacian_split_group(laplacian_spec, num_eigenvectors)
    device = torch.device(
        device
        if device is not None
        else os.environ.get("PLAID_LAPLACIAN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    sample = Sample(
        pos=pos.to(device=device, dtype=torch.float32),
        y=torch.zeros(pos.shape[0], 1, device=device),
        edge_index=edge_index.to(device=device, dtype=torch.long),
        sample_id=str(int(sample_idx)),
        kind=SampleKind.STATIC,
        extras={"cells": cells.to(device=device, dtype=torch.long)},
    )
    service.ensure(
        dataset_name,
        split_group,
        sample_ids=[int(sample_idx)],
        spec=laplacian_spec,
        K=int(num_eigenvectors),
        samples={str(int(sample_idx)): sample},
    )


def _load_plaid_laplacian(
    *,
    dataset_dir: str,
    dataset_name: str,
    sample_idx: int,
    num_eigenvectors: int,
    laplacian_spec: str,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Load cached PLAID Laplacian eigenpairs. Fail-loud on miss."""
    service = _plaid_laplacian_service(dataset_dir)
    split_group = _plaid_laplacian_split_group(laplacian_spec, num_eigenvectors)
    return service.load(
        dataset_name,
        split_group,
        int(sample_idx),
        laplacian_spec,
        int(num_eigenvectors),
    )


def _attach_laplacian_features(
    graph,
    *,
    dataset_dir: str,
    dataset_name: str,
    sample_idx: int,
    laplacian_eig_dim: int,
    laplacian_spec: str,
):
    """Load cached Laplacian features onto ``graph``. Never computes on miss."""
    if int(laplacian_eig_dim) <= 0:
        return graph
    service = _plaid_laplacian_service(dataset_dir)
    split_group = _plaid_laplacian_split_group(laplacian_spec, laplacian_eig_dim)
    pos = graph.pos
    sample = Sample(
        pos=pos,
        y=torch.zeros(pos.shape[0], 1, dtype=torch.float32),
        edge_index=graph.edge_index,
        sample_id=str(int(sample_idx)),
        kind=SampleKind.STATIC,
    )
    sample = service.attach(
        sample,
        canonical_dataset=dataset_name,
        split_group=split_group,
        spec=laplacian_spec,
        K=int(laplacian_eig_dim),
    )
    graph.laplacian_eig = sample.laplacian_eig.float()
    if sample.laplacian_eigvals is not None:
        graph.laplacian_eigvals = sample.laplacian_eigvals.reshape(1, -1).float()
    else:
        graph.laplacian_eigvals = torch.zeros(1, int(laplacian_eig_dim), dtype=torch.float32)
    return graph
