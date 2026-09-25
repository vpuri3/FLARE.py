"""On-disk path helpers for Laplacian caches."""

from __future__ import annotations

from pathlib import Path

from pdebench.dataset.laplacian.spec import laplacian_spec_dim, laplacian_spec_slug


def torchfile_laplacian_cache_path(
    dataset_dir: str | Path,
    sample_id: int | str,
    *,
    laplacian_spec: str,
    num_eigenvectors: int,
) -> Path:
    """PLAID TorchFile layout under ``static_cache/laplacian_pt``.

    ``{dataset_dir}/static_cache/laplacian_pt/{spec_slug}/K{K}/sample_{id:08d}.pt``
    """
    total_dim = laplacian_spec_dim(laplacian_spec, int(num_eigenvectors))
    spec_slug = laplacian_spec_slug(laplacian_spec, int(num_eigenvectors))
    return (
        Path(dataset_dir)
        / "static_cache"
        / "laplacian_pt"
        / spec_slug
        / f"K{int(total_dim)}"
        / f"sample_{int(sample_id):08d}.pt"
    )
