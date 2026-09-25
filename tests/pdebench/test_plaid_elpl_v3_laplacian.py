from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    not os.path.exists("data/plaid/2D_ElastoPlastoDynamics/README.md"),
    reason="PLAID elasto-plasto dataset not present",
)
def test_k64_shard_eigvecs_match_laplacian_pt() -> None:

    import torch

    from pdebench.dataset.plaid_elpl_v3.laplacian import load_laplacian_bundle
    from pdebench.dataset.plaid_elpl_v3.parse import load_shard_payload
    from pdebench.dataset.plaid_elpl_v3.paths import shard_path

    dataset_dir = "data/plaid/2D_ElastoPlastoDynamics"
    shard_file = shard_path(dataset_dir, split_seed=5, shard_id=0, laplacian_eig_dim=64, laplacian_spec="graph")
    if not shard_file.is_file():
        pytest.skip("K64 v3 shard cache not built")
    shard = load_shard_payload(shard_file)
    assert shard.laplacian is not None
    sim_id = int(shard.sim_ids[0])
    local = shard.laplacian[0]
    assert local is not None
    standalone = load_laplacian_bundle(
        dataset_dir=dataset_dir,
        sim_id=sim_id,
        laplacian_eig_dim=64,
        laplacian_spec="graph",
    )
    assert standalone is not None
    torch.testing.assert_close(local.eigenvalues, standalone.eigenvalues, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(local.eigenvectors, standalone.eigenvectors, rtol=1e-5, atol=1e-5)
