from __future__ import annotations

import os

import pytest

from pdebench.dataset.plaid_elpl_v3.dataset import ElPlTransitionDataset


@pytest.mark.skipif(
    not os.path.exists("data/plaid/2D_ElastoPlastoDynamics/README.md"),
    reason="PLAID elasto-plasto dataset not present",
)
def test_load_mesh_static_dataset_v3_counts() -> None:
    from pdebench.dataset.plaid_datasets import load_mesh_static_dataset

    train, val, meta = load_mesh_static_dataset(
        data_root="data",
        dataset_name="plaid_el_pl_dynamics",
        split_seed=5,
        graph_backend="pyg",
        use_sdf_features=True,
        max_samples=0,
    )
    assert len(train) == 32000
    assert len(val) == 8000
    assert meta["c_in"] == 8
    assert isinstance(train, ElPlTransitionDataset)


@pytest.mark.skipif(
    not os.path.exists("data/plaid/2D_ElastoPlastoDynamics/README.md"),
    reason="PLAID elasto-plasto dataset not present",
)
def test_load_mesh_static_dataset_v3_sdf_off_c_in() -> None:
    from pdebench.dataset.plaid_datasets import load_mesh_static_dataset

    _train, _val, meta = load_mesh_static_dataset(
        data_root="data",
        dataset_name="plaid_el_pl_dynamics",
        split_seed=5,
        graph_backend="pyg",
        use_sdf_features=False,
        max_samples=0,
        laplacian_eig_dim=0,
    )
    assert meta["c_in"] == 5
    assert meta["plaid_use_sdf_features"] is False
