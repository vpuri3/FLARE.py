from __future__ import annotations

import os

import pytest

from pdebench.dataset.plaid_elpl_terminal.assemble import assemble_terminal_graph
from pdebench.dataset.plaid_elpl_terminal.dataset import ElPlTerminalDataset
from pdebench.dataset.plaid_elpl_terminal.norm import fit_terminal_norm_stats_from_trajectories
from tests.pdebench.test_plaid_elpl_v3_schema import _dummy_traj


def test_terminal_requested_laplacian_rejects_missing_bundle() -> None:
    traj = _dummy_traj(sim_id=9)
    stats = fit_terminal_norm_stats_from_trajectories([traj])
    with pytest.raises(FileNotFoundError, match=r"Incomplete Laplacian cache.*precompute"):
        assemble_terminal_graph(
            traj,
            stats=stats,
            runtime_y_normalizer=stats.cache_y_normalizer,
            use_sdf_features=False,
            graph_backend="pyg",
            target_fields=("U_x",),
            laplacian_eig_dim=8,
            laplacian_spec="graph",
            bandwidth=1.0,
        )


@pytest.mark.skipif(
    not os.path.exists("data/plaid/2D_ElastoPlastoDynamics/README.md"),
    reason="PLAID elasto-plasto dataset not present",
)
def test_load_mesh_static_dataset_terminal_counts() -> None:
    from pdebench.dataset.plaid_datasets import load_mesh_static_dataset

    train, val, meta = load_mesh_static_dataset(
        data_root="data",
        dataset_name="plaid_elpl_terminal",
        split_seed=5,
        graph_backend="pyg",
        use_sdf_features=False,
        max_samples=0,
    )
    assert len(train) == 800
    assert len(val) == 200
    assert meta["c_in"] == 2
    assert meta["time_cond"] is False
    assert meta["plaid_temporal_one_step"] is False
    assert meta["plaid_terminal_prediction"] is True
    assert meta["plaid_terminal_target_fields"] == ["U_x"]
    assert meta["target_fields"] == ["U_x"]
    assert meta["c_out"] == 1
    assert meta["plaid_use_sdf_features"] is False
    assert isinstance(train, ElPlTerminalDataset)

    graph = train[0]
    assert graph.x.shape[-1] == 2
    assert graph.y.shape[-1] == 1
    assert graph.output_fields_names == ["U_x"]
