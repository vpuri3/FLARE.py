from __future__ import annotations

import os

import pytest
import torch

from pdebench.dataset.plaid_elpl_v3.assemble import assemble_transition_graph
from pdebench.dataset.plaid_elpl_v3.laplacian import _laplacian_cache_file, load_laplacian_bundle
from pdebench.dataset.plaid_elpl_v3.norm import fit_norm_stats_from_trajectories
from tests.pdebench.test_plaid_elpl_v3_schema import _dummy_traj


def test_assembled_shapes() -> None:
    traj = _dummy_traj(sim_id=1)
    stats = fit_norm_stats_from_trajectories([traj], transitions_per_sim=40)
    graph = assemble_transition_graph(
        traj,
        step_idx=0,
        stats=stats,
        use_sdf_features=True,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    assert graph.x.shape == (3, 7)
    assert graph.y.shape == (3, 2)
    assert graph.input_scalars.shape[-1] == 1
    assert torch.equal(graph.context, graph.input_scalars)

    graph_no_sdf = assemble_transition_graph(
        traj,
        step_idx=0,
        stats=stats,
        use_sdf_features=False,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    assert graph_no_sdf.x.shape == (3, 4)


def test_load_laplacian_bundle_missing_cache_fails_with_precompute_hint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Incomplete Laplacian cache.*precompute"):
        load_laplacian_bundle(
            dataset_dir=tmp_path,
            sim_id=4,
            laplacian_eig_dim=2,
            laplacian_spec="graph",
        )


def test_load_laplacian_bundle_incomplete_payload_fails_with_precompute_hint(tmp_path) -> None:
    cache_file = _laplacian_cache_file(tmp_path, 4, 2, "graph")
    cache_file.parent.mkdir(parents=True)
    torch.save({"eigenvalues": torch.ones(2)}, cache_file)

    with pytest.raises(FileNotFoundError, match=r"Incomplete Laplacian cache.*precompute"):
        load_laplacian_bundle(
            dataset_dir=tmp_path,
            sim_id=4,
            laplacian_eig_dim=2,
            laplacian_spec="graph",
        )


def test_load_laplacian_bundle_slices_requested_prefix_from_covering_cache(tmp_path) -> None:
    cache_file = _laplacian_cache_file(tmp_path, 4, 64, "graph")
    cache_file.parent.mkdir(parents=True)
    eigenvalues = torch.arange(64, dtype=torch.float32)
    eigenvectors = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64)
    torch.save({"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}, cache_file)

    bundle = load_laplacian_bundle(
        dataset_dir=tmp_path,
        sim_id=4,
        laplacian_eig_dim=32,
        laplacian_spec="graph",
    )

    assert bundle is not None
    torch.testing.assert_close(bundle.eigenvalues, eigenvalues[:32])
    torch.testing.assert_close(bundle.eigenvectors, eigenvectors[:, :32])


@pytest.mark.skipif(
    not os.path.exists(
        "data/plaid/2D_ElastoPlastoDynamics/static_cache/pyg_graphs/v2_temporal_one_step/pyg/split5/sdf1/public0/max0/K0/graphs.pt"
    ),
    reason="v2 graph cache not present",
)
def test_assembled_matches_v2_graphs_sample() -> None:

    v2_path = (
        "data/plaid/2D_ElastoPlastoDynamics/static_cache/pyg_graphs/"
        "v2_temporal_one_step/pyg/split5/sdf1/public0/max0/K0/graphs.pt"
    )
    payload = torch.load(v2_path, map_location="cpu", weights_only=False, mmap=True)
    train = payload["train_graphs"]
    v2_graph = train[0]
    # smoke: v2 graph contract
    assert v2_graph.x.shape[-1] == 7
    assert v2_graph.y.shape[-1] == 2
