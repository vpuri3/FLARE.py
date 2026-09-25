from __future__ import annotations

from pdebench.dataset.plaid_elpl_terminal.assemble import assemble_terminal_graph
from pdebench.dataset.plaid_elpl_terminal.constants import FINAL_STEP_IDX
from pdebench.dataset.plaid_elpl_terminal.norm import fit_terminal_norm_stats_from_trajectories
from pdebench.models.graph_models.glt import GLT, GLTConfig
from pdebench.models.graph_models.glt import NonePEConfig
from tests.pdebench.test_plaid_elpl_v3_schema import _dummy_traj


def test_terminal_assembled_shapes_sdf_off() -> None:
    traj = _dummy_traj(sim_id=1)
    stats = fit_terminal_norm_stats_from_trajectories([traj])
    graph = assemble_terminal_graph(
        traj,
        stats=stats,
        runtime_y_normalizer=stats.cache_y_normalizer,
        use_sdf_features=False,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    assert graph.x.shape == (3, 2)
    assert graph.y.shape == (3, 2)
    assert not hasattr(graph, "input_scalars")
    assert graph.metadata["final_step_idx"] == FINAL_STEP_IDX
    assert graph.metadata["dataset"] == "plaid_elpl_terminal"


def test_terminal_sdf_off_c_in() -> None:
    traj = _dummy_traj(sim_id=2)
    stats = fit_terminal_norm_stats_from_trajectories([traj])
    graph = assemble_terminal_graph(
        traj,
        stats=stats,
        runtime_y_normalizer=stats.cache_y_normalizer,
        use_sdf_features=False,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    pos_dim = int(graph.pos.shape[-1])
    fun_dim = int(graph.x.shape[-1]) - pos_dim
    assert fun_dim == 0
    assert pos_dim + fun_dim == 2

    cfg = GLTConfig(pe=NonePEConfig(), pe_inject_mode="concat_input")
    params = sum(p.numel() for p in GLT(cfg, metadata={"c_in": 2, "c_out": 2}).parameters())
    assert params > 0


def test_terminal_assembled_shapes_ux_only() -> None:
    from pdebench.dataset.plaid_elpl_terminal.constants import terminal_target_field_indices
    from pdebench.dataset.plaid_elpl_terminal.norm import slice_y_normalizer

    traj = _dummy_traj(sim_id=4)
    stats = fit_terminal_norm_stats_from_trajectories([traj])
    target_fields = ("U_x",)
    idx = terminal_target_field_indices(target_fields)
    y_norm = slice_y_normalizer(stats.cache_y_normalizer, idx)
    graph = assemble_terminal_graph(
        traj,
        stats=stats,
        runtime_y_normalizer=y_norm,
        use_sdf_features=False,
        graph_backend="pyg",
        target_fields=target_fields,
        bandwidth=1.0,
    )
    assert graph.y.shape == (3, 1)
    assert graph.output_fields_names == ["U_x"]
    assert graph.metadata["target_fields"] == ["U_x"]

    cfg = GLTConfig(pe=NonePEConfig(), pe_inject_mode="concat_input")
    params = sum(p.numel() for p in GLT(cfg, metadata={"c_in": 2, "c_out": 1}).parameters())
    assert params > 0


def test_terminal_assembled_shapes_sdf_on() -> None:
    traj = _dummy_traj(sim_id=3)
    stats = fit_terminal_norm_stats_from_trajectories([traj])
    graph = assemble_terminal_graph(
        traj,
        stats=stats,
        runtime_y_normalizer=stats.cache_y_normalizer,
        use_sdf_features=True,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    assert graph.x.shape == (3, 5)
