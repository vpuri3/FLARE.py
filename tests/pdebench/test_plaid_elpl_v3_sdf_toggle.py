from __future__ import annotations

from pdebench.dataset.plaid_elpl_v3.assemble import assemble_transition_graph
from pdebench.dataset.plaid_elpl_v3.norm import fit_norm_stats_from_trajectories
from pdebench.models.graph_models.glt import GLT, GLTConfig
from pdebench.models.graph_models.glt import NonePEConfig
from tests.pdebench.test_plaid_elpl_v3_schema import _dummy_traj


def test_sdf_toggle_drops_geom_channels_and_c_in() -> None:
    traj = _dummy_traj(sim_id=2)
    stats = fit_norm_stats_from_trajectories([traj], transitions_per_sim=40)
    on = assemble_transition_graph(
        traj,
        step_idx=0,
        stats=stats,
        use_sdf_features=True,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    off = assemble_transition_graph(
        traj,
        step_idx=0,
        stats=stats,
        use_sdf_features=False,
        graph_backend="pyg",
        target_fields=("U_x", "U_y"),
        bandwidth=1.0,
    )
    assert on.x.shape == (3, 7)
    assert off.x.shape == (3, 4)

    pos_dim = int(on.pos.shape[-1])
    on_fun_dim = int(on.x.shape[-1]) - pos_dim + int(on.input_scalars.shape[-1])
    off_fun_dim = int(off.x.shape[-1]) - pos_dim + int(off.input_scalars.shape[-1])
    assert on_fun_dim == 6
    assert off_fun_dim == 3
    assert pos_dim + on_fun_dim == 8
    assert pos_dim + off_fun_dim == 5

    glt_meta_on = {"c_in": 8, "c_out": 2}
    glt_meta_off = {"c_in": 5, "c_out": 2}
    cfg = GLTConfig(pe=NonePEConfig(), pe_inject_mode="concat_input")
    on_params = sum(p.numel() for p in GLT(cfg, metadata=glt_meta_on).parameters())
    off_params = sum(p.numel() for p in GLT(cfg, metadata=glt_meta_off).parameters())
    assert off_params < on_params
