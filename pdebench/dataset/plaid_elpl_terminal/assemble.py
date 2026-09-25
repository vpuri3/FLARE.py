"""Assemble PyG terminal graphs: mesh (+ optional SDF) -> u(T)."""

from __future__ import annotations

from typing import Any

import torch

from pdebench.dataset.plaid_elpl_terminal.constants import (
    FINAL_STEP_IDX,
    FINAL_TIME,
    NUM_FIELD_SNAPSHOTS,
    terminal_target_field_indices,
)
from pdebench.dataset.plaid_elpl_terminal.norm import (
    ElPlTerminalNormStats,
    YFieldNormalizer,
    encode_static_node_features,
    encode_y_on_cpu,
    slice_y_normalizer,
)
from pdebench.dataset.plaid_elpl_v3.laplacian import require_laplacian_bundle
from pdebench.dataset.plaid_elpl_v3.schema import LaplacianBundle, TrajectoryBundle


def assemble_terminal_graph(
    traj: TrajectoryBundle,
    *,
    stats: ElPlTerminalNormStats,
    runtime_y_normalizer: YFieldNormalizer,
    use_sdf_features: bool,
    graph_backend: str,
    target_fields: tuple[str, ...],
    laplacian: LaplacianBundle | None = None,
    bandwidth: float,
    laplacian_eig_dim: int = 0,
    laplacian_spec: str = "graph",
) -> Any:
    laplacian = require_laplacian_bundle(
        laplacian,
        canonical_dataset="plaid_elpl_terminal",
        split_group="elpl_terminal_shards",
        sample_id=traj.sim_id,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    if int(traj.u_traj.shape[0]) != int(NUM_FIELD_SNAPSHOTS):
        raise ValueError(
            f"Expected u_traj with {NUM_FIELD_SNAPSHOTS} snapshots for sim {traj.sim_id}, "
            f"got {int(traj.u_traj.shape[0])}."
        )
    traj.ensure_edges(bandwidth=float(bandwidth))
    assert traj.edge_index is not None and traj.edge_attr is not None

    geom = torch.cat([traj.sdf, traj.proj], dim=-1).cpu() if use_sdf_features else None
    field_idx = terminal_target_field_indices(target_fields)
    y_final = traj.u_traj[int(FINAL_STEP_IDX), :, list(field_idx)].cpu()
    cache_y_normalizer = slice_y_normalizer(stats.cache_y_normalizer, field_idx)
    x_enc = encode_static_node_features(
        pos=traj.pos.cpu(),
        geom=geom,
        stats=stats,
        use_sdf_features=use_sdf_features,
    )
    y_enc = encode_y_on_cpu(
        raw_y=y_final,
        cache_y_normalizer=cache_y_normalizer,
        runtime_y_normalizer=runtime_y_normalizer,
    )

    if graph_backend != "pyg":
        raise ValueError(f"Unsupported graph backend '{graph_backend}'.")

    import torch_geometric as pyg

    graph = pyg.data.Data(
        pos=traj.pos.cpu().float(),
        x=x_enc.float(),
        y=y_enc.float(),
        edge_index=traj.edge_index.long().cpu(),
        edge_attr=traj.edge_attr.float().cpu(),
        cells=traj.cells.long().cpu(),
        output_fields=y_enc.float(),
        output_fields_names=list(target_fields),
        output_scalars_names=[],
        sample_id=int(traj.sim_id),
        metadata={
            "dataset": "plaid_elpl_terminal",
            "sample_index": int(traj.sim_id),
            "target_fields": list(target_fields),
            "target_scalar_fields": [],
            "scalar_names": (),
            "scalar_values": [],
            "boundary_tags": list(traj.boundary_tags),
            "boundary_ids": traj.boundary_ids.tolist(),
            "raw_format": "plaid_elpl_terminal",
            "final_time": float(FINAL_TIME),
            "final_step_idx": int(FINAL_STEP_IDX),
        },
    )
    graph.timestep_list = list(traj.timestep_list)
    if laplacian is not None:
        graph.laplacian_eigvals = laplacian.eigenvalues.float()
        graph.laplacian_eig = laplacian.eigenvectors.float()
    return graph
