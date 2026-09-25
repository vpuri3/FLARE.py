"""Assemble PyG transition graphs from trajectory bundles.

Time conditioning lives only in graph-level ``input_scalars`` and the matching
``context`` alias. It is never concatenated into per-node ``x`` features.
Mesh GLT forward reads ``batch.context`` as the temporal context channel.
"""

from __future__ import annotations

from typing import Any

import torch

from pdebench.dataset.plaid_elpl_v3.laplacian import require_laplacian_bundle
from pdebench.dataset.plaid_elpl_v3.norm import ElPlNormStats, encode_node_features, encode_on_cpu
from pdebench.dataset.plaid_elpl_v3.schema import LaplacianBundle, TrajectoryBundle


def assemble_raw_transition(
    traj: TrajectoryBundle,
    *,
    step_idx: int,
    use_sdf_features: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step = int(step_idx)
    u0 = traj.u_traj[step]
    y = traj.u_traj[step + 1]
    t0 = traj.times[step].reshape(1)
    if use_sdf_features:
        geom = torch.cat([traj.sdf, traj.proj], dim=-1)
        x_raw = torch.cat([traj.pos, geom, u0], dim=-1)
    else:
        geom = torch.empty((traj.pos.shape[0], 0), dtype=traj.pos.dtype)
        x_raw = torch.cat([traj.pos, u0], dim=-1)
    return x_raw, y, t0, geom


def assemble_transition_graph(
    traj: TrajectoryBundle,
    *,
    step_idx: int,
    stats: ElPlNormStats,
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
        canonical_dataset="plaid_el_pl_dynamics",
        split_group="elpl_v3_shards",
        sample_id=traj.sim_id,
        laplacian_eig_dim=laplacian_eig_dim,
        laplacian_spec=laplacian_spec,
    )
    traj.ensure_edges(bandwidth=float(bandwidth))
    assert traj.edge_index is not None and traj.edge_attr is not None
    step = int(step_idx)
    geom = torch.cat([traj.sdf, traj.proj], dim=-1).cpu() if use_sdf_features else None
    u0 = traj.u_traj[step].cpu()
    y = traj.u_traj[step + 1].cpu()
    t0 = float(traj.times[step].item())
    x_enc = encode_node_features(
        pos=traj.pos.cpu(),
        geom=geom,
        u_field=u0,
        stats=stats,
        use_sdf_features=use_sdf_features,
    )
    y_enc = encode_on_cpu(stats.y_normalizer, y)
    # Graph-level temporal context (Sample.context); not part of node features.
    scalar_enc = encode_on_cpu(stats.input_scalar_normalizer, torch.tensor([[t0]], dtype=torch.float32))

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
        input_scalars=scalar_enc.reshape(1, -1).float(),
        input_scalars_names=("time",),
        output_fields=y_enc.float(),
        output_fields_names=list(target_fields),
        output_scalars_names=[],
        sample_id=int(traj.sim_id),
        metadata={
            "dataset": "plaid_el_pl_dynamics",
            "sample_index": int(traj.sim_id),
            "target_fields": list(target_fields),
            "target_scalar_fields": [],
            "scalar_names": ("time",),
            "scalar_values": [t0],
            "boundary_tags": list(traj.boundary_tags),
            "boundary_ids": traj.boundary_ids.tolist(),
            "raw_format": "plaid_elpl_v3_transition",
        },
    )
    graph.time = t0
    graph.context = graph.input_scalars
    graph.timestep_list = list(traj.timestep_list)
    if laplacian is not None:
        graph.laplacian_eigvals = laplacian.eigenvalues.float()
        graph.laplacian_eig = laplacian.eigenvectors.float()
    return graph
