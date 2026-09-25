"""RIGNO (Region Interaction Graph Neural Operator) over flat graph tensors.

No MeshGraphNet dependency: message-passing blocks and regional graphs are owned here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .neighbors import knn_graph
from .utils import (
    _make_activation,
    _make_norm,
    _norm_type_from_rmsnorm,
    _region_grid,
    graph_node_input,
)

__all__ = [
    "RIGNOConfig",
    "RIGNOModel",
]


class RIGNOMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        norm_type: str | None,
    ):
        super().__init__()
        if hidden_layers < 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
            for _ in range(hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn])
            layers.append(nn.Linear(hidden_dim, output_dim))

        if norm_type is None:
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Sequential(*layers, _make_norm(norm_type, output_dim))

    def forward(self, x):
        return self.net(x)


class RIGNOSplitMLP(nn.Module):
    def __init__(
        self,
        input_dims: tuple[int, ...],
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        norm_type: str | None,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("RIGNOSplitMLP requires hidden_layers >= 1.")
        self.input_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for input_dim in input_dims])
        layers = [activation_fn]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn])
        layers.append(nn.Linear(hidden_dim, output_dim))
        if norm_type is not None:
            layers.append(_make_norm(norm_type, output_dim))
        self.post = nn.Sequential(*layers)

    def forward(self, *inputs):
        x = self.input_layers[0](inputs[0])
        for layer, input_tensor in zip(self.input_layers[1:], inputs[1:]):
            x = x + layer(input_tensor)
        return self.post(x)


class RIGNOEdgeBlock(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        norm_type: str,
    ):
        super().__init__()
        self.edge_mlp = RIGNOSplitMLP(
            input_dims=(edge_dim, node_dim, node_dim),
            output_dim=edge_dim,
            hidden_dim=edge_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, edge_features, node_features, src, dst):
        return edge_features + self.edge_mlp(edge_features, node_features[src], node_features[dst])


class RIGNONodeBlock(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_layers: int,
        aggregation: str,
        activation_fn: nn.Module,
        norm_type: str,
    ):
        super().__init__()
        if aggregation != "sum":
            raise ValueError("RIGNONodeBlock currently supports only sum aggregation.")
        self.node_mlp = RIGNOSplitMLP(
            input_dims=(node_dim, edge_dim),
            output_dim=node_dim,
            hidden_dim=edge_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, edge_features, node_features, dst):
        aggregated = torch.zeros(
            node_features.shape[0],
            edge_features.shape[-1],
            dtype=edge_features.dtype,
            device=edge_features.device,
        )
        aggregated.index_add_(0, dst, edge_features)
        return node_features + self.node_mlp(node_features, aggregated)


class RIGNOProcessor(nn.Module):
    """Residual GraphNet edge/node processor owned by RIGNO.

    On a directed graph :math:`G=(V,E)` with node states :math:`h_i` and edge
    states :math:`e_{ij}`, each layer applies:

    1. Edge update (residual)::

           e'_{ij} = e_{ij} + MLP_e(e_{ij}, h_i, h_j)

    2. Node update with sum aggregation (residual)::

           \\bar e_j = \\sum_{i:(i\\to j)\\in E} e'_{ij},\\qquad
           h'_j = h_j + MLP_n(h_j, \\bar e_j).
    """

    def __init__(
        self,
        processor_size: int,
        input_dim_node: int,
        input_dim_edge: int,
        num_layers_node: int,
        num_layers_edge: int,
        aggregation: str,
        norm_type: str,
        activation_fn: nn.Module,
    ):
        super().__init__()
        self.processor_layers = nn.ModuleList()
        for _ in range(processor_size):
            self.processor_layers.append(
                RIGNOEdgeBlock(
                    node_dim=input_dim_node,
                    edge_dim=input_dim_edge,
                    hidden_layers=num_layers_edge,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                )
            )
            self.processor_layers.append(
                RIGNONodeBlock(
                    node_dim=input_dim_node,
                    edge_dim=input_dim_edge,
                    hidden_layers=num_layers_node,
                    aggregation=aggregation,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                )
            )

    def forward(self, node_features, edge_features, edge_index):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        for i in range(0, len(self.processor_layers), 2):
            edge_features = self.processor_layers[i](edge_features, node_features, src, dst)
            node_features = self.processor_layers[i + 1](edge_features, node_features, dst)
        return node_features


@dataclass
class RIGNOConfig:
    model: str = "rigno"
    num_blocks: int = 4
    channel_dim: int = 128
    num_layers_node: int = 2
    num_layers_edge: int = 2
    act: Optional[str] = None
    rmsnorm: bool = False
    num_slices: int = 64
    region_knn: int = 8


class RIGNOModel(nn.Module):
    """Region Interaction Graph Neural Operator on flat PDEBench tensors.

    Dataflow (structural fidelity to RIGNO; single-scale knn regional graph):

    1. Encode nodes: ``h = MLP(concat(pos, feats))``.
    2. Assign a uniform 2D regional grid (``num_slices`` per axis); regional
       centers are the mean of assigned coordinates (empty cells filled from
       the regular grid).
    3. **p2r** bipartite physical→region (by assignment); message
       ``MLP(h_i, δ_i)`` aggregated onto regions.
    4. **Regional graph** from region centers via knn (``region_knn``, default 8)
       — single scale (not official multi-scale Delaunay).
    5. **Processor:** RIGNO-owned residual edge/node MP (GraphNet equations;
       local classes, not imported from MeshGraphNet modules).
    6. **r2p** bipartite region→physical + decode MLP.

    Forward takes ``pos`` / ``feats`` / optional ``batch_index``; dataset
    ``edge_index`` / ``edge_attr`` are not required. There is no MeshGraphNet
    dependency.
    """

    def __init__(self, config: RIGNOConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        in_dim = int(metadata.get("c_in", metadata.get("point_input_dim", 1)))
        out_dim = int(metadata.get("c_out", 1))
        hidden_dim = int(config.channel_dim)
        num_layers = int(config.num_blocks)
        num_layers_node = int(config.num_layers_node)
        num_layers_edge = int(config.num_layers_edge)
        num_regions_per_axis = int(config.num_slices)
        activation = "silu" if config.act is None else config.act
        rmsnorm = bool(config.rmsnorm)
        activation_fn = _make_activation(activation)
        norm_type = _norm_type_from_rmsnorm(rmsnorm)
        self.num_regions_per_axis = int(num_regions_per_axis)
        self.region_knn = int(config.region_knn)
        self.node_encoder = RIGNOMLP(
            in_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.p2r_mlp = RIGNOSplitMLP(
            input_dims=(hidden_dim, 2),
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        # Relative geom edge features: (dx, dy, length) / scale → dim 3.
        self.region_edge_encoder = RIGNOMLP(
            3,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.processor = RIGNOProcessor(
            processor_size=num_layers,
            input_dim_node=hidden_dim,
            input_dim_edge=hidden_dim,
            num_layers_node=num_layers_node,
            num_layers_edge=num_layers_edge,
            aggregation="sum",
            norm_type=norm_type,
            activation_fn=activation_fn,
        )
        self.r2p_mlp = RIGNOSplitMLP(
            input_dims=(hidden_dim, hidden_dim, 2),
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.node_decoder = RIGNOMLP(
            hidden_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type=None,
        )

    def _assign_regions(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        side = self.num_regions_per_axis
        coord = pos[:, :2]
        mins = coord.min(dim=0).values
        maxs = coord.max(dim=0).values
        span = (maxs - mins).clamp_min(1e-6)
        ij = torch.floor((coord - mins) / span * side).long().clamp_(0, side - 1)
        assign = ij[:, 0] * side + ij[:, 1]

        num_regions = side * side
        region_pos = torch.zeros(num_regions, 2, dtype=coord.dtype, device=coord.device)
        counts = torch.zeros(num_regions, 1, dtype=coord.dtype, device=coord.device)
        region_pos.index_add_(0, assign, coord)
        counts.index_add_(0, assign, torch.ones(assign.shape[0], 1, dtype=coord.dtype, device=coord.device))
        region_pos = region_pos / counts.clamp_min(1.0)

        empty = counts.squeeze(-1) == 0
        if bool(empty.any()):
            region_pos[empty] = _region_grid(coord, side)[empty]
        return assign, region_pos

    def _region_geom_edge_attr(self, region_pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        diff = region_pos[dst] - region_pos[src]
        length = diff.norm(dim=-1, keepdim=True)
        scale = length[length > 0].mean().clamp_min(1e-6) if bool((length > 0).any()) else length.new_tensor(1.0)
        return torch.cat([diff / scale, length / scale], dim=-1)

    def _forward_one(self, pos: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        assign, region_pos = self._assign_regions(pos)

        region_input = torch.zeros(
            region_pos.shape[0], node_features.shape[-1], dtype=node_features.dtype, device=node_features.device
        )
        counts = torch.zeros(region_pos.shape[0], 1, dtype=node_features.dtype, device=node_features.device)
        rel = (pos[:, :2] - region_pos[assign]).to(dtype=node_features.dtype)
        p2r_msg = self.p2r_mlp(node_features, rel)
        region_input.index_add_(0, assign, p2r_msg)
        counts.index_add_(0, assign, torch.ones(assign.shape[0], 1, dtype=node_features.dtype, device=node_features.device))
        region_features = region_input / counts.clamp_min(1.0)

        region_edge_index = knn_graph(region_pos, self.region_knn)
        if region_edge_index.numel() == 0:
            edge_features = node_features.new_zeros((0, node_features.shape[-1]))
        else:
            geom_attr = self._region_geom_edge_attr(region_pos, region_edge_index).to(dtype=node_features.dtype)
            edge_features = self.region_edge_encoder(geom_attr)
        region_features = self.processor(region_features, edge_features, region_edge_index)

        region_at_nodes = region_features[assign]
        decoded = self.r2p_mlp(node_features, region_at_nodes, rel)
        return self.node_decoder(decoded)

    def forward(
        self,
        data=None,
        *,
        pos=None,
        feats=None,
        edge_index=None,
        edge_attr=None,
        batch_index=None,
        **kwargs,
    ):
        del kwargs, edge_index, edge_attr

        if pos is not None:
            if pos.ndim != 2 or pos.shape[-1] < 2:
                raise ValueError(f"RIGNO requires pos [N, >=2], got shape {tuple(pos.shape)}.")
            node_input = graph_node_input(pos, feats)
            spatial = pos
        elif data is not None and hasattr(data, "pos") and data.pos is not None:
            spatial = data.pos
            if spatial.ndim != 2 or spatial.shape[-1] < 2:
                raise ValueError(f"RIGNO requires pos [N, >=2], got shape {tuple(spatial.shape)}.")
            node_feats = getattr(data, "x", None)
            # If data.x already includes coordinates, prefer it as encoded input width.
            if node_feats is not None and node_feats.shape[-1] > spatial.shape[-1]:
                node_input = node_feats
            else:
                node_input = graph_node_input(spatial, node_feats)
            batch_index = data.batch if hasattr(data, "batch") else batch_index
        else:
            raise ValueError(f"Unsupported graph data type for RIGNOModel: {type(data)}")

        device = next(self.parameters()).device
        for name, tensor in (("node features", node_input), ("pos", spatial)):
            if tensor.device != device:
                raise ValueError(
                    f"RIGNOModel expected {name} on {device}, found {tensor.device}. "
                    "Move the batch to the model device before calling forward."
                )
        if batch_index is not None and batch_index.device != device:
            raise ValueError(
                f"RIGNOModel expected batch_index on {device}, found {batch_index.device}. "
                "Move the batch to the model device before calling forward."
            )

        node_features = self.node_encoder(node_input)
        if batch_index is None:
            return self._forward_one(spatial, node_features)

        outputs = None
        for graph_idx in torch.unique(batch_index, sorted=True):
            idx = torch.nonzero(batch_index == graph_idx, as_tuple=False).flatten()
            y = self._forward_one(spatial[idx], node_features[idx])
            if outputs is None:
                outputs = torch.empty(node_input.shape[0], y.shape[-1], dtype=y.dtype, device=y.device)
            outputs[idx] = y
        return outputs
