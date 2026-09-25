"""MeshGraphNet model and message-passing blocks over flat graph tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .utils import _make_activation, _make_norm, _norm_type_from_rmsnorm, graph_node_input

__all__ = [
    "MeshEdgeBlock",
    "MeshGraphMLP",
    "MeshGraphNetConfig",
    "MeshGraphNetModel",
    "MeshGraphNetProcessor",
    "MeshGraphSplitInputMLP",
    "MeshNodeBlock",
]


class MeshGraphMLP(nn.Module):
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


class MeshGraphSplitInputMLP(nn.Module):
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
            raise ValueError("MeshGraphSplitInputMLP requires hidden_layers >= 1.")
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


class MeshEdgeBlock(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module,
        norm_type: str,
    ):
        super().__init__()
        self.edge_mlp = MeshGraphSplitInputMLP(
            input_dims=(edge_dim, node_dim, node_dim),
            output_dim=edge_dim,
            hidden_dim=edge_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(self, edge_features, node_features, src, dst):
        return edge_features + self.edge_mlp(edge_features, node_features[src], node_features[dst])


class MeshNodeBlock(nn.Module):
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
            raise ValueError("MeshNodeBlock currently supports only sum aggregation.")
        self.node_mlp = MeshGraphSplitInputMLP(
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


class MeshGraphNetProcessor(nn.Module):
    """Encode–process core of MeshGraphNets (Interaction Network style).

    On a directed graph :math:`G=(V,E)` with node states :math:`h_i\\in\\mathbb{R}^{d_n}`
    and edge states :math:`e_{ij}\\in\\mathbb{R}^{d_e}`, each processor layer applies:

    1. **Edge update** (residual)::

           e'_{ij} = e_{ij} + \\mathrm{MLP}_e\\bigl(e_{ij},\\, h_i,\\, h_j\\bigr)

       where :math:`i=\\mathrm{src}`, :math:`j=\\mathrm{dst}` in ``edge_index``.

    2. **Node update** with sum aggregation (residual)::

           \\bar e_j = \\sum_{i:(i\\to j)\\in E} e'_{ij},\\qquad
           h'_j = h_j + \\mathrm{MLP}_n\\bigl(h_j,\\, \\bar e_j\\bigr).

    Stacking ``processor_size`` such (edge, node) pairs expands the receptive field
    along mesh connectivity. This is the Sanchez-Gonzalez / Battaglia Graph Network
    processor used by MeshGraphNets; other models may reuse it as a *generic* MP block,
    but that reuse is an implementation choice, not a requirement of those papers.
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
                MeshEdgeBlock(
                    node_dim=input_dim_node,
                    edge_dim=input_dim_edge,
                    hidden_layers=num_layers_edge,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                )
            )
            self.processor_layers.append(
                MeshNodeBlock(
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
class MeshGraphNetConfig:
    model: str = "meshgraphnet"
    num_blocks: int = 4
    channel_dim: int = 128
    num_layers_node: int = 2
    num_layers_edge: int = 2
    act: Optional[str] = None
    rmsnorm: bool = False


class MeshGraphNetModel(nn.Module):
    """MeshGraphNets surrogate on a fixed mesh (encode–process–decode).

    Learns a map from nodal fields on a simulation mesh to nodal outputs by message
    passing on the *same* mesh graph. With node inputs
    :math:`x_i=\\mathrm{concat}(\\mathrm{pos}_i,\\mathrm{feats}_i)` and edge attributes
    :math:`a_{ij}` (typically relative geometry),

    .. math::

        h_i^{(0)} = \\mathrm{MLP}_n(x_i),\\qquad
        e_{ij}^{(0)} = \\mathrm{MLP}_e(a_{ij}),

        h^{(L)} = \\mathrm{Proc}^{(L)}\\bigl(h^{(0)}, e^{(0)}, E\\bigr),\\qquad
        y_i = \\mathrm{MLP}_{\\mathrm{dec}}\\bigl(h_i^{(L)}\\bigr),

    where :math:`\\mathrm{Proc}` is :class:`MeshGraphNetProcessor`. No latent coarsening:
    all communication is through the provided ``edge_index``.

    Inputs
    ------
    Flat tensors ``pos`` / ``feats`` / ``edge_index`` / ``edge_attr``, or a PyG-like
    batch with ``.x``, ``.edge_index``, ``.edge_attr``.
    """

    def __init__(self, config: MeshGraphNetConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        in_dim = int(metadata.get("c_in", metadata.get("point_input_dim", 1)))
        out_dim = int(metadata.get("c_out", 1))
        hidden_dim = int(config.channel_dim)
        num_layers = int(config.num_blocks)
        num_layers_node = int(config.num_layers_node)
        num_layers_edge = int(config.num_layers_edge)
        edge_dim = int(metadata.get("c_edge", 1))
        activation = "silu" if config.act is None else config.act
        rmsnorm = bool(config.rmsnorm)
        activation_fn = _make_activation(activation)
        norm_type = _norm_type_from_rmsnorm(rmsnorm)
        self.edge_encoder = MeshGraphMLP(
            edge_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.node_encoder = MeshGraphMLP(
            in_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=num_layers,
            input_dim_node=hidden_dim,
            input_dim_edge=hidden_dim,
            num_layers_node=num_layers_node,
            num_layers_edge=num_layers_edge,
            aggregation="sum",
            norm_type=norm_type,
            activation_fn=activation_fn,
        )
        self.node_decoder = MeshGraphMLP(
            hidden_dim,
            output_dim=out_dim,
            hidden_dim=hidden_dim,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type=None,
        )

    def forward(self, data=None, *, pos=None, feats=None, edge_index=None, edge_attr=None, **kwargs):
        del kwargs

        if pos is not None:
            node_features = graph_node_input(pos, feats)
            if edge_index is None:
                raise ValueError("MeshGraphNet forward requires edge_index.")
            if edge_attr is None:
                raise ValueError("MeshGraphNet forward requires edge_attr.")
            edge_features = edge_attr
        elif hasattr(data, "edge_index"):
            node_features = data.x
            edge_index = data.edge_index
            edge_features = data.edge_attr
            if edge_features is None:
                raise ValueError("Graph batch is missing edge_attr required for MeshGraphNet.")
        else:
            raise ValueError(f"Unsupported graph data type for MeshGraphNetModel: {type(data)}")

        if edge_index.numel() == 0:
            raise ValueError("MeshGraphNet requires non-empty edge connectivity.")

        device = next(self.parameters()).device
        if node_features.device != device:
            raise ValueError(
                f"MeshGraphNetModel expected node features on {device}, found {node_features.device}. "
                "Move the batch to the model device before calling forward."
            )
        if edge_features.device != device:
            raise ValueError(
                f"MeshGraphNetModel expected edge features on {device}, found {edge_features.device}. "
                "Move the batch to the model device before calling forward."
            )
        if edge_index.device != device:
            raise ValueError(
                f"MeshGraphNetModel expected edge_index on {device}, found {edge_index.device}. "
                "Move the batch to the model device before calling forward."
            )

        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        node_features = self.processor(node_features, edge_features, edge_index)
        return self.node_decoder(node_features)
