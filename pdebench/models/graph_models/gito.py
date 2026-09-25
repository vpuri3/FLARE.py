"""GITO (graph-informed transformer operator) model.

No MeshGraphNet dependency: HGT and TNO blocks are owned here.

Structural dataflow (same-mesh queries only):
1. Encode nodes/edges with ``nn.Linear`` (in→hidden, edge→hidden).
2. HGT: GATv2 local + LinAttn global + fusion LinAttn (requires mesh edges when
   ``num_blocks_hgt > 0``).
3. TNO: each block applies cross-attn then self-attn —
   ``q ← q + LinAttn_cross(Norm(q), context=hgt_tokens)`` then
   ``q ← q + LinAttn_self(Norm(q))``, with ``context`` = post-HGT node states.
4. Decode with ``nn.Linear`` (hidden→out).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .utils import _make_activation, _make_norm, _norm_type_from_rmsnorm, graph_node_input

__all__ = [
    "GITOConfig",
    "GITOModel",
    "GITOLinearAttention",
    "GITOGATv2Layer",
    "GITOHGTBlock",
    "GITOTNOBlock",
]


@dataclass
class GITOConfig:
    """GITO hyperparameters.

    TNO blocks are cross-attention then self-attention on the same mesh tokens.
    Node/edge encoders and the decoder are single ``nn.Linear`` layers.
    """

    model: str = "gito"
    num_blocks_hgt: int = 2
    num_blocks_self_attn: int = 0
    channel_dim: int = 128
    num_heads: int = 8
    act: Optional[str] = None
    rmsnorm: bool = False


class GITOLinearAttention(nn.Module):
    """Linear-complexity attention used by GITO/GNOT-style operator blocks."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"GITO hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def _attend_one(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query(query).view(query.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        k = self.key(context).view(context.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        v = self.value(context).view(context.shape[0], self.num_heads, self.head_dim).transpose(0, 1)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)
        normalizer = 1.0 / context.shape[0]
        kv = k.transpose(-2, -1) @ v
        y = (q @ kv) * normalizer + q
        y = y.transpose(0, 1).reshape(query.shape[0], self.num_heads * self.head_dim)
        return self.proj(y).to(dtype=query.dtype)

    @torch.compiler.disable
    def _attend_batched(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_batch: torch.Tensor,
        context_batch: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query(query).view(query.shape[0], self.num_heads, self.head_dim)
        k = self.key(context).view(context.shape[0], self.num_heads, self.head_dim)
        v = self.value(context).view(context.shape[0], self.num_heads, self.head_dim)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        context_labels, context_inverse = torch.unique(context_batch, sorted=True, return_inverse=True)
        query_group = torch.searchsorted(context_labels, query_batch)
        if bool((query_group == context_labels.numel()).any()) or bool((context_labels[query_group] != query_batch).any()):
            raise ValueError("GITOLinearAttention query_batch contains labels missing from context_batch.")

        num_groups = context_labels.numel()
        kv = torch.zeros(
            num_groups,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            dtype=k.dtype,
            device=k.device,
        )
        kv.index_add_(0, context_inverse, k.unsqueeze(-1) * v.unsqueeze(-2))
        counts = torch.bincount(context_inverse, minlength=num_groups).to(dtype=q.dtype, device=q.device).clamp_min(1)

        y = torch.einsum("nhd,nhde->nhe", q, kv[query_group])
        y = y / counts[query_group].view(-1, 1, 1) + q
        y = y.reshape(query.shape[0], self.num_heads * self.head_dim)
        return self.proj(y).to(dtype=query.dtype)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        query_batch: torch.Tensor | None = None,
        context_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_is_query = context is None
        context = query if context is None else context
        if query_batch is None:
            return self._attend_one(query, context)
        if context_batch is None:
            if not context_is_query and context.shape[0] != query_batch.shape[0]:
                raise ValueError("GITOLinearAttention cross-attention with batched context requires context_batch.")
            context_batch = query_batch
        return self._attend_batched(query, context, query_batch, context_batch)


class GITOGATv2Layer(nn.Module):
    """Edge-aware local graph layer matching the paper's GATv2 HGT component."""

    def __init__(self, hidden_dim: int, num_heads: int, activation_fn: nn.Module):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"GITO hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.activation = activation_fn
        nn.init.xavier_uniform_(self.attn)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0].long(), edge_index[1].long()
        num_nodes = node_features.shape[0]
        src_h = self.src(node_features).view(num_nodes, self.num_heads, self.head_dim)
        dst_h = self.dst(node_features).view(num_nodes, self.num_heads, self.head_dim)
        edge_h = self.edge(edge_features).view(edge_features.shape[0], self.num_heads, self.head_dim)
        value_h = self.value(node_features).view(num_nodes, self.num_heads, self.head_dim)

        score_h = self.activation(src_h[src] + dst_h[dst] + edge_h)
        score = (score_h * self.attn.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.head_dim)
        max_score = torch.full((num_nodes, self.num_heads), -torch.inf, dtype=score.dtype, device=score.device)
        max_score.scatter_reduce_(0, dst[:, None].expand(-1, self.num_heads), score, reduce="amax", include_self=True)
        weight = torch.exp(score - max_score[dst])
        denom = torch.zeros((num_nodes, self.num_heads), dtype=score.dtype, device=score.device)
        denom.index_add_(0, dst, weight)
        weight = weight / denom[dst].clamp_min(1e-12)

        messages = weight.unsqueeze(-1) * (value_h[src] + edge_h)
        aggregated = torch.zeros((num_nodes, self.num_heads, self.head_dim), dtype=messages.dtype, device=messages.device)
        aggregated.index_add_(0, dst, messages)
        return self.out(aggregated.reshape(num_nodes, -1)).to(dtype=node_features.dtype)


class GITOHGTBlock(nn.Module):
    """Hybrid graph transformer: GATv2 local + LinAttn global + fusion LinAttn."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        activation_fn: nn.Module,
        norm_type: str,
    ):
        super().__init__()
        self.local_norm = _make_norm(norm_type, hidden_dim)
        self.local = GITOGATv2Layer(hidden_dim, num_heads, activation_fn)
        self.global_norm = _make_norm(norm_type, hidden_dim)
        self.global_attn = GITOLinearAttention(hidden_dim, num_heads)
        self.fusion_in = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_norm = _make_norm(norm_type, hidden_dim)
        self.fusion_attn = GITOLinearAttention(hidden_dim, num_heads)
        self.out_norm = _make_norm(norm_type, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor | None,
    ) -> torch.Tensor:
        local = self.local(self.local_norm(x), edge_features, edge_index)
        global_ = self.global_attn(self.global_norm(x), query_batch=batch_index)
        fused = self.fusion_in(torch.cat([local, global_], dim=-1))
        return self.out_norm(x + self.fusion_attn(self.fusion_norm(fused), query_batch=batch_index))


class GITOTNOBlock(nn.Module):
    """Transformer neural operator block: cross-attention then self-attention.

    Cross-attends queries to fixed post-HGT context tokens on the same mesh, then
    applies residual self-attention on the updated queries.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        norm_type: str,
    ):
        super().__init__()
        self.cross_norm = _make_norm(norm_type, hidden_dim)
        self.cross_attn = GITOLinearAttention(hidden_dim, num_heads)
        self.self_norm = _make_norm(norm_type, hidden_dim)
        self.self_attn = GITOLinearAttention(hidden_dim, num_heads)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_batch: torch.Tensor | None = None,
        context_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = query + self.cross_attn(
            self.cross_norm(query),
            context=context,
            query_batch=query_batch,
            context_batch=context_batch,
        )
        return q + self.self_attn(self.self_norm(q), query_batch=query_batch)


class GITOModel(nn.Module):
    """Graph-informed transformer operator (no MeshGraphNet dependency).

    Structural dataflow (same-mesh queries only):
    1. Encode nodes/edges with ``nn.Linear`` (in→hidden, edge→hidden).
    2. HGT: GATv2 local + LinAttn global + fusion LinAttn (mesh edges required when
       ``num_blocks_hgt > 0``).
    3. TNO: each block does
       ``q ← q + LinAttn_cross(Norm(q), context=hgt_tokens)`` then
       ``q ← q + LinAttn_self(Norm(q))``, with ``context`` = post-HGT node states.
    4. Decode with ``nn.Linear`` (hidden→out).
    """

    def __init__(self, config: GITOConfig, metadata=None):
        super().__init__()
        metadata = {} if metadata is None else dict(metadata)
        in_dim = int(metadata.get("c_in", metadata.get("point_input_dim", 1)))
        out_dim = int(metadata.get("c_out", 1))
        hidden_dim = int(config.channel_dim)
        num_blocks_hgt = int(config.num_blocks_hgt)
        num_blocks_self_attn = int(config.num_blocks_self_attn)
        num_heads = int(config.num_heads)
        if num_blocks_hgt < 0:
            raise ValueError("GITOConfig.num_blocks_hgt must be >= 0.")
        if num_blocks_self_attn < 0:
            raise ValueError("GITOConfig.num_blocks_self_attn must be >= 0.")
        edge_dim = int(metadata.get("c_edge", 1))
        activation = "silu" if config.act is None else config.act
        rmsnorm = bool(config.rmsnorm)
        activation_fn = _make_activation(activation)
        norm_type = _norm_type_from_rmsnorm(rmsnorm)
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.hgt_blocks = nn.ModuleList(
            [
                GITOHGTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                )
                for _ in range(num_blocks_hgt)
            ]
        )
        self.tno_blocks = nn.ModuleList(
            [
                GITOTNOBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    norm_type=norm_type,
                )
                for _ in range(num_blocks_self_attn)
            ]
        )
        self.node_decoder = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        data=None,
        *,
        pos=None,
        feats=None,
        edge_index=None,
        edge_attr=None,
        batch_index=None,
        use_flash_varlen: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ):
        del kwargs

        needs_edges = len(self.hgt_blocks) > 0
        if pos is not None:
            node_input = graph_node_input(pos, feats)
            if needs_edges and edge_index is None:
                raise ValueError("Tensor GITO forward requires edge_index.")
            if needs_edges and edge_attr is None:
                raise ValueError("Tensor GITO forward requires edge_attr.")
        elif hasattr(data, "edge_index"):
            node_input = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch_index = data.batch if hasattr(data, "batch") else batch_index
            if needs_edges and edge_attr is None:
                raise ValueError("PyG mesh graph is missing edge_attr required for GITO.")
        else:
            raise ValueError(f"Unsupported graph data type for GITOModel: {type(data)}")

        if needs_edges and edge_index.numel() == 0:
            raise ValueError("GITO requires non-empty edge connectivity.")

        device = next(self.parameters()).device
        tensors = [("node features", node_input)]
        if needs_edges:
            tensors.extend([("edge_index", edge_index), ("edge features", edge_attr)])
        for name, tensor in tensors:
            if tensor.device != device:
                raise ValueError(f"GITOModel expected {name} on {device}, found {tensor.device}.")
        if batch_index is not None and batch_index.device != device:
            raise ValueError(f"GITOModel expected batch_index on {device}, found {batch_index.device}.")

        query = self.node_encoder(node_input)
        edge_features = self.edge_encoder(edge_attr) if needs_edges else None
        for block in self.hgt_blocks:
            query = block(
                query,
                edge_features,
                edge_index,
                batch_index=batch_index,
            )
        context = query
        for block in self.tno_blocks:
            query = block(
                query,
                context=context,
                query_batch=batch_index,
                context_batch=batch_index,
            )
        return self.node_decoder(query)
