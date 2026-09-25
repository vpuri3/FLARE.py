"""Exact truncated-BFS hop neighborhoods for GLT positional encodings.

Neighborhood construction is a fixed-depth (``MAX_HOP``) vectorized multi-source
BFS over packed graphs. There is no per-node Python BFS. The only structural
Python loops are the constant hop iteration ``for hop in 1..MAX_HOP`` and the
fixed 3-scale encode loop.

Compile boundaries
------------------
* The entire ``MultiscaleHopPE`` forward path is ``@torch.compiler.disable``.
  Pair lists are ragged (millions of edges on bumper), and letting Dynamo /
  Inductor see ``searchsorted`` / ``scatter`` over them causes ``SliceView``
  failures or catastrophic graph breaks that slow the surrounding GLT trunk.
* Keep the PE eager; let ``torch.compile`` specialize the transformer blocks
  on stable ``[N, C]`` activations only.
* Neighborhood cache keys are **content** fingerprints (not ``data_ptr``), so
  static meshes still hit when the dataloader reallocates ``edge_index``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import LongTensor, Tensor, nn

try:
    from torch_geometric.utils import scatter
except ImportError:  # PyG is optional; only MultiscaleHopPE needs it.
    scatter = None

from pdebench.dataset.sample import FeatureRequest

from .pe_base import GraphPE, TopologyFeatures, _packed_node_count, _resolve_packed_indices

DEFAULT_HOP_SCALES: tuple[tuple[int, int], ...] = ((1, 2), (3, 6), (7, 10))
MAX_HOP: int = 10

__all__ = [
    "DEFAULT_HOP_SCALES",
    "MAX_HOP",
    "MultiscaleHopPE",
    "MultiscaleHopPEConfig",
    "ScaleNeighborhoods",
    "build_multiscale_hop_neighborhoods",
]


@dataclass
class ScaleNeighborhoods:
    center_index: LongTensor
    neighbor_index: LongTensor
    hop_distance: LongTensor


@dataclass
class MultiscaleHopPEConfig:
    """Configuration for the multi-scale hop-neighborhood positional encoding."""

    kind: Literal["multiscale_hop_pe"] = "multiscale_hop_pe"
    pointnet_hidden_dim: int = 32
    normalize_by_mean_edge_length: bool = True
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        hidden_dim = int(self.pointnet_hidden_dim)
        if hidden_dim != 32:
            raise ValueError(f"pointnet_hidden_dim must be 32 for v1; got {hidden_dim}.")
        if float(self.rms_norm_eps) <= 0:
            raise ValueError(f"rms_norm_eps must be > 0; got {self.rms_norm_eps}.")

    @property
    def out_dim(self) -> int:
        return len(DEFAULT_HOP_SCALES) * 2 * int(self.pointnet_hidden_dim)

    def to_feature_request(self) -> FeatureRequest:
        return FeatureRequest(edges=True, laplacian_k=0)


def _pair_keys(centers: Tensor, nodes: Tensor, num_nodes: int) -> Tensor:
    return centers * int(num_nodes) + nodes


def _dedupe_by_key(centers: Tensor, nodes: Tensor, keys: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the first occurrence of each key (via argsort)."""
    if keys.numel() == 0:
        return centers, nodes, keys
    keys_sorted, order = torch.sort(keys)
    keep = torch.ones(keys_sorted.shape[0], dtype=torch.bool, device=keys.device)
    keep[1:] = keys_sorted[1:] != keys_sorted[:-1]
    order = order[keep]
    return centers[order], nodes[order], keys_sorted[keep]


def _filter_new_pairs(
    centers: Tensor,
    nodes: Tensor,
    visited_sorted: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Drop duplicates and pairs already present in sorted ``visited_sorted`` keys."""
    if centers.numel() == 0:
        empty = centers.new_empty(0)
        return empty, empty, empty
    keys = _pair_keys(centers, nodes, num_nodes)
    centers, nodes, keys = _dedupe_by_key(centers, nodes, keys)
    if visited_sorted.numel() == 0:
        return centers, nodes, keys
    pos = torch.searchsorted(visited_sorted, keys)
    in_range = pos < visited_sorted.numel()
    matched = torch.zeros_like(keys, dtype=torch.bool)
    safe_pos = torch.minimum(pos, torch.full_like(pos, visited_sorted.numel() - 1))
    matched[in_range] = visited_sorted[safe_pos[in_range]] == keys[in_range]
    keep = ~matched
    return centers[keep], nodes[keep], keys[keep]


def _merge_sorted_keys(visited_sorted: Tensor, new_keys: Tensor) -> Tensor:
    if new_keys.numel() == 0:
        return visited_sorted
    if visited_sorted.numel() == 0:
        return torch.sort(new_keys).values
    return torch.sort(torch.cat([visited_sorted, new_keys], dim=0)).values


def _csr_from_directed_edges(src: Tensor, dst: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Build CSR where ``edge_index`` is already the directed adjacency list."""
    order = torch.argsort(src)
    src_s = src[order]
    col = dst[order]
    deg = torch.bincount(src_s, minlength=num_nodes)
    rowptr = torch.empty(num_nodes + 1, dtype=torch.long, device=src.device)
    rowptr[0] = 0
    torch.cumsum(deg, dim=0, out=rowptr[1:])
    return rowptr, col


def _expand_frontier(
    centers: Tensor,
    nodes: Tensor,
    rowptr: Tensor,
    col: Tensor,
) -> tuple[Tensor, Tensor]:
    """Vectorized CSR expansion: ``(c, u)`` → ``(c, v)`` for all neighbors ``v`` of ``u``."""
    if centers.numel() == 0:
        return centers, nodes
    deg = rowptr[nodes + 1] - rowptr[nodes]
    if not bool((deg > 0).any()):
        empty = centers.new_empty(0)
        return empty, empty
    new_centers = torch.repeat_interleave(centers, deg)
    starts = rowptr[nodes]
    offsets = torch.zeros(nodes.numel() + 1, dtype=torch.long, device=centers.device)
    offsets[1:] = torch.cumsum(deg, dim=0)
    idx = torch.arange(new_centers.numel(), device=centers.device, dtype=torch.long)
    seg = torch.searchsorted(offsets[1:].contiguous(), idx, right=True)
    local = idx - offsets[seg]
    new_nodes = col[starts[seg] + local]
    return new_centers, new_nodes


def _same_batch_edges(
    edge_index: Tensor,
    num_nodes: int,
    batch_index: Tensor | None,
) -> tuple[Tensor, Tensor]:
    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    if batch_index is not None:
        batch_index = batch_index.to(device=src.device, dtype=torch.long)
        keep = batch_index[src] == batch_index[dst]
        src, dst = src[keep], dst[keep]
    keep = (src != dst) & (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    return src[keep], dst[keep]


@torch.compiler.disable
def build_multiscale_hop_neighborhoods(
    edge_index: Tensor,
    num_nodes: int,
    batch_index: Tensor | None = None,
    *,
    max_hop: int = MAX_HOP,
    scales: tuple[tuple[int, int], ...] = DEFAULT_HOP_SCALES,
) -> list[ScaleNeighborhoods]:
    """Exact min-hop neighborhoods via vectorized multi-source BFS (packed / varlen)."""
    if max_hop < 1:
        raise ValueError(f"max_hop must be >= 1; got {max_hop}.")
    device = edge_index.device
    src, dst = _same_batch_edges(edge_index, num_nodes, batch_index)
    if src.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return [ScaleNeighborhoods(empty, empty, empty) for _ in scales]

    rowptr, col = _csr_from_directed_edges(src, dst, num_nodes)

    # Centers start visited so back-edges never re-emit the center as a neighbor.
    node_ids = torch.arange(num_nodes, device=device, dtype=torch.long)
    visited_sorted = torch.sort(_pair_keys(node_ids, node_ids, num_nodes)).values

    hop_centers: list[Tensor] = []
    hop_nodes: list[Tensor] = []
    hop_dists: list[Tensor] = []

    frontier_c, frontier_n, new_keys = _filter_new_pairs(src, dst, visited_sorted, num_nodes)
    visited_sorted = _merge_sorted_keys(visited_sorted, new_keys)

    for hop in range(1, max_hop + 1):
        if frontier_c.numel() == 0:
            break
        hop_centers.append(frontier_c)
        hop_nodes.append(frontier_n)
        hop_dists.append(torch.full((frontier_c.numel(),), hop, dtype=torch.long, device=device))
        if hop == max_hop:
            break
        cand_c, cand_n = _expand_frontier(frontier_c, frontier_n, rowptr, col)
        frontier_c, frontier_n, new_keys = _filter_new_pairs(cand_c, cand_n, visited_sorted, num_nodes)
        visited_sorted = _merge_sorted_keys(visited_sorted, new_keys)

    if not hop_centers:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return [ScaleNeighborhoods(empty, empty, empty) for _ in scales]

    centers = torch.cat(hop_centers, dim=0)
    nodes = torch.cat(hop_nodes, dim=0)
    hops = torch.cat(hop_dists, dim=0)

    out: list[ScaleNeighborhoods] = []
    for lo, hi in scales:
        mask = (hops >= lo) & (hops <= hi)
        out.append(
            ScaleNeighborhoods(
                center_index=centers[mask],
                neighbor_index=nodes[mask],
                hop_distance=hops[mask],
            )
        )
    return out


class MultiscaleHopPE(GraphPE):
    """Pool learned relative geometry over exact multi-scale hop neighborhoods."""

    def __init__(self, config: MultiscaleHopPEConfig, *, pos_dim: int = 3) -> None:
        if scatter is None:
            raise ImportError("MultiscaleHopPE requires torch_geometric (scripts/install.sh, PyG stack).")
        super().__init__()
        self.config = config
        self.pos_dim = int(pos_dim)
        if self.pos_dim <= 0:
            raise ValueError(f"pos_dim must be > 0; got {self.pos_dim}.")
        hidden_dim = int(config.pointnet_hidden_dim)
        self.pointnets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.pos_dim + 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                for _ in DEFAULT_HOP_SCALES
            ]
        )
        rms_norm = getattr(nn, "RMSNorm", None)
        if rms_norm is None:
            raise RuntimeError("MultiscaleHopPE requires torch.nn.RMSNorm.")
        self.norms = nn.ModuleList(
            [rms_norm(2 * hidden_dim, eps=float(config.rms_norm_eps)) for _ in DEFAULT_HOP_SCALES]
        )
        self.out_dim = config.out_dim
        self._neighborhood_cache_key: tuple | None = None
        self._neighborhood_cache: list[ScaleNeighborhoods] | None = None

    @staticmethod
    def _topology_key(
        edge_index: Tensor,
        cu_seqlens: Tensor,
        num_nodes: int,
    ) -> tuple:
        """Content fingerprint so static meshes hit across dataloader reallocations."""
        nnz = int(edge_index.shape[-1]) if edge_index.ndim == 2 else 0
        if nnz == 0:
            edge_fp = (0, 0, 0, 0)
        else:
            flat = edge_index.reshape(-1)
            edge_fp = (
                nnz,
                int(flat[0].item()),
                int(flat[-1].item()),
                int(flat.sum().item()),
            )
        # cu_seqlens layout without depending on storage pointer.
        n_graphs = max(int(cu_seqlens.numel()) - 1, 0)
        last = int(cu_seqlens[-1].item()) if cu_seqlens.numel() else 0
        return (int(num_nodes), edge_fp, n_graphs, last)

    @torch.compiler.disable
    def _cache_or_build_neighborhoods(
        self,
        edge_index: Tensor,
        *,
        num_nodes: int,
        batch_index: Tensor,
        cu_seqlens: Tensor,
    ) -> list[ScaleNeighborhoods]:
        key = self._topology_key(edge_index, cu_seqlens, num_nodes)
        if key != self._neighborhood_cache_key or self._neighborhood_cache is None:
            self._neighborhood_cache = build_multiscale_hop_neighborhoods(edge_index, num_nodes, batch_index)
            self._neighborhood_cache_key = key
        return self._neighborhood_cache

    @staticmethod
    def _mean_edge_length(
        pos: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        num_graphs: int,
    ) -> Tensor:
        """Mean undirected edge length per graph (bidirected edges share length → mean ok)."""
        source, target = edge_index
        same_graph = batch_index[source] == batch_index[target]
        source, target = source[same_graph], target[same_graph]
        if source.numel() == 0:
            return pos.new_ones((num_graphs,))
        edge_lengths = torch.linalg.vector_norm(pos[target] - pos[source], dim=-1)
        graph_ids = batch_index[source]
        mean_lengths = scatter(edge_lengths, graph_ids, dim=0, dim_size=num_graphs, reduce="mean")
        return mean_lengths.clamp_min(1e-8)

    @staticmethod
    def _diagnostics(neighborhood: ScaleNeighborhoods, num_nodes: int) -> dict[str, float | int]:
        degrees = torch.bincount(neighborhood.center_index, minlength=num_nodes)
        degree_values = degrees.float()
        return {
            "num_pairs": int(neighborhood.center_index.numel()),
            "mean_degree": float(degree_values.mean().item()) if degrees.numel() else 0.0,
            "median_degree": float(degree_values.median().item()) if degrees.numel() else 0.0,
            "p95_degree": float(torch.quantile(degree_values, 0.95).item()) if degrees.numel() else 0.0,
            "p99_degree": float(torch.quantile(degree_values, 0.99).item()) if degrees.numel() else 0.0,
            "max_degree": float(degree_values.max().item()) if degrees.numel() else 0.0,
            "empty_fraction": float((degrees == 0).float().mean().item()) if degrees.numel() else 0.0,
        }

    def _pool_scale(
        self,
        *,
        pos: Tensor,
        pointnet: nn.Module,
        centers: Tensor,
        neighbor_index: Tensor,
        hop_distance: Tensor,
        mean_edge_lengths: Tensor,
        batch_index: Tensor,
        num_nodes: int,
        hidden: int,
    ) -> Tensor:
        """Vectorized PointNet + mean‖max scatter over flat pair lists."""
        del hidden  # inferred from messages
        deltas = pos[neighbor_index] - pos[centers]
        scales = mean_edge_lengths[batch_index[centers]].unsqueeze(-1)
        deltas = deltas / scales
        distances = torch.linalg.vector_norm(deltas, dim=-1, keepdim=True)
        hops = hop_distance.to(dtype=pos.dtype).unsqueeze(-1) / float(MAX_HOP)
        features = torch.cat([deltas, distances, hops], dim=-1)
        messages = pointnet(features.to(dtype=pointnet[0].weight.dtype)).to(dtype=pos.dtype)
        mean = scatter(messages, centers, dim=0, dim_size=num_nodes, reduce="mean")
        maximum = scatter(messages, centers, dim=0, dim_size=num_nodes, reduce="max")
        counts = torch.bincount(centers, minlength=num_nodes).unsqueeze(-1)
        present = counts > 0
        mean = torch.where(present, mean, torch.zeros_like(mean))
        maximum = torch.where(present & torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=-1)

    def _encode_scales(
        self,
        *,
        pos: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        neighborhoods: list[ScaleNeighborhoods],
        return_diagnostics: bool,
    ) -> Tensor | tuple[Tensor, list[dict[str, float | int]]]:
        num_nodes = int(pos.shape[0])
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        mean_edge_lengths = self._mean_edge_length(pos, edge_index, batch_index, max(num_graphs, 1))
        if not self.config.normalize_by_mean_edge_length:
            mean_edge_lengths = torch.ones_like(mean_edge_lengths)

        parts: list[Tensor] = []
        diagnostics: list[dict[str, float | int]] = []
        hidden = int(self.config.pointnet_hidden_dim)
        for neighborhood, pointnet, norm in zip(neighborhoods, self.pointnets, self.norms, strict=True):
            centers = neighborhood.center_index
            if centers.numel() == 0:
                pooled = pos.new_zeros((num_nodes, hidden * 2))
            else:
                pooled = self._pool_scale(
                    pos=pos,
                    pointnet=pointnet,
                    centers=centers,
                    neighbor_index=neighborhood.neighbor_index,
                    hop_distance=neighborhood.hop_distance,
                    mean_edge_lengths=mean_edge_lengths,
                    batch_index=batch_index,
                    num_nodes=num_nodes,
                    hidden=hidden,
                )
            parts.append(norm(pooled))
            if return_diagnostics:
                diagnostics.append(self._diagnostics(neighborhood, num_nodes))

        pe = torch.cat(parts, dim=-1)
        return (pe, diagnostics) if return_diagnostics else pe

    @torch.compiler.disable
    def forward(
        self,
        *,
        pos: Tensor,
        edge_index: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int | None = None,
        num_total_nodes: int | None = None,
        topology_features: TopologyFeatures = None,
        topology_eigenvalues: Tensor | None = None,
        batch_index: Tensor | None = None,
        return_diagnostics: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, list[dict[str, float | int]]]:
        del topology_features, topology_eigenvalues, kwargs
        if int(pos.shape[-1]) != self.pos_dim:
            raise ValueError(
                f"MultiscaleHopPE expected pos.shape[-1] == {self.pos_dim}, got {int(pos.shape[-1])}."
            )
        num_nodes = _packed_node_count(cu_seqlens, num_total_nodes)
        packed = _resolve_packed_indices(
            cu_seqlens,
            num_nodes,
            pos.device,
            batch_index=batch_index,
            max_seqlen=max_seqlen,
        )
        neighborhoods = self._cache_or_build_neighborhoods(
            edge_index,
            num_nodes=num_nodes,
            batch_index=packed.batch_index,
            cu_seqlens=cu_seqlens,
        )
        return self._encode_scales(
            pos=pos,
            edge_index=edge_index,
            batch_index=packed.batch_index,
            neighborhoods=neighborhoods,
            return_diagnostics=return_diagnostics,
        )
