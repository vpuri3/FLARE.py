"""Unified probe-distance PE for GLT (Euclidean + optional geodesic)."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor, nn

from pdebench.dataset.sample import FeatureRequest

from .pe_base import GraphPE, _packed_node_count

__all__ = [
    "ProbeDistGraphPE",
    "ProbeDistPE",
    "ProbeDistPEConfig",
    "_build_probe_dist_pe",
]


def _logit(u: Tensor) -> Tensor:
    return torch.log(u) - torch.log1p(-u)


def _validate_positive_int_count(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value}")
    return int(value)


def _validate_packed_graph(pos: Tensor, edge_index: Tensor, cu_seqlens: Tensor, dim: int) -> None:
    if pos.ndim != 2 or pos.shape[1] != dim:
        raise ValueError(f"pos must be [N,{dim}]; got {tuple(pos.shape)}")
    if not torch.isfinite(pos).all():
        raise ValueError("pos must contain only finite values")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E]; got {tuple(edge_index.shape)}")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("edge_index must have an integer dtype")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional tensor with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must have an integer dtype")
    if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != pos.shape[0]:
        raise ValueError("cu_seqlens must start at zero and end at the node count")
    if torch.any(cu_seqlens[1:] <= cu_seqlens[:-1]):
        raise ValueError("packed graphs must be nonempty")
    if edge_index.numel() and (torch.any(edge_index < 0) or torch.any(edge_index >= pos.shape[0])):
        raise ValueError("edge_index contains an out-of-range node index")
    if edge_index.numel():
        graph_boundaries = cu_seqlens[1:].to(edge_index.device)
        endpoint_graphs = torch.bucketize(edge_index, graph_boundaries, right=True)
        if torch.any(endpoint_graphs[0] != endpoint_graphs[1]):
            raise ValueError("edge_index contains an edge that crosses packed graphs")


def _physical_edge_lengths(pos: Tensor, edge_index: Tensor, scale: Tensor, distance_eps: float) -> Tensor:
    source, destination = edge_index.long()
    delta = (pos[source] - pos[destination]) * scale
    return delta.float().square().sum(dim=-1).clamp_min(distance_eps).sqrt().to(pos.dtype)


def _soft_anchor_weights(candidate_dist2: Tensor, temperature2: Tensor, dtype: torch.dtype) -> Tensor:
    return torch.softmax(-candidate_dist2 / temperature2, dim=0).to(dtype)


def _truncated_anchor_distances(
    num_nodes: int,
    edge_index: Tensor,
    edge_lengths: Tensor,
    anchor_nodes: Tensor,
    max_geodesic_hops: int,
    cap: Tensor,
) -> Tensor:
    num_fields = anchor_nodes.numel()
    dist = cap.to(dtype=edge_lengths.dtype, device=edge_lengths.device).expand(num_nodes, num_fields).clone()
    dist.scatter_(0, anchor_nodes.reshape(1, -1), 0.0)
    source, destination = edge_index.long()
    both_source = torch.cat((source, destination))
    both_destination = torch.cat((destination, source))
    both_lengths = torch.cat((edge_lengths, edge_lengths))
    for _ in range(max_geodesic_hops):
        proposals = dist[both_source] + both_lengths[:, None]
        next_dist = cap.to(dtype=dist.dtype, device=dist.device).expand_as(dist).clone()
        next_dist.scatter_reduce_(
            0,
            both_destination[:, None].expand(-1, dist.shape[1]),
            proposals,
            reduce="amin",
            include_self=False,
        )
        dist = torch.minimum(dist, next_dist)
    return dist


class ProbeDistPE(nn.Module):
    def __init__(
        self,
        num_probes: int,
        dim: int,
        domain_min: Tensor,
        domain_max: Tensor,
        scale: Tensor,
        shift: Tensor,
        *,
        euclidean_feats: bool = True,
        geodesic_feats: bool = False,
        num_anchor_candidates: int = 4,
        max_geodesic_hops: int = 32,
        temperature: float = 1.0,
        distance_cap: float = 2.0,
        init_eps: float = 1e-4,
        distance_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not euclidean_feats:
            raise ValueError("euclidean_feats must be True")
        d = int(dim)
        if d not in (2, 3):
            raise ValueError(f"dim must be 2 or 3; got {d}")
        k = _validate_positive_int_count("num_probes", num_probes)
        candidates = _validate_positive_int_count("num_anchor_candidates", num_anchor_candidates)
        hops = _validate_positive_int_count("max_geodesic_hops", max_geodesic_hops)
        for name, value in (
            ("temperature", temperature),
            ("distance_cap", distance_cap),
            ("init_eps", init_eps),
            ("distance_eps", distance_eps),
        ):
            if not torch.isfinite(torch.tensor(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite; got {value}")
        if init_eps >= 0.5:
            raise ValueError(f"init_eps must be less than 0.5; got {init_eps}")

        buffers = {}
        for name, value in (
            ("domain_min", domain_min),
            ("domain_max", domain_max),
            ("scale", scale),
            ("shift", shift),
        ):
            value = value.detach().float().reshape(d)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
            buffers[name] = value
        if not torch.all(buffers["domain_max"] > buffers["domain_min"]):
            raise ValueError("domain_max must be > domain_min elementwise")
        if not torch.all(buffers["scale"] != 0):
            raise ValueError("scale must be nonzero elementwise")

        self.dim = d
        self.num_probes = k
        self.euclidean_feats = bool(euclidean_feats)
        self.geodesic_feats = bool(geodesic_feats)
        self.num_anchor_candidates = candidates
        self.max_geodesic_hops = hops
        self.temperature = float(temperature)
        self.distance_cap = float(distance_cap)
        self.init_eps = float(init_eps)
        self.distance_eps = float(distance_eps)
        for name, value in buffers.items():
            self.register_buffer(name, value)
        b = buffers["scale"].abs() * (buffers["domain_max"] - buffers["domain_min"])
        self.register_buffer("reference_length", torch.linalg.vector_norm(b))
        u = torch.empty(k, d).uniform_(self.init_eps, 1.0 - self.init_eps)
        self.probe_logits = nn.Parameter(_logit(u))

    @property
    def output_dim(self) -> int:
        euc = self.num_probes * (self.dim + 1)
        geo = 2 * self.num_probes if self.geodesic_feats else 0
        return euc + geo

    def probe_locations(self) -> Tensor:
        u = torch.sigmoid(self.probe_logits)
        return self.domain_min + u * (self.domain_max - self.domain_min)

    def physical_probe_locations(self) -> Tensor:
        return self.probe_locations() * self.scale + self.shift

    def _euclidean_features(self, pos: Tensor) -> Tensor:
        if pos.ndim != 2 or pos.shape[-1] != self.dim:
            raise ValueError(f"pos must be [N,{self.dim}]; got {tuple(pos.shape)}")
        a = self.probe_locations()  # [K,d]
        delta = (pos.unsqueeze(1) - a.unsqueeze(0)) * self.scale.view(1, 1, -1)
        ell = self.reference_length.clamp_min(self.distance_eps)
        delta_n = delta / ell
        r2 = delta.float().square().sum(dim=-1).add(self.distance_eps)
        r_n = r2.sqrt().to(dtype=pos.dtype) / ell
        if self.dim == 2:
            dx, dy = delta_n.unbind(dim=-1)
            feat = torch.stack([dx, dy, r_n], dim=-1)
        else:
            dx, dy, dz = delta_n.unbind(dim=-1)
            feat = torch.stack([dx, dy, dz, r_n], dim=-1)
        return feat.reshape(pos.shape[0], self.num_probes * (self.dim + 1))

    def _geodesic_features(self, pos: Tensor, edge_index: Tensor, cu_seqlens: Tensor) -> Tensor:
        return self._forward_geodesic_packed_eager(pos, edge_index, cu_seqlens)

    @torch.compiler.disable
    def _forward_geodesic_packed_eager(self, pos: Tensor, edge_index: Tensor, cu_seqlens: Tensor) -> Tensor:
        """Validate and evaluate variable-size packed graphs outside Dynamo capture.

        Python slicing boundaries are data-dependent values from ``cu_seqlens``;
        keeping this method eager avoids scalar graph breaks while allowing the
        surrounding training graph to compile before and after this PE call.
        """
        _validate_packed_graph(pos, edge_index, cu_seqlens, self.dim)
        edge_index = edge_index.long()
        edge_lengths = _physical_edge_lengths(pos, edge_index, self.scale, self.distance_eps)
        if not torch.isfinite(edge_lengths).all():
            raise ValueError("physical edge lengths must contain only finite values")
        probes = self.physical_probe_locations()
        if not torch.isfinite(probes).all():
            raise ValueError("physical probe locations must contain only finite values")
        physical_pos = pos * self.scale + self.shift
        if not torch.isfinite(physical_pos).all():
            raise ValueError("physical positions must contain only finite values")
        output = pos.new_empty((pos.shape[0], 2 * self.num_probes))

        for graph_index in range(cu_seqlens.numel() - 1):
            start = int(cu_seqlens[graph_index])
            stop = int(cu_seqlens[graph_index + 1])
            graph_nodes = stop - start
            edge_mask = (
                (edge_index[0] >= start)
                & (edge_index[0] < stop)
                & (edge_index[1] >= start)
                & (edge_index[1] < stop)
            )
            graph_edges = edge_index[:, edge_mask] - start
            graph_edge_lengths = edge_lengths[edge_mask]
            usable_edges = graph_edges[0] != graph_edges[1]
            graph_edges = graph_edges[:, usable_edges]
            graph_edge_lengths = graph_edge_lengths[usable_edges]
            if graph_nodes > 1 and not graph_edge_lengths.numel():
                raise ValueError(f"packed graph {graph_index} has no usable non-self edges")
            graph_physical_pos = physical_pos[start:stop]
            reference_length = torch.linalg.vector_norm(
                graph_physical_pos.max(dim=0).values - graph_physical_pos.min(dim=0).values
            ).clamp_min(self.distance_eps)
            if graph_nodes == 1:
                mean_edge_length = reference_length
            else:
                mean_edge_length = graph_edge_lengths.mean().detach()

            probe_delta = graph_physical_pos[:, None, :] - probes[None, :, :]
            probe_dist2 = probe_delta.float().square().sum(dim=-1)
            candidate_count = min(self.num_anchor_candidates, graph_nodes)
            candidate_dist2, candidate_nodes = torch.topk(
                probe_dist2, candidate_count, dim=0, largest=False, sorted=False
            )
            temperature2 = (self.temperature * mean_edge_length.square()).clamp_min(self.distance_eps)
            weights = _soft_anchor_weights(candidate_dist2, temperature2, pos.dtype)
            off_mesh = (weights * candidate_dist2.sqrt()).sum(dim=0)

            flattened_anchors = candidate_nodes.transpose(0, 1).reshape(-1)
            candidate_fields = _truncated_anchor_distances(
                graph_nodes,
                graph_edges,
                graph_edge_lengths,
                flattened_anchors,
                self.max_geodesic_hops,
                self.distance_cap * reference_length,
            ).reshape(graph_nodes, self.num_probes, candidate_count).detach()
            mesh = (candidate_fields * weights.transpose(0, 1).unsqueeze(0)).sum(dim=-1)
            features = torch.stack(
                (mesh / reference_length, (off_mesh / reference_length).expand(graph_nodes, -1)), dim=-1
            )
            output[start:stop] = features.reshape(graph_nodes, 2 * self.num_probes)
        return output

    def forward(
        self,
        pos: Tensor,
        edge_index: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor:
        euc = self._euclidean_features(pos)
        if not self.geodesic_feats:
            return euc
        if edge_index is None or cu_seqlens is None:
            raise ValueError("geodesic_feats requires edge_index and cu_seqlens")
        geo = self._geodesic_features(pos, edge_index, cu_seqlens)
        return torch.cat((euc, geo), dim=-1)


@dataclass
class ProbeDistPEConfig:
    kind: Literal["probe_dist"] = "probe_dist"
    num_probes: int = 32
    euclidean_feats: bool = True
    geodesic_feats: bool = False
    init_eps: float = 1e-4
    distance_eps: float = 1e-12
    num_anchor_candidates: int = 4
    max_geodesic_hops: int = 32
    temperature: float = 1.0
    distance_cap: float = 2.0

    def __post_init__(self) -> None:
        if self.kind != "probe_dist":
            raise ValueError(f"ProbeDistPEConfig kind must be 'probe_dist'; got {self.kind!r}.")
        if not self.euclidean_feats:
            raise ValueError("euclidean_feats must be True")
        _validate_positive_int_count("num_probes", self.num_probes)
        _validate_positive_int_count("num_anchor_candidates", self.num_anchor_candidates)
        _validate_positive_int_count("max_geodesic_hops", self.max_geodesic_hops)

    def to_feature_request(self) -> FeatureRequest:
        return FeatureRequest(edges=bool(self.geodesic_feats), laplacian_k=0, pos_domain=True)


class ProbeDistGraphPE(GraphPE):
    def __init__(self, core: ProbeDistPE) -> None:
        super().__init__()
        self.core = core
        self.out_dim = int(core.output_dim)

    def forward(
        self,
        *,
        pos: Tensor,
        edge_index: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int | None = None,
        num_total_nodes: int | None = None,
        **kwargs,
    ) -> Tensor:
        del max_seqlen, kwargs
        n = _packed_node_count(cu_seqlens, num_total_nodes)
        if pos.shape[0] != n:
            raise ValueError(f"pos rows {pos.shape[0]} != packed node count {n}")
        if self.core.geodesic_feats:
            return self.core(pos, edge_index, cu_seqlens)
        return self.core(pos)


def _build_probe_dist_pe(
    config: ProbeDistPEConfig,
    *,
    pos_dim: int,
    act: str,
    pos_domain=None,
) -> ProbeDistGraphPE:
    del act
    if pos_domain is None:
        raise ValueError("probe_dist requires PosDomain")
    d = int(pos_domain.scale.numel())
    if int(pos_dim) != d:
        raise ValueError(f"pos_dim {pos_dim} != PosDomain dim {d}")
    core = ProbeDistPE(
        num_probes=config.num_probes,
        dim=d,
        domain_min=pos_domain.normalized_pos_expanse[:, 0].reshape(d),
        domain_max=pos_domain.normalized_pos_expanse[:, 1].reshape(d),
        scale=pos_domain.scale.reshape(d),
        shift=pos_domain.shift.reshape(d),
        euclidean_feats=True,
        geodesic_feats=bool(config.geodesic_feats),
        num_anchor_candidates=config.num_anchor_candidates,
        max_geodesic_hops=config.max_geodesic_hops,
        temperature=float(config.temperature),
        distance_cap=float(config.distance_cap),
        init_eps=float(config.init_eps),
        distance_eps=float(config.distance_eps),
    )
    return ProbeDistGraphPE(core)
