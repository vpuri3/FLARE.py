"""Non-eigen GLT PE kinds (none, geo_transolver_pe)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from pdebench.dataset.sample import FeatureRequest

from ..geo_transolver import _GEO_PAD, _as_xyz, _pack_batch, _unpack_batch
from ..geotransolver_pn.ball_query import BQWarp
from ..geotransolver_pn.pn_compat import Mlp
from .pe_base import (
    GraphPE,
    TopologyFeatures,
    _packed_node_count,
    _resolve_packed_indices,
)

__all__ = [
    "GeoTransolverPE",
    "GeoTransolverPEConfig",
    "NonePE",
    "NonePEConfig",
    "_build_geo_transolver_pe",
    "_build_none_pe",
]


@dataclass
class NonePEConfig:
    kind: Literal["none"] = "none"

    def to_feature_request(self) -> FeatureRequest:
        return FeatureRequest(edges=False, laplacian_k=0)


class NonePE(GraphPE):
    """Explicit no positional encoding: ``out_dim=0``, returns ``[N, 0]``."""

    def __init__(self) -> None:
        super().__init__()
        self.out_dim = 0

    def forward(
        self,
        *,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        num_total_nodes: int | None = None,
        topology_features: TopologyFeatures = None,
        topology_eigenvalues: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del edge_index, max_seqlen, topology_features, topology_eigenvalues, batch_index, kwargs
        n = _packed_node_count(cu_seqlens, num_total_nodes)
        return pos.new_zeros((n, 0))

def _as_tuple_floats(value: tuple[float, ...] | list[float] | float) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


def _as_tuple_ints(value: tuple[int, ...] | list[int] | int) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)


class _ScaleBallQueryMLP(nn.Module):
    """Single-scale BQWarp + MLP over flattened neighbor coordinates."""

    def __init__(self, radius: float, neighbors_in_radius: int, geometry_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.bq_warp = BQWarp(radius=float(radius), neighbors_in_radius=int(neighbors_in_radius))
        self.mlp = Mlp(
            in_features=int(geometry_dim) * int(neighbors_in_radius),
            hidden_features=[int(hidden_dim), int(hidden_dim) // 2],
            out_features=int(hidden_dim),
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(
        self,
        pos: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, neighbors = self.bq_warp(pos, pos, query_mask=query_mask, key_mask=key_mask)
        flat = neighbors.reshape(neighbors.shape[0], neighbors.shape[1], -1)
        return torch.tanh(self.mlp(flat))


@dataclass
class GeoTransolverPEConfig:
    kind: Literal["geo_transolver_pe"] = "geo_transolver_pe"
    radii: tuple[float, ...] = (0.05, 0.25)
    neighbors_in_radius: tuple[int, ...] = (8, 32)
    n_hidden_local: int = 32
    geometry_dim: int = 3

    def __post_init__(self) -> None:
        self.radii = _as_tuple_floats(self.radii)
        self.neighbors_in_radius = _as_tuple_ints(self.neighbors_in_radius)
        if int(self.geometry_dim) != 3:
            raise ValueError(f"geometry_dim must be 3 for BQWarp; got {self.geometry_dim}.")
        if int(self.n_hidden_local) <= 0:
            raise ValueError(f"n_hidden_local must be > 0; got {self.n_hidden_local}.")
        if len(self.radii) == 0:
            raise ValueError("radii must be non-empty.")
        if len(self.radii) != len(self.neighbors_in_radius):
            raise ValueError(
                f"radii and neighbors_in_radius length mismatch: "
                f"{len(self.radii)} vs {len(self.neighbors_in_radius)}."
            )
        if any(k <= 0 for k in self.neighbors_in_radius):
            raise ValueError(f"neighbors_in_radius entries must be > 0; got {self.neighbors_in_radius}.")

    @property
    def out_dim_local(self) -> int:
        return int(self.n_hidden_local) * len(self.radii)

    def to_feature_request(self) -> FeatureRequest:
        return FeatureRequest(edges=False, laplacian_k=0)


class GeoTransolverPE(GraphPE):
    """Multi-scale ball-query local PE for packed GLT graphs."""

    def __init__(self, config: GeoTransolverPEConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.n_hidden_local)
        self.scales = nn.ModuleList(
            [
                _ScaleBallQueryMLP(r, k, config.geometry_dim, h)
                for r, k in zip(config.radii, config.neighbors_in_radius, strict=True)
            ]
        )
        self.out_dim = int(config.out_dim_local)

    def encode_local(
        self,
        pos: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``pos: [B, N, 3] → local: [B, N, n_hidden_local * R]``."""
        if pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError(f"encode_local expects pos [B, N, 3]; got {tuple(pos.shape)}.")
        parts = [scale(pos, query_mask=query_mask, key_mask=key_mask) for scale in self.scales]
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        *,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
        num_total_nodes: int | None = None,
        topology_features: TopologyFeatures = None,
        topology_eigenvalues: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del edge_index, topology_features, topology_eigenvalues, kwargs
        n = _packed_node_count(cu_seqlens, num_total_nodes)
        packed = _resolve_packed_indices(
            cu_seqlens,
            n,
            pos.device,
            batch_index=batch_index,
            max_seqlen=max_seqlen,
        )
        xyz = _as_xyz(pos, self.config.geometry_dim)
        pos_b, mask, counts = _pack_batch(xyz, packed.batch_index, pad_value=_GEO_PAD)
        local_b = self.encode_local(pos_b, query_mask=mask, key_mask=mask)
        return _unpack_batch(local_b, counts).to(dtype=pos.dtype)



def _build_none_pe(config: NonePEConfig, *, pos_dim: int, act: str, pos_domain=None) -> NonePE:
    del config, pos_dim, act, pos_domain
    return NonePE()


def _build_geo_transolver_pe(
    config: GeoTransolverPEConfig, *, pos_dim: int, act: str, pos_domain=None
) -> GeoTransolverPE:
    del pos_dim, act, pos_domain
    return GeoTransolverPE(config)
