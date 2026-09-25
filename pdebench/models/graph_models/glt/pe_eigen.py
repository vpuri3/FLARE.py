"""Eigenvector / spectral PE kinds for GLT (raw_eigen, spe)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from pdebench.dataset.laplacian.spec import laplacian_spec_dim
from pdebench.dataset.sample import FeatureRequest

from .pe_base import (
    GraphPE,
    TopologyFeatures,
    _normalize_pe,
    _packed_node_count,
)

__all__ = [
    "PE_SPE_FILTER_TYPES",
    "PE_SPE_MODES",
    "RawEigenPE",
    "RawEigenPEConfig",
    "SpectralFilterPE",
    "SpectralFilterPEConfig",
    "_build_raw_eigen_pe",
    "_build_spe_pe",
    "_parse_pe_spe_filter_type",
    "_parse_pe_spe_mode",
]

PE_SPE_MODES: tuple[str, ...] = ("query", "multihop")
PE_SPE_FILTER_TYPES: tuple[str, ...] = ("band", "spatial")


def _parse_pe_spe_mode(pe_spe_mode: str) -> str:
    mode = str(pe_spe_mode).strip().lower()
    if mode not in PE_SPE_MODES:
        raise ValueError(f"pe_spe_mode must be one of {PE_SPE_MODES}; got {pe_spe_mode!r}.")
    return mode


def _parse_pe_spe_filter_type(pe_spe_filter_type: str) -> str:
    filter_type = str(pe_spe_filter_type).strip().lower()
    if filter_type not in PE_SPE_FILTER_TYPES:
        raise ValueError(f"pe_spe_filter_type must be one of {PE_SPE_FILTER_TYPES}; got {pe_spe_filter_type!r}.")
    return filter_type


def _activation(act: str | None) -> type[nn.Module]:
    return nn.SiLU if act == "silu" else nn.GELU

def _first_key(mapping: dict, *keys: str, default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default




def _split_topology_inputs(
    topology_features: TopologyFeatures,
    topology_eigenvalues: torch.Tensor | None,
    *,
    model_name: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if isinstance(topology_features, dict):
        eigenvectors = _first_key(topology_features, "eigenvectors", "eigvecs", "vectors")
        eigenvalues = _first_key(
            topology_features,
            "eigenvalues",
            "eigvals",
            "values",
            default=topology_eigenvalues,
        )
        return eigenvectors, eigenvalues
    if isinstance(topology_features, (tuple, list)):
        if len(topology_features) != 2:
            raise RuntimeError(f"{model_name} topology_features tuple must be (eigenvalues, eigenvectors).")
        eigenvalues, eigenvectors = topology_features
        return eigenvectors, eigenvalues
    return topology_features, topology_eigenvalues


def _raw_topology_features(
    *,
    model_name: str,
    num_eigenvectors: int,
    edge_index: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_total_nodes: int | None = None,
    topology_features: torch.Tensor | None = None,
) -> torch.Tensor:
    num_total_nodes = _packed_node_count(cu_seqlens, num_total_nodes)
    if int(num_eigenvectors) == 0:
        return edge_index.new_empty((num_total_nodes, 0), dtype=torch.float32).to(device=edge_index.device)
    if topology_features is None:
        raise RuntimeError(
            f"{model_name} requires topology features. Enable the GINOT Laplacian cache path or run "
            "python -m pdebench.dataset.laplacian.precompute first."
        )
    if topology_features.ndim != 2:
        raise RuntimeError(
            f"{model_name} topology_features must be [N_tot, K], got {tuple(topology_features.shape)}."
        )
    if int(topology_features.shape[0]) != num_total_nodes:
        raise RuntimeError(
            f"{model_name} topology_features row count must match packed point count: "
            f"{int(topology_features.shape[0])} vs {num_total_nodes}."
        )
    if int(topology_features.shape[1]) < int(num_eigenvectors):
        raise RuntimeError(
            f"{model_name} topology_features has too few eigenvectors: "
            f"{int(topology_features.shape[1])} < {int(num_eigenvectors)}."
        )
    return topology_features[:, : int(num_eigenvectors)].to(device=edge_index.device)


def _raw_topology_eigendecomp(
    *,
    model_name: str,
    num_eigenvectors: int,
    edge_index: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_total_nodes: int | None = None,
    topology_features: TopologyFeatures = None,
    topology_eigenvalues: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvectors_input, eigenvalues = _split_topology_inputs(
        topology_features,
        topology_eigenvalues,
        model_name=model_name,
    )
    eigenvectors = _raw_topology_features(
        model_name=model_name,
        num_eigenvectors=num_eigenvectors,
        edge_index=edge_index,
        cu_seqlens=cu_seqlens,
        num_total_nodes=num_total_nodes,
        topology_features=eigenvectors_input,
    )
    if eigenvalues is None:
        raise RuntimeError(
            f"{model_name} requires precomputed topology eigenvalues in addition to eigenvectors. "
            "Pass topology_eigenvalues or topology_features=(eigenvalues, eigenvectors)."
        )
    if eigenvalues.ndim not in {1, 2}:
        raise RuntimeError(f"{model_name} topology_eigenvalues must be [K] or [B, K], got {tuple(eigenvalues.shape)}.")
    if int(eigenvalues.shape[-1]) < int(num_eigenvectors):
        raise RuntimeError(
            f"{model_name} topology_eigenvalues has too few values: "
            f"{int(eigenvalues.shape[-1])} < {int(num_eigenvectors)}."
        )
    batch_size = int(cu_seqlens.numel() - 1)
    eigenvalues = eigenvalues[..., : int(num_eigenvectors)].to(device=edge_index.device)
    if eigenvalues.ndim == 1:
        eigenvalues = eigenvalues.unsqueeze(0).expand(batch_size, -1)
    elif int(eigenvalues.shape[0]) != batch_size:
        raise RuntimeError(
            f"{model_name} topology_eigenvalues batch count must match cu_seqlens: "
            f"{int(eigenvalues.shape[0])} vs {batch_size}."
        )
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Adjacency helpers (SPE multihop mode)
# ---------------------------------------------------------------------------

def _same_graph_edges(
    edge_index: torch.Tensor,
    batch_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    src = edge_index[0].to(device=device, dtype=torch.long)
    dst = edge_index[1].to(device=device, dtype=torch.long)
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]
    same_graph = batch_index[src] == batch_index[dst]
    return src[same_graph], dst[same_graph]


def _adjacency_aggregate(
    edge_index: torch.Tensor,
    batch_index: torch.Tensor,
    num_nodes: int,
    features: torch.Tensor,
) -> torch.Tensor:
    src, dst = _same_graph_edges(edge_index, batch_index, num_nodes)
    out = torch.zeros_like(features)
    out.index_add_(0, src, features[dst])
    return out


def _build_query_features(pos: torch.Tensor, poly_order: int = 1) -> torch.Tensor:
    ones = torch.ones(pos.shape[0], 1, device=pos.device, dtype=pos.dtype)
    parts = [ones, pos]
    if poly_order >= 2:
        parts.append(pos.square())
        d = pos.shape[-1]
        for i in range(d):
            for j in range(i + 1, d):
                parts.append((pos[:, i] * pos[:, j]).unsqueeze(-1))
    return torch.cat(parts, dim=-1)

# ---------------------------------------------------------------------------
# Raw eigenvector PE — cached eigenvectors + graph_rms normalization
# ---------------------------------------------------------------------------

@dataclass
class RawEigenPEConfig:
    kind: Literal["raw_eigen"] = "raw_eigen"
    num_eigenmodes: int = 32
    laplacian_spec: str = "graph"

    def __post_init__(self) -> None:
        if int(self.num_eigenmodes) <= 0:
            raise ValueError(
                f"RawEigenPEConfig requires num_eigenmodes > 0; got {self.num_eigenmodes}. "
                "Use NonePEConfig (kind='none') for no PE."
            )

    def to_feature_request(self) -> FeatureRequest:
        return FeatureRequest(
            edges=True,
            laplacian_k=int(self.num_eigenmodes),
            laplacian_spec=str(self.laplacian_spec),
        )


class RawEigenPE(GraphPE):
    """Cached Laplacian eigenvectors passed through with graph_rms normalization."""

    def __init__(self, num_eigenmodes: int, *, laplacian_spec: str = "graph"):
        super().__init__()
        self.num_eigenmodes = int(num_eigenmodes)
        if self.num_eigenmodes <= 0:
            raise ValueError(
                f"RawEigenPE requires num_eigenmodes > 0; got {self.num_eigenmodes}. "
                "Use NonePE (kind='none') for no PE."
            )
        self.laplacian_spec = str(laplacian_spec)
        self.out_dim = int(laplacian_spec_dim(self.laplacian_spec, self.num_eigenmodes))

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
        del max_seqlen, kwargs
        eigenvectors_input, _ = _split_topology_inputs(
            topology_features,
            topology_eigenvalues,
            model_name="RawEigenPE",
        )
        raw_eigen = _raw_topology_features(
            model_name="RawEigenPE",
            num_eigenvectors=self.out_dim,
            edge_index=edge_index,
            cu_seqlens=cu_seqlens,
            num_total_nodes=num_total_nodes,
            topology_features=eigenvectors_input,
        )
        return _normalize_pe(raw_eigen.to(dtype=pos.dtype), cu_seqlens, "graph_rms", batch_index=batch_index)


# ---------------------------------------------------------------------------
# Learnable SPE — basis-invariant spectral filter PE
# ---------------------------------------------------------------------------

@dataclass
class SpectralFilterPEConfig:
    kind: Literal["spe"] = "spe"
    num_eigenmodes: int = 32
    laplacian_spec: str = "graph"
    filter_type: str = "band"
    mode: str = "query"
    num_filters: int = 16
    hidden: int = 64
    band_sigma_init: float = 1e-6
    poly_order: int = 2
    num_hops: int = 3

    def to_feature_request(self) -> FeatureRequest:
        if int(self.num_eigenmodes) <= 0:
            raise ValueError("SpectralFilterPEConfig requires num_eigenmodes > 0")
        return FeatureRequest(
            edges=True,
            laplacian_k=int(self.num_eigenmodes),
            laplacian_spec=str(self.laplacian_spec),
        )


class SpectralFilterPE(GraphPE):
    """Learnable SPE with query or multihop readout."""

    def __init__(
        self,
        num_modes: int,
        *,
        pos_dim: int,
        filter_type: str = "band",
        spe_mode: str = "query",
        num_filters: int = 16,
        hidden_dim: int = 64,
        act: str = "gelu",
        band_sigma_init: float = 1e-6,
        poly_order: int = 2,
        num_hops: int = 3,
    ):
        super().__init__()
        if int(num_modes) <= 0:
            raise ValueError(f"SpectralFilterPE requires num_modes > 0, got {num_modes}.")
        if int(num_filters) < 0:
            raise ValueError("SpectralFilterPE num_filters must be >= 0.")
        self.filter_type = _parse_pe_spe_filter_type(filter_type)
        self.spe_mode = _parse_pe_spe_mode(spe_mode)
        self.num_modes = int(num_modes)
        self.num_filters = int(num_filters)
        self.num_hops = int(num_hops)
        self.pos_dim = int(pos_dim)
        self.poly_order = int(poly_order)
        d = self.pos_dim
        cross_terms = d * (d - 1) // 2 if self.poly_order >= 2 else 0
        self.query_dim = 1 + d + (d + cross_terms if self.poly_order >= 2 else 0)
        act_cls = _activation(act)

        if self.filter_type == "band":
            self.num_total_filters = self.num_modes
            init = float(max(band_sigma_init, 1e-8))
            self.band_log_sigma = nn.Parameter(torch.log(torch.expm1(torch.tensor(init))))
        else:
            if self.num_filters <= 0:
                raise ValueError("SpectralFilterPE filter_type='spatial' requires num_filters > 0.")
            self.num_total_filters = int(num_filters)
            spatial_in = self.query_dim * 2
            self.spatial_phi = nn.Sequential(
                nn.Linear(spatial_in, int(hidden_dim)),
                act_cls(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                act_cls(),
                nn.Linear(int(hidden_dim), self.num_total_filters),
            )

        if self.spe_mode == "query":
            self.out_feature_dim = self.num_total_filters * self.query_dim
            self.norm_mode = "graph_rms"
        else:
            self.out_feature_dim = self.num_total_filters * (self.num_hops + 1)
            self.norm_mode = "node_rms"

    @property
    def out_dim(self) -> int:
        return self.out_feature_dim

    def _band_phi(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Return band-pass phi(lambda) with shape [B, K, K]."""
        lam = eigenvalues.float()
        sigma = torch.nn.functional.softplus(self.band_log_sigma) + 1e-12
        diff = lam[:, :, None] - lam[:, None, :]
        return torch.softmax(-diff.square() / sigma, dim=-1)

    def _resolve_phi(
        self,
        *,
        v: torch.Tensor,
        pos: torch.Tensor | None,
        batch_index: torch.Tensor,
        batch_size: int,
        eigenvalues: torch.Tensor,
        work_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.filter_type == "spatial":
            if pos is None:
                raise RuntimeError("SpectralFilterPE spatial filter requires pos.")
            x = _build_query_features(pos, poly_order=self.poly_order).to(device=v.device, dtype=work_dtype)
            vq = v.new_zeros(batch_size, self.num_modes, x.shape[-1])
            vq.index_add_(0, batch_index, v.unsqueeze(-1) * x.unsqueeze(-2))
            vq2 = vq.square()
            s_k = torch.cat([vq2, vq2.square()], dim=-1)
            return self.spatial_phi(s_k).transpose(1, 2).to(dtype=work_dtype)

        eigenvalues = eigenvalues.to(device=v.device)
        if eigenvalues.ndim == 1:
            eigenvalues = eigenvalues.unsqueeze(0).expand(batch_size, -1)
        eigenvalues = eigenvalues[:, : self.num_modes]
        return self._band_phi(eigenvalues).to(dtype=work_dtype)

    def _forward_query(
        self,
        v: torch.Tensor,
        phi_nodes: torch.Tensor,
        pos: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int,
        cu: torch.Tensor,
        out_dtype: torch.dtype,
        work_dtype: torch.dtype,
    ) -> torch.Tensor:
        x = _build_query_features(pos, poly_order=self.poly_order).to(device=v.device, dtype=work_dtype)
        vq = v.new_zeros(batch_size, self.num_modes, x.shape[-1])
        vq.index_add_(0, batch_index, v.unsqueeze(-1) * x.unsqueeze(-2))
        contrib = v.unsqueeze(-1) * vq.index_select(0, batch_index)
        z_query_modes = torch.einsum("nlk,nkc->nlc", phi_nodes, contrib)
        pe = z_query_modes.reshape(v.shape[0], -1).to(dtype=out_dtype)
        return _normalize_pe(pe, cu, "graph_rms", batch_index=batch_index)

    def _forward_multihop(
        self,
        v: torch.Tensor,
        phi_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        cu: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        num_nodes = int(v.shape[0])
        pe_parts: list[torch.Tensor] = []
        walk = v
        for power in range(self.num_hops + 1):
            if power == 0:
                mode_contrib = v.square()
            else:
                walk = _adjacency_aggregate(edge_index, batch_index, num_nodes, walk)
                mode_contrib = v * walk
            pe_parts.append(torch.einsum("nlk,nk->nl", phi_nodes, mode_contrib))
        pe = torch.cat(pe_parts, dim=-1).to(dtype=out_dtype)
        return _normalize_pe(pe, cu, "node_rms", batch_index=batch_index)

    def _forward_spe(
        self,
        eigenvectors: torch.Tensor,
        *,
        pos: torch.Tensor | None,
        edge_index: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        eigenvalues: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SPE math on already-loaded eigenvectors/eigenvalues. Unchanged from the original impl."""
        if eigenvectors.ndim != 2:
            raise RuntimeError(f"SpectralFilterPE expects eigenvectors [N, K], got {tuple(eigenvectors.shape)}.")
        if int(eigenvectors.shape[-1]) != self.num_modes:
            raise RuntimeError(
                f"SpectralFilterPE expects K={self.num_modes}, got K={int(eigenvectors.shape[-1])}."
            )
        if self.spe_mode == "query" and pos is None:
            raise RuntimeError("SpectralFilterPE query mode requires pos.")
        if self.spe_mode == "multihop" and edge_index is None:
            raise RuntimeError("SpectralFilterPE multihop mode requires edge_index.")
        if self.filter_type == "spatial" and pos is None:
            raise RuntimeError("SpectralFilterPE spatial filter requires pos.")

        out_dtype = eigenvectors.dtype
        device = eigenvectors.device
        work_dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
        v = eigenvectors.to(dtype=work_dtype)
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        batch_size = int(cu.numel() - 1)
        if batch_index is None:
            graph_lens = cu[1:] - cu[:-1]
            batch_index = torch.repeat_interleave(
                torch.arange(batch_size, device=device, dtype=torch.long), graph_lens
            )
        else:
            batch_index = batch_index.to(device=device, dtype=torch.long)
        phi = self._resolve_phi(
            v=v,
            pos=pos,
            batch_index=batch_index,
            batch_size=batch_size,
            eigenvalues=eigenvalues,
            work_dtype=work_dtype,
        )
        phi_nodes = phi.index_select(0, batch_index)

        if self.spe_mode == "query":
            return self._forward_query(
                v,
                phi_nodes,
                pos,
                batch_index,
                batch_size,
                cu,
                out_dtype,
                work_dtype,
            )
        return self._forward_multihop(v, phi_nodes, edge_index, batch_index, cu, out_dtype)

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
        """``GraphPE`` contract: load eigenvectors/eigenvalues, then run the SPE math unchanged."""
        del max_seqlen, kwargs
        eigenvalues, eigenvectors = _raw_topology_eigendecomp(
            model_name="SpectralFilterPE",
            num_eigenvectors=self.num_modes,
            edge_index=edge_index,
            cu_seqlens=cu_seqlens,
            num_total_nodes=num_total_nodes,
            topology_features=topology_features,
            topology_eigenvalues=topology_eigenvalues,
        )
        return self._forward_spe(
            eigenvectors,
            pos=pos,
            edge_index=edge_index,
            cu_seqlens=cu_seqlens,
            eigenvalues=eigenvalues,
            batch_index=batch_index,
        )


def _build_raw_eigen_pe(config: RawEigenPEConfig, *, pos_dim: int, act: str, pos_domain=None) -> RawEigenPE:
    del pos_dim, act, pos_domain
    return RawEigenPE(num_eigenmodes=config.num_eigenmodes, laplacian_spec=config.laplacian_spec)


def _build_spe_pe(config: SpectralFilterPEConfig, *, pos_dim: int, act: str, pos_domain=None) -> SpectralFilterPE:
    del pos_domain
    return SpectralFilterPE(
        config.num_eigenmodes,
        pos_dim=pos_dim,
        filter_type=config.filter_type,
        spe_mode=config.mode,
        num_filters=config.num_filters,
        hidden_dim=config.hidden,
        act=act,
        band_sigma_init=config.band_sigma_init,
        poly_order=config.poly_order,
        num_hops=config.num_hops,
    )
