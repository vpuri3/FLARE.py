"""Shared GraphPE base and packed-graph helpers for GLT positional encodings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = [
    "GraphPE",
    "PackedGraphIndices",
    "TopologyFeatures",
    "_normalize_pe",
    "_packed_graph_ids",
    "_packed_graph_indices",
    "_packed_node_count",
    "_resolve_packed_indices",
]

TopologyFeatures = (
    torch.Tensor
    | dict[str, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
    | list[torch.Tensor]
    | None
)


def _packed_node_count(cu_seqlens: torch.Tensor, num_total_nodes: int | None) -> int:
    return int(num_total_nodes if num_total_nodes is not None else cu_seqlens[-1].item())


@dataclass
class PackedGraphIndices:
    batch_index: torch.Tensor
    local_index: torch.Tensor
    lengths: torch.Tensor
    max_seqlen: int


def _resolve_packed_indices(
    cu_seqlens: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    *,
    batch_index: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> PackedGraphIndices:
    """Map packed nodes to graph ids without repeat_interleave (torch.compile friendly)."""
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    node_ids = torch.arange(int(num_nodes), device=device, dtype=torch.long)
    num_graphs = int(lengths.numel())
    if batch_index is not None:
        batch_index = batch_index.to(device=device, dtype=torch.long)
        local_index = node_ids - cu_seqlens[batch_index]
    else:
        batch_index = torch.searchsorted(cu_seqlens[1:].contiguous(), node_ids, right=True)
        batch_index = batch_index.clamp_max(num_graphs - 1)
        local_index = node_ids - cu_seqlens[batch_index]
    # Avoid lengths.max().item() under torch.compile (dynamo graph break). Callers that
    # need a Python max_seqlen must pass it explicitly (GLT already has it outside compile).
    if max_seqlen is not None:
        resolved_max_seqlen = int(max_seqlen)
    elif lengths.numel() == 0:
        resolved_max_seqlen = 0
    else:
        resolved_max_seqlen = 0
    return PackedGraphIndices(batch_index, local_index, lengths, resolved_max_seqlen)


def _packed_graph_indices(
    cu_seqlens: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    max_seqlen: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    packed = _resolve_packed_indices(cu_seqlens, num_nodes, device, max_seqlen=max_seqlen)
    return packed.batch_index, packed.local_index, packed.lengths, packed.max_seqlen


def _packed_graph_ids(
    cu_seqlens: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = _resolve_packed_indices(cu_seqlens, num_nodes, device)
    return packed.batch_index, packed.lengths


def _normalize_pe(
    pe: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pe_norm_mode: str,
    batch_index: torch.Tensor | None = None,
) -> torch.Tensor:
    cu_seqlens = cu_seqlens.to(device=pe.device, dtype=torch.long)
    packed = _resolve_packed_indices(
        cu_seqlens,
        int(pe.shape[0]),
        pe.device,
        batch_index=batch_index,
        max_seqlen=0,
    )
    graph_ids = packed.batch_index
    lengths = packed.lengths

    if pe_norm_mode == "node_rms":
        rms = pe.square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return pe / rms

    if pe_norm_mode != "graph_rms":
        raise ValueError(f"GLT pe_norm_mode must be 'graph_rms' or 'node_rms'. Got {pe_norm_mode!r}.")

    sum_sq = torch.segment_reduce(
        pe.square(),
        reduce="sum",
        offsets=cu_seqlens,
        axis=0,
    )
    rms = (sum_sq / lengths.float().unsqueeze(-1)).sqrt().clamp_min(1e-6)
    return pe / rms[graph_ids].to(dtype=pe.dtype)

class GraphPE(nn.Module):
    """Base class for pluggable graph PE modules.

    Subclasses set ``self.out_dim`` in ``__init__`` and implement ``forward`` returning
    packed per-node PE features ``[N_tot, out_dim]``.
    """

    out_dim: int

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError
