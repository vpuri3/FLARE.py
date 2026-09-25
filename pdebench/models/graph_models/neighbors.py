"""Pairwise knn / ball-query helpers for graph operators."""
from __future__ import annotations

import torch


@torch.compiler.disable
def knn_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Return edge_index [2, E] for knn among rows of pos [N, D] (exclude self).

    Disabled under ``torch.compile`` because neighbor construction uses data-dependent
    indexing / topk and must not be functionalized into the training graph.
    """
    n = pos.shape[0]
    k = min(max(int(k), 1), max(n - 1, 1))
    if n <= 1:
        return pos.new_empty((2, 0), dtype=torch.long)
    dist = torch.cdist(pos.detach(), pos.detach())
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    dist = dist.masked_fill(eye, float("inf"))
    nbr = dist.topk(k, largest=False).indices  # [N, k]
    src = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(nbr).reshape(-1)
    dst = nbr.reshape(-1)
    return torch.stack([src, dst], dim=0)
