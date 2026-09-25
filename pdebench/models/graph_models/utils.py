"""Shared graph-model helpers (node inputs, activations, norms)."""
from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "graph_node_input",
    "_make_activation",
    "_make_norm",
    "_norm_type_from_rmsnorm",
    "_region_grid",
]


def graph_node_input(pos: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
    """Concatenate spatial coordinates with optional per-node feature channels."""
    if feats is None or feats.numel() == 0:
        return pos
    if feats.ndim != 2:
        raise ValueError(f"feats must be [N, C_f], got shape {tuple(feats.shape)}.")
    if feats.shape[0] != pos.shape[0]:
        raise ValueError(
            f"feats and pos must share node count, got {feats.shape[0]} and {pos.shape[0]}."
        )
    if feats.shape[-1] == 0:
        return pos
    return torch.cat([pos, feats], dim=-1)


def _make_activation(name: str):
    act_name = str(name).lower()
    if act_name in {"relu"}:
        return nn.ReLU()
    if act_name in {"elu"}:
        return nn.ELU()
    if act_name in {"leaky", "leaky_relu", "lrelu"}:
        return nn.LeakyReLU(0.05)
    if act_name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported MeshGraphNet activation '{name}'.")


def _norm_type_from_rmsnorm(rmsnorm: bool) -> str:
    return "RMSNorm" if rmsnorm else "LayerNorm"


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "LayerNorm":
        return nn.LayerNorm(dim)
    if norm_type == "RMSNorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"Unsupported MeshGraphNet norm_type '{norm_type}'.")


def _region_grid(coord: torch.Tensor, side: int) -> torch.Tensor:
    mins = coord.min(dim=0).values
    maxs = coord.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    lin = (torch.arange(side, dtype=coord.dtype, device=coord.device) + 0.5) / side
    grid_x, grid_y = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    return mins + grid * span
