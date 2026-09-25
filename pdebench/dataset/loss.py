"""Physical channel-mean Rel-L2 for the unified PDEBench data pipeline.

Hard rules:
- Rel-L2 is always computed in physical space (decode pred/target first).
- One formula: per-graph channel-mean Rel-L2.
- When ``c_out == 1``, matches ``pdebench.utils.RelL2Loss``.
- Optional node mask via ``LossSpec.mask`` (same formula).

Supports:
- padded ``[B, N, C]`` (+ optional ``mask`` ``[B, N]``)
- packed ``[N_tot, C]`` (+ ``batch_index`` / ``cu_seqlens``)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from pdebench.dataset.sample import LossSpec

__all__ = [
    "channel_mean_rel_l2",
    "packed_per_graph_channel_mean_rel_l2",
    "packed_channel_mean_rel_l2",
    "field_rel_l2",
    "compute_loss",
    "compute_packed_loss",
    "compute_field_loss",
    "batch_index_from_cu_seqlens",
]


def batch_index_from_cu_seqlens(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Build packed ``batch_index`` and ``num_graphs`` from flash-attn ``cu_seqlens``."""
    cu = cu_seqlens.to(dtype=torch.long)
    if cu.ndim != 1 or cu.numel() < 2:
        raise ValueError(f"cu_seqlens must be 1D with length >= 2; got shape {tuple(cu.shape)}")
    lengths = (cu[1:] - cu[:-1]).tolist()
    num_graphs = len(lengths)
    if num_graphs == 0:
        return cu.new_empty((0,), dtype=torch.long), 0
    parts = [
        torch.full((int(n),), i, device=cu.device, dtype=torch.long)
        for i, n in enumerate(lengths)
    ]
    return torch.cat(parts, dim=0), num_graphs


def channel_mean_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-graph channel-mean Rel-L2 on padded ``[B, N, C]`` tensors.

    For each batch item and channel: ``||err||_2 / ||target||_2`` over nodes
    (optionally masked), then mean over channels, then mean over batch.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"channel_mean_rel_l2 expects padded [B, N, C]; got shape {tuple(pred.shape)}")

    diff_sq = (pred - target).square()
    target_sq = target.square()

    if mask is not None:
        mask_b = mask.to(device=pred.device, dtype=torch.bool)
        if mask_b.shape != pred.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask_b.shape)} must match pred batch/nodes {tuple(pred.shape[:2])}"
            )
        weight = mask_b.to(dtype=pred.dtype).unsqueeze(-1)
        diff_sq = diff_sq * weight
        target_sq = target_sq * weight

    # Sum over nodes → [B, C]
    err_l2 = diff_sq.sum(dim=1).sqrt()
    tgt_l2 = target_sq.sum(dim=1).sqrt().clamp_min(eps)
    if mask is not None:
        valid = mask_b.any(dim=1)
    else:
        valid = torch.ones(pred.shape[0], device=pred.device, dtype=torch.bool)
    channel_rel = err_l2 / tgt_l2
    per_graph = channel_rel.mean(dim=-1)[valid]
    if per_graph.numel() == 0:
        return torch.full((), float("nan"), device=pred.device, dtype=pred.dtype)
    return per_graph.mean()


def packed_per_graph_channel_mean_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-graph channel-mean Rel-L2 on packed ``[N_tot, C]``; returns ``[n_valid]``."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 2:
        raise ValueError(
            f"packed_per_graph_channel_mean_rel_l2 expects [N_tot, C]; got shape {tuple(pred.shape)}"
        )
    if num_graphs <= 0 or pred.numel() == 0:
        return pred.new_empty((0,))

    batch_index = batch_index.to(device=pred.device, dtype=torch.long)
    diff_sq = (pred - target).square()
    target_sq = target.square()
    if mask is not None:
        mask_b = mask.to(device=pred.device, dtype=torch.bool)
        if mask_b.shape != pred.shape[:1]:
            raise ValueError(f"mask shape {tuple(mask_b.shape)} must match N_tot={pred.shape[0]}")
        weight = mask_b.to(dtype=pred.dtype).unsqueeze(-1)
        diff_sq = diff_sq * weight
        target_sq = target_sq * weight
        node_weight = mask_b.to(dtype=pred.dtype)
    else:
        node_weight = torch.ones(pred.shape[0], device=pred.device, dtype=pred.dtype)

    c_out = pred.shape[-1]
    per_graph_diff_sq = torch.zeros((num_graphs, c_out), device=pred.device, dtype=diff_sq.dtype)
    per_graph_target_sq = torch.zeros((num_graphs, c_out), device=pred.device, dtype=target_sq.dtype)
    per_graph_diff_sq.index_add_(0, batch_index, diff_sq)
    per_graph_target_sq.index_add_(0, batch_index, target_sq)
    counts = torch.zeros((num_graphs,), device=pred.device, dtype=pred.dtype)
    counts.index_add_(0, batch_index, node_weight)
    valid = counts > 0
    channel_rel = per_graph_diff_sq.sqrt() / per_graph_target_sq.sqrt().clamp_min(eps)
    return channel_rel.mean(dim=-1)[valid]


def packed_channel_mean_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-graph channel-mean Rel-L2 on packed ``[N_tot, C]`` tensors; returns mean over graphs."""
    per_graph = packed_per_graph_channel_mean_rel_l2(
        pred,
        target,
        batch_index=batch_index,
        num_graphs=num_graphs,
        mask=mask,
        eps=eps,
    )
    if per_graph.numel() == 0:
        return torch.full((), float("nan"), device=pred.device, dtype=pred.dtype)
    return per_graph.mean()


def field_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    batch_index: Optional[torch.Tensor] = None,
    num_graphs: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dispatch channel-mean Rel-L2 for padded or packed layouts."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    if pred.ndim == 3:
        return channel_mean_rel_l2(pred, target, mask=mask, eps=eps)

    if pred.ndim != 2:
        raise ValueError(f"field_rel_l2 expects [B, N, C] or [N_tot, C]; got shape {tuple(pred.shape)}")

    if cu_seqlens is not None:
        batch_index, inferred_graphs = batch_index_from_cu_seqlens(cu_seqlens)
        if num_graphs is None:
            num_graphs = inferred_graphs
    if batch_index is None:
        batch_index = torch.zeros(pred.shape[0], device=pred.device, dtype=torch.long)
        num_graphs = 1 if num_graphs is None else int(num_graphs)
    elif num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0

    return packed_channel_mean_rel_l2(
        pred,
        target,
        batch_index=batch_index,
        num_graphs=int(num_graphs),
        mask=mask,
        eps=eps,
    )


def _resolve_mask(
    loss_spec: LossSpec,
    masks: Optional[Mapping[str, torch.Tensor]],
) -> Optional[torch.Tensor]:
    if loss_spec.mask is None:
        return None
    if masks is None or loss_spec.mask not in masks:
        raise KeyError(f"LossSpec.mask={loss_spec.mask!r} missing from masks")
    return masks[loss_spec.mask]


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_normalizer: Any,
    loss_spec: LossSpec,
    masks: Optional[Mapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode to physical space, then apply channel-mean Rel-L2 (+ optional mask)."""
    return compute_field_loss(pred, target, y_normalizer, loss_spec, masks=masks)


def compute_packed_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_normalizer: Any,
    loss_spec: LossSpec,
    batch_index: torch.Tensor,
    num_graphs: int,
    masks: Optional[Mapping[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode packed preds/targets, then packed channel-mean Rel-L2."""
    return compute_field_loss(
        pred,
        target,
        y_normalizer,
        loss_spec,
        masks=masks,
        batch_index=batch_index,
        num_graphs=num_graphs,
    )


def compute_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_normalizer: Any,
    loss_spec: Optional[LossSpec] = None,
    masks: Optional[Mapping[str, torch.Tensor]] = None,
    *,
    batch_index: Optional[torch.Tensor] = None,
    num_graphs: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Decode both sides to physical space, then ``field_rel_l2`` (padded or packed)."""
    if loss_spec is None:
        loss_spec = LossSpec()
    yn = y_normalizer.to(pred.device) if hasattr(y_normalizer, "to") else y_normalizer
    yh = yn.decode(pred)
    y = yn.decode(target)
    mask = _resolve_mask(loss_spec, masks)
    return field_rel_l2(
        yh,
        y,
        mask=mask,
        batch_index=batch_index,
        num_graphs=num_graphs,
        cu_seqlens=cu_seqlens,
        eps=eps,
    )
