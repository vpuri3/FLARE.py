from __future__ import annotations

from typing import Any

import torch


def _ginot_node_tensors(batch: dict[str, Any], *, use_flat_pos: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    pos = batch["flat_pos"] if use_flat_pos else batch["pos"]
    feats = batch.get("flat_feats") if use_flat_pos else batch.get("feats")
    return pos, feats


def ginot_model_forward(cfg, model, batch: dict[str, Any]):
    if not isinstance(batch, dict):
        raise ValueError("GINOT batches must be dicts returned by ginot_collate_fn.")

    flag_source = model
    while True:
        next_source = getattr(flag_source, "_orig_mod", None)
        if next_source is None:
            next_source = getattr(flag_source, "module", None)
        if next_source is None or next_source is flag_source:
            break
        flag_source = next_source
    requires_edge_info = bool(getattr(model, "requires_edge_info", getattr(flag_source, "requires_edge_info", False)))

    model_type = cfg.model.model
    # Flat pos/feats graph operators (edges only when requires_edge_info).
    flat_graph_models = {
        "meshgraphnet",
        "rigno",
        "gito",
        "geo_transolver",
    }

    if requires_edge_info or model_type in {"rigno", "geo_transolver"}:
        use_flat_pos = bool(batch.get("use_flash_varlen")) or (
            model_type in flat_graph_models and "flat_pos" in batch
        )
        pos, feats = _ginot_node_tensors(batch, use_flat_pos=use_flat_pos)
        kwargs = {"pos": pos, "feats": feats}
        if batch.get("mask") is not None:
            kwargs["mask"] = batch["mask"]
        if batch.get("use_flash_varlen"):
            num_total_nodes = int(batch["cu_seqlens"][-1].item())
            pos, feats = _ginot_node_tensors(batch, use_flat_pos=True)
            kwargs.update(
                pos=pos,
                feats=feats,
                use_flash_varlen=True,
                cu_seqlens=batch["cu_seqlens"],
                max_seqlen=batch["max_seqlen"],
            )
            if model_type == "glt":
                kwargs["num_total_nodes"] = num_total_nodes
        if requires_edge_info:
            kwargs.update(edge_index=batch["edge_index"], edge_attr=batch["edge_attr"])
        if batch.get("sample_id") is not None:
            kwargs["sample_ids"] = batch["sample_id"]
        if model_type in {
            "glt",
            "rigno",
            "gito",
            "geo_transolver",
        } and batch.get("batch_index") is not None:
            kwargs["batch_index"] = batch["batch_index"]
        if model_type == "glt" and batch.get("flat_laplacian_eig") is not None:
            kwargs["topology_features"] = batch["flat_laplacian_eig"]
        if model_type == "glt" and batch.get("flat_laplacian_eigvals") is not None:
            kwargs["topology_eigenvalues"] = batch["flat_laplacian_eigvals"]
        yh = model(**kwargs)
    else:
        # Split-input models (Transolver, …): x=pos, f=feats when present.
        mask = batch.get("mask")
        if batch.get("use_flash_varlen"):
            pos, feats = _ginot_node_tensors(batch, use_flat_pos=True)
            if feats is not None:
                yh = model(
                    pos,
                    feats,
                    use_flash_varlen=True,
                    cu_seqlens=batch["cu_seqlens"],
                    max_seqlen=batch["max_seqlen"],
                )
            else:
                yh = model(
                    pos,
                    use_flash_varlen=True,
                    cu_seqlens=batch["cu_seqlens"],
                    max_seqlen=batch["max_seqlen"],
                )
        else:
            pos, feats = _ginot_node_tensors(batch, use_flat_pos=False)
            if feats is not None:
                yh = model(pos, feats, mask=mask) if mask is not None else model(pos, feats)
            else:
                yh = model(pos, mask=mask) if mask is not None else model(pos)

    if isinstance(yh, (tuple, list)):
        yh = yh[0]
    y = batch["y"] if "y" in batch else batch["flat_y"]
    if yh.ndim == 3:
        valid_mask = batch.get("mask")
        if valid_mask is None:
            yh = yh.reshape(-1, yh.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        else:
            yh = yh[valid_mask]
            y = y[valid_mask]
    else:
        y = batch.get("flat_y", y)
    return yh, y, batch["batch_index"], int(batch["num_graphs"])


def ginot_apply_deform_plate_boundary_conditions(
    yh: torch.Tensor,
    batch: dict[str, Any],
    *,
    y_normalizer,
) -> torch.Tensor:
    bc_mask = batch.get("flat_bc_mask")
    u_prescribed = batch.get("flat_u_prescribed")
    if bc_mask is None or u_prescribed is None:
        return yh
    prescribed = y_normalizer.encode(u_prescribed.to(device=yh.device, dtype=yh.dtype))
    return torch.where(bc_mask.unsqueeze(-1), prescribed, yh)


def ginot_postprocess_displacement(
    yh: torch.Tensor,
    batch: dict[str, Any],
    *,
    y_normalizer,
) -> torch.Tensor:
    if batch.get("flat_bc_mask") is None:
        return yh
    return ginot_apply_deform_plate_boundary_conditions(yh, batch, y_normalizer=y_normalizer)


def ginot_per_graph_free_node_rel_l2(
    yh: torch.Tensor,
    y: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    free_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if num_graphs <= 0 or yh.numel() == 0:
        return torch.full((1,), float("nan"), device=yh.device)
    batch_index = batch_index.to(device=yh.device, dtype=torch.long)
    free_mask = free_mask.to(device=yh.device, dtype=torch.bool)
    diff_sq = (yh - y).square()[free_mask]
    target_sq = y.square()[free_mask]
    if diff_sq.numel() == 0:
        return torch.full((1,), float("nan"), device=yh.device)

    graph_ids = batch_index[free_mask]
    per_graph_diff_sq = torch.zeros((num_graphs,), device=yh.device, dtype=diff_sq.dtype)
    per_graph_target_sq = torch.zeros((num_graphs,), device=yh.device, dtype=target_sq.dtype)
    per_graph_diff_sq.index_add_(0, graph_ids, diff_sq.sum(dim=-1))
    per_graph_target_sq.index_add_(0, graph_ids, target_sq.sum(dim=-1))
    counts = torch.zeros((num_graphs,), device=yh.device, dtype=yh.dtype)
    counts.index_add_(0, graph_ids, torch.ones_like(graph_ids, dtype=yh.dtype))
    valid_graphs = counts > 0
    diff_l2 = per_graph_diff_sq.sqrt()
    target_l2 = per_graph_target_sq.sqrt()
    rel = diff_l2 / (target_l2 + eps)
    return rel[valid_graphs]


def ginot_per_graph_channel_rel_l2(
    yh: torch.Tensor,
    y: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Packed channel-mean Rel-L2 per graph (unified ``packed_per_graph_channel_mean_rel_l2``)."""
    from pdebench.dataset.loss import packed_per_graph_channel_mean_rel_l2

    if num_graphs <= 0 or yh.numel() == 0:
        return torch.full((1,), float("nan"), device=yh.device)
    return packed_per_graph_channel_mean_rel_l2(
        yh,
        y,
        batch_index=batch_index,
        num_graphs=num_graphs,
        eps=eps,
    )
