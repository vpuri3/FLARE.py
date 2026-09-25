from __future__ import annotations

from typing import Any

import torch


def ginot_collate_fn(
    samples: list[dict[str, torch.Tensor]],
    pad_to_nodes: int | None = None,
    pad_to_boundary_nodes: int | None = None,
    use_flash_varlen: bool = False,
    include_padded_boundary: bool = True,
) -> dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("Cannot collate an empty GINOT batch.")

    space_dim = int(samples[0]["pos"].shape[-1])
    dtype = samples[0]["pos"].dtype
    device = samples[0]["pos"].device

    padded_pos = None
    padded_y = None
    mask = None
    padded_boundary_pos = None
    boundary_mask = None
    if not use_flash_varlen:
        max_nodes = max(int(sample["pos"].shape[0]) for sample in samples)
        max_boundary_nodes = max(int(sample["boundary_pos"].shape[0]) for sample in samples)
        if pad_to_nodes is not None:
            max_nodes = max(max_nodes, int(pad_to_nodes))
        if pad_to_boundary_nodes is not None:
            max_boundary_nodes = max(max_boundary_nodes, int(pad_to_boundary_nodes))
        target_dim = int(samples[0]["y"].shape[-1])
        y_dtype = samples[0]["y"].dtype

        padded_pos = torch.zeros((len(samples), max_nodes, space_dim), dtype=dtype, device=device)
        padded_y = torch.zeros((len(samples), max_nodes, target_dim), dtype=y_dtype, device=device)
        mask = torch.zeros((len(samples), max_nodes), dtype=torch.bool, device=device)
        padded_feats = None
        feats_dim = int(samples[0]["feats"].shape[-1]) if "feats" in samples[0] else 0
        if feats_dim > 0:
            padded_feats = torch.zeros((len(samples), max_nodes, feats_dim), dtype=dtype, device=device)
        padded_boundary_pos = torch.zeros((len(samples), max_boundary_nodes, space_dim), dtype=dtype, device=device)
        boundary_mask = torch.zeros((len(samples), max_boundary_nodes), dtype=torch.bool, device=device)

    flat_y_parts = []
    flat_pos_parts = []
    flat_feats_parts = []
    flat_boundary_pos_parts = []
    edge_index_parts = []
    edge_attr_parts = []
    batch_index_parts = []
    boundary_batch_parts = []
    ptr = [0]
    boundary_ptr = [0]
    node_lengths = []
    boundary_lengths = []
    sample_ids = []
    edge_cache_keys = []
    graph_cache_dirs = []
    laplacian_parts = []
    laplacian_eigval_parts = []
    free_mask_parts = []
    bc_mask_parts = []
    u_prescribed_parts = []
    node_offset = 0

    for graph_idx, sample in enumerate(samples):
        pos = sample["pos"]
        boundary_pos = sample["boundary_pos"]
        y = sample["y"]
        feats = sample.get("feats")
        num_nodes = int(pos.shape[0])
        num_boundary_nodes = int(boundary_pos.shape[0])
        if not use_flash_varlen:
            padded_pos[graph_idx, :num_nodes] = pos
            padded_y[graph_idx, :num_nodes] = y
            mask[graph_idx, :num_nodes] = True
            if padded_feats is not None:
                if feats is None:
                    raise ValueError("GINOT collate expected per-node feats for every sample in the batch.")
                padded_feats[graph_idx, :num_nodes] = feats
            padded_boundary_pos[graph_idx, :num_boundary_nodes] = boundary_pos
            boundary_mask[graph_idx, :num_boundary_nodes] = True
        node_lengths.append(num_nodes)
        boundary_lengths.append(num_boundary_nodes)
        flat_pos_parts.append(pos)
        if feats is not None and feats.numel() > 0:
            flat_feats_parts.append(feats)
        flat_boundary_pos_parts.append(boundary_pos)
        flat_y_parts.append(y)
        batch_index_parts.append(torch.full((pos.shape[0],), graph_idx, dtype=torch.long))
        boundary_batch_parts.append(torch.full((boundary_pos.shape[0],), graph_idx, dtype=torch.long))
        ptr.append(ptr[-1] + int(pos.shape[0]))
        boundary_ptr.append(boundary_ptr[-1] + int(boundary_pos.shape[0]))
        sample_ids.append(sample["sample_id"].reshape(()))
        edge_cache_keys.append(sample.get("edge_cache_key"))
        graph_cache_dirs.append(sample.get("graph_cache_dir"))
        if "laplacian_eig" in sample:
            laplacian_parts.append(sample["laplacian_eig"])
            if "laplacian_eigvals" in sample:
                laplacian_eigval_parts.append(sample["laplacian_eigvals"])
        if "free_mask" in sample:
            free_mask_parts.append(sample["free_mask"])
        if "bc_mask" in sample:
            bc_mask_parts.append(sample["bc_mask"])
        if "u_prescribed" in sample:
            u_prescribed_parts.append(sample["u_prescribed"])

        edge_index = sample["edge_index"]
        if edge_index.numel() > 0:
            edge_index_parts.append(edge_index + node_offset)
            edge_attr_parts.append(sample["edge_attr"])
        node_offset += int(pos.shape[0])

    if edge_index_parts:
        edge_index = torch.cat(edge_index_parts, dim=1)
        edge_attr = torch.cat(edge_attr_parts, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, space_dim + 1), dtype=dtype, device=device)

    batch = {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "flat_pos": torch.cat(flat_pos_parts, dim=0),
        "flat_boundary_pos": torch.cat(flat_boundary_pos_parts, dim=0),
        "flat_y": torch.cat(flat_y_parts, dim=0),
        "batch_index": torch.cat(batch_index_parts, dim=0),
        "boundary_batch_index": torch.cat(boundary_batch_parts, dim=0),
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "boundary_ptr": torch.tensor(boundary_ptr, dtype=torch.long),
        "sample_id": torch.stack(sample_ids),
        "num_graphs": len(samples),
    }
    if all(key is not None for key in edge_cache_keys):
        batch["graph_cache_key"] = (
            "ginot:"
            + "|".join(str(key) for key in edge_cache_keys)
            + ":nodes="
            + ",".join(str(length) for length in node_lengths)
        )
    if all(path is not None for path in graph_cache_dirs):
        unique_dirs = sorted(set(str(path) for path in graph_cache_dirs))
        if len(unique_dirs) != 1:
            raise ValueError(f"Expected one graph_cache_dir per batch, got {unique_dirs}.")
        batch["graph_cache_dir"] = unique_dirs[0]
    if laplacian_parts:
        if len(laplacian_parts) != len(samples):
            raise ValueError("Either every GINOT sample must provide laplacian_eig, or none may provide it.")
        batch["flat_laplacian_eig"] = torch.cat(laplacian_parts, dim=0)
        if laplacian_eigval_parts:
            if len(laplacian_eigval_parts) != len(samples):
                raise ValueError("Either every GINOT sample must provide laplacian_eigvals, or none may provide it.")
            batch["flat_laplacian_eigvals"] = torch.stack(laplacian_eigval_parts, dim=0)
    if not use_flash_varlen:
        batch.update(
            pos=padded_pos,
            boundary_pos=padded_boundary_pos,
            y=padded_y,
            mask=mask,
            boundary_mask=boundary_mask,
        )
        if padded_feats is not None:
            batch["feats"] = padded_feats
    if flat_feats_parts:
        if len(flat_feats_parts) != len(samples):
            raise ValueError("Either every GINOT sample must provide feats, or none may provide feats.")
        batch["flat_feats"] = torch.cat(flat_feats_parts, dim=0)
    if free_mask_parts:
        if len(free_mask_parts) != len(samples):
            raise ValueError("Either every GINOT sample must provide free_mask, or none may provide it.")
        batch["flat_free_mask"] = torch.cat(free_mask_parts, dim=0)
    if bc_mask_parts:
        if len(bc_mask_parts) != len(samples):
            raise ValueError("Either every GINOT sample must provide bc_mask, or none may provide it.")
        batch["flat_bc_mask"] = torch.cat(bc_mask_parts, dim=0)
    if u_prescribed_parts:
        if len(u_prescribed_parts) != len(samples):
            raise ValueError("Either every GINOT sample must provide u_prescribed, or none may provide it.")
        batch["flat_u_prescribed"] = torch.cat(u_prescribed_parts, dim=0)
    if use_flash_varlen and include_padded_boundary:
        max_boundary_nodes = max(int(sample["boundary_pos"].shape[0]) for sample in samples)
        if pad_to_boundary_nodes is not None:
            max_boundary_nodes = max(max_boundary_nodes, int(pad_to_boundary_nodes))
        padded_boundary_pos = torch.zeros((len(samples), max_boundary_nodes, space_dim), dtype=dtype, device=device)
        boundary_mask = torch.zeros((len(samples), max_boundary_nodes), dtype=torch.bool, device=device)
        for graph_idx, sample in enumerate(samples):
            boundary_pos = sample["boundary_pos"]
            num_boundary_nodes = int(boundary_pos.shape[0])
            padded_boundary_pos[graph_idx, :num_boundary_nodes] = boundary_pos
            boundary_mask[graph_idx, :num_boundary_nodes] = True
        batch["boundary_pos"] = padded_boundary_pos
        batch["boundary_mask"] = boundary_mask
    if use_flash_varlen:
        node_lengths_tensor = torch.tensor(node_lengths, dtype=torch.int32, device=device)
        boundary_lengths_tensor = torch.tensor(boundary_lengths, dtype=torch.int32, device=device)
        batch.update(
            use_flash_varlen=True,
            node_lengths=node_lengths_tensor,
            boundary_lengths=boundary_lengths_tensor,
            cu_seqlens=torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.cumsum(node_lengths_tensor, dim=0, dtype=torch.int32),
                ]
            ),
            boundary_cu_seqlens=torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.cumsum(boundary_lengths_tensor, dim=0, dtype=torch.int32),
                ]
            ),
            max_seqlen=int(max(node_lengths)),
            boundary_max_seqlen=int(max(boundary_lengths)),
        )
    return batch
