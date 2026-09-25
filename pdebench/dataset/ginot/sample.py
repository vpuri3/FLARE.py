from __future__ import annotations

import numpy as np
import torch

from pdebench.dataset.ginot.bumper_beam import encode_bumper_beam_target
from pdebench.dataset.ginot.deform_plate import (
    encode_deform_plate_boundary_positions,
    encode_deform_plate_positions,
)
from pdebench.dataset.ginot.features import encode_deform_plate_boundary_tensors, encode_sample_feats
from pdebench.dataset.ginot.mesh import (
    build_edge_attr,
    build_edge_index_for_sample,
    build_edge_index_from_normalized_cells,
    normalize_cells,
    select_cells,
)
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer
from pdebench.dataset.ginot.utils import (
    as_float_array,
    is_bumper_beam_raw,
    is_deform_plate_raw,
    is_micro_puc_fixed_raw,
    is_micro_puc_raw,
)


def read_raw_sample_arrays(raw: GinotRawDataset, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_np = as_float_array(raw.query_points[idx], dims=raw.space_dim)
    boundary_np = as_float_array(raw.point_clouds[idx], dims=raw.space_dim)
    y_np = as_float_array(raw.targets[idx])
    if pos_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"GINOT sample {idx} has inconsistent point counts: pos={pos_np.shape}, y={y_np.shape}.")
    return pos_np, boundary_np, y_np


def encode_raw_sample(
    raw: GinotRawDataset,
    idx: int,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    *,
    to_cpu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_np, boundary_np, y_np = read_raw_sample_arrays(raw, idx)
    if is_bumper_beam_raw(raw):
        pos = pos_normalizer.encode(torch.from_numpy(pos_np).float())
        boundary_pos = boundary_pos_normalizer.encode(torch.from_numpy(boundary_np).float())
        y = encode_bumper_beam_target(raw, int(idx), pos_normalizer)
    elif is_deform_plate_raw(raw):
        pos = encode_deform_plate_positions(raw, int(idx), pos_normalizer)
        boundary_pos = encode_deform_plate_boundary_positions(raw, int(idx), boundary_pos_normalizer)
        y = y_normalizer.encode(torch.from_numpy(y_np).float())
    else:
        pos = pos_normalizer.encode(torch.from_numpy(pos_np).float())
        boundary_pos = boundary_pos_normalizer.encode(torch.from_numpy(boundary_np).float())
        y = torch.from_numpy(y_np).float()
        if raw.normalize_targets:
            y = y_normalizer.encode(y)
    if to_cpu:
        pos = pos.cpu()
        boundary_pos = boundary_pos.cpu()
        y = y.cpu()
    return pos, boundary_pos, y


def build_periodic_x_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Minimum-image edge attributes for micro_puc_fixed on the [0, 1)^2 torus."""
    if edge_index.numel() == 0:
        return torch.empty((0, pos.shape[-1] + 1), dtype=pos.dtype, device=pos.device)
    src, dst = edge_index
    diff = pos[src] - pos[dst]
    diff = diff.clone()
    for axis in range(min(2, diff.shape[-1])):
        diff[:, axis] = torch.remainder(diff[:, axis] + 0.5, 1.0) - 0.5
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    scale = max(float(torch.median(length.squeeze(-1)).item()), 1e-8)
    return torch.cat([diff / scale, length / scale], dim=-1)


def build_sample_edge_tensors(
    raw: GinotRawDataset,
    idx: int,
    pos: torch.Tensor,
    *,
    dataset_name: str | None = None,
    shared_edge_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray | None]:
    if shared_edge_index is not None:
        edge_index = shared_edge_index
    elif is_micro_puc_fixed_raw(raw):
        edge_index = build_edge_index_for_sample(raw, idx, pos)
    else:
        cells_np = normalize_cells(select_cells(raw, int(idx)), num_nodes=int(pos.shape[0]))
        if cells_np is None and raw.precomputed_edge_index is not None:
            edge_np = np.asarray(raw.precomputed_edge_index[int(idx)], dtype=np.int64)
            if edge_np.ndim != 2 or edge_np.shape[0] != 2 or edge_np.shape[1] == 0:
                shape = getattr(edge_np, "shape", None)
                raise ValueError(f"GINOT sample {idx} has invalid precomputed_edge_index shape {shape}.")
            edge_index = torch.from_numpy(edge_np).long()
            edge_attr = build_edge_attr(pos, edge_index).float()
            return edge_index, edge_attr, None
        if cells_np is None:
            raise ValueError(f"GINOT sample {idx} has no mesh cell connectivity; refusing to synthesize graph edges.")
        edge_index = build_edge_index_from_normalized_cells(
            cells_np,
            num_nodes=int(pos.shape[0]),
            perimeter_edges=is_micro_puc_raw(raw) or is_bumper_beam_raw(raw),
        )

    if is_micro_puc_fixed_raw(raw):
        edge_attr = build_periodic_x_edge_attr(pos, edge_index).float()
        cells_np = normalize_cells(select_cells(raw, int(idx)), num_nodes=int(pos.shape[0]))
    else:
        edge_attr = build_edge_attr(pos, edge_index).float()
        cells_np = normalize_cells(select_cells(raw, int(idx)), num_nodes=int(pos.shape[0]))
    return edge_index, edge_attr, cells_np


def build_graph_sample_dict(
    raw: GinotRawDataset,
    idx: int,
    pos_normalizer: StandardNormalizer,
    boundary_pos_normalizer: StandardNormalizer,
    y_normalizer: StandardNormalizer,
    *,
    feats_normalizer: StandardNormalizer | None = None,
    dataset_name: str | None = None,
    shared_edge_index: torch.Tensor | None = None,
    to_cpu: bool = False,
) -> dict[str, torch.Tensor]:
    pos, boundary_pos, y = encode_raw_sample(
        raw,
        idx,
        pos_normalizer,
        boundary_pos_normalizer,
        y_normalizer,
        to_cpu=to_cpu,
    )
    edge_index, edge_attr, _ = build_sample_edge_tensors(
        raw,
        idx,
        pos,
        dataset_name=dataset_name,
        shared_edge_index=shared_edge_index,
    )
    if to_cpu:
        edge_index = edge_index.cpu()
        edge_attr = edge_attr.float().cpu()
    sample = {
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "boundary_pos": boundary_pos,
        "y": y,
        "sample_id": torch.tensor(int(idx), dtype=torch.long),
    }
    if feats_normalizer is not None or is_deform_plate_raw(raw):
        sample["feats"] = encode_sample_feats(
            raw,
            int(idx),
            feats_normalizer,
            int(pos.shape[0]),
            to_cpu=to_cpu,
            y_normalizer=y_normalizer,
        )
    if is_deform_plate_raw(raw):
        sample.update(encode_deform_plate_boundary_tensors(raw, int(idx), to_cpu=to_cpu))
    return sample
