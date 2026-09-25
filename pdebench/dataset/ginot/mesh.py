from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pdebench.dataset.ginot.types import MICRO_PUC_CANONICAL_MESH_ROWS, GinotRawDataset
from pdebench.dataset.ginot.utils import is_bumper_beam_raw, is_micro_puc_raw

QUADRATIC_TET_NODE_COUNT = 10
LINEAR_TET_NODE_COUNT = 4


@dataclass(frozen=True)
class LinearTetMesh:
    vertices: np.ndarray
    cells: np.ndarray
    nodal_values: np.ndarray | None = None
    boundary_vertices: np.ndarray | None = None


def downgrade_quadratic_tet_mesh(
    vertices: np.ndarray,
    cells: np.ndarray,
    *,
    nodal_values: np.ndarray | None = None,
    surface_node_ids: np.ndarray | None = None,
) -> LinearTetMesh:
    """Downgrade VTK C3D10 volume meshes to linear C3D4 by dropping mid-edge nodes."""
    vertices = np.asarray(vertices, dtype=np.float32)
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] < LINEAR_TET_NODE_COUNT:
        raise ValueError(f"Expected 2D cell array with at least 4 nodes per cell; got shape {cells.shape}.")
    if vertices.ndim != 2:
        raise ValueError(f"Expected vertices with shape [N, D]; got {vertices.shape}.")
    if nodal_values is not None and int(nodal_values.shape[0]) != int(vertices.shape[0]):
        raise ValueError(
            f"nodal_values rows ({nodal_values.shape[0]}) must match vertices ({vertices.shape[0]})."
        )

    linear_cells = cells[:, :LINEAR_TET_NODE_COUNT]
    if cells.shape[1] == QUADRATIC_TET_NODE_COUNT and np.any(cells[:, LINEAR_TET_NODE_COUNT:] < 0):
        raise ValueError("Quadratic tet cells must not use negative mid-edge node indices.")
    if np.any(linear_cells < 0) or np.any(linear_cells >= vertices.shape[0]):
        raise ValueError("Linear tet corner indices are out of range for the provided vertices.")

    corner_ids = np.unique(linear_cells.reshape(-1))
    old_to_new = np.full(int(vertices.shape[0]), -1, dtype=np.int64)
    old_to_new[corner_ids] = np.arange(int(corner_ids.size), dtype=np.int64)

    vertices_out = vertices[corner_ids]
    cells_out = old_to_new[linear_cells]
    values_out = None if nodal_values is None else np.asarray(nodal_values, dtype=np.float32)[corner_ids]

    boundary_out = None
    if surface_node_ids is not None:
        surface_ids = np.asarray(surface_node_ids, dtype=np.int64).reshape(-1)
        surface_ids = surface_ids[(surface_ids >= 0) & (surface_ids < vertices.shape[0])]
        if surface_ids.size:
            corner_mask = np.zeros(int(vertices.shape[0]), dtype=bool)
            corner_mask[corner_ids] = True
            surface_corners = np.unique(surface_ids[corner_mask[surface_ids]])
            boundary_out = vertices_out[old_to_new[surface_corners]]
        else:
            boundary_out = vertices_out

    return LinearTetMesh(
        vertices=vertices_out,
        cells=cells_out,
        nodal_values=values_out,
        boundary_vertices=boundary_out,
    )


def build_micro_puc_mesh_idx(sample_ids: np.ndarray) -> np.ndarray:
    """Map each Micro-PUC row to a canonical mesh index in [0, 9999]."""
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if sample_ids.ndim != 1:
        raise ValueError(f"Micro-PUC sample_ids must be 1D; got shape {sample_ids.shape}.")
    if sample_ids.shape[0] < MICRO_PUC_CANONICAL_MESH_ROWS:
        raise ValueError(
            f"Micro-PUC requires at least {MICRO_PUC_CANONICAL_MESH_ROWS} rows in sample_ids.npy; "
            f"found {sample_ids.shape[0]}."
        )
    sid_to_mesh = {int(sample_ids[row]): row for row in range(MICRO_PUC_CANONICAL_MESH_ROWS)}
    missing = sorted({int(sid) for sid in np.unique(sample_ids)} - set(sid_to_mesh))
    if missing:
        preview = missing[:8]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(
            "Micro-PUC sample_ids reference base geometry ids without a canonical mesh in "
            f"rows 0..{MICRO_PUC_CANONICAL_MESH_ROWS - 1}: {preview}{suffix}"
        )
    return np.array([sid_to_mesh[int(sid)] for sid in sample_ids], dtype=np.int32)


def cell_storage_row(raw: GinotRawDataset, idx: int) -> int:
    if raw.micro_puc_mesh_idx is not None:
        return int(raw.micro_puc_mesh_idx[int(idx)])
    return int(idx)


def lookup_cells(cells, row: int, *, num_samples: int):
    if isinstance(cells, np.ndarray):
        if cells.dtype == object:
            return cells[row]
        if cells.ndim == 3 and cells.shape[0] > row:
            return cells[row]
        return cells
    if isinstance(cells, (list, tuple)):
        if len(cells) == 0:
            return None
        if len(cells) == num_samples:
            return cells[row]
        if len(cells) > row:
            candidate = cells[row]
            if np.asarray(candidate).ndim >= 2:
                return candidate
        return cells
    if hasattr(cells, "__getitem__") and hasattr(cells, "__len__") and not isinstance(cells, (str, bytes)):
        try:
            if len(cells) == num_samples:
                return cells[row]
        except TypeError:
            pass
        return cells[row]
    return cells


def select_cells(raw: GinotRawDataset, idx: int):
    if raw.cells is None:
        return None
    row = cell_storage_row(raw, idx)
    return lookup_cells(raw.cells, row, num_samples=len(raw.query_points))


def cells_are_sample_indexed(raw: GinotRawDataset) -> bool:
    cells = raw.cells
    if cells is None:
        return False
    num_samples = len(raw.query_points)
    if isinstance(cells, np.ndarray):
        return bool(cells.dtype == object or cells.ndim == 3 or (cells.ndim == 2 and cells.shape[0] == num_samples))
    if isinstance(cells, (list, tuple)) and len(cells) > 0:
        return len(cells) == num_samples or np.asarray(cells[0]).ndim >= 2
    if hasattr(cells, "__getitem__") and hasattr(cells, "__len__") and not isinstance(cells, (str, bytes)):
        try:
            return len(cells) == num_samples
        except TypeError:
            return False
    return False


def _pad_cell_rows(rows: list[np.ndarray]) -> np.ndarray | None:
    if not rows:
        return None
    max_width = max(int(row.shape[0]) for row in rows)
    if max_width < 2:
        return None
    padded = np.full((len(rows), max_width), -1, dtype=np.int64)
    for row_idx, row in enumerate(rows):
        if row.ndim != 1 or row.shape[0] < 2:
            return None
        padded[row_idx, : row.shape[0]] = row
    return padded


def _cell_row_lengths(cells_np: np.ndarray) -> np.ndarray | None:
    if cells_np.ndim != 2 or cells_np.shape[1] < 2:
        return None
    lengths = np.empty((cells_np.shape[0],), dtype=np.int64)
    for row_idx, row in enumerate(cells_np):
        valid_mask = row >= 0
        valid_count = int(valid_mask.sum())
        if valid_count < 2:
            return None
        if not np.all(valid_mask[:valid_count]) or np.any(valid_mask[valid_count:]):
            return None
        lengths[row_idx] = valid_count
    return lengths


def normalize_cells(cells, num_nodes: int) -> np.ndarray | None:
    if cells is None:
        return None
    cells_np = np.asarray(cells, dtype=np.int64)
    if cells_np.ndim == 1:
        if cells_np.size < 3:
            return None
        cell_width = int(cells_np[0]) + 1
        if cell_width >= 3 and cells_np.size % cell_width == 0 and np.all(cells_np[::cell_width] == cell_width - 1):
            cells_np = cells_np.reshape(-1, cell_width)[:, 1:]
        else:
            parsed = []
            offset = 0
            while offset < cells_np.size:
                width = int(cells_np[offset])
                next_offset = offset + width + 1
                if width < 2 or next_offset > cells_np.size:
                    return None
                parsed.append(cells_np[offset + 1:next_offset])
                offset = next_offset
            if not parsed:
                return None
            cells_np = _pad_cell_rows([np.asarray(row, dtype=np.int64) for row in parsed])
            if cells_np is None:
                return None
    if cells_np.ndim != 2 or cells_np.shape[1] < 2:
        return None
    cells_np = cells_np.copy()
    if cells_np.shape[1] >= 3 and np.all(cells_np[:, 0] == cells_np.shape[1] - 1):
        cells_np = cells_np[:, 1:]
    lengths = _cell_row_lengths(cells_np)
    if lengths is None:
        return None
    valid_entries = cells_np[cells_np >= 0]
    if valid_entries.size == 0:
        return None
    if valid_entries.min() >= 1 and valid_entries.max() >= num_nodes:
        cells_np[cells_np >= 0] -= 1
        valid_entries = cells_np[cells_np >= 0]
    if valid_entries.min() < 0 or valid_entries.max() >= num_nodes:
        return None
    return cells_np


def build_edge_index_from_normalized_cells(
    cells_np: np.ndarray | None,
    num_nodes: int | None = None,
    *,
    perimeter_edges: bool = False,
) -> torch.Tensor:
    if cells_np is None:
        return torch.empty((2, 0), dtype=torch.long)

    from pdebench.dataset.plaid_core import build_edge_index_from_cell_cliques

    row_lengths = _cell_row_lengths(cells_np)
    if row_lengths is None:
        return torch.empty((2, 0), dtype=torch.long)

    # Clique path matches historical GINOT ordering exactly (shared with plaid_core).
    if not perimeter_edges and np.all(row_lengths == cells_np.shape[1]):
        return build_edge_index_from_cell_cliques(cells_np)

    if perimeter_edges and cells_np.shape[1] > 2 and np.all(row_lengths == cells_np.shape[1]):
        # Preserve historical GINOT ring / micro_puc perimeter edge ordering (unique-by-key)
        # for the released uniform-width path.
        edges = []
        cell_pairs = [(i, (i + 1) % cells_np.shape[1]) for i in range(cells_np.shape[1])]
        for i, j in cell_pairs:
            a = cells_np[:, i]
            b = cells_np[:, j]
            edges.append(np.stack([a, b], axis=0))
            edges.append(np.stack([b, a], axis=0))
        edge_np = np.concatenate(edges, axis=1).astype(np.int64, copy=False)
        if edge_np.size == 0:
            return torch.empty((2, 0), dtype=torch.long)
        if num_nodes is None:
            num_nodes = int(edge_np.max()) + 1
        keys = edge_np[0] * np.int64(num_nodes) + edge_np[1]
        _, unique_idx = np.unique(keys, return_index=True)
        edge_np = edge_np[:, np.sort(unique_idx)]
        return torch.from_numpy(edge_np).long()

    edge_pairs: list[tuple[int, int]] = []
    for row, row_len in zip(cells_np, row_lengths.tolist(), strict=False):
        valid = row[:row_len]
        if perimeter_edges and row_len > 2:
            for col_idx in range(row_len):
                a = int(valid[col_idx])
                b = int(valid[(col_idx + 1) % row_len])
                edge_pairs.append((a, b))
                edge_pairs.append((b, a))
            continue
        for start in range(row_len):
            for end in range(start + 1, row_len):
                a = int(valid[start])
                b = int(valid[end])
                edge_pairs.append((a, b))
                edge_pairs.append((b, a))
    if not edge_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    if num_nodes is None:
        num_nodes = max(max(a, b) for a, b in edge_pairs) + 1
    ordered_unique: list[tuple[int, int]] = []
    seen: set[int] = set()
    for a, b in edge_pairs:
        key = a * int(num_nodes) + b
        if key in seen:
            continue
        seen.add(key)
        ordered_unique.append((a, b))
    edge_np = np.asarray(ordered_unique, dtype=np.int64).T
    return torch.from_numpy(edge_np).long()

def build_edge_index_from_cells(cells, num_nodes: int) -> torch.Tensor:
    cells_np = normalize_cells(cells, num_nodes)
    if cells_np is None:
        raise ValueError("Could not normalize mesh cell connectivity; refusing to synthesize graph edges.")
    return build_edge_index_from_normalized_cells(cells_np, num_nodes=num_nodes)


def build_edge_index_for_sample(raw: GinotRawDataset, idx: int, pos: torch.Tensor) -> torch.Tensor:
    num_nodes = int(pos.shape[0])
    cells_np = normalize_cells(select_cells(raw, idx), num_nodes=num_nodes)
    if cells_np is None:
        raise ValueError(f"GINOT sample {idx} has no mesh cell connectivity; refusing to synthesize graph edges.")
    return build_edge_index_from_normalized_cells(
        cells_np,
        num_nodes=num_nodes,
        perimeter_edges=is_micro_puc_raw(raw) or is_bumper_beam_raw(raw),
    )


def build_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    from pdebench.dataset.plaid_core import build_edge_attr as _plaid_build_edge_attr

    return _plaid_build_edge_attr(pos, edge_index, bandwidth=None)


def topology_key_for_row(raw: GinotRawDataset, dataset_name: str, row_idx: int) -> int | None:
    if dataset_name == "poisson_structured":
        return 0
    if dataset_name == "micro_puc":
        if raw.micro_puc_mesh_idx is None:
            return None
        return int(raw.micro_puc_mesh_idx[int(row_idx)])
    if dataset_name == "micro_puc_fixed":
        if raw.micro_puc_source_geometry_ids is None:
            return None
        return int(raw.micro_puc_source_geometry_ids[int(row_idx)])
    return None


def unique_topology_representatives(raw: GinotRawDataset, dataset_name: str, num_rows: int) -> dict[int, int]:
    reps: dict[int, int] = {}
    for row_idx in range(int(num_rows)):
        key = topology_key_for_row(raw, dataset_name, row_idx)
        if key is None:
            continue
        if key not in reps:
            reps[key] = int(row_idx)
    return reps
