"""Shared PLAID CGNS parsing utilities for MGN and GINOT loaders."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from glob import glob
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

__all__ = [
    "PlaidParsedSample",
    "PlaidReadmeMeta",
    "build_edge_attr",
    "build_edge_index_from_cells",
    "distance_to_boundary",
    "extract_boundary_edges",
    "extract_boundary_tag_ids",
    "extract_node_types_and_boundary_ids",
    "iter_cgns_nodes",
    "load_plaid_readme_meta",
    "parse_frontmatter_yaml",
    "parse_plaid_sample_bytes",
    "resolve_target_fields",
    "split_labeled_train_test",
    "target_scalar_names_for_dataset",
    "build_plaid_mgn_node_features",
    "build_plaid_benchmark_node_features",
    "build_plaid_elpl_sdf_features",
    "list_plaid_mesh_times",
    "list_plaid_parquet_paths",
    "parse_plaid_elpl_temporal_graphs",
    "parse_plaid_elpl_trajectory",
]


@dataclass(frozen=True)
class PlaidParsedSample:
    pos: np.ndarray
    cells: np.ndarray
    targets: np.ndarray | None
    target_scalars: np.ndarray | None
    boundary_pos: np.ndarray
    scalar_values: np.ndarray
    boundary_tag_ids: dict[str, np.ndarray]
    boundary_ids: np.ndarray
    boundary_tags: tuple[str, ...]
    node_type_ids: np.ndarray
    boundary_distance: np.ndarray
    target_fields: tuple[str, ...]
    space_dim: int


@dataclass(frozen=True)
class PlaidReadmeMeta:
    scalar_names: tuple[str, ...]
    out_fields: tuple[str, ...]
    split_map: dict[str, list[int]]


def parse_frontmatter_yaml(readme_path: str) -> dict:
    with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    return yaml.safe_load(text[4:end]) or {}


def load_plaid_readme_meta(dataset_dir: str) -> PlaidReadmeMeta:
    readme_path = f"{dataset_dir}/README.md"
    fm = parse_frontmatter_yaml(readme_path)
    desc = ((fm.get("dataset_info") or {}).get("description") or {})
    split_raw = desc.get("split") or {}
    split_map = {str(name): [int(i) for i in ids] for name, ids in split_raw.items()}
    scalar_names = tuple(str(x) for x in (desc.get("in_scalars_names") or []))
    out_fields = tuple(str(x) for x in (desc.get("out_fields_names") or []))
    return PlaidReadmeMeta(
        scalar_names=scalar_names,
        out_fields=out_fields,
        split_map=split_map,
    )


def resolve_target_fields(benchmark_fields: tuple[str, ...], out_fields: list[str] | tuple[str, ...]) -> list[str]:
    target_fields = list(benchmark_fields)
    out_set = set(out_fields)
    missing = [name for name in target_fields if name not in out_set]
    if missing:
        raise ValueError(
            f"Target field(s) {missing} are not available. Available out fields: {list(out_fields)}"
        )
    return target_fields


def iter_cgns_nodes(node, path: str = ""):
    if not isinstance(node, list) or len(node) != 4:
        return
    name, value, children, node_type = node
    current_path = f"{path}/{name}"
    yield current_path, name, value, node_type
    if isinstance(children, list):
        for child in children:
            yield from iter_cgns_nodes(child, current_path)


def extract_node_types_and_boundary_ids(
    cgns_nodes: list[tuple[str, str, Any, str]],
    num_nodes: int,
):
    tag_label_map = {
        "Airfoil": 2,
        "Holes": 2,
        "Top": 2,
        "Inlet": 4,
        "Bottom": 4,
        "Inflow": 4,
        "Ext_bound": 6,
        "Outflow": 6,
        "Intrado": 1,
        "Extrado": 3,
        "Periodic_1": 5,
        "Periodic_2": 7,
    }

    labels = np.zeros((num_nodes,), dtype=np.int64)
    boundary_ids = []
    boundary_tags = []

    for path, _, value, node_type in cgns_nodes:
        if node_type != "IndexArray_t" or not path.endswith("/PointList"):
            continue
        parts = path.split("/")
        if len(parts) < 2:
            continue
        bc_name = parts[-2]

        idx = np.asarray(value, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            continue
        if idx.min() >= 1:
            idx = idx - 1
        idx = idx[(idx >= 0) & (idx < num_nodes)]
        if idx.size == 0:
            continue

        boundary_ids.append(idx)
        boundary_tags.append(bc_name)
        labels[idx] = tag_label_map.get(bc_name, 0)

    if len(boundary_ids) == 0:
        return labels, np.array([], dtype=np.int64), []

    boundary = np.unique(np.concatenate(boundary_ids, axis=0))
    return labels, boundary, sorted(set(boundary_tags))


def extract_boundary_tag_ids(
    cgns_nodes: list[tuple[str, str, Any, str]],
    num_nodes: int,
) -> dict[str, np.ndarray]:
    tag_ids: dict[str, list[np.ndarray]] = {}
    for path, _, value, node_type in cgns_nodes:
        if node_type != "IndexArray_t" or not path.endswith("/PointList"):
            continue
        parts = path.split("/")
        if len(parts) < 2:
            continue
        tag_name = parts[-2]
        idx = np.asarray(value, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            continue
        if idx.min() >= 1:
            idx = idx - 1
        idx = idx[(idx >= 0) & (idx < num_nodes)]
        if idx.size == 0:
            continue
        tag_ids.setdefault(tag_name, []).append(idx)

    return {
        tag_name: np.unique(np.concatenate(chunks, axis=0)).astype(np.int64, copy=False)
        for tag_name, chunks in tag_ids.items()
    }


def extract_boundary_edges(cells: np.ndarray) -> np.ndarray:
    if cells.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    num_nodes_per_cell = cells.shape[1]
    edges = []
    for i in range(num_nodes_per_cell):
        j = (i + 1) % num_nodes_per_cell
        e = cells[:, [i, j]]
        e = np.sort(e, axis=1)
        edges.append(e)
    edges = np.concatenate(edges, axis=0)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def distance_to_boundary(pos: np.ndarray, cells: np.ndarray) -> np.ndarray:
    boundary_edges = extract_boundary_edges(cells)
    if boundary_edges.shape[0] == 0:
        return np.zeros((pos.shape[0], 1), dtype=np.float32)

    pos_t = torch.from_numpy(pos).float()
    seg_a = pos_t[torch.from_numpy(boundary_edges[:, 0]).long()]
    seg_b = pos_t[torch.from_numpy(boundary_edges[:, 1]).long()]
    seg_ab = seg_b - seg_a
    seg_ab_sq = torch.sum(seg_ab * seg_ab, dim=-1).clamp_min(1e-12)

    chunk = 2048
    mins = []
    for start in range(0, pos_t.shape[0], chunk):
        p = pos_t[start : start + chunk]
        ap = p[:, None, :] - seg_a[None, :, :]
        t = torch.sum(ap * seg_ab[None, :, :], dim=-1) / seg_ab_sq[None, :]
        t = t.clamp_(0.0, 1.0)
        proj = seg_a[None, :, :] + t[..., None] * seg_ab[None, :, :]
        d = torch.sqrt(torch.sum((p[:, None, :] - proj) ** 2, dim=-1).clamp_min(1e-20))
        mins.append(d.min(dim=1).values)

    dist = torch.cat(mins, dim=0).unsqueeze(-1)
    return dist.cpu().numpy().astype(np.float32, copy=False)


def build_edge_index_from_cells(cells: np.ndarray) -> torch.Tensor:
    """Build a bidirectional edge index from mesh cell connectivity (cell-ring edges)."""
    if cells.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    num_nodes_per_cell = cells.shape[1]
    ring_edges = []
    for i in range(num_nodes_per_cell):
        j = (i + 1) % num_nodes_per_cell
        ring_edges.append(cells[:, [i, j]])

    undirected = np.concatenate(ring_edges, axis=0)
    src = np.minimum(undirected[:, 0], undirected[:, 1])
    dst = np.maximum(undirected[:, 0], undirected[:, 1])
    undirected = np.unique(np.stack([src, dst], axis=1), axis=0)

    directed = np.concatenate([undirected, undirected[:, [1, 0]]], axis=0)
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    return torch.from_numpy(directed.T.astype(np.int64, copy=False))


def build_edge_index_from_cell_cliques(cells: np.ndarray) -> torch.Tensor:
    """Bidirectional edges for every pair of nodes within each cell (GINOT default)."""
    from itertools import combinations

    if cells.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = []
    for i, j in combinations(range(cells.shape[1]), 2):
        a = cells[:, i]
        b = cells[:, j]
        edges.append(np.stack([a, b], axis=0))
        edges.append(np.stack([b, a], axis=0))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_np = np.concatenate(edges, axis=1).astype(np.int64, copy=False)
    if edge_np.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    num_nodes = int(edge_np.max()) + 1
    keys = edge_np[0] * np.int64(num_nodes) + edge_np[1]
    _, unique_idx = np.unique(keys, return_index=True)
    edge_np = edge_np[:, np.sort(unique_idx)]
    return torch.from_numpy(edge_np).long()


def build_edge_attr(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    bandwidth: float | None = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0, pos.shape[-1] + 1), dtype=pos.dtype)

    src, dst = edge_index
    diff = pos[src] - pos[dst]
    length = torch.linalg.norm(diff, dim=-1, keepdim=True)
    if bandwidth is None:
        scale = float(torch.median(length.squeeze(-1)).item())
    else:
        scale = float(bandwidth)
    scale = max(scale, 1e-8)
    return torch.cat([diff / scale, length / scale], dim=-1)


def _boundary_positions(pos: np.ndarray, cells: np.ndarray, boundary_ids: np.ndarray) -> np.ndarray:
    if boundary_ids.size > 0:
        return pos[boundary_ids].astype(np.float32, copy=False)
    perimeter = extract_boundary_edges(cells)
    if perimeter.size == 0:
        return pos.astype(np.float32, copy=False)
    node_ids = np.unique(perimeter.reshape(-1))
    return pos[node_ids].astype(np.float32, copy=False)


def _select_mesh_entry(meshes: dict) -> Any:
    if not meshes:
        raise ValueError("PLAID sample has no mesh entries.")
    times = sorted(meshes.keys(), key=float)
    return meshes[times[-1]]


def _collect_cgns_arrays(cgns_nodes: list[tuple[str, str, Any, str]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path, _, value, node_type in cgns_nodes:
        if node_type == "DataArray_t":
            arrays[path] = np.asarray(value)
    return arrays


def _mesh_has_geometry(mesh_entry: Any) -> bool:
    for path, _, _, node_type in iter_cgns_nodes(mesh_entry):
        if node_type == "DataArray_t" and path.endswith("/CoordinateX"):
            return True
    return False


def _extract_point_field_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    point_arrays: dict[str, np.ndarray] = {}
    for path, arr in arrays.items():
        if "/PointData/" not in path and "/VertexFields/" not in path:
            continue
        field_name = path.split("/")[-1]
        if field_name == "GridLocation":
            continue
        point_arrays[field_name] = np.asarray(arr)
    return point_arrays


def _mesh_has_point_fields(mesh_entry: Any, target_fields: list[str] | tuple[str, ...]) -> bool:
    arrays = _collect_cgns_arrays(list(iter_cgns_nodes(mesh_entry)))
    point_arrays = _extract_point_field_arrays(arrays)
    return all(field in point_arrays for field in target_fields)


def _select_plaid_mesh_entries(
    meshes: dict,
    target_fields: list[str] | tuple[str, ...],
) -> tuple[Any, Any]:
    """Return (geometry_mesh, field_mesh) entries from a possibly time-dependent sample."""
    if not meshes:
        raise ValueError("PLAID sample has no mesh entries.")

    ordered = sorted(meshes.keys(), key=float)
    geom_key = next((key for key in ordered if _mesh_has_geometry(meshes[key])), None)
    if geom_key is None:
        raise ValueError("PLAID sample has no mesh entry with coordinates and connectivity.")

    field_key = next(
        (key for key in reversed(ordered) if _mesh_has_point_fields(meshes[key], target_fields)),
        geom_key,
    )
    return meshes[geom_key], meshes[field_key]


_PLAID_TARGET_SCALAR_NAMES: dict[str, tuple[str, ...]] = {
    "plaid_tensile2d": ("max_von_mises", "max_U2_top", "max_sig22_top"),
    "plaid_hyperelasticity": ("effective_energy",),
}


def target_scalar_names_for_dataset(dataset_name: str) -> tuple[str, ...]:
    """Vi-Transf / PLAID benchmark output scalars for a canonical dataset name."""
    return _PLAID_TARGET_SCALAR_NAMES.get(str(dataset_name), ())


def parse_plaid_sample_bytes(
    sample_bytes: bytes,
    *,
    target_fields: list[str] | tuple[str, ...],
    scalar_names: list[str] | tuple[str, ...],
    dataset_name: str,
    sample_idx: int,
    require_targets: bool = True,
) -> PlaidParsedSample | None:
    sample = pickle.loads(sample_bytes)
    target_names = list(target_fields)
    geom_mesh_entry, field_mesh_entry = _select_plaid_mesh_entries(sample["meshes"], target_names)
    cgns_nodes = list(iter_cgns_nodes(geom_mesh_entry))
    arrays = _collect_cgns_arrays(cgns_nodes)
    field_arrays = (
        arrays
        if field_mesh_entry is geom_mesh_entry
        else _collect_cgns_arrays(list(iter_cgns_nodes(field_mesh_entry)))
    )

    def find_array(mesh_arrays: dict[str, np.ndarray], suffix: str):
        for path, arr in mesh_arrays.items():
            if path.endswith(suffix):
                return arr
        return None

    x_coord = find_array(arrays, "/CoordinateX")
    y_coord = find_array(arrays, "/CoordinateY")
    z_coord = find_array(arrays, "/CoordinateZ")
    if x_coord is None or y_coord is None:
        raise ValueError(f"Could not locate coordinates in PLAID sample {sample_idx} ({dataset_name}).")

    if z_coord is None:
        pos = np.stack([x_coord, y_coord], axis=-1).astype(np.float32, copy=False)
    else:
        pos = np.stack([x_coord, y_coord, z_coord], axis=-1).astype(np.float32, copy=False)

    element_connectivity = None
    element_parent_name = None
    for path, arr in arrays.items():
        if path.endswith("/ElementConnectivity"):
            element_connectivity = arr
            element_parent_name = path.split("/")[-2]
            break
    if element_connectivity is None:
        raise ValueError(f"Could not locate ElementConnectivity in PLAID sample {sample_idx} ({dataset_name}).")

    num_nodes_per_cell = 3
    if element_parent_name is not None:
        match = re.search(r"_(\d+)$", element_parent_name)
        if match is not None:
            num_nodes_per_cell = int(match.group(1))
    if element_connectivity.size % num_nodes_per_cell != 0:
        if element_connectivity.size % 3 == 0:
            num_nodes_per_cell = 3
        elif element_connectivity.size % 4 == 0:
            num_nodes_per_cell = 4
        else:
            raise ValueError(
                f"Invalid element connectivity size={element_connectivity.size} "
                f"in sample {sample_idx} ({dataset_name})."
            )

    cells = element_connectivity.reshape(-1, num_nodes_per_cell).astype(np.int64, copy=False)
    if cells.min() >= 1:
        cells = cells - 1

    point_arrays = _extract_point_field_arrays(field_arrays)
    has_all_targets = all(field in point_arrays for field in target_names)
    if require_targets and (not has_all_targets):
        return None

    target = None
    if has_all_targets:
        target_components = []
        for field in target_names:
            arr = point_arrays[field]
            if arr.ndim == 1:
                arr = arr[:, None]
            target_components.append(arr.astype(np.float32, copy=False))
        target = np.concatenate(target_components, axis=-1).astype(np.float32, copy=False)

    scalar_dict = {str(k): float(v) for (k, v) in (sample.get("scalars") or {}).items()}
    scalar_values = np.array([scalar_dict.get(name, 0.0) for name in scalar_names], dtype=np.float32)
    target_scalar_names = target_scalar_names_for_dataset(dataset_name)
    has_all_target_scalars = all(name in scalar_dict for name in target_scalar_names)
    target_scalars = None
    if target_scalar_names and has_all_target_scalars:
        target_scalars = np.array([scalar_dict[name] for name in target_scalar_names], dtype=np.float32)
    if require_targets and target_scalar_names and (not has_all_target_scalars):
        return None
    node_type_ids, boundary_ids, boundary_tags = extract_node_types_and_boundary_ids(cgns_nodes, pos.shape[0])
    boundary_tag_ids = extract_boundary_tag_ids(cgns_nodes, pos.shape[0])
    boundary_pos = _boundary_positions(pos, cells, boundary_ids)
    boundary_distance = distance_to_boundary(pos, cells)

    return PlaidParsedSample(
        pos=pos,
        cells=cells,
        targets=target,
        target_scalars=target_scalars,
        boundary_pos=boundary_pos,
        scalar_values=scalar_values,
        boundary_tag_ids=boundary_tag_ids,
        boundary_ids=boundary_ids,
        boundary_tags=tuple(boundary_tags),
        node_type_ids=node_type_ids,
        boundary_distance=boundary_distance,
        target_fields=tuple(target_names),
        space_dim=int(pos.shape[-1]),
    )


def split_labeled_train_test(
    indices: list[int],
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}.")
    if len(indices) < 2:
        raise ValueError("Need at least two labeled samples to split train/test.")
    np.random.seed(seed)
    train_ids, test_ids = train_test_split(list(indices), test_size=test_ratio)
    return [int(i) for i in train_ids], [int(i) for i in test_ids]


def build_plaid_mgn_node_features(parsed: PlaidParsedSample) -> np.ndarray:
    node_type_onehot = np.eye(9, dtype=np.float32)[parsed.node_type_ids]
    scalar_features = np.repeat(parsed.scalar_values[None, :], parsed.pos.shape[0], axis=0)
    return np.concatenate(
        [parsed.pos, node_type_onehot, parsed.boundary_distance, scalar_features],
        axis=-1,
    ).astype(np.float32, copy=False)


def build_plaid_benchmark_node_features(parsed: PlaidParsedSample, *, use_sdf_features: bool = True) -> np.ndarray:
    if parsed.space_dim != 2:
        raise ValueError(f"PLAID benchmark point features currently require 2D positions, got {parsed.space_dim}D.")

    if use_sdf_features:
        boundary_ids = parsed.boundary_tag_ids.get("Holes")
        if boundary_ids is None or boundary_ids.size == 0:
            boundary_ids = parsed.boundary_ids
        if boundary_ids.size == 0:
            sdf = np.zeros((parsed.pos.shape[0], 1), dtype=np.float32)
            projection_vectors = np.zeros((parsed.pos.shape[0], parsed.space_dim), dtype=np.float32)
        else:
            boundary_vertices = parsed.pos[boundary_ids]
            search_index = KDTree(boundary_vertices)
            sdf, projection_id = search_index.query(parsed.pos, return_distance=True)
            projection_vertices = boundary_vertices[projection_id.ravel()]
            projection_vectors = projection_vertices - parsed.pos
            projection_norm = np.linalg.norm(projection_vectors, axis=1)
            projection_norm[projection_norm == 0] = 1.0
            projection_vectors = projection_vectors / projection_norm[:, None]
            sdf = sdf.astype(np.float32, copy=False)
            projection_vectors = projection_vectors.astype(np.float32, copy=False)
        field_features = np.concatenate([parsed.pos, sdf, projection_vectors], axis=-1)
    else:
        field_features = parsed.pos
    scalar_features = np.repeat(parsed.scalar_values[None, :], parsed.pos.shape[0], axis=0)
    return np.concatenate([field_features, scalar_features], axis=-1).astype(np.float32, copy=False)


def list_plaid_parquet_paths(dataset_dir: str) -> list[str]:
    return sorted(glob(f"{dataset_dir}/data/all_samples-*.parquet"))


def list_plaid_mesh_times(meshes: dict) -> list[float]:
    if not meshes:
        raise ValueError("PLAID sample has no mesh entries.")
    return sorted(float(key) for key in meshes.keys())


def build_plaid_elpl_sdf_features(
    pos: np.ndarray,
    *,
    boundary_tag_ids: dict[str, np.ndarray],
    boundary_ids: np.ndarray,
    use_sdf_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if not use_sdf_features:
        return (
            np.zeros((pos.shape[0], 1), dtype=np.float32),
            np.zeros((pos.shape[0], pos.shape[-1]), dtype=np.float32),
        )
    holes = boundary_tag_ids.get("Holes")
    if holes is not None and holes.size > 0:
        use_ids = holes
    else:
        use_ids = boundary_ids
    if use_ids.size == 0:
        return (
            np.zeros((pos.shape[0], 1), dtype=np.float32),
            np.zeros((pos.shape[0], pos.shape[-1]), dtype=np.float32),
        )
    boundary_vertices = pos[use_ids]
    search_index = KDTree(boundary_vertices)
    sdf, projection_id = search_index.query(pos, return_distance=True)
    projection_vertices = boundary_vertices[projection_id.ravel()]
    projection_vectors = projection_vertices - pos
    projection_norm = np.linalg.norm(projection_vectors, axis=1)
    projection_norm[projection_norm == 0] = 1.0
    projection_vectors = (projection_vectors / projection_norm[:, None]).astype(np.float32, copy=False)
    return sdf.astype(np.float32, copy=False).reshape(-1, 1), projection_vectors


def _extract_point_targets_from_mesh(mesh_entry: Any, target_fields: list[str] | tuple[str, ...]) -> np.ndarray | None:
    arrays = _collect_cgns_arrays(list(iter_cgns_nodes(mesh_entry)))
    point_arrays = _extract_point_field_arrays(arrays)
    if not all(field in point_arrays for field in target_fields):
        return None
    target_components = []
    for field in target_fields:
        arr = np.asarray(point_arrays[field])
        if arr.ndim == 1:
            arr = arr[:, None]
        target_components.append(arr.astype(np.float32, copy=False))
    return np.concatenate(target_components, axis=-1).astype(np.float32, copy=False)


PLAID_ELPL_TARGET_FIELDS = ("U_x", "U_y")
PLAID_ELPL_INPUT_SCALAR_NAMES = ("time",)


def _parse_plaid_geometry_from_sample(
    sample: dict,
    *,
    sample_idx: int,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
    meshes = sample["meshes"]
    target_fields = PLAID_ELPL_TARGET_FIELDS
    geom_mesh_entry, _ = _select_plaid_mesh_entries(meshes, target_fields)
    cgns_nodes = list(iter_cgns_nodes(geom_mesh_entry))
    arrays = _collect_cgns_arrays(cgns_nodes)

    def find_array(mesh_arrays: dict[str, np.ndarray], suffix: str):
        for path, arr in mesh_arrays.items():
            if path.endswith(suffix):
                return arr
        return None

    x_coord = find_array(arrays, "/CoordinateX")
    y_coord = find_array(arrays, "/CoordinateY")
    z_coord = find_array(arrays, "/CoordinateZ")
    if x_coord is None or y_coord is None:
        raise ValueError(f"Could not locate coordinates in PLAID sample {sample_idx} ({dataset_name}).")
    if z_coord is None:
        pos = np.stack([x_coord, y_coord], axis=-1).astype(np.float32, copy=False)
    else:
        pos = np.stack([x_coord, y_coord, z_coord], axis=-1).astype(np.float32, copy=False)

    element_connectivity = None
    element_parent_name = None
    for path, arr in arrays.items():
        if path.endswith("/ElementConnectivity"):
            element_connectivity = arr
            element_parent_name = path.split("/")[-2]
            break
    if element_connectivity is None:
        raise ValueError(f"Could not locate ElementConnectivity in PLAID sample {sample_idx} ({dataset_name}).")

    num_nodes_per_cell = 3
    if element_parent_name is not None:
        match = re.search(r"_(\d+)$", element_parent_name)
        if match is not None:
            num_nodes_per_cell = int(match.group(1))
    if element_connectivity.size % num_nodes_per_cell != 0:
        if element_connectivity.size % 3 == 0:
            num_nodes_per_cell = 3
        elif element_connectivity.size % 4 == 0:
            num_nodes_per_cell = 4
        else:
            raise ValueError(
                f"Invalid element connectivity size={element_connectivity.size} "
                f"in sample {sample_idx} ({dataset_name})."
            )
    cells = element_connectivity.reshape(-1, num_nodes_per_cell).astype(np.int64, copy=False)
    if cells.min() >= 1:
        cells = cells - 1

    _, boundary_ids, boundary_tags = extract_node_types_and_boundary_ids(cgns_nodes, pos.shape[0])
    boundary_tag_ids = extract_boundary_tag_ids(cgns_nodes, pos.shape[0])
    return pos, cells, boundary_ids, boundary_tag_ids, boundary_tags


def parse_plaid_elpl_trajectory(
    sample_bytes: bytes,
    *,
    sample_idx: int,
    require_targets: bool = True,
) -> dict[str, Any]:
    """Parse one elasto-plasto simulation into a trajectory bundle (mesh + SDF + U[41,N,2])."""
    sample = pickle.loads(sample_bytes)
    meshes = sample["meshes"]
    time_keys = sorted(meshes.keys(), key=float)
    if len(time_keys) < 2:
        if require_targets:
            raise ValueError(f"PLAID el-pl sample {sample_idx} has fewer than 2 mesh times.")
        raise ValueError(f"PLAID el-pl sample {sample_idx} has fewer than 2 mesh times.")

    pos, cells, boundary_ids, boundary_tag_ids, boundary_tags = _parse_plaid_geometry_from_sample(
        sample,
        sample_idx=sample_idx,
        dataset_name="plaid_el_pl_dynamics",
    )
    sdf, projection_vectors = build_plaid_elpl_sdf_features(
        pos,
        boundary_tag_ids=boundary_tag_ids,
        boundary_ids=boundary_ids,
        use_sdf_features=True,
    )

    targets_by_time: dict[float, np.ndarray] = {}
    for key in time_keys:
        targets = _extract_point_targets_from_mesh(meshes[key], PLAID_ELPL_TARGET_FIELDS)
        if targets is None:
            if require_targets:
                raise ValueError(
                    f"PLAID el-pl sample {sample_idx} is missing targets at time {float(key)}."
                )
            continue
        targets_by_time[float(key)] = targets

    timestep_list = [float(key) for key in time_keys]
    if not targets_by_time:
        raise ValueError(f"PLAID el-pl sample {sample_idx} has no target fields.")
    u_slices = []
    times = []
    for t in timestep_list:
        if t not in targets_by_time:
            if require_targets:
                raise ValueError(f"PLAID el-pl sample {sample_idx} missing target fields at time {t}.")
            continue
        u_slices.append(targets_by_time[t])
        times.append(t)
    u_traj = np.stack(u_slices, axis=0).astype(np.float32, copy=False)
    return dict(
        pos=pos,
        cells=cells,
        sdf=sdf,
        proj=projection_vectors,
        u_traj=u_traj,
        times=np.asarray(times, dtype=np.float32),
        timestep_list=timestep_list,
        sample_id=int(sample_idx),
        boundary_ids=boundary_ids,
        boundary_tags=boundary_tags,
    )


def parse_plaid_elpl_temporal_graphs(
    sample_bytes: bytes,
    *,
    sample_idx: int,
    bandwidth: float = 1.0,
    use_sdf_features: bool = True,
    require_targets: bool = True,
):
    """Expand one elasto-plasto simulation into one-step transition graphs (Vi-Transf parity)."""
    del bandwidth  # edge_attr built in plaid_datasets with dataset bandwidth
    try:
        traj = parse_plaid_elpl_trajectory(
            sample_bytes,
            sample_idx=sample_idx,
            require_targets=require_targets,
        )
    except ValueError:
        if require_targets:
            raise
        return []
    pos = traj["pos"]
    cells = traj["cells"]
    sdf = traj["sdf"]
    projection_vectors = traj["proj"]
    if not use_sdf_features:
        sdf = np.zeros_like(sdf)
        projection_vectors = np.zeros_like(projection_vectors)
    static_prefix = np.concatenate([pos, sdf, projection_vectors], axis=-1).astype(np.float32, copy=False)
    u_traj = traj["u_traj"]
    times = traj["times"]
    timestep_list = traj["timestep_list"]

    graphs = []
    for step in range(len(times) - 1):
        t0 = float(times[step])
        output_fields_t0 = u_traj[step]
        output_fields = u_traj[step + 1]
        node_x = np.concatenate([static_prefix, output_fields_t0], axis=-1).astype(np.float32, copy=False)
        graphs.append(
            dict(
                pos=pos,
                cells=cells,
                x=node_x,
                y=output_fields.copy(),
                input_scalars=np.array([t0], dtype=np.float32),
                time=t0,
                timestep_list=timestep_list,
                sample_id=int(sample_idx),
                boundary_ids=traj["boundary_ids"],
                boundary_tags=traj["boundary_tags"],
            )
        )
    return graphs
