from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from pdebench.dataset.ginot.types import (
    FIXED_DATASET_DIRNAME,
    MANIFEST_NAME,
    MICRO_PUC_MESH_SAMPLES,
    MICRO_PUC_TOTAL_SAMPLES,
)
from pdebench.dataset.ginot.storage import ShardedMicroPucFixedSequence

__all__ = [
    "FIXED_DATASET_DIRNAME",
    "MANIFEST_NAME",
    "MICRO_PUC_MESH_SAMPLES",
    "MicroPucFixedConfig",
    "MicroPucFixedStats",
    "ShardedMicroPucFixedSequence",
    "audit_micro_puc_periodic_boundaries",
    "build_micro_puc_fixed_dataset",
    "build_periodic_node_map",
    "fix_one_sample",
]


@dataclass(frozen=True)
class MicroPucFixedConfig:
    """Configuration for constructing the x/y-periodic quotient Micro-PUC final-step dataset.

    The raw PeriodUnitCell archive contains final-step fields and coordinates for 73,879 rows.
    mesh_cells10K.pkl stores the first 10,000 canonical mesh connectivities; sample_ids.npy maps
    every shifted row back to one of those canonical meshes. This builder constructs a torus
    quotient map for each canonical mesh and applies it to every shifted row sharing that sample id.
    """

    boundary_atol: float = 1e-6
    match_atol: float = 1e-3
    fallback_boundary_atol: float = 5e-3
    fallback_match_atol: float = 5e-3
    shard_size: int = 512
    num_workers: int = 64
    max_samples: int = 0


@dataclass(frozen=True)
class MicroPucFixedStats:
    sample_id: int
    source_geometry_id: int
    canonical_row: int
    old_nodes: int
    new_nodes: int
    x0_nodes: int
    x1_nodes: int
    x1_matched_nodes: int
    x1_unmatched_nodes: int
    y0_nodes: int
    y1_nodes: int
    y1_matched_nodes: int
    y1_unmatched_nodes: int
    edge_source: str
    old_directed_edges: int
    remapped_directed_edges: int
    unique_edges: int
    self_loops_after_remap: int


def _load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _as_xy(pos: Any) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected coordinates shaped [N,D>=2], got {arr.shape}.")
    return arr[:, :2].astype(np.float32, copy=False)


def _as_field(field: Any) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected field shaped [N,C], got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _normalize_cells(cells: Any, num_nodes: int) -> np.ndarray:
    if cells is None:
        raise ValueError("Micro-PUC fixed requires mesh cell connectivity; got None.")
    cells_np = np.asarray(cells, dtype=np.int64)
    if cells_np.ndim == 1:
        if cells_np.size < 3:
            raise ValueError("Cell connectivity is too short.")
        width = int(cells_np[0]) + 1
        if width >= 3 and cells_np.size % width == 0 and np.all(cells_np[::width] == width - 1):
            cells_np = cells_np.reshape(-1, width)[:, 1:]
        else:
            parsed = []
            offset = 0
            while offset < cells_np.size:
                width = int(cells_np[offset])
                next_offset = offset + width + 1
                if width < 2 or next_offset > cells_np.size:
                    raise ValueError("Could not parse ragged VTK-style cell connectivity.")
                parsed.append(cells_np[offset + 1:next_offset])
                offset = next_offset
            cells_np = np.asarray(parsed, dtype=np.int64)
    if cells_np.ndim != 2 or cells_np.shape[1] < 2:
        raise ValueError(f"Expected cell connectivity shaped [E,K>=2], got {cells_np.shape}.")
    cell_min = int(cells_np.min())
    cell_max = int(cells_np.max())
    if cell_min >= 1 and cell_max >= num_nodes:
        cells_np = cells_np - 1
        cell_min = int(cells_np.min())
        cell_max = int(cells_np.max())
    if cell_min < 0 or cell_max >= num_nodes:
        raise ValueError(
            f"Cell indices outside node range [0, {num_nodes}): min={cell_min}, max={cell_max}."
        )
    return cells_np.astype(np.int64, copy=False)


def _directed_edges_from_cells(cells_np: np.ndarray) -> np.ndarray:
    edges = []
    if cells_np.shape[1] == 2:
        edge_offsets = [(0, 1)]
    else:
        edge_offsets = [(i, (i + 1) % cells_np.shape[1]) for i in range(cells_np.shape[1])]
    for i, j in edge_offsets:
        a = cells_np[:, i]
        b = cells_np[:, j]
        keep = a != b
        if np.any(keep):
            edges.append(np.stack([a[keep], b[keep]], axis=1))
            edges.append(np.stack([b[keep], a[keep]], axis=1))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(edges, axis=0).astype(np.int64, copy=False)


def _unique_edge_pairs(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = edges.astype(np.int64, copy=False)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    keep = np.ones(edges.shape[0], dtype=bool)
    keep[1:] = np.any(edges[1:] != edges[:-1], axis=1)
    return edges[keep]


def build_periodic_node_map(pos: Any, config: MicroPucFixedConfig) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Map duplicated x=1 and y=1 boundary nodes onto x=0 / y=0 representatives.

    Micro-PUC is represented on [0, 1]^2 with periodic x and y. Canonical meshes duplicate
    boundary cuts; this quotient keeps (x=0, y=0) representatives and remaps matching boundary
    duplicates (x=1 by y, then y=1 by x). Corners collapse to a single node, e.g. (0,0) absorbs
    (0,1), (1,0), and (1,1).
    """

    pos_xy = _as_xy(pos)
    atol = float(config.boundary_atol)
    if pos_xy.min(initial=0.0) < -atol or pos_xy.max(initial=0.0) > 1.0 + atol:
        raise ValueError(
            "Micro-PUC periodic quotient expects coordinates normalized to [0, 1]^2; "
            f"found min={float(pos_xy.min())}, max={float(pos_xy.max())}."
        )
    x = pos_xy[:, 0]
    y = pos_xy[:, 1]
    x0 = np.flatnonzero(np.isclose(x, 0.0, atol=atol))
    x1 = np.flatnonzero(np.isclose(x, 1.0, atol=atol))
    y0 = np.flatnonzero(np.isclose(y, 0.0, atol=atol))
    y1 = np.flatnonzero(np.isclose(y, 1.0, atol=atol))

    parent = np.arange(pos_xy.shape[0], dtype=np.int64)

    def map_matching_boundary(
        source: np.ndarray,
        target: np.ndarray,
        match_coord: np.ndarray,
        match_atol: float,
    ) -> tuple[int, np.ndarray]:
        if len(source) == 0 or len(target) == 0:
            return 0, target
        source_sorted = source[np.argsort(match_coord[source], kind="stable")]
        source_values = match_coord[source_sorted]
        matched = 0
        unmatched = []
        for target_idx in target[np.argsort(match_coord[target], kind="stable")]:
            coord = float(match_coord[target_idx])
            insert = int(np.searchsorted(source_values, coord))
            candidates = []
            if insert < len(source_sorted):
                candidates.append(int(source_sorted[insert]))
            if insert > 0:
                candidates.append(int(source_sorted[insert - 1]))
            if not candidates:
                unmatched.append(int(target_idx))
                continue
            source_idx = min(candidates, key=lambda idx: abs(float(match_coord[idx]) - coord))
            if abs(float(match_coord[source_idx]) - coord) <= float(match_atol):
                parent[int(target_idx)] = int(source_idx)
                matched += 1
            else:
                unmatched.append(int(target_idx))
        return matched, np.asarray(unmatched, dtype=np.int64)

    x_matched, x_unmatched = map_matching_boundary(x0, x1, y, float(config.match_atol))
    if len(x_unmatched) > 0 and float(config.fallback_boundary_atol) > atol:
        near_x0 = np.flatnonzero(np.isclose(x, 0.0, atol=float(config.fallback_boundary_atol)))
        fallback_matched, x_unmatched = map_matching_boundary(near_x0, x_unmatched, y, float(config.fallback_match_atol))
        x_matched += fallback_matched

    y_matched, y_unmatched = map_matching_boundary(y0, y1, x, float(config.match_atol))
    if len(y_unmatched) > 0 and float(config.fallback_boundary_atol) > atol:
        near_y0 = np.flatnonzero(np.isclose(y, 0.0, atol=float(config.fallback_boundary_atol)))
        fallback_matched, y_unmatched = map_matching_boundary(near_y0, y_unmatched, x, float(config.fallback_match_atol))
        y_matched += fallback_matched

    for i in range(parent.shape[0]):
        while parent[parent[i]] != parent[i]:
            parent[i] = parent[parent[i]]

    keep = parent == np.arange(parent.shape[0], dtype=np.int64)
    old_to_new = np.full(parent.shape[0], -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    remap = old_to_new[parent]
    stats = {
        "x0_nodes": int(len(x0)),
        "x1_nodes": int(len(x1)),
        "x1_matched_nodes": int(x_matched),
        "x1_unmatched_nodes": int(len(x1) - x_matched),
        "y0_nodes": int(len(y0)),
        "y1_nodes": int(len(y1)),
        "y1_matched_nodes": int(y_matched),
        "y1_unmatched_nodes": int(len(y1) - y_matched),
        "kept_nodes": int(keep.sum()),
        "removed_nodes": int((~keep).sum()),
    }
    return remap, keep, stats


def _remap_edges_from_cells(cells: Any, num_nodes: int, remap: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    cells_np = _normalize_cells(cells, num_nodes=num_nodes)
    old_edges = _directed_edges_from_cells(cells_np)
    remapped_edges = old_edges.copy()
    remapped_edges[:, 0] = remap[remapped_edges[:, 0]]
    remapped_edges[:, 1] = remap[remapped_edges[:, 1]]
    self_loops = int(np.sum(remapped_edges[:, 0] == remapped_edges[:, 1]))
    unique_edges = _unique_edge_pairs(remapped_edges)
    return unique_edges, int(old_edges.shape[0]), int(remapped_edges.shape[0]), self_loops


def fix_one_sample(
    sample_id: int,
    pos: Any,
    field: Any,
    cells: Any,
    config: MicroPucFixedConfig,
    *,
    canonical_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pos_xy = _as_xy(pos)
    y_np = _as_field(field)
    if pos_xy.shape[0] != y_np.shape[0]:
        raise ValueError(f"Sample {sample_id} has inconsistent pos/y nodes: {pos_xy.shape} vs {y_np.shape}.")
    if canonical_map is None:
        if cells is None:
            raise ValueError("Micro-PUC fixed requires mesh cell connectivity; got None.")
        remap, keep, map_stats = build_periodic_node_map(pos_xy, config)
        unique_edges, old_edge_count, remapped_edge_count, self_loops = _remap_edges_from_cells(
            cells,
            num_nodes=int(pos_xy.shape[0]),
            remap=remap,
        )
        source_geometry_id = int(sample_id)
        canonical_row = int(sample_id)
    else:
        keep = canonical_map["keep"]
        unique_edges = canonical_map["edges"]
        map_stats = canonical_map["stats"]
        old_edge_count = int(canonical_map["old_directed_edges"])
        remapped_edge_count = int(canonical_map["remapped_directed_edges"])
        self_loops = int(canonical_map["self_loops_after_remap"])
        source_geometry_id = int(canonical_map["source_geometry_id"])
        canonical_row = int(canonical_map["canonical_row"])
        if keep.shape[0] != pos_xy.shape[0]:
            raise ValueError(
                f"Sample {sample_id} has {pos_xy.shape[0]} nodes but canonical row {canonical_row} "
                f"for source geometry {source_geometry_id} has {keep.shape[0]} nodes."
            )

    fixed = {
        "pos": pos_xy[keep].astype(np.float32, copy=False),
        "y": y_np[keep].astype(np.float32, copy=False),
        "edges": unique_edges.astype(np.int64, copy=False),
    }
    fixed["stats"] = MicroPucFixedStats(
        sample_id=int(sample_id),
        source_geometry_id=source_geometry_id,
        canonical_row=canonical_row,
        old_nodes=int(pos_xy.shape[0]),
        new_nodes=int(fixed["pos"].shape[0]),
        x0_nodes=int(map_stats["x0_nodes"]),
        x1_nodes=int(map_stats["x1_nodes"]),
        x1_matched_nodes=int(map_stats["x1_matched_nodes"]),
        x1_unmatched_nodes=int(map_stats["x1_unmatched_nodes"]),
        y0_nodes=int(map_stats["y0_nodes"]),
        y1_nodes=int(map_stats["y1_nodes"]),
        y1_matched_nodes=int(map_stats["y1_matched_nodes"]),
        y1_unmatched_nodes=int(map_stats["y1_unmatched_nodes"]),
        edge_source="cells",
        old_directed_edges=old_edge_count,
        remapped_directed_edges=remapped_edge_count,
        unique_edges=int(unique_edges.shape[0]),
        self_loops_after_remap=self_loops,
    )
    return fixed


def audit_micro_puc_periodic_boundaries(data_root: str | Path = "data", max_samples: int = MICRO_PUC_MESH_SAMPLES) -> dict[str, Any]:
    src_dir = Path(data_root) / "PeriodUnitCell"
    coords = _load_pickle(src_dir / "mesh_coords.pkl")
    cells_path = src_dir / "mesh_cells10K.pkl"
    if not cells_path.exists():
        raise FileNotFoundError(f"Micro-PUC mesh connectivity is required: {cells_path}")
    cells = _load_pickle(cells_path)
    num_samples = min(int(max_samples), len(coords), len(cells), MICRO_PUC_MESH_SAMPLES)
    failures = []
    for sample_id in range(num_samples):
        try:
            _normalize_cells(cells[sample_id], num_nodes=int(_as_xy(coords[sample_id]).shape[0]))
            build_periodic_node_map(coords[sample_id], MicroPucFixedConfig())
        except Exception as exc:
            failures.append({"sample_id": int(sample_id), "error": str(exc)})
            if len(failures) >= 10:
                break
    return {"checked": num_samples, "failures": failures}


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(coords, fields, cells, config_dict):
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        coords=coords,
        fields=fields,
        sample_ids=np.asarray(cells["sample_ids"], dtype=np.int64),
        canonical_maps=cells["canonical_maps"],
        config=MicroPucFixedConfig(**config_dict),
    )


def _process_shard(task):
    shard_id, row_ids, out_root_str = task
    out_root = Path(out_root_str)
    coords = _WORKER_STATE["coords"]
    fields = _WORKER_STATE["fields"]
    sample_ids = _WORKER_STATE["sample_ids"]
    canonical_maps = _WORKER_STATE["canonical_maps"]
    config = _WORKER_STATE["config"]
    shard_pos = []
    shard_y = []
    shard_edges = []
    stats = []
    for row_id in row_ids:
        source_geometry_id = int(sample_ids[int(row_id)])
        fixed = fix_one_sample(
            int(row_id),
            coords[int(row_id)],
            fields[int(row_id)],
            None,
            config,
            canonical_map=canonical_maps[source_geometry_id],
        )
        shard_pos.append(torch.from_numpy(fixed["pos"]))
        shard_y.append(torch.from_numpy(fixed["y"]))
        shard_edges.append(torch.from_numpy(fixed["edges"]))
        stats.append(asdict(fixed["stats"]))
    shard_path = out_root / "shards" / f"shard_{int(shard_id):06d}.pt"
    tmp_path = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
    torch.save({"sample_ids": list(map(int, row_ids)), "pos": shard_pos, "y": shard_y, "cells": shard_edges}, tmp_path)
    os.replace(tmp_path, shard_path)
    return {"shard_id": int(shard_id), "num_samples": len(row_ids), "stats": stats}


def build_micro_puc_fixed_dataset(
    data_root: str | Path = "data",
    output_dir: str | Path | None = None,
    *,
    config: MicroPucFixedConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Build the reusable periodic-quotient Micro-PUC final-step dataset."""

    config = MicroPucFixedConfig() if config is None else config
    data_root = Path(data_root)
    src_dir = data_root / "PeriodUnitCell"
    out_root = Path(output_dir) if output_dir is not None else data_root / FIXED_DATASET_DIRNAME
    manifest_path = out_root / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "shards").mkdir(parents=True, exist_ok=True)
    log_path = out_root / "processing_stats.jsonl"
    if overwrite:
        for path in (out_root / "shards").glob("shard_*.pt"):
            path.unlink()
        if log_path.exists():
            log_path.unlink()

    coords = _load_pickle(src_dir / "mesh_coords.pkl")
    su_data = _load_pickle(src_dir / "mises_disp_laststep.pkl")
    fields = su_data["mises_disp"] if isinstance(su_data, dict) else su_data
    cells_path = src_dir / "mesh_cells10K.pkl"
    sample_ids_path = src_dir / "sample_ids.npy"
    if not cells_path.exists():
        raise FileNotFoundError(f"Micro-PUC fixed requires mesh connectivity: {cells_path}")
    if not sample_ids_path.exists():
        raise FileNotFoundError(f"Micro-PUC fixed requires sample id mapping: {sample_ids_path}")
    cells = _load_pickle(cells_path)
    row_sample_ids = np.load(sample_ids_path).astype(np.int64, copy=False)
    if len(coords) != len(fields) or len(coords) != len(row_sample_ids):
        raise ValueError(
            "Micro-PUC fixed requires aligned mesh_coords, mises_disp, and sample_ids rows; "
            f"found coords={len(coords)} fields={len(fields)} sample_ids={len(row_sample_ids)}."
        )
    source_limit = len(coords) if int(config.max_samples) <= 0 else min(int(config.max_samples), len(coords))
    if source_limit <= 0:
        raise ValueError("Micro-PUC fixed requires at least one sample row.")
    if source_limit > MICRO_PUC_TOTAL_SAMPLES:
        raise ValueError(f"Micro-PUC fixed source limit {source_limit} exceeds expected {MICRO_PUC_TOTAL_SAMPLES}.")
    if len(coords) < source_limit or len(fields) < source_limit:
        raise ValueError(
            f"Micro-PUC fixed needs {source_limit} aligned coordinate/field rows; "
            f"found coords={len(coords)} fields={len(fields)}."
        )

    canonical_limit = min(MICRO_PUC_MESH_SAMPLES, len(cells), len(coords), len(row_sample_ids))
    if canonical_limit <= 0:
        raise ValueError("Micro-PUC fixed requires at least one canonical mesh in mesh_cells10K.pkl.")
    sid_to_row = {int(row_sample_ids[row]): int(row) for row in range(canonical_limit)}
    required_sids = {int(sid) for sid in row_sample_ids[:source_limit]}
    missing = sorted(required_sids - set(sid_to_row))
    if missing:
        preview = missing[:8]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(
            "Micro-PUC fixed sample_ids reference source geometry ids without canonical mesh "
            f"in rows 0..{canonical_limit - 1}: {preview}{suffix}"
        )

    canonical_maps: dict[int, dict[str, Any]] = {}
    for source_geometry_id in sorted(required_sids):
        canonical_row = sid_to_row[source_geometry_id]
        canonical_pos = _as_xy(coords[canonical_row])
        remap, keep, map_stats = build_periodic_node_map(canonical_pos, config)
        unique_edges, old_edge_count, remapped_edge_count, self_loops = _remap_edges_from_cells(
            cells[canonical_row],
            num_nodes=int(canonical_pos.shape[0]),
            remap=remap,
        )
        canonical_maps[source_geometry_id] = {
            "keep": keep,
            "edges": unique_edges,
            "stats": map_stats,
            "old_directed_edges": old_edge_count,
            "remapped_directed_edges": remapped_edge_count,
            "self_loops_after_remap": self_loops,
            "source_geometry_id": int(source_geometry_id),
            "canonical_row": int(canonical_row),
        }

    row_ids = list(range(source_limit))
    num_samples = len(row_ids)
    shards = [row_ids[i:i + int(config.shard_size)] for i in range(0, num_samples, int(config.shard_size))]
    tasks = [(shard_id, ids, str(out_root)) for shard_id, ids in enumerate(shards)]
    ctx = mp.get_context("fork")
    num_workers = max(1, min(int(config.num_workers), os.cpu_count() or 1, len(tasks)))
    totals = {
        "samples": 0,
        "old_nodes": 0,
        "new_nodes": 0,
        "cell_edge_samples": 0,
        "x1_unmatched_nodes": 0,
        "y1_unmatched_nodes": 0,
        "self_loops_after_remap": 0,
    }
    with log_path.open("w", encoding="utf-8") as log_file:
        worker_cells = {"sample_ids": row_sample_ids, "canonical_maps": canonical_maps}
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(coords, fields, worker_cells, asdict(config))) as pool:
            for result in tqdm(pool.imap_unordered(_process_shard, tasks), total=len(tasks), desc="micro_puc_fixed shards", ncols=100):
                for item in result["stats"]:
                    log_file.write(json.dumps(item, sort_keys=True) + "\n")
                    totals["samples"] += 1
                    totals["old_nodes"] += int(item["old_nodes"])
                    totals["new_nodes"] += int(item["new_nodes"])
                    totals["x1_unmatched_nodes"] += int(item["x1_unmatched_nodes"])
                    totals["y1_unmatched_nodes"] += int(item["y1_unmatched_nodes"])
                    totals["self_loops_after_remap"] += int(item["self_loops_after_remap"])
                    if item["edge_source"] == "cells":
                        totals["cell_edge_samples"] += 1

    manifest = {
        "name": "micro_puc_fixed",
        "source_dir": str(src_dir),
        "num_samples": num_samples,
        "source_sample_limit": source_limit,
        "canonical_mesh_rows": canonical_limit,
        "source_geometry_ids": sorted(map(int, required_sids)),
        "excluded_sample_ids": [],
        "quotient_axes": ["x", "y"],
        "shard_size": int(config.shard_size),
        "num_shards": len(shards),
        "keys": ["pos", "y", "cells"],
        "target_fields": ["mises_stress", "disp_x", "disp_y"],
        "target_shift": su_data.get("shift").tolist() if isinstance(su_data, dict) and "shift" in su_data else None,
        "target_scaler": su_data.get("scaler").tolist() if isinstance(su_data, dict) and "scaler" in su_data else None,
        "config": asdict(config),
        "totals": totals,
        "notes": [
            "Coordinates and final-step fields come from PeriodUnitCell mesh_coords.pkl and mises_disp_laststep.pkl.",
            "sample_ids.npy maps every row to one of the first 10,000 canonical mesh rows in mesh_cells10K.pkl.",
            "The x/y-periodic quotient is built once per canonical mesh and applied to every shifted row with that source geometry id.",
            "Duplicated x=1 nodes map onto x=0 by y; duplicated y=1 nodes map onto y=0 by x (corners collapse to one node).",
            "Edges are remapped strictly from mesh_cells10K.pkl cell perimeters; no kNN or coordinate-neighbor connectivity is synthesized.",
            "Downstream edge attributes should use minimum-image displacement on both axes for the [0, 1)^2 torus.",
        ],
    }
    tmp_manifest = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp_manifest.replace(manifest_path)
    return out_root


def _parse_args():
    parser = argparse.ArgumentParser(description="Build periodic-quotient Micro-PUC dataset.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--boundary-atol", type=float, default=1e-6)
    parser.add_argument("--match-atol", type=float, default=1e-3)
    parser.add_argument("--fallback-boundary-atol", type=float, default=5e-3)
    parser.add_argument("--fallback-match-atol", type=float, default=5e-3)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit source rows for debugging; 0 means all rows.")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = MicroPucFixedConfig(
        boundary_atol=float(args.boundary_atol),
        match_atol=float(args.match_atol),
        fallback_boundary_atol=float(args.fallback_boundary_atol),
        fallback_match_atol=float(args.fallback_match_atol),
        shard_size=int(args.shard_size),
        num_workers=int(args.num_workers),
        max_samples=int(args.max_samples),
    )
    if args.audit_only:
        print(json.dumps(audit_micro_puc_periodic_boundaries(args.data_root, max_samples=config.max_samples), indent=2))
        return
    out = build_micro_puc_fixed_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config=config,
        overwrite=bool(args.overwrite),
    )
    print(f"micro_puc_fixed written to {out}")


if __name__ == "__main__":
    main()
