"""Torch-free deform_plate parsing and cache build (used from the tfrecord sub-env)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .deform_plate_tfrecord_io import load_meshgraphnets_tfrecord
except ImportError:
    load_meshgraphnets_tfrecord = None  # type: ignore[assignment,misc]

NORMAL = 0
OBSTACLE = 1
HANDLE = 3
VALID_NODE_TYPES = frozenset({NORMAL, OBSTACLE, HANDLE})
INITIAL_WORLD_STEP = 0
TARGET_STEP = 360  # peak scripted actuator displacement in the DeepMind release (Pfaff et al.)
TRAJECTORY_LENGTH = 400

SPLIT_COUNTS = {
    "train": 1200,
    "valid": 100,
    "test": 100,
}
GINOT_TRAIN_COUNT = SPLIT_COUNTS["train"]
GINOT_VALID_OFFSET = GINOT_TRAIN_COUNT
GINOT_VALID_COUNT = SPLIT_COUNTS["valid"]
GINOT_INDEXABLE_COUNT = GINOT_TRAIN_COUNT + GINOT_VALID_COUNT

DATASET_DIRNAME = "deforming_plate"
CACHE_DIRNAME = "ginot_cache"
MANIFEST_NAME = "manifest.json"
TARGET_FIELD_PREFIX = "disp"

NODE_TYPE_TO_ONEHOT = {
    NORMAL: 0,
    OBSTACLE: 1,
    HANDLE: 2,
}
ONEHOT_DIM = 3
# Node feats = one_hot(type) || u_prescribed; pos (centered x0) is encoded separately for GLT.
FEAT_DIM = ONEHOT_DIM + 3

GINOT_CACHE_SAMPLE_FIELDS: tuple[str, ...] = (
    "x0",
    "mesh_pos",
    "cells",
    "node_type",
    "free_mask",
    "actuator_mask",
    "handle_mask",
    "bc_mask",
    "plate_mask",
    "plate_nodes",
    "plate_cells",
    "plate_mesh_pos",
    "u_prescribed",
    "u_target",
)
FORBIDDEN_CACHE_FIELDS = frozenset({
    "world_pos",
    "stress",
    "velocity",
    "pressure",
    "density",
    "world_pos_trajectory",
    "trajectory",
    "time_idx",
    "node_type_onehot",
})


@dataclass(frozen=True)
class DeformPlateSplitSpec:
    cache_tag: str = "official_t360"
    split_label: str = "official_train_valid"


def deform_plate_dataset_dir(data_root: str | os.PathLike[str]) -> Path:
    return Path(data_root) / DATASET_DIRNAME


def deform_plate_cache_dir(data_root: str | os.PathLike[str]) -> Path:
    return deform_plate_dataset_dir(data_root) / CACHE_DIRNAME


def _load_tfrecord(path: str | os.PathLike[str], meta: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    loader = load_meshgraphnets_tfrecord
    if loader is None:
        try:
            from deform_plate_tfrecord_io import load_meshgraphnets_tfrecord as loader
        except ImportError as exc:
            raise ImportError(
                "deform_plate TFRecord loading requires the tfrecord package. "
                "Run: bash scripts/setup_deform_plate_tfrecord_venv.sh "
                "and build the cache via the sub-env worker."
            ) from exc
    return loader(path, meta)


def _collapse_static_fields(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(sample)
    for key in ("cells", "mesh_pos", "node_type"):
        arr = np.asarray(out[key])
        if arr.ndim >= 1 and arr.shape[0] == TRAJECTORY_LENGTH:
            out[key] = arr[0]
    return out


def _node_masks(node_type: np.ndarray) -> dict[str, np.ndarray]:
    node_type = np.asarray(node_type, dtype=np.int64).reshape(-1)
    free_mask = node_type == NORMAL
    actuator_mask = node_type == OBSTACLE
    handle_mask = node_type == HANDLE
    bc_mask = actuator_mask | handle_mask
    plate_mask = free_mask | handle_mask
    return {
        "free_mask": free_mask,
        "actuator_mask": actuator_mask,
        "handle_mask": handle_mask,
        "bc_mask": bc_mask,
        "plate_mask": plate_mask,
    }


def _plate_topology(
    mesh_pos: np.ndarray,
    cells: np.ndarray,
    node_type: np.ndarray,
    plate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=np.int64)
    node_type = np.asarray(node_type, dtype=np.int64).reshape(-1)
    plate_nodes = np.flatnonzero(plate_mask)
    old_to_new = np.full(int(node_type.shape[0]), -1, dtype=np.int64)
    old_to_new[plate_nodes] = np.arange(int(plate_nodes.size), dtype=np.int64)

    cell_types = node_type[cells]
    actuator_cell = (cell_types == OBSTACLE).all(axis=1)
    plate_cell = (cell_types != OBSTACLE).all(axis=1)
    if not bool((actuator_cell | plate_cell).all()):
        raise ValueError("Found a tetrahedron connecting the actuator and plate.")

    plate_cells_global = cells[plate_cell]
    plate_cells_local = old_to_new[plate_cells_global]
    if np.any(plate_cells_local < 0):
        raise ValueError("Plate cell references a non-plate node.")
    plate_mesh_pos = np.asarray(mesh_pos, dtype=np.float32)[plate_nodes]
    return plate_nodes.astype(np.int64, copy=False), plate_cells_local.astype(np.int64, copy=False), plate_mesh_pos


def _connected_components(cells: np.ndarray, num_nodes: int) -> int:
    parent = np.arange(int(num_nodes), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for tet in np.asarray(cells, dtype=np.int64):
        a = int(tet[0])
        for b in tet[1:]:
            union(a, int(b))
    roots = {find(i) for i in range(int(num_nodes))}
    return len(roots)


def resolve_target_step(
    world_pos: np.ndarray,
    node_type: np.ndarray,
    *,
    target_step: int | None = None,
) -> int:
    """Return the quasi-static equilibrium frame used for direct displacement prediction."""
    del node_type  # fixed TARGET_STEP for this benchmark; kept for API compatibility.
    step = TARGET_STEP if target_step is None else int(target_step)
    if step < 0 or step >= int(world_pos.shape[0]):
        raise ValueError(f"target_step={step} is out of range for trajectory length {world_pos.shape[0]}.")
    return step


def parse_deform_plate_trajectory(
    sample: dict[str, np.ndarray],
    *,
    target_step: int | None = None,
) -> dict[str, np.ndarray]:
    sample = _collapse_static_fields(sample)
    cells = np.asarray(sample["cells"], dtype=np.int64)
    mesh_pos = np.asarray(sample["mesh_pos"], dtype=np.float32)
    node_type = np.asarray(sample["node_type"], dtype=np.int64).reshape(-1)
    world_pos = np.asarray(sample["world_pos"], dtype=np.float32)

    if world_pos.shape[0] != TRAJECTORY_LENGTH:
        raise ValueError(f"Expected {TRAJECTORY_LENGTH} trajectory states, got {world_pos.shape[0]}.")

    resolved_target_step = resolve_target_step(world_pos, node_type, target_step=target_step)

    unique_types = set(int(v) for v in np.unique(node_type))
    if not unique_types.issubset(VALID_NODE_TYPES):
        raise ValueError(f"Unexpected node types {sorted(unique_types)}; expected subset of {sorted(VALID_NODE_TYPES)}.")

    masks = _node_masks(node_type)
    plate_nodes, plate_cells, plate_mesh_pos = _plate_topology(mesh_pos, cells, node_type, masks["plate_mask"])

    # Total displacement in world space: u = world_pos[t_final] - world_pos[t_initial].
    # mesh_pos is the static reference (u_i) for topology only; do not use mesh_pos deltas.
    x0 = world_pos[int(INITIAL_WORLD_STEP)].astype(np.float32, copy=False)
    x_target = world_pos[int(resolved_target_step)].astype(np.float32, copy=False)
    u_target = x_target - x0
    u_prescribed = np.zeros_like(u_target, dtype=np.float32)
    u_prescribed[masks["bc_mask"]] = u_target[masks["bc_mask"]]
    u_prescribed[masks["free_mask"]] = 0.0

    return {
        "x0": x0,
        "mesh_pos": mesh_pos.astype(np.float32, copy=False),
        "cells": cells,
        "node_type": node_type,
        **masks,
        "plate_nodes": plate_nodes,
        "plate_cells": plate_cells,
        "plate_mesh_pos": plate_mesh_pos,
        "u_prescribed": u_prescribed,
        "u_target": u_target,
        "target_step": np.int64(resolved_target_step),
    }


def equilibrium_arrays_for_cache(parsed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = [name for name in GINOT_CACHE_SAMPLE_FIELDS if name not in parsed]
    if missing:
        raise ValueError(f"deform_plate equilibrium sample is missing cache fields: {missing}.")
    arrays = {name: np.asarray(parsed[name]) for name in GINOT_CACHE_SAMPLE_FIELDS}
    validate_cache_sample_arrays(arrays)
    return arrays


def validate_cache_sample_arrays(arrays: dict[str, np.ndarray]) -> None:
    forbidden = sorted(FORBIDDEN_CACHE_FIELDS.intersection(arrays))
    if forbidden:
        raise ValueError(f"deform_plate cache must not store trajectory/intermediate fields: {forbidden}.")
    extra = sorted(set(arrays) - set(GINOT_CACHE_SAMPLE_FIELDS))
    if extra:
        raise ValueError(f"deform_plate cache contains unexpected fields: {extra}.")
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] == TRAJECTORY_LENGTH:
            raise ValueError(
                f"deform_plate cache field {key!r} has trajectory length {TRAJECTORY_LENGTH} on axis 0; "
                "only equilibrium arrays may be stored."
            )


def audit_deform_plate_trajectory(parsed: dict[str, np.ndarray]) -> dict[str, Any]:
    node_type = parsed["node_type"]
    mesh_pos = np.asarray(parsed["mesh_pos"], dtype=np.float32)
    x0 = np.asarray(parsed["x0"], dtype=np.float32)
    cells = np.asarray(parsed["cells"], dtype=np.int64)
    return {
        "num_nodes": int(node_type.shape[0]),
        "num_tets": int(cells.shape[0]),
        "normal_count": int(np.sum(node_type == NORMAL)),
        "handle_count": int(np.sum(node_type == HANDLE)),
        "obstacle_count": int(np.sum(node_type == OBSTACLE)),
        "num_components": _connected_components(cells, int(node_type.shape[0])),
        "max_initial_reference_difference": float(np.max(np.abs(mesh_pos - x0))),
    }


def one_hot_node_type(node_type: np.ndarray) -> np.ndarray:
    node_type = np.asarray(node_type, dtype=np.int64).reshape(-1)
    out = np.zeros((int(node_type.shape[0]), ONEHOT_DIM), dtype=np.float32)
    for value, channel in NODE_TYPE_TO_ONEHOT.items():
        out[node_type == value, channel] = 1.0
    return out


def build_node_feature_matrix(
    node_type: np.ndarray,
    u_prescribed: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            one_hot_node_type(node_type),
            np.asarray(u_prescribed, dtype=np.float32),
        ],
        axis=-1,
    )


def _sample_cache_path(cache_root: Path, split_name: str, local_idx: int) -> Path:
    return cache_root / split_name / f"sample_{int(local_idx):04d}.npz"


def build_deform_plate_cache(
    data_root: str | os.PathLike[str],
    *,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    overwrite: bool = False,
) -> Path:
    dataset_dir = deform_plate_dataset_dir(data_root)
    cache_root = deform_plate_cache_dir(data_root)
    meta_path = dataset_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}. Download the official MeshGraphNets deforming_plate release with "
            "python scripts/download_pdebench_dataset.py --dataset deforming_plate "
            f"--data-root {Path(data_root).resolve()}"
        )
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "deform_plate_ginot_v1",
        "target_step": TARGET_STEP,
        "trajectory_length": TRAJECTORY_LENGTH,
        "stored_fields": list(GINOT_CACHE_SAMPLE_FIELDS),
        "excluded_fields": sorted(FORBIDDEN_CACHE_FIELDS),
        "note": (
            "TFRecord trajectories are collapsed to a single equilibrium sample per graph; "
            "intermediate world_pos/stress time series are not written to ginot_cache."
        ),
        "splits": {},
        "audits": [],
    }

    for split_name in splits:
        if split_name not in SPLIT_COUNTS:
            raise ValueError(f"Unsupported split {split_name!r}.")
        tfrecord_path = dataset_dir / f"{split_name}.tfrecord"
        if not tfrecord_path.is_file():
            raise FileNotFoundError(f"Missing TFRecord split file: {tfrecord_path}")
        raw_trajectories = _load_tfrecord(tfrecord_path, meta)
        expected = int(SPLIT_COUNTS[split_name])
        if len(raw_trajectories) != expected:
            raise ValueError(
                f"Split {split_name!r} should contain {expected} trajectories; found {len(raw_trajectories)}."
            )

        split_dir = cache_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_audits: list[dict[str, Any]] = []
        for local_idx, trajectory in enumerate(raw_trajectories):
            out_path = _sample_cache_path(cache_root, split_name, local_idx)
            if out_path.is_file() and not overwrite:
                with np.load(out_path) as cached:
                    parsed = {key: cached[key] for key in cached.files}
                    validate_cache_sample_arrays(parsed)
            else:
                parsed = parse_deform_plate_trajectory(trajectory)
                np.savez_compressed(out_path, **equilibrium_arrays_for_cache(parsed))
                parsed = {key: np.asarray(parsed[key]) for key in GINOT_CACHE_SAMPLE_FIELDS}
            audit = audit_deform_plate_trajectory(parsed)
            audit["split"] = split_name
            audit["local_idx"] = int(local_idx)
            split_audits.append(audit)

        manifest["splits"][split_name] = {
            "count": expected,
            "dir": split_name,
        }
        manifest["audits"].extend(split_audits)

    manifest_path = cache_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cache_root
