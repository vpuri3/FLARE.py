"""MeshGraphNets DeformingPlate → direct equilibrium GINOT benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from pdebench.dataset.ginot import deform_plate_core as core
from pdebench.dataset.ginot.mesh import (
    build_edge_index_from_normalized_cells,
    normalize_cells,
)
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer
from pdebench.dataset.ginot.utils import target_fields

# Re-export torch-free API used by tests and other ginot modules.
DATASET_DIRNAME = core.DATASET_DIRNAME
CACHE_DIRNAME = core.CACHE_DIRNAME
MANIFEST_NAME = core.MANIFEST_NAME
TARGET_FIELD_PREFIX = core.TARGET_FIELD_PREFIX
NORMAL = core.NORMAL
OBSTACLE = core.OBSTACLE
HANDLE = core.HANDLE
VALID_NODE_TYPES = core.VALID_NODE_TYPES
TARGET_STEP = core.TARGET_STEP
INITIAL_WORLD_STEP = core.INITIAL_WORLD_STEP
TRAJECTORY_LENGTH = core.TRAJECTORY_LENGTH
resolve_target_step = core.resolve_target_step
SPLIT_COUNTS = core.SPLIT_COUNTS
GINOT_TRAIN_COUNT = core.GINOT_TRAIN_COUNT
GINOT_VALID_OFFSET = core.GINOT_VALID_OFFSET
GINOT_VALID_COUNT = core.GINOT_VALID_COUNT
GINOT_INDEXABLE_COUNT = core.GINOT_INDEXABLE_COUNT
NODE_TYPE_TO_ONEHOT = core.NODE_TYPE_TO_ONEHOT
ONEHOT_DIM = core.ONEHOT_DIM
FEAT_DIM = core.FEAT_DIM
GINOT_CACHE_SAMPLE_FIELDS = core.GINOT_CACHE_SAMPLE_FIELDS
FORBIDDEN_CACHE_FIELDS = core.FORBIDDEN_CACHE_FIELDS
DeformPlateSplitSpec = core.DeformPlateSplitSpec
deform_plate_dataset_dir = core.deform_plate_dataset_dir
deform_plate_cache_dir = core.deform_plate_cache_dir
parse_deform_plate_trajectory = core.parse_deform_plate_trajectory
equilibrium_arrays_for_cache = core.equilibrium_arrays_for_cache
validate_cache_sample_arrays = core.validate_cache_sample_arrays
audit_deform_plate_trajectory = core.audit_deform_plate_trajectory
one_hot_node_type = core.one_hot_node_type
build_node_feature_matrix = core.build_node_feature_matrix
build_deform_plate_cache = core.build_deform_plate_cache

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUBENV_PYTHON = _REPO_ROOT / "out/pdebench/.venv-deform-plate-tfrecord/bin/python"
_CACHE_WORKER = _REPO_ROOT / "scripts/deform_plate_build_ginot_cache.py"
_SETUP_SCRIPT = _REPO_ROOT / "scripts/setup_deform_plate_tfrecord_venv.sh"


def _ensure_tfrecord_subenv() -> Path:
    if not _SUBENV_PYTHON.is_file():
        if not _SETUP_SCRIPT.is_file():
            raise RuntimeError(
                "Missing deform_plate tfrecord sub-env. "
                f"Expected setup script at {_SETUP_SCRIPT}."
            )
        subprocess.run(["bash", str(_SETUP_SCRIPT)], check=True)
    if not _SUBENV_PYTHON.is_file():
        raise RuntimeError(
            "deform_plate tfrecord sub-env is not installed. "
            f"Run: bash {_SETUP_SCRIPT}"
        )
    return _SUBENV_PYTHON


def run_deform_plate_cache_build(
    data_root: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert TFRecords to ginot_cache using the isolated tfrecord sub-env."""
    subenv_python = _ensure_tfrecord_subenv()
    cmd = [
        str(subenv_python),
        str(_CACHE_WORKER),
        "--data-root",
        str(data_root),
    ]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)
    return deform_plate_cache_dir(data_root)


class _DeformPlateNpzStore:
    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root)
        manifest_path = self.cache_root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing deform_plate cache manifest at {manifest_path}. "
                "Build it with: bash out/pdebench/run_deform_plate_precompute.sh"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_count = int(self.manifest["splits"]["train"]["count"])
        valid_count = int(self.manifest["splits"]["valid"]["count"])
        self._index: list[tuple[str, int]] = []
        for local_idx in range(train_count):
            self._index.append(("train", local_idx))
        for local_idx in range(valid_count):
            self._index.append(("valid", local_idx))

    def __len__(self) -> int:
        return len(self._index)

    def _load(self, global_idx: int) -> dict[str, np.ndarray]:
        split_name, local_idx = self._index[int(global_idx)]
        path = core._sample_cache_path(self.cache_root, split_name, local_idx)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached deform_plate sample: {path}")
        with np.load(path) as data:
            parsed = {key: data[key] for key in data.files}
        validate_cache_sample_arrays(parsed)
        return parsed

    def field(self, global_idx: int, name: str) -> np.ndarray:
        return self._load(global_idx)[name]


class DeformPlateSequence:
    def __init__(self, store: _DeformPlateNpzStore, field: str):
        self.store = store
        self.field = str(field)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int):
        sample = self.store._load(int(idx))
        if self.field == "query_points":
            return sample["x0"]
        if self.field == "point_clouds":
            bc_mask = sample["bc_mask"]
            return sample["x0"][bc_mask] + sample["u_prescribed"][bc_mask]
        if self.field == "targets":
            return sample["u_target"]
        if self.field == "cells":
            return sample["cells"]
        if self.field == "mesh_pos":
            return sample["mesh_pos"]
        if self.field == "node_type":
            return sample["node_type"]
        if self.field == "free_mask":
            return sample["free_mask"]
        if self.field == "bc_mask":
            return sample["bc_mask"]
        if self.field == "u_prescribed":
            return sample["u_prescribed"]
        if self.field == "plate_nodes":
            return sample["plate_nodes"]
        if self.field == "plate_cells":
            return sample["plate_cells"]
        if self.field == "plate_mesh_pos":
            return sample["plate_mesh_pos"]
        raise KeyError(self.field)


def resolve_deform_plate_splits(
    _data_root: str,
    _split_seed: int,
) -> tuple[list[int], list[int]]:
    del _data_root, _split_seed
    train_ids = list(range(GINOT_TRAIN_COUNT))
    test_ids = list(range(GINOT_VALID_OFFSET, GINOT_INDEXABLE_COUNT))
    return train_ids, test_ids


def resolve_deform_plate_official_test_indices(data_root: str) -> list[int]:
    del data_root
    return list(range(GINOT_VALID_COUNT))


def compute_deform_plate_normalizers(
    raw: GinotRawDataset,
    train_ids: list[int],
) -> tuple[StandardNormalizer, StandardNormalizer, StandardNormalizer, float]:
    if raw.deform_plate_store is None:
        raise ValueError("deform_plate normalizers require deform_plate_store on GinotRawDataset.")

    store = raw.deform_plate_store
    centered_parts: list[np.ndarray] = []
    free_disp_parts: list[np.ndarray] = []
    for idx in train_ids:
        sample = store._load(int(idx))
        x0 = sample["x0"]
        plate_mask = sample["plate_mask"]
        center = x0[plate_mask].mean(axis=0, keepdims=True)
        centered = x0 - center
        centered_parts.append(centered)
        free_disp_parts.append(sample["u_target"][sample["free_mask"]])

    centered_all = np.concatenate(centered_parts, axis=0)
    radius_sq = np.sum(centered_all * centered_all, axis=-1)
    global_length_scale = float(max(np.sqrt(np.mean(radius_sq)), 1e-8))

    free_disp = np.concatenate(free_disp_parts, axis=0).astype(np.float32, copy=False)
    mean = free_disp.mean(axis=0, keepdims=True)
    std = free_disp.std(axis=0, keepdims=True).clip(min=1e-8)

    pos_normalizer = StandardNormalizer(
        mean=torch.zeros(1, 3),
        std=torch.full((1, 3), global_length_scale),
    )
    boundary_pos_normalizer = pos_normalizer
    y_normalizer = StandardNormalizer(
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
    )
    return pos_normalizer, boundary_pos_normalizer, y_normalizer, global_length_scale


def encode_deform_plate_positions(
    raw: GinotRawDataset,
    idx: int,
    pos_normalizer: StandardNormalizer,
) -> torch.Tensor:
    if raw.deform_plate_store is None:
        raise ValueError("encode_deform_plate_positions requires deform_plate_store.")
    sample = raw.deform_plate_store._load(int(idx))
    x0 = sample["x0"]
    center = x0[sample["plate_mask"]].mean(axis=0, keepdims=True)
    centered = torch.from_numpy((x0 - center).astype(np.float32, copy=False)).float()
    return pos_normalizer.encode(centered)


def encode_deform_plate_boundary_positions(
    raw: GinotRawDataset,
    idx: int,
    pos_normalizer: StandardNormalizer,
) -> torch.Tensor:
    if raw.deform_plate_store is None:
        raise ValueError("encode_deform_plate_boundary_positions requires deform_plate_store.")
    sample = raw.deform_plate_store._load(int(idx))
    x0 = sample["x0"]
    center = x0[sample["plate_mask"]].mean(axis=0, keepdims=True)
    bc_pos = x0[sample["bc_mask"]] + sample["u_prescribed"][sample["bc_mask"]]
    centered = torch.from_numpy((bc_pos - center).astype(np.float32, copy=False)).float()
    return pos_normalizer.encode(centered)


def encode_deform_plate_feats(
    raw: GinotRawDataset,
    idx: int,
    y_normalizer: StandardNormalizer,
    num_nodes: int,
) -> torch.Tensor:
    if raw.deform_plate_store is None:
        raise ValueError("encode_deform_plate_feats requires deform_plate_store.")
    sample = raw.deform_plate_store._load(int(idx))
    node_feats = build_node_feature_matrix(
        sample["node_type"],
        sample["u_prescribed"],
    )
    feats = torch.from_numpy(node_feats).float()
    prescribed = feats[:, ONEHOT_DIM:]
    prescribed = y_normalizer.encode(prescribed)
    feats[:, ONEHOT_DIM:] = prescribed
    if int(feats.shape[0]) != int(num_nodes):
        raise ValueError(f"deform_plate feats node count mismatch: {feats.shape[0]} vs {num_nodes}.")
    return feats


def scatter_plate_laplacian_eigenvectors(
    plate_nodes: np.ndarray,
    plate_eigvecs: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    full = torch.zeros((int(num_nodes), int(plate_eigvecs.shape[-1])), dtype=plate_eigvecs.dtype)
    full[torch.from_numpy(np.asarray(plate_nodes, dtype=np.int64))] = plate_eigvecs
    return full


def compute_deform_plate_plate_laplacian_eigendecomp(
    raw: GinotRawDataset,
    idx: int,
    pos: torch.Tensor,
    name: str,
    count: int,
    *,
    init_eigenvectors: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from pdebench.dataset.laplacian import compute_laplacian_eigendecomp_part

    if raw.deform_plate_store is None:
        raise ValueError("compute_deform_plate_plate_laplacian_eigendecomp requires deform_plate_store.")
    sample = raw.deform_plate_store._load(int(idx))
    plate_nodes = np.asarray(sample["plate_nodes"], dtype=np.int64)
    plate_cells = normalize_cells(sample["plate_cells"], num_nodes=int(plate_nodes.shape[0]))
    if plate_cells is None:
        raise ValueError(f"deform_plate sample {idx} is missing plate cell connectivity.")
    plate_mesh_pos = torch.from_numpy(np.asarray(sample["plate_mesh_pos"], dtype=np.float32)).float()
    plate_edge_index = build_edge_index_from_normalized_cells(
        plate_cells,
        num_nodes=int(plate_nodes.shape[0]),
        perimeter_edges=False,
    )
    if name.startswith("fem"):
        cells_tensor = torch.from_numpy(plate_cells).long()
        plate_eigvals, plate_eigvecs = compute_laplacian_eigendecomp_part(
            plate_edge_index,
            plate_mesh_pos.to(device=plate_edge_index.device),
            cells_tensor,
            name,
            count,
            init_eigenvectors=init_eigenvectors,
        )
    else:
        # Graph Laplacian uses reference plate geometry (mesh_pos), not dynamic world_pos[0].
        plate_pos = plate_mesh_pos.to(device=plate_edge_index.device)
        plate_init = None
        if init_eigenvectors is not None:
            plate_init = init_eigenvectors[torch.from_numpy(plate_nodes).long()]
        plate_eigvals, plate_eigvecs = compute_laplacian_eigendecomp_part(
            plate_edge_index,
            plate_pos,
            None,
            name,
            count,
            init_eigenvectors=plate_init,
        )
    full_eigvecs = scatter_plate_laplacian_eigenvectors(plate_nodes, plate_eigvecs.cpu(), int(pos.shape[0]))
    return plate_eigvals.cpu(), full_eigvecs


@lru_cache(maxsize=4)
def _cached_store(data_root: str) -> _DeformPlateNpzStore:
    cache_root = deform_plate_cache_dir(data_root)
    return _DeformPlateNpzStore(cache_root)


def load_deform_plate_equilibrium_arrays(raw: GinotRawDataset, idx: int) -> dict[str, np.ndarray]:
    if raw.deform_plate_store is None:
        raise ValueError("load_deform_plate_equilibrium_arrays requires deform_plate_store.")
    return raw.deform_plate_store._load(int(idx))


def load_deform_plate(data_root: str) -> GinotRawDataset:
    store = _cached_store(data_root)
    return GinotRawDataset(
        dataset_dir=str(deform_plate_cache_dir(data_root)),
        query_points=DeformPlateSequence(store, "query_points"),
        point_clouds=DeformPlateSequence(store, "point_clouds"),
        targets=DeformPlateSequence(store, "targets"),
        cells=DeformPlateSequence(store, "cells"),
        input_params=None,
        target_fields=target_fields(TARGET_FIELD_PREFIX, 3),
        space_dim=3,
        normalize_pos=False,
        normalize_boundary_pos=False,
        deform_plate_store=store,
        mesh_pos=DeformPlateSequence(store, "mesh_pos"),
        node_types=DeformPlateSequence(store, "node_type"),
        free_mask=DeformPlateSequence(store, "free_mask"),
        bc_mask=DeformPlateSequence(store, "bc_mask"),
        u_prescribed=DeformPlateSequence(store, "u_prescribed"),
        plate_nodes=DeformPlateSequence(store, "plate_nodes"),
        plate_cells=DeformPlateSequence(store, "plate_cells"),
        plate_mesh_pos=DeformPlateSequence(store, "plate_mesh_pos"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deform_plate GINOT cache from TFRecords.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.build_cache:
        parser.error("Pass --build-cache to convert TFRecords into ginot_cache/.")
    cache_root = run_deform_plate_cache_build(args.data_root, overwrite=bool(args.overwrite))
    print(f"Wrote deform_plate GINOT cache under {cache_root}")


if __name__ == "__main__":
    main()
