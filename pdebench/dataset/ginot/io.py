from __future__ import annotations

import json
import os

import numpy as np

from pdebench.dataset.ginot.bumper_beam import load_bumper_beam
from pdebench.dataset.ginot.deform_plate import load_deform_plate
from pdebench.dataset.ginot.mesh import build_micro_puc_mesh_idx
from pdebench.dataset.ginot.storage import ShardedMicroPucFixedSequence
from pdebench.dataset.ginot.types import (
    FIXED_DATASET_DIRNAME,
    MICRO_PUC_MESH_SAMPLES,
    MICRO_PUC_TOTAL_SAMPLES,
    GinotRawDataset,
)
from pdebench.dataset.ginot.utils import (
    as_sample_list,
    load_pickle,
    require_files,
    shift_scaler_normalizer,
    target_dim,
    target_fields,
)
from pdebench.dataset.lpbf import load_lpbf_ginot


def load_poisson(data_root: str) -> GinotRawDataset:
    path = os.path.join(data_root, "poisson", "poisson_geo_unstruc_msh.pkl")
    require_files(path)

    data = load_pickle(path)
    targets = as_sample_list(data["solutions"])
    return GinotRawDataset(
        dataset_dir=os.path.dirname(path),
        query_points=as_sample_list(data["nodes"], dims=2),
        point_clouds=as_sample_list(data["point_clouds"], dims=2),
        targets=targets,
        cells=data.get("cells"),
        input_params=None,
        target_fields=target_fields("u", target_dim(targets[0])),
        space_dim=2,
    )


def _cells_fingerprint(cells_i) -> tuple:
    a = np.asarray(cells_i)
    return (a.shape, a.dtype.str, int(a.sum()), int(a.min()), int(a.max()), hash(a.tobytes()))


def _validate_poisson_structured_shared_mesh(query_points, cells) -> None:
    n = len(query_points)
    if cells is None or len(cells) != n:
        raise ValueError(
            f"poisson_structured requires cells list length == n_samples ({n}); "
            f"got {0 if cells is None else len(cells)}."
        )
    n0 = int(np.asarray(query_points[0]).shape[0])
    for i in range(n):
        ni = int(np.asarray(query_points[i]).shape[0])
        if ni != n0:
            raise ValueError(
                f"poisson_structured requires constant node count; sample 0 has {n0}, sample {i} has {ni}."
            )
    fps = {_cells_fingerprint(c) for c in cells}
    if len(fps) != 1:
        raise ValueError(
            "poisson_structured requires a single shared mesh connectivity across all samples; "
            f"found {len(fps)} unique cells fingerprints."
        )


def load_poisson_structured(data_root: str) -> GinotRawDataset:
    path = os.path.join(data_root, "poisson", "poisson_geo_struc_msh.pkl")
    require_files(path)
    data = load_pickle(path)
    targets = as_sample_list(data["solutions"])
    query_points = as_sample_list(data["nodes"], dims=2)
    cells = data.get("cells")
    _validate_poisson_structured_shared_mesh(query_points, cells)
    return GinotRawDataset(
        dataset_dir=os.path.dirname(path),
        query_points=query_points,
        point_clouds=as_sample_list(data["point_clouds"], dims=2),
        targets=targets,
        cells=cells,
        input_params=None,
        target_fields=target_fields("u", target_dim(targets[0])),
        space_dim=2,
    )


def load_micro_puc(data_root: str) -> GinotRawDataset:
    dataset_dir = os.path.join(data_root, "PeriodUnitCell")
    paths = {
        "targets": os.path.join(dataset_dir, "mises_disp_laststep.pkl"),
        "point_cloud": os.path.join(dataset_dir, "points_cloud.pkl"),
        "coords": os.path.join(dataset_dir, "mesh_coords.pkl"),
        "cells": os.path.join(dataset_dir, "mesh_cells10K.pkl"),
        "sample_ids": os.path.join(dataset_dir, "sample_ids.npy"),
    }
    require_files(*paths.values())

    su_data = load_pickle(paths["targets"])
    targets = as_sample_list(su_data["mises_disp"] if isinstance(su_data, dict) else su_data)
    target_normalizer = None
    if isinstance(su_data, dict) and "shift" in su_data and "scaler" in su_data:
        target_normalizer = shift_scaler_normalizer(su_data["shift"], su_data["scaler"])

    cells = load_pickle(paths["cells"])
    if len(cells) < MICRO_PUC_MESH_SAMPLES:
        raise ValueError(
            f"Micro-PUC requires the first {MICRO_PUC_MESH_SAMPLES} mesh cell arrays from mesh_cells10K.pkl; "
            f"found {len(cells)}."
        )

    query_points = as_sample_list(load_pickle(paths["coords"]), dims=2)
    sample_ids = np.load(paths["sample_ids"])
    if len(query_points) != len(sample_ids):
        raise ValueError(
            f"Micro-PUC sample_ids length {len(sample_ids)} does not match mesh_coords length {len(query_points)}."
        )

    return GinotRawDataset(
        dataset_dir=dataset_dir,
        query_points=query_points,
        point_clouds=as_sample_list(load_pickle(paths["point_cloud"]), dims=2),
        targets=targets,
        cells=cells,
        input_params=None,
        target_fields=target_fields("mises_disp", target_dim(targets[0])),
        space_dim=2,
        target_normalizer=target_normalizer,
        normalize_pos=False,
        normalize_boundary_pos=False,
        micro_puc_mesh_idx=build_micro_puc_mesh_idx(sample_ids),
    )


def load_micro_puc_fixed(data_root: str) -> GinotRawDataset:
    dataset_dir = os.path.join(data_root, FIXED_DATASET_DIRNAME)
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Could not find fixed Micro-PUC manifest: {manifest_path}. "
            "Build it with: python -m pdebench.dataset.ginot.micro_puc_fixed --data-root data"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    target_normalizer = None
    if manifest.get("target_shift") is not None and manifest.get("target_scaler") is not None:
        target_normalizer = shift_scaler_normalizer(manifest["target_shift"], manifest["target_scaler"])
    sample_ids_path = os.path.join(data_root, "PeriodUnitCell", "sample_ids.npy")
    if not os.path.exists(sample_ids_path):
        raise FileNotFoundError(
            f"Micro-PUC fixed requires source geometry mapping: {sample_ids_path}. "
            "Expected aligned sample_ids.npy from the original PeriodUnitCell archive."
        )
    source_geometry_ids = np.load(sample_ids_path).astype(np.int64, copy=False)
    if source_geometry_ids.shape[0] != MICRO_PUC_TOTAL_SAMPLES:
        raise ValueError(
            f"Micro-PUC fixed requires {MICRO_PUC_TOTAL_SAMPLES} source geometry ids; "
            f"found {source_geometry_ids.shape[0]} in {sample_ids_path}."
        )
    return GinotRawDataset(
        dataset_dir=dataset_dir,
        query_points=ShardedMicroPucFixedSequence(dataset_dir, "pos"),
        point_clouds=ShardedMicroPucFixedSequence(dataset_dir, "pos"),
        targets=ShardedMicroPucFixedSequence(dataset_dir, "y"),
        cells=ShardedMicroPucFixedSequence(dataset_dir, "cells"),
        input_params=None,
        target_fields=tuple(manifest.get("target_fields", ("mises_stress", "disp_x", "disp_y"))),
        space_dim=2,
        target_normalizer=target_normalizer,
        normalize_pos=False,
        normalize_boundary_pos=False,
        micro_puc_source_geometry_ids=source_geometry_ids,
    )


def load_bracket_lug(data_root: str) -> GinotRawDataset:
    dataset_dir = os.path.join(data_root, "PLASTIC_LUG")
    data_path = os.path.join(dataset_dir, "LUG_node_S_PC.pkl")
    cells_path = os.path.join(dataset_dir, "LUG_cells.pkl")
    params_path = os.path.join(dataset_dir, "input_params.npy")
    require_files(data_path, cells_path, params_path)

    data = load_pickle(data_path)
    params = np.load(params_path).astype(np.float32, copy=False)
    if params.ndim == 1:
        params = params.reshape(params.shape[0], 1)
    targets = as_sample_list(data["nodal_stress"])
    return GinotRawDataset(
        dataset_dir=dataset_dir,
        query_points=as_sample_list(data["vertices"], dims=3),
        point_clouds=as_sample_list(data["points_cloud"], dims=3),
        targets=targets,
        cells=load_pickle(cells_path),
        input_params=params,
        target_fields=target_fields("mises_stress", target_dim(targets[0])),
        space_dim=3,
    )


_LOADERS = {
    "poisson_unstructured": load_poisson,
    "poisson_structured": load_poisson_structured,
    "bracket_lug": load_bracket_lug,
    "micro_puc": load_micro_puc,
    "micro_puc_fixed": load_micro_puc_fixed,
    "deform_plate": load_deform_plate,
    "bumper_beam": load_bumper_beam,
    "lpbf": load_lpbf_ginot,
}


def load_raw_dataset(dataset_name: str, data_root: str) -> GinotRawDataset:
    loader = _LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unsupported GINOT dataset '{dataset_name}'.")
    return loader(data_root)


def expected_micro_puc_fixed_samples() -> int:
    return MICRO_PUC_TOTAL_SAMPLES
