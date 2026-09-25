from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pyvista as pv
import torch

from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer

REQUIRED_TIMES = tuple(range(0, 101, 10))
TARGET_TIMES = REQUIRED_TIMES[1:]
EXPECTED_RUNS = 131
GLOBAL_FIELDS = ("velocity_x", "thickness_scale", "rwall_origin_y")
BUMPER_BEAM_CACHE_TAG = "curated_vtp_v1_n7_y50_80_20"
TARGET_FIELDS = tuple(
    field
    for time in TARGET_TIMES
    for field in (
        f"position_x_t{time}",
        f"position_y_t{time}",
        f"position_z_t{time}",
        f"effective_plastic_strain_t{time}",
        f"stress_vm_t{time}",
    )
)


@dataclass(frozen=True)
class BumperBeamRun:
    reference_points: np.ndarray
    positions: np.ndarray
    thickness: np.ndarray
    global_features: np.ndarray | None
    cells: np.ndarray
    targets: np.ndarray


def _run_number(path: Path) -> int:
    match = re.fullmatch(r"Run(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Expected bumper-beam filename RunNNN.vtp, got {path.name!r}.")
    return int(match.group(1))


def discover_bumper_beam_files(data_root: str) -> tuple[Path, list[Path]]:
    curated = Path(data_root) / "bumper_beam" / "CURATED_DATA_VTP"
    if not curated.is_dir():
        curated = Path(data_root) / "CURATED_DATA_VTP"
    metadata = curated / "GLOBAL_FEATURES.json"
    paths = [*curated.glob("TRAINING_DATA/*.vtp"), *curated.glob("VALIDATION_DATA/*.vtp")]
    paths.sort(key=_run_number)
    if not metadata.is_file() or not paths:
        raise FileNotFoundError(f"Missing curated bumper-beam data below {curated}.")
    return metadata, paths


def _polygon_cells(mesh: pv.PolyData, path: Path) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    rows: list[np.ndarray] = []
    cursor = 0
    while cursor < faces.size:
        width = int(faces[cursor])
        row = faces[cursor + 1 : cursor + 1 + width]
        if width < 3 or row.size != width:
            raise ValueError(f"{path.name}: invalid polygon at flattened offset {cursor}.")
        rows.append(row)
        cursor += width + 1
    if not rows:
        raise ValueError(f"{path.name}: expected non-empty polygon connectivity.")
    max_width = max(int(row.size) for row in rows)
    cells = np.full((len(rows), max_width), -1, dtype=np.int64)
    for row_idx, row in enumerate(rows):
        cells[row_idx, : row.size] = row
    return cells


def _validate_run(
    path: Path,
    reference: np.ndarray,
    positions: np.ndarray,
    thickness: np.ndarray,
    cells: np.ndarray,
    targets: np.ndarray,
) -> None:
    num_nodes = int(reference.shape[0])
    expected = {
        "reference_points": (num_nodes, 3),
        "positions": (len(REQUIRED_TIMES), num_nodes, 3),
        "thickness": (num_nodes, 1),
        "targets": (num_nodes, 50),
    }
    actual = {
        "reference_points": reference.shape,
        "positions": positions.shape,
        "thickness": thickness.shape,
        "targets": targets.shape,
    }
    bad_shapes = {key: (actual[key], shape) for key, shape in expected.items() if actual[key] != shape}
    if num_nodes == 0 or bad_shapes:
        raise ValueError(f"{path.name}: inconsistent bumper-beam shapes: {bad_shapes}.")
    arrays = (reference, positions, thickness, targets)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path.name}: non-finite values in required bumper-beam arrays.")
    if cells.ndim != 2 or cells.shape[0] == 0:
        raise ValueError(f"{path.name}: polygon connectivity contains invalid node indices.")
    for row in cells:
        valid = row[row >= 0]
        if valid.size < 3:
            raise ValueError(f"{path.name}: polygon connectivity must keep at least three nodes per cell.")
        if np.any(row[: valid.size] < 0) or np.any(row[valid.size:] >= 0):
            raise ValueError(f"{path.name}: polygon connectivity must use trailing-only padding.")
        if np.any(valid >= num_nodes):
            raise ValueError(f"{path.name}: polygon connectivity contains invalid node indices.")


def read_bumper_beam_run(path: Path) -> BumperBeamRun:
    mesh = pv.read(path)
    reference = np.asarray(mesh.points, dtype=np.float32)
    required_point = [f"displacement_t{time}.000" for time in REQUIRED_TIMES] + ["thickness"]
    required_cell = [
        name
        for time in REQUIRED_TIMES
        for name in (f"cell_effective_plastic_strain_t{time}.000", f"cell_stress_vm_t{time}.000")
    ]
    missing = [name for name in required_point if name not in mesh.point_data]
    missing += [name for name in required_cell if name not in mesh.cell_data]
    if missing:
        raise ValueError(f"{path.name}: missing required arrays: {missing}.")

    point_mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
    positions = np.stack(
        [
            reference
            if time == 0
            else reference + np.asarray(mesh.point_data[f"displacement_t{time}.000"], dtype=np.float32)
            for time in REQUIRED_TIMES
        ]
    ).astype(np.float32, copy=False)

    dynamic = []
    for state_index, time in enumerate(TARGET_TIMES, start=1):
        strain = np.asarray(
            point_mesh.point_data[f"cell_effective_plastic_strain_t{time}.000"],
            dtype=np.float32,
        ).reshape(-1, 1)
        stress = np.asarray(
            point_mesh.point_data[f"cell_stress_vm_t{time}.000"],
            dtype=np.float32,
        ).reshape(-1, 1)
        dynamic.append(np.concatenate((positions[state_index], strain, stress), axis=1))

    thickness = np.asarray(mesh.point_data["thickness"], dtype=np.float32).reshape(-1, 1)
    cells = _polygon_cells(mesh, path)
    targets = np.concatenate(dynamic, axis=1).astype(np.float32, copy=False)
    _validate_run(path, reference, positions, thickness, cells, targets)
    return BumperBeamRun(
        reference_points=reference,
        positions=positions,
        thickness=thickness,
        global_features=None,
        cells=cells,
        targets=targets,
    )


class BumperBeamStore:
    """Decoded curated VTP runs shared by raw and edge GINOT paths.

    Both flows eventually read through this store:

    - ``include_edges=False`` (e.g. GeoTransolver): ``GinotDataset`` hits VTPs
      every training step.
    - ``include_edges=True`` (e.g. GLT): LMDB graph-cache *build* reads VTPs
      (+ cells) once; warm-cache training then uses ``GraphCacheDataset``.

    Keep an unbounded decode cache — ``EXPECTED_RUNS`` is small (~0.5GB on
    disk). A tiny LRU re-parses VTPs every miss and dominates step time.
    Eager ``preload()`` is opt-in via ``preload_bumper_beam_store`` so warm
    edge-cache startups are not forced to decode all runs again.
    """

    def __init__(self, paths: list[Path], global_features: dict[str, dict[str, float]]):
        self.paths = tuple(paths)
        self.global_features = global_features

    def __len__(self) -> int:
        return len(self.paths)

    @lru_cache(maxsize=None)
    def load(self, idx: int) -> BumperBeamRun:
        path = self.paths[int(idx)]
        run = read_bumper_beam_run(path)
        try:
            values = np.array([self.global_features[path.stem][key] for key in GLOBAL_FIELDS], dtype=np.float32)
        except KeyError as exc:
            raise ValueError(f"{path}: missing global metadata key {exc.args[0]!r}.") from exc
        return replace(run, global_features=values)

    def preload(self) -> None:
        """Decode every curated run once (idempotent; fills ``load`` cache)."""
        for idx in range(len(self)):
            self.load(idx)


def preload_bumper_beam_store(raw: GinotRawDataset | None) -> None:
    """Warm ``BumperBeamStore`` when present; no-op for other GINOT datasets."""
    store = getattr(raw, "bumper_beam_store", None) if raw is not None else None
    if store is not None:
        store.preload()


class BumperBeamSequence:
    def __init__(self, store: BumperBeamStore, field: Literal["reference_points", "targets", "cells"]):
        self.store, self.field = store, field

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int):
        return getattr(self.store.load(int(idx)), self.field)


def compute_bumper_beam_normalizers(
    raw: GinotRawDataset,
    train_ids: list[int],
) -> tuple[StandardNormalizer, StandardNormalizer, StandardNormalizer, StandardNormalizer]:
    store = raw.bumper_beam_store
    if store is None or not train_ids:
        raise ValueError("bumper_beam normalizers require a non-empty training split and bumper_beam_store.")
    coord_min = np.full((3,), np.inf, dtype=np.float64)
    coord_max = np.full((3,), -np.inf, dtype=np.float64)
    thickness_sum = thickness_sq = thickness_count = 0.0
    globals_rows = []
    for idx in train_ids:
        run = store.load(int(idx))
        coords = run.positions.reshape(-1, 3).astype(np.float64)
        coord_min = np.minimum(coord_min, coords.min(axis=0))
        coord_max = np.maximum(coord_max, coords.max(axis=0))
        thickness = run.thickness.astype(np.float64)
        thickness_sum += float(thickness.sum())
        thickness_sq += float(np.square(thickness).sum())
        thickness_count += float(thickness.size)
        globals_rows.append(run.global_features)
    pos = StandardNormalizer(
        mean=torch.from_numpy(coord_min.astype(np.float32)).reshape(1, 3),
        std=torch.from_numpy(np.maximum(coord_max - coord_min, 1e-8).astype(np.float32)).reshape(1, 3),
    )
    thickness_mean = thickness_sum / thickness_count
    thickness_std = max((thickness_sq / thickness_count - thickness_mean**2) ** 0.5, 1e-8)
    globals_np = np.stack(globals_rows).astype(np.float32)
    feat_mean = np.concatenate(([thickness_mean], globals_np.mean(axis=0))).astype(np.float32)
    feat_std = np.concatenate(([thickness_std], globals_np.std(axis=0).clip(min=1e-8))).astype(np.float32)
    feats = StandardNormalizer(torch.from_numpy(feat_mean).reshape(1, 4), torch.from_numpy(feat_std).reshape(1, 4))
    identity_y = StandardNormalizer(torch.zeros(1, 50), torch.ones(1, 50))
    return pos, pos, identity_y, feats


def encode_bumper_beam_target(raw: GinotRawDataset, idx: int, pos_normalizer: StandardNormalizer) -> torch.Tensor:
    store = raw.bumper_beam_store
    if store is None:
        raise ValueError("bumper_beam target encoding requires bumper_beam_store.")
    target = torch.from_numpy(store.load(int(idx)).targets.copy()).float()
    for base in range(0, 50, 5):
        target[:, base : base + 3] = pos_normalizer.encode(target[:, base : base + 3])
    return target


def decode_bumper_beam_target(target: torch.Tensor, pos_normalizer: StandardNormalizer) -> torch.Tensor:
    """Inverse of ``encode_bumper_beam_target`` for position channels (strain/stress unchanged)."""
    out = target.detach().clone().float()
    width = int(out.shape[-1])
    for base in range(0, width, 5):
        out[..., base : base + 3] = pos_normalizer.decode(out[..., base : base + 3])
    return out


def encode_bumper_beam_feats(
    raw: GinotRawDataset,
    idx: int,
    feats_normalizer: StandardNormalizer,
    num_nodes: int,
) -> torch.Tensor:
    store = raw.bumper_beam_store
    if store is None:
        raise ValueError("bumper_beam feature encoding requires bumper_beam_store.")
    run = store.load(int(idx))
    globals_by_node = np.broadcast_to(run.global_features.reshape(1, 3), (num_nodes, 3))
    raw_feats = torch.from_numpy(np.concatenate((run.thickness, globals_by_node), axis=1).copy()).float()
    return feats_normalizer.encode(raw_feats)


def load_bumper_beam(data_root: str) -> GinotRawDataset:
    metadata_path, paths = discover_bumper_beam_files(data_root)
    global_features = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(paths) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} curated bumper-beam VTP runs, found {len(paths)}.")
    missing = [path.stem for path in paths if path.stem not in global_features]
    if missing:
        raise ValueError(f"GLOBAL_FEATURES.json is missing runs: {missing}.")
    invalid = {
        path.stem: [field for field in GLOBAL_FIELDS if field not in global_features[path.stem]]
        for path in paths
        if any(field not in global_features[path.stem] for field in GLOBAL_FIELDS)
    }
    if invalid:
        raise ValueError(f"GLOBAL_FEATURES.json has missing scalar fields: {invalid}.")
    store = BumperBeamStore(paths, global_features)
    return GinotRawDataset(
        dataset_dir=str(metadata_path.parent.parent),
        query_points=BumperBeamSequence(store, "reference_points"),
        point_clouds=BumperBeamSequence(store, "reference_points"),
        targets=BumperBeamSequence(store, "targets"),
        cells=BumperBeamSequence(store, "cells"),
        input_params=None,
        target_fields=TARGET_FIELDS,
        space_dim=3,
        normalize_targets=False,
        bumper_beam_store=store,
    )
