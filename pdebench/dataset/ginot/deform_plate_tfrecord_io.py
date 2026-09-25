"""MeshGraphNets TFRecord reader using the lightweight ``tfrecord`` package (no TensorFlow)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from tfrecord.reader import example_loader
from tfrecord.tools.tfrecord2idx import create_index

_DTYPE_MAP = {
    "int32": np.int32,
    "float32": np.float32,
    "int64": np.int64,
    "float64": np.float64,
}


def _tfrecord_index_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".idx")


def _ensure_tfrecord_index(path: Path) -> Path:
    idx_path = _tfrecord_index_path(path)
    if not idx_path.is_file() or idx_path.stat().st_mtime < path.stat().st_mtime:
        create_index(str(path), str(idx_path))
    return idx_path


def _decode_meshgraphnets_field(
    raw_bytes: bytes,
    field: dict[str, Any],
    *,
    trajectory_length: int,
) -> np.ndarray:
    dtype_name = str(field["dtype"])
    if dtype_name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported MeshGraphNets dtype {dtype_name!r}.")
    arr = np.frombuffer(raw_bytes, dtype=_DTYPE_MAP[dtype_name])
    shape = [int(dim) if int(dim) > 0 else -1 for dim in field["shape"]]
    known_dims = [dim for dim in shape if dim > 0]
    known_prod = int(np.prod(known_dims, dtype=np.int64)) if known_dims else 1
    if any(dim < 0 for dim in shape):
        if arr.size % known_prod != 0:
            raise ValueError(
                f"Cannot infer variable shape {shape} from {arr.size} elements "
                f"(known product {known_prod})."
            )
        inferred = arr.size // known_prod
        shape = [inferred if dim < 0 else dim for dim in shape]
    decoded = arr.reshape(shape)
    if field.get("type") == "static" and decoded.shape[0] == 1:
        decoded = np.repeat(decoded, int(trajectory_length), axis=0)
    return decoded


def decode_meshgraphnets_example(
    raw: dict[str, bytes],
    meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    trajectory_length = int(meta["trajectory_length"])
    sample: dict[str, np.ndarray] = {}
    for key in meta["field_names"]:
        field = meta["features"][key]
        sample[key] = _decode_meshgraphnets_field(
            raw[key],
            field,
            trajectory_length=trajectory_length,
        )
    return sample


def load_meshgraphnets_tfrecord(
    path: str | os.PathLike[str],
    meta: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    tfrecord_path = Path(path)
    if not tfrecord_path.is_file():
        raise FileNotFoundError(f"Missing TFRecord file: {tfrecord_path}")
    idx_path = _ensure_tfrecord_index(tfrecord_path)
    trajectories: list[dict[str, np.ndarray]] = []
    for raw in example_loader(str(tfrecord_path), str(idx_path)):
        trajectories.append(decode_meshgraphnets_example(raw, meta))
    return trajectories
