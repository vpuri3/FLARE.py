"""Pinned schema constants for the NASA-CRM AASM Case 4 HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

import pdebench

GLOBAL_ATTRS = (
    "Mach",
    "AlphaMean",
    "aileronInboard",
    "aileronOutboard",
    "htp",
    "elevator",
)

ARRAY_COORDS = ("CoordinateX", "CoordinateY", "CoordinateZ")
ARRAY_NORMALS = ("NormalX", "NormalY", "NormalZ")
ARRAY_CP = "PressureCoefficient"
ARRAY_CF = ("cfx", "cfy", "cfz")
ARRAY_SURFACE = "Surface"
ARRAY_KEYS = ARRAY_COORDS + ARRAY_NORMALS + (ARRAY_CP, ARRAY_SURFACE) + ARRAY_CF

TRAIN_SAMPLES = 105
TEST_SAMPLES = 44
N = 454_404
CF_COMPONENTS = 3
C_OUT = 1 + CF_COMPONENTS

TRAIN_H5 = "trainingData_NASA-CRM.h5"
TEST_H5 = "testData_NASA-CRM.h5"
_EPS = 1e-8


def _broadcast_globals(globals_vec: np.ndarray, num_nodes: int) -> np.ndarray:
    """Broadcast six scalar operating conditions over all mesh nodes."""
    globals_row = np.asarray(globals_vec, dtype=np.float32).reshape(1, len(GLOBAL_ATTRS))
    return np.broadcast_to(globals_row, (num_nodes, len(GLOBAL_ATTRS))).copy()


def _unit_normals(normals: np.ndarray) -> np.ndarray:
    """L2-normalize each normal vector to unit magnitude."""
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norms, _EPS)


def _read_xyz(group: h5py.Group) -> np.ndarray:
    return np.stack([np.asarray(group[key], dtype=np.float32) for key in ARRAY_COORDS], axis=-1)


def _read_y(group: h5py.Group) -> np.ndarray:
    cp = np.asarray(group[ARRAY_CP], dtype=np.float32).reshape(-1, 1)
    cf = np.stack([np.asarray(group[key], dtype=np.float32) for key in ARRAY_CF], axis=-1)
    return np.concatenate((cp, cf), axis=-1)


class NasaCrmDataset(torch.utils.data.Dataset):
    """Lazy sample-level reader for one NASA-CRM HDF5 split."""

    def __init__(
        self,
        h5_path: str | Path,
        sample_keys: list[str],
        *,
        xyz_min: np.ndarray,
        xyz_max: np.ndarray,
        g_mean: np.ndarray,
        g_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        c_out: int,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.sample_keys = tuple(sample_keys)
        self.xyz_min = np.asarray(xyz_min, dtype=np.float32)
        self.xyz_max = np.asarray(xyz_max, dtype=np.float32)
        self.g_mean = np.asarray(g_mean, dtype=np.float32)
        self.g_std = np.asarray(g_std, dtype=np.float32)
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_std = np.asarray(y_std, dtype=np.float32)
        self.c_out = c_out
        # Open lazily so DataLoader workers get a fresh handle after fork/spawn.
        self._h5: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.sample_keys)

    def _ensure_h5(self) -> h5py.File:
        handle = self._h5
        if handle is None:
            handle = h5py.File(self.h5_path, "r")
            self._h5 = handle
        return handle

    def close(self) -> None:
        handle = self._h5
        if handle is not None:
            handle.close()
            self._h5 = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        group = self._ensure_h5()[self.sample_keys[idx]]
        xyz = _read_xyz(group)
        normals = np.stack(
            [np.asarray(group[key], dtype=np.float32) for key in ARRAY_NORMALS],
            axis=-1,
        )
        raw_globals = np.asarray([group.attrs[key] for key in GLOBAL_ATTRS], dtype=np.float32)
        y = _read_y(group)

        xyz = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + _EPS)
        normals = _unit_normals(normals)
        # Per-channel unit Gaussian on the six operating-condition scalars, then broadcast.
        globals_vec = (raw_globals - self.g_mean) / self.g_std
        x = np.concatenate((xyz, normals, _broadcast_globals(globals_vec, xyz.shape[0])), axis=-1)
        y = (y - self.y_mean) / self.y_std
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def _sample_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(handle.keys())


def _validate_training_schema(train_path: Path) -> None:
    with h5py.File(train_path, "r") as handle:
        sample_keys = sorted(handle.keys())
        if not sample_keys:
            raise ValueError(f"{TRAIN_H5} contains no sample groups")

        first_key = sample_keys[0]
        group = handle[first_key]
        missing_arrays = [name for name in ARRAY_KEYS if name not in group]
        if missing_arrays:
            raise KeyError(
                f"{TRAIN_H5} sample {first_key!r} is missing required arrays: {missing_arrays}"
            )

        missing_attrs = [name for name in GLOBAL_ATTRS if name not in group.attrs]
        if missing_attrs:
            raise KeyError(
                f"{TRAIN_H5} sample {first_key!r} is missing required attributes: {missing_attrs}"
            )


def _training_statistics(
    train_path: Path,
    sample_keys: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    g_sum = np.zeros(len(GLOBAL_ATTRS), dtype=np.float64)
    g_sum_sq = np.zeros(len(GLOBAL_ATTRS), dtype=np.float64)
    y_sum = np.zeros(C_OUT, dtype=np.float64)
    y_sum_sq = np.zeros(C_OUT, dtype=np.float64)
    y_count = 0
    max_n = 0

    with h5py.File(train_path, "r") as handle:
        for key in sample_keys:
            group = handle[key]
            xyz = _read_xyz(group).astype(np.float64, copy=False)
            globals_vec = np.asarray([group.attrs[name] for name in GLOBAL_ATTRS], dtype=np.float64)
            y = _read_y(group).astype(np.float64, copy=False)

            xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
            xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
            g_sum += globals_vec
            g_sum_sq += globals_vec * globals_vec
            y_sum += y.sum(axis=0)
            y_sum_sq += np.square(y).sum(axis=0)
            y_count += y.shape[0]
            max_n = max(max_n, y.shape[0])

    sample_count = len(sample_keys)
    g_mean = g_sum / sample_count
    g_var = np.maximum(g_sum_sq / sample_count - np.square(g_mean), 0.0)
    g_std = np.maximum(np.sqrt(g_var), _EPS)
    y_mean = y_sum / y_count
    y_var = np.maximum(y_sum_sq / y_count - np.square(y_mean), 0.0)
    y_std = np.maximum(np.sqrt(y_var), _EPS)
    return xyz_min, xyz_max, g_mean, g_std, y_mean, y_std, max_n


def _max_nodes(path: Path, sample_keys: list[str]) -> int:
    max_n = 0
    with h5py.File(path, "r") as handle:
        for key in sample_keys:
            max_n = max(max_n, handle[key][ARRAY_CP].shape[0])
    return max_n


def assert_official_split(
    data_root: str | Path,
    *,
    train_samples: int = TRAIN_SAMPLES,
    test_samples: int = TEST_SAMPLES,
) -> None:
    """Verify downloaded NASA-CRM HDF5 files contain the expected sample counts."""
    root = Path(data_root)
    train_path = root / TRAIN_H5
    test_path = root / TEST_H5

    with h5py.File(train_path, "r") as train_file:
        actual = len(train_file.keys())
        assert actual == train_samples, (
            f"{TRAIN_H5} has {actual} sample keys, expected {train_samples}"
        )

    with h5py.File(test_path, "r") as test_file:
        actual = len(test_file.keys())
        assert actual == test_samples, (
            f"{TEST_H5} has {actual} sample keys, expected {test_samples}"
        )


def load_nasa_crm_dataset(
    data_root: str,
    *,
    strict_split: bool = True,
) -> tuple[NasaCrmDataset, NasaCrmDataset, dict]:
    """Load lazy train/test datasets and training-derived normalization statistics."""
    root = Path(data_root)
    train_path = root / TRAIN_H5
    test_path = root / TEST_H5
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    _validate_training_schema(train_path)
    if strict_split:
        assert_official_split(root)

    train_keys = _sample_keys(train_path)
    test_keys = _sample_keys(test_path)
    xyz_min, xyz_max, g_mean, g_std, y_mean, y_std, max_n = _training_statistics(
        train_path,
        train_keys,
    )
    max_n = max(max_n, _max_nodes(test_path, test_keys))

    dataset_kwargs = {
        "xyz_min": xyz_min,
        "xyz_max": xyz_max,
        "g_mean": g_mean,
        "g_std": g_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "c_out": C_OUT,
    }
    train_dataset = NasaCrmDataset(train_path, train_keys, **dataset_kwargs)
    test_dataset = NasaCrmDataset(test_path, test_keys, **dataset_kwargs)

    y_normalizer = pdebench.UnitGaussianNormalizer(torch.zeros(2, 1, C_OUT))
    y_normalizer.mean = torch.as_tensor(y_mean, dtype=torch.float32).reshape(1, 1, C_OUT)
    y_normalizer.std = torch.as_tensor(y_std, dtype=torch.float32).reshape(1, 1, C_OUT)
    metadata = {
        "x_normalizer": pdebench.IdentityNormalizer(),
        "y_normalizer": y_normalizer,
        "c_in": 12,
        "c_out": C_OUT,
        "space_dim": 12,
        "pos_dim": 3,
        "time_cond": False,
        "max_length": max_n,
        # Transolver-3 reporting: cp Rel-L2 on channel 0; cf Rel-L2 on per-node |cf|.
        "y_field_slices": {"cp": slice(0, 1), "cf": slice(1, None)},
        "y_field_metrics": {
            "cp": {"kind": "slice", "slice": slice(0, 1)},
            "cf": {"kind": "vector_norm", "slice": slice(1, None)},
        },
    }
    return train_dataset, test_dataset, metadata
