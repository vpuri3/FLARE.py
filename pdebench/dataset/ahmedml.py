"""AhmedML surface dataset: full-mesh storage + paper-style amortized training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import pdebench

SUBSET_SIZE = 100_000
NUM_PARTS = 10  # Transolver-style stride when iid_samples=False
STATS_READY = True
ARRAY_P = "pMean"
ARRAY_TAU = "wallShearStressMean"
XYZ_MIN = np.array([-1.1991983652114868, -0.25003325939178467, 0.049643803387880325], dtype=np.float32)
XYZ_MAX = np.array([0.00018266533152200282, 0.25, 0.36486977338790894], dtype=np.float32)
# Train-split float64 moments over all surface cells (population mean/std).
# Do not recompute via float32 concat+mean — that saturates (~4e8 cells) and
# historically produced Y_MEAN≈-0.036 / Y_STD≈0.148 instead of ≈-0.101 / ≈0.188.
Y_MEAN = np.array(
    [-0.10098538007356549, -0.0015292095383488604, -4.081109131044544e-08, -5.835234911180056e-05],
    dtype=np.float32,
)
Y_STD = np.array(
    [0.1882101637964642, 0.001175221402256852, 0.0006525062926520155, 0.0007134350298949246],
    dtype=np.float32,
)
SPLIT_PATH = Path(__file__).with_name("splits") / "ahmedml.json"
_EPS = 1e-8


def load_ahmedml_split(path: str | Path) -> dict:
    """Read the pinned AhmedML split (AB-UPT / Noether seed-42, 400/50/50).

    Matches Emmi-AI/noether ``AhmedMLDefaultSplitIDs``
    (``torch.randperm(500, seed=42) + 1`` → train/val/test).
    """
    with Path(path).open() as handle:
        return json.load(handle)


def unit_normals(normals: np.ndarray) -> np.ndarray:
    """L2-normalize normals while preserving zero vectors."""
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(norms, _EPS)


def sample_indices_without_replacement(n: int, subset_size: int, *, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample of ``min(subset_size, n)`` distinct cell indices."""
    if n <= 0:
        raise ValueError(f"expected n > 0, got {n}")
    take = min(int(subset_size), int(n))
    return rng.choice(n, size=take, replace=False)


def sample_indices_strided_part(
    n: int,
    *,
    num_parts: int = NUM_PARTS,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one random stride part ``k::num_parts`` as cell indices."""
    if n <= 0:
        raise ValueError(f"expected n > 0, got {n}")
    if num_parts < 1:
        raise ValueError(f"num_parts must be >= 1, got {num_parts}")
    k = int(rng.integers(0, num_parts))
    return np.arange(k, n, num_parts, dtype=np.int64)


class AhmedMLSurfaceRunDataset(torch.utils.data.Dataset):
    """Lazy full-surface reader: one item per run, all cells in original order."""

    def __init__(
        self,
        data_root: str | Path,
        run_ids: list[str],
        *,
        xyz_min: np.ndarray | None = None,
        xyz_max: np.ndarray | None = None,
        y_mean: np.ndarray | None = None,
        y_std: np.ndarray | None = None,
    ) -> None:
        seen: set[str] = set()
        for run_id in run_ids:
            if run_id in seen:
                raise ValueError(f"duplicate source run: {run_id}")
            seen.add(run_id)
        self.data_root = Path(data_root)
        self.source_run_ids = tuple(run_ids)
        self.xyz_min = np.asarray(XYZ_MIN if xyz_min is None else xyz_min, dtype=np.float32)
        self.xyz_max = np.asarray(XYZ_MAX if xyz_max is None else xyz_max, dtype=np.float32)
        self.y_mean = np.asarray(Y_MEAN if y_mean is None else y_mean, dtype=np.float32)
        self.y_std = np.asarray(Y_STD if y_std is None else y_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.source_run_ids)

    def _field_path(self, run_id: str, field: str) -> Path:
        boundary_id = run_id.removeprefix("run_")
        return self.data_root / run_id / f"boundary_{boundary_id}_{field}.npy"

    def _mesh_length(self, run_id: str) -> int:
        return int(np.load(self._field_path(run_id, "points"), mmap_mode="r").shape[0])

    def _load_arrays(self, run_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            xyz = np.load(self._field_path(run_id, "points"), mmap_mode="r")
            normals = np.load(self._field_path(run_id, "normals"), mmap_mode="r")
            pressure = np.load(self._field_path(run_id, "p"), mmap_mode="r")
            tau = np.load(self._field_path(run_id, "tau"), mmap_mode="r")
        except (FileNotFoundError, OSError) as error:
            raise ValueError(f"invalid AhmedML {run_id}: {error}") from error
        return xyz, normals, pressure, tau

    def _encode(
        self,
        xyz: np.ndarray,
        normals: np.ndarray,
        pressure: np.ndarray,
        tau: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = np.asarray(xyz, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        pressure = np.asarray(pressure, dtype=np.float32).reshape(-1, 1)
        tau = np.asarray(tau, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"expected points [N, 3], got {tuple(xyz.shape)}")
        if normals.shape != xyz.shape:
            raise ValueError(f"expected normals {tuple(xyz.shape)}, got {tuple(normals.shape)}")
        if tau.ndim != 2 or tau.shape != (xyz.shape[0], 3):
            raise ValueError(f"expected tau [N, 3], got {tuple(tau.shape)}")
        if pressure.shape[0] != xyz.shape[0]:
            raise ValueError(f"expected pressure length {xyz.shape[0]}, got {pressure.shape[0]}")

        xyz = (xyz - self.xyz_min) / np.maximum(self.xyz_max - self.xyz_min, _EPS)
        x = np.concatenate((xyz, unit_normals(normals)), axis=-1)
        y = np.concatenate((pressure, tau), axis=-1)
        y = (y - self.y_mean) / np.maximum(self.y_std, _EPS)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        run_id = self.source_run_ids[idx]
        try:
            return self._encode(*self._load_arrays(run_id))
        except ValueError as error:
            raise ValueError(f"invalid AhmedML {run_id}: {error}") from error


class AhmedMLSurfaceAmortizedTrainDataset(torch.utils.data.Dataset):
    """Paper geometry-amortized training with IID or strided cell sampling.

    Epoch length equals ``#runs``. Default IID draws ``min(subset_size, N)``
    distinct cells; ``sampling='strided_parts'`` uses ``k::NUM_PARTS`` (K=10).
    """

    def __init__(
        self,
        run_dataset: AhmedMLSurfaceRunDataset,
        *,
        subset_size: int | None = None,
        seed: int | None = None,
        sampling: Literal["iid", "strided_parts"] = "iid",
        num_parts: int | None = None,
    ) -> None:
        resolved = SUBSET_SIZE if subset_size is None else int(subset_size)
        if resolved <= 0:
            raise ValueError(f"subset_size must be positive, got {resolved}")
        self.run_dataset = run_dataset
        self.subset_size = resolved
        self.sampling = sampling
        self.num_parts = NUM_PARTS if num_parts is None else int(num_parts)
        self._seed = seed

    @property
    def source_run_ids(self) -> tuple[str, ...]:
        return self.run_dataset.source_run_ids

    def __len__(self) -> int:
        return len(self.run_dataset)

    def _rng(self, idx: int) -> np.random.Generator:
        if self._seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._seed + int(idx))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        run_id = self.run_dataset.source_run_ids[idx]
        xyz, normals, pressure, tau = self.run_dataset._load_arrays(run_id)
        n = int(xyz.shape[0])
        if self.sampling == "strided_parts":
            indices = sample_indices_strided_part(n, num_parts=self.num_parts, rng=self._rng(idx))
            indices = np.sort(indices)
        elif self.sampling == "iid":
            indices = sample_indices_without_replacement(n, self.subset_size, rng=self._rng(idx))
            # Sort for contiguous mmap reads; set membership is unchanged.
            indices = np.sort(indices)
        else:
            raise ValueError(f"unknown sampling={self.sampling!r}")
        return self.run_dataset._encode(xyz[indices], normals[indices], pressure[indices], tau[indices])


def _max_subset_length(run_dataset: AhmedMLSurfaceRunDataset, subset_size: int) -> int:
    if len(run_dataset) == 0:
        return 0
    return min(subset_size, max(run_dataset._mesh_length(run_id) for run_id in run_dataset.source_run_ids))


def _max_strided_length(run_dataset: AhmedMLSurfaceRunDataset, num_parts: int) -> int:
    if len(run_dataset) == 0:
        return 0
    return max(
        (run_dataset._mesh_length(run_id) + num_parts - 1) // num_parts
        for run_id in run_dataset.source_run_ids
    )


def load_ahmedml_surface_dataset(
    data_root: str,
    *,
    subset_size: int | None = None,
    iid_samples: bool = True,
) -> tuple[AhmedMLSurfaceAmortizedTrainDataset, AhmedMLSurfaceAmortizedTrainDataset, dict]:
    """Load AhmedML with paper-style amortized train/test samples.

    Full-mesh Rel-L2 uses ``ahmedml_*_run_data`` (complete surfaces). Default
    ``iid_samples=True`` IID-subsamples ``min(subset_size, N)`` cells per step;
    ``False`` uses Transolver-style stride parts with ``K=NUM_PARTS``.
    """
    if not STATS_READY:
        raise RuntimeError(
            "AhmedML statistics are not ready; run surface prep and paste stats into pdebench/dataset/ahmedml.py."
        )

    resolved = SUBSET_SIZE if subset_size is None else int(subset_size)
    if resolved <= 0:
        raise ValueError(f"subset_size must be positive, got {resolved}")
    use_iid = bool(iid_samples)
    sampling: Literal["iid", "strided_parts"] = "iid" if use_iid else "strided_parts"

    split = load_ahmedml_split(SPLIT_PATH)
    train_runs = AhmedMLSurfaceRunDataset(data_root, split["train"])
    test_runs = AhmedMLSurfaceRunDataset(data_root, split["test"])
    train_dataset = AhmedMLSurfaceAmortizedTrainDataset(
        train_runs, subset_size=resolved, sampling=sampling, num_parts=NUM_PARTS
    )
    test_dataset = AhmedMLSurfaceAmortizedTrainDataset(
        test_runs, subset_size=resolved, sampling=sampling, num_parts=NUM_PARTS
    )
    if use_iid:
        max_length = max(
            _max_subset_length(train_runs, resolved),
            _max_subset_length(test_runs, resolved),
        )
    else:
        max_length = max(
            _max_strided_length(train_runs, NUM_PARTS),
            _max_strided_length(test_runs, NUM_PARTS),
        )
    y_normalizer = pdebench.UnitGaussianNormalizer(torch.zeros(2, 1, 4))
    y_normalizer.mean = torch.as_tensor(Y_MEAN, dtype=torch.float32).reshape(1, 1, 4)
    y_normalizer.std = torch.as_tensor(Y_STD, dtype=torch.float32).reshape(1, 1, 4)
    metadata = {
        "x_normalizer": pdebench.IdentityNormalizer(),
        "y_normalizer": y_normalizer,
        "c_in": 6,
        "c_out": 4,
        "space_dim": 6,
        "pos_dim": 3,
        "time_cond": False,
        "max_length": max_length,
        "ahmedml_subset_size": resolved,
        "ahmedml_iid_samples": use_iid,
        "ahmedml_train_run_data": train_runs,
        "ahmedml_test_run_data": test_runs,
        "y_field_slices": {"p": slice(0, 1), "tau": slice(1, None)},
        "y_field_metrics": {
            "p": {"kind": "slice", "slice": slice(0, 1)},
            "tau": {"kind": "vector_norm", "slice": slice(1, None)},
        },
    }
    if not use_iid:
        metadata["ahmedml_num_parts"] = NUM_PARTS
    return train_dataset, test_dataset, metadata
