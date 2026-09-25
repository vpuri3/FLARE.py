"""DrivAerML surface dataset: full-mesh storage + paper-style amortized training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import pdebench

SUBSET_SIZE = 100_000
NUM_PARTS = 80  # default K at SUBSET_SIZE; was 20 with PART_STRIDE=4
PART_BUDGET = NUM_PARTS * SUBSET_SIZE  # 8_000_000
# Eval / ts3 stride parts ``k::NUM_PARTS`` are expected near this cell count
# (~mean mesh / 80 ≈ 8.6M / 80).
TARGET_PART_SIZE = 100_000  # ~ideal mesh/K at defaults
PART_SIZE_REL_TOL = 0.10
STATS_READY = True
ARRAY_P = "pMeanTrim"
ARRAY_TAU = "wallShearStressMeanTrim"
XYZ_MIN = np.array([-0.9425785541534424, -1.131676197052002, -0.31757670640945435], dtype=np.float32)
XYZ_MAX = np.array([4.132784843444824, 1.131676197052002, 1.249011754989624], dtype=np.float32)
# Train-split float64 moments over all surface cells (population mean/std).
# Do not recompute via float32 concat+mean — that saturates (~3e9 cells) and
# historically produced Y_MEAN≈-10 / Y_STD≈132 instead of ≈-230 / ≈269.
# Baked on the previous 400-run train split (including run_45); after removing
# run_45 the moments are slightly stale (~1/400) and may be recomputed later via prep.
Y_MEAN = np.array(
    [-229.78107299, -1.20065446, 0.0015083475, -0.0720732683],
    dtype=np.float32,
)
Y_STD = np.array(
    [269.38979926, 2.07687109, 1.35638411, 1.11433413],
    dtype=np.float32,
)
SPLIT_PATH = Path(__file__).with_name("splits") / "drivaerml.json"
_EPS = 1e-8


def load_drivaerml_split(path: str | Path) -> dict:
    """Read the pinned AB-UPT DrivAerML split."""
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


def num_parts_from_subset_size(subset_size: int) -> int:
    """Derive stride part count ``K = PART_BUDGET // subset_size`` after divisor checks."""
    size = int(subset_size)
    if size <= 0:
        raise ValueError(f"subset_size must be positive, got {size}")
    if size % SUBSET_SIZE != 0:
        raise ValueError(f"subset_size must be a multiple of {SUBSET_SIZE}, got {size}")
    if PART_BUDGET % size != 0:
        raise ValueError(
            f"subset_size={size} does not divide PART_BUDGET={PART_BUDGET} into an integer K"
        )
    return PART_BUDGET // size


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


def sample_train_indices(
    n: int,
    *,
    subset_size: int,
    num_parts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Strided part plus optional IID pad to ``min(subset_size, n)`` when the part is short."""
    part = sample_indices_strided_part(n, num_parts=num_parts, rng=rng)
    need = int(subset_size) - int(part.shape[0])
    if need > 0:
        mask = np.ones(n, dtype=bool)
        mask[part] = False
        complement = np.flatnonzero(mask)
        take = min(need, int(complement.shape[0]))
        if take > 0:
            pad = rng.choice(complement, size=take, replace=False)
            part = np.concatenate([part, pad.astype(np.int64, copy=False)])
    return np.sort(part)


def stride_part_lengths(n: int, num_parts: int = NUM_PARTS) -> list[int]:
    """Cell counts for stride parts ``k::num_parts`` (same split as ts3 eval)."""
    if n < 0:
        raise ValueError(f"expected n >= 0, got {n}")
    if num_parts < 1:
        raise ValueError(f"num_parts must be >= 1, got {num_parts}")
    return [len(range(part_id, n, num_parts)) for part_id in range(num_parts)]


def surprising_stride_parts(
    n: int,
    num_parts: int = NUM_PARTS,
    *,
    target: int = TARGET_PART_SIZE,
    rel_tol: float = PART_SIZE_REL_TOL,
) -> list[tuple[int, int]]:
    """Return ``(part_id, length)`` for empty parts or length outside ``target±rel_tol``."""
    if target < 1:
        raise ValueError(f"target must be >= 1, got {target}")
    if rel_tol < 0:
        raise ValueError(f"rel_tol must be >= 0, got {rel_tol}")
    lo = target * (1.0 - rel_tol)
    hi = target * (1.0 + rel_tol)
    surprises: list[tuple[int, int]] = []
    for part_id, length in enumerate(stride_part_lengths(n, num_parts)):
        if length == 0 or length < lo or length > hi:
            surprises.append((part_id, length))
    return surprises


class DrivAerMLSurfaceRunDataset(torch.utils.data.Dataset):
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
            raise ValueError(f"invalid DrivAerML surface {run_id}: {error}") from error
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
            raise ValueError(f"invalid DrivAerML surface {run_id}: {error}") from error


class DrivAerMLSurfaceAmortizedTrainDataset(torch.utils.data.Dataset):
    """Paper geometry-amortized training with configurable cell sampling."""

    def __init__(
        self,
        run_dataset: DrivAerMLSurfaceRunDataset,
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
            indices = sample_train_indices(
                n,
                subset_size=self.subset_size,
                num_parts=self.num_parts,
                rng=self._rng(idx),
            )
        elif self.sampling == "iid":
            indices = sample_indices_without_replacement(n, self.subset_size, rng=self._rng(idx))
            indices = np.sort(indices)
        else:
            raise ValueError(f"unknown sampling={self.sampling!r}")
        return self.run_dataset._encode(xyz[indices], normals[indices], pressure[indices], tau[indices])


def _max_subset_length(run_dataset: DrivAerMLSurfaceRunDataset, subset_size: int) -> int:
    if len(run_dataset) == 0:
        return 0
    return min(subset_size, max(run_dataset._mesh_length(run_id) for run_id in run_dataset.source_run_ids))


def _max_strided_length(
    run_dataset: DrivAerMLSurfaceRunDataset, num_parts: int, subset_size: int
) -> int:
    if len(run_dataset) == 0:
        return 0
    return max(
        max(
            (run_dataset._mesh_length(run_id) + num_parts - 1) // num_parts,
            min(subset_size, run_dataset._mesh_length(run_id)),
        )
        for run_id in run_dataset.source_run_ids
    )


def load_drivaerml_surface_dataset(
    data_root: str,
    *,
    subset_size: int | None = None,
    iid_samples: bool = True,
) -> tuple[DrivAerMLSurfaceAmortizedTrainDataset, DrivAerMLSurfaceAmortizedTrainDataset, dict]:
    """Load DrivAerML surface data with amortized train/test samples.

    Train sampling defaults to IID; set ``iid_samples=False`` for strided parts.
    Test is always IID. Only ``split["train"]`` and ``split["test"]`` are wired;
    ``val`` / ``hidden_test`` stay unused.
    """
    if not STATS_READY:
        raise RuntimeError(
            "DrivAerML surface statistics are not ready; run surface prep and paste stats into "
            "pdebench/dataset/drivaerml_surface.py."
        )

    split = load_drivaerml_split(SPLIT_PATH)
    # Explicitly ignore val/hidden_test — fullbatch statsfun only sees train/test run data.
    train_ids = list(split["train"])
    test_ids = list(split["test"])
    val_ids = set(split.get("val") or [])
    if val_ids & set(train_ids) or val_ids & set(test_ids):
        raise ValueError("drivaerml split val overlaps train/test; val must remain unused")
    resolved = SUBSET_SIZE if subset_size is None else int(subset_size)
    if resolved <= 0:
        raise ValueError(f"subset_size must be positive, got {resolved}")
    use_iid_train = bool(iid_samples)
    train_runs = DrivAerMLSurfaceRunDataset(data_root, train_ids)
    test_runs = DrivAerMLSurfaceRunDataset(data_root, test_ids)
    if use_iid_train:
        train_dataset = DrivAerMLSurfaceAmortizedTrainDataset(
            train_runs, subset_size=resolved, sampling="iid"
        )
        max_length = max(
            _max_subset_length(train_runs, resolved),
            _max_subset_length(test_runs, resolved),
        )
        num_parts_meta = None
    else:
        k = num_parts_from_subset_size(resolved)
        train_dataset = DrivAerMLSurfaceAmortizedTrainDataset(
            train_runs, subset_size=resolved, sampling="strided_parts", num_parts=k
        )
        max_length = max(
            _max_strided_length(train_runs, k, resolved),
            _max_subset_length(test_runs, resolved),
        )
        num_parts_meta = k
    test_dataset = DrivAerMLSurfaceAmortizedTrainDataset(test_runs, subset_size=resolved, sampling="iid")
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
        "drivaerml_subset_size": resolved,
        "drivaerml_iid_samples": use_iid_train,
        "drivaerml_train_run_data": train_runs,
        "drivaerml_test_run_data": test_runs,
        "y_field_slices": {"p": slice(0, 1), "tau": slice(1, None)},
        "y_field_metrics": {
            "p": {"kind": "slice", "slice": slice(0, 1)},
            "tau": {"kind": "vector_norm", "slice": slice(1, None)},
        },
    }
    if num_parts_meta is not None:
        metadata["drivaerml_num_parts"] = num_parts_meta
    return train_dataset, test_dataset, metadata
