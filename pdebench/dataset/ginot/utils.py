from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pdebench.dataset.ginot.types import FIXED_DATASET_DIRNAME, GinotRawDataset, StandardNormalizer


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def require_files(*paths: str) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Could not find required GINOT file(s): {', '.join(missing)}")


def as_float_array(x, dims: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if dims is not None:
        arr = arr[..., :dims]
    return arr.astype(np.float32, copy=False)


def as_sample_list(values, dims: int | None = None):
    del dims
    # Keep large dense GINOT arrays lazy. Eager conversion can require hundreds of GB.
    return values


def target_fields(prefix: str, dim: int) -> tuple[str, ...]:
    if dim == 1:
        return (prefix,)
    if prefix == "mises_disp" and dim == 3:
        return ("mises_stress", "disp_x", "disp_y")
    return tuple(f"{prefix}_{idx}" for idx in range(dim))


def target_dim(target_sample) -> int:
    return int(as_float_array(target_sample).shape[-1])


def shift_scaler_normalizer(shift, scaler) -> StandardNormalizer:
    return StandardNormalizer(
        mean=torch.as_tensor(shift, dtype=torch.float32),
        std=torch.as_tensor(scaler, dtype=torch.float32),
    )


def split_indices(num_samples: int, seed: int, train_size: int, test_size: int) -> tuple[list[int], list[int]]:
    required = train_size + test_size
    if num_samples < required:
        raise ValueError(
            f"GINOT split requires at least {required} samples for {train_size} train and {test_size} test; "
            f"found {num_samples}."
        )
    perm = np.random.default_rng(seed).permutation(np.arange(num_samples, dtype=np.int64))
    return perm[:train_size].tolist(), perm[train_size:train_size + test_size].tolist()


def fractional_split_indices(num_samples: int, seed: int, test_size: float) -> tuple[list[int], list[int]]:
    train_ids, test_ids = train_test_split(np.arange(num_samples), test_size=test_size, random_state=seed)
    return train_ids.tolist(), test_ids.tolist()


def num_indexable_samples(raw: GinotRawDataset) -> int:
    return len(raw.query_points)


def compute_normalizer(raw: GinotRawDataset, indices: list[int], getter) -> StandardNormalizer:
    total = None
    total_sq = None
    count = 0
    for idx in indices:
        arr = torch.from_numpy(getter(raw, idx)).float()
        if total is None:
            total = arr.sum(dim=0, keepdim=True)
            total_sq = (arr * arr).sum(dim=0, keepdim=True)
        else:
            total += arr.sum(dim=0, keepdim=True)
            total_sq += (arr * arr).sum(dim=0, keepdim=True)
        count += int(arr.shape[0])
    if total is None or total_sq is None or count == 0:
        raise ValueError("Cannot build GINOT normalizer from an empty split.")
    mean = total / count
    var = (total_sq / count) - mean * mean
    std = torch.sqrt(var.clamp_min(0.0)).clamp_min(1e-8)
    return StandardNormalizer(mean=mean, std=std)


def compute_minmax_normalizer(raw: GinotRawDataset, indices: list[int], getters) -> StandardNormalizer:
    coord_min = None
    coord_max = None
    for idx in indices:
        for getter in getters:
            arr = torch.from_numpy(getter(raw, idx)).float()
            sample_min = arr.amin(dim=0, keepdim=True)
            sample_max = arr.amax(dim=0, keepdim=True)
            coord_min = sample_min if coord_min is None else torch.minimum(coord_min, sample_min)
            coord_max = sample_max if coord_max is None else torch.maximum(coord_max, sample_max)
    if coord_min is None or coord_max is None:
        raise ValueError("Cannot build GINOT min/max normalizer from an empty split.")
    scale = (coord_max - coord_min).clamp_min(1e-8)
    return StandardNormalizer(mean=coord_min, std=scale)


def is_micro_puc_fixed_raw(raw: GinotRawDataset) -> bool:
    return os.path.basename(os.path.normpath(str(raw.dataset_dir))) == FIXED_DATASET_DIRNAME


def is_micro_puc_raw(raw: GinotRawDataset) -> bool:
    return os.path.basename(os.path.normpath(str(raw.dataset_dir))) == "PeriodUnitCell"


def is_deform_plate_raw(raw: GinotRawDataset) -> bool:
    return raw.deform_plate_store is not None


def is_bumper_beam_raw(raw: GinotRawDataset) -> bool:
    return raw.bumper_beam_store is not None


def identity_normalizer(space_dim: int) -> StandardNormalizer:
    return StandardNormalizer(mean=torch.zeros(1, space_dim), std=torch.ones(1, space_dim))
