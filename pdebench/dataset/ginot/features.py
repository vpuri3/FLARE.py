from __future__ import annotations

import os

import numpy as np
import torch

from pdebench.dataset.ginot.bumper_beam import encode_bumper_beam_feats
from pdebench.dataset.ginot.deform_plate import FEAT_DIM, encode_deform_plate_feats
from pdebench.dataset.ginot.types import GinotRawDataset, StandardNormalizer
from pdebench.dataset.ginot.utils import is_bumper_beam_raw, is_deform_plate_raw

# bracket_lug input_params columns: 0=Rh, 1=shank (Lx-0.025), 2=Lz, 3=applied load
GEOMETRY_PARAM_COLUMNS: tuple[int, ...] = (0, 1, 2)
LOAD_PARAM_COLUMN: int = 3


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def input_params_include_geometry() -> bool:
    return _env_flag("GINOT_INPUT_PARAMS_INCLUDE_GEOMETRY", False)


def input_params_include_load() -> bool:
    return _env_flag("GINOT_INPUT_PARAMS_INCLUDE_LOAD", True)


def input_params_feat_columns() -> tuple[int, ...]:
    cols: list[int] = []
    if input_params_include_geometry():
        cols.extend(GEOMETRY_PARAM_COLUMNS)
    if input_params_include_load():
        cols.append(LOAD_PARAM_COLUMN)
    return tuple(cols)


def sample_param_dim(raw: GinotRawDataset) -> int:
    if raw.input_params is None:
        return 0
    if isinstance(raw.input_params, np.ndarray):
        params = raw.input_params
        if params.ndim == 1:
            return 1
        return int(params.shape[1])
    row = np.asarray(raw.input_params[0], dtype=np.float32).reshape(-1)
    return int(row.shape[0])


def active_feats_dim(raw: GinotRawDataset) -> int:
    if is_bumper_beam_raw(raw):
        return 4
    if is_deform_plate_raw(raw):
        return FEAT_DIM
    dim = sample_param_dim(raw)
    if dim == 0:
        return 0
    cols = input_params_feat_columns()
    if not cols:
        return 0
    if max(cols) >= dim:
        return dim
    return len(cols)


def read_sample_param_row(raw: GinotRawDataset, idx: int) -> np.ndarray:
    dim = sample_param_dim(raw)
    if dim == 0:
        return np.empty((0,), dtype=np.float32)
    row = np.asarray(raw.input_params[int(idx)], dtype=np.float32).reshape(-1)
    if row.shape[0] != dim:
        raise ValueError(
            f"GINOT sample {idx} input_params has length {row.shape[0]}, expected {dim}."
        )
    return row


def select_input_params_feat_row(row: np.ndarray) -> np.ndarray:
    cols = input_params_feat_columns()
    if not cols:
        return np.empty((0,), dtype=np.float32)
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if max(cols) >= row.shape[0]:
        return row.astype(np.float32, copy=False)
    return row[list(cols)].astype(np.float32, copy=False)


def compute_feats_normalizer(raw: GinotRawDataset, train_ids: list[int]) -> StandardNormalizer | None:
    if is_deform_plate_raw(raw):
        return None
    feat_dim = active_feats_dim(raw)
    if feat_dim == 0:
        return None
    rows = [select_input_params_feat_row(read_sample_param_row(raw, int(idx))) for idx in train_ids]
    train = np.stack(rows, axis=0).astype(np.float32, copy=False)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True).clip(min=1e-8)
    return StandardNormalizer(
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
    )


def encode_deform_plate_boundary_tensors(
    raw: GinotRawDataset,
    idx: int,
    *,
    to_cpu: bool = False,
) -> dict[str, torch.Tensor]:
    store = raw.deform_plate_store
    if store is None:
        raise ValueError("deform_plate sample is missing deform_plate_store.")
    parsed = store._load(int(idx))
    out = {
        "free_mask": torch.from_numpy(parsed["free_mask"]).bool(),
        "bc_mask": torch.from_numpy(parsed["bc_mask"]).bool(),
        "u_prescribed": torch.from_numpy(parsed["u_prescribed"]).float(),
    }
    if to_cpu:
        return {key: value.cpu() for key, value in out.items()}
    return out


def encode_sample_feats(
    raw: GinotRawDataset,
    idx: int,
    feats_normalizer: StandardNormalizer | None,
    num_nodes: int,
    *,
    to_cpu: bool = False,
    y_normalizer: StandardNormalizer | None = None,
) -> torch.Tensor:
    if is_bumper_beam_raw(raw):
        if feats_normalizer is None:
            raise ValueError("bumper_beam feats require a train-fitted feature normalizer.")
        feats = encode_bumper_beam_feats(raw, int(idx), feats_normalizer, int(num_nodes))
        return feats.cpu() if to_cpu else feats
    if is_deform_plate_raw(raw):
        if y_normalizer is None:
            raise ValueError("deform_plate feats require y_normalizer for prescribed displacement scaling.")
        feats = encode_deform_plate_feats(raw, int(idx), y_normalizer, int(num_nodes))
        return feats.cpu() if to_cpu else feats
    if feats_normalizer is None or active_feats_dim(raw) == 0:
        return torch.empty((int(num_nodes), 0), dtype=torch.float32)
    row = torch.from_numpy(select_input_params_feat_row(read_sample_param_row(raw, int(idx)))).float().reshape(1, -1)
    encoded = feats_normalizer.encode(row)
    feats = encoded.expand(int(num_nodes), -1)
    return feats.cpu() if to_cpu else feats


def attach_sample_feats(
    sample: dict[str, torch.Tensor],
    raw: GinotRawDataset,
    idx: int,
    feats_normalizer: StandardNormalizer | None,
    *,
    y_normalizer: StandardNormalizer | None = None,
) -> dict[str, torch.Tensor]:
    need_feats = is_deform_plate_raw(raw) or feats_normalizer is not None
    if need_feats and "feats" not in sample:
        num_nodes = int(sample["pos"].shape[0])
        sample["feats"] = encode_sample_feats(
            raw,
            int(idx),
            feats_normalizer,
            num_nodes,
            y_normalizer=y_normalizer,
        )
    if is_deform_plate_raw(raw):
        missing_bc = any(key not in sample for key in ("free_mask", "bc_mask", "u_prescribed"))
        if missing_bc:
            sample.update(encode_deform_plate_boundary_tensors(raw, int(idx)))
    return sample
