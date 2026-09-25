from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pdebench.callbacks import _rel_l2_by_field_metrics, _rel_l2_by_slices
from pdebench.dataset import utils as dataset_utils
from pdebench.dataset.nasa_crm import (
    ARRAY_CF,
    ARRAY_COORDS,
    ARRAY_CP,
    ARRAY_KEYS,
    ARRAY_NORMALS,
    ARRAY_SURFACE,
    GLOBAL_ATTRS,
    load_nasa_crm_dataset,
)
from pdebench.utils import RelL2Loss


def _write_split(path: Path, count: int, n: int) -> None:
    with h5py.File(path, "w") as handle:
        for sample_index in range(count):
            group = handle.create_group(f"Sample{sample_index:03d}")
            for channel, key in enumerate(ARRAY_COORDS):
                group.create_dataset(key, data=np.arange(n, dtype=np.float32) + channel)
            for channel, key in enumerate(ARRAY_NORMALS):
                group.create_dataset(key, data=np.full(n, channel + 1, dtype=np.float32))
            group.create_dataset(ARRAY_CP, data=np.arange(n, dtype=np.float32) + sample_index)
            group.create_dataset(ARRAY_SURFACE, data=np.ones(n, dtype=np.int32))
            for channel, key in enumerate(ARRAY_CF):
                group.create_dataset(key, data=np.full(n, sample_index + channel + 1, dtype=np.float32))
            for attr_index, key in enumerate(GLOBAL_ATTRS):
                group.attrs[key] = np.float32(10 * sample_index + attr_index)


def _write_tiny_nasa_crm(root: Path, *, n_train: int = 2, n_test: int = 1, n: int = 8) -> None:
    _write_split(root / "trainingData_NASA-CRM.h5", n_train, n)
    _write_split(root / "testData_NASA-CRM.h5", n_test, n)


def test_nasa_crm_adapter_routes(monkeypatch, tmp_path: Path) -> None:
    from pdebench.dataset.adapters import get_adapter

    calls = {}

    def fake(data_root):
        calls["data_root"] = data_root
        return "train", "test", {"c_in": 12}

    monkeypatch.setattr(dataset_utils, "load_nasa_crm_dataset", fake)
    train, test, meta = get_adapter("nasa_crm").load(str(tmp_path))

    assert calls["data_root"] == str(tmp_path)
    assert (train, test, meta["c_in"]) == ("train", "test", 12)


def test_split_sizes_and_feature_layout(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)

    train, test, meta = load_nasa_crm_dataset(str(tmp_path), strict_split=False)
    x, y = train[0]

    assert len(train) == 2
    assert len(test) == 1
    assert x.shape == (8, 12)
    assert y.shape == (8, meta["c_out"])
    assert meta["c_in"] == 12
    assert meta["space_dim"] == 12
    assert meta["pos_dim"] == 3
    assert meta["max_length"] == 8
    assert meta["y_field_slices"] == {"cp": slice(0, 1), "cf": slice(1, None)}
    assert meta["y_field_metrics"] == {
        "cp": {"kind": "slice", "slice": slice(0, 1)},
        "cf": {"kind": "vector_norm", "slice": slice(1, None)},
    }
    assert torch.allclose(x[:, 6:], x[0:1, 6:].expand_as(x[:, 6:]))
    decoded_y = meta["y_normalizer"].decode(y.unsqueeze(0)).squeeze(0)
    expected_y = torch.column_stack(
        (
            torch.arange(8, dtype=torch.float32),
            torch.ones(8),
            torch.full((8,), 2.0),
            torch.full((8,), 3.0),
        )
    )
    assert torch.allclose(decoded_y, expected_y)
    assert dataset_utils.load_nasa_crm_dataset is load_nasa_crm_dataset


def test_globals_affect_input(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)
    train, _, _ = load_nasa_crm_dataset(str(tmp_path), strict_split=False)

    first_x, _ = train[0]
    second_x, _ = train[1]

    assert not torch.allclose(first_x[:, 6], second_x[:, 6])


def test_persistent_h5_handle_reused(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)
    train, _, _ = load_nasa_crm_dataset(str(tmp_path), strict_split=False)

    assert train._h5 is None
    _ = train[0]
    handle = train._h5
    assert handle is not None
    _ = train[1]
    assert train._h5 is handle
    train.close()
    assert train._h5 is None


def test_normals_are_unit_vectors(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)
    train, _, _ = load_nasa_crm_dataset(str(tmp_path), strict_split=False)
    x, _ = train[0]
    norms = torch.linalg.vector_norm(x[:, 3:6], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_globals_use_per_channel_unit_gaussian(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)
    train, _, _ = load_nasa_crm_dataset(str(tmp_path), strict_split=False)

    raw = np.stack(
        [
            np.asarray([10 * sample_index + attr_index for attr_index in range(6)], dtype=np.float64)
            for sample_index in range(2)
        ],
        axis=0,
    )
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    expected = (raw[0] - mean) / np.maximum(std, 1e-8)

    x, _ = train[0]
    assert torch.allclose(x[0, 6:], torch.as_tensor(expected, dtype=torch.float32), atol=1e-5)


def test_targets_use_training_statistics(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)
    train, test, meta = load_nasa_crm_dataset(str(tmp_path), strict_split=False)

    _, train_y = train[0]
    _, test_y = test[0]

    assert train_y.dtype == torch.float32
    assert test_y.dtype == torch.float32
    assert meta["y_normalizer"].mean.shape[-1] == 4
    assert meta["y_normalizer"].std.shape[-1] == 4
    assert torch.isfinite(train_y).all()
    assert torch.isfinite(test_y).all()


def test_strict_split_is_enabled_by_default(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)

    with pytest.raises(AssertionError, match="expected 105"):
        load_nasa_crm_dataset(str(tmp_path))


def test_missing_data_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nasa_crm_dataset(str(tmp_path / "missing"), strict_split=False)


def test_missing_array_fails_schema_validation(tmp_path: Path) -> None:
    _write_tiny_nasa_crm(tmp_path)
    missing_key = ARRAY_KEYS[-1]
    with h5py.File(tmp_path / "trainingData_NASA-CRM.h5", "a") as handle:
        del handle["Sample000"][missing_key]

    with pytest.raises(KeyError, match=missing_key):
        load_nasa_crm_dataset(str(tmp_path), strict_split=False)


def test_rel_l2_by_slices_separates_cp_and_cf_errors() -> None:
    y = torch.ones(1, 3, 4)
    yh = y.clone()
    yh[..., :1] = 0

    errors = _rel_l2_by_slices(yh, y, {"cp": slice(0, 1), "cf": slice(1, None)})

    assert errors["cp"].item() == pytest.approx(1.0)
    assert errors["cf"].item() == pytest.approx(0.0)


def test_rel_l2_vector_norm_cf_matches_magnitude_fields() -> None:
    # Transolver-3 Cf: Rel-L2 on |cf| scalars, not vector-difference Rel-L2.
    y = torch.tensor([[[1.0, 3.0, 4.0, 0.0], [2.0, 0.0, 0.0, 5.0]]])  # [1, 2, 4]
    yh = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 3.0, 4.0]]])

    metrics = {
        "cp": {"kind": "slice", "slice": slice(0, 1)},
        "cf": {"kind": "vector_norm", "slice": slice(1, None)},
    }
    errors = _rel_l2_by_field_metrics(yh, y, metrics)

    lf = RelL2Loss()
    y_cf_mag = torch.linalg.vector_norm(y[..., 1:], ord=2, dim=-1, keepdim=True)
    yh_cf_mag = torch.linalg.vector_norm(yh[..., 1:], ord=2, dim=-1, keepdim=True)
    expect_cf = lf(yh_cf_mag, y_cf_mag)
    expect_cp = lf(yh[..., :1], y[..., :1])

    assert errors["cp"].item() == pytest.approx(expect_cp.item())
    assert errors["cf"].item() == pytest.approx(expect_cf.item())
    # Magnitude Rel-L2 differs from joint 3-channel Rel-L2 on this example.
    assert errors["cf"].item() != pytest.approx(lf(yh[..., 1:], y[..., 1:]).item())


@pytest.mark.skipif(
    not Path("data/NASA_CRM/trainingData_NASA-CRM.h5").is_file(),
    reason="NASA-CRM data not downloaded",
)
def test_real_nasa_crm_shapes() -> None:
    train, test, meta = load_nasa_crm_dataset("data/NASA_CRM")
    x, y = train[0]

    assert len(train) == 105
    assert len(test) == 44
    assert x.shape[-1] == 12
    assert x.shape[0] > 400_000
    assert y.shape[-1] == meta["c_out"]
    assert meta["y_field_metrics"]["cf"]["kind"] == "vector_norm"
