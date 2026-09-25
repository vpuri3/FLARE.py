from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from pdebench.dataset.nasa_crm import assert_official_split


def _write_sample_groups(path: Path, count: int) -> None:
    with h5py.File(path, "w") as handle:
        for index in range(1, count + 1):
            handle.create_group(f"Sample{index:03d}")


def test_assert_official_split_passes_with_matching_counts(tmp_path: Path) -> None:
    _write_sample_groups(tmp_path / "trainingData_NASA-CRM.h5", 2)
    _write_sample_groups(tmp_path / "testData_NASA-CRM.h5", 1)

    assert_official_split(tmp_path, train_samples=2, test_samples=1)


def test_assert_official_split_raises_on_wrong_train_count(tmp_path: Path) -> None:
    _write_sample_groups(tmp_path / "trainingData_NASA-CRM.h5", 1)
    _write_sample_groups(tmp_path / "testData_NASA-CRM.h5", 1)

    with pytest.raises(AssertionError, match=r"trainingData_NASA-CRM\.h5 has 1 sample keys, expected 2"):
        assert_official_split(tmp_path, train_samples=2, test_samples=1)


def test_assert_official_split_raises_on_wrong_test_count(tmp_path: Path) -> None:
    _write_sample_groups(tmp_path / "trainingData_NASA-CRM.h5", 2)
    _write_sample_groups(tmp_path / "testData_NASA-CRM.h5", 2)

    with pytest.raises(AssertionError, match=r"testData_NASA-CRM\.h5 has 2 sample keys, expected 1"):
        assert_official_split(tmp_path, train_samples=2, test_samples=1)
