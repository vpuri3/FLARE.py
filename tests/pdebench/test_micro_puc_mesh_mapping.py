from __future__ import annotations

import numpy as np
import pytest

from pdebench.dataset.ginot import (
    GinotRawDataset,
    MICRO_PUC_CANONICAL_MESH_ROWS,
    _build_micro_puc_mesh_idx,
    _select_cells,
)


def test_build_micro_puc_mesh_idx_maps_augmented_rows_to_canonical_meshes() -> None:
    sample_ids = np.concatenate(
        [
            np.array([0, 14, 15, 17], dtype=np.int64),
            np.full(MICRO_PUC_CANONICAL_MESH_ROWS - 4, 99, dtype=np.int64),
            np.array([0, 14, 15], dtype=np.int64),
        ]
    )
    mesh_idx = _build_micro_puc_mesh_idx(sample_ids)
    assert mesh_idx.shape == sample_ids.shape
    assert mesh_idx[0] == 0
    assert mesh_idx[1] == 1
    assert mesh_idx[MICRO_PUC_CANONICAL_MESH_ROWS] == 0
    assert mesh_idx[MICRO_PUC_CANONICAL_MESH_ROWS + 1] == 1


def test_select_cells_uses_mesh_idx_for_micro_puc() -> None:
    cells = [np.array([[i + 1, i + 2]], dtype=np.int64) for i in range(3)]
    raw = GinotRawDataset(
        query_points=[np.zeros((2, 2), dtype=np.float32) for _ in range(5)],
        point_clouds=[np.zeros((1, 2), dtype=np.float32) for _ in range(5)],
        targets=[np.zeros((2, 1), dtype=np.float32) for _ in range(5)],
        cells=cells,
        input_params=None,
        target_fields=("u",),
        space_dim=2,
        micro_puc_mesh_idx=np.array([2, 0, 1, 2, 0], dtype=np.int32),
    )
    assert np.array_equal(_select_cells(raw, 4), cells[0])
    assert np.array_equal(_select_cells(raw, 0), cells[2])


def test_build_micro_puc_mesh_idx_rejects_unknown_base_ids() -> None:
    sample_ids = np.arange(MICRO_PUC_CANONICAL_MESH_ROWS + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="without a canonical mesh"):
        _build_micro_puc_mesh_idx(sample_ids)
