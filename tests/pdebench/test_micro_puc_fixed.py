from __future__ import annotations

import numpy as np
import pytest
import torch

from pdebench.dataset.ginot.micro_puc_fixed import (
    MicroPucFixedConfig,
    build_micro_puc_fixed_dataset,
    build_periodic_node_map,
    fix_one_sample,
)
from pdebench.dataset.ginot.sample import build_periodic_x_edge_attr


def _grid_3x3_pos() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_periodic_node_map_collapses_xy_boundaries() -> None:
    pos = _grid_3x3_pos()
    remap, keep, stats = build_periodic_node_map(pos, MicroPucFixedConfig())

    corner = 0
    assert remap[corner] == remap[2]
    assert remap[corner] == remap[6]
    assert remap[corner] == remap[8]
    assert remap[3] == remap[5]
    assert remap[1] == remap[7]
    assert remap[0] != remap[1]
    assert int(keep.sum()) == 4
    assert stats["removed_nodes"] == 5
    assert stats["x1_matched_nodes"] == 3
    assert stats["y1_matched_nodes"] == 3


def test_periodic_node_map_matches_near_x0_boundary_fallback() -> None:
    pos = np.asarray(
        [
            [0.0, 0.69467145],
            [0.00043920, 0.73283756],
            [0.00087840, 0.77100360],
            [0.00131760, 0.80916965],
            [0.00351360, 1.0],
            [0.00351360, 0.0],
            [1.0, 0.69467145],
            [1.0, 0.73243684],
            [1.0, 0.77020216],
            [1.0, 0.80796754],
            [1.0, 0.99679428],
        ],
        dtype=np.float32,
    )

    remap, keep, stats = build_periodic_node_map(pos, MicroPucFixedConfig())

    assert stats["x1_unmatched_nodes"] == 0
    assert remap[6] == remap[0]
    assert remap[7] == remap[1]
    assert remap[8] == remap[2]
    assert remap[9] == remap[3]
    assert remap[10] == remap[5]
    assert int(keep.sum()) == 5


def test_fix_one_sample_remaps_fields_and_preserves_directed_edge_rows_before_dedupe() -> None:
    pos = _grid_3x3_pos()
    field = np.arange(27, dtype=np.float32).reshape(9, 3)
    cells = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 5],
            [3, 4, 7],
            [4, 5, 8],
        ],
        dtype=np.int64,
    )

    fixed = fix_one_sample(0, pos, field, cells, MicroPucFixedConfig())

    assert fixed["pos"].shape == (4, 2)
    assert fixed["y"].shape == (4, 3)
    assert fixed["edges"].ndim == 2
    assert fixed["edges"].shape[1] == 2
    assert fixed["edges"].min() >= 0
    assert fixed["edges"].max() < fixed["pos"].shape[0]
    assert not np.any(fixed["edges"][:, 0] == fixed["edges"][:, 1])
    assert fixed["stats"].edge_source == "cells"
    assert fixed["stats"].old_directed_edges > 0
    assert fixed["stats"].remapped_directed_edges == fixed["stats"].old_directed_edges


def test_fix_one_sample_reuses_canonical_map_for_shifted_duplicate_nodes() -> None:
    canonical_pos = np.asarray(
        [
            [0.0, 0.25],
            [0.5, 0.25],
            [1.0, 0.25],
            [0.0, 0.75],
            [0.5, 0.75],
            [1.0, 0.75],
        ],
        dtype=np.float32,
    )
    shifted_pos = np.asarray(
        [
            [0.9, 0.25],
            [0.4, 0.25],
            [0.9, 0.25],
            [0.9, 0.75],
            [0.4, 0.75],
            [0.9, 0.75],
        ],
        dtype=np.float32,
    )
    field = np.arange(12, dtype=np.float32).reshape(6, 2)
    cells = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    config = MicroPucFixedConfig()
    remap, keep, map_stats = build_periodic_node_map(canonical_pos, config)
    fixed_canonical = fix_one_sample(1169, canonical_pos, field, cells, config)
    canonical_map = {
        "keep": keep,
        "edges": fixed_canonical["edges"],
        "stats": map_stats,
        "old_directed_edges": fixed_canonical["stats"].old_directed_edges,
        "remapped_directed_edges": fixed_canonical["stats"].remapped_directed_edges,
        "self_loops_after_remap": fixed_canonical["stats"].self_loops_after_remap,
        "source_geometry_id": 1169,
        "canonical_row": 925,
    }

    fixed_shifted = fix_one_sample(12972, shifted_pos, field, None, config, canonical_map=canonical_map)

    assert fixed_shifted["pos"].shape == (4, 2)
    assert np.unique(fixed_shifted["pos"], axis=0).shape[0] == 4
    assert fixed_shifted["stats"].sample_id == 12972
    assert fixed_shifted["stats"].source_geometry_id == 1169
    assert fixed_shifted["stats"].canonical_row == 925
    assert remap.tolist() == [0, 1, 0, 2, 3, 2]


def test_fix_one_sample_accepts_one_based_cells() -> None:
    pos = _grid_3x3_pos()
    field = np.arange(27, dtype=np.float32).reshape(9, 3)
    cells = np.asarray([[1, 2, 5], [2, 3, 6], [4, 5, 8], [5, 6, 9]], dtype=np.int64)

    fixed = fix_one_sample(0, pos, field, cells, MicroPucFixedConfig())

    assert fixed["edges"].shape[1] == 2
    assert fixed["edges"].min() >= 0
    assert fixed["edges"].max() < fixed["pos"].shape[0]


def test_fix_one_sample_requires_mesh_cells() -> None:
    pos = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    field = np.asarray([[0.0], [1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="requires mesh cell connectivity"):
        fix_one_sample(0, pos, field, None, MicroPucFixedConfig())


def test_build_micro_puc_fixed_dataset_fails_if_requested_mesh_cells_are_missing(tmp_path) -> None:
    import pickle

    src = tmp_path / "PeriodUnitCell"
    src.mkdir()
    coords = [np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32) for _ in range(2)]
    fields = [np.asarray([[0.0], [1.0]], dtype=np.float32) for _ in range(2)]
    cells = [np.asarray([[0, 1]], dtype=np.int64)]
    with (src / "mesh_coords.pkl").open("wb") as f:
        pickle.dump(coords, f)
    with (src / "mises_disp_laststep.pkl").open("wb") as f:
        pickle.dump({"mises_disp": fields}, f)
    with (src / "mesh_cells10K.pkl").open("wb") as f:
        pickle.dump(cells, f)
    np.save(src / "sample_ids.npy", np.asarray([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="without canonical mesh"):
        build_micro_puc_fixed_dataset(
            data_root=tmp_path,
            config=MicroPucFixedConfig(max_samples=2, num_workers=1),
            overwrite=True,
        )


def test_build_micro_puc_fixed_dataset_applies_canonical_maps_to_all_rows(tmp_path) -> None:
    import json
    import pickle

    src = tmp_path / "PeriodUnitCell"
    src.mkdir()
    coords = [
        np.asarray([[0.0, 0.25], [0.5, 0.25], [1.0, 0.25]], dtype=np.float32),
        np.asarray([[0.0, 0.75], [0.5, 0.75], [1.0, 0.75]], dtype=np.float32),
        np.asarray([[0.9, 0.25], [0.4, 0.25], [0.9, 0.25]], dtype=np.float32),
    ]
    fields = [np.arange(6, dtype=np.float32).reshape(3, 2) for _ in range(3)]
    cells = [np.asarray([[0, 1, 2]], dtype=np.int64), np.asarray([[0, 1, 2]], dtype=np.int64)]
    with (src / "mesh_coords.pkl").open("wb") as f:
        pickle.dump(coords, f)
    with (src / "mises_disp_laststep.pkl").open("wb") as f:
        pickle.dump({"mises_disp": fields}, f)
    with (src / "mesh_cells10K.pkl").open("wb") as f:
        pickle.dump(cells, f)
    np.save(src / "sample_ids.npy", np.asarray([10, 11, 10], dtype=np.int64))

    out = build_micro_puc_fixed_dataset(
        data_root=tmp_path,
        config=MicroPucFixedConfig(max_samples=3, num_workers=1, shard_size=2),
        overwrite=True,
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["num_samples"] == 3
    assert manifest["canonical_mesh_rows"] == 2
    assert manifest["source_geometry_ids"] == [10, 11]
    assert manifest["quotient_axes"] == ["x", "y"]


def test_periodic_edge_attr_uses_minimum_image_displacement_on_x_and_y() -> None:
    pos = np.asarray(
        [[0.95, 0.5], [0.0, 0.5], [0.25, 0.5], [0.5, 0.95], [0.5, 0.0]],
        dtype=np.float32,
    )
    edge_index = np.asarray([[0, 1, 2, 3], [1, 0, 0, 4]], dtype=np.int64)
    edge_attr = build_periodic_x_edge_attr(
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index).long(),
    )

    assert edge_attr.shape == (4, 3)
    assert edge_attr[0, 0].item() < 0.0
    assert edge_attr[1, 0].item() > 0.0
    assert abs(edge_attr[0, 2].item() - edge_attr[1, 2].item()) < 1e-5
    dy = float(pos[3, 1] - pos[4, 1])
    wrapped_dy = (dy + 0.5) % 1.0 - 0.5
    assert wrapped_dy < 0.0
    assert abs(wrapped_dy) < 0.1
    assert edge_attr[3, 1].item() < 0.0
