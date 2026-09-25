from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from pdebench.dataset.adapters import get_adapter
from pdebench.dataset.ginot.graph_cache import graph_cache_num_shards
from pdebench.dataset.ginot.io import load_poisson_structured, load_raw_dataset
from pdebench.dataset.ginot.types import GINOT_DATASETS
from pdebench.dataset.laplacian.spec import DATASET_LAPLACIAN_SPECS, default_laplacian_specs_for_dataset


def _write_struc_pickle(path: Path, *, n: int = 3, mutate_cells_at: int | None = None, n_nodes: int = 4) -> None:
    cells0 = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
    cells = [cells0.copy() for _ in range(n)]
    if mutate_cells_at is not None:
        cells[mutate_cells_at] = cells0.copy()
        cells[mutate_cells_at][0] = 1  # break uniqueness
    nodes = [np.random.default_rng(i).random((n_nodes, 2), dtype=np.float32) for i in range(n)]
    payload = {
        "cells": cells,
        "nodes": nodes,
        "point_clouds": [np.random.default_rng(i).random((5, 2), dtype=np.float32) for i in range(n)],
        "solutions": [np.zeros((n_nodes,), dtype=np.float32) for _ in range(n)],
        "radius": np.ones(n, dtype=np.float32),
        "celltypes": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_poisson_structured_registered_surfaces() -> None:
    assert "poisson_structured" in GINOT_DATASETS
    assert default_laplacian_specs_for_dataset("poisson_structured") == "graph:64"
    assert DATASET_LAPLACIAN_SPECS["poisson_structured"] == "graph:64"
    assert graph_cache_num_shards("poisson_structured", 6001) == 2
    assert get_adapter("poisson_structured") is not None


def test_load_poisson_structured_ok(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_struc_pickle(root / "poisson" / "poisson_geo_struc_msh.pkl", n=3)
    raw = load_poisson_structured(str(root))
    assert len(raw.query_points) == 3
    assert raw.space_dim == 2
    assert raw.target_fields == ("u",)
    assert load_raw_dataset("poisson_structured", str(root)) is not None


def test_load_poisson_structured_rejects_nonunique_cells(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_struc_pickle(root / "poisson" / "poisson_geo_struc_msh.pkl", n=3, mutate_cells_at=1)
    with pytest.raises(ValueError, match="single shared mesh"):
        load_poisson_structured(str(root))


def test_load_poisson_structured_rejects_varying_node_counts(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = root / "poisson" / "poisson_geo_struc_msh.pkl"
    cells0 = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32)
    nodes = [
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((5, 2), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
    ]
    payload = {
        "cells": [cells0.copy() for _ in range(3)],
        "nodes": nodes,
        "point_clouds": [np.zeros((5, 2), dtype=np.float32) for _ in range(3)],
        "solutions": [np.zeros((n.shape[0],), dtype=np.float32) for n in nodes],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    with pytest.raises(ValueError, match="node count"):
        load_poisson_structured(str(root))
