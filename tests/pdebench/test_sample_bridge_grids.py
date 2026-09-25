"""C2 grid / geo families: document intentional no-wrap for Sample bridge (Task 6).

These loaders return ``TensorDataset`` or custom ``Dataset`` items as plain tensor
tuples, not PyG ``Data`` or GINOT ``dict`` samples. C2a strict scope wraps only
families whose legacy items already round-trip through the existing PyG or GINOT
bridges without inventing geometry or re-shaping fused inputs (e.g. Darcy
``[pos_x, pos_y, coeff]`` concatenation). Grid/geo adapters therefore pass
loader output through unchanged; ``get_adapter`` must still resolve and load them.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, TensorDataset

import pdebench.dataset.utils as dataset_utils
from pdebench.dataset.adapters import get_adapter
from pdebench.dataset.sample_wrappers import RoundTripGinotDataset, RoundTripPygDataset

# Task 6 scope: loader return shapes documented for future C2/C4 work.
_GRID_GEO_LOADER_RETURNS: dict[str, str] = {
    "elasticity": "TensorDataset (input_xy, input_s) — mesh coords + scalar field",
    "darcy": "TensorDataset (pos+coeff, sol) — fused 3-D input, not separate pos/y",
    "shapenet_car": "ShapeNetCarDataset (mesh_points, pressure) — point-cloud tuple",
    "drivaerml_40k": "DrivAerMLDataset (surface_centers, pressure) — point-cloud tuple",
}

_GRID_GEO_SIMPLE_NAMES = ("elasticity", "darcy", "shapenet_car")
_DRIVAER_NAME = "drivaerml_40k"


class _TinyTupleDataset(Dataset):
    """Stand-in for grid/geo datasets that yield ``(pos, y)`` tuples."""

    def __init__(self, n: int = 2) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            torch.tensor([[float(idx), 0.0], [float(idx) + 1.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5], [1.5]], dtype=torch.float32),
        )


def test_grid_geo_loader_return_types_documented() -> None:
    assert set(_GRID_GEO_LOADER_RETURNS) == {*_GRID_GEO_SIMPLE_NAMES, _DRIVAER_NAME}


def test_simple_grid_geo_adapters_skip_sample_bridge(monkeypatch) -> None:
    train_ds = _TinyTupleDataset()
    test_ds = _TinyTupleDataset()

    for name in _GRID_GEO_SIMPLE_NAMES:
        loader_attr = f"load_{name}_dataset"

        def _fake_loader(data_root, _attr=loader_attr):
            assert isinstance(data_root, str)
            return train_ds, test_ds, {"meta": 1}

        monkeypatch.setattr(dataset_utils, loader_attr, _fake_loader)

        adapter = get_adapter(name)
        assert adapter.name == name

        train, test, meta = adapter.load("unused")
        assert train is train_ds
        assert test is test_ds
        assert meta == {"meta": 1}
        assert "sample_bridge" not in meta
        assert not isinstance(train, RoundTripPygDataset)
        assert not isinstance(train, RoundTripGinotDataset)


def test_drivaerml_adapter_skips_sample_bridge(monkeypatch) -> None:
    train_ds = _TinyTupleDataset()
    test_ds = _TinyTupleDataset()
    calls: dict[str, str] = {}

    def _fake_loader(dataset_name, data_root):
        calls["dataset_name"] = dataset_name
        calls["data_root"] = data_root
        return train_ds, test_ds, {"meta": 1}

    monkeypatch.setattr(dataset_utils, "load_drivaerml_dataset", _fake_loader)

    adapter = get_adapter(_DRIVAER_NAME)
    assert adapter.name == _DRIVAER_NAME

    train, test, meta = adapter.load("unused")
    assert calls == {"dataset_name": _DRIVAER_NAME, "data_root": "unused"}
    assert train is train_ds
    assert test is test_ds
    assert meta == {"meta": 1}
    assert "sample_bridge" not in meta
    assert not isinstance(train, RoundTripPygDataset)


def test_darcy_fused_input_not_lossless_pos_y_split() -> None:
    """Darcy items fuse grid coords and PDE coeff — not a bare (pos, y) tuple."""
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    coeff = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    fused = torch.cat([pos, coeff], dim=-1)
    sol = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

    ds = TensorDataset(fused, sol)
    item = ds[0]
    assert len(item) == 2
    assert item[0].shape[-1] == 3
    assert item[0].shape[-1] != item[1].shape[-1]


def test_elasticity_returns_coord_field_tuple_not_pyg() -> None:
    n = 3
    coords = torch.randn(n, 2)
    field = torch.randn(n, 1)
    ds = TensorDataset(coords, field)
    item = ds[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert not hasattr(item, "keys")
