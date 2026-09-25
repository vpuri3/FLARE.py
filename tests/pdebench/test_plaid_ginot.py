"""PLAID parse helpers (shared by mesh-static PLAID; GINOT adapter removed)."""

from __future__ import annotations

import pickle

import numpy as np

from pdebench.dataset.plaid_core import parse_plaid_sample_bytes


def _make_cgns_tree(
    *,
    pos: np.ndarray,
    cells: np.ndarray,
    targets: dict[str, np.ndarray],
    scalars: dict[str, float] | None = None,
):
    x, y = pos[:, 0], pos[:, 1]
    connectivity = cells.reshape(-1)
    if connectivity.min() == 0:
        connectivity = connectivity + 1

    point_data_children = []
    for name, values in targets.items():
        point_data_children.append(
            [name, np.asarray(values), [], "DataArray_t"],
        )

    zone_children = [
        ["CoordinateX", x, [], "DataArray_t"],
        ["CoordinateY", y, [], "DataArray_t"],
        ["ElementConnectivity", connectivity, [], "DataArray_t"],
        ["PointData", None, point_data_children, "Zone_t"],
        [
            "Inflow",
            None,
            [["PointList", np.array([1, 2], dtype=np.int64), [], "IndexArray_t"]],
            "ZoneBC_t",
        ],
    ]
    mesh = ["Zone", None, zone_children, "Zone_t"]
    return {"meshes": {0.0: mesh}, "scalars": scalars or {}}


def _make_dynamic_cgns_tree(
    *,
    pos: np.ndarray,
    cells: np.ndarray,
    initial_targets: dict[str, np.ndarray],
    final_targets: dict[str, np.ndarray],
):
    x, y = pos[:, 0], pos[:, 1]
    connectivity = cells.reshape(-1)
    if connectivity.min() == 0:
        connectivity = connectivity + 1

    def _point_children(targets: dict[str, np.ndarray]):
        return [[name, np.asarray(values), [], "DataArray_t"] for name, values in targets.items()]

    geom_zone_children = [
        ["CoordinateX", x, [], "DataArray_t"],
        ["CoordinateY", y, [], "DataArray_t"],
        ["ElementConnectivity", connectivity, [], "DataArray_t"],
        ["PointData", None, _point_children(initial_targets), "Zone_t"],
    ]
    geom_mesh = ["Zone", None, geom_zone_children, "Zone_t"]

    field_zone_children = [
        ["VertexFields", None, _point_children(final_targets), "Zone_t"],
    ]
    field_mesh = ["Zone", None, field_zone_children, "Zone_t"]
    return {"meshes": {0.0: geom_mesh, 1.0: field_mesh}, "scalars": {}}


def test_parse_plaid_sample_bytes_uses_final_timestep_fields_for_dynamic_samples() -> None:
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    initial = {
        "U_x": np.zeros(3, dtype=np.float32),
        "U_y": np.zeros(3, dtype=np.float32),
    }
    final = {
        "U_x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "U_y": np.array([4.0, 5.0, 6.0], dtype=np.float32),
    }
    sample_bytes = pickle.dumps(
        _make_dynamic_cgns_tree(
            pos=pos,
            cells=cells,
            initial_targets=initial,
            final_targets=final,
        )
    )
    parsed = parse_plaid_sample_bytes(
        sample_bytes,
        target_fields=("U_x", "U_y"),
        scalar_names=(),
        dataset_name="plaid_el_pl_dynamics",
        sample_idx=0,
    )
    assert parsed is not None
    assert parsed.pos.shape == (3, 2)
    assert parsed.cells.shape == (1, 3)
    np.testing.assert_allclose(parsed.targets[:, 0], final["U_x"])
    np.testing.assert_allclose(parsed.targets[:, 1], final["U_y"])


def test_parse_plaid_sample_bytes_builds_fields() -> None:
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.2, 0.3]], dtype=np.float32)
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    targets = {
        "Mach": np.linspace(0.1, 0.4, 4, dtype=np.float32),
        "Pressure": np.linspace(1.0, 4.0, 4, dtype=np.float32),
        "Velocity-x": np.zeros(4, dtype=np.float32),
        "Velocity-y": np.ones(4, dtype=np.float32),
    }
    scalars = {
        "max_von_mises": 1.0,
        "max_U2_top": 2.0,
        "max_sig22_top": 3.0,
    }
    sample_bytes = pickle.dumps(_make_cgns_tree(pos=pos, cells=cells, targets=targets, scalars=scalars))
    parsed = parse_plaid_sample_bytes(
        sample_bytes,
        target_fields=("Mach", "Pressure", "Velocity-x", "Velocity-y"),
        scalar_names=(),
        dataset_name="plaid_tensile2d",
        sample_idx=0,
    )
    assert parsed is not None
    assert parsed.pos.shape == (4, 2)
    assert parsed.cells.shape == (2, 3)
    assert parsed.targets is not None and parsed.targets.shape == (4, 4)
    assert parsed.boundary_pos.shape[0] >= 2
    assert parsed.space_dim == 2
    assert parsed.target_scalars is not None
    np.testing.assert_allclose(parsed.target_scalars, [1.0, 2.0, 3.0])
