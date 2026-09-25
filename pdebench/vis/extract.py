"""Per-dataset geometry and field extraction for visualization bundles.

Every function here is pure numpy: it takes the model-space arrays that the
dataloader and the model produced, and returns the *physical* geometry and
fields a renderer needs. Nothing here loads data or touches torch, so it is
testable off-cluster.

The per-dataset knowledge that lives in this file, and nowhere else:

  - where the point coordinates hide inside ``x``, and how to undo whatever
    normalization the loader applied to them;
  - whether the points form a structured grid, and of what shape;
  - what the output channels are called;
  - which extra input channel is worth carrying for the benchmark gallery
    (e.g. the Darcy permeability field).

Getting the coordinates wrong is silent and ruinous: ``figs/pdebench_vis.ipynb``
drew Airfoil and Pipe with ``imshow`` on the index grid, i.e. in computational
space, which is not the shape the solver saw.
"""

from __future__ import annotations

import numpy as np

# Datasets this module knows how to unpack.
DATASETS: tuple[str, ...] = (
    "elasticity",
    "darcy",
    "airfoil_steady",
    "pipe",
    "drivaerml_40k",
    "drivaerml_surface",
    "nasa_crm",
    "ahmedml_surface",
)

# Output channel names, in channel order, per dataset.
#
# airfoil_steady takes channel 4 of Geo-FNO's ``NACA_Cylinder_Q``, which is the
# Mach number; pipe takes channel 0 of ``Pipe_Q``, the streamwise velocity.
_FIELD_NAMES: dict[str, tuple[str, ...]] = {
    "elasticity": ("sigma",),
    "darcy": ("u",),
    "airfoil_steady": ("mach",),
    "pipe": ("u_x",),
    "drivaerml_40k": ("p",),
    "drivaerml_surface": ("p", "tau_x", "tau_y", "tau_z"),
    "nasa_crm": ("cp", "cf_x", "cf_y", "cf_z"),
    "ahmedml_surface": ("p", "tau_x", "tau_y", "tau_z"),
}

# Spatial dimension of the point set.
_NDIM: dict[str, int] = {
    "elasticity": 2,
    "darcy": 2,
    "airfoil_steady": 2,
    "pipe": 2,
    "drivaerml_40k": 3,
    "drivaerml_surface": 3,
    "nasa_crm": 3,
    "ahmedml_surface": 3,
}


def field_names(dataset: str) -> tuple[str, ...]:
    _check(dataset)
    return _FIELD_NAMES[dataset]


def ndim(dataset: str) -> int:
    _check(dataset)
    return _NDIM[dataset]


def is_surface(dataset: str) -> bool:
    """True for the 3D surface benchmarks, which need normals to render."""
    return ndim(dataset) == 3


def _check(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported visualization dataset {dataset!r}; expected one of {DATASETS}")


def _as_np(value) -> np.ndarray:
    """Accept a torch tensor or an array without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _minmax_denorm(xyz_unit: np.ndarray, metadata: dict, dataset: str) -> np.ndarray:
    """Undo the per-axis min-max map the 3D loaders apply to coordinates.

    Per-axis min-max does not preserve aspect ratio, so rendering the encoded
    coordinates directly would show a stretched car or aircraft.
    """
    lo = metadata.get("vis_xyz_min")
    hi = metadata.get("vis_xyz_max")
    if lo is None or hi is None:
        raise KeyError(
            f"{dataset}: metadata is missing vis_xyz_min / vis_xyz_max; "
            "pdebench.vis.runner must attach them from the dataset object before extraction"
        )
    lo = _as_np(lo).reshape(1, 3).astype(np.float64)
    hi = _as_np(hi).reshape(1, 3).astype(np.float64)
    return (xyz_unit.astype(np.float64) * (hi - lo) + lo).astype(np.float32)


def extract_geometry(dataset: str, x: np.ndarray, metadata: dict) -> dict:
    """Physical coordinates (+ normals, grid shape, gallery inputs) from ``x``.

    ``x`` is one sample, shape ``[N, C]``, exactly as handed to the model.
    """
    _check(dataset)
    x = _as_np(x)
    if x.ndim != 2:
        raise ValueError(f"{dataset}: expected x of shape [N, C], got {x.shape}")

    out: dict = {"normals": None, "grid_shape": None, "inputs": {}}

    if dataset == "elasticity":
        # Raw unit-cell coordinates; IdentityNormalizer, nothing to undo.
        out["coords"] = x[:, :2].astype(np.float32)

    elif dataset == "darcy":
        # x = [pos_x, pos_y, coeff]; pos is the raw unit square, only the
        # coefficient channel went through x_normalizer.
        out["coords"] = x[:, :2].astype(np.float32)
        out["grid_shape"] = _grid_shape(metadata, dataset)
        out["inputs"]["a"] = _decode_darcy_coeff(x[:, 2:3], metadata).reshape(-1).astype(np.float32)

    elif dataset == "airfoil_steady":
        # IdentityNormalizer on x: already the body-fitted physical mesh.
        out["coords"] = x[:, :2].astype(np.float32)
        out["grid_shape"] = _grid_shape(metadata, dataset)

    elif dataset == "pipe":
        # x went through a UnitGaussianNormalizer; decode to get the real mesh.
        out["coords"] = _decode_x(x[:, :2], metadata).astype(np.float32)
        out["grid_shape"] = _grid_shape(metadata, dataset)

    elif dataset == "drivaerml_40k":
        # x = [xyz] only. No normals in this pipeline; the renderer estimates
        # them from local neighbourhoods (see figs/vis/render3d.py).
        out["coords"] = _minmax_denorm(x[:, :3], metadata, dataset)

    elif dataset in ("ahmedml_surface", "drivaerml_surface"):
        # x = [xyz_unit(3), unit_normals(3)]
        out["coords"] = _minmax_denorm(x[:, :3], metadata, dataset)
        out["normals"] = x[:, 3:6].astype(np.float32)

    elif dataset == "nasa_crm":
        # x = [xyz_unit(3), unit_normals(3), broadcast operating conditions(6)]
        out["coords"] = _minmax_denorm(x[:, :3], metadata, dataset)
        out["normals"] = x[:, 3:6].astype(np.float32)
        # The six operating conditions are constant over the mesh; keep one row
        # so a caption can state the flight condition of the shown case.
        out["inputs"]["globals"] = x[0, 6:12].astype(np.float32)

    expected = _NDIM[dataset]
    if out["coords"].shape[1] != expected:
        raise ValueError(f"{dataset}: expected {expected}D coordinates, got shape {out['coords'].shape}")

    grid = out["grid_shape"]
    if grid is not None and grid[0] * grid[1] != out["coords"].shape[0]:
        raise ValueError(
            f"{dataset}: grid shape {grid} does not match {out['coords'].shape[0]} points; "
            "the loader's H/W metadata and the sample disagree"
        )

    return out


def _grid_shape(metadata: dict, dataset: str) -> tuple[int, int]:
    h, w = metadata.get("H"), metadata.get("W")
    if h is None or w is None:
        raise KeyError(f"{dataset}: structured dataset but metadata has no H/W")
    return (int(h), int(w))


def _decode_x(x: np.ndarray, metadata: dict) -> np.ndarray:
    """Undo metadata['x_normalizer'] on the coordinate channels."""
    normalizer = metadata["x_normalizer"]
    mean = getattr(normalizer, "mean", None)
    std = getattr(normalizer, "std", None)
    if mean is None or std is None:  # IdentityNormalizer
        return x
    mean = _as_np(mean).reshape(-1)[: x.shape[1]]
    std = _as_np(std).reshape(-1)[: x.shape[1]]
    return x * std + mean


def _decode_darcy_coeff(coeff: np.ndarray, metadata: dict) -> np.ndarray:
    """Darcy's x_normalizer was fit on the 1-channel coefficient alone."""
    normalizer = metadata["x_normalizer"]
    mean = getattr(normalizer, "mean", None)
    std = getattr(normalizer, "std", None)
    if mean is None or std is None:
        return coeff
    return coeff * _as_np(std).reshape(-1)[0] + _as_np(mean).reshape(-1)[0]


def extract_bundle(
    dataset: str,
    *,
    x: np.ndarray,
    y_ref: np.ndarray,
    predictions: dict[str, np.ndarray],
    metadata: dict,
    tri: np.ndarray | None = None,
) -> dict:
    """Assemble the ``.npz`` payload for one case.

    ``y_ref`` and every entry of ``predictions`` must already be decoded to
    physical units. Keys of ``predictions`` become ``y_<model>`` arrays.
    """
    _check(dataset)
    y_ref = _as_np(y_ref).astype(np.float32)
    if y_ref.ndim != 2:
        raise ValueError(f"{dataset}: expected y of shape [N, F], got {y_ref.shape}")

    names = _FIELD_NAMES[dataset]
    if y_ref.shape[1] != len(names):
        raise ValueError(
            f"{dataset}: expected {len(names)} output channels {names}, got {y_ref.shape[1]}"
        )

    geom = extract_geometry(dataset, x, metadata)
    n_points = geom["coords"].shape[0]
    if y_ref.shape[0] != n_points:
        raise ValueError(f"{dataset}: {y_ref.shape[0]} field values against {n_points} points")

    bundle: dict = {
        "coords": geom["coords"],
        "y_ref": y_ref,
        "field_names": np.array(names, dtype=object),
        "dataset": np.array(dataset),
    }

    for model, pred in predictions.items():
        pred = _as_np(pred).astype(np.float32)
        if pred.shape != y_ref.shape:
            raise ValueError(f"{dataset}/{model}: prediction shape {pred.shape} != reference {y_ref.shape}")
        bundle[f"y_{model}"] = pred

    if geom["normals"] is not None:
        bundle["normals"] = geom["normals"]
    if geom["grid_shape"] is not None:
        bundle["grid_shape"] = np.asarray(geom["grid_shape"], dtype=np.int64)
    if tri is not None:
        bundle["tri"] = _as_np(tri).astype(np.int32)
    for key, value in geom["inputs"].items():
        bundle[f"input_{key}"] = np.asarray(value, dtype=np.float32)

    return bundle
