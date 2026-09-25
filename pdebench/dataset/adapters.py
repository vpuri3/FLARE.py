"""Thin ``DatasetAdapter`` registry wrapping today's ``load_dataset`` loaders (C1 shim).

Zero-behavior-change facade: every path in ``utils.load_dataset`` resolves to
``get_adapter(name).load(...)`` instead of calling loader functions directly.

Adapters MUST resolve loaders via ``pdebench.dataset.utils`` at call time
(lazily), because existing tests monkeypatch loader names (e.g.
``load_mesh_static_dataset``, ``load_ginot_dataset``, ``load_lpbf_dataset``,
``load_elasticity_dataset``) as attributes on the ``utils`` module. Importing
those loaders directly at this module's top level would bind a separate
reference that monkeypatches on ``utils`` cannot reach.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Protocol

from torch.utils.data import Dataset

from pdebench.dataset.registry import CANONICAL, RegistryError, resolve_dataset_name
from pdebench.dataset.sample import FeatureRequest, SampleKind
from pdebench.dataset.sample_collate import collate_plaid_static, collate_samples_ginot
from pdebench.dataset.sample_wrappers import RoundTripGinotDataset, RoundTripPygDataset

# C2a strict scope: all canonical PLAID mesh-static families round-trip through Sample.
_SAMPLE_BRIDGE_PLAID_KINDS: dict[str, SampleKind] = {
    "plaid_tensile2d": SampleKind.STATIC,
    "plaid_hyperelasticity": SampleKind.STATIC,
    "plaid_el_pl_dynamics": SampleKind.STATIC,
    "plaid_elpl_terminal": SampleKind.TERMINAL,
}

# C4 scope (Tasks 9 + 11): all canonical PLAID mesh-static families.
_SAMPLE_COLLATE_PLAID_NAMES: frozenset[str] = frozenset(_SAMPLE_BRIDGE_PLAID_KINDS)

# C2a strict scope: all canonical GINOT families round-trip through Sample.
_SAMPLE_BRIDGE_GINOT_KINDS: dict[str, SampleKind] = {
    "poisson_unstructured": SampleKind.STATIC,
    "poisson_structured": SampleKind.STATIC,
    "bracket_lug": SampleKind.STATIC,
    "micro_puc": SampleKind.STATIC,
    "micro_puc_fixed": SampleKind.STATIC,
    "deform_plate": SampleKind.STATIC,
    "bumper_beam": SampleKind.STATIC,
}

# C4 scope (this task): every canonical GINOT family (same names as the C2a bridge).
_SAMPLE_COLLATE_GINOT_NAMES: frozenset[str] = frozenset(_SAMPLE_BRIDGE_GINOT_KINDS)


def _sample_bridge_enabled() -> bool:
    """Return False when ``PDEBENCH_SAMPLE_BRIDGE=0`` (C2a wrappers off; default on)."""
    return os.environ.get("PDEBENCH_SAMPLE_BRIDGE", "1") not in ("0", "false", "False")


def _sample_collate_enabled_for(name: str, *, family_names: frozenset[str], family_alias: str) -> bool:
    """Return True when C4 Sample collate is active for ``name`` in ``family_names``.

    ``PDEBENCH_SAMPLE_COLLATE`` unset or ``"0"``/``"false"`` -> disabled (default).
    ``"1"``/``"true"`` -> enabled for every name in ``family_names``.
    ``family_alias`` (e.g. ``"plaid_static"`` / ``"ginot"``) -> enabled for every
    name in ``family_names`` for that family only. Otherwise treated as a
    comma-separated list of canonical dataset names to enable individually.
    """
    if name not in family_names:
        return False
    raw = os.environ.get("PDEBENCH_SAMPLE_COLLATE", "0").strip()
    if raw in ("0", "false", "False", ""):
        return False
    if raw in ("1", "true", "True", family_alias):
        return True
    requested = {token.strip() for token in raw.split(",") if token.strip()}
    return name in requested


def _sample_collate_enabled(name: str) -> bool:
    """C4 Sample collate flag for canonical PLAID mesh-static families (Tasks 9 + 11)."""
    return _sample_collate_enabled_for(name, family_names=_SAMPLE_COLLATE_PLAID_NAMES, family_alias="plaid_static")


def _sample_collate_enabled_ginot(name: str) -> bool:
    """C4 Sample collate flag for canonical GINOT families (see Task 10)."""
    return _sample_collate_enabled_for(name, family_names=_SAMPLE_COLLATE_GINOT_NAMES, family_alias="ginot")


def _utils():
    import pdebench.dataset.utils as u

    return u


def _wrap_sample_bridge(dataset: Any, *, kind: SampleKind = SampleKind.STATIC, yield_sample: bool = False) -> Any:
    """Wrap ``dataset`` with the C2a round-trip bridge, skipping non-Dataset stand-ins.

    Tests monkeypatch ``load_mesh_static_dataset`` to return plain sentinel values
    (e.g. ``"train"``); only real ``Dataset`` instances go through the bridge.
    ``yield_sample=True`` is the C4 mode: ``__getitem__`` returns ``Sample``
    instead of round-tripping back to ``Data`` (pairs with ``collate_plaid_static``).
    """
    if isinstance(dataset, Dataset):
        return RoundTripPygDataset(dataset, kind=kind, yield_sample=yield_sample)
    return dataset


def _wrap_ginot_sample_bridge(dataset: Any, *, kind: SampleKind = SampleKind.STATIC, yield_sample: bool = False) -> Any:
    """Wrap ``dataset`` with the C2a GINOT round-trip bridge, skipping non-Dataset stand-ins.

    Tests monkeypatch ``load_ginot_dataset`` to return plain sentinel values
    (e.g. ``"train"``); only real ``Dataset`` instances go through the bridge.
    ``yield_sample=True`` is the C4 mode: ``__getitem__`` returns ``Sample``
    instead of round-tripping back to ``dict`` (pairs with ``collate_ginot``).
    """
    if isinstance(dataset, Dataset):
        return RoundTripGinotDataset(dataset, kind=kind, yield_sample=yield_sample)
    return dataset


class DatasetAdapter(Protocol):
    name: str

    def load(self, data_root: str, **kwargs: Any) -> tuple[Any, Any, Any]:
        ...


class _PlaidCanonicalAdapter:
    """``plaid_tensile2d`` / ``plaid_hyperelasticity`` / ``plaid_el_pl_dynamics`` / ``plaid_elpl_terminal``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def load(
        self,
        data_root: str,
        *,
        mesh_split_seed: int,
        mesh_graph_backend: str,
        plaid_use_sdf_features: bool,
        plaid_max_samples: int,
        plaid_load_public_test: bool,
        plaid_terminal_y_norm: str,
        plaid_terminal_target_fields: list[str] | None,
        feature_request: FeatureRequest,
        **_unused: Any,
    ):
        mesh_kwargs = dict(
            dataset_name=self.name,
            data_root=data_root,
            split_seed=mesh_split_seed,
            graph_backend=mesh_graph_backend,
            use_sdf_features=plaid_use_sdf_features,
            max_samples=plaid_max_samples,
            load_public_test=plaid_load_public_test,
            laplacian_eig_dim=feature_request.laplacian_k,
            laplacian_spec=feature_request.laplacian_spec,
        )
        if self.name == "plaid_elpl_terminal":
            mesh_kwargs["y_norm_mode"] = plaid_terminal_y_norm
            mesh_kwargs["terminal_target_fields"] = plaid_terminal_target_fields
        train, test, meta = _utils().load_mesh_static_dataset(**mesh_kwargs)
        if _sample_bridge_enabled() and self.name in _SAMPLE_BRIDGE_PLAID_KINDS:
            kind = _SAMPLE_BRIDGE_PLAID_KINDS[self.name]
            use_sample_collate = _sample_collate_enabled(self.name)
            train = _wrap_sample_bridge(train, kind=kind, yield_sample=use_sample_collate)
            test = _wrap_sample_bridge(test, kind=kind, yield_sample=use_sample_collate)
            meta = dict(meta)
            if "mesh_test_data" in meta:
                meta["mesh_test_data"] = _wrap_sample_bridge(
                    meta["mesh_test_data"], kind=kind, yield_sample=use_sample_collate
                )
            meta["sample_bridge"] = "c2a_pyg"
            if use_sample_collate:
                # C4: dataset yields Sample; training must route the batch through
                # collate_plaid_static instead of torch_geometric's Collater (see
                # __main__.py's is_plaid gnn_loader bypass for sample_collate).
                meta["sample_collate"] = True
                meta["collate_fn_name"] = "plaid_static_sample"
                meta["train_collate_fn"] = collate_plaid_static
                meta["eval_collate_fn"] = collate_plaid_static
        return train, test, meta


class _GinotCanonicalAdapter:
    """GINOT static datasets: ``poisson_unstructured``, ``poisson_structured``,
    ``bracket_lug``, ``micro_puc``, ``micro_puc_fixed``, ``deform_plate``,
    ``bumper_beam``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def load(
        self,
        data_root: str,
        *,
        mesh_split_seed: int,
        ginot_max_samples: int,
        ginot_use_flash_varlen: bool,
        feature_request: FeatureRequest,
        **_unused: Any,
    ):
        train, test, meta = _utils().load_ginot_dataset(
            dataset_name=self.name,
            data_root=data_root,
            split_seed=mesh_split_seed,
            max_samples=ginot_max_samples,
            include_edges=feature_request.edges,
            use_flash_varlen=ginot_use_flash_varlen,
            laplacian_eig_dim=feature_request.laplacian_k,
            laplacian_spec=feature_request.laplacian_spec,
            include_padded_boundary=feature_request.boundary,
        )
        if _sample_bridge_enabled() and self.name in _SAMPLE_BRIDGE_GINOT_KINDS:
            kind = _SAMPLE_BRIDGE_GINOT_KINDS[self.name]
            use_sample_collate = _sample_collate_enabled_ginot(self.name)
            train = _wrap_ginot_sample_bridge(train, kind=kind, yield_sample=use_sample_collate)
            test = _wrap_ginot_sample_bridge(test, kind=kind, yield_sample=use_sample_collate)
            meta = dict(meta)
            meta["sample_bridge"] = "c2a_ginot"
            if use_sample_collate:
                # C4: dataset yields Sample; replace load_ginot_dataset's
                # partial(ginot_collate_fn, ...) with the equivalent
                # partial(collate_samples_ginot, ...), preserving the same
                # pad_to_nodes/pad_to_boundary_nodes/use_flash_varlen/
                # include_padded_boundary kwargs it was built with.
                legacy_collate_fn = meta.get("train_collate_fn")
                collate_kwargs = dict(getattr(legacy_collate_fn, "keywords", {}) or {})
                meta["sample_collate"] = True
                meta["collate_fn_name"] = "ginot_sample"
                sample_collate_fn = partial(collate_samples_ginot, **collate_kwargs)
                meta["train_collate_fn"] = sample_collate_fn
                meta["eval_collate_fn"] = sample_collate_fn
        return train, test, meta


class _LpbfAdapter:
    """LPBF: always ``load_lpbf_dataset`` (FLARE HF path or GLT graph-cache via ``FeatureRequest.edges``)."""

    name = "lpbf"

    def load(
        self,
        data_root: str,
        *,
        ginot_use_flash_varlen: bool,
        mesh_split_seed: int,
        ginot_max_samples: int,
        feature_request: FeatureRequest,
        **_unused: Any,
    ):
        train, test, meta = _utils().load_lpbf_dataset(
            data_root=data_root,
            include_edges=feature_request.edges,
            use_flash_varlen=ginot_use_flash_varlen,
            laplacian_eig_dim=feature_request.laplacian_k,
            laplacian_spec=feature_request.laplacian_spec,
            include_padded_boundary=feature_request.boundary,
            mesh_split_seed=mesh_split_seed,
            max_samples=ginot_max_samples,
        )
        # LPBF default (FLARE) path returns PyG ``Data``; the GLT/graph-cache path
        # (``feature_request.edges=True``) returns GINOT dict datasets — wrap those only.
        if _sample_bridge_enabled() and feature_request.edges:
            train = _wrap_ginot_sample_bridge(train, kind=SampleKind.STATIC)
            test = _wrap_ginot_sample_bridge(test, kind=SampleKind.STATIC)
            meta = dict(meta)
            meta["sample_bridge"] = "c2a_ginot"
        return train, test, meta


class _AhmedMLSurfaceAdapter:
    """AhmedML surface: ``load_ahmedml_surface_dataset`` with ``subset_size`` / ``iid_samples``."""

    name = "ahmedml_surface"

    def load(
        self,
        data_root: str,
        *,
        subset_size: int = 100_000,
        iid_samples: bool = True,
        **_unused: Any,
    ):
        return _utils().load_ahmedml_surface_dataset(
            data_root, subset_size=subset_size, iid_samples=iid_samples
        )


class _DrivAerMLSurfaceAdapter:
    """DrivAerML surface: ``load_drivaerml_surface_dataset`` with ``subset_size`` / ``iid_samples``."""

    name = "drivaerml_surface"

    def load(
        self,
        data_root: str,
        *,
        subset_size: int = 100_000,
        iid_samples: bool = True,
        **_unused: Any,
    ):
        return _utils().load_drivaerml_surface_dataset(
            data_root, subset_size=subset_size, iid_samples=iid_samples
        )


class _SimpleAdapter:
    """Single-arg loaders: ``elasticity`` / ``plasticity`` / ``pipe`` / ``airfoil_steady`` /
    ``darcy`` / ``navier_stokes`` / ``shapenet_car``.

    Resolved lazily off the ``utils`` module by attribute name so monkeypatches
    on e.g. ``dataset_utils.load_elasticity_dataset`` keep working.

    C2 (grids/geo): these loaders return ``TensorDataset`` / custom tuple items, not PyG
    ``Data`` or GINOT ``dict`` samples. C2a round-trip wrappers are intentionally skipped
    here — see ``tests/pdebench/test_sample_bridge_grids.py``.
    """

    def __init__(self, name: str, loader_attr: str) -> None:
        self.name = name
        self._loader_attr = loader_attr

    def load(self, data_root: str, **_unused: Any):
        loader = getattr(_utils(), self._loader_attr)
        return loader(data_root)


class _DrivAerAdapter:
    """``drivaerml_*``: resolved dynamically by name prefix, not pre-registered.

    C2 (grids/geo): ``DrivAerMLDataset.__getitem__`` returns a ``(pos, y)`` tensor tuple,
    not PyG/dict. C2a round-trip wrappers are skipped — see
    ``tests/pdebench/test_sample_bridge_grids.py``.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def load(self, data_root: str, **_unused: Any):
        return _utils().load_drivaerml_dataset(self.name, data_root)


_ADAPTERS: dict[str, DatasetAdapter] = {}


def _register_defaults() -> None:
    if _ADAPTERS:
        return
    for name in (
        "plaid_tensile2d",
        "plaid_hyperelasticity",
        "plaid_el_pl_dynamics",
        "plaid_elpl_terminal",
    ):
        _ADAPTERS[name] = _PlaidCanonicalAdapter(name)
    for name in (
        "poisson_unstructured",
        "poisson_structured",
        "bracket_lug",
        "micro_puc",
        "micro_puc_fixed",
        "deform_plate",
        "bumper_beam",
    ):
        _ADAPTERS[name] = _GinotCanonicalAdapter(name)
    _ADAPTERS["lpbf"] = _LpbfAdapter()
    _ADAPTERS["elasticity"] = _SimpleAdapter("elasticity", "load_elasticity_dataset")
    _ADAPTERS["plasticity"] = _SimpleAdapter("plasticity", "load_plasticity_dataset")
    _ADAPTERS["pipe"] = _SimpleAdapter("pipe", "load_pipe_dataset")
    _ADAPTERS["airfoil_steady"] = _SimpleAdapter("airfoil_steady", "load_airfoil_steady_dataset")
    _ADAPTERS["darcy"] = _SimpleAdapter("darcy", "load_darcy_dataset")
    _ADAPTERS["navier_stokes"] = _SimpleAdapter("navier_stokes", "load_navier_stokes_dataset")
    _ADAPTERS["shapenet_car"] = _SimpleAdapter("shapenet_car", "load_shapenet_car_dataset")
    _ADAPTERS["nasa_crm"] = _SimpleAdapter("nasa_crm", "load_nasa_crm_dataset")
    _ADAPTERS["ahmedml_surface"] = _AhmedMLSurfaceAdapter()
    _ADAPTERS["drivaerml_surface"] = _DrivAerMLSurfaceAdapter()

    assert set(_ADAPTERS) >= CANONICAL, "every CANONICAL dataset name must have a registered adapter"


def get_adapter(name: str) -> DatasetAdapter:
    """Resolve ``name`` (raw or canonical) to its registered ``DatasetAdapter``.

    Raises:
        RegistryError: if ``name`` was removed (via ``resolve_dataset_name``) or is
            unknown to the registry. ``RegistryError`` subclasses ``ValueError``, so
            callers that only expect ``ValueError`` for unknown names keep working.
    """
    _register_defaults()
    key = resolve_dataset_name(str(name).lower())
    if key in _ADAPTERS:
        return _ADAPTERS[key]
    if key.startswith("drivaerml"):
        return _DrivAerAdapter(key)
    raise RegistryError(f"Dataset {key} not found.")
