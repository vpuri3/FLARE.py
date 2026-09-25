"""GLT PE registry: ``GRAPH_PE_BY_KIND`` + ``build_pe`` factory.

See ``docs/superpowers/specs/2026-07-15-glt-pe-experiment-requirements-design.md``.
"""
from __future__ import annotations

from collections.abc import Callable

from .pe_base import GraphPE
from .pe_dist import (
    ProbeDistGraphPE,
    ProbeDistPEConfig,
    _build_probe_dist_pe,
)
from .pe_eigen import (
    RawEigenPE,
    RawEigenPEConfig,
    SpectralFilterPE,
    SpectralFilterPEConfig,
    _build_raw_eigen_pe,
    _build_spe_pe,
)
from .pe_hop import MultiscaleHopPE, MultiscaleHopPEConfig
from .pe_other import (
    GeoTransolverPE,
    GeoTransolverPEConfig,
    NonePE,
    NonePEConfig,
    _build_geo_transolver_pe,
    _build_none_pe,
)

__all__ = [
    "GRAPH_PE_BY_KIND",
    "GraphPEConfig",
    "build_pe",
]

_PEBuildFn = Callable[..., GraphPE]


def _build_multiscale_hop_pe(
    config: MultiscaleHopPEConfig, *, pos_dim: int, act: str, pos_domain=None
) -> MultiscaleHopPE:
    del act, pos_domain
    return MultiscaleHopPE(config, pos_dim=pos_dim)


GRAPH_PE_BY_KIND: dict[str, tuple[type, type, _PEBuildFn]] = {
    "none": (NonePEConfig, NonePE, _build_none_pe),
    "raw_eigen": (RawEigenPEConfig, RawEigenPE, _build_raw_eigen_pe),
    "spe": (SpectralFilterPEConfig, SpectralFilterPE, _build_spe_pe),
    "geo_transolver_pe": (GeoTransolverPEConfig, GeoTransolverPE, _build_geo_transolver_pe),
    "multiscale_hop_pe": (MultiscaleHopPEConfig, MultiscaleHopPE, _build_multiscale_hop_pe),
    "probe_dist": (ProbeDistPEConfig, ProbeDistGraphPE, _build_probe_dist_pe),
}
_GRAPH_PE_BY_CONFIG_TYPE: dict[type, tuple[str, type, _PEBuildFn]] = {
    config_cls: (kind, module_cls, build_fn)
    for kind, (config_cls, module_cls, build_fn) in GRAPH_PE_BY_KIND.items()
}
GraphPEConfig = (
    NonePEConfig
    | RawEigenPEConfig
    | SpectralFilterPEConfig
    | GeoTransolverPEConfig
    | MultiscaleHopPEConfig
    | ProbeDistPEConfig
)


def _resolve_pe_registry_entry(config: object) -> tuple[str, type, _PEBuildFn]:
    entry = _GRAPH_PE_BY_CONFIG_TYPE.get(type(config))
    if entry is not None:
        return entry
    kind = getattr(config, "kind", None)
    if isinstance(kind, str) and kind in GRAPH_PE_BY_KIND:
        config_cls, module_cls, build_fn = GRAPH_PE_BY_KIND[kind]
        if isinstance(config, config_cls):
            return kind, module_cls, build_fn
    raise TypeError(f"build_pe: unsupported PE config type {type(config)!r}.")


def build_pe(
    config: GraphPEConfig,
    *,
    pos_dim: int,
    act: str = "gelu",
    pos_domain=None,
) -> GraphPE:
    _, _, build_fn = _resolve_pe_registry_entry(config)
    if bool(config.to_feature_request().pos_domain) and pos_domain is None:
        raise ValueError(
            f"PE config {type(config).__name__} requested pos_domain but none was provided in metadata."
        )
    return build_fn(config, pos_dim=pos_dim, act=act, pos_domain=pos_domain)
