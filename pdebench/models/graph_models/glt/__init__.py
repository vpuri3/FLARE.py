"""GLT package: backbone + pluggable positional encodings.

Modules
-------
- ``backbone`` — ``GLT`` / ``GLTConfig`` / attention / injection
- ``pe_eigen`` — ``raw_eigen`` and ``spe``
- ``pe_hop`` — ``multiscale_hop_pe``
- ``pe_dist`` — unified ``probe_dist`` (Euclidean + optional geodesic)
- ``pe_other`` — ``none`` and ``geo_transolver_pe``
- ``registry`` — ``GRAPH_PE_BY_KIND`` / ``build_pe``
- ``pe_base`` — shared ``GraphPE`` contract and packed-graph helpers

Experiment intake:
``docs/superpowers/specs/2026-07-15-glt-pe-experiment-requirements-design.md``.
"""
from .backbone import (
    GLT,
    PE_INJECT_MODES,
    GLTBlock,
    GLTConfig,
    GLTMHAAttention,
)
from .pe_base import GraphPE, TopologyFeatures, _normalize_pe, _resolve_packed_indices
from .pe_dist import (
    ProbeDistGraphPE,
    ProbeDistPE,
    ProbeDistPEConfig,
)
from .pe_eigen import (
    RawEigenPE,
    RawEigenPEConfig,
    SpectralFilterPE,
    SpectralFilterPEConfig,
    _parse_pe_spe_mode,
)
from .pe_hop import MultiscaleHopPE, MultiscaleHopPEConfig
from .pe_other import (
    GeoTransolverPE,
    GeoTransolverPEConfig,
    NonePE,
    NonePEConfig,
)
from .registry import GRAPH_PE_BY_KIND, GraphPEConfig, build_pe

__all__ = [
    "GRAPH_PE_BY_KIND",
    "GLT",
    "GLTBlock",
    "GLTConfig",
    "GLTMHAAttention",
    "GeoTransolverPE",
    "GeoTransolverPEConfig",
    "GraphPE",
    "GraphPEConfig",
    "MultiscaleHopPE",
    "MultiscaleHopPEConfig",
    "NonePE",
    "NonePEConfig",
    "PE_INJECT_MODES",
    "ProbeDistGraphPE",
    "ProbeDistPE",
    "ProbeDistPEConfig",
    "RawEigenPE",
    "RawEigenPEConfig",
    "SpectralFilterPE",
    "SpectralFilterPEConfig",
    "TopologyFeatures",
    "build_pe",
    "_normalize_pe",
    "_parse_pe_spe_mode",
    "_resolve_packed_indices",
]
