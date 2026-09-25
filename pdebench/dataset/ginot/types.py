from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pdebench.dataset.normalizer import MeanStdNormalizer

MICRO_PUC_MESH_SAMPLES = 10_000
MICRO_PUC_CANONICAL_MESH_ROWS = MICRO_PUC_MESH_SAMPLES
MICRO_PUC_TOTAL_SAMPLES = 73_879
FIXED_DATASET_DIRNAME = "PeriodUnitCell_fixed"
MANIFEST_NAME = "manifest.json"
MICRO_PUC_FIXED_SOURCE_SAMPLES = MICRO_PUC_TOTAL_SAMPLES

GINOT_DATASETS = {
    "poisson_unstructured",
    "poisson_structured",
    "bracket_lug",
    "micro_puc",
    "micro_puc_fixed",
    "deform_plate",
    "bumper_beam",
}

# Compat alias — shared encode/decode lives in ``pdebench.dataset.normalizer``.
StandardNormalizer = MeanStdNormalizer


@dataclass(frozen=True)
class GinotRawDataset:
    query_points: Any
    point_clouds: Any
    targets: Any
    cells: Any
    input_params: np.ndarray | None
    target_fields: tuple[str, ...]
    space_dim: int
    target_normalizer: StandardNormalizer | None = None
    normalize_pos: bool = True
    normalize_boundary_pos: bool = True
    normalize_targets: bool = True
    dataset_dir: str = ""
    micro_puc_mesh_idx: np.ndarray | None = None
    micro_puc_source_geometry_ids: np.ndarray | None = None
    deform_plate_store: Any | None = None
    bumper_beam_store: Any | None = None
    mesh_pos: Any | None = None
    node_types: Any | None = None
    free_mask: Any | None = None
    bc_mask: Any | None = None
    u_prescribed: Any | None = None
    plate_nodes: Any | None = None
    plate_cells: Any | None = None
    plate_mesh_pos: Any | None = None
    precomputed_edge_index: Any | None = None
