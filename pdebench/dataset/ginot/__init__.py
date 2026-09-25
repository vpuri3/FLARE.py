"""GINOT dataset package.

Module layout:
- types, utils: shared dataclasses, splits, normalizers
- mesh: connectivity, topology keys, edge construction
- io: raw dataset loaders
- sample: sample encoding for live reads and static cache build
- graph_cache: sharded LMDB graph static cache
- dataset: GinotDataset, GraphCacheDataset, batch samplers
- loader: train/test dataset construction
- collate, forward, stats: training batching and metrics
- precompute, visualize: offline cache build and QA plots

Laplacian eigenfeatures live in ``pdebench.dataset.laplacian`` (not under ginot).
"""

from pdebench.dataset.ginot.collate import ginot_collate_fn
from pdebench.dataset.ginot.dataset import GinotDataset, GraphCacheDataset, LengthBucketBatchSampler
from pdebench.dataset.ginot.forward import (
    ginot_apply_deform_plate_boundary_conditions,
    ginot_model_forward,
    ginot_per_graph_channel_rel_l2,
    ginot_per_graph_free_node_rel_l2,
    ginot_postprocess_displacement,
)
from pdebench.dataset.ginot.graph_cache import (
    ensure_graph_cache as _ensure_graph_cache,
)
from pdebench.dataset.ginot.graph_cache import (
    graph_cache_root as _graph_cache_root,
)
from pdebench.dataset.ginot.graph_cache import (
    load_graph_cache_sample as _load_graph_cache_sample,
)
from pdebench.dataset.ginot.io import load_raw_dataset as _load_raw_dataset
from pdebench.dataset.ginot.loader import load_ginot_dataset, load_ginot_precompute_context
from pdebench.dataset.ginot.mesh import (
    build_micro_puc_mesh_idx as _build_micro_puc_mesh_idx,
)
from pdebench.dataset.ginot.mesh import (
    normalize_cells as _normalize_cells,
)
from pdebench.dataset.ginot.mesh import (
    select_cells as _select_cells,
)
from pdebench.dataset.ginot.stats import make_ginot_statsfun
from pdebench.dataset.ginot.storage import ShardedMicroPucFixedSequence
from pdebench.dataset.ginot.types import (
    FIXED_DATASET_DIRNAME,
    GINOT_DATASETS,
    MANIFEST_NAME,
    MICRO_PUC_CANONICAL_MESH_ROWS,
    MICRO_PUC_FIXED_SOURCE_SAMPLES,
    MICRO_PUC_MESH_SAMPLES,
    MICRO_PUC_TOTAL_SAMPLES,
    GinotRawDataset,
    StandardNormalizer,
)
from pdebench.dataset.ginot.utils import (
    as_float_array as _as_float_array,
)
from pdebench.dataset.ginot.utils import (
    is_micro_puc_fixed_raw as _is_micro_puc_fixed_raw,
)
from pdebench.dataset.ginot.utils import (
    is_micro_puc_raw as _is_micro_puc_raw,
)
from pdebench.dataset.ginot.utils import (
    num_indexable_samples as _num_indexable_samples,
)

_MICRO_PUC_FIXED_EXPORTS = {
    "MANIFEST_NAME": "MANIFEST_NAME",
    "MicroPucFixedConfig": "MicroPucFixedConfig",
    "MicroPucFixedStats": "MicroPucFixedStats",
    "audit_micro_puc_periodic_boundaries": "audit_micro_puc_periodic_boundaries",
    "build_micro_puc_fixed_dataset": "build_micro_puc_fixed_dataset",
    "build_periodic_node_map": "build_periodic_node_map",
    "fix_one_sample": "fix_one_sample",
}


def __getattr__(name: str):
    if name in _MICRO_PUC_FIXED_EXPORTS:
        from pdebench.dataset.ginot import micro_puc_fixed

        value = getattr(micro_puc_fixed, _MICRO_PUC_FIXED_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GINOT_DATASETS",
    "FIXED_DATASET_DIRNAME",
    "GinotDataset",
    "GraphCacheDataset",
    "GinotRawDataset",
    "LengthBucketBatchSampler",
    "MANIFEST_NAME",
    "MICRO_PUC_CANONICAL_MESH_ROWS",
    "MICRO_PUC_FIXED_SOURCE_SAMPLES",
    "MICRO_PUC_MESH_SAMPLES",
    "MICRO_PUC_TOTAL_SAMPLES",
    "MicroPucFixedConfig",
    "MicroPucFixedStats",
    "ShardedMicroPucFixedSequence",
    "StandardNormalizer",
    "audit_micro_puc_periodic_boundaries",
    "build_micro_puc_fixed_dataset",
    "build_periodic_node_map",
    "fix_one_sample",
    "ginot_collate_fn",
    "ginot_model_forward",
    "ginot_apply_deform_plate_boundary_conditions",
    "ginot_postprocess_displacement",
    "ginot_per_graph_channel_rel_l2",
    "ginot_per_graph_free_node_rel_l2",
    "load_ginot_dataset",
    "load_ginot_precompute_context",
    "make_ginot_statsfun",
    "_as_float_array",
    "_build_micro_puc_mesh_idx",
    "_ensure_graph_cache",
    "_graph_cache_root",
    "_load_graph_cache_sample",
    "_is_micro_puc_fixed_raw",
    "_is_micro_puc_raw",
    "_load_raw_dataset",
    "_normalize_cells",
    "_num_indexable_samples",
    "_select_cells",
]
