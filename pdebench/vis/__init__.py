"""Qualitative-visualization export.

Stage A of the manuscript figure pipeline: load trained checkpoints, score the
test split, pick the cases to show, and write self-describing ``.npz`` bundles.

Deliberately free of matplotlib / VTK / OpenGL. Rendering is stage B and runs
off-cluster from the bundles this package writes.
"""

from pdebench.vis.extract import DATASETS, extract_bundle, field_names
from pdebench.vis.select import per_sample_errors, select_cases

__all__ = [
    "DATASETS",
    "extract_bundle",
    "field_names",
    "per_sample_errors",
    "select_cases",
]
