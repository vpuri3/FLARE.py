"""Per-sample scoring and case selection.

The selection rule is fixed here rather than chosen per figure, so that a
caption can state it and a reader can check it. Never the best case.

Metric definitions mirror ``pdebench/callbacks.py`` exactly, because the whole
point of stage A is that the figure and the reported table describe the same
model on the same data:

  - ``rel_l2``          -- ``RelL2Loss``, over all output channels jointly.
  - ``pressure_rel_l2`` -- channel 0 only (surface benchmarks).
  - ``wall_shear_rel_l2`` -- on the *norm* of channels 1: , not componentwise,
    matching ``_surface_metric_sums``.
"""

from __future__ import annotations

import numpy as np

# Datasets scored with the surface metric split of ``_surface_metric_sums``.
SURFACE_METRIC_DATASETS: frozenset[str] = frozenset(
    {"nasa_crm", "ahmedml_surface", "drivaerml_surface", "drivaerml_40k"}
)

# The metric each dataset is selected and reported on.
PRIMARY_METRIC: dict[str, str] = {
    "nasa_crm": "pressure_rel_l2",
    "ahmedml_surface": "pressure_rel_l2",
    "drivaerml_surface": "pressure_rel_l2",
    "drivaerml_40k": "pressure_rel_l2",
}
_DEFAULT_PRIMARY = "rel_l2"


def primary_metric(dataset: str) -> str:
    return PRIMARY_METRIC.get(dataset, _DEFAULT_PRIMARY)


def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.sqrt(np.sum((pred - target) ** 2)))
    den = float(np.sqrt(np.sum(target**2)))
    return num / den


def sample_metrics(dataset: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """All metrics for one sample, in physical units. ``[N, F]`` arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {target.shape}")

    metrics = {"rel_l2": _rel_l2(pred, target)}

    if dataset in SURFACE_METRIC_DATASETS:
        if target.shape[1] < 4:
            raise ValueError(f"{dataset}: surface metrics need >=4 channels, got {target.shape[1]}")
        metrics["full_rel_l2"] = metrics["rel_l2"]
        metrics["pressure_rel_l2"] = _rel_l2(pred[:, 0], target[:, 0])
        pred_tau = np.linalg.norm(pred[:, 1:], axis=-1)
        target_tau = np.linalg.norm(target[:, 1:], axis=-1)
        metrics["wall_shear_rel_l2"] = _rel_l2(pred_tau, target_tau)

    return metrics


def per_sample_errors(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Transpose a list of per-sample metric dicts into arrays."""
    if not records:
        raise ValueError("no samples scored")
    keys = records[0].keys()
    return {key: np.asarray([r[key] for r in records], dtype=np.float64) for key in keys}


def select_cases(
    errors_ref: np.ndarray,
    errors_other: np.ndarray | None = None,
) -> dict[str, int]:
    """Pick the sample indices to visualize.

    ``errors_ref`` is the per-sample error of the model the selection is keyed
    on (FLARE++). ``errors_other`` (FLARE) enables the improvement cases.

    The median is an actual sample -- the ``n//2``-th order statistic -- not an
    interpolated value, so it can be rendered.
    """
    errors_ref = np.asarray(errors_ref, dtype=np.float64)
    if errors_ref.ndim != 1 or errors_ref.size == 0:
        raise ValueError(f"expected a non-empty 1D error array, got shape {errors_ref.shape}")

    order = np.argsort(errors_ref, kind="stable")
    cases = {
        "median": int(order[errors_ref.size // 2]),
        "max": int(order[-1]),
    }

    if errors_other is not None:
        errors_other = np.asarray(errors_other, dtype=np.float64)
        if errors_other.shape != errors_ref.shape:
            raise ValueError(f"error arrays disagree: {errors_other.shape} vs {errors_ref.shape}")
        gain = errors_other - errors_ref  # positive where the ref model is better
        cases["best_gain"] = int(np.argmax(gain))
        cases["worst_gain"] = int(np.argmin(gain))

    return cases


def check_against_reported(
    dataset: str,
    model: str,
    metric: str,
    recomputed_mean: float,
    reported: float | None,
    *,
    tol: float = 1e-2,
) -> str:
    """Compare the recomputed test mean with the manuscript value.

    Returns a human-readable line. Never silently reconciles the two: a
    mismatch is reported as a mismatch, because a figure drawn from a
    checkpoint that does not reproduce the table is evidence of nothing.
    """
    line = f"{dataset:18s} {model:10s} {metric:20s} recomputed={100 * recomputed_mean:.3f}%"
    if reported is None:
        return line + "  reported=n/a"
    delta = abs(recomputed_mean - reported)
    verdict = "OK" if delta <= tol * max(reported, 1e-12) else "MISMATCH"
    return line + f"  reported={100 * reported:.3f}%  {verdict}"
