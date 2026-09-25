"""Sentinel values for metrics that were never computed (distinct from numerical NaN)."""

from __future__ import annotations

import math
from typing import Any


def unset_metric() -> None:
    return None


def metric_is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (int,)):
        return True
    return True


def normalize_metric(value: Any) -> Any:
    """Map non-finite floats to ``None`` for JSON / logging."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def format_scalar(value: Any, *, precision: int = 6) -> str:
    if not metric_is_set(value):
        return "null"
    return f"{float(value):.{precision}e}"


def format_metric(value: Any) -> str:
    return format_scalar(value, precision=6)
