"""Locked design parameters for PLAID el-pl terminal prediction."""

from __future__ import annotations

CACHE_TAG = "terminal"
FINAL_STEP_IDX = 40
FINAL_TIME = 0.04
NUM_FIELD_SNAPSHOTS = 41
TARGET_FIELDS = ("U_x", "U_y")
FULL_TARGET_FIELDS = TARGET_FIELDS
DEFAULT_TERMINAL_TARGET_FIELDS = ("U_x",)


def parse_terminal_target_fields(
    raw: str | None,
    *,
    default: tuple[str, ...] = DEFAULT_TERMINAL_TARGET_FIELDS,
) -> tuple[str, ...]:
    """Parse comma-separated terminal target field names (subset of U_x, U_y)."""
    if raw is None or not str(raw).strip():
        return default
    fields = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not fields:
        return default
    allowed = set(TARGET_FIELDS)
    unknown = [name for name in fields if name not in allowed]
    if unknown:
        raise ValueError(f"Unknown terminal target field(s) {unknown} (allowed: {TARGET_FIELDS}).")
    if len(set(fields)) != len(fields):
        raise ValueError(f"Duplicate terminal target fields in {fields!r}.")
    return fields


def terminal_target_field_indices(target_fields: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(TARGET_FIELDS.index(name) for name in target_fields)

# Frozen terminal cache always stores z-score y stats (split5_sdf0_norm_stats.pt).
CACHE_Y_NORM_VERSION = 1

# Runtime y-normalization modes (applied at graph assembly; no cache rebuild).
RUNTIME_Y_NORM_MODES = ("cache", "asinh_iqr")
DEFAULT_RUNTIME_Y_NORM = "asinh_iqr"

# asinh+IQR runtime normalizer parameters.
UX_CLIP_LO = 0.0
UX_CLIP_HI = 20.0
ROBUST_SCALE_FACTOR = 1.4826
Y_NORM_IQR_FLOOR = 1e-8
Y_NORM_SCALE_FLOOR = 1e-6
RUNTIME_ASINH_IQR_VERSION = 3
