"""Locked design parameters for PLAID el-pl v3 cache."""

from __future__ import annotations

CACHE_FORMAT = "elpl_v3"
CACHE_SCHEMA_VERSION = 1
TRAJECTORIES_PER_SHARD = 128
NUM_FIELD_SNAPSHOTS = 41
PLAID_ELPL_TARGET_FIELDS = ("U_x", "U_y")
PLAID_ELPL_INPUT_SCALAR_NAMES = ("time",)
