#!/usr/bin/env bash
# Shared GINOT dataloader env for out/pdebench/run_glt.sh and sweep jobs.
#
# Worker / prefetch: run_glt.sh defaults NUM_WORKERS=8. Override with NUM_WORKERS /
# PREFETCH_FACTOR for a one-off experiment (this helper does not set them).
#
# GINOT_INPUT_PARAMS_INCLUDE_GEOMETRY (default 0)
#   Include bracket_lug input_params cols 0–2 (Rh, shank, Lz) in per-node feats.
# GINOT_INPUT_PARAMS_INCLUDE_LOAD (default 1)
#   Include bracket_lug input_params col 3 (applied load) in per-node feats.
#
# GINOT_LMDB_OPEN_SHARD_LRU (default 128): open-shard LRU for high-shard LMDB caches.

export GINOT_INPUT_PARAMS_INCLUDE_GEOMETRY="${GINOT_INPUT_PARAMS_INCLUDE_GEOMETRY:-0}"
export GINOT_INPUT_PARAMS_INCLUDE_LOAD="${GINOT_INPUT_PARAMS_INCLUDE_LOAD:-1}"
export GINOT_LMDB_OPEN_SHARD_LRU="${GINOT_LMDB_OPEN_SHARD_LRU:-128}"
