#!/usr/bin/env bash
# BATCH_SIZE is the global batch (sum over DDP ranks) unless context parallel is on.
# With CP, batch_size is per-rank (same sample on each rank, sequence-sharded).
# TRAINING_BATCH_SIZE is what --training.batch_size receives.
set -euo pipefail

TORCHRUN_NPROC="${TORCHRUN_NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
USE_CONTEXT_PARALLEL="${USE_CONTEXT_PARALLEL:-false}"

if [[ "${USE_CONTEXT_PARALLEL}" == "true" || "${USE_CONTEXT_PARALLEL}" == "1" ]]; then
  # CP: batch_size is per-rank; no DDP split of the batch across ranks.
  export PER_RANK_BATCH_SIZE="${BATCH_SIZE}"
  export TRAINING_BATCH_SIZE="${BATCH_SIZE}"
else
  if (( BATCH_SIZE % TORCHRUN_NPROC != 0 )); then
    echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by TORCHRUN_NPROC=${TORCHRUN_NPROC}" >&2
    exit 1
  fi
  export PER_RANK_BATCH_SIZE=$(( BATCH_SIZE / TORCHRUN_NPROC ))
  export TRAINING_BATCH_SIZE="${BATCH_SIZE}"
fi
