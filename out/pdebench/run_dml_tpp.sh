#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

for NUM_BLOCKS in 4 8; do
#======================================================================#
NPROC="${NPROC:-4}"
DATASET="${DATASET:-drivaerml_1m}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EMA="${EMA:-false}"
EPOCH="${EPOCH:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
RMSNORM="${RMSNORM:-true}"
NUM_WORKERS="${NUM_WORKERS:-8}"

COMMON_ARGS="--dataset.dataset ${DATASET}
      --training.mixed_precision ${MIXED_PRECISION}
      --training.amp_dtype ${AMP_DTYPE}
      --model.rmsnorm ${RMSNORM}
      --training.epochs ${EPOCH}
      --training.batch_size ${BATCH_SIZE}
      --training.num_workers ${NUM_WORKERS}
      --optimizer.weight_decay ${WEIGHT_DECAY}
      --training.ema ${EMA}
      --model.num_blocks ${NUM_BLOCKS}"

#======================================================================#
# Transolver++
#======================================================================#
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
NUM_CHANNELS="${NUM_CHANNELS:-128}"
NUM_SLICES="${NUM_SLICES:-128}"
NUM_HEADS="${NUM_HEADS:-8}"
MLP_RATIO="${MLP_RATIO:-2.0}"

MODEL_ARGS="--model.model transolver++
          --model.channel_dim ${NUM_CHANNELS}
          --model.num_slices ${NUM_SLICES}
          --model.num_heads ${NUM_HEADS}
          --model.mlp_ratio ${MLP_RATIO}
          --optimizer.learning_rate ${LEARNING_RATE}"

# torchrun --standalone --nproc_per_node="${NPROC}" -m pdebench \
#     --run.train true ${ARGS} ${TSR_ARGS} \
#     --training.use_context_parallel false \
#     --run.exp_name model_transolverpp_dp${NPROC}

    torchrun --standalone --nproc_per_node="${NPROC}" -m pdebench \
    --run.train true ${COMMON_ARGS} ${MODEL_ARGS} \
    --training.use_context_parallel true \
    --training.context_parallel_size "${NPROC}" \
    --training.cp_sequence_dim 1 \
    --run.exp_name model_transolverpp_cp${NPROC}_B${NUM_BLOCKS}
#======================================================================#
done # NUM_BLOCKS
