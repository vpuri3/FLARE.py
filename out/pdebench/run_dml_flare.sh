#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

#======================================================================#
DATASET="${DATASET:-drivaerml_1m}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
AMP_DTYPE="${AMP_DTYPE:-fp16}" # baseline-compatible autocast behavior
EMA="${EMA:-false}"
EPOCH="${EPOCH:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
RMSNORM="${RMSNORM:-true}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
ONE_CYCLE_PCT_START="${ONE_CYCLE_PCT_START:-0.05}"
ONE_CYCLE_DIV_FACTOR="${ONE_CYCLE_DIV_FACTOR:-10000.0}"
ONE_CYCLE_FINAL_DIV_FACTOR="${ONE_CYCLE_FINAL_DIV_FACTOR:-10000.0}"
ONE_CYCLE_THREE_PHASE="${ONE_CYCLE_THREE_PHASE:-false}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

COMMON_ARGS="--dataset.dataset ${DATASET}
      --training.mixed_precision ${MIXED_PRECISION}
	  --training.amp_dtype ${AMP_DTYPE}
      --model.rmsnorm ${RMSNORM}
      --training.epochs ${EPOCH}
      --training.batch_size ${BATCH_SIZE}
      --optimizer.weight_decay ${WEIGHT_DECAY}
      --training.ema ${EMA}
      --optimizer.learning_rate ${LEARNING_RATE}
      --scheduler.pct_start ${ONE_CYCLE_PCT_START}
      --scheduler.div_factor ${ONE_CYCLE_DIV_FACTOR}
      --scheduler.final_div_factor ${ONE_CYCLE_FINAL_DIV_FACTOR}
      --scheduler.three_phase ${ONE_CYCLE_THREE_PHASE}
      --training.num_workers ${NUM_WORKERS}"

#======================================================================#
# FLARE
#======================================================================#
# NUM_BLOCKS="${NUM_BLOCKS:-4}"
# NUM_CHANNELS="${NUM_CHANNELS:-64}"
# NUM_LATENTS="${NUM_LATENTS:-128}"
# NUM_HEADS="${NUM_HEADS:-8}"
#
# FLARE_ARGS="--model.model flare
#           --model.num_blocks ${NUM_BLOCKS}
#           --model.channel_dim ${NUM_CHANNELS}
#           --model.num_latents ${NUM_LATENTS}
#           --model.num_heads ${NUM_HEADS}"
#
# EXP_NAME="${EXP_NAME:-dml1m_flare_B4_C64_M128_H8}"
#
# python -m pdebench --run.train true ${ARGS} ${FLARE_ARGS} --run.exp_name ${EXP_NAME} ${EXTRA_ARGS}

#======================================================================#
# Transolver
#======================================================================#
NUM_BLOCKS="${NUM_BLOCKS:-4}"
NUM_CHANNELS="${NUM_CHANNELS:-256}"
NUM_SLICES="${NUM_SLICES:-32}"
NUM_HEADS="${NUM_HEADS:-8}"
MLP_RATIO="${MLP_RATIO:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"

MODEL_ARGS="--model.model transolver
          --model.num_blocks ${NUM_BLOCKS}
          --model.channel_dim ${NUM_CHANNELS}
          --model.num_slices ${NUM_SLICES}
          --model.num_heads ${NUM_HEADS}
          --model.mlp_ratio ${MLP_RATIO}
          --optimizer.learning_rate ${LEARNING_RATE}"

EXP_NAME="${EXP_NAME:-dml1m_tsr_B4_C256_H8}"

python -m pdebench --run.train true ${COMMON_ARGS} ${MODEL_ARGS} --run.exp_name ${EXP_NAME} ${EXTRA_ARGS}

#======================================================================#
