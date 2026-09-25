#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

#======================================================================#
NPROC="${NPROC:-1}"
DATASET="${DATASET:-drivaerml_1m}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EMA="${EMA:-false}"
EPOCH="${EPOCH:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
RMSNORM="${RMSNORM:-true}"
ONE_CYCLE_PCT_START="${ONE_CYCLE_PCT_START:-0.4}"
ONE_CYCLE_DIV_FACTOR="${ONE_CYCLE_DIV_FACTOR:-25}"
ONE_CYCLE_FINAL_DIV_FACTOR="${ONE_CYCLE_FINAL_DIV_FACTOR:-10}"
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
      --scheduler.pct_start ${ONE_CYCLE_PCT_START}
      --scheduler.div_factor ${ONE_CYCLE_DIV_FACTOR}
      --scheduler.final_div_factor ${ONE_CYCLE_FINAL_DIV_FACTOR}"

#======================================================================#
# Transolver
#======================================================================#
run_transolver() {
    local blocks="$1"
    local channels="$2"
    local slices="$3"
    local mlp_ratio="$4"
    local num_heads="${NUM_HEADS:-8}"
    local learning_rate="${LEARNING_RATE:-5e-4}"

    local MODEL_ARGS="--model.model transolver
          --model.num_blocks ${blocks}
          --model.channel_dim ${channels}
          --model.num_slices ${slices}
          --model.num_heads ${num_heads}
          --model.mlp_ratio ${mlp_ratio}
          --optimizer.learning_rate ${learning_rate}"

    # torchrun --standalone --nproc_per_node="${NPROC}" -m pdebench \
    #     --run.train true ${ARGS} ${tsr_args} \
    #     --training.use_context_parallel false \
    #     --run.exp_name model_transolver_B${blocks}_dp${NPROC}

    torchrun --standalone --nproc_per_node="${NPROC}" -m pdebench \
        --run.train true ${COMMON_ARGS} ${MODEL_ARGS} \
        --training.use_context_parallel false \
        --training.context_parallel_size "${NPROC}" \
        --training.cp_sequence_dim 1 \
        --run.exp_name model_transolver_B${blocks}_C${channels}_M${slices}
}

run0() {
	run_transolver "${NUM_BLOCKS:-1}" "${NUM_CHANNELS:-128}" "${NUM_SLICES:-32}" "${MLP_RATIO:-2.0}"
	run_transolver "${NUM_BLOCKS:-1}" "${NUM_CHANNELS:-256}" "${NUM_SLICES:-32}" "${MLP_RATIO:-1.0}"
}

run1() {
	run_transolver "${NUM_BLOCKS:-2}" "${NUM_CHANNELS:-128}" "${NUM_SLICES:-32}" "${MLP_RATIO:-2.0}"
	run_transolver "${NUM_BLOCKS:-2}" "${NUM_CHANNELS:-256}" "${NUM_SLICES:-32}" "${MLP_RATIO:-1.0}"
}

run2() {
	run_transolver "${NUM_BLOCKS:-4}" "${NUM_CHANNELS:-128}" "${NUM_SLICES:-32}" "${MLP_RATIO:-2.0}"
	run_transolver "${NUM_BLOCKS:-4}" "${NUM_CHANNELS:-256}" "${NUM_SLICES:-32}" "${MLP_RATIO:-1.0}"
}

run3() {
	run_transolver "${NUM_BLOCKS:-8}" "${NUM_CHANNELS:-128}" "${NUM_SLICES:-32}" "${MLP_RATIO:-2.0}"
	run_transolver "${NUM_BLOCKS:-8}" "${NUM_CHANNELS:-256}" "${NUM_SLICES:-32}" "${MLP_RATIO:-1.0}"
}

CUDA_VISIBLE_DEVICES=0 run0 &
CUDA_VISIBLE_DEVICES=1 run1 &
CUDA_VISIBLE_DEVICES=2 run2 &
CUDA_VISIBLE_DEVICES=3 run3 &

wait

#======================================================================#
