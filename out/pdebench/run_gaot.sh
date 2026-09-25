#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# Upstream reference:
# config/examples/time_indep/elasticity.json
# - coord_dim: 2
# - latent_tokens_size: [64, 64]
# - magno.hidden_size / lifting_channels: 64
# - transformer.patch_size: 2
# - transformer.hidden_size: 256 (= 64 * 2 * 2)
# - seed: 42

MODEL_TYPE=gaot

DATASET=elasticity
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=gpu --master_port 29501 -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model gaot --run.exp_name model_${MODEL_TYPE}_${DATASET} --use_puri2025flare_config true &

DATASET=darcy
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=gpu --master_port 29502 -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model gaot --run.exp_name model_${MODEL_TYPE}_${DATASET} --use_puri2025flare_config true &

wait

DATASET=airfoil_steady
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=gpu --master_port 29503 -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model gaot --run.exp_name model_${MODEL_TYPE}_${DATASET} --use_puri2025flare_config true &

DATASET=pipe
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=gpu --master_port 29504 -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model gaot --run.exp_name model_${MODEL_TYPE}_${DATASET} --use_puri2025flare_config true &

wait

DATASET=lpbf
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=gpu --master_port 29505 -m pdebench \
  --dataset.dataset "${DATASET}" --run.train true --model.model gaot --run.exp_name model_${MODEL_TYPE}_${DATASET} --use_puri2025flare_config true &

wait
