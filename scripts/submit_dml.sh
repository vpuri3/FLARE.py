#!/bin/bash

#============================#
# Advanced Queue
#============================#
#SBATCH --account=lkara
#SBATCH --partition=advanced
#SBATCH --qos=adv_4gpu_qos
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=FLARE

# #============================#
# # General Queue
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=general
# #SBATCH --qos=general_qos
# #SBATCH --gres=gpu:1
# #SBATCH --time=12:00:00
# #SBATCH --job-name=FLARE

# #============================#
# # Preempt Queue
# #============================#
# #SBATCH --account=lkara
# #SBATCH --partition=preempt
# #SBATCH --qos=preempt_qos
# #SBATCH --gres=gpu:1
# #SBATCH --time=18:00:00
# #SBATCH --job-name=FLARE

#============================#
# srun --pty --overlap --jobid <job_id> bash
#============================#

#============================#
# Setup
#============================#
echo "Setting up environment"
cd /project/community/$(whoami)/FLARE-dev.py

#============================#
# Run
#============================#
# sleep 12h
# ./out/pdebench/run_dml.sh
# uv run python ./ablation/time_memory_lca_flash.py --run true --plot true
#============================#

# DATASET=drivaerml_1m
# LR=5.0e-4
# CLIP=1.0
# BATCH_SIZE=1
# WEIGHT_DECAY=1e-5
# PCT=0.05
# EPOCH=500

# # B = 1,2,4,8
# # M = 128, 512, 1024, 2048
# # C = 64
# # H = C/8 = 8

# NUM_BLOCKS=8
# NUM_LATENTS=2048
# CHANNEL_DIM=64
# NUM_HEADS=8

# uv run python -m pdebench --dataset ${DATASET} --train true --model_type 2 \
#     --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} --learning_rate ${LR} \
#     --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
#     --mixed_precision true --attn_backend flash --clip_grad_norm ${CLIP} --opt_eps 1e-6 --num_workers 5 \
#     --one_cycle_pct_start ${PCT} --seed 0 --exp_name lca_${DATASET}_B_${NUM_BLOCKS}_C_${CHANNEL_DIM}_L_${NUM_LATENTS}_H_${NUM_HEADS} &

# # restart
# uv run python -m pdebench --restart true \
#     --exp_name lca_${DATASET}_B_${NUM_BLOCKS}_C_${CHANNEL_DIM}_L_${NUM_LATENTS}_H_${NUM_HEADS} &

#============================#
wait
#============================#
#