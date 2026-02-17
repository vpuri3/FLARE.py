#
#======================================================================#
source ~/.bash_profile
source .venv/bin/activate
#======================================================================#

#======================================================================#
# Elasticity
#======================================================================#
DATASET=elasticity
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

CHANNEL_DIM=64
NUM_LATENTS=64
NUM_HEADS=8

#======================================================================#
# NUM_BLOCKS = 1
#======================================================================#
DEVICE=0
NUM_BLOCKS=1

SHARED_LATENTS=False
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

SHARED_LATENTS=True
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

#======================================================================#
# NUM_BLOCKS = 2
#======================================================================#
DEVICE=1
NUM_BLOCKS=2

SHARED_LATENTS=False
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

SHARED_LATENTS=True
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

#======================================================================#
# NUM_BLOCKS = 4
#======================================================================#
DEVICE=0
NUM_BLOCKS=4

SHARED_LATENTS=False
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

SHARED_LATENTS=True
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

#======================================================================#
# NUM_BLOCKS = 8
#======================================================================#
DEVICE=0
NUM_BLOCKS=8

SHARED_LATENTS=False
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

SHARED_LATENTS=True
NUM_LATENT_BLOCKS=0
EXP_NAME=sl/model_7_${DATASET}_SL_${SHARED_LATENTS}_LB_${NUM_LATENT_BLOCKS}_BCMH_${NUM_BLOCKS}_${CHANNEL_DIM}_${NUM_LATENTS}_${NUM_HEADS}

CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --dataset ${DATASET} --train true --model_type flare_ablations \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${CHANNEL_DIM} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --shared_att ${SHARED_LATENTS} --num_passes ${NUM_LATENT_BLOCKS} \
    --seed 0 --exp_name ${EXP_NAME} &

# CUDA_VISIBLE_DEVICES=${DEVICE} python -m pdebench --evaluate true --exp_name ${EXP_NAME}

#======================#
wait
#======================#

#======================================================================#
#