#
# MODEL_TYPES:
#
# transolver
# lno
# flare
# transformer
# gnot
# upt (not implemented)
# perceiverio
#
#======================================================================#
# Comparison models
# hyperparameters are hard coded in pdebench/__main__.py for now
# for all models but FLARE.
#======================================================================#

for DATASET in elasticity darcy airfoil_steady pipe drivaerml_40k lpbf; do
for MODEL_TYPE in transolver lno gnot perceiverio; do

    uv run python -m pdebench --dataset ${DATASET} --train true \
        --model_type ${MODEL_TYPE} --exp_name model_${MODEL_TYPE}_${DATASET}

done
done

###
# Transolver with conv2d and/or unified_pos
###

uv run python -m pdebench --dataset darcy --train true \
    --conv2d true --unified_pos true --model_type transolver --exp_name model_transolver_conv_darcy

uv run python -m pdebench --dataset airfoil_steady --train true \
    --conv2d true --model_type transolver --exp_name model_transolver_conv_airfoil_steady

uv run python -m pdebench --dataset pipe --train true \
    --conv2d true --model_type transolver --exp_name model_transolver_conv_pipe

###
# Vanilla Transformer
###

uv run python -m pdebench --dataset elasticity --train true \
    --model_type transformer --exp_name model_transformer_elasticity

uv run python -m pdebench --dataset darcy --train true \
    --model_type transformer --exp_name model_transformer_darcy

uv run python -m pdebench --dataset airfoil_steady --train true \
    --model_type transformer --exp_name model_transformer_airfoil_steady

#======================================================================#
# FLARE
#======================================================================#
DATASET=elasticity
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=64
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type flare \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=darcy
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=16

uv run python -m pdebench --dataset ${DATASET} --train true --model_type flare \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=airfoil_steady
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type flare \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=pipe
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=128
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type flare \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=drivaerml_40k
EPOCH=500
BATCH_SIZE=1
WEIGHT_DECAY=1e-4

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type flare \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=lpbf
EPOCH=250
BATCH_SIZE=1
WEIGHT_DECAY=1e-4

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=128
NUM_HEADS=8

uv run python -m pdebench --dataset ${DATASET} --train true --model_type flare \
    --epochs ${EPOCH} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} \
    --channel_dim ${NUM_CHANNELS} --num_latents ${NUM_LATENTS} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} \
    --seed 0 --exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
#
