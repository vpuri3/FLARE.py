#
#============================#
# Setup
#============================#
# cd /project/community/$(whoami)/FLARE-dev.py
source .venv/bin/activate

#======================================================================#
# MODEL_TYPES:
# transolver: Transolver
# lno: LNO
# flare: FLARE
# transformer: Vanilla Transformer
# gnot: GNOT
# upt: UPT (not implemented)
# perceiverio: PerceiverIO
# transolver++: TransolverPlusPlus
# Comparison models
# hyperparameters are hard coded in pdebench/__main__.py
#======================================================================#

for DATASET in elasticity darcy airfoil_steady pipe drivaerml_40k lpbf; do
for MODEL_TYPE in transolver lno gnot perceiverio transolver++; do

    TRANSOLVER_CONV_FLAG=""
    if [ "${MODEL_TYPE}" = "transolver" ]; then
        case "${DATASET}" in
            darcy|airfoil_steady|pipe)
                TRANSOLVER_CONV_FLAG="--model.conv2d true"
                ;;
        esac
    fi

    python -m pdebench --dataset.dataset ${DATASET} --run.train true --use_puri2025flare_config true \
        --model.model ${MODEL_TYPE} ${TRANSOLVER_CONV_FLAG} --run.exp_name model_${MODEL_TYPE}_${DATASET}

done
done

###
# Transolver structured 2D mesh (conv2d) — explicit conv runs beyond the comparison loop
###

python -m pdebench --dataset.dataset darcy --run.train true --use_puri2025flare_config true \
    --model.conv2d true --model.unified_pos true --model.model transolver --run.exp_name model_transolver_conv_darcy

python -m pdebench --dataset.dataset airfoil_steady --run.train true --use_puri2025flare_config true \
    --model.conv2d true --model.model transolver --run.exp_name model_transolver_conv_airfoil_steady

python -m pdebench --dataset.dataset pipe --run.train true --use_puri2025flare_config true \
    --model.conv2d true --model.model transolver --run.exp_name model_transolver_conv_pipe

###
# Vanilla Transformer
###

python -m pdebench --dataset.dataset elasticity --run.train true --use_puri2025flare_config true \
    --model.model transformer --run.exp_name model_transformer_elasticity

python -m pdebench --dataset.dataset darcy --run.train true --use_puri2025flare_config true \
    --model.model transformer --run.exp_name model_transformer_darcy

python -m pdebench --dataset.dataset airfoil_steady --run.train true --use_puri2025flare_config true \
    --model.model transformer --run.exp_name model_transformer_airfoil_steady

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

python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model flare \
    --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
    --model.channel_dim ${NUM_CHANNELS} --model.num_latents ${NUM_LATENTS} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} \
    --run.seed 0 --run.exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=darcy
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=16

python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model flare \
    --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
    --model.channel_dim ${NUM_CHANNELS} --model.num_latents ${NUM_LATENTS} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} \
    --run.seed 0 --run.exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=airfoil_steady
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=8

python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model flare \
    --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
    --model.channel_dim ${NUM_CHANNELS} --model.num_latents ${NUM_LATENTS} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} \
    --run.seed 0 --run.exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=pipe
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=128
NUM_HEADS=8

python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model flare \
    --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
    --model.channel_dim ${NUM_CHANNELS} --model.num_latents ${NUM_LATENTS} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} \
    --run.seed 0 --run.exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=drivaerml_40k
EPOCH=500
BATCH_SIZE=1
WEIGHT_DECAY=1e-4

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=256
NUM_HEADS=8

python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model flare \
    --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
    --model.channel_dim ${NUM_CHANNELS} --model.num_latents ${NUM_LATENTS} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} \
    --run.seed 0 --run.exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
DATASET=lpbf
EPOCH=250
BATCH_SIZE=1
WEIGHT_DECAY=1e-4

NUM_BLOCKS=8
NUM_CHANNELS=64
NUM_LATENTS=128
NUM_HEADS=8

    python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model flare \
    --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
    --model.channel_dim ${NUM_CHANNELS} --model.num_latents ${NUM_LATENTS} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} \
    --run.seed 0 --run.exp_name model_flare_${DATASET}_B_${NUM_BLOCKS}_C_${NUM_CHANNELS}_M_${NUM_LATENTS}_H_${NUM_HEADS}

#======================================================================#
#
