#
#============================#
# Setup
#============================#
source ~/.bash_profile
cd /project/community/$(whoami)/FLARE-dev.py
source .venv/bin/activate

#========================================================#
#========================================================#
# SUDOKU
#========================================================#
#========================================================#
# rm -rf data/Sudoku/*

TASK=sudoku

STEPS=1_00
EPOCHS=0
BATCH_SIZE=256
WEIGHT_DECAY=1e-0
LEARNING_RATE=1e-4
MIXED_PRECISION=true

SCHEDULE=OneCycleLR
ONE_CYCLE_OVERRIDE_MIN_LR=1e-6
ONE_CYCLE_DIV_FACTOR=100.0
ONE_CYCLE_FINAL_DIV_FACTOR=0.01
ONE_CYCLE_MAX_MOMENTUM=0.95
ONE_CYCLE_BASE_MOMENTUM=0.85
ONE_CYCLE_CYCLE_MOMENTUM=true
ONE_CYCLE_ANNEAL_STRATEGY=cos
ONE_CYCLE_PCT_START=0.1
ONE_CYCLE_THREE_PHASE=false

NUM_BLOCKS=2
CHANNEL_DIM=256
NUM_HEADS=8
ATTN_DROP=0.0
PROJ_DROP=0.0

# TO simulate vanilla model: TRM_N_STEPS=1, TRM_n=0, TRM_T=1

TRM=true
TRM_N_STEPS=5
TRM_N=2
TRM_T=1

EMA=true
EMA_DECAY=0.999

#============================#
# TRANSFORMER
#============================#
MODEL_TYPE=transformer
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
    --schedule ${SCHEDULE} \
    --one_cycle_div_factor ${ONE_CYCLE_DIV_FACTOR} \
    --one_cycle_final_div_factor ${ONE_CYCLE_FINAL_DIV_FACTOR} \
    --one_cycle_max_momentum ${ONE_CYCLE_MAX_MOMENTUM} \
    --one_cycle_base_momentum ${ONE_CYCLE_BASE_MOMENTUM} \
    --one_cycle_cycle_momentum ${ONE_CYCLE_CYCLE_MOMENTUM} \
    --one_cycle_anneal_strategy ${ONE_CYCLE_ANNEAL_STRATEGY} \
    --one_cycle_pct_start ${ONE_CYCLE_PCT_START} \
    --one_cycle_three_phase ${ONE_CYCLE_THREE_PHASE} \
    --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} \
    --trm ${TRM} --trm_N_steps ${TRM_N_STEPS} --trm_n ${TRM_N} --trm_T ${TRM_T} \
    --exp_name ${TASK}/trm_N_${TRM_N_STEPS}_n_${TRM_N}_T_${TRM_T}

#============================#
exit
#============================#
#