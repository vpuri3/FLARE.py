#
#============================#
# Setup
#============================#
source ~/.bash_profile
cd /project/community/$(whoami)/FLARE-dev.py
source .venv/bin/activate

#========================================================#
#========================================================#
# FUNCTION COMPOSITION
#========================================================#
#========================================================#
TASK=function_composition

STEPS=0
EPOCHS=1_000
BATCH_SIZE=2500
WEIGHT_DECAY=0e-5
LEARNING_RATE=1e-3
# SCHEDULE=ConstantLR
SCHEDULE=OneCycleLR
MIXED_PRECISION=false
RMSNORM=${MIXED_PRECISION}

POOL=cls
NUM_BLOCKS=1
CHANNEL_DIM=16
NUM_HEADS=1
ATTN_DROP=0.3
PROJ_DROP=0.3

EMA=false
EMA_DECAY=0.999

#============================#
# TRANSFORMER
#============================#
WEIGHT_DECAY=1e-5
ATTN_DROP=0.0
PROJ_DROP=0.0

# MODEL_TYPE=transformer
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} \
#     --exp_name ${TASK}/trans

#============================#
# THIRD ORDER ATTENTION
#============================#
WEIGHT_DECAY=1e-5
ATTN_DROP=0.0
PROJ_DROP=0.0

# MODEL_TYPE=third_order
# THIRD_ORDER_METHOD=strassen
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
#     --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

# MODEL_TYPE=third_order
# THIRD_ORDER_METHOD=third_order
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
#     --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

# #============================#
# # TRIPLE
# #============================#
# WEIGHT_DECAY=1e-5
# ATTN_DROP=0.0
# PROJ_DROP=0.0

# MODEL_TYPE=triple
# NUM_LAYERS_KV_PROJ=-1; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; QK_DIM_RATIO=1.0 
# KERNEL=identity
# NORM_Q=true
# NORM_K=true
# USE_TRITON=false
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} --mixed_precision ${MIXED_PRECISION} \
#     --epochs ${EPOCHS} --steps ${STEPS} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} --schedule ${SCHEDULE} \
#     --learning_rate "${LEARNING_RATE}" --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --channel_dim ${CHANNEL_DIM} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} --rmsnorm ${RMSNORM} \
#     --model_type ${MODEL_TYPE} --kernel ${KERNEL} --norm_q ${NORM_Q} --norm_k ${NORM_K} --qk_dim_ratio ${QK_DIM_RATIO} --use_triton ${USE_TRITON} \
#     --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
#     --exp_name ${TASK}/${MODEL_TYPE}

#========================================================#
#========================================================#
# BINARY RELATION COMPOSITION
#========================================================#
#========================================================#
TASK=binary_relation_composition

STEPS=0
EPOCHS=200
BATCH_SIZE=2500
WEIGHT_DECAY=0e-5
LEARNING_RATE=1e-3
# SCHEDULE=ConstantLR
SCHEDULE=OneCycleLR
MIXED_PRECISION=false
RMSNORM=${MIXED_PRECISION}

NUM_BLOCKS=1
CHANNEL_DIM=16
NUM_HEADS=1
ATTN_DROP=0.3
PROJ_DROP=0.3

EMA=false
EMA_DECAY=0.999

# #============================#
# # TRANSFORMER
# #============================#
# WEIGHT_DECAY=1e-5
# ATTN_DROP=0.0
# PROJ_DROP=0.0

# MODEL_TYPE=transformer
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} \
#     --exp_name ${TASK}/trans

#============================#
# THIRD ORDER ATTENTION
#============================#
WEIGHT_DECAY=1e-5
ATTN_DROP=0.0
PROJ_DROP=0.0

# MODEL_TYPE=third_order
# THIRD_ORDER_METHOD=strassen
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
#     --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

MODEL_TYPE=third_order
THIRD_ORDER_METHOD=third_order
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
    --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
    --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
    --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
    --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

# #============================#
# # TRIPLE
# #============================#
# WEIGHT_DECAY=1e-5
# ATTN_DROP=0.0
# PROJ_DROP=0.0

# MODEL_TYPE=triple
# NUM_LAYERS_KV_PROJ=-1; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; QK_DIM_RATIO=1.0 
# KERNEL=identity
# NORM_Q=true
# NORM_K=true
# USE_TRITON=false
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} --mixed_precision ${MIXED_PRECISION} \
#     --epochs ${EPOCHS} --steps ${STEPS} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} --schedule ${SCHEDULE} \
#     --learning_rate "${LEARNING_RATE}" --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --channel_dim ${CHANNEL_DIM} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} --rmsnorm ${RMSNORM} \
#     --model_type ${MODEL_TYPE} --kernel ${KERNEL} --norm_q ${NORM_Q} --norm_k ${NORM_K} --qk_dim_ratio ${QK_DIM_RATIO} --use_triton ${USE_TRITON} \
#     --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
#     --exp_name ${TASK}/${MODEL_TYPE}

#========================================================#
#========================================================#
# QUOTIENT BINARY RELATION COMPOSITION
#========================================================#
#========================================================#
TASK=quotient_binary_relation_composition

STEPS=0
EPOCHS=3000
BATCH_SIZE=2000
WEIGHT_DECAY=0e-5
LEARNING_RATE=1e-3
# SCHEDULE=ConstantLR
SCHEDULE=OneCycleLR
MIXED_PRECISION=false
RMSNORM=${MIXED_PRECISION}

NUM_BLOCKS=1
CHANNEL_DIM=16
NUM_HEADS=1
ATTN_DROP=0.3
PROJ_DROP=0.3

EMA=false
EMA_DECAY=0.999


# #============================#
# # TRANSFORMER
# #============================#
# WEIGHT_DECAY=1e-5
# ATTN_DROP=0.0
# PROJ_DROP=0.0

# MODEL_TYPE=transformer
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} \
#     --exp_name ${TASK}/trans

#============================#
# THIRD ORDER ATTENTION
#============================#
WEIGHT_DECAY=1e-5
ATTN_DROP=0.0
PROJ_DROP=0.0

# MODEL_TYPE=third_order
# THIRD_ORDER_METHOD=strassen
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
#     --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

MODEL_TYPE=third_order
THIRD_ORDER_METHOD=third_order
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
    --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
    --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
    --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
    --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

# #============================#
# # TRIPLE
# #============================#
# WEIGHT_DECAY=1e-5
# ATTN_DROP=0.0
# PROJ_DROP=0.0

# MODEL_TYPE=triple
# NUM_LAYERS_KV_PROJ=-1; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; QK_DIM_RATIO=1.0 
# KERNEL=identity
# NORM_Q=true
# NORM_K=true
# USE_TRITON=false
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} --mixed_precision ${MIXED_PRECISION} \
#     --epochs ${EPOCHS} --steps ${STEPS} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} --schedule ${SCHEDULE} \
#     --learning_rate "${LEARNING_RATE}" --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --channel_dim ${CHANNEL_DIM} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} --rmsnorm ${RMSNORM} \
#     --model_type ${MODEL_TYPE} --kernel ${KERNEL} --norm_q ${NORM_Q} --norm_k ${NORM_K} --qk_dim_ratio ${QK_DIM_RATIO} --use_triton ${USE_TRITON} \
#     --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
#     --exp_name ${TASK}/${MODEL_TYPE}

#========================================================#
#========================================================#
# MATCH 3
#========================================================#
#========================================================#
TASK=match3

STEPS=0
EPOCHS=500
BATCH_SIZE=2500
WEIGHT_DECAY=0e-5
LEARNING_RATE=1e-3
# SCHEDULE=ConstantLR
SCHEDULE=OneCycleLR
MIXED_PRECISION=false
RMSNORM=${MIXED_PRECISION}

NUM_BLOCKS=1
CHANNEL_DIM=128
NUM_HEADS=1
ATTN_DROP=0.4
PROJ_DROP=0.4

# EMA=true
# EMA_DECAY=0.999

EMA=false
EMA_DECAY=0.999

# #============================#
# # TRANSFORMER
# #============================#
# WEIGHT_DECAY=1e-5
# ATTN_DROP=0.0
# PROJ_DROP=0.0

# MODEL_TYPE=transformer
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu --master_port=12341 -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} \
#     --exp_name ${TASK}/trans

#============================#
# THIRD ORDER ATTENTION
#============================#
WEIGHT_DECAY=1e-5
ATTN_DROP=0.0
PROJ_DROP=0.0

# MODEL_TYPE=third_order
# THIRD_ORDER_METHOD=strassen
# MLP_RATIO=4.0
# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
#     --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
#     --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
#     --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

MODEL_TYPE=third_order
THIRD_ORDER_METHOD=third_order
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} --mixed_precision ${MIXED_PRECISION} \
    --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} --schedule ${SCHEDULE} --one_cycle_override_min_lr 1e-6 \
    --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} --pool ${POOL} \
    --model_type ${MODEL_TYPE} --mlp_ratio ${MLP_RATIO} --third_order_method ${THIRD_ORDER_METHOD} \
    --exp_name ${TASK}/${MODEL_TYPE}_${THIRD_ORDER_METHOD}

# #============================#
# # TRIPLE
# #============================#
# # WEIGHT_DECAY=1e-5
# # ATTN_DROP=0.0
# # PROJ_DROP=0.0

# MODEL_TYPE=triple
# NUM_LAYERS_KV_PROJ=-1; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; QK_DIM_RATIO=1.0 
# KERNEL=identity
# NORM_Q=true
# NORM_K=true
# USE_TRITON=false
# torchrun --nproc-per-node gpu --master_port=12342 -m lra --train true --task ${TASK} --mixed_precision ${MIXED_PRECISION} \
#     --epochs ${EPOCHS} --steps ${STEPS} --weight_decay ${WEIGHT_DECAY} --batch_size ${BATCH_SIZE} --schedule ${SCHEDULE} \
#     --learning_rate "${LEARNING_RATE}" --one_cycle_override_min_lr 1e-6 \
#     --attn_drop ${ATTN_DROP} --proj_drop ${PROJ_DROP} --ema ${EMA} --ema_decay ${EMA_DECAY} \
#     --channel_dim ${CHANNEL_DIM} --num_blocks ${NUM_BLOCKS} --num_heads ${NUM_HEADS} --rmsnorm ${RMSNORM} \
#     --model_type ${MODEL_TYPE} --kernel ${KERNEL} --norm_q ${NORM_Q} --norm_k ${NORM_K} --qk_dim_ratio ${QK_DIM_RATIO} --use_triton ${USE_TRITON} \
#     --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
#     --exp_name ${TASK}/${MODEL_TYPE}

#============================#
exit
#============================#
#