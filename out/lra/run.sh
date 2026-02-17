#
#============================#
# Setup
#============================#
source ~/.bash_profile
cd /project/community/$(whoami)/FLARE-dev.py
source .venv/bin/activate

# Retrieval B=4, C=128, H=4, MLP ratio=4.0, BS=32, Steps=5K, LR=5e-1, WD=1e-4
#   FLARE: WD: 1e-4, M: 128, H: 8, Scale: 1.0
# Image: B=4, C=128, H=8, MLP ratio=4.0, BS=32, Steps=10K, LR=5e-4, WD=1e-1

#----------------------------------------------------------------------------|#
# | Task          | B |  C  | H | MLP ratio | BS | Steps/Epochs |   LR, WD   |
# ----------------|---|-----|---|-----------|----|--------------|------------|
# | listops       | 6 | 512 | 8 |    4.0    | 32 |  10K steps   | 5e-4, 1e-4 |
# | text          | 6 | 512 | 8 |    4.0    | 32 |  20K steps   | 5e-2, 1e-1 |
# | retrieval     | 4 | 128 | 4 |    4.0    | 32 |  5K steps    | 5e-1, _e-_ |
# | image         | 3 |  64 | 4 |    1.0    | 32 |  200 epochs  | 1e-2, 1e-1 |
# | pathfinder32  | 4 | 128 | 8 |    1.0    | 32 |  200 epochs  | 1e-2, _e-_ |
# | pathfinder128 | 4 | 128 | 8 |    1.0    | 32 |  200 epochs  | 1e-2, _e-_ |
#----------------------------------------------------------------------------|#

#========================================================#
#========================================================#
# LISTOPS
#========================================================#
#========================================================#
TASK=listops

EPOCHS=0
STEPS=10_000
LR=5e-4
BATCH_SIZE=32
WEIGHT_DECAY=1e-5

NUM_BLOCKS=4
CHANNEL_DIM=128
NUM_HEADS=8

#============================#
# TRANSFORMER
#============================#
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transformer \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/trans
#============================#
# LINEAR ATTENTION
#============================#
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/linear

#============================#
# LINFORMER
#============================#
MLP_RATIO=4.0
LINFORMER_K=128
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} \
    --exp_name ${TASK}/linformer

#============================#
# FLARE
#============================#
NUM_HEADS=8
NUM_LATENTS=128
NUM_LAYERS_KV_PROJ=3
NUM_LAYERS_FFN=3
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=1.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --exp_name ${TASK}/flare

#============================#
# PERFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.0 --proj_drop 0.0 \
    --exp_name ${TASK}/performer

#============================#
# MLA (norm attention)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type multilinear \
    --num_states 1 \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --kernel identity \
    --norm_q true \
    --norm_k true \
    --attn_scale one \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/mla_ns1

# #============================#
# exit
# #============================#

#========================================================#
#========================================================#
# IMAGE
#========================================================#
#========================================================#
TASK=image

EPOCHS=0
STEPS=20_000
LR=1e-3
BATCH_SIZE=32
WEIGHT_DECAY=5e-2

NUM_BLOCKS=3
CHANNEL_DIM=64
NUM_HEADS=4

#============================#
# TRANSFORMER
#============================#
MLP_RATIO=2.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transformer \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/trans

#============================#
# LINEAR ATTENTION
#============================#
MLP_RATIO=2.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/linear

#============================#
# LINFORMER
#============================#
MLP_RATIO=2.0
LINFORMER_K=256
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} \
    --exp_name ${TASK}/linformer

#============================#
# FLARE
#============================#
NUM_HEADS=8
NUM_LATENTS=256
NUM_LAYERS_KV_PROJ=2
NUM_LAYERS_FFN=1
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=1.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --exp_name ${TASK}/flare

#============================#
# PERFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.1 --proj_drop 0.1 \
    --exp_name ${TASK}/performer

#============================#
# MLA (norm attention)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type multilinear \
    --num_states 1 \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --kernel identity \
    --norm_q true \
    --norm_k true \
    --attn_scale one \
    --mlp_ratio ${MLP_RATIO} \
    --attn_drop 0.1 --proj_drop 0.1 \
    --exp_name ${TASK}/mla_ns1

#========================================================#
#========================================================#
# RETRIEVAL
#========================================================#
#========================================================#
TASK=retrieval

EPOCHS=0
STEPS=10_000
LR=5e-4
BATCH_SIZE=32
WEIGHT_DECAY=1e-4

NUM_BLOCKS=4
CHANNEL_DIM=128
NUM_HEADS=4

#============================#
# TRANSFORMER
#============================#
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transformer \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/trans

#============================#
# LINEAR ATTENTION
#============================#
MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/linear

#============================#
# LINFORMER
#============================#
LINEAR_LR=4.2e-5
LINEAR_WEIGHT_DECAY=1e-6
MLP_RATIO=4.0
LINFORMER_K=128
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LINEAR_LR} --batch_size ${BATCH_SIZE} --weight_decay ${LINEAR_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} \
    --exp_name ${TASK}/linformer

#============================#
# FLARE
#============================#
NUM_HEADS=8
NUM_LATENTS=128
NUM_LAYERS_KV_PROJ=3
NUM_LAYERS_FFN=3
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=1.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --exp_name ${TASK}/flare

#============================#
# PERFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.0 --proj_drop 0.0 \
    --exp_name ${TASK}/performer

#============================#
# MLA (norm attention)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type multilinear \
    --num_states 1 \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --kernel identity \
    --norm_q true \
    --norm_k true \
    --attn_scale one \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/mla_ns1


#========================================================#
#========================================================#
# TEXT - CLS TOKEN IMPLEMENTATION
#========================================================#
#========================================================#
TASK=text

EPOCHS=0
STEPS=20_000
LR=1e-5
BATCH_SIZE=32


NUM_BLOCKS=4
CHANNEL_DIM=128
NUM_HEADS=8
MLP_RATIO=4.0
POOL=cls
POS_EMBED=rope

============================#
TRANSFORMER
============================#
WEIGHT_DECAY=1e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transformer \
    --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/trans

============================#
LINEAR
============================#
LINEAR_POS_EMBED=abs
LINEAR_KERNEL=identity
LINEAR_LR=1e-5
WEIGHT_DECAY=1e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LINEAR_LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed ${LINEAR_POS_EMBED} \
    --kernel elu --norm_q true --norm_k true \
    --exp_name ${TASK}/linear

============================#
LINFORMER
============================#
LINFORMER_K=128
LR=5e-7
WEIGHT_DECAY=5e-3      

torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/linformer

============================#
FLARE
============================#
NUM_LATENTS=128
NUM_LAYERS_KV_PROJ=3
NUM_LAYERS_FFN=3
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=1.0
WEIGHT_DECAY=1e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale sqrt \
    --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/flare

#============================#
# PERFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.0 --proj_drop 0.0 \
    --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/performer

#============================#
# MLA (norm attention)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type multilinear \
    --num_states 1 \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --kernel identity \
    --norm_q true \
    --norm_k true \
    --attn_scale one \
    --mlp_ratio ${MLP_RATIO} \
    --pool ${POOL} \
    --exp_name ${TASK}/mla_ns1

# #============================#
# # MLA
# #============================#
# NUM_HEADS=8
# NUM_LAYERS_KV_PROJ=3
# NUM_LAYERS_FFN=3       # MLA doesn't use below
# KV_PROJ_MLP_RATIO=1.0
# FFN_MLP_RATIO=1.0

# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} \
#     --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
#     --model_type mla_1 \
#     --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
#     --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
#     --exp_name ${TASK}/mla_1

# torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
#     --epochs ${EPOCHS} --steps ${STEPS} \
#     --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
#     --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
#     --model_type mla_2 \
#     --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
#     --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
#     --exp_name ${TASK}/mla_2

#========================================================#
#========================================================#
# PATHFINDER32
#========================================================#
#========================================================#
TASK=pathfinder32

STEPS=0
EPOCHS=200
NUM_BLOCKS=4
CHANNEL_DIM=128
NUM_HEADS=8

#============================#
# TRANSFORMER
#============================#

LR=6e-4
BATCH_SIZE=64
WEIGHT_DECAY=1.5e-4
MLP_RATIO=1.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule ConstantLR \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transformer \
    --mlp_ratio ${MLP_RATIO} \
    --clip_grad_norm 1.0 \
    --rmsnorm true \
    --ema true --ema_decay 0.999 \
    --emb_drop 0.05 --attn_drop 0.05 --proj_drop 0.05 \
    --exp_name ${TASK}/trans

# #============================#
# # LINEAR ATTENTION
# #============================#

LR=5e-4
BATCH_SIZE=32
WEIGHT_DECAY=1e-4
MLP_RATIO=1.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --mlp_ratio ${MLP_RATIO} \
    --num_workers 0 \
    --kernel elu --norm_q true --norm_k true \
    --clip_grad_norm 0.5 \
    --rmsnorm true \
    --mixed_precision false \
    --exp_name ${TASK}/linear

# #============================#
# # LINFORMER
# #============================#

LR=1e-3
BATCH_SIZE=64
WEIGHT_DECAY=1e-4
MLP_RATIO=1.0
LINFORMER_K=256
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule ConstantLR \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} \
    --clip_grad_norm 1.0 \
    --rmsnorm true \
    --ema true --ema_decay 0.999 \
    --exp_name ${TASK}/linformer


# #============================#
# # FLARE
# #============================#
LR=5e-4
BATCH_SIZE=32
WEIGHT_DECAY=5e-4
MLP_RATIO=1.0
NUM_LATENTS=128
NUM_LAYERS_KV_PROJ=3
NUM_LAYERS_FFN=3
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=1.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare --mlp_ratio ${MLP_RATIO} \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --num_workers 0 \
    --attn_drop 0.1 --emb_drop 0.1 --proj_drop 0.1 \
    --mixed_precision false \
    --clip_grad_norm 0.5 \
    --rmsnorm false \
    --exp_name ${TASK}/flare

#============================#
# PERFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate 5e-4 --batch_size 32 --weight_decay 5e-4 \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio 1.0 --attn_drop 0.1 --proj_drop 0.1 \
    --clip_grad_norm 0.5 --mixed_precision false \
    --exp_name ${TASK}/performer

#============================#
# MLA (norm attention)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate 1e-5 --batch_size 32 --weight_decay 1e-5 \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type multilinear \
    --num_states 1 \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --kernel identity \
    --norm_q true \
    --norm_k true \
    --attn_scale one \
    --mlp_ratio 1.0 \
    --clip_grad_norm 0.5 \
    --mixed_precision false \
    --attn_drop 0.1 --proj_drop 0.1 --emb_drop 0.1 \
    --exp_name ${TASK}/mla_ns1

#============================#
exit
#============================#

#
