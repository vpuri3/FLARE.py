#
#============================#
# Setup
#============================#
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

if [ ! -f .venv/bin/activate ]; then
    echo "Missing ${REPO_ROOT}/.venv/bin/activate"
    echo "Create the repo-local env first with scripts/install.sh"
    exit 1
fi

source .venv/bin/activate

has_transformers() {
    python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("transformers") is not None else 1)
PY
}

EXTERNAL_COMPILE=false
EXTERNAL_STATIC_GRAPH=false

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
# TRANSOLVER
#============================#
TRANSOLVER_LR=1e-3
TRANSOLVER_WEIGHT_DECAY=1e-5
TRANSOLVER_NUM_SLICES=128
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${TRANSOLVER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${TRANSOLVER_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transolver \
    --num_slices ${TRANSOLVER_NUM_SLICES} --mlp_ratio ${MLP_RATIO} --pos_embed abs \
    --exp_name ${TASK}/transolver_tuned
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
# COSFORMER
#============================#
COSFORMER_LR=1e-4
COSFORMER_MLP_RATIO=2.0
COSFORMER_ATTN_DROP=0.05
COSFORMER_PROJ_DROP=0.05
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${COSFORMER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type cosformer \
    --mlp_ratio ${COSFORMER_MLP_RATIO} \
    --attn_drop ${COSFORMER_ATTN_DROP} --proj_drop ${COSFORMER_PROJ_DROP} \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/cosformer

#============================#
# HEDGEHOG
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate 1e-4 --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --kernel hedgehog --q_norm true --k_norm true \
    --mlp_ratio ${MLP_RATIO} \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/hedgehog

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
# LINFORMER (SHARED KV)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} --linformer_share_kv true \
    --exp_name ${TASK}/linformer_shared

#============================#
# FLARE
#============================#
NUM_HEADS=8
NUM_LATENTS=64
NUM_LAYERS_KV_PROJ=-1
NUM_LAYERS_FFN=0
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=4.0
FLARE_LR=2e-4
FLARE_WEIGHT_DECAY=3e-5
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${FLARE_LR} --batch_size ${BATCH_SIZE} --weight_decay ${FLARE_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --q_norm true --k_norm true \
    --attn_drop 0.05 \
    --num_workers 4 --prefetch_factor 2 \
    --rmsnorm true \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/flare

#============================#
# FLAREPP
#============================#
NUM_HEADS=8
NUM_LATENTS=64
NUM_LAYERS_FFN=0
FFN_MLP_RATIO=4.0
FLARE_LR=2e-4
FLARE_WEIGHT_DECAY=3e-5
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${FLARE_LR} --batch_size ${BATCH_SIZE} --weight_decay ${FLARE_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flarepp \
    --num_latents ${NUM_LATENTS} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
    --attn_drop 0.05 \
    --num_workers 4 --prefetch_factor 2 \
    --rmsnorm true \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/flarepp

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
# PERFORMER (FAVOR++)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate 4.5e-4 --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_feature_map favor_pp \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio 2.0 --attn_drop 0.05 --proj_drop 0.05 --cls_drop 0.05 --emb_drop 0.0 \
    --q_norm true --k_norm true --rmsnorm true \
    --num_workers 4 --prefetch_factor 2 \
    --mixed_precision false \
    --compile_model true --static_graph true \
    --seed 2 \
    --exp_name ${TASK}/performer_favorpp

#============================#
# NORMATTENTION
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type normattention \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/normattention

#============================#
# FUNNEL / REFORMER (HF)
#============================#
if has_transformers; then
    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type funnel_hf \
        --mlp_ratio ${MLP_RATIO} \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/funnel_hf

    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type reformer_hf \
        --mlp_ratio ${MLP_RATIO} \
        --reformer_num_hashes 2 \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/reformer_hf
else
    echo "Skipping ${TASK}/funnel_hf and ${TASK}/reformer_hf: transformers is not installed"
fi

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
# TRANSOLVER
#============================#
TRANSOLVER_LR=5e-4
TRANSOLVER_WEIGHT_DECAY=1e-1
TRANSOLVER_EMB_DROP=0.05
TRANSOLVER_NUM_SLICES=64
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${TRANSOLVER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${TRANSOLVER_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transolver \
    --num_slices ${TRANSOLVER_NUM_SLICES} --mlp_ratio ${MLP_RATIO} --emb_drop ${TRANSOLVER_EMB_DROP} --pos_embed abs \
    --exp_name ${TASK}/transolver_tuned

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
# COSFORMER
#============================#
COSFORMER_LR=3e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${COSFORMER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type cosformer \
    --mlp_ratio ${MLP_RATIO} \
    --attn_drop 0.05 --proj_drop 0.05 \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/cosformer

#============================#
# HEDGEHOG
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate 5e-4 --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --kernel hedgehog --q_norm true --k_norm true \
    --mlp_ratio ${MLP_RATIO} \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/hedgehog

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
# LINFORMER (SHARED KV)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} --linformer_share_kv true \
    --exp_name ${TASK}/linformer_shared

#============================#
# FLARE
#============================#
NUM_HEADS=8
NUM_LATENTS=192
NUM_LAYERS_KV_PROJ=-1
NUM_LAYERS_FFN=0
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=2.0
FLARE_LR=1.5e-3
FLARE_WEIGHT_DECAY=6e-2
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${FLARE_LR} --batch_size ${BATCH_SIZE} --weight_decay ${FLARE_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --q_norm true --k_norm true \
    --emb_drop 0.08 --cls_drop 0.08 --attn_drop 0.08 --proj_drop 0.08 \
    --num_workers 8 --prefetch_factor 4 \
    --rmsnorm true \
    --mixed_precision false \
    --compile_model true --static_graph true \
    --seed 6 \
    --exp_name ${TASK}/flare

#============================#
# FLAREPP
#============================#
NUM_HEADS=8
NUM_LATENTS=192
NUM_LAYERS_FFN=0
FFN_MLP_RATIO=2.0
FLARE_LR=1.5e-3
FLARE_WEIGHT_DECAY=6e-2
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${FLARE_LR} --batch_size ${BATCH_SIZE} --weight_decay ${FLARE_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flarepp \
    --num_latents ${NUM_LATENTS} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
    --emb_drop 0.08 --cls_drop 0.08 --attn_drop 0.08 --proj_drop 0.08 \
    --num_workers 8 --prefetch_factor 4 \
    --rmsnorm true \
    --mixed_precision false \
    --compile_model true --static_graph true \
    --seed 6 \
    --exp_name ${TASK}/flarepp

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
# PERFORMER (FAVOR++)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_feature_map favor_pp \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.1 --proj_drop 0.1 \
    --exp_name ${TASK}/performer_favorpp

#============================#
# NORMATTENTION
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type normattention \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --mlp_ratio ${MLP_RATIO} \
    --attn_drop 0.1 --proj_drop 0.1 \
    --exp_name ${TASK}/normattention

#============================#
# FUNNEL / REFORMER (HF)
#============================#
if has_transformers; then
    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type funnel_hf \
        --mlp_ratio ${MLP_RATIO} \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/funnel_hf

    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type reformer_hf \
        --mlp_ratio ${MLP_RATIO} \
        --reformer_num_hashes 2 \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/reformer_hf
else
    echo "Skipping ${TASK}/funnel_hf and ${TASK}/reformer_hf: transformers is not installed"
fi

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
# TRANSOLVER
#============================#
TRANSOLVER_LR=8e-4
TRANSOLVER_WEIGHT_DECAY=1e-4
TRANSOLVER_NUM_SLICES=128
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${TRANSOLVER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${TRANSOLVER_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transolver \
    --num_slices ${TRANSOLVER_NUM_SLICES} --mlp_ratio ${MLP_RATIO} --pos_embed abs \
    --exp_name ${TASK}/transolver_tuned

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
# COSFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type cosformer \
    --mlp_ratio ${MLP_RATIO} \
    --cls_drop 0.05 --emb_drop 0.05 --proj_drop 0.05 \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/cosformer

#============================#
# HEDGEHOG
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate 1e-4 --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --kernel hedgehog --q_norm true --k_norm true \
    --mlp_ratio ${MLP_RATIO} \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/hedgehog

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
# LINFORMER (SHARED KV)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LINEAR_LR} --batch_size ${BATCH_SIZE} --weight_decay ${LINEAR_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} --linformer_share_kv true \
    --exp_name ${TASK}/linformer_shared

#============================#
# FLARE
#============================#
NUM_HEADS=8
NUM_LATENTS=128
NUM_LAYERS_KV_PROJ=-1
NUM_LAYERS_FFN=0
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --q_norm true --k_norm true \
    --num_workers 8 --prefetch_factor 4 \
    --rmsnorm true \
    --compile_model true --static_graph true \
    --exp_name ${TASK}/flare

#============================#
# FLAREPP
#============================#
NUM_HEADS=8
NUM_LATENTS=128
NUM_LAYERS_FFN=0
FFN_MLP_RATIO=4.0
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flarepp \
    --num_latents ${NUM_LATENTS} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
    --num_workers 8 --prefetch_factor 4 \
    --rmsnorm true \
    --compile_model true --static_graph true \
    --exp_name ${TASK}/flarepp

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
# PERFORMER (FAVOR++)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_feature_map favor_pp \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.0 --proj_drop 0.0 \
    --exp_name ${TASK}/performer_favorpp

#============================#
# NORMATTENTION
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type normattention \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --mlp_ratio ${MLP_RATIO} \
    --exp_name ${TASK}/normattention

#============================#
# FUNNEL / REFORMER (HF)
#============================#
if has_transformers; then
    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate ${LR} --batch_size 16 --weight_decay ${WEIGHT_DECAY} \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type funnel_hf \
        --mlp_ratio ${MLP_RATIO} \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/funnel_hf

    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type reformer_hf \
        --mlp_ratio ${MLP_RATIO} \
        --reformer_num_hashes 2 \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/reformer_hf
else
    echo "Skipping ${TASK}/funnel_hf and ${TASK}/reformer_hf: transformers is not installed"
fi


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

#============================#
# TRANSFORMER
#============================#
WEIGHT_DECAY=1e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transformer \
    --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/trans

#============================#
# TRANSOLVER
#============================#
TRANSOLVER_LR=2e-5
TRANSOLVER_WEIGHT_DECAY=1e-4
TRANSOLVER_NUM_SLICES=64
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${TRANSOLVER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${TRANSOLVER_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transolver \
    --num_slices ${TRANSOLVER_NUM_SLICES} --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed abs \
    --exp_name ${TASK}/transolver_tuned

#============================#
# LINEAR
#============================#
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
    --kernel elu --q_norm true --k_norm true \
    --exp_name ${TASK}/linear

#============================#
# COSFORMER
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate 1e-5 --batch_size ${BATCH_SIZE} --weight_decay 1e-4 \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type cosformer \
    --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed ${POS_EMBED} \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/cosformer

#============================#
# HEDGEHOG
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate 1e-5 --batch_size ${BATCH_SIZE} --weight_decay 1e-4 \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linear \
    --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed ${LINEAR_POS_EMBED} \
    --kernel hedgehog --q_norm true --k_norm true \
    --mixed_precision false \
    --compile_model false --static_graph false \
    --exp_name ${TASK}/hedgehog

#============================#
# LINFORMER
#============================#
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

#============================#
# LINFORMER (SHARED KV)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type linformer \
    --mlp_ratio ${MLP_RATIO} --linformer_k ${LINFORMER_K} --linformer_share_kv true --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/linformer_shared

#============================#
# FLARE
#============================#
NUM_LATENTS=128
NUM_LAYERS_KV_PROJ=-1
NUM_LAYERS_FFN=0
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=4.0
FLARE_STEPS=40_000
FLARE_LR=1.4e-5
FLARE_WEIGHT_DECAY=3e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${FLARE_STEPS} \
    --learning_rate ${FLARE_LR} --batch_size ${BATCH_SIZE} --weight_decay ${FLARE_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --q_norm true --k_norm true \
    --emb_drop 0.08 --cls_drop 0.08 --attn_drop 0.08 --proj_drop 0.02 \
    --num_workers 8 --prefetch_factor 4 \
    --rmsnorm true \
    --compile_model true --static_graph true \
    --one_cycle_pct_start 0.2 \
    --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/flare

#============================#
# FLAREPP
#============================#
NUM_LATENTS=128
NUM_LAYERS_FFN=0
FFN_MLP_RATIO=4.0
FLARE_STEPS=40_000
FLARE_LR=1.4e-5
FLARE_WEIGHT_DECAY=3e-4
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${FLARE_STEPS} \
    --learning_rate ${FLARE_LR} --batch_size ${BATCH_SIZE} --weight_decay ${FLARE_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flarepp \
    --num_latents ${NUM_LATENTS} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
    --emb_drop 0.08 --cls_drop 0.08 --attn_drop 0.08 --proj_drop 0.02 \
    --num_workers 8 --prefetch_factor 4 \
    --rmsnorm true \
    --compile_model true --static_graph true \
    --one_cycle_pct_start 0.2 \
    --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/flarepp

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
# PERFORMER (FAVOR++)
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type performer \
    --performer_nb_features 256 \
    --performer_feature_map favor_pp \
    --performer_redraw_interval 0 \
    --performer_normalize_inputs true \
    --mlp_ratio ${MLP_RATIO} --attn_drop 0.0 --proj_drop 0.0 \
    --pool ${POOL} --pos_embed ${POS_EMBED} \
    --exp_name ${TASK}/performer_favorpp

#============================#
# NORMATTENTION
#============================#
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type normattention \
    --num_layers_kv_proj -1 \
    --num_layers_ffn 0 \
    --ffn_mlp_ratio 4.0 \
    --qk_dim_ratio 1.0 \
    --mlp_ratio ${MLP_RATIO} \
    --pool ${POOL} \
    --exp_name ${TASK}/normattention

#============================#
# FUNNEL / REFORMER (HF)
#============================#
if has_transformers; then
    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate 1e-5 --batch_size ${BATCH_SIZE} --weight_decay 1e-4 \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type funnel_hf \
        --mlp_ratio ${MLP_RATIO} --pool ${POOL} --pos_embed ${POS_EMBED} \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/funnel_hf

    torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
        --epochs ${EPOCHS} --steps ${STEPS} \
        --learning_rate 1e-5 --batch_size ${BATCH_SIZE} --weight_decay 1e-4 \
        --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
        --model_type reformer_hf \
        --mlp_ratio ${MLP_RATIO} --reformer_num_hashes 2 --pool ${POOL} --pos_embed ${POS_EMBED} \
        --compile_model ${EXTERNAL_COMPILE} --static_graph ${EXTERNAL_STATIC_GRAPH} \
        --exp_name ${TASK}/reformer_hf
else
    echo "Skipping ${TASK}/funnel_hf and ${TASK}/reformer_hf: transformers is not installed"
fi

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

# Pathfinder runs are intentionally disabled in this script.
: <<'PATHFINDER_DISABLED'
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

#============================#
# TRANSOLVER (BEST STABLE PATHFINDER CHECKPOINT)
#============================#
TRANSOLVER_LR=1e-3
TRANSOLVER_WEIGHT_DECAY=1.5e-4
TRANSOLVER_NUM_SLICES=64
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate ${TRANSOLVER_LR} --batch_size ${BATCH_SIZE} --weight_decay ${TRANSOLVER_WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type transolver \
    --num_slices ${TRANSOLVER_NUM_SLICES} --mlp_ratio ${MLP_RATIO} --pool cls --pos_embed abs \
    --clip_grad_norm 1.0 \
    --rmsnorm false \
    --ema false \
    --emb_drop 0.0 \
    --exp_name ${TASK}/transolver_pf_s64_lr1e3_cls

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
    --kernel elu --q_norm true --k_norm true \
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
NUM_LAYERS_KV_PROJ=-1
NUM_LAYERS_FFN=0
KV_PROJ_MLP_RATIO=1.0
FFN_MLP_RATIO=${MLP_RATIO}
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate 4.5e-4 --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flare --mlp_ratio ${MLP_RATIO} \
    --num_latents ${NUM_LATENTS} --num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} --num_layers_ffn ${NUM_LAYERS_FFN} \
    --kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} --ffn_mlp_ratio ${FFN_MLP_RATIO} --attn_scale one \
    --q_norm true --k_norm true \
    --num_workers 0 \
    --attn_drop 0.11 --emb_drop 0.11 --proj_drop 0.11 \
    --mixed_precision false \
    --clip_grad_norm 0.5 \
    --ema true --ema_decay 0.999 \
    --rmsnorm false \
    --exp_name ${TASK}/flare

#============================#
# FLAREPP
#============================#
LR=3e-4
BATCH_SIZE=32
WEIGHT_DECAY=1e-4
MLP_RATIO=1.0
NUM_LATENTS=128
NUM_LAYERS_FFN=0
FFN_MLP_RATIO=${MLP_RATIO}
torchrun --nproc-per-node gpu -m lra --train true --task ${TASK} \
    --epochs ${EPOCHS} --steps ${STEPS} \
    --schedule OneCycleLR \
    --learning_rate ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
    --num_blocks ${NUM_BLOCKS} --channel_dim ${CHANNEL_DIM} --num_heads ${NUM_HEADS} \
    --model_type flarepp --mlp_ratio ${MLP_RATIO} \
    --num_latents ${NUM_LATENTS} --num_layers_ffn ${NUM_LAYERS_FFN} --ffn_mlp_ratio ${FFN_MLP_RATIO} \
    --num_workers 0 \
    --attn_drop 0.11 --emb_drop 0.11 --proj_drop 0.11 \
    --mixed_precision false \
    --clip_grad_norm 0.5 \
    --ema true --ema_decay 0.999 \
    --rmsnorm false \
    --exp_name ${TASK}/flarepp

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
    --q_norm true \
    --k_norm true \
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
PATHFINDER_DISABLED
