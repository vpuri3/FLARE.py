#
#============================#
# GPU job tracking setup
#============================#
cd /project/community/vedantpu/FLARE-dev.py
source .venv/bin/activate

# GPU configuration
GPU_LIST=(0 1 2 3)  # 4 GPUs
MAX_JOBS_PER_GPU=2  # Maximum concurrent jobs per GPU
GPU_TRACK_DIR="/tmp/gpu_job_tracking_$$"  # Temporary directory for tracking
mkdir -p "$GPU_TRACK_DIR"

# Cleanup function
cleanup() {
    rm -rf "$GPU_TRACK_DIR"
}
trap cleanup EXIT

# Function to count jobs on a GPU
count_gpu_jobs() {
    local gpu_id=$1
    local count=0
    local temp_file="$GPU_TRACK_DIR/gpu_${gpu_id}.jobs.tmp"
    
    if [ -f "$GPU_TRACK_DIR/gpu_${gpu_id}.jobs" ]; then
        # Count non-zero PIDs (processes that are still running) and clean up dead ones
        > "$temp_file"
        while IFS= read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                count=$((count + 1))
                echo "$pid" >> "$temp_file"
            fi
        done < "$GPU_TRACK_DIR/gpu_${gpu_id}.jobs"
        mv "$temp_file" "$GPU_TRACK_DIR/gpu_${gpu_id}.jobs" 2>/dev/null || true
    fi
    echo $count
}

# Function to get next available GPU
get_next_gpu() {
    local min_jobs=999
    local selected_gpu=${GPU_LIST[0]}
    
    for gpu in "${GPU_LIST[@]}"; do
        local job_count=$(count_gpu_jobs $gpu)
        if [ $job_count -lt $min_jobs ] && [ $job_count -lt $MAX_JOBS_PER_GPU ]; then
            min_jobs=$job_count
            selected_gpu=$gpu
        fi
    done
    
    echo $selected_gpu
}

# Function to wait for GPU availability
wait_for_gpu() {
    while true; do
        for gpu in "${GPU_LIST[@]}"; do
            local job_count=$(count_gpu_jobs $gpu)
            if [ $job_count -lt $MAX_JOBS_PER_GPU ]; then
                return 0
            fi
        done
        sleep 1m  # Wait 1 minute before checking again
    done
}

# Function to run job on specific GPU
run_on_gpu() {
    local gpu_id=$1
    local exp_name=$2
    shift 2
    
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "[GPU $gpu_id] Starting: $exp_name"
        "$@"
        echo "[GPU $gpu_id] Finished: $exp_name"
    ) &
    
    local pid=$!
    # Track the PID in the GPU's job file
    echo "$pid" >> "$GPU_TRACK_DIR/gpu_${gpu_id}.jobs"
}

#======================================================================#
# Elasticity/Pipe/Airfoil/Darcy
#======================================================================#

# training hyperparameters
EPOCH=500
BATCH_SIZE=2
WEIGHT_DECAY=1e-5
LEARNING_RATE=1e-3
MIXED_PRECISION=false
RMSNORM=${MIXED_PRECISION}

EMA=true
EMA_DECAY=0.999

# model hyperparameters
NUM_HEADS=4
CHANNEL_DIM=128
NUM_BLOCKS_LIST=(1 2 4 6 8 12 16 20 24 32 40)

#==============================#
for DATASET in elasticity; do
for ((i=0; i<${#NUM_BLOCKS_LIST[@]}; i++)); do
#==============================#
NUM_BLOCKS=${NUM_BLOCKS_LIST[$i]}

# #======================================================================#
# # Vanilla Transformer
# #======================================================================#
# MODEL=transformer
# WEIGHT_DECAY=1e-5
# EXP_NAME=loopy_${DATASET}/${MODEL}_C_${CHANNEL_DIM}_H_${NUM_HEADS}/B_${NUM_BLOCKS}
# #
# wait_for_gpu
# GPU=$(get_next_gpu)
# 
# if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
#     if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
#         run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
#     fi
# else
#     run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
#         --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
#         --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
#         --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
#         --run.exp_name ${EXP_NAME}
# fi

# #======================================================================#
# # FLARE (STANDARD KVPROJ, STANDARD FFN)
# #======================================================================#
# MODEL=flare
# WEIGHT_DECAY=1e-5
# ATTN_SCALE=sqrt
# NUM_LATENTS=256
# NUM_LAYERS_KV_PROJ=-1; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; KV_PROJ_MLP_RATIO=1.0

# EXP_NAME=loopy_${DATASET}/${MODEL}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
# EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
# EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
# EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}
# #
# wait_for_gpu
# GPU=$(get_next_gpu)

# if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
#     if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
#         run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
#     fi
# else
#     run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
#         --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
#         --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
#         --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
#         --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
#         --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
#         --run.exp_name ${EXP_NAME}
# fi

# #======================================================================#
# # FLARE (AS PROPOSED: DEEP KVPROJ, DEEP FFN)
# #======================================================================#
# MODEL=flare
# WEIGHT_DECAY=1e-5
# ATTN_SCALE=sqrt
# NUM_LATENTS=256
# NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=3; FFN_MLP_RATIO=1.0; KV_PROJ_MLP_RATIO=1.0 

# EXP_NAME=loopy_${DATASET}/${MODEL}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
# EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
# EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
# EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}
# #
# wait_for_gpu
# GPU=$(get_next_gpu)

# if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
#     if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
#         run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
#     fi
# else
#     run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
#         --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
#         --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
#         --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
#         --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
#         --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
#         --run.exp_name ${EXP_NAME}
# fi

# #======================================================================#
# # FLARE (DEEP KVPROJ, STANDARD FFN)
# #======================================================================#
# MODEL=flare
# WEIGHT_DECAY=1e-5
# ATTN_SCALE=sqrt
# NUM_LATENTS=256
# NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; KV_PROJ_MLP_RATIO=1.0 

# EXP_NAME=loopy_${DATASET}/${MODEL}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
# EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
# EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
# EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}
#
# wait_for_gpu
# GPU=$(get_next_gpu)

# if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
#     if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
#         run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
#     fi
# else
#     run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
#         --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
#         --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
#         --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
#         --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
#         --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
#         --run.exp_name ${EXP_NAME}
# fi

#======================================================================#
# UNLOOPY (DEEP KVPROJ, DEEP FFN) SHARED ATT
#======================================================================#
MODEL=unloopy
WEIGHT_DECAY=1e-5
ATTN_SCALE=sqrt
NUM_LATENTS=256
NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=3; FFN_MLP_RATIO=1.0; KV_PROJ_MLP_RATIO=1.0 

SHARED_FFN=false; SHARED_ATT=true;
NAME=flare_tied_att

EXP_NAME=loopy_${DATASET}/${NAME}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}
#
wait_for_gpu
GPU=$(get_next_gpu)

if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
    if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
        run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
    fi
else
    run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
        --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
        --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
        --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
        --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
        --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
        --model.shared_ffn ${SHARED_FFN} --model.shared_att ${SHARED_ATT} \
        --run.exp_name ${EXP_NAME}
fi

#======================================================================#
# UNLOOPY (DEEP KVPROJ, STANDARD FFN) SHARED ATT
#======================================================================#
MODEL=unloopy
WEIGHT_DECAY=1e-5
ATTN_SCALE=sqrt
NUM_LATENTS=256
NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; KV_PROJ_MLP_RATIO=1.0 

SHARED_FFN=false; SHARED_ATT=true;
NAME=flare_tied_att

EXP_NAME=loopy_${DATASET}/${NAME}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}

wait_for_gpu
GPU=$(get_next_gpu)

if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
    if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
        run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
    fi
else
    run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
        --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
        --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
        --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
        --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
        --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
        --model.shared_ffn ${SHARED_FFN} --model.shared_att ${SHARED_ATT} \
        --run.exp_name ${EXP_NAME}
fi

#======================================================================#
# UNLOOPY (DEEP KVPROJ, DEEP FFN) SHARED FFN
#======================================================================#
MODEL=unloopy
WEIGHT_DECAY=1e-5
ATTN_SCALE=sqrt
NUM_LATENTS=256
NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=3; FFN_MLP_RATIO=1.0; KV_PROJ_MLP_RATIO=1.0 

SHARED_FFN=true; SHARED_ATT=false;
NAME=flare_tied_ffn

EXP_NAME=loopy_${DATASET}/${NAME}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}
#
wait_for_gpu
GPU=$(get_next_gpu)

if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
    if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
        run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
    fi
else
    run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
        --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
        --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
        --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
        --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
        --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
        --model.shared_ffn ${SHARED_FFN} --model.shared_att ${SHARED_ATT} \
        --run.exp_name ${EXP_NAME}
fi

#======================================================================#
# UNLOOPY (DEEP KVPROJ, STANDARD FFN) SHARED FFN
#======================================================================#
MODEL=unloopy
WEIGHT_DECAY=1e-5
ATTN_SCALE=sqrt
NUM_LATENTS=256
NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; KV_PROJ_MLP_RATIO=1.0 

SHARED_FFN=true; SHARED_ATT=false;
NAME=flare_tied_ffn

EXP_NAME=loopy_${DATASET}/${NAME}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}

wait_for_gpu
GPU=$(get_next_gpu)

if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
    if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
        run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
    fi
else
    run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
        --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
        --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
        --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
        --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
        --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
        --model.shared_ffn ${SHARED_FFN} --model.shared_att ${SHARED_ATT} \
        --run.exp_name ${EXP_NAME}
fi

# #======================================================================#
# # LOOPY FLARE
# #======================================================================#
# NUM_BLOCKS_EFF=$NUM_BLOCKS
# NUM_BLOCKS=4
# if [ $((NUM_BLOCKS_EFF % NUM_BLOCKS)) -ne 0 ]; then
#     continue
# fi
# NUM_PASSES=$((NUM_BLOCKS_EFF / NUM_BLOCKS))

# #======================================================================#
# # LOOPY FLARE (DEEP KVPROJ, DEEP FFN)
# #======================================================================#
# MODEL=loopy
# WEIGHT_DECAY=1e-5
# ATTN_SCALE=sqrt
# NUM_LATENTS=256
# NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=3; FFN_MLP_RATIO=1.0; KV_PROJ_MLP_RATIO=1.0 

# EXP_NAME=loopy_${DATASET}/${MODEL}_B_${NUM_BLOCKS}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
# EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
# EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
# EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}_P_${NUM_PASSES}
# #
# wait_for_gpu
# GPU=$(get_next_gpu)

# if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
#     if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
#         run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
#     fi
# else
#     run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
#         --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
#         --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
#         --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
#         --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
#         --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
#         --model.num_passes ${NUM_PASSES} \
#         --run.exp_name ${EXP_NAME}
# fi

# #======================================================================#
# # LOOPY FLARE (DEEP KVPROJ, STANDARD FFN)
# #======================================================================#
# MODEL=loopy
# WEIGHT_DECAY=1e-5
# ATTN_SCALE=sqrt
# NUM_LATENTS=256
# NUM_LAYERS_KV_PROJ=3; NUM_LAYERS_FFN=0; FFN_MLP_RATIO=4.0; KV_PROJ_MLP_RATIO=1.0 

# EXP_NAME=loopy_${DATASET}/${MODEL}_B_${NUM_BLOCKS}_C_${CHANNEL_DIM}_H_${NUM_HEADS}_M_${NUM_LATENTS}_ATTNSCALE_${ATTN_SCALE}
# EXP_NAME=${EXP_NAME}_KVLAYERS_${NUM_LAYERS_KV_PROJ}_FFNLAYERS_${NUM_LAYERS_FFN}
# EXP_NAME=${EXP_NAME}_KVRATIO_${KV_PROJ_MLP_RATIO}_FFNRATIO_${FFN_MLP_RATIO}
# EXP_NAME=${EXP_NAME}/B_${NUM_BLOCKS}_P_${NUM_PASSES}
# #
# wait_for_gpu
# GPU=$(get_next_gpu)

# if [ -f "out/pdebench/${EXP_NAME}/config.yaml" ]; then
#     if [ ! -f "out/pdebench/${EXP_NAME}/ckpt10/rel_error.json" ]; then
#         run_on_gpu $GPU "$EXP_NAME" python -m pdebench --run.restart true --run.exp_name ${EXP_NAME}
#     fi
# else
#     run_on_gpu $GPU "$EXP_NAME" python -m pdebench --dataset.dataset ${DATASET} --run.train true --model.model ${MODEL} --training.mixed_precision ${MIXED_PRECISION} \
#         --training.epochs ${EPOCH} --optimizer.weight_decay ${WEIGHT_DECAY} --training.batch_size ${BATCH_SIZE} \
#         --optimizer.learning_rate "${LEARNING_RATE}" --scheduler.override_min_lr 1e-6 --training.ema ${EMA} --training.ema_decay ${EMA_DECAY} \
#         --model.channel_dim ${CHANNEL_DIM} --model.num_blocks ${NUM_BLOCKS} --model.num_heads ${NUM_HEADS} --model.rmsnorm ${RMSNORM} \
#         --model.attn_scale ${ATTN_SCALE} --model.num_latents ${NUM_LATENTS} --model.num_layers_kv_proj ${NUM_LAYERS_KV_PROJ} \
#         --model.num_layers_ffn ${NUM_LAYERS_FFN} --model.ffn_mlp_ratio ${FFN_MLP_RATIO} --model.kv_proj_mlp_ratio ${KV_PROJ_MLP_RATIO} \
#         --model.num_passes ${NUM_PASSES} \
#         --run.exp_name ${EXP_NAME}
# fi

#==============================#
done
done
#==============================#

wait
#======================================================================#
#
