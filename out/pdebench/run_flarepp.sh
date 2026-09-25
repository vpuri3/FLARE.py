#!/usr/bin/env bash
# PDEBench FLARE/FLAREPP train launcher (nasa_crm, ahmedml_surface, or drivaerml_surface).
#
# Usage:
#   DATASET=nasa_crm MODEL=flarepp bash out/pdebench/run_flarepp.sh
#   DATASET=ahmedml_surface MODEL=flarepp bash out/pdebench/run_flarepp.sh
#   DATASET=drivaerml_surface MODEL=flarepp bash out/pdebench/run_flarepp.sh
#   DATASET=nasa_crm MODEL=flare USE_CONTEXT_PARALLEL=true NUM_HEADS=32 bash out/pdebench/run_flarepp.sh
#   DATASET=nasa_crm MODEL=flarepp USE_CONTEXT_PARALLEL=true bash out/pdebench/run_flarepp.sh
#   DATASET=nasa_crm MODEL=flare NUM_LATENTS=128 EPOCH=500 bash out/pdebench/run_flarepp.sh
#   DATASET=ahmedml_surface MODEL=flare SUBSET_SIZE=50000 bash out/pdebench/run_flarepp.sh
#   DATASET=drivaerml_surface MODEL=flare SUBSET_SIZE=200000 bash out/pdebench/run_flarepp.sh  # K=40
#
# Prepare nasa_crm: python scripts/download_nasa_crm.py --data-root data/NASA_CRM
# Prepare ahmedml_surface: python scripts/download_ahmedml_surface.py --data-root data/AhmedML/raw
#   && python scripts/prep_ahmedml_surface.py --data-root data/AhmedML/raw --out-root data/AhmedML/surface_full
# Prepare drivaerml_surface:
#   python scripts/download_drivaerml_surface.py --data-root data/DrivAerML/raw
#   python scripts/prep_drivaerml_surface.py --data-root data/DrivAerML/raw --out-root data/DrivAerML/surface_full
#
# Slurm (full cpuset — do not train from a single agent CPU slot):
#   srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-26}" --gpus-per-task=4 --overlap \
#     bash -c 'DATASET=nasa_crm MODEL=flare bash out/pdebench/run_flarepp.sh'
#
# Defaults:
#   DATASET required (nasa_crm|ahmedml_surface|drivaerml_surface); per-dataset DATA_ROOT; MODEL required: flare|flarepp|abupt_surface_mixer
#   NUM_BLOCKS=8  CHANNEL_DIM=128  NUM_HEADS=8  NUM_LATENTS=128
#   NUM_LAYERS_FFN=0  MLP_RATIO_FFN=4.0
#   ATTN_SCALE=sqrt (flare only; flarepp uses inv-sqrt head_dim internally)
#   BATCH_SIZE=1  NUM_WORKERS=8  MIXED_PRECISION=true  AMP_DTYPE=fp16
#   TORCHRUN_NPROC=4  EPOCH=500  LR=1e-3  WD=1e-5
#   OneCycleLR (pct_start=0.05, override_min_lr=1e-6, cycle_momentum=false)  EMA=true  compile=true
#   flare: QK_NORM=false; flarepp: hop-1 norms always on; K_NORM=true Q_FIXED_NORM=true SHARE_K0_V0=true GATE_LOGIT_INIT=0.25
#   ahmedml_surface defaults: WEIGHT_DECAY=5e-2; SUBSET_SIZE=100000; IID_SAMPLES=true
#   drivaerml_surface defaults: EMA=false; SUBSET_SIZE=100000 (K=80); IID_SAMPLES=true
#   surface: REL_L2_LOSS=false (normalized MSE); true → 0.5*(p+τ) physical Rel-L2
#   USE_CONTEXT_PARALLEL=false (flare|flarepp; set CONTEXT_PARALLEL_SIZE / TORCHRUN_NPROC)
#   DATASET=drivaerml_surface IID_SAMPLES=false …  # strided train parts
#   DATASET=ahmedml_surface REL_L2_LOSS=true …     # Rel-L2 train + full-mesh loss
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
_COMPILE_FROM_CALLER="${COMPILE_MODEL-}"
_CUDA_VISIBLE_DEVICES_FROM_CALLER="${CUDA_VISIBLE_DEVICES-}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
elif [[ -f /project/community/vedantpu/FLARE-dev.py/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /project/community/vedantpu/FLARE-dev.py/.venv/bin/activate
else
  echo "No .venv found under ${ROOT} or main checkout" >&2
  exit 1
fi
if [[ -n "${_CUDA_VISIBLE_DEVICES_FROM_CALLER}" ]]; then
  export CUDA_VISIBLE_DEVICES="${_CUDA_VISIBLE_DEVICES_FROM_CALLER}"
fi

# ---------------------------------------------------------------------------
# Locked dataset / model
# ---------------------------------------------------------------------------
if [[ -z "${DATASET:-}" ]]; then
  echo "run_flarepp.sh requires DATASET=nasa_crm|ahmedml_surface|drivaerml_surface" >&2
  exit 1
fi
case "${DATASET}" in
  nasa_crm) DATA_ROOT="${DATA_ROOT:-${ROOT}/data/NASA_CRM}" ;;
  ahmedml_surface) DATA_ROOT="${DATA_ROOT:-${ROOT}/data/AhmedML/surface_full}" ;;
  drivaerml_surface) DATA_ROOT="${DATA_ROOT:-${ROOT}/data/DrivAerML/surface_full}" ;;
  *)
    echo "run_flarepp.sh only supports DATASET=nasa_crm|ahmedml_surface|drivaerml_surface; got DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

if [[ -z "${MODEL:-}" ]]; then
  echo "run_flarepp.sh requires MODEL=flare|flarepp|abupt_surface_mixer" >&2
  exit 1
fi
if [[ "${MODEL}" != "flare" && "${MODEL}" != "flarepp" && "${MODEL}" != "abupt_surface_mixer" ]]; then
  echo "run_flarepp.sh only supports MODEL=flare|flarepp|abupt_surface_mixer; got MODEL=${MODEL}" >&2
  exit 1
fi
if [[ "${MODEL}" == "abupt_surface_mixer" && "${DATASET}" != "drivaerml_surface" ]]; then
  echo "${MODEL} only supports DATASET=drivaerml_surface; got DATASET=${DATASET}" >&2
  exit 1
fi
MIXED_PRECISION="${MIXED_PRECISION:-true}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
STATIC_GRAPH="${STATIC_GRAPH:-true}"
STEPS="${STEPS:-0}"
SEED="${SEED:-0}"
RESTART="${RESTART:-false}"
if [[ "${MODEL}" == "abupt_surface_mixer" ]]; then
  TORCHRUN_NPROC="${TORCHRUN_NPROC:-1}"
else
  TORCHRUN_NPROC="${TORCHRUN_NPROC:-4}"
fi
USE_CONTEXT_PARALLEL="${USE_CONTEXT_PARALLEL:-false}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${TORCHRUN_NPROC}}"
CP_SEQUENCE_DIM="${CP_SEQUENCE_DIM:-1}"

STATS_ON_START="${STATS_ON_START:-true}"
FULLBATCH_STATS_TRAIN="${FULLBATCH_STATS_TRAIN:-true}"
FULLBATCH_STATS_TEST="${FULLBATCH_STATS_TEST:-true}"
FULLBATCH_STATS_ON_START="${FULLBATCH_STATS_ON_START:-false}"

DEFAULT_EPOCH=500
DEFAULT_BATCH_SIZE=1
DEFAULT_AMP_DTYPE=fp16
DEFAULT_WEIGHT_DECAY=1e-5
DEFAULT_LEARNING_RATE=1e-3
DEFAULT_COMPILE_MODEL=true
DEFAULT_SCHEDULE=OneCycleLR
DEFAULT_OPTIMIZER=adamw
DEFAULT_MIN_LR=0.0
DEFAULT_OVERRIDE_MIN_LR=1e-6
DEFAULT_EMA=true
DEFAULT_NUM_BLOCKS=8
DEFAULT_CHANNEL_DIM=128
DEFAULT_NUM_HEADS=8
DEFAULT_NUM_LATENTS=128
DEFAULT_NUM_LAYERS_FFN=0
DEFAULT_MLP_RATIO_FFN=4.0
DEFAULT_ATTN_SCALE=sqrt
DEFAULT_NUM_LAYERS_IN_OUT_PROJ=2
DEFAULT_OUT_PROJ_NORM=true
DEFAULT_NUM_LAYERS_K_PROJ=-1
DEFAULT_NUM_LAYERS_V_PROJ=-1
DEFAULT_K_PROJ_MLP_RATIO=1.0
DEFAULT_V_PROJ_MLP_RATIO=1.0
DEFAULT_QK_NORM_FLARE=false
DEFAULT_K_NORM_FLAREPP=true
DEFAULT_Q_FIXED_NORM_FLAREPP=true
DEFAULT_SHARE_K0_V0_FLAREPP=true

ABUPT_COORDINATE_SCALE="${ABUPT_COORDINATE_SCALE:-1000.0}"
ABUPT_MIXER_NUM_LAYERS_FFN="${ABUPT_MIXER_NUM_LAYERS_FFN:-${DEFAULT_NUM_LAYERS_FFN}}"
ABUPT_MIXER_FFN_MLP_RATIO="${ABUPT_MIXER_FFN_MLP_RATIO:-2.0}"
ABUPT_MIXER_NUM_SURFACE_ANCHORS="${ABUPT_MIXER_NUM_SURFACE_ANCHORS:-1024}"

# Dataset-specific defaults (overrideable via env).
if [[ "${DATASET}" == "ahmedml_surface" ]]; then
  DEFAULT_WEIGHT_DECAY=5e-2
fi
if [[ "${DATASET}" == "drivaerml_surface" ]]; then
  DEFAULT_EMA=false
fi
SUBSET_SIZE="${SUBSET_SIZE:-100000}"
IID_SAMPLES="${IID_SAMPLES:-true}"
REL_L2_LOSS="${REL_L2_LOSS:-false}"

AMP_DTYPE="${AMP_DTYPE:-${DEFAULT_AMP_DTYPE}}"
EPOCH="${EPOCH:-${DEFAULT_EPOCH}}"
COMPILE_MODEL="${_COMPILE_FROM_CALLER:-${DEFAULT_COMPILE_MODEL}}"
export COMPILE_MODEL
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-${DEFAULT_WEIGHT_DECAY}}"
LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LEARNING_RATE}}"
SCHEDULE="${SCHEDULE:-${DEFAULT_SCHEDULE}}"
OPTIMIZER="${OPTIMIZER:-${DEFAULT_OPTIMIZER}}"
MIN_LR="${MIN_LR:-${DEFAULT_MIN_LR}}"
OVERRIDE_MIN_LR="${OVERRIDE_MIN_LR:-${DEFAULT_OVERRIDE_MIN_LR}}"
EMA="${EMA:-${DEFAULT_EMA}}"
NUM_BLOCKS="${NUM_BLOCKS:-${DEFAULT_NUM_BLOCKS}}"
CHANNEL_DIM="${CHANNEL_DIM:-${DEFAULT_CHANNEL_DIM}}"
NUM_HEADS="${NUM_HEADS:-${DEFAULT_NUM_HEADS}}"
NUM_LATENTS="${NUM_LATENTS:-${DEFAULT_NUM_LATENTS}}"
NUM_LAYERS_FFN="${NUM_LAYERS_FFN:-${DEFAULT_NUM_LAYERS_FFN}}"
MLP_RATIO_FFN="${MLP_RATIO_FFN:-${DEFAULT_MLP_RATIO_FFN}}"
NUM_LAYERS_IN_OUT_PROJ="${NUM_LAYERS_IN_OUT_PROJ:-${DEFAULT_NUM_LAYERS_IN_OUT_PROJ}}"
OUT_PROJ_NORM="${OUT_PROJ_NORM:-${DEFAULT_OUT_PROJ_NORM}}"
NUM_LAYERS_K_PROJ="${NUM_LAYERS_K_PROJ:-${DEFAULT_NUM_LAYERS_K_PROJ}}"
NUM_LAYERS_V_PROJ="${NUM_LAYERS_V_PROJ:-${DEFAULT_NUM_LAYERS_V_PROJ}}"
K_PROJ_MLP_RATIO="${K_PROJ_MLP_RATIO:-${DEFAULT_K_PROJ_MLP_RATIO}}"
V_PROJ_MLP_RATIO="${V_PROJ_MLP_RATIO:-${DEFAULT_V_PROJ_MLP_RATIO}}"

# attn_scale: FLAREModel defaults to sqrt in this harness; flarepp uses inv-sqrt head_dim internally.
if [[ "${MODEL}" == "flare" ]]; then
  ATTN_SCALE="${ATTN_SCALE:-${DEFAULT_ATTN_SCALE}}"
  QK_NORM="${QK_NORM:-${DEFAULT_QK_NORM_FLARE}}"
  K_NORM="${K_NORM-}"
  Q_FIXED_NORM="${Q_FIXED_NORM-}"
  SHARE_K0_V0="${SHARE_K0_V0-}"
elif [[ "${MODEL}" == "flarepp" ]]; then
  ATTN_SCALE="${ATTN_SCALE-}"
  QK_NORM="${QK_NORM-}"
  K_NORM="${K_NORM:-${DEFAULT_K_NORM_FLAREPP}}"
  Q_FIXED_NORM="${Q_FIXED_NORM:-${DEFAULT_Q_FIXED_NORM_FLAREPP}}"
  SHARE_K0_V0="${SHARE_K0_V0:-${DEFAULT_SHARE_K0_V0_FLAREPP}}"
else
  ATTN_SCALE=""
  QK_NORM=""
  K_NORM=""
  Q_FIXED_NORM=""
  SHARE_K0_V0=""
fi
GATE_LOGIT_INIT="${GATE_LOGIT_INIT:-0.25}"

if [[ "${USE_CONTEXT_PARALLEL}" == "true" || "${USE_CONTEXT_PARALLEL}" == "1" ]]; then
  if [[ "${MODEL}" == "abupt_surface_mixer" ]]; then
    echo "${MODEL} does not support context parallelism yet" >&2
    exit 1
  fi
  if [[ "${MODEL}" != "flare" && "${MODEL}" != "flarepp" ]]; then
    echo "USE_CONTEXT_PARALLEL requires MODEL=flare|flarepp; got MODEL=${MODEL}" >&2
    exit 1
  fi
  if [[ "${TORCHRUN_NPROC}" -lt 2 ]]; then
    echo "USE_CONTEXT_PARALLEL requires TORCHRUN_NPROC>=2; got ${TORCHRUN_NPROC}" >&2
    exit 1
  fi
  if (( TORCHRUN_NPROC % CONTEXT_PARALLEL_SIZE != 0 )); then
    echo "TORCHRUN_NPROC=${TORCHRUN_NPROC} must be divisible by CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE}" >&2
    exit 1
  fi
fi
export USE_CONTEXT_PARALLEL

default_stats_every() {
  local n="$1"
  local every=$(( n / 10 ))
  if [[ "${every}" -lt 1 ]]; then
    every=1
  fi
  echo "${every}"
}

if [[ "${STEPS}" != "0" ]]; then
  STATS_EVERY="${STATS_EVERY:-$(default_stats_every "${STEPS}")}"
else
  STATS_EVERY="${STATS_EVERY:-$(default_stats_every "${EPOCH}")}"
fi

# shellcheck source=out/pdebench/batch_size_env.sh
source out/pdebench/batch_size_env.sh

OVERLAP_DATALOAD="${OVERLAP_DATALOAD:-true}"
CONTINUOUS_TRAIN_BATCHES="${CONTINUOUS_TRAIN_BATCHES:-auto}"
# Transolver-3-style: 5% warmup, cosine anneal to override_min_lr (via div_factor rewrite in __main__).
ONE_CYCLE_PCT_START="${ONE_CYCLE_PCT_START:-0.05}"
ONE_CYCLE_DIV_FACTOR="${ONE_CYCLE_DIV_FACTOR:-10000.0}"
ONE_CYCLE_FINAL_DIV_FACTOR="${ONE_CYCLE_FINAL_DIV_FACTOR:-10000.0}"
ONE_CYCLE_CYCLE_MOMENTUM="${ONE_CYCLE_CYCLE_MOMENTUM:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${STEPS}" != "0" ]]; then
  TRAINING_ARGS=(--training.steps "${STEPS}" --training.epochs 0)
else
  TRAINING_ARGS=(--training.steps 0 --training.epochs "${EPOCH}")
fi

BASE_ARGS=(
  --dataset.dataset "${DATASET}"
  --dataset.data_root "${DATA_ROOT}"
  --training.mixed_precision "${MIXED_PRECISION}"
  --training.amp_dtype "${AMP_DTYPE}"
  --training.compile_model "${COMPILE_MODEL}"
  --training.static_graph "${STATIC_GRAPH}"
  --training.stats_on_start "${STATS_ON_START}"
  --training.fullbatch_stats_on_start "${FULLBATCH_STATS_ON_START}"
  --training.fullbatch_stats_train "${FULLBATCH_STATS_TRAIN}"
  --training.fullbatch_stats_test "${FULLBATCH_STATS_TEST}"
  --training.batch_size "${TRAINING_BATCH_SIZE}"
  --training.num_workers "${NUM_WORKERS}"
  --training.ema "${EMA}"
  --training.overlap_train_dataloader "${OVERLAP_DATALOAD}"
  --optimizer.optimizer "${OPTIMIZER}"
  --optimizer.weight_decay "${WEIGHT_DECAY}"
  --optimizer.learning_rate "${LEARNING_RATE}"
  --optimizer.opt_beta1 0.9
  --optimizer.opt_beta2 0.999
  --optimizer.opt_eps 1e-8
  --scheduler.schedule "${SCHEDULE}"
  --run.seed "${SEED}"
)

if [[ "${DATASET}" == "ahmedml_surface" || "${DATASET}" == "drivaerml_surface" ]]; then
  BASE_ARGS+=(
    --dataset.subset_size "${SUBSET_SIZE}"
    --dataset.iid_samples "${IID_SAMPLES}"
    --dataset.rel_l2_loss "${REL_L2_LOSS}"
  )
fi

if [[ "${USE_CONTEXT_PARALLEL}" == "true" || "${USE_CONTEXT_PARALLEL}" == "1" ]]; then
  BASE_ARGS+=(
    --training.use_context_parallel true
    --training.context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
    --training.cp_sequence_dim "${CP_SEQUENCE_DIM}"
  )
fi

if [[ "${SCHEDULE}" == "OneCycleLR" ]]; then
  BASE_ARGS+=(
    --scheduler.pct_start "${ONE_CYCLE_PCT_START}"
    --scheduler.div_factor "${ONE_CYCLE_DIV_FACTOR}"
    --scheduler.final_div_factor "${ONE_CYCLE_FINAL_DIV_FACTOR}"
    --scheduler.cycle_momentum "${ONE_CYCLE_CYCLE_MOMENTUM}"
  )
  if [[ -n "${OVERRIDE_MIN_LR}" && "${OVERRIDE_MIN_LR}" != "0" && "${OVERRIDE_MIN_LR}" != "0.0" ]]; then
    BASE_ARGS+=(--scheduler.override_min_lr "${OVERRIDE_MIN_LR}")
  fi
fi

if [[ "${SCHEDULE}" == "CosineAnnealingLR" ]]; then
  BASE_ARGS+=(--scheduler.min_lr "${MIN_LR}")
fi

if [[ "${STATS_EVERY}" != "0" ]]; then
  BASE_ARGS+=(--training.stats_every "${STATS_EVERY}")
fi

if [[ "${CONTINUOUS_TRAIN_BATCHES}" != "auto" ]]; then
  BASE_ARGS+=(--training.continuous_train_batches "${CONTINUOUS_TRAIN_BATCHES}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
else
  EXTRA_ARGS_ARRAY=()
fi

run_pdebench() {
  if [[ "${TORCHRUN_NPROC}" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="${TORCHRUN_NPROC}" -m pdebench "$@"
  else
    python -m pdebench "$@"
  fi
}

if [[ "${MODEL}" == "flare" ]]; then
  if [[ "${USE_CONTEXT_PARALLEL}" == "true" || "${USE_CONTEXT_PARALLEL}" == "1" ]]; then
    exp_name="${EXP_NAME:-${DATASET}_flare_cp${CONTEXT_PARALLEL_SIZE}_B${NUM_BLOCKS}_C${CHANNEL_DIM}_H${NUM_HEADS}_M${NUM_LATENTS}}"
  else
    exp_name="${EXP_NAME:-${DATASET}_flare_B${NUM_BLOCKS}_C${CHANNEL_DIM}_H${NUM_HEADS}_M${NUM_LATENTS}}"
  fi
  MODEL_ARGS=(
    --model.model flare
    --model.num_blocks "${NUM_BLOCKS}"
    --model.channel_dim "${CHANNEL_DIM}"
    --model.num_heads "${NUM_HEADS}"
    --model.num_latents "${NUM_LATENTS}"
    --model.num_layers_ffn "${NUM_LAYERS_FFN}"
    --model.ffn_mlp_ratio "${MLP_RATIO_FFN}"
    --model.num_layers_in_out_proj "${NUM_LAYERS_IN_OUT_PROJ}"
    --model.out_proj_norm "${OUT_PROJ_NORM}"
    --model.num_layers_k_proj "${NUM_LAYERS_K_PROJ}"
    --model.num_layers_v_proj "${NUM_LAYERS_V_PROJ}"
    --model.k_proj_mlp_ratio "${K_PROJ_MLP_RATIO}"
    --model.v_proj_mlp_ratio "${V_PROJ_MLP_RATIO}"
    --model.qk_norm "${QK_NORM}"
  )
elif [[ "${MODEL}" == "flarepp" ]]; then
  # flarepp
  share_tag="_sharek0v0_${SHARE_K0_V0}"
  if [[ "${USE_CONTEXT_PARALLEL}" == "true" || "${USE_CONTEXT_PARALLEL}" == "1" ]]; then
    exp_name="${EXP_NAME:-${DATASET}_flarepp_cp${CONTEXT_PARALLEL_SIZE}_B${NUM_BLOCKS}_C${CHANNEL_DIM}_H${NUM_HEADS}_M${NUM_LATENTS}${share_tag}}"
  else
    exp_name="${EXP_NAME:-${DATASET}_flarepp_B${NUM_BLOCKS}_C${CHANNEL_DIM}_H${NUM_HEADS}_M${NUM_LATENTS}${share_tag}}"
  fi
  MODEL_ARGS=(
    --model.model flarepp
    --model.num_blocks "${NUM_BLOCKS}"
    --model.channel_dim "${CHANNEL_DIM}"
    --model.num_heads "${NUM_HEADS}"
    --model.num_latents "${NUM_LATENTS}"
    --model.num_layers_ffn "${NUM_LAYERS_FFN}"
    --model.ffn_mlp_ratio "${MLP_RATIO_FFN}"
    --model.num_layers_in_out_proj "${NUM_LAYERS_IN_OUT_PROJ}"
    --model.out_proj_norm "${OUT_PROJ_NORM}"
    --model.k_norm "${K_NORM}"
    --model.q_fixed_norm "${Q_FIXED_NORM}"
    --model.share_k0_v0 "${SHARE_K0_V0}"
    --model.gate_logit_init "${GATE_LOGIT_INIT}"
  )
else
  exp_name="${EXP_NAME:-${DATASET}_abupt_surface_mixer_B${NUM_BLOCKS}_C${CHANNEL_DIM}_H${NUM_HEADS}_A${ABUPT_MIXER_NUM_SURFACE_ANCHORS}}"
  MODEL_ARGS=(
    --model.model abupt_surface_mixer
    --model.num_blocks "${NUM_BLOCKS}"
    --model.channel_dim "${CHANNEL_DIM}"
    --model.num_heads "${NUM_HEADS}"
    --model.num_layers_ffn "${ABUPT_MIXER_NUM_LAYERS_FFN}"
    --model.ffn_mlp_ratio "${ABUPT_MIXER_FFN_MLP_RATIO}"
    --model.num_layers_in_out_proj "${NUM_LAYERS_IN_OUT_PROJ}"
    --model.out_proj_norm "${OUT_PROJ_NORM}"
    --model.coordinate_scale "${ABUPT_COORDINATE_SCALE}"
  )
  if [[ -n "${ABUPT_MIXER_NUM_SURFACE_ANCHORS}" ]]; then
    exp_name+="_A${ABUPT_MIXER_NUM_SURFACE_ANCHORS}"
    MODEL_ARGS+=(--model.num_surface_anchors "${ABUPT_MIXER_NUM_SURFACE_ANCHORS}")
  fi
fi

if [[ -n "${ATTN_SCALE}" ]]; then
  MODEL_ARGS+=(--model.attn_scale "${ATTN_SCALE}")
fi
if [[ -n "${RMSNORM+x}" ]]; then
  MODEL_ARGS+=(--model.rmsnorm "${RMSNORM}")
fi

if [[ "${MODEL}" == "abupt_surface_mixer" ]]; then
  echo "[run_flarepp] model=${MODEL} sequence=S${NUM_BLOCKS} dataset=${DATASET} data_root=${DATA_ROOT} batch_size=${BATCH_SIZE} per_rank_batch_size=${PER_RANK_BATCH_SIZE} num_workers=${NUM_WORKERS} torchrun_nproc=${TORCHRUN_NPROC} cp=${USE_CONTEXT_PARALLEL} B=${NUM_BLOCKS} C=${CHANNEL_DIM} H=${NUM_HEADS} anchors=${ABUPT_MIXER_NUM_SURFACE_ANCHORS} ffn=${ABUPT_MIXER_NUM_LAYERS_FFN}/${ABUPT_MIXER_FFN_MLP_RATIO} lr=${LEARNING_RATE} schedule=${SCHEDULE} amp=${AMP_DTYPE} ema=${EMA} stats_every=${STATS_EVERY} subset_size=${SUBSET_SIZE} iid_samples=${IID_SAMPLES} rel_l2_loss=${REL_L2_LOSS} wd=${WEIGHT_DECAY}"
else
  echo "[run_flarepp] model=${MODEL} dataset=${DATASET} data_root=${DATA_ROOT} batch_size=${BATCH_SIZE} per_rank_batch_size=${PER_RANK_BATCH_SIZE} num_workers=${NUM_WORKERS} torchrun_nproc=${TORCHRUN_NPROC} cp=${USE_CONTEXT_PARALLEL} cp_size=${CONTEXT_PARALLEL_SIZE} B=${NUM_BLOCKS} C=${CHANNEL_DIM} H=${NUM_HEADS} M=${NUM_LATENTS} ffn=${NUM_LAYERS_FFN}/${MLP_RATIO_FFN} attn_scale=${ATTN_SCALE:--} qk_norm=${QK_NORM:--} k_norm=${K_NORM:--} q_fixed_norm=${Q_FIXED_NORM:--} share_k0_v0=${SHARE_K0_V0:--} gate_logit_init=${GATE_LOGIT_INIT} lr=${LEARNING_RATE} override_min_lr=${OVERRIDE_MIN_LR:--} pct_start=${ONE_CYCLE_PCT_START:--} cycle_momentum=${ONE_CYCLE_CYCLE_MOMENTUM:--} schedule=${SCHEDULE} amp=${AMP_DTYPE} ema=${EMA} stats_every=${STATS_EVERY} subset_size=${SUBSET_SIZE} iid_samples=${IID_SAMPLES} rel_l2_loss=${REL_L2_LOSS}"
fi

if [[ "${RESTART}" == "true" || "${RESTART}" == "1" ]]; then
  RUN_MODE_ARGS=(--run.restart true)
else
  RUN_MODE_ARGS=(--run.train true)
fi

run_pdebench "${RUN_MODE_ARGS[@]}" "${BASE_ARGS[@]}" "${TRAINING_ARGS[@]}" "${MODEL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" --run.exp_name "${exp_name}"
