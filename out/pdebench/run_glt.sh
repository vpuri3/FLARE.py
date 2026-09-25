#!/usr/bin/env bash
# PDEBench train launcher (GLT today; MODEL= selects the architecture).
#
# DO NOT add new run_glt_*.sh / dated sweep scripts under out/pdebench/.
# Variants = env overrides. Multi-GPU sweeps = jobs.tsv + run_gpu_sweep.sh.
# See AGENTS.md ("PDEBench GLT launches") and skills/pdebench-gpu-jobs.
#
# Usage:
#   DATASET=lpbf bash out/pdebench/run_glt.sh
#   DATASET=micro_puc_fixed GLT_PE_INJECT_MODE=concat_qk GLT_PE=spe bash out/pdebench/run_glt.sh
#   DATASET=bumper_beam GLT_PE=none bash out/pdebench/run_glt.sh
#   DATASET=bumper_beam GLT_PE=geo_transolver_pe bash out/pdebench/run_glt.sh
#   DATASET=bumper_beam GLT_PE=multiscale_hop_pe bash out/pdebench/run_glt.sh
#   DATASET=plaid_elpl_terminal bash out/pdebench/run_glt.sh
#   MODEL=glt DATASET=lpbf bash out/pdebench/run_glt.sh   # MODEL defaults to glt
#   MODEL=transolver DATASET=bumper_beam bash out/pdebench/run_glt.sh
#   MODEL=geo_transolver DATASET=bumper_beam bash out/pdebench/run_glt.sh
#   MODEL=geo_transolver DATASET=bumper_beam GEO_USE_GEO=false bash out/pdebench/run_glt.sh
#   MODEL=gito DATASET=bumper_beam bash out/pdebench/run_glt.sh
#
# Slurm (full cpuset — do not train from a single agent CPU slot):
#   srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-26}" --gpus-per-task=4 --overlap \
#     bash -c 'DATASET=lpbf bash out/pdebench/run_glt.sh'
#
# Batch size: BATCH_SIZE is the **global** batch (sum over DDP ranks). See batch_size_env.sh.
#
# Shared training defaults (all datasets unless noted):
#   MODEL=glt  TORCHRUN_NPROC=4  MIXED_PRECISION=true  AMP_DTYPE=fp16  EMA=true  SEED=0
#   STATS_ON_START=true  STATS_EVERY=epochs//5 (or steps//5)
#   OPTIMIZER=adamw  (bumper_beam → muon; see per-dataset)
#   NUM_WORKERS=8  (override with NUM_WORKERS=…; PREFETCH still trainer default)
#   PLAID_USE_SDF_FEATURES=false on all PLAID datasets
#
# Shared architecture size (MODEL=glt, geo_transolver, transolver, gito):
#   NUM_BLOCKS=2  CHANNEL_DIM=128  NUM_HEADS=8  (micro_puc(_fixed) → B4; bumper_beam → B8 C128 H8)
#   GITO: NUM_BLOCKS → num_blocks_hgt; launcher always passes num_blocks_self_attn=0
# GLT-only: PE_INJECT_MODE=concat_input  PE=raw_eigen|none|multiscale_hop_pe|probe_dist  K=64|-  laplacian_spec=graph  PE_UPDATE=false  ATTN=mha (lpbf → flare128)
# GeoTransolver-only: NUM_SLICES=128; use_geo=true; include_local_features=true; concat_local_features=true;
# bumper ball radii=[0.05,0.25] ks=[8,32], global_dim=3; geometry_dim=3
#   GEO_USE_GEO=false → no GlobalContextBuilder / BQ locals / GALE cross-attn (self-attn only)
# Transolver-only: NUM_SLICES=128
# GITO-only: num_blocks_hgt from NUM_BLOCKS / GITO_NUM_BLOCKS; num_blocks_self_attn hardwired to 0
#   rmsnorm: omit flag → factory auto-enables under mixed precision + fp16/bf16
#
# Per-dataset training (batch / wd / epoch / lr / compile / schedule) — shared across models:
#   micro_puc(_fixed)     BS=64  WD=1e-5  E=200   LR=1e-3  compile  OneCycleLR  AMP=bf16
#   poisson_unstructured|poisson_structured  BS=16  WD=1e-5  E=100   LR=1e-3  compile  OneCycleLR
#   bracket_lug           BS=8   WD=1e-5  E=100   LR=1e-3  compile  OneCycleLR
#   bumper_beam           BS=1   WD=1e-5  E=2000  LR=1e-3  compile  OneCycleLR  AMP=bf16  OPT=adam
#                         arch: B8 C128 H8
#   deform_plate          BS=8   WD=1e-5  E=500   LR=1e-3  compile  OneCycleLR
#   lpbf                  BS=4   WD=1e-4  E=100   LR=1e-3  compile  OneCycleLR
#   plaid_tensile2d       BS=16  WD=1e-5  E=100   LR=1e-3  compile  OneCycleLR
#   plaid_hyperelasticity BS=8   WD=1e-5  E=1000  LR=1e-3  compile  OneCycleLR
#   plaid_el_pl_dynamics  BS=16  WD=1e-5  E=5     LR=1e-4  no-compile  ConstantLR
#   plaid_elpl_terminal   BS=8   WD=1e-5  E=200   LR=1e-4  no-compile  OneCycleLR
#                         + y_norm=asinh_iqr  target_fields=U_x
#
# Prep / viz (no shell wrappers — call Python directly):
#   python -m pdebench.dataset.laplacian.precompute --datasets <name>
#   python -m pdebench.dataset.visualize --dataset <name> …
#   bumper_beam: python -m pdebench.dataset.laplacian.precompute --datasets bumper_beam
#   deform_plate: python -m pdebench.dataset.ginot.deform_plate --data-root data --build-cache
#                 then laplacian.precompute --datasets deform_plate
#   plaid_el_pl_dynamics: build elpl_v3 shards, then
#     python -m pdebench.dataset.plaid_elpl_v3.laplacian_precompute …
#
# Subset smoke: GINOT_MAX_SAMPLES=1024 (disables fullbatch stats; stats_every=EPOCH)
# Override stats cadence: STATS_EVERY=<n>  (0 = omit flag)
# Local Micro-PUC staging: STAGE_DATA_TO_LOCAL=true
#   Optional destination: PDEBENCH_LOCAL_DATA_ROOT=/tmp/pdebench-data
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
# Shared training / dataset / optimizer / scheduler
# ---------------------------------------------------------------------------
MODEL="${MODEL:-glt}"
DATASET="${DATASET:-micro_puc}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
STATIC_GRAPH="${STATIC_GRAPH:-true}"
STEPS="${STEPS:-0}"
GINOT_MAX_SAMPLES="${GINOT_MAX_SAMPLES:-0}"
SEED="${SEED:-0}"
TORCHRUN_NPROC="${TORCHRUN_NPROC:-4}"
STAGE_DATA_TO_LOCAL="${STAGE_DATA_TO_LOCAL:-false}"
PDEBENCH_LOCAL_DATA_ROOT="${PDEBENCH_LOCAL_DATA_ROOT:-/tmp/pdebench-data}"

if [[ "${STAGE_DATA_TO_LOCAL}" == "true" ]]; then
  if [[ "${DATASET}" != "micro_puc" && "${DATASET}" != "micro_puc_fixed" ]]; then
    echo "Local staging supports only micro_puc and micro_puc_fixed; got DATASET=${DATASET}" >&2
    exit 1
  fi
  STAGE_SOURCE_ROOT="${DATA_ROOT:-${ROOT}/data}"
  STAGE_LAPLACIAN_K=0
  if [[ "${MODEL}" == "glt" && ( "${GLT_PE:-raw_eigen}" == "raw_eigen" || "${GLT_PE:-raw_eigen}" == "spe" ) ]]; then
    STAGE_LAPLACIAN_K="${GLT_PE_NUM_EIGENMODES:-64}"
  fi
  echo "[run_glt] staging dataset=${DATASET} source=${STAGE_SOURCE_ROOT} destination=${PDEBENCH_LOCAL_DATA_ROOT} K=${STAGE_LAPLACIAN_K}"
  python -m pdebench.dataset.local_stage \
    --dataset "${DATASET}" \
    --source-root "${STAGE_SOURCE_ROOT}" \
    --destination-root "${PDEBENCH_LOCAL_DATA_ROOT}" \
    --laplacian-k "${STAGE_LAPLACIAN_K}"
  DATA_ROOT="${PDEBENCH_LOCAL_DATA_ROOT}"
fi
STATS_ON_START="${STATS_ON_START:-true}"

FULLBATCH_STATS_TRAIN="${FULLBATCH_STATS_TRAIN:-true}"
FULLBATCH_STATS_TEST="${FULLBATCH_STATS_TEST:-true}"
FULLBATCH_STATS_ON_START="${FULLBATCH_STATS_ON_START:-false}"

DEFAULT_EPOCH=100
DEFAULT_BATCH_SIZE=16
DEFAULT_AMP_DTYPE=bf16
DEFAULT_WEIGHT_DECAY=1e-5
DEFAULT_LEARNING_RATE=1e-3
DEFAULT_COMPILE_MODEL=true
DEFAULT_SCHEDULE=OneCycleLR
DEFAULT_OPTIMIZER=adamw
DEFAULT_MIN_LR=0.0
DEFAULT_EMA=true

case "${DATASET}" in
  micro_puc|micro_puc_fixed)
	DEFAULT_EPOCH=200
    DEFAULT_BATCH_SIZE=64
    DEFAULT_LEARNING_RATE=1e-3
    ;;
  poisson_unstructured|poisson_structured)
    DEFAULT_BATCH_SIZE=16
    ;;
  bracket_lug)
    DEFAULT_BATCH_SIZE=8
    ;;
  bumper_beam)
    DEFAULT_BATCH_SIZE=1
    DEFAULT_EPOCH=2000
    DEFAULT_OPTIMIZER=adam
    ;;
  plaid_hyperelasticity)
    DEFAULT_BATCH_SIZE=8
    DEFAULT_EPOCH=1000
    ;;
  plaid_el_pl_dynamics)
    DEFAULT_BATCH_SIZE=16
    DEFAULT_EPOCH=5
    DEFAULT_LEARNING_RATE=1e-4
    DEFAULT_COMPILE_MODEL=false
    DEFAULT_SCHEDULE=ConstantLR
    ;;
  plaid_tensile2d)
    DEFAULT_BATCH_SIZE=16
    ;;
  plaid_elpl_terminal)
    DEFAULT_BATCH_SIZE=8
    DEFAULT_EPOCH=200
    DEFAULT_LEARNING_RATE=1e-4
    DEFAULT_COMPILE_MODEL=false
    ;;
  deform_plate)
    DEFAULT_BATCH_SIZE=8
    DEFAULT_EPOCH=500
    ;;
  lpbf)
    DEFAULT_BATCH_SIZE=4
    DEFAULT_WEIGHT_DECAY=1e-4
    ;;
  *)
    ;;
esac

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
EMA="${EMA:-${DEFAULT_EMA}}"

default_stats_every() {
  local n="$1"
  local every=$(( n / 5 ))
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

if (( GINOT_MAX_SAMPLES > 0 )); then
  FULLBATCH_STATS_TRAIN=false
  FULLBATCH_STATS_TEST=false
  STATS_EVERY="${EPOCH}"
fi

# shellcheck source=out/pdebench/ginot_dataloader_env.sh
source out/pdebench/ginot_dataloader_env.sh
# shellcheck source=out/pdebench/batch_size_env.sh
source out/pdebench/batch_size_env.sh

OVERLAP_DATALOAD="${OVERLAP_DATALOAD:-true}"
CONTINUOUS_TRAIN_BATCHES="${CONTINUOUS_TRAIN_BATCHES:-auto}"
ONE_CYCLE_PCT_START="${ONE_CYCLE_PCT_START:-0.10}"
ONE_CYCLE_DIV_FACTOR="${ONE_CYCLE_DIV_FACTOR:-10000.0}"
ONE_CYCLE_FINAL_DIV_FACTOR="${ONE_CYCLE_FINAL_DIV_FACTOR:-10000.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${STEPS}" != "0" ]]; then
  TRAINING_ARGS=(--training.steps "${STEPS}" --training.epochs 0)
else
  TRAINING_ARGS=(--training.steps 0 --training.epochs "${EPOCH}")
fi

BASE_ARGS=(
  --dataset.dataset "${DATASET}"
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

if [[ "${SCHEDULE}" == "OneCycleLR" ]]; then
  BASE_ARGS+=(
    --scheduler.pct_start "${ONE_CYCLE_PCT_START}"
    --scheduler.div_factor "${ONE_CYCLE_DIV_FACTOR}"
    --scheduler.final_div_factor "${ONE_CYCLE_FINAL_DIV_FACTOR}"
  )
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

if [[ "${GINOT_MAX_SAMPLES}" != "0" ]]; then
  BASE_ARGS+=(--dataset.max_samples "${GINOT_MAX_SAMPLES}")
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  BASE_ARGS+=(--dataset.data_root "${DATA_ROOT}")
fi

if [[ "${DATASET}" == plaid_* ]]; then
  BASE_ARGS+=(--dataset.plaid_use_sdf_features "${PLAID_USE_SDF_FEATURES:-false}")
fi

if [[ "${DATASET}" == "plaid_elpl_terminal" ]]; then
  BASE_ARGS+=(--dataset.plaid_terminal_y_norm "${PLAID_TERMINAL_Y_NORM:-asinh_iqr}")
  BASE_ARGS+=(--dataset.plaid_terminal_target_fields "${PLAID_TERMINAL_TARGET_FIELDS:-U_x}")
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

# ---------------------------------------------------------------------------
# Shared model size (GLT + GeoTransolver)
# ---------------------------------------------------------------------------
DEFAULT_NUM_BLOCKS=2
DEFAULT_CHANNEL_DIM=128
DEFAULT_NUM_HEADS=8
case "${DATASET}" in
  micro_puc|micro_puc_fixed)
    DEFAULT_NUM_BLOCKS=4
    ;;
  bumper_beam)
    DEFAULT_NUM_BLOCKS=8
    DEFAULT_CHANNEL_DIM=128
    DEFAULT_NUM_HEADS=8
    ;;
esac
NUM_BLOCKS="${NUM_BLOCKS:-${DEFAULT_NUM_BLOCKS}}"
CHANNEL_DIM="${CHANNEL_DIM:-${DEFAULT_CHANNEL_DIM}}"
NUM_HEADS="${NUM_HEADS:-${DEFAULT_NUM_HEADS}}"

# ---------------------------------------------------------------------------
# GLT model
# ---------------------------------------------------------------------------
# PE production (model.pe) is pluggable; injection into attention (model.pe_inject_mode /
# model.pe_update) is separate. See docs/superpowers/specs/2026-07-14-glt-pe-inject-split-design.md.
DEFAULT_GLT_ATTN_TYPE=mha
DEFAULT_GLT_PE_NUM_EIGENMODES=64
if [[ "${DATASET}" == "lpbf" ]]; then
  DEFAULT_GLT_ATTN_TYPE=flare128
fi

GLT_PE_INJECT_MODE="${GLT_PE_INJECT_MODE:-concat_input}"
GLT_PE_UPDATE="${GLT_PE_UPDATE:-false}"
# GLT_PE=raw_eigen | spe | none | geo_transolver_pe | multiscale_hop_pe | probe_dist   (default raw_eigen)
GLT_PE="${GLT_PE:-raw_eigen}"   # raw_eigen | spe | none | geo_transolver_pe | multiscale_hop_pe | probe_dist
GLT_PE_NUM_EIGENMODES="${GLT_PE_NUM_EIGENMODES:-${DEFAULT_GLT_PE_NUM_EIGENMODES}}"
GLT_PE_LAPLACIAN_SPEC="${GLT_PE_LAPLACIAN_SPEC:-graph}"
GLT_NUM_BLOCKS="${GLT_NUM_BLOCKS:-${NUM_BLOCKS}}"
GLT_CHANNEL_DIM="${GLT_CHANNEL_DIM:-${CHANNEL_DIM}}"
GLT_NUM_HEADS="${GLT_NUM_HEADS:-${NUM_HEADS}}"
GLT_PE_FILTER_TYPE="${GLT_PE_FILTER_TYPE:-band}"
GLT_PE_MODE="${GLT_PE_MODE:-query}"
# geo_transolver_pe knobs (ball-query local; no Laplacian)
GLT_PE_RADII="${GLT_PE_RADII:-[0.05,0.25]}"
GLT_PE_NEIGHBORS="${GLT_PE_NEIGHBORS:-[8,32]}"
GLT_PE_N_HIDDEN_LOCAL="${GLT_PE_N_HIDDEN_LOCAL:-32}"
# multiscale_hop_pe knobs (PointNet hidden width is currently fixed at 32)
GLT_PE_POINTNET_HIDDEN="${GLT_PE_POINTNET_HIDDEN:-32}"
GLT_PE_NORMALIZE_EDGE_LEN="${GLT_PE_NORMALIZE_EDGE_LEN:-true}"
# probe_dist knobs
GLT_PE_NUM_PROBES="${GLT_PE_NUM_PROBES:-32}"
GLT_PE_GEODESIC_FEATS="${GLT_PE_GEODESIC_FEATS:-false}"
GLT_PE_NUM_ANCHOR_CANDIDATES="${GLT_PE_NUM_ANCHOR_CANDIDATES:-4}"
GLT_PE_MAX_GEODESIC_HOPS="${GLT_PE_MAX_GEODESIC_HOPS:-32}"
GLT_PE_TEMPERATURE="${GLT_PE_TEMPERATURE:-1.0}"
GLT_PE_DISTANCE_CAP="${GLT_PE_DISTANCE_CAP:-2.0}"
GLT_ATTN_TYPE="${GLT_ATTN_TYPE:-${DEFAULT_GLT_ATTN_TYPE}}"

run_glt() {
  local exp_name
  exp_name="${EXP_NAME:-${DATASET}_${GLT_LABEL:-GLT}_B${GLT_NUM_BLOCKS}_C${GLT_CHANNEL_DIM}_H${GLT_NUM_HEADS}_I${GLT_PE_INJECT_MODE}}"

  local -a GLT_MODEL_ARGS=(
    --model.model glt
    --model.num_blocks "${GLT_NUM_BLOCKS}"
    --model.channel_dim "${GLT_CHANNEL_DIM}"
    --model.num_heads "${GLT_NUM_HEADS}"
    --model.attn_type "${GLT_ATTN_TYPE}"
    --model.mlp_ratio "${GLT_MLP_RATIO:-2.0}"
    --model.pe_inject_mode "${GLT_PE_INJECT_MODE}"
    --model.pe_update "${GLT_PE_UPDATE}"
    --model.pe.kind "${GLT_PE}"
  )
  # Optional explicit rmsnorm override; default is unset → factory auto under MP+fp16.
  if [[ -n "${GLT_RMSNORM+x}" ]]; then
    GLT_MODEL_ARGS+=(--model.rmsnorm "${GLT_RMSNORM}")
  fi
  if [[ "${GLT_PE}" == "raw_eigen" || "${GLT_PE}" == "spe" ]]; then
    GLT_MODEL_ARGS+=(
      --model.pe.num_eigenmodes "${GLT_PE_NUM_EIGENMODES}"
      --model.pe.laplacian_spec "${GLT_PE_LAPLACIAN_SPEC}"
    )
  fi
  if [[ "${GLT_PE}" == "spe" ]]; then
    GLT_MODEL_ARGS+=(
      --model.pe.filter_type "${GLT_PE_FILTER_TYPE}"
      --model.pe.mode "${GLT_PE_MODE}"
    )
    if [[ -n "${GLT_PE_BAND_SIGMA_INIT+x}" ]]; then
      GLT_MODEL_ARGS+=(--model.pe.band_sigma_init "${GLT_PE_BAND_SIGMA_INIT}")
    fi
    if [[ -n "${GLT_PE_NUM_FILTERS+x}" ]]; then
      GLT_MODEL_ARGS+=(--model.pe.num_filters "${GLT_PE_NUM_FILTERS}")
    fi
    if [[ -n "${GLT_PE_HIDDEN+x}" ]]; then
      GLT_MODEL_ARGS+=(--model.pe.hidden "${GLT_PE_HIDDEN}")
    fi
    if [[ -n "${GLT_PE_POLY_ORDER+x}" ]]; then
      GLT_MODEL_ARGS+=(--model.pe.poly_order "${GLT_PE_POLY_ORDER}")
    fi
    if [[ -n "${GLT_PE_NUM_HOPS+x}" ]]; then
      GLT_MODEL_ARGS+=(--model.pe.num_hops "${GLT_PE_NUM_HOPS}")
    fi
  fi
  if [[ "${GLT_PE}" == "geo_transolver_pe" ]]; then
    GLT_MODEL_ARGS+=(
      --model.pe.radii "${GLT_PE_RADII}"
      --model.pe.neighbors_in_radius "${GLT_PE_NEIGHBORS}"
      --model.pe.n_hidden_local "${GLT_PE_N_HIDDEN_LOCAL}"
    )
  fi
  if [[ "${GLT_PE}" == "multiscale_hop_pe" ]]; then
    GLT_MODEL_ARGS+=(
      --model.pe.pointnet_hidden_dim "${GLT_PE_POINTNET_HIDDEN}"
      --model.pe.normalize_by_mean_edge_length "${GLT_PE_NORMALIZE_EDGE_LEN}"
    )
  fi
  if [[ "${GLT_PE}" == "probe_dist" ]]; then
    GLT_MODEL_ARGS+=(
      --model.pe.num_probes "${GLT_PE_NUM_PROBES}"
      --model.pe.geodesic_feats "${GLT_PE_GEODESIC_FEATS}"
    )
    if [[ "${GLT_PE_GEODESIC_FEATS}" == "true" ]]; then
      GLT_MODEL_ARGS+=(
        --model.pe.num_anchor_candidates "${GLT_PE_NUM_ANCHOR_CANDIDATES}"
        --model.pe.max_geodesic_hops "${GLT_PE_MAX_GEODESIC_HOPS}"
        --model.pe.temperature "${GLT_PE_TEMPERATURE}"
        --model.pe.distance_cap "${GLT_PE_DISTANCE_CAP}"
      )
    fi
  fi

  local pe_k_display="${GLT_PE_NUM_EIGENMODES}"
  if [[ "${GLT_PE}" == "none" || "${GLT_PE}" == "geo_transolver_pe" || "${GLT_PE}" == "multiscale_hop_pe" || "${GLT_PE}" == "probe_dist" ]]; then
    pe_k_display="-"
  fi
  echo "[run_glt] model=glt attn=${GLT_ATTN_TYPE} pe_inject_mode=${GLT_PE_INJECT_MODE} pe=${GLT_PE} K=${pe_k_display} pe_update=${GLT_PE_UPDATE} B=${GLT_NUM_BLOCKS} C=${GLT_CHANNEL_DIM} H=${GLT_NUM_HEADS}"
  run_pdebench --run.train true "${BASE_ARGS[@]}" "${TRAINING_ARGS[@]}" "${GLT_MODEL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" --run.exp_name "${exp_name}"
}

# ---------------------------------------------------------------------------
# GeoTransolver model
# ---------------------------------------------------------------------------
GEO_NUM_BLOCKS="${GEO_NUM_BLOCKS:-${NUM_BLOCKS}}"
GEO_CHANNEL_DIM="${GEO_CHANNEL_DIM:-${CHANNEL_DIM}}"
GEO_NUM_HEADS="${GEO_NUM_HEADS:-${NUM_HEADS}}"
GEO_NUM_SLICES="${GEO_NUM_SLICES:-128}"
GEO_USE_GEO="${GEO_USE_GEO:-true}"
GEO_INCLUDE_LOCAL_FEATURES="${GEO_INCLUDE_LOCAL_FEATURES:-true}"
GEO_CONCAT_LOCAL_FEATURES="${GEO_CONCAT_LOCAL_FEATURES:-true}"
GEO_GEOMETRY_DIM="${GEO_GEOMETRY_DIM:-3}"
GEO_N_HIDDEN_LOCAL="${GEO_N_HIDDEN_LOCAL:-32}"
GEO_STATE_MIXING_MODE="${GEO_STATE_MIXING_MODE:-weighted}"
DEFAULT_GEO_BALL_RADII="0.05,0.25"
DEFAULT_GEO_BALL_KS="8,32"
DEFAULT_GEO_GLOBAL_DIM=""
if [[ "${DATASET}" == "bumper_beam" ]]; then
  # Paper §6.1 / PhysicsNeMo bumper GeoTransolver multi-scale ball query + globals.
  DEFAULT_GEO_BALL_RADII="0.05,0.25"
  DEFAULT_GEO_BALL_KS="8,32"
  DEFAULT_GEO_GLOBAL_DIM="3"
fi
GEO_BALL_RADII="${GEO_BALL_RADII:-${DEFAULT_GEO_BALL_RADII}}"
GEO_BALL_KS="${GEO_BALL_KS:-${DEFAULT_GEO_BALL_KS}}"
GEO_GLOBAL_DIM="${GEO_GLOBAL_DIM:-${DEFAULT_GEO_GLOBAL_DIM}}"

run_geo_transolver() {
  local exp_name
  exp_name="${EXP_NAME:-${DATASET}_${GEO_LABEL:-GeoTS}_B${GEO_NUM_BLOCKS}_C${GEO_CHANNEL_DIM}_H${GEO_NUM_HEADS}_S${GEO_NUM_SLICES}}"

  # jsonargparse wants a single list literal, not space-separated values.
  local -a radii_args=(--model.ball_radii="[${GEO_BALL_RADII}]")
  local -a ks_args=(--model.ball_ks="[${GEO_BALL_KS}]")

  local -a GEO_MODEL_ARGS=(
    --model.model geo_transolver
    --model.num_blocks "${GEO_NUM_BLOCKS}"
    --model.channel_dim "${GEO_CHANNEL_DIM}"
    --model.num_heads "${GEO_NUM_HEADS}"
    --model.num_slices "${GEO_NUM_SLICES}"
    --model.mlp_ratio "${GEO_MLP_RATIO:-2.0}"
    --model.geometry_dim "${GEO_GEOMETRY_DIM}"
    --model.use_geo "${GEO_USE_GEO}"
    --model.include_local_features "${GEO_INCLUDE_LOCAL_FEATURES}"
    --model.concat_local_features "${GEO_CONCAT_LOCAL_FEATURES}"
    --model.n_hidden_local "${GEO_N_HIDDEN_LOCAL}"
    --model.state_mixing_mode "${GEO_STATE_MIXING_MODE}"
    "${radii_args[@]}"
    "${ks_args[@]}"
  )
  if [[ -n "${GEO_GLOBAL_DIM}" ]]; then
    GEO_MODEL_ARGS+=(--model.global_dim "${GEO_GLOBAL_DIM}")
  fi
  if [[ -n "${GEO_RMSNORM+x}" ]]; then
    GEO_MODEL_ARGS+=(--model.rmsnorm "${GEO_RMSNORM}")
  fi

  echo "[run_glt] model=geo_transolver B=${GEO_NUM_BLOCKS} C=${GEO_CHANNEL_DIM} H=${GEO_NUM_HEADS} slices=${GEO_NUM_SLICES} use_geo=${GEO_USE_GEO} ball_radii=${GEO_BALL_RADII} ball_ks=${GEO_BALL_KS} include_local=${GEO_INCLUDE_LOCAL_FEATURES} concat_local=${GEO_CONCAT_LOCAL_FEATURES} global_dim=${GEO_GLOBAL_DIM:-none}"
  run_pdebench --run.train true "${BASE_ARGS[@]}" "${TRAINING_ARGS[@]}" "${GEO_MODEL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" --run.exp_name "${exp_name}"
}

# ---------------------------------------------------------------------------
# Transolver model
# ---------------------------------------------------------------------------
TS_NUM_BLOCKS="${TS_NUM_BLOCKS:-${NUM_BLOCKS}}"
TS_CHANNEL_DIM="${TS_CHANNEL_DIM:-${CHANNEL_DIM}}"
TS_NUM_HEADS="${TS_NUM_HEADS:-${NUM_HEADS}}"
TS_NUM_SLICES="${TS_NUM_SLICES:-128}"

run_transolver() {
  local exp_name
  exp_name="${EXP_NAME:-${DATASET}_${TS_LABEL:-Transolver}_B${TS_NUM_BLOCKS}_C${TS_CHANNEL_DIM}_H${TS_NUM_HEADS}_S${TS_NUM_SLICES}}"

  local -a TS_MODEL_ARGS=(
    --model.model transolver
    --model.num_blocks "${TS_NUM_BLOCKS}"
    --model.channel_dim "${TS_CHANNEL_DIM}"
    --model.num_heads "${TS_NUM_HEADS}"
    --model.num_slices "${TS_NUM_SLICES}"
    --model.mlp_ratio "${TS_MLP_RATIO:-2.0}"
  )
  if [[ -n "${TS_RMSNORM+x}" ]]; then
    TS_MODEL_ARGS+=(--model.rmsnorm "${TS_RMSNORM}")
  fi

  echo "[run_glt] model=transolver B=${TS_NUM_BLOCKS} C=${TS_CHANNEL_DIM} H=${TS_NUM_HEADS} slices=${TS_NUM_SLICES}"
  run_pdebench --run.train true "${BASE_ARGS[@]}" "${TRAINING_ARGS[@]}" "${TS_MODEL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" --run.exp_name "${exp_name}"
}

# ---------------------------------------------------------------------------
# GITO model
# ---------------------------------------------------------------------------
GITO_NUM_BLOCKS="${GITO_NUM_BLOCKS:-${NUM_BLOCKS}}"  # → num_blocks_hgt
GITO_CHANNEL_DIM="${GITO_CHANNEL_DIM:-${CHANNEL_DIM}}"
GITO_NUM_HEADS="${GITO_NUM_HEADS:-${NUM_HEADS}}"

run_gito() {
  local exp_name
  exp_name="${EXP_NAME:-${DATASET}_${GITO_LABEL:-GITO}_B${GITO_NUM_BLOCKS}_C${GITO_CHANNEL_DIM}_H${GITO_NUM_HEADS}}"

  local -a GITO_MODEL_ARGS=(
    --model.model gito
    --model.num_blocks_hgt "${GITO_NUM_BLOCKS}"
    # Launcher always uses 0 self-attn (TNO) blocks; no GITO_NUM_BLOCKS_SELF_ATTN override.
    --model.num_blocks_self_attn 0
    --model.channel_dim "${GITO_CHANNEL_DIM}"
    --model.num_heads "${GITO_NUM_HEADS}"
  )
  if [[ -n "${GITO_RMSNORM+x}" ]]; then
    GITO_MODEL_ARGS+=(--model.rmsnorm "${GITO_RMSNORM}")
  fi

  echo "[run_glt] model=gito B_hgt=${GITO_NUM_BLOCKS} B_self_attn=0 C=${GITO_CHANNEL_DIM} H=${GITO_NUM_HEADS}"
  run_pdebench --run.train true "${BASE_ARGS[@]}" "${TRAINING_ARGS[@]}" "${GITO_MODEL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" --run.exp_name "${exp_name}"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
echo "[run_glt] model=${MODEL} dataset=${DATASET} batch_size=${BATCH_SIZE} per_rank_batch_size=${PER_RANK_BATCH_SIZE} num_workers=${NUM_WORKERS} torchrun_nproc=${TORCHRUN_NPROC} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all} compile_model=${COMPILE_MODEL} optimizer=${OPTIMIZER} lr=${LEARNING_RATE} min_lr=${MIN_LR} schedule=${SCHEDULE} amp=${AMP_DTYPE} ema=${EMA} stats_every=${STATS_EVERY} max_samples=${GINOT_MAX_SAMPLES}"

case "${MODEL}" in
  glt) run_glt ;;
  geo_transolver) run_geo_transolver ;;
  transolver) run_transolver ;;
  gito) run_gito ;;
  *)
    echo "Unsupported MODEL=${MODEL} (supported: glt, geo_transolver, transolver, gito)" >&2
    exit 1
    ;;
esac
