#!/usr/bin/env bash
# PDEBench mixer_backbone standard-benchmark launcher.
#
# Usage:
#   DATASET=elasticity MIXER=simplifiedflarepp bash out/pdebench/run_flarepp_standard.sh
#   DATASET=elasticity MIXER=flarepp_ablations bash out/pdebench/run_flarepp_standard.sh
#   DATASET=elasticity MIXER=flarepp bash out/pdebench/run_flarepp_standard.sh
#   DATASET=darcy MIXER=flare bash out/pdebench/run_flarepp_standard.sh
#   DATASET=pipe MIXER=mha bash out/pdebench/run_flarepp_standard.sh
#   DATASET=elasticity MIXER=transolver bash out/pdebench/run_flarepp_standard.sh
#   DATASET=elasticity MIXER=transolverpp bash out/pdebench/run_flarepp_standard.sh
#   DATASET=elasticity MIXER=transolver3 bash out/pdebench/run_flarepp_standard.sh
#   DATASET=elasticity MIXER=luna bash out/pdebench/run_flarepp_standard.sh
#
# Sweep (one job per GPU via run_gpu_sweep.sh):
#   LOG_DIR=/tmp/flarepp_standard bash out/pdebench/run_gpu_sweep.sh \
#     --jobs-file /tmp/flarepp_standard/jobs.tsv
#
# Config baseline for this suite (outer C=128 H=8 IO=2 FFN=0 OPN=true MR=2);
# plot harness: ablation/sweep_flarepp.py. Suite overrides:
#   locked: compile_model=true  run.deterministic=false  (all mixers / fp32 / AMP)
#   default: mixed_precision=false  amp_dtype=None (omit); MHA: mixed_precision=false qk_norm=false
#   RESTART=true EXP_NAME=flarepp_standard/<existing_case>  # resume from latest ckpt
#   FLARE: qk_norm=false
#   LUNA: native --model.model luna (not mixer_backbone); qk_norm=false;
#         same outer C/H/IO/FFN/OPN/MR and per-dataset B/M/bs/wd/epochs as FLARE
#   FLAREPP: defaults match flarepp_ablations ABLATION_CONFIG=qk0_norm —
#            qk0_norm=true share_k0_v0=false qk_norm=false (gate/v0_norm fixed off in FLAREPPMixer;
#            SHARE_K0_V0=true supported via env)
#   FLAREPPAblations: ABLATION_CONFIG=default|qk0_norm|qk0_norm_qk_norm|gate (default: default).
#                     Shared: all use_bias=false, elementwise_affine=false, use_residual=true, share_k0_v0=false.
#                     default: gate=false, all norms/q_fixed_norm=false
#                     qk0_norm: gate=false, q0+k0 norm
#                     qk0_norm_qk_norm: gate=false, q0+k0+q+k norm
#                     gate: use_gate=true gate_logit_init=0.25 q_fixed_norm=true; norms q0+k0+v0+k
#                     Explicit env knobs still override the preset.
#   FLAREPP_ANCHORED: hop-1 q0/k0 norms always on (q0 affine, k0/v0 no affine);
#                     k_norm=true share_k0_v0=true gate_logit_init=0.25
#                     q_fixed_norm=true (env Q_FIXED_NORM; always elementwise_affine=false);
#                     gate always on; mix is q_fixed + sigmoid(gate)*q_dynamic;
#                     when share_k0_v0, v0 aliases post-norm k0 (skips separate v0_norm/v0_proj).
#                     v0_norm module kept for the non-share path (not an env knob).
#   TRANSOLVER/TRANSOLVERPP/TRANSOLVER3: qk_norm=false qk0_norm=false v0_norm=false use_gate=false; FP32 by default
#   this suite: num_layers_in_out_proj=2  out_proj_norm=true
#   num_latents=64 (darcy → 128)
#   num_blocks: elasticity/darcy/airfoil_steady=8; pipe=2; drivaerml_40k=4; lpbf=2
#
# Outputs: out/pdebench/flarepp_standard/<exp_name>/
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
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
# Locked model / dataset
# ---------------------------------------------------------------------------
if [[ -z "${DATASET:-}" ]]; then
  echo "run_flarepp_standard.sh requires DATASET=elasticity|darcy|airfoil_steady|pipe|drivaerml_40k|lpbf" >&2
  exit 1
fi
case "${DATASET}" in
  elasticity|darcy|airfoil_steady|pipe|drivaerml_40k|lpbf) ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

if [[ -z "${MIXER:-}" ]]; then
  echo "run_flarepp_standard.sh requires MIXER=mha|flare|simplifiedflarepp|flarepp_ablations|flarepp|transolver|transolverpp|transolver3|luna" >&2
  exit 1
fi
case "${MIXER}" in
  mha|flare|simplifiedflarepp|flarepp_ablations|flarepp|transolver|transolverpp|transolver3|luna) ;;
  *)
    echo "Unsupported MIXER=${MIXER}" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Dataset / mixer defaults for this experiment
# ---------------------------------------------------------------------------
STATIC_GRAPH="${STATIC_GRAPH:-true}"
STEPS="${STEPS:-0}"
SEED="${SEED:-0}"
TORCHRUN_NPROC="${TORCHRUN_NPROC:-1}"

STATS_ON_START="${STATS_ON_START:-true}"
FULLBATCH_STATS_TRAIN="${FULLBATCH_STATS_TRAIN:-true}"
FULLBATCH_STATS_TEST="${FULLBATCH_STATS_TEST:-true}"
FULLBATCH_STATS_ON_START="${FULLBATCH_STATS_ON_START:-false}"

DEFAULT_CHANNEL_DIM=128
DEFAULT_NUM_HEADS=8
DEFAULT_NUM_LAYERS_IN_OUT_PROJ=2
DEFAULT_OUT_PROJ_NORM=true
DEFAULT_NUM_LAYERS_FFN=0
DEFAULT_MLP_RATIO_FFN=2.0
DEFAULT_LEARNING_RATE=1e-3
DEFAULT_SCHEDULE=OneCycleLR
DEFAULT_OPTIMIZER=adamw
DEFAULT_EMA=true
DEFAULT_NUM_LATENTS=64
# Precision defaults (fp32). Env MIXED_PRECISION / AMP_DTYPE still win.
DEFAULT_MIXED_PRECISION=false
DEFAULT_AMP_DTYPE=""
# Locked for every run in this suite (not overridable via env).
COMPILE_MODEL=true
DETERMINISTIC=false
export COMPILE_MODEL

case "${DATASET}" in
  elasticity|airfoil_steady)
    DEFAULT_NUM_BLOCKS=8
    DEFAULT_BATCH_SIZE=2
    DEFAULT_WEIGHT_DECAY=1e-5
    DEFAULT_EPOCH=500
    ;;
  darcy)
    DEFAULT_NUM_BLOCKS=8
    DEFAULT_BATCH_SIZE=2
    DEFAULT_WEIGHT_DECAY=1e-5
    DEFAULT_EPOCH=500
    DEFAULT_NUM_LATENTS=128
    ;;
  pipe)
    DEFAULT_NUM_BLOCKS=2
    DEFAULT_BATCH_SIZE=2
    DEFAULT_WEIGHT_DECAY=1e-5
    DEFAULT_EPOCH=500
    ;;
  drivaerml_40k)
    DEFAULT_NUM_BLOCKS=4
    DEFAULT_BATCH_SIZE=1
    DEFAULT_WEIGHT_DECAY=1e-2
    DEFAULT_EPOCH=500
    ;;
  lpbf)
    DEFAULT_NUM_BLOCKS=2
    DEFAULT_BATCH_SIZE=1
    DEFAULT_WEIGHT_DECAY=1e-4
    DEFAULT_EPOCH=250
    ;;
esac

EPOCH="${EPOCH:-${DEFAULT_EPOCH}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-${DEFAULT_WEIGHT_DECAY}}"
LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LEARNING_RATE}}"
SCHEDULE="${SCHEDULE:-${DEFAULT_SCHEDULE}}"
OPTIMIZER="${OPTIMIZER:-${DEFAULT_OPTIMIZER}}"
EMA="${EMA:-${DEFAULT_EMA}}"
NUM_BLOCKS="${NUM_BLOCKS:-${DEFAULT_NUM_BLOCKS}}"
CHANNEL_DIM="${CHANNEL_DIM:-${DEFAULT_CHANNEL_DIM}}"
NUM_HEADS="${NUM_HEADS:-${DEFAULT_NUM_HEADS}}"
NUM_LATENTS="${NUM_LATENTS:-${DEFAULT_NUM_LATENTS}}"
NUM_LAYERS_FFN="${NUM_LAYERS_FFN:-${DEFAULT_NUM_LAYERS_FFN}}"
MLP_RATIO_FFN="${MLP_RATIO_FFN:-${DEFAULT_MLP_RATIO_FFN}}"
NUM_LAYERS_IN_OUT_PROJ="${NUM_LAYERS_IN_OUT_PROJ:-${DEFAULT_NUM_LAYERS_IN_OUT_PROJ}}"
OUT_PROJ_NORM="${OUT_PROJ_NORM:-${DEFAULT_OUT_PROJ_NORM}}"

# Mixer-specific defaults for this standard suite.
case "${MIXER}" in
  mha)
    QK_NORM="${QK_NORM:-false}"
    QK0_NORM="${QK0_NORM:-false}"
    V0_NORM="${V0_NORM:-false}"
    USE_GATE="${USE_GATE:-false}"
    ;;
  flare|luna)
    QK_NORM="${QK_NORM:-false}"
    QK0_NORM="${QK0_NORM:-false}"
    V0_NORM="${V0_NORM:-false}"
    USE_GATE="${USE_GATE:-false}"
    ;;
  simplifiedflarepp)
    QK_NORM="${QK_NORM:-false}"
    QK0_NORM="${QK0_NORM:-true}"
    V0_NORM="${V0_NORM:-false}"
    SHARE_K0_V0="${SHARE_K0_V0:-false}"
    USE_GATE="${USE_GATE:-false}"
    GATE_LOGIT_INIT="${GATE_LOGIT_INIT:--1.0}"
    if { [[ "${USE_GATE}" != "false" && "${USE_GATE}" != "0" ]] \
      || [[ "${V0_NORM}" != "false" && "${V0_NORM}" != "0" ]] \
      || [[ "${GATE_LOGIT_INIT}" != "-1.0" ]]; }; then
      echo "MIXER=simplifiedflarepp no longer supports USE_GATE/V0_NORM/GATE_LOGIT_INIT; use MIXER=flarepp_ablations" >&2
      exit 1
    fi
    ;;
  flarepp_ablations)
    ABLATION_CONFIG="${ABLATION_CONFIG:-default}"
    case "${ABLATION_CONFIG}" in
      default)
        _abl_q0=false
        _abl_k0=false
        _abl_v0=false
        _abl_q=false
        _abl_k=false
        _abl_q_fixed=false
        _abl_gate=false
        _abl_gate_init=0.25
        ;;
      qk0_norm)
        _abl_q0=true
        _abl_k0=true
        _abl_v0=false
        _abl_q=false
        _abl_k=false
        _abl_q_fixed=false
        _abl_gate=false
        _abl_gate_init=0.25
        ;;
      qk0_norm_qk_norm)
        _abl_q0=true
        _abl_k0=true
        _abl_v0=false
        _abl_q=true
        _abl_k=true
        _abl_q_fixed=false
        _abl_gate=false
        _abl_gate_init=0.25
        ;;
      gate)
        _abl_q0=true
        _abl_k0=true
        _abl_v0=true
        _abl_q=false
        _abl_k=true
        _abl_q_fixed=true
        _abl_gate=true
        _abl_gate_init=0.25
        ;;
      *)
        echo "Unsupported ABLATION_CONFIG=${ABLATION_CONFIG} (expected default|qk0_norm|qk0_norm_qk_norm|gate)" >&2
        exit 1
        ;;
    esac
    Q0_NORM="${Q0_NORM:-${_abl_q0}}"
    K0_NORM="${K0_NORM:-${_abl_k0}}"
    V0_NORM="${V0_NORM:-${_abl_v0}}"
    Q_NORM="${Q_NORM:-${_abl_q}}"
    K_NORM="${K_NORM:-${_abl_k}}"
    Q_FIXED_NORM="${Q_FIXED_NORM:-${_abl_q_fixed}}"
    USE_GATE="${USE_GATE:-${_abl_gate}}"
    GATE_LOGIT_INIT="${GATE_LOGIT_INIT:-${_abl_gate_init}}"
    # Shared across all ablations presets (env can still override).
    Q0_ELEMENTWISE_AFFINE="${Q0_ELEMENTWISE_AFFINE:-false}"
    K0_ELEMENTWISE_AFFINE="${K0_ELEMENTWISE_AFFINE:-false}"
    V0_ELEMENTWISE_AFFINE="${V0_ELEMENTWISE_AFFINE:-false}"
    Q_ELEMENTWISE_AFFINE="${Q_ELEMENTWISE_AFFINE:-false}"
    K_ELEMENTWISE_AFFINE="${K_ELEMENTWISE_AFFINE:-false}"
    Q_FIXED_ELEMENTWISE_AFFINE="${Q_FIXED_ELEMENTWISE_AFFINE:-false}"
    K0_USE_BIAS="${K0_USE_BIAS:-false}"
    V0_USE_BIAS="${V0_USE_BIAS:-false}"
    K_USE_BIAS="${K_USE_BIAS:-false}"
    V_USE_BIAS="${V_USE_BIAS:-false}"
    K0_USE_RESIDUAL="${K0_USE_RESIDUAL:-true}"
    V0_USE_RESIDUAL="${V0_USE_RESIDUAL:-true}"
    K_USE_RESIDUAL="${K_USE_RESIDUAL:-true}"
    V_USE_RESIDUAL="${V_USE_RESIDUAL:-true}"
    SHARE_K0_V0="${SHARE_K0_V0:-false}"
    # qk_norm unused by ablations; set a harmless default for echo tags:
    QK_NORM="${QK_NORM:-false}"
    ;;
  flarepp)
    K_NORM="${K_NORM:-true}"
    SHARE_K0_V0="${SHARE_K0_V0:-true}"
    GATE_LOGIT_INIT="${GATE_LOGIT_INIT:-0.25}"
    Q_FIXED_NORM="${Q_FIXED_NORM:-true}"
    ;;
  transolver|transolverpp|transolver3)
    QK_NORM="${QK_NORM:-false}"
    QK0_NORM="${QK0_NORM:-false}"
    V0_NORM="${V0_NORM:-false}"
    SHARE_K0_V0="${SHARE_K0_V0:-false}"
    USE_GATE="${USE_GATE:-false}"
    ;;
esac
MIXED_PRECISION="${MIXED_PRECISION:-${DEFAULT_MIXED_PRECISION}}"
# amp_dtype: if unset, use mixer default only when mixed precision is on; omit otherwise.
# Explicit AMP_DTYPE=... (including empty) always wins.
if [[ -z "${AMP_DTYPE+x}" ]]; then
  if [[ "${MIXED_PRECISION}" == "true" || "${MIXED_PRECISION}" == "1" ]]; then
    AMP_DTYPE="${DEFAULT_AMP_DTYPE}"
  else
    AMP_DTYPE=""
  fi
fi
GATE_LOGIT_INIT="${GATE_LOGIT_INIT:-0.25}"
SHARE_K0_V0="${SHARE_K0_V0:-false}"
RESTART="${RESTART:-false}"

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
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${STEPS}" != "0" ]]; then
  TRAINING_ARGS=(--training.steps "${STEPS}" --training.epochs 0)
else
  TRAINING_ARGS=(--training.steps 0 --training.epochs "${EPOCH}")
fi

_mr_tag() {
  echo "${MLP_RATIO_FFN}" | tr '.' 'p'
}

_opn_tag() {
  if [[ "${OUT_PROJ_NORM}" == "true" || "${OUT_PROJ_NORM}" == "1" ]]; then
    echo 1
  else
    echo 0
  fi
}

_prec_tag() {
  if [[ "${MIXED_PRECISION}" == "true" || "${MIXED_PRECISION}" == "1" ]]; then
    echo "amp_${AMP_DTYPE:-default}"
  else
    echo "fp32"
  fi
}

# Fully descriptive exp stem (saved under out/pdebench/flarepp_standard/).
build_exp_name() {
  local mixer_tag prec opn mr extras
  prec="$(_prec_tag)"
  opn="$(_opn_tag)"
  mr="$(_mr_tag)"
  if [[ "${MIXER}" == "mha" ]]; then
    mixer_tag="mha"
  else
    mixer_tag="${MIXER}_${NUM_LATENTS}"
  fi
  extras=""
  if [[ "${MIXER}" == "mha" || "${MIXER}" == "flare" || "${MIXER}" == "simplifiedflarepp" || "${MIXER}" == "luna" ]]; then
    extras="${extras}_qknorm_${QK_NORM}"
  fi
  if [[ "${MIXER}" == "simplifiedflarepp" ]]; then
    extras="${extras}_qk0_${QK0_NORM}"
    extras="${extras}_v0_${V0_NORM}"
    extras="${extras}_sharek0v0_${SHARE_K0_V0}"
    if [[ "${USE_GATE}" == "true" || "${USE_GATE}" == "1" ]]; then
      extras="${extras}_gate"
    else
      extras="${extras}_gate_false"
    fi
  fi
  if [[ "${MIXER}" == "flarepp_ablations" ]]; then
    extras="${extras}_cfg_${ABLATION_CONFIG:-default}"
    extras="${extras}_q0_${Q0_NORM}_k0_${K0_NORM}_v0_${V0_NORM}_q_${Q_NORM}_k_${K_NORM}"
    extras="${extras}_sharek0v0_${SHARE_K0_V0}"
    if [[ "${USE_GATE}" == "true" || "${USE_GATE}" == "1" ]]; then
      extras="${extras}_gate_${GATE_LOGIT_INIT}"
    else
      extras="${extras}_gate_false"
    fi
    if [[ "${K0_USE_BIAS}" == "true" || "${K0_USE_BIAS}" == "1" ]]; then
      extras="${extras}_k0bias"
    fi
    if [[ "${V0_USE_BIAS}" == "true" || "${V0_USE_BIAS}" == "1" ]]; then
      extras="${extras}_v0bias"
    fi
    if [[ "${K_USE_BIAS}" == "true" || "${K_USE_BIAS}" == "1" ]]; then
      extras="${extras}_kbias"
    fi
    if [[ "${V_USE_BIAS}" == "true" || "${V_USE_BIAS}" == "1" ]]; then
      extras="${extras}_vbias"
    fi
    # Preset default: all residuals on — only tag the off cases.
    if [[ "${K0_USE_RESIDUAL}" == "false" || "${K0_USE_RESIDUAL}" == "0" ]]; then
      extras="${extras}_k0res0"
    fi
    if [[ "${V0_USE_RESIDUAL}" == "false" || "${V0_USE_RESIDUAL}" == "0" ]]; then
      extras="${extras}_v0res0"
    fi
    if [[ "${K_USE_RESIDUAL}" == "false" || "${K_USE_RESIDUAL}" == "0" ]]; then
      extras="${extras}_kres0"
    fi
    if [[ "${V_USE_RESIDUAL}" == "false" || "${V_USE_RESIDUAL}" == "0" ]]; then
      extras="${extras}_vres0"
    fi
    # Default is affine off; only tag the non-default on case.
    if [[ "${Q0_ELEMENTWISE_AFFINE}" == "true" || "${Q0_ELEMENTWISE_AFFINE}" == "1" ]]; then
      extras="${extras}_q0aff"
    fi
    if [[ "${K0_ELEMENTWISE_AFFINE}" == "true" || "${K0_ELEMENTWISE_AFFINE}" == "1" ]]; then
      extras="${extras}_k0aff"
    fi
    if [[ "${V0_ELEMENTWISE_AFFINE}" == "true" || "${V0_ELEMENTWISE_AFFINE}" == "1" ]]; then
      extras="${extras}_v0aff"
    fi
    if [[ "${Q_ELEMENTWISE_AFFINE}" == "true" || "${Q_ELEMENTWISE_AFFINE}" == "1" ]]; then
      extras="${extras}_qaff"
    fi
    if [[ "${K_ELEMENTWISE_AFFINE}" == "true" || "${K_ELEMENTWISE_AFFINE}" == "1" ]]; then
      extras="${extras}_kaff"
    fi
    # q_fixed_norm varies by preset; tag off for override visibility. Affine default false → tag when on.
    if [[ "${Q_FIXED_NORM}" == "false" || "${Q_FIXED_NORM}" == "0" ]]; then
      extras="${extras}_qfnorm0"
    fi
    if [[ "${Q_FIXED_ELEMENTWISE_AFFINE}" == "true" || "${Q_FIXED_ELEMENTWISE_AFFINE}" == "1" ]]; then
      extras="${extras}_qfaff"
    fi
  fi
  if [[ "${MIXER}" == "flarepp" ]]; then
    extras="${extras}_k_${K_NORM}"
    extras="${extras}_sharek0v0_${SHARE_K0_V0}"
    extras="${extras}_gate_${GATE_LOGIT_INIT}"
    # Fixed naming tag: mix is non-convex (q_fixed + g*q_dynamic). Kept for stem continuity.
    extras="${extras}_convex_false"
    # Default q_fixed_norm=true; only tag the off case.
    if [[ "${Q_FIXED_NORM}" == "false" || "${Q_FIXED_NORM}" == "0" ]]; then
      extras="${extras}_qfnorm0"
    fi
  fi
  if [[ "${SEED}" != "0" ]]; then
    extras="${extras}_seed${SEED}"
  fi
  echo "flarepp_standard/${DATASET}_${mixer_tag}_C${CHANNEL_DIM}_B${NUM_BLOCKS}_H${NUM_HEADS}_IO_${NUM_LAYERS_IN_OUT_PROJ}_FFN_${NUM_LAYERS_FFN}_OPN_${opn}_MR_${mr}_${prec}${extras}"
}

exp_name="${EXP_NAME:-$(build_exp_name)}"

BASE_ARGS=(
  --dataset.dataset "${DATASET}"
  --training.mixed_precision "${MIXED_PRECISION}"
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
  --run.deterministic "${DETERMINISTIC}"
)

if [[ -n "${AMP_DTYPE}" ]]; then
  BASE_ARGS+=(--training.amp_dtype "${AMP_DTYPE}")
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  BASE_ARGS+=(--dataset.data_root "${DATA_ROOT}")
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

if [[ "${MIXER}" == "luna" ]]; then
  MODEL_ARGS=(
    --model.model luna
    --model.num_blocks "${NUM_BLOCKS}"
    --model.channel_dim "${CHANNEL_DIM}"
    --model.num_heads "${NUM_HEADS}"
    --model.num_layers_ffn "${NUM_LAYERS_FFN}"
    --model.ffn_mlp_ratio "${MLP_RATIO_FFN}"
    --model.num_layers_in_out_proj "${NUM_LAYERS_IN_OUT_PROJ}"
    --model.out_proj_norm "${OUT_PROJ_NORM}"
    --model.num_latents "${NUM_LATENTS}"
    --model.qk_norm "${QK_NORM}"
  )
else
MODEL_ARGS=(
  --model.model mixer_backbone
  --model.mixer.kind "${MIXER}"
  --model.num_blocks "${NUM_BLOCKS}"
  --model.channel_dim "${CHANNEL_DIM}"
  --model.num_heads "${NUM_HEADS}"
  --model.num_layers_ffn "${NUM_LAYERS_FFN}"
  --model.mlp_ratio_ffn "${MLP_RATIO_FFN}"
  --model.num_layers_in_out_proj "${NUM_LAYERS_IN_OUT_PROJ}"
  --model.out_proj_norm "${OUT_PROJ_NORM}"
)
fi

# num_latents / qk_norm belong on mixer configs that declare them
case "${MIXER}" in
  luna)
    ;;
  mha)
    MODEL_ARGS+=(--model.mixer.qk_norm "${QK_NORM}")
    ;;
  flare)
    MODEL_ARGS+=(
      --model.mixer.num_latents "${NUM_LATENTS}"
      --model.mixer.qk_norm "${QK_NORM}"
    )
    ;;
  simplifiedflarepp)
    MODEL_ARGS+=(
      --model.mixer.num_latents "${NUM_LATENTS}"
      --model.mixer.qk_norm "${QK_NORM}"
      --model.mixer.qk0_norm "${QK0_NORM}"
      --model.mixer.share_k0_v0 "${SHARE_K0_V0}"
    )
    ;;
  flarepp_ablations)
    MODEL_ARGS+=(
      --model.mixer.num_latents "${NUM_LATENTS}"
      --model.mixer.q0_norm "${Q0_NORM}"
      --model.mixer.k0_norm "${K0_NORM}"
      --model.mixer.v0_norm "${V0_NORM}"
      --model.mixer.q_norm "${Q_NORM}"
      --model.mixer.k_norm "${K_NORM}"
      --model.mixer.q_fixed_norm "${Q_FIXED_NORM}"
      --model.mixer.q0_elementwise_affine "${Q0_ELEMENTWISE_AFFINE}"
      --model.mixer.k0_elementwise_affine "${K0_ELEMENTWISE_AFFINE}"
      --model.mixer.v0_elementwise_affine "${V0_ELEMENTWISE_AFFINE}"
      --model.mixer.q_elementwise_affine "${Q_ELEMENTWISE_AFFINE}"
      --model.mixer.k_elementwise_affine "${K_ELEMENTWISE_AFFINE}"
      --model.mixer.q_fixed_elementwise_affine "${Q_FIXED_ELEMENTWISE_AFFINE}"
      --model.mixer.k0_use_bias "${K0_USE_BIAS}"
      --model.mixer.v0_use_bias "${V0_USE_BIAS}"
      --model.mixer.k_use_bias "${K_USE_BIAS}"
      --model.mixer.v_use_bias "${V_USE_BIAS}"
      --model.mixer.k0_use_residual "${K0_USE_RESIDUAL}"
      --model.mixer.v0_use_residual "${V0_USE_RESIDUAL}"
      --model.mixer.k_use_residual "${K_USE_RESIDUAL}"
      --model.mixer.v_use_residual "${V_USE_RESIDUAL}"
      --model.mixer.share_k0_v0 "${SHARE_K0_V0}"
      --model.mixer.use_gate "${USE_GATE}"
      --model.mixer.gate_logit_init "${GATE_LOGIT_INIT}"
    )
    ;;
  flarepp)
    MODEL_ARGS+=(
      --model.mixer.num_latents "${NUM_LATENTS}"
      --model.mixer.k_norm "${K_NORM}"
      --model.mixer.share_k0_v0 "${SHARE_K0_V0}"
      --model.mixer.gate_logit_init "${GATE_LOGIT_INIT}"
      --model.mixer.q_fixed_norm "${Q_FIXED_NORM}"
    )
    ;;
  transolver|transolverpp|transolver3)
    MODEL_ARGS+=(--model.mixer.num_latents "${NUM_LATENTS}")
    ;;
  *)
    echo "Unsupported MIXER=${MIXER}" >&2
    exit 1
    ;;
esac

if [[ -n "${RMSNORM+x}" ]]; then
  MODEL_ARGS+=(--model.rmsnorm "${RMSNORM}")
fi

if [[ "${DIAGNOSTICS:-}" == "true" || "${DIAGNOSTICS:-}" == "1" ]]; then
  MODEL_ARGS+=(--model.diagnostics true)
fi

run_pdebench() {
  if [[ "${TORCHRUN_NPROC}" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="${TORCHRUN_NPROC}" -m pdebench "$@"
  else
    python -m pdebench "$@"
  fi
}

if [[ "${RESTART}" == "true" || "${RESTART}" == "1" ]]; then
  if [[ -z "${EXP_NAME:-}" ]]; then
    echo "RESTART=true requires EXP_NAME=flarepp_standard/<existing_case>" >&2
    exit 1
  fi
  echo "[run_flarepp_standard] RESTART exp=${EXP_NAME}"
  run_pdebench --run.restart true --run.exp_name "${EXP_NAME}"
  exit 0
fi

if [[ "${MIXER}" == "flarepp_ablations" ]]; then
  echo "[run_flarepp_standard] mixer=${MIXER} ablation_config=${ABLATION_CONFIG:-default} dataset=${DATASET} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} B=${NUM_BLOCKS} C=${CHANNEL_DIM} H=${NUM_HEADS} M=${NUM_LATENTS} ffn=${NUM_LAYERS_FFN}/${MLP_RATIO_FFN} q0_norm=${Q0_NORM} k0_norm=${K0_NORM} v0_norm=${V0_NORM} q_norm=${Q_NORM} k_norm=${K_NORM} q_fixed_norm=${Q_FIXED_NORM} share_k0_v0=${SHARE_K0_V0} use_gate=${USE_GATE} gate_logit_init=${GATE_LOGIT_INIT} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} epochs=${EPOCH} mp=${MIXED_PRECISION} amp=${AMP_DTYPE:--} ema=${EMA} compile=${COMPILE_MODEL} deterministic=${DETERMINISTIC} exp=${exp_name}"
elif [[ "${MIXER}" == "simplifiedflarepp" ]]; then
  echo "[run_flarepp_standard] mixer=${MIXER} dataset=${DATASET} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} B=${NUM_BLOCKS} C=${CHANNEL_DIM} H=${NUM_HEADS} M=${NUM_LATENTS} ffn=${NUM_LAYERS_FFN}/${MLP_RATIO_FFN} qk_norm=${QK_NORM} qk0_norm=${QK0_NORM:-} share_k0_v0=${SHARE_K0_V0} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} epochs=${EPOCH} mp=${MIXED_PRECISION} amp=${AMP_DTYPE:--} ema=${EMA} compile=${COMPILE_MODEL} deterministic=${DETERMINISTIC} exp=${exp_name}"
elif [[ "${MIXER}" == "flarepp" ]]; then
  echo "[run_flarepp_standard] mixer=${MIXER} dataset=${DATASET} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} B=${NUM_BLOCKS} C=${CHANNEL_DIM} H=${NUM_HEADS} M=${NUM_LATENTS} ffn=${NUM_LAYERS_FFN}/${MLP_RATIO_FFN} k_norm=${K_NORM} q_fixed_norm=${Q_FIXED_NORM} share_k0_v0=${SHARE_K0_V0} gate_logit_init=${GATE_LOGIT_INIT} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} epochs=${EPOCH} mp=${MIXED_PRECISION} amp=${AMP_DTYPE:--} ema=${EMA} compile=${COMPILE_MODEL} deterministic=${DETERMINISTIC} exp=${exp_name}"
else
  echo "[run_flarepp_standard] mixer=${MIXER} dataset=${DATASET} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} B=${NUM_BLOCKS} C=${CHANNEL_DIM} H=${NUM_HEADS} M=${NUM_LATENTS} ffn=${NUM_LAYERS_FFN}/${MLP_RATIO_FFN} qk_norm=${QK_NORM} qk0_norm=${QK0_NORM:-} v0_norm=${V0_NORM:-} share_k0_v0=${SHARE_K0_V0} use_gate=${USE_GATE} gate_logit_init=${GATE_LOGIT_INIT} lr=${LEARNING_RATE} wd=${WEIGHT_DECAY} epochs=${EPOCH} mp=${MIXED_PRECISION} amp=${AMP_DTYPE:--} ema=${EMA} compile=${COMPILE_MODEL} deterministic=${DETERMINISTIC} exp=${exp_name}"
fi

run_pdebench --run.train true "${BASE_ARGS[@]}" "${TRAINING_ARGS[@]}" "${MODEL_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" --run.exp_name "${exp_name}"
