#!/usr/bin/env bash
# Launch tab-separated sweep jobs with one training process per GPU slot.
#
# jobs.tsv format (one job per line):
#   label<TAB>KEY=val ...<TAB>command
#
# Example:
#   LOG_DIR=/tmp/my_sweep bash out/pdebench/run_gpu_sweep.sh --jobs-file /tmp/my_sweep/jobs.tsv
#
set -euo pipefail

JOBS_FILE=""
NUM_GPUS="${NUM_GPUS:-}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
LOG_DIR="${LOG_DIR:-/tmp/pdebench_gpu_sweep_$(date +%Y%m%d_%H%M%S)}"

usage() {
  echo "Usage: LOG_DIR=/tmp/sweep bash out/pdebench/run_gpu_sweep.sh --jobs-file /path/jobs.tsv" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs-file)
      JOBS_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "${JOBS_FILE}" || ! -f "${JOBS_FILE}" ]]; then
  echo "Missing or unreadable --jobs-file ${JOBS_FILE}" >&2
  exit 1
fi

if [[ -z "${NUM_GPUS}" ]]; then
  if [[ -n "${SWEEP_GPU_IDS:-}" ]]; then
    IFS=',' read -ra _SWEEP_GPU_PARSED <<< "${SWEEP_GPU_IDS}"
    NUM_GPUS="${#_SWEEP_GPU_PARSED[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    NUM_GPUS=1
  fi
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=1
fi

declare -a GPU_IDS=()
if [[ -n "${SWEEP_GPU_IDS:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "${SWEEP_GPU_IDS}"
  NUM_GPUS="${#GPU_IDS[@]}"
else
  for g in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_IDS+=("${g}")
  done
fi

mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"

log_master() {
  echo "[$(date -Iseconds)] $*" | tee -a "${MASTER_LOG}"
}

run_job() {
  local gpu="$1"
  local label="$2"
  local env_str="$3"
  local cmd="$4"
  local job_log="${LOG_DIR}/${label}.log"

  log_master "START gpu=${gpu} label=${label}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TORCHRUN_NPROC=1
    # shellcheck disable=SC2086
    eval export ${env_str}
    exec > >(tee -a "${job_log}") 2>&1
    echo "[job] gpu=${gpu} label=${label} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TORCHRUN_NPROC=${TORCHRUN_NPROC}"
    echo "[job] env: ${env_str}"
    echo "[job] cmd: ${cmd}"
    eval "${cmd}"
  )
  local rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    log_master "DONE gpu=${gpu} label=${label}"
  else
    log_master "FAIL gpu=${gpu} label=${label} exit=${rc}"
  fi
  return "${rc}"
}

if [[ "${SLOTS_PER_GPU}" -lt 1 ]]; then
  SLOTS_PER_GPU=1
fi
MAX_PARALLEL=$((NUM_GPUS * SLOTS_PER_GPU))

count_jobs_on_gpu() {
  local target_gpu="$1"
  local i count=0
  for i in "${!PIDS[@]}"; do
    if [[ "${PHYSICAL_GPUS[$i]}" == "${target_gpu}" ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

pick_gpu() {
  local best_gpu="" min_jobs=999 job_count gpu
  for gpu in "${GPU_IDS[@]}"; do
    job_count="$(count_jobs_on_gpu "${gpu}")"
    if [[ "${job_count}" -lt "${SLOTS_PER_GPU}" && "${job_count}" -lt "${min_jobs}" ]]; then
      min_jobs="${job_count}"
      best_gpu="${gpu}"
    fi
  done
  echo "${best_gpu}"
}

mapfile -t JOB_LINES < "${JOBS_FILE}"
TOTAL="${#JOB_LINES[@]}"
log_master "sweep start jobs=${TOTAL} num_gpus=${NUM_GPUS} slots_per_gpu=${SLOTS_PER_GPU} max_parallel=${MAX_PARALLEL} gpu_ids=${GPU_IDS[*]} log_dir=${LOG_DIR} jobs_file=${JOBS_FILE}"

declare -a PIDS=()
declare -a LABELS=()
declare -a PHYSICAL_GPUS=()
FAILURES=0
NEXT_JOB=0

reap_finished_jobs() {
  local i
  for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      if wait "${PIDS[$i]}"; then
        log_master "slot free gpu=${PHYSICAL_GPUS[$i]} label=${LABELS[$i]} status=ok"
      else
        log_master "slot free gpu=${PHYSICAL_GPUS[$i]} label=${LABELS[$i]} status=fail"
        FAILURES=$((FAILURES + 1))
      fi
      unset 'PIDS[i]' 'LABELS[i]' 'PHYSICAL_GPUS[i]'
    fi
  done
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    PIDS=("${PIDS[@]}")
    LABELS=("${LABELS[@]}")
    PHYSICAL_GPUS=("${PHYSICAL_GPUS[@]}")
  else
    PIDS=()
    LABELS=()
    PHYSICAL_GPUS=()
  fi
}

while [[ "${NEXT_JOB}" -lt "${TOTAL}" || "${#PIDS[@]}" -gt 0 ]]; do
  reap_finished_jobs

  while [[ "${NEXT_JOB}" -lt "${TOTAL}" && "${#PIDS[@]}" -lt "${MAX_PARALLEL}" ]]; do
    physical_gpu="$(pick_gpu)"
    if [[ -z "${physical_gpu}" ]]; then
      break
    fi

    line="${JOB_LINES[$NEXT_JOB]}"
    NEXT_JOB=$((NEXT_JOB + 1))
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line}" =~ ^# ]] && continue

    IFS=$'\t' read -r label env_str cmd <<< "${line}"
    if [[ -z "${label}" || -z "${env_str}" || -z "${cmd}" ]]; then
      log_master "SKIP malformed line: ${line}"
      FAILURES=$((FAILURES + 1))
      continue
    fi

    run_job "${physical_gpu}" "${label}" "${env_str}" "${cmd}" &
    PIDS+=("$!")
    LABELS+=("${label}")
    PHYSICAL_GPUS+=("${physical_gpu}")
  done

  if [[ "${#PIDS[@]}" -eq 0 && "${NEXT_JOB}" -ge "${TOTAL}" ]]; then
    break
  fi
  sleep 2
done

log_master "sweep finished failures=${FAILURES}/${TOTAL}"
exit $((FAILURES > 0))
