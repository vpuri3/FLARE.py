#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ovl JOB_ID [COMMAND...]

Attach an overlapping step to an existing Slurm allocation.
Examples:
  ovl 106539
  ovl 106539 nvidia-smi
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

job_id="$1"
shift

# Keep input strict to avoid accidental `srun --jobid` surprises.
if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
  echo "Error: JOB_ID must be numeric, got '${job_id}'." >&2
  exit 2
fi

# Read the live allocation so we can match its CPU geometry.
job_desc="$(scontrol show job -o "${job_id}" 2>/dev/null || true)"
if [[ -z "${job_desc}" ]]; then
  echo "Error: unable to query job ${job_id}." >&2
  exit 3
fi

cpus_per_task=1
if [[ "${job_desc}" =~ CPUs/Task=([0-9]+) ]]; then
  cpus_per_task="${BASH_REMATCH[1]}"
fi

# Default to an interactive shell; otherwise run the user-provided command.
if [[ $# -eq 0 ]]; then
  cmd=(bash)
else
  cmd=("$@")
fi

# Avoid inheriting stale binding masks from another job/step environment.
# This is the root cause of "CPU binding outside of job step allocation".
unset SLURM_CPU_BIND
unset SLURM_CPU_BIND_LIST
unset SLURM_MEM_BIND

# Ask Slurm for one task and the target job's CPUs-per-task, with bind disabled.
exec srun \
  --pty \
  --overlap \
  --jobid="${job_id}" \
  -n1 \
  -c"${cpus_per_task}" \
  --cpu-bind=none \
  --mem-bind=none \
  "${cmd[@]}"
