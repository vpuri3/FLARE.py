#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [--clear]

Options:
  --clear   Recreate .venv before installing dependencies.
  -h, --help
            Show this help text.
EOF
}

CLEAR_ENV=0
for arg in "$@"; do
  case "$arg" in
    --clear)
      CLEAR_ENV=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

#=========================================#
# CUDA/PyTorch selection
#=========================================#
# Eagle
# TORCH_VERSION=2.6
# CUDA_VERSION=cu124

# GCloud H100
TORCH_VERSION=2.8
CUDA_VERSION=cu128

# Personal servers
# TORCH_VERSION=2.8
# CUDA_VERSION=cu129

#=========================================#
# Cache redirection (avoid home quota pressure)
#=========================================#
PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY_CACHE_BASE="${PROJ_DIR}/../cache"

export PIP_CACHE_DIR="${PY_CACHE_BASE}/pip"
export UV_CACHE_DIR="${PY_CACHE_BASE}/uv"
export XDG_CACHE_HOME="${PY_CACHE_BASE}"
export HF_HOME="${PY_CACHE_BASE}/huggingface"
export HUGGINGFACE_HUB_CACHE="${PY_CACHE_BASE}/huggingface"
export TORCH_HOME="${PY_CACHE_BASE}/torch"
export WANDB_CACHE_DIR="${PY_CACHE_BASE}/wandb"
export TRITON_CACHE_DIR="${PY_CACHE_BASE}/triton"
export DATASETS_CACHE="${PY_CACHE_BASE}/datasets"
export MPLCONFIGDIR="${PY_CACHE_BASE}/matplotlib"
export HF_DATASETS_CACHE="${PY_CACHE_BASE}/datasets"
export HF_HUB_CACHE="${PY_CACHE_BASE}/huggingface"

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${WANDB_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${DATASETS_CACHE}" \
  "${MPLCONFIGDIR}"

#=========================================#
# Concurrency tuning (auto CPU detection)
#=========================================#
detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
    return
  fi
  echo 1
}

CPU_COUNT=$(detect_cpu_count)
if ! [[ ${CPU_COUNT} =~ ^[0-9]+$ ]] || [[ ${CPU_COUNT} -lt 1 ]]; then
  CPU_COUNT=1
fi

# Keep build parallelism conservative to reduce OOM risk from heavy native builds.
UV_CONCURRENT_BUILDS=$((CPU_COUNT < 8 ? CPU_COUNT : 8))
UV_CONCURRENT_DOWNLOADS=$((CPU_COUNT * 2))
if [[ ${UV_CONCURRENT_DOWNLOADS} -gt 32 ]]; then
  UV_CONCURRENT_DOWNLOADS=32
fi
UV_CONCURRENT_INSTALLS=$((CPU_COUNT < 16 ? CPU_COUNT : 16))

export UV_CONCURRENT_BUILDS
export UV_CONCURRENT_DOWNLOADS
export UV_CONCURRENT_INSTALLS

echo "Detected ${CPU_COUNT} CPU cores."
echo "uv concurrency: builds=${UV_CONCURRENT_BUILDS}, downloads=${UV_CONCURRENT_DOWNLOADS}, installs=${UV_CONCURRENT_INSTALLS}"

#=========================================#
# Environment bootstrap
#=========================================#
echo "Updating uv..."
uv self update || echo "Warning: uv self-update failed (likely multiple uv installs); continuing."

if [[ ${CLEAR_ENV} -eq 1 ]]; then
  echo "Recreating virtual environment (.venv) because --clear was provided."
  uv venv --clear --python 3.11
elif [[ ! -d .venv ]]; then
  uv venv --python 3.11
fi

#=========================================#
# Torch first (CUDA-specific index)
#=========================================#
uv pip install --python .venv/bin/python \
  torch==${TORCH_VERSION} torchvision \
  --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

#=========================================#
# Optional extras
#=========================================#
SYNC_ARGS=(--python .venv/bin/python --inexact --extra dev --extra test)

read -p "Install PyG stack (torch_geometric + extensions)? [y/N] " install_pyg
if [[ ${install_pyg} == [Yy]* ]]; then
  SYNC_ARGS+=(--extra pyg)
fi

read -p "Install vision extras (diffusers/pillow/clean-fid/dali/accelerate/webdataset)? [y/N] " install_vision
if [[ ${install_vision} == [Yy]* ]]; then
  SYNC_ARGS+=(--extra vision)
fi

read -p "Install TensorFlow (meshgraphnet extra)? [y/N] " install_tf
if [[ ${install_tf} == [Yy]* ]]; then
  SYNC_ARGS+=(--extra meshgraphnet)
fi

read -p "Install AM SDF extras (trimesh/rtree)? [y/N] " install_am_sdf
if [[ ${install_am_sdf} == [Yy]* ]]; then
  SYNC_ARGS+=(--extra am_sdf)
fi

uv sync "${SYNC_ARGS[@]}"

if [[ ${install_pyg} == [Yy]* ]]; then
  uv pip install --python .venv/bin/python \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.0+${CUDA_VERSION}.html
fi

read -p "Install Mamba extras (mamba-ssm/causal-conv1d)? [y/N] " install_mamba
if [[ ${install_mamba} == [Yy]* ]]; then
  # Build against the already-installed torch in this env to avoid ABI mismatch.
  # Keep deps frozen to avoid resolver upgrading torch/CUDA runtime packages.
  uv pip install --python .venv/bin/python ninja
  uv pip install --python .venv/bin/python --no-deps --no-cache \
    --no-build-isolation --no-binary mamba-ssm --no-binary causal-conv1d \
    mamba-ssm causal-conv1d
fi

read -p "Install Flash Attention? [y/N] " install_flash_attn
if [[ ${install_flash_attn} == [Yy]* ]]; then
  # Avoid resolver churn that can remove optional extras chosen above.
  uv pip install --python .venv/bin/python flash-attn --no-build-isolation --no-deps
fi

read -p "Install LaTeX for publication-quality plots? (requires sudo) [y/N] " install_latex
if [[ ${install_latex} == [Yy]* ]]; then
  sudo apt update && sudo apt install -y \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    texlive-fonts-extra cm-super dvipng
fi

echo "Install complete. Activate with: source .venv/bin/activate"
echo "Optional: bash scripts/install_agent_tools.sh  # rg, fd, ast-grep, ruff, pytest, graphify"
