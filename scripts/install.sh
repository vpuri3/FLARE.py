#!/bin/bash
set -euo pipefail

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
# Environment bootstrap
#=========================================#
cd "${PROJ_DIR}"
uv self update
uv python install 3.11

if [[ ! -d .venv ]]; then
  uv venv --python 3.11
fi

#=========================================#
# Torch first (CUDA-specific index)
#=========================================#
uv pip install --python .venv/bin/python \
  torch==${TORCH_VERSION} torchvision \
  --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

#=========================================#
# Install from pyproject (core + dev + test)
#=========================================#
uv sync --python .venv/bin/python --extra dev --extra test

#=========================================#
# Optional extras
#=========================================#
read -p "Install PyG stack (torch_geometric + extensions)? [y/N] " install_pyg
if [[ ${install_pyg} == [Yy]* ]]; then
  uv pip install --python .venv/bin/python torch-geometric
  uv pip install --python .venv/bin/python \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.0+${CUDA_VERSION}.html
fi

read -p "Install vision extras (diffusers/pillow/clean-fid/dali/accelerate)? [y/N] " install_vision
if [[ ${install_vision} == [Yy]* ]]; then
  uv sync --python .venv/bin/python --extra vision
fi

read -p "Install TensorFlow (meshgraphnet extra)? [y/N] " install_tf
if [[ ${install_tf} == [Yy]* ]]; then
  uv sync --python .venv/bin/python --extra meshgraphnet
fi

read -p "Install AM SDF extras (trimesh/rtree)? [y/N] " install_am_sdf
if [[ ${install_am_sdf} == [Yy]* ]]; then
  uv sync --python .venv/bin/python --extra am_sdf
fi

read -p "Install Flash Attention? [y/N] " install_flash_attn
if [[ ${install_flash_attn} == [Yy]* ]]; then
  uv pip install --python .venv/bin/python flash-attn --no-build-isolation
fi

read -p "Install LaTeX for publication-quality plots? (requires sudo) [y/N] " install_latex
if [[ ${install_latex} == [Yy]* ]]; then
  sudo apt update && sudo apt install -y \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    texlive-fonts-extra cm-super dvipng
fi

echo "Install complete. Activate with: source .venv/bin/activate"
