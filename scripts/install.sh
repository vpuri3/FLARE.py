#!/bin/bash
#=========================================#
# # Eagle
# TORCH_VERSION=2.6
# CUDA_VERSION=cu124
#=========================================#
# GCloud H100

TORCH_VERSION=2.8
CUDA_VERSION=cu128

# Personal Servers
# TORCH_VERSION=2.8
# CUDA_VERSION=cu129

#=========================================#
# Redirect all caches to project space to avoid disk quota issues
#=========================================#
PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY_CACHE_BASE="${PROJ_DIR}/../cache"

echo "Setting cache base to: ${PY_CACHE_BASE}"

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

mkdir -p "${PY_CACHE_BASE}"
mkdir -p "${PIP_CACHE_DIR}"
mkdir -p "${UV_CACHE_DIR}"
mkdir -p "${XDG_CACHE_HOME}"
mkdir -p "${TORCH_HOME}"
mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${DATASETS_CACHE}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${HF_HUB_CACHE}"

#=========================================#
# Update uv
#=========================================#
echo "Updating uv..."
uv self update

set -e  # Exit on any error

#=========================================#
# Create virtual environment
#=========================================#
echo "#------------------------#"
echo "Cleaning up previous installation..."
echo "#------------------------#"
rm -rf .venv uv.lock main.py pyproject.toml

echo "#------------------------#"
echo "Initializing project..."
echo "#------------------------#"
uv init --name flare --python=3.11 --no-readme
uv python install

echo "#------------------------#"
echo "Creating virtual environment..."
echo "#------------------------#"
uv venv

#=========================================#
# Install packages
#=========================================#

echo "#------------------------#"
echo "Installing PyTorch with CUDA support..."
echo "#------------------------#"
uv pip install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

echo "#------------------------#"
echo "Installing torch_geometric with PyTorch ${TORCH_VERSION} + ${CUDA_VERSION} compatibility..."
echo "#------------------------#"
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.0+${CUDA_VERSION}.html

echo "#------------------------#"
echo "Installing core packages from PyPI..."
echo "#------------------------#"
uv pip install triton einops
uv pip install timm datasets webdataset transformers
uv pip install tqdm jsonargparse setuptools packaging
uv pip install scipy pandas seaborn pyvista matplotlib
uv pip install diffusers pillow accelerate
uv pip install clean-fid
uv pip install nvidia-dali-cuda120


# Interactive tools
echo "#------------------------#"
echo "Installing interactive tools..."
echo "#------------------------#"
uv pip install ipython gpustat

# Testing tools
echo "#------------------------#"
echo "Installing testing tools..."
echo "#------------------------#"
uv pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-randomly

# Misc packages
echo "#------------------------#"
echo "Installing additional dataset processing packages..."
echo "#------------------------#"
uv pip install meshio
uv pip install open3d

# Meshgraphnet datasets
read -p "Install TensorFlow for Meshgraphnet datasets? [y/N] " install_tensorflow
if [[ $install_tensorflow == [Yy]* ]]; then
    echo "Installing TensorFlow..."
    uv pip install tensorflow
    echo "TensorFlow installation completed."
else
    echo "Skipping TensorFlow installation."
fi

# AM SDF stuff
read -p "Install 3D mesh processing libraries (trimesh, rtree) for AM SDF functionality? [y/N] " install_mesh_libs
if [[ $install_mesh_libs == [Yy]* ]]; then
    echo "Installing 3D mesh processing libraries..."
    uv pip install trimesh rtree
    echo "3D mesh processing libraries installation completed."
else
    echo "Skipping 3D mesh processing libraries installation."
fi

# Flash Attention with proper build configuration
read -p "Install Flash Attention for faster transformer models? [y/N] " install_flash_attn
if [[ $install_flash_attn == [Yy]* ]]; then
    echo "Installing Flash Attention..."
    uv pip install flash-attn --no-build-isolation
    echo "Flash Attention installation completed."
else
    echo "Skipping Flash Attention installation."
fi

# LaTeX for high-quality plots (requires sudo access)
read -p "Install LaTeX for publication-quality plots? (requires sudo access) [y/N] " install_latex
if [[ $install_latex == [Yy]* ]]; then
    echo "Installing LaTeX packages..."
    sudo apt update && sudo apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
    sudo apt install -y texlive-fonts-extra texlive-latex-extra cm-super
    sudo apt install -y dvipng
    echo "LaTeX installation completed."
else
    echo "Skipping LaTeX installation."
fi

echo "#------------------------#"
echo "Installation completed successfully!"
echo "Activate the environment with: source .venv/bin/activate"
echo "#------------------------#"
#
