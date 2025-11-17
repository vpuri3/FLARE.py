#!/bin/bash
#=========================================#
# # Eagle
# TORCH_VERSION=2.6
# CUDA_VERSION=cu124
#=========================================#
# # GCloud H100
# TORCH_VERSION=2.7
# CUDA_VERSION=cu128
#=========================================#
# Orchard
TORCH_VERSION=2.6
CUDA_VERSION=cu124

#=========================================#
# Update uv
#=========================================#
echo "Updating uv..."
uv self update

set -e  # Exit on any error

#=========================================#
# Create virtual environment
#=========================================#
echo "Cleaning up previous installation..."
rm -rf .venv uv.lock main.py pyproject.toml

echo "Initializing project..."
uv init --name flare --python=3.11 --no-readme

echo "Creating virtual environment..."
uv venv

#=========================================#
# Install packages
#=========================================#

# Install PyTorch first directly into virtual environment to avoid dependency conflicts
echo "Installing PyTorch with CUDA support..."
uv pip install torch==${TORCH_VERSION} torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Now add other packages from PyPI (this will use default PyPI index)
echo "Installing core packages from PyPI..."
uv add torch-geometric timm datasets
uv add tqdm jsonargparse einops setuptools packaging
uv add scipy pandas seaborn pyvista matplotlib
# uv add causal_conv1d mamba_ssm

# Interactive tools
echo "Installing interactive tools..."
uv add ipython gpustat

# Misc packages
echo "Installing additional packages for dataset management..."
uv add meshio
uv add open3d

echo "Installation completed successfully!"
echo "Activate the environment with: source .venv/bin/activate"

# # Meshgraphnet datasets
# uv add tensorflow

# # AM SDF stuff
# uv add trimesh rtree

# Flash Attention with proper build configuration
read -p "Install Flash Attention for faster transformer models? [y/N] " install_flash_attn
if [[ $install_flash_attn == [Yy]* ]]; then
    echo "Installing Flash Attention..."
    uv add flash-attn==2.8.1 --no-build-isolation
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
#