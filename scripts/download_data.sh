#!/bin/bash

#=========================================#
PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=${PROJ_DIR}/data

#=========================================#
# Redirect all caches to project space to avoid disk quota issues
#=========================================#
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
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

#=========================================#
# # Check if authenticated with Hugging Face
# echo "Checking Hugging Face authentication..."
# HF_STATUS=$(uv run hf whoami 2>&1)
# if echo "$HF_STATUS" | grep -q "Not logged in"; then
#     echo "❌ Not logged in to Hugging Face!"
#     echo "Please run: uv run hf login"
#     echo "You'll need a Hugging Face account and token from https://huggingface.co/settings/tokens"
#     exit 1
# fi

# echo "✅ Authenticated with Hugging Face"

#=========================================#
# Geo-FNO
#=========================================#
mkdir -p Geo-FNO
echo "Downloading Geo-FNO dataset..."
uv run hf download --repo-type dataset vedantpuri/PDESurrogates Geo-FNO.tar.gz --local-dir .
echo "Extracting Geo-FNO dataset..."
tar -xzf Geo-FNO.tar.gz
rm -rf Geo-FNO.tar.gz

#=========================================#
# FNO
#=========================================#
mkdir -p FNO
echo "Downloading FNO dataset..."
uv run hf download --repo-type dataset vedantpuri/PDESurrogates FNO.tar.gz --local-dir .
echo "Extracting FNO dataset..."
tar -xzf FNO.tar.gz
rm -rf FNO.tar.gz

#=========================================#
# SHAPENET-CAR
# https://github.com/ml-jku/UPT/blob/main/SETUP_DATA.md
#=========================================#
mkdir -p ShapeNet-Car
echo "Downloading ShapeNet-Car dataset..."
cd ShapeNet-Car
wget http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
unzip mlcfd_data.zip
rm -rf __MACOSX
cd mlcfd_data
cd training_data
tar -xvzf param0.tar.gz
tar -xvzf param1.tar.gz
tar -xvzf param2.tar.gz
tar -xvzf param3.tar.gz
tar -xvzf param4.tar.gz
tar -xvzf param5.tar.gz
tar -xvzf param6.tar.gz
tar -xvzf param7.tar.gz
tar -xvzf param8.tar.gz
# remove folders without quadpress_smpl.vtk
rm -rf ./param2/854bb96a96a4d1b338acbabdc1252e2f
rm -rf ./param2/85bb9748c3836e566f81b21e2305c824
rm -rf ./param5/9ec13da6190ab1a3dd141480e2c154d3
rm -rf ./param8/c5079a5b8d59220bc3fb0d224baae2a

rm -rf mlcfd_data.zip
rm -rf mlcfd_data/training_data/*tar.gz

cd $DATA_DIR

#=========================================#
# DRIVAERML
#=========================================#
mkdir -p DrivAerML
echo "Downloading and extracting DrivAerML dataset..."

cd DrivAerML

for num_points in 10k 40k 100k 200k 300k 400k 500k 1m; do
    uv run hf download --repo-type dataset vedantpuri/PDESurrogates drivaerml_surface_presampled_${num_points}.tar.gz --local-dir .
    tar -xzf drivaerml_surface_presampled_${num_points}.tar.gz
    rm -rf drivaerml_surface_presampled_${num_points}.tar.gz
done

#=========================================#
cd $DATA_DIR
#=========================================#
#