#!/bin/sh
#=========================================#
CWD=$(pwd)
DATADIR=$CWD/data
export HF_HOME=$DATADIR/huggingface

#=========================================#
# Check if authenticated with Hugging Face
#=========================================#
echo "Checking Hugging Face authentication..."
HF_STATUS=$(uv run hf whoami 2>&1)
if echo "$HF_STATUS" | grep -q "Not logged in"; then
    echo "❌ Not logged in to Hugging Face!"
    echo "Please run: uv run hf login"
    echo "You'll need a Hugging Face account and token from https://huggingface.co/settings/tokens"
    exit 1
fi

echo "✅ Authenticated with Hugging Face"

#=========================================#
mkdir -p data
cd data

#=========================================#
# MODELNET40
#=========================================#
mkdir -p modelnet40_normal_resampled
echo "Downloading ModelNet40 dataset..."
uv run hf download --repo-type dataset Pointcept/modelnet40_normal_resampled-compressed modelnet40_normal_resampled.tar.gz --local-dir .
echo "Extracting ModelNet40 dataset..."
tar -xzf modelnet40_normal_resampled.tar.gz -C modelnet40_normal_resampled/

#=========================================#
# SCANNET
#=========================================#
mkdir -p scannet
echo "Downloading ScanNet dataset..."
uv run hf download --repo-type dataset Pointcept/scannet-compressed scannet.tar.gz --local-dir .
echo "Extracting ScanNet dataset..."
tar -xzf scannet.tar.gz -C scannet/

#=========================================#
# S3DIS
#=========================================#
mkdir -p s3dis
echo "Downloading S3DIS dataset..."
uv run hf download --repo-type dataset Pointcept/s3dis-compressed s3dis.tar.gz --local-dir .
echo "Extracting S3DIS dataset..."
tar -xzf s3dis.tar.gz -C s3dis/

# #=========================================#
# # NUSCENES
# #=========================================#
# mkdir -p nuscenes
# echo "Downloading Nuscenes dataset..."
# uv run hf download --repo-type dataset Pointcept/nuscenes-compressed nuscenes.tar.gz --local-dir .
# echo "Extracting Nuscenes dataset..."
# tar -xzf nuscenes.tar.gz -C nuscenes/

#