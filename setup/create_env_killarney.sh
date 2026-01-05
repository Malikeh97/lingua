#!/bin/bash
# Create lingua environment for Killarney cluster

#SBATCH --job-name=lingua_gpu_env_setup
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/logs/lingua_gpu_env_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/logs/lingua_gpu_env_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --account=aip-craffel
#SBATCH --time=01:00:00


set -e

echo "============================================"
echo "Creating Lingua environment at $(date)"
echo "============================================"

# Load modules BEFORE creating/activating venv
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

# Set paths
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export ENV_DIR="$SCRATCH/envs/lingua_gpu_env"
export TMPDIR="$SCRATCH/tmp"

mkdir -p $SCRATCH/logs
mkdir -p $SCRATCH/envs
mkdir -p $TMPDIR

# Create venv if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment at $ENV_DIR..."
    python -m venv --system-site-packages "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch and xformers
pip install torch==2.7.0 xformers

# Install ninja for faster builds
pip install ninja

# Install lingua requirements (exclude pyarrow - provided by arrow module)
echo "Installing lingua requirements..."
grep -v "pyarrow" /project/aip-craffel/ehghaghi/lingua/requirements.txt | pip install -r /dev/stdin

# Install additional dependencies for enc_dec
pip install --no-build-isolation datasets transformers

echo "============================================"
echo "Environment created successfully!"
echo "Activate with: source $ENV_DIR/bin/activate"
echo "============================================"
