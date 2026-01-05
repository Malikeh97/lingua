#!/bin/bash
#SBATCH --job-name=enc_dec_smoke_1b
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/enc_dec_smoke_1b/logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/enc_dec_smoke_1b/logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --account=aip-craffel
#SBATCH --time=02:00:00
#SBATCH --signal=USR2@120

echo "============================================"
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"
echo "============================================"
nvidia-smi

# Load modules
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

# Activate lingua environment
source /home/ehghaghi/scratch/ehghaghi/envs/lingua_env/bin/activate

# Set environment variables
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
export TMPDIR="$SCRATCH/tmp"
export OMP_NUM_THREADS=1 # Limit CPU threads for better performance
export MKL_NUM_THREADS=1 # Limit MKL threads for better performance
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # Avoid NCCL record streams issues
export NCCL_DEBUG=INFO # Enable NCCL debug info

# Create directories
mkdir -p $SCRATCH/enc_dec_smoke_1b/logs
mkdir -p $SCRATCH/.cache
mkdir -p $SCRATCH/tmp

cd /project/6104653/ehghaghi/lingua

echo "============================================"
echo "Starting training at $(date)"
echo "Config: apps/enc_dec/configs/smoke_test_1b.yaml"
echo "============================================"

# Run training with torchrun (4 GPUs on single node)
torchrun --nproc-per-node 1 \
    --master-port 29500 \
    -m apps.enc_dec.train \
    config=apps/enc_dec/configs/smoke_test_1b.yaml

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
