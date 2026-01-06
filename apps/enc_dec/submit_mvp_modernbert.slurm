#!/bin/bash
#SBATCH --job-name=mvp_modernbert
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert/logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert/logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --account=aip-craffel
#SBATCH --time=04:00:00
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
export HF_HOME=$SCRATCH/.cache/huggingface
export TMPDIR="$SCRATCH/tmp"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=INFO

# Create directories
mkdir -p $SCRATCH/mvp_modernbert/logs
mkdir -p $SCRATCH/.cache/huggingface
mkdir -p $SCRATCH/tmp

cd /project/6104653/ehghaghi/lingua

echo "============================================"
echo "MVP: Frozen ModernBERT encoder + trainable decoder"
echo "Config: apps/enc_dec/configs/mvp_modernbert.yaml"
echo "============================================"

# Pre-download ModernBERT model (optional, helps with logging)
echo "Checking ModernBERT model availability..."
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base'); print('ModernBERT tokenizer ready')" 2>/dev/null || echo "Will download during training"

echo "============================================"
echo "Starting training at $(date)"
echo "============================================"

# Run training
torchrun --nproc-per-node 1 \
    --master-port 29500 \
    -m apps.enc_dec.train \
    config=apps/enc_dec/configs/mvp_modernbert.yaml

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
