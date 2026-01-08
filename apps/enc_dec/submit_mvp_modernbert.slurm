#!/bin/bash
#SBATCH --job-name=mvp_modernbert
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert/logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert/logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --account=aip-craffel
#SBATCH --time=03:00:00
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
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" 
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$SCRATCH/.cache"


export TMPDIR="$SCRATCH/tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=INFO

# Create directories
mkdir -p $SCRATCH/mvp_modernbert/logs
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p $SCRATCH/tmp
mkdir -p $SCRATCH/tmp/triton
mkdir -p $SCRATCH/tmp/torch_extensions

cd /project/aip-craffel/ehghaghi/lingua

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
