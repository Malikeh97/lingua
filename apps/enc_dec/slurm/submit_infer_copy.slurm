#!/bin/bash
#SBATCH --job-name=infer_copy
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/extractive_copy_modernbert_300M/logs/infer_%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/extractive_copy_modernbert_300M/logs/infer_%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --account=aip-craffel
#SBATCH --time=01:00:00

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

mkdir -p $SCRATCH/extractive_copy_modernbert_300M/logs
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p $SCRATCH/tmp

cd /project/aip-craffel/ehghaghi/lingua

# Set checkpoint path - UPDATE THIS to your actual checkpoint
CHECKPOINT_DIR="$SCRATCH/extractive_copy_modernbert_300M/checkpoints"
# Find latest checkpoint
LATEST_STEP=$(ls -d ${CHECKPOINT_DIR}/0* 2>/dev/null | sort -V | tail -n1)

if [ -z "$LATEST_STEP" ]; then
    echo "ERROR: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

CHECKPOINT="${LATEST_STEP}/consolidated/consolidated.pth"
CONFIG="${LATEST_STEP}/params.json"

echo "============================================"
echo "Copy Mechanism Inference"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "============================================"

# Run inference
python -m apps.enc_dec.infer_copy \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --num_examples 10 \
    --max_new_tokens 64 \
    --temperature 1.0

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
