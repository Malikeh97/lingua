#!/bin/bash
#SBATCH --job-name=infer_modernbert_300M
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert_pretrained_dec_300M/logs/infer_%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert_pretrained_dec_300M/logs/infer_%x_%j.err
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
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch_extensions"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Create directories
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p $SCRATCH/tmp

cd /project/aip-craffel/ehghaghi/lingua

# Set checkpoint path - uses params.json saved with checkpoint (no external YAML needed)
CKPT_DIR="$SCRATCH/mvp_modernbert_pretrained_dec_300M/checkpoints"
# Find latest checkpoint step
LATEST_STEP=$(ls -d ${CKPT_DIR}/0* 2>/dev/null | sort -V | tail -1)
CONSOLIDATED_CKPT="${LATEST_STEP}/consolidated/consolidated.pth"
CONFIG_FILE="${LATEST_STEP}/params.json"

echo "============================================"
echo "Running Inference"
echo "Checkpoint: $CONSOLIDATED_CKPT"
echo "Config: $CONFIG_FILE"
echo "============================================"

# Run inference with SQuAD examples
python -m apps.enc_dec.infer \
    --checkpoint "$CONSOLIDATED_CKPT" \
    --config "$CONFIG_FILE" \
    --num_examples 5 \
    --max_new_tokens 64 \
    --device cuda

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
