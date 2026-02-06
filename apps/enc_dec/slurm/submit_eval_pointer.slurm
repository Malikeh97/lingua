#!/bin/bash
#SBATCH --job-name=eval_pointer
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/extractive_pointer_modernbert_300M/logs/eval_%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/extractive_pointer_modernbert_300M/logs/eval_%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --account=aip-craffel
#SBATCH --time=08:00:00

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

# Set checkpoint path - UPDATE THIS TO YOUR POINTER MODEL CHECKPOINT
CKPT_DIR="$SCRATCH/extractive_pointer_modernbert_300M/checkpoints"
LATEST_STEP=$(ls -d ${CKPT_DIR}/0* 2>/dev/null | sort -V | tail -1)
CONSOLIDATED_CKPT="${LATEST_STEP}/consolidated/consolidated.pth"
CONFIG_FILE="${LATEST_STEP}/params.json"

# SQuAD dev file path
SQUAD_DEV_FILE="/home/ehghaghi/projects/aip-craffel/ehghaghi/lingua/apps/enc_dec/data/dev-v2.0.json"

# Output directory for results
OUTPUT_DIR="$SCRATCH/extractive_pointer_modernbert_300M/eval_results/$(basename $LATEST_STEP)"

# Percentage of dev data to evaluate (1-100)
EVAL_PERCENT=100

# Create log directory if it doesn't exist
mkdir -p "$SCRATCH/extractive_pointer_modernbert_300M/logs"

echo "============================================"
echo "Running Pointer Mechanism SQuAD Evaluation"
echo "Checkpoint: $CONSOLIDATED_CKPT"
echo "Config: $CONFIG_FILE"
echo "SQuAD Dev File: $SQUAD_DEV_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "Eval Percent: $EVAL_PERCENT%"
echo "============================================"

# Run evaluation on SQuAD dev set
python -u -m apps.enc_dec.eval_squad_pointer \
    --checkpoint "$CONSOLIDATED_CKPT" \
    --config "$CONFIG_FILE" \
    --data_file "$SQUAD_DEV_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --eval_percent "$EVAL_PERCENT" \
    --max_new_tokens 32 \
    --device cuda

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
