#!/bin/bash
#SBATCH --job-name=infer_custom
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert_pretrained_dec_300M/logs/infer_%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/mvp_modernbert_pretrained_dec_300M/logs/infer_%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --account=aip-craffel
#SBATCH --time=00:30:00

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

# Set checkpoint path
CKPT_DIR="$SCRATCH/mvp_modernbert_pretrained_dec_300M/checkpoints"
LATEST_STEP=$(ls -d ${CKPT_DIR}/0* 2>/dev/null | sort -V | tail -1)
CONSOLIDATED_CKPT="${LATEST_STEP}/consolidated/consolidated.pth"
CONFIG_FILE="${LATEST_STEP}/params.json"

# Custom document and question
DOCUMENT="At the 52nd Annual Grammy Awards, Beyoncé received ten nominations, including Album of the Year for I Am... Sasha Fierce, Record of the Year for "Halo", and Song of the Year for \"Single Ladies (Put a Ring on It)\", among others. She tied with Lauryn Hill for most Grammy nominations in a single year by a female artist. In 2010, Beyoncé was featured on Lady Gaga's single "Telephone" and its music video. The song topped the US Pop Songs chart, becoming the sixth number-one for both Beyoncé and Gaga, tying them with Mariah Carey for most number-ones since the Nielsen Top 40 airplay chart launched in 1992. \"Telephone\" received a Grammy Award nomination for Best Pop Collaboration with Vocals."

QUESTION="How many awards was Beyonce nominated for at the 52nd Grammy Awards?"

echo "============================================"
echo "Running Custom Inference"
echo "Checkpoint: $CONSOLIDATED_CKPT"
echo "Config: $CONFIG_FILE"
echo "============================================"
echo "Document: $DOCUMENT"
echo "Question: $QUESTION"
echo "============================================"

# Run inference with custom document and question
python -m apps.enc_dec.infer \
    --checkpoint "$CONSOLIDATED_CKPT" \
    --config "$CONFIG_FILE" \
    --document "$DOCUMENT" \
    --question "$QUESTION" \
    --max_new_tokens 64 \
    --device cuda

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
