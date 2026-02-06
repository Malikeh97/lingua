#!/bin/bash
# Compute Canada specific setup for enc-dec with uv
# This script is optimized for the Compute Canada HPC environment
#
# Usage:
#   ./scripts/setup_cc.sh              # Uses default env location
#   ENV_PATH=/path/to/env ./scripts/setup_cc.sh  # Custom location

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LINGUA_DIR="$(dirname "$(dirname "$PROJECT_DIR")")"

# Default environment path on scratch (faster I/O)
ENV_PATH="${ENV_PATH:-$SCRATCH/envs/lingua_uv}"

echo "=== Compute Canada enc-dec Setup ==="
echo "Project: $PROJECT_DIR"
echo "Lingua:  $LINGUA_DIR"
echo "Env:     $ENV_PATH"
echo ""

# Load required modules
echo "Loading modules..."
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Create virtual environment
if [[ ! -d "$ENV_PATH" ]]; then
    echo "Creating virtual environment..."
    uv venv "$ENV_PATH" --python 3.11
fi

# Activate
source "$ENV_PATH/bin/activate"

# Force all uv pip installs to target the venv, overriding any
# project-level [tool.uv.pip] system=true that would write to /cvmfs
UV_PYTHON="$ENV_PATH/bin/python"

# Install PyTorch with CUDA 12.6 support
echo "Installing PyTorch..."
uv pip install --python "$UV_PYTHON" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install xformers (CUDA-specific)
echo "Installing xformers..."
uv pip install --python "$UV_PYTHON" xformers --index-url https://download.pytorch.org/whl/cu126 || \
    uv pip install --python "$UV_PYTHON" xformers

# Install flash-attn (requires CUDA)
echo "Installing flash-attention..."
uv pip install --python "$UV_PYTHON" flash-attn --no-build-isolation || \
    echo "Warning: flash-attn installation failed (may need to build from source)"

# Install project in editable mode
echo "Installing enc-dec project..."
cd "$PROJECT_DIR"
uv pip install --python "$UV_PYTHON" -e .

# Install lingua package from parent
echo "Installing lingua package..."
cd "$LINGUA_DIR"
uv pip install --python "$UV_PYTHON" -e . 2>/dev/null || uv pip install --python "$UV_PYTHON" -r requirements.txt

# Additional useful packages for training
echo "Installing additional packages..."
uv pip install --python "$UV_PYTHON" \
    deepspeed \
    peft \
    trl \
    evaluate \
    einops

echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import xformers; print(f'xformers: {xformers.__version__}')" 2>/dev/null || echo "xformers: not available"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate this environment:"
echo "  module load cuda/12.6"
echo "  module load gcc arrow/19.0.1 python/3.11"
echo "  source $ENV_PATH/bin/activate"
echo ""
echo "Add to your SLURM scripts:"
echo "  source $ENV_PATH/bin/activate"
