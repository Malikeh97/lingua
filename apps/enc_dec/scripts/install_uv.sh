#!/bin/bash
# Quick uv installation for enc-dec project
# Usage: ./scripts/install_uv.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== enc-dec environment setup with uv ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

cd "$PROJECT_DIR"

# Create venv and sync
echo "Creating virtual environment and installing dependencies..."
uv venv .venv --python 3.11 2>/dev/null || uv venv .venv

# Install with PyTorch CUDA support
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .

echo ""
echo "=== Installation complete ==="
echo "Activate with: source .venv/bin/activate"
echo ""

# Quick verification
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
