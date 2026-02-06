#!/bin/bash
# Setup script for enc-dec environment using uv
# Supports both local development and Compute Canada HPC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${ENV_NAME:-enc_dec_env}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Detect if running on Compute Canada
is_compute_canada() {
    [[ -d /cvmfs/soft.computecanada.ca ]] || [[ -n "$CC_CLUSTER" ]]
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_warn "uv not found. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    log_info "uv version: $(uv --version)"
}

# Setup for Compute Canada
setup_compute_canada() {
    log_info "Detected Compute Canada environment"

    # Load required modules
    log_info "Loading modules..."
    module load cuda/12.6
    module load gcc arrow/19.0.1 python/3.11

    # Default env location on scratch
    if [[ -z "$VIRTUAL_ENV" ]]; then
        ENV_PATH="${ENV_PATH:-$SCRATCH/envs/$ENV_NAME}"
    else
        ENV_PATH="$VIRTUAL_ENV"
    fi

    log_info "Environment path: $ENV_PATH"

    # Create virtual environment if it doesn't exist
    if [[ ! -d "$ENV_PATH" ]]; then
        log_info "Creating virtual environment..."
        uv venv "$ENV_PATH" --python 3.11
    fi

    # Activate environment
    source "$ENV_PATH/bin/activate"

    # Force all uv pip installs to target the venv, overriding any
    # project-level [tool.uv.pip] system=true that would write to /cvmfs
    local UV_PYTHON="$ENV_PATH/bin/python"

    # Install with Compute Canada optimizations
    log_info "Installing dependencies..."

    # Use Compute Canada's pre-built PyTorch wheels when available
    if [[ -f /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/python/3.11.5/lib/python3.11/site-packages/torch/__init__.py ]]; then
        log_info "Using system PyTorch from Compute Canada modules"
        # Install everything except torch
        uv pip install --python "$UV_PYTHON" -e "$PROJECT_DIR" --no-deps
        uv pip install --python "$UV_PYTHON" -r <(grep -v "^torch" "$PROJECT_DIR/pyproject.toml" 2>/dev/null || echo "")
    else
        # Install with PyTorch index for CUDA 12.6
        log_info "Installing PyTorch from pip..."
        uv pip install --python "$UV_PYTHON" torch --index-url https://download.pytorch.org/whl/cu126
        uv pip install --python "$UV_PYTHON" -e "$PROJECT_DIR"
    fi

    log_info "Installation complete!"
    log_info "Activate with: source $ENV_PATH/bin/activate"
}

# Setup for local development
setup_local() {
    log_info "Setting up local development environment"

    ENV_PATH="${ENV_PATH:-.venv}"

    # Create virtual environment
    if [[ ! -d "$ENV_PATH" ]]; then
        log_info "Creating virtual environment at $ENV_PATH..."
        uv venv "$ENV_PATH"
    fi

    # Activate
    source "$ENV_PATH/bin/activate"

    # Detect CUDA version and install appropriate PyTorch
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
        log_info "Detected NVIDIA driver: $CUDA_VERSION"

        # Install PyTorch with CUDA support
        log_info "Installing PyTorch with CUDA support..."
        uv pip install torch --index-url https://download.pytorch.org/whl/cu126
    else
        log_warn "No NVIDIA GPU detected, installing CPU-only PyTorch"
        uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install project dependencies
    log_info "Installing project dependencies..."
    uv pip install -e "$PROJECT_DIR"

    log_info "Installation complete!"
    log_info "Activate with: source $ENV_PATH/bin/activate"
}

# Main
main() {
    cd "$PROJECT_DIR"
    check_uv

    if is_compute_canada; then
        setup_compute_canada
    else
        setup_local
    fi

    # Verify installation
    log_info "Verifying installation..."
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    python -c "import transformers; print(f'Transformers {transformers.__version__}')"

    log_info "Setup complete!"
}

main "$@"
