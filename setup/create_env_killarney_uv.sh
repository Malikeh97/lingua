set -e

echo "============================================"
echo "Creating Lingua environment with UV at $(date)"
echo "============================================"

# Load modules BEFORE creating/activating venv
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

mkdir -p logs

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "UV version: $(uv --version)"


# Install project with dependencies using uv sync
echo "Installing lingua with dependencies..."
uv sync

source .venv/bin/activate

echo "Python: $(which python)"

echo "============================================"
echo "Environment created successfully!"
echo "Activate with: source .venv/bin/activate"
echo "============================================"
