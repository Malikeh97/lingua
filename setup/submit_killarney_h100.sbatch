#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --partition=gpubase_h100_b1
#SBATCH --account=aip-craffel

# Show node name
hostname

# Set offline mode
export OFFLINE_MODE=0

# Source start env
source setup/start_env.sh

# Execute the command passed as argument
eval "$@"