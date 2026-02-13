#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=compute
#SBATCH --account=def-craffel

# Show node name
hostname

# Set offline mode
export OFFLINE_MODE=1

# Source start env
source setup/start_env.sh

# Execute the command passed as argument
eval "$@"