#!/bin/bash
#SBATCH --job-name=finesearch-multinode
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --account=aip-craffel
#SBATCH --output=outputs/finesearch_multinode/slurm-%j.out
#SBATCH --error=outputs/finesearch_multinode/slurm-%j.err

hostname

export OFFLINE_MODE=0
source setup/start_env.sh

# Multi-node torchrun
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/test_multinode.yaml \
    dump_dir=outputs/finesearch_multinode
