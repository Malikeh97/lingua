#!/bin/bash
#SBATCH --job-name=finesearch-ring-attn
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --account=aip-craffel
#SBATCH --output=outputs/finesearch_ring_attn/slurm-%j.out
#SBATCH --error=outputs/finesearch_ring_attn/slurm-%j.err

hostname

export OFFLINE_MODE=0
source setup/start_env.sh

torchrun --nproc_per_node=4 \
    -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/test_ring_attn.yaml \
    dump_dir=outputs/finesearch_ring_attn
