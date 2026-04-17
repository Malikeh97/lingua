#!/bin/bash
#SBATCH --job-name=finesearch-multitask-perf
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --account=aip-craffel
#SBATCH --output=outputs/finesearch_multitask_perf/slurm-%j.out
#SBATCH --error=outputs/finesearch_multitask_perf/slurm-%j.err

hostname

export OFFLINE_MODE=0
source setup/start_env.sh

# Train
torchrun --nproc_per_node=4 \
    -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/test_multitask_perf.yaml \
    dump_dir=outputs/finesearch_multitask_perf

# Eval on squad
python -m apps.finesearch.eval \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_eval.yaml \
    ckpt_dir=outputs/finesearch_multitask_perf/checkpoints/0000005000 \
    dump_dir=outputs/finesearch_multitask_perf/eval
