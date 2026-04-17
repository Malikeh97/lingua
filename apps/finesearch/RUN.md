# FineSearch: Training & Evaluation Commands

## Training (1 node, 4 GPUs)

```bash
torchrun --nproc_per_node=4 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_train.yaml
```

## Multi-task training (1 node, 4 GPUs)

```bash
torchrun --nproc_per_node=4 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_multitask.yaml
```

## FineInstructions-only training (1 GPU)

```bash
torchrun --nproc_per_node=1 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_fineinstructions.yaml
```

## Debug training (1 GPU, prints batch stats)

```bash
DEBUG=1 torchrun --nproc_per_node=1 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_train.yaml
```

```bash
DEBUG=1 torchrun --nproc_per_node=1 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_multitask.yaml
```

```bash
DEBUG=1 torchrun --nproc_per_node=1 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_fineinstructions.yaml
```

## Evaluation

```bash
python -m apps.finesearch.eval \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_eval.yaml \
    ckpt_dir=outputs/finesearch_debug/checkpoints/0000005000
```

## Quick sanity check (1 GPU, 100 examples, DEBUG tokenization)

```bash
DEBUG=1 python -m apps.finesearch.eval \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_eval.yaml \
    ckpt_dir=outputs/finesearch_debug/checkpoints/0000005000 \
    max_samples=100
```

---

## Verification Tests (sbatch)

### 1. Multi-node training (2 nodes × 4 GPUs)

```bash
sbatch apps/finesearch/sbatch/multinode.sbatch
```

### 2. Multi-node checkpointing (train 250 → checkpoint → resume → 500)

```bash
sbatch apps/finesearch/sbatch/multinode_checkpoint.sbatch
```

### 3. Ring attention (1 node × 4 GPUs, sp_size=4)

```bash
sbatch apps/finesearch/sbatch/ring_attn.sbatch
```

### 4. Multi-task performance (1 node × 4 GPUs, train + eval)

```bash
sbatch apps/finesearch/sbatch/multitask_perf.sbatch
```
