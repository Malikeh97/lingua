# FineSearch: Training & Evaluation Commands

## Training (1 node, 4 GPUs)

```bash
torchrun --nproc_per_node=4 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_train.yaml
```

## Debug training (1 GPU, prints first batch tokenization)

```bash
DEBUG=1 torchrun --nproc_per_node=1 -m apps.finesearch.train \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_train.yaml
```

## Evaluation

```bash
python -m apps.finesearch.eval \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_eval.yaml \
    eval.ckpt_dir=outputs/finesearch_debug/checkpoints/0000005000
```

## Quick sanity check (1 GPU, 100 examples, DEBUG tokenization)

```bash
DEBUG=1 python -m apps.finesearch.eval \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_eval.yaml \
    eval.ckpt_dir=outputs/finesearch_debug/checkpoints/0000005000 \
    eval.max_samples=100
```
