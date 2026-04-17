# Addon Tests

## Test packed encoder (compare HF vs our implementation)

```bash
python -m apps.addon_tests.test_packed_encoder
```

## Test data pipeline (no model, dumps batches for inspection)

```bash
# Single task
python -m apps.addon_tests.test_data_pipeline \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_train.yaml \
    num_batches=5

# Multi-task
python -m apps.addon_tests.test_data_pipeline \
    config=apps/finesearch/configs/debug.yaml,apps/finesearch/configs/debug_multitask.yaml \
    num_batches=5
```
