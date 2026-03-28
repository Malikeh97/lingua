# FineSearch: Long-Context Encoder-Decoder Project Plan

## Directory Structure

```
lingua/
├── addons/
│   ├── models/                        # Model components
│   │   ├── config.py                  # ModelArgs, GenerationArgs
│   │   ├── packed_attention.py        # BlockDiagonalMask-based, no padding
│   │   ├── cross_attention.py         # Gated cross-attention adapter
│   │   ├── encoder.py                 # Wraps HF encoders (ModernBERT, etc.)
│   │   ├── decoder.py                 # Decoder with cross-attn injection
│   │   ├── encoder_decoder.py         # Full model
│   │   └── ring_attention.py          # Sequence parallelism
│   │
│   ├── data/                          # Data pipeline
│   │   ├── collate.py                 # PackedSequences, TokenizedBatch
│   │   └── ray_pipeline.py            # DatasetReader, Mixer, Packer, PipelineConfig
│   │
│   └── tasks/                         # Task = schema + reading + mapping + eval
│       ├── schema.py                  # ContextBasedExample, BatchedExamples
│       ├── base.py                    # BaseTask (init_state, read, map, eval)
│       ├── registry.py                # Task registry
│       ├── metrics.py                 # Metric functions
│       └── <task>.py                  # Per-task implementations
│
├── apps/
│   └── finesearch/
│       ├── train.py
│       ├── eval.py
│       ├── infer.py
│       └── configs/
```

## Architecture Overview

### Model
- Write custom encoder-decoder, can reuse from cepe
- Decoder inherited from pretrained model
- FSDP for model parallelism
- Ring attention for sequence parallelism

### Data Pipeline (Ray-based)

| Component | Type | Why |
|-----------|------|-----|
| DatasetReader | Actor | Holds task state, calls task.read() |
| Mixer | Actor | RNG state for sampling by weights |
| Packer | Actor | Buffer state for bin-packing |

```
DatasetReaders ──► Mixer Actor ──► Packer Actor ──► Training
(per source)       (sampling)      (bin-pack)
```

### Tasks (Functional State-Passing)

Tasks define:
- `init_state(split, **kwargs)` → state dict (task decides contents)
- `read(state, batch_size)` → (examples, next_state)
- `map_example(raw)` → ContextBasedExample
- `evaluate(predictions, examples)` → metrics

State examples:
- Small dataset: `{"dataset": <ref>, "idx": 0, "epoch": 0}`
- Streaming: `{"iterator": <iter>, "buffer": [...]}`
- With doc cache: `{"idx": 0, "doc_cache": {...}}`

### Trainer
- Built upon lingua components (FSDP, checkpoint, optim)
- Reference: apps/main/train.py

### Evaluation
- Short term: model.generate()
- Long term: SGLang integration

## Layer Responsibilities

| Layer | Rules | Contains |
|-------|-------|----------|
| `addons/models/` | Pure PyTorch, supports FSDP + ring attention | Encoder, decoder, cross-attention |
| `addons/data/` | Ray-based pipeline, tokenization | Collation, Ray actors |
| `addons/tasks/` | HF datasets, schema, metrics | Schema, data loading, format mapping, evaluation |
| `apps/finesearch/` | Entry points | train.py, eval.py, infer.py |

## Key Design Decisions

| Aspect | Decision |
|--------|----------|
| Variable-length handling | Packed sequences + BlockDiagonalMask |
| Document deduplication | Per-batch, hash-based |
| Encoder | Pretrained (ModernBERT, etc.) |
| Decoder | Inherited from pretrained, add cross-attention |
| Cross-attention | Gated injection |
| Model parallelism | FSDP |
| Sequence parallelism | Ring attention |
| Data pipeline | Ray actors with functional state-passing |
| Task reading | (state) → (examples, next_state) |

## Data Flow

```
Task.init_state(split)
        │
        ▼
DatasetReader Actor (holds state)
        │
        ▼ Task.read(state, n)
(examples, next_state)
        │
        ▼ sample by weights
Mixer Actor
        │
        ▼ bin-pack to target tokens
Packer Actor
        │
        ▼ batch ready
Training
        │
        ├── Encode documents (ring attention if needed)
        ├── Cross-attention to encoder output
        └── Decode with loss
```

## Checkpointing

```
checkpoint = {
    "model": model_state_dict,
    "optimizer": optimizer_state_dict,
    "train_state": {"step": N, ...},
    "data_state": {
        "readers": {name: task_state, ...},
        "mixer": {"rng_state": ..., "counts": ...},
        "packer": {"buffer": ..., "batches_emitted": ...},
    }
}
```

Coordination: Training-driven or epoch boundaries.

## Open Questions

1. Encoder architecture: ModernBERT (8k) vs. alternatives?
2. Cross-attention frequency: every layer or every N layers?
3. Linear attention for encoder? (simplifies ring attention)
4. Data mixing ratios for pretraining vs. post-training?
