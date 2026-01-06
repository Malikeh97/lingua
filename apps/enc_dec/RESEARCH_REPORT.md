# Encoder-Decoder QA System: Technical Report

**Application:** `apps/enc_dec`
**Built on:** Lingua Framework
**Author:** Malikeh Ehghaghi
**Date:** January 2026

---

## Executive Summary

This document provides a comprehensive overview of the encoder-decoder QA system built on top of Lingua. The system supports three encoder architectures (trainable transformer, pretrained HuggingFace models, embedding-only), enabling extensive ablation studies on encoder-decoder architectures for question answering tasks.

**Key Capabilities:**
- Flexible encoder architecture selection (trainable, pretrained, embedding-only)
- Support for frozen pretrained encoders (e.g., ModernBERT)
- Configurable model dimensions, depths, and attention mechanisms
- Distributed training with FSDP and tensor parallelism
- Step-based training with gradient accumulation
- Comprehensive checkpointing and resumption

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Configuration System](#2-configuration-system)
3. [Model Configurability](#3-model-configurability)
4. [Training Pipeline Options](#4-training-pipeline-options)
5. [Data Pipeline](#5-data-pipeline)
6. [Ablation Study Design Guide](#6-ablation-study-design-guide)
7. [Practical Usage](#7-practical-usage)
8. [File Reference](#8-file-reference)

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Encoder-Decoder QA Model                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                    ┌──────────────────────┐  │
│  │   ENCODER    │                    │       DECODER        │  │
│  │              │                    │                      │  │
│  │ Three modes: │    Encoder         │  ┌────────────────┐  │  │
│  │ • trainable  │    Hidden     ────►│  │ Cross-Attention│  │  │
│  │ • pretrained │    States          │  └────────────────┘  │  │
│  │ • embed_only │                    │         │            │  │
│  │              │                    │         ▼            │  │
│  │   Context    │                    │  Causal Self-Attn    │  │
│  │   Document   │                    │         │            │  │
│  └──────────────┘                    │         ▼            │  │
│                                      │       FFN            │  │
│                                      │         │            │  │
│                                      │         ▼            │  │
│                                      │  Output Projection   │  │
│                                      │  (Question→Answer)   │  │
│                                      └──────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Encoder Types

| Type | Description | Use Case | Parameters |
|------|-------------|----------|------------|
| `trainable` | Full transformer encoder trained from scratch | Custom domain adaptation | All trainable |
| `pretrained` | Frozen HuggingFace model (e.g., ModernBERT) | Leverage pretrained representations | Frozen (decoder trainable) |
| `embedding_only` | Embeddings + LayerNorm, no transformer layers | Ablation baseline, fast debugging | Minimal |

### 1.3 Data Flow

```
Input: (context_document, question, answer)
       ↓
Encoder: context_document → encoder_hidden_states [B, enc_seq, dim]
       ↓
Decoder: question + answer → output logits [B, dec_seq, vocab]
       ↓
Loss: Cross-entropy on answer tokens only (question masked with -100)
```

---

## 2. Configuration System

### 2.1 Configuration Hierarchy

```
Dataclass Defaults (Python)
        ↓
YAML Configuration File
        ↓
CLI Overrides
```

### 2.2 Available Configuration Files

| Config | Purpose | Total Params | Training Steps | GPU Memory |
|--------|---------|--------------|----------------|------------|
| `debug.yaml` | Quick local testing | ~10M | 100 | 8GB |
| `mvp_embedding_only.yaml` | Fast ablation baseline | ~50M | 500 | 16GB |
| `mvp_modernbert.yaml` | MVP with pretrained encoder | ~250M | 2,000 | 48GB |
| `enc_dec_base.yaml` | Production training | ~300M | 50,000 | 24GB+ |
| `smoke_test_1b.yaml` | Large scale validation | ~1.3B | 200 | 48GB |

### 2.3 Configuration Structure

```yaml
# Top-level configuration
name: str                    # Experiment name
dump_dir: str                # Output directory
seed: int                    # Random seed
steps: int                   # Total training steps
grad_acc_steps: int          # Gradient accumulation

model:                       # Model architecture
  dim: int                   # Hidden dimension
  encoder_type: str          # "trainable" | "pretrained" | "embedding_only"
  encoder: {...}             # Encoder-specific config
  decoder: {...}             # Decoder-specific config
  pretrained_encoder: {...}  # Pretrained encoder config
  weight_tying: bool         # Tie output/embedding weights
  share_embeddings: bool     # Share encoder/decoder embeddings

data:                        # Data configuration
  dataset_name: str          # HuggingFace dataset name
  max_encoder_len: int       # Max context length
  max_decoder_len: int       # Max question+answer length
  batch_size: int
  encoder_tokenizer_name: str  # Optional separate encoder tokenizer

optim:                       # Optimization
  lr: float                  # Learning rate
  warmup: int                # Warmup steps
  clip: float                # Gradient clipping

distributed:                 # Distributed training
  fsdp_type: str             # "full_shard" | "no_shard" | "hybrid"
  model_dtype: str           # "bf16" | "fp32"
  compile: bool              # torch.compile

checkpoint:                  # Checkpointing
  dump: {every: int}         # Training checkpoint frequency
  eval: {every: int}         # Evaluation checkpoint frequency
```

---

## 3. Model Configurability

### 3.1 Encoder Configuration

#### Trainable Encoder
```yaml
model:
  encoder_type: trainable
  encoder:
    n_layers: 6              # Number of transformer layers
    n_heads: 12              # Attention heads
    n_kv_heads: 8            # KV heads for GQA (null = MHA)
    head_dim: null           # Head dimension (auto: dim/n_heads)
    ffn_dim_multiplier: 1.5  # FFN hidden size multiplier
    multiple_of: 256         # Round FFN dim to multiple
    norm_eps: 1e-5           # RMSNorm epsilon
    rope_theta: 10000.0      # RoPE base frequency
    init_base_std: null      # Weight init std (auto-calculated)
    init_std_factor: "disabled"  # Init scaling strategy
```

#### Pretrained Encoder
```yaml
model:
  encoder_type: pretrained
  pretrained_encoder:
    model_name: "answerdotai/ModernBERT-base"  # HF model identifier
    encoder_dim: 768         # HF model's hidden dimension
    pooling: "none"          # "none" | "mean" | "cls"
    use_flash_attention: true
```

**Supported Pretrained Encoders:**
- `answerdotai/ModernBERT-base` (768D, 149M params)
- `answerdotai/ModernBERT-large` (1024D, 395M params)
- `bert-base-uncased` (768D)
- `roberta-base` (768D)
- Any HuggingFace encoder model

#### Embedding-Only Encoder
```yaml
model:
  encoder_type: embedding_only
  share_embeddings: true     # Use decoder's embeddings
```

### 3.2 Decoder Configuration

```yaml
model:
  decoder:
    n_layers: 6              # Number of transformer layers
    n_heads: 12              # Attention heads
    n_kv_heads: 8            # KV heads for GQA
    head_dim: null           # Head dimension
    ffn_dim_multiplier: 1.5  # FFN size multiplier
    multiple_of: 256
    norm_eps: 1e-5
    rope_theta: 10000.0
    init_base_std: null
    init_std_factor: "disabled"
```

### 3.3 Shared Configuration

```yaml
model:
  dim: 768                   # Shared hidden dimension (encoder output = decoder input)
  vocab_size: -1             # Auto-detect from tokenizer
  weight_tying: true         # Tie embedding and output projection weights
  share_embeddings: true     # Share embeddings between encoder and decoder
  max_encoder_seqlen: 2048   # Maximum encoder sequence length
  max_decoder_seqlen: 512    # Maximum decoder sequence length
```

### 3.4 Attention Mechanism Options

| Option | Description |
|--------|-------------|
| MHA (Multi-Head Attention) | `n_kv_heads: null` or `n_kv_heads == n_heads` |
| GQA (Grouped Query Attention) | `n_kv_heads < n_heads` (must divide evenly) |
| MQA (Multi-Query Attention) | `n_kv_heads: 1` |

---

## 4. Training Pipeline Options

### 4.1 Optimization Configuration

```yaml
optim:
  lr: 3e-4                   # Peak learning rate
  warmup: 2000               # Linear warmup steps
  lr_min_ratio: 0.000001     # Min LR ratio for cosine annealing
  clip: 1.0                  # Gradient clipping max norm
  weight_decay: 0.1          # AdamW weight decay
  beta1: 0.9                 # Adam beta1
  beta2: 0.95                # Adam beta2
  eps: 1e-8                  # Adam epsilon
```

### 4.2 Training Control

```yaml
steps: 50000                 # Total optimizer steps
grad_acc_steps: 4            # Gradient accumulation steps
gc_collect_freq: 1000        # Garbage collection frequency
```

**Effective Batch Size:** `batch_size × grad_acc_steps × world_size`

### 4.3 Distributed Training

```yaml
distributed:
  fsdp_type: full_shard      # FSDP strategy
  model_dtype: bf16          # Model precision
  compile: true              # Enable torch.compile
  selective_activation_checkpointing: true  # Memory optimization
```

**FSDP Strategies:**
| Strategy | Description | Use Case |
|----------|-------------|----------|
| `full_shard` | Full sharding across all ranks | Multi-GPU, memory constrained |
| `no_shard` | No sharding (data parallel) | Single GPU, small models |
| `hybrid` | TP within node, FSDP across nodes | Large scale training |

### 4.4 Checkpointing

```yaml
checkpoint:
  dump:
    every: 1000              # Save every N steps
    keep: 3                  # Keep last K checkpoints
  eval:
    every: 2500              # Evaluation checkpoint frequency
```

### 4.5 Logging

```yaml
logging:
  freq: 10                   # Log every N steps
```

**Logged Metrics:**
- `loss/out`: Training loss
- `speed/wps`: Words per second (throughput)
- `speed/FLOPS`: Estimated FLOPS
- `optim/grad_norm`: Gradient norm
- `optim/lr`: Current learning rate
- `optim/total_tokens`: Cumulative tokens processed
- `memory/max_active_pct`: GPU memory utilization

---

## 5. Data Pipeline

### 5.1 Dataset Configuration

```yaml
data:
  dataset_name: squad              # HuggingFace dataset name
  dataset_config: null             # Dataset configuration (if applicable)
  dataset_split: train             # train | validation | test
  max_samples: null                # Limit samples (null = use all)
  question_column: question        # Column name for questions
  answer_column: answers           # Column name for answers
  context_column: context          # Column name for context/documents
```

### 5.2 Sequence Lengths

```yaml
data:
  max_encoder_len: 2048            # Max context tokens
  max_decoder_len: 512             # Max question + answer tokens
```

### 5.3 Tokenization

```yaml
data:
  add_bos: true                    # Add BOS token
  add_eos: true                    # Add EOS token
  tokenizer:
    name: bytes                    # Tokenizer type
  encoder_tokenizer_name: null     # Optional: separate encoder tokenizer
```

**Dual Tokenizer Support:**
For pretrained encoders, use separate tokenizers:
```yaml
data:
  encoder_tokenizer_name: answerdotai/ModernBERT-base  # Encoder uses HF tokenizer
  tokenizer:
    name: bytes                    # Decoder uses lingua tokenizer
```

### 5.4 Data Loading

```yaml
data:
  batch_size: 8
  num_workers: 4                   # DataLoader workers
  prefetch_factor: 2               # Prefetch batches per worker
  seed: 42
```

### 5.5 Label Masking

Training labels use `-100` for question tokens (ignored in loss):

```
Decoder Input: [BOS] question tokens [SEP] answer tokens [EOS]
Labels:        [-100] [-100 × Q_len]  [SEP] answer tokens [EOS]
```

---

## 6. Ablation Study Design Guide

### 6.1 Encoder Architecture Ablations

| Ablation | Config Change | Research Question |
|----------|---------------|-------------------|
| Trainable vs. Pretrained | `encoder_type: trainable` vs `pretrained` | Does pretrained encoder help? |
| Encoder Depth | `encoder.n_layers: 2/4/6/12` | How deep should encoder be? |
| Encoder Width | `model.dim: 256/512/768/1024` | Impact of hidden dimension |
| Attention Type | `encoder.n_kv_heads: null/8/4/1` | MHA vs GQA vs MQA |
| Embedding Only | `encoder_type: embedding_only` | Is encoder even necessary? |

### 6.2 Decoder Architecture Ablations

| Ablation | Config Change | Research Question |
|----------|---------------|-------------------|
| Decoder Depth | `decoder.n_layers: 4/6/12/22` | Optimal decoder depth |
| Decoder Width | Varies with `model.dim` | Decoder capacity needs |
| Weight Tying | `weight_tying: true/false` | Does tying help smaller models? |
| Shared Embeddings | `share_embeddings: true/false` | Encoder-decoder embedding sharing |

### 6.3 Encoder-Decoder Asymmetry Ablations

| Config | Encoder | Decoder | Purpose |
|--------|---------|---------|---------|
| `enc_heavy` | 12 layers | 6 layers | Emphasize encoding |
| `dec_heavy` | 6 layers | 12 layers | Emphasize generation |
| `balanced` | 6 layers | 6 layers | Baseline |
| `minimal_enc` | embedding_only | 6 layers | Extreme decoder-focused |

**Example Configuration for Asymmetric Model:**
```yaml
model:
  dim: 768
  encoder:
    n_layers: 12
    n_heads: 12
  decoder:
    n_layers: 6
    n_heads: 12
```

### 6.4 Pretrained Encoder Ablations

| Ablation | Config Change | Research Question |
|----------|---------------|-------------------|
| Encoder Choice | `model_name: ModernBERT-base/large` | Which pretrained encoder? |
| Frozen vs. Fine-tuned | Custom code modification | Should encoder be frozen? |
| Pooling Strategy | `pooling: none/mean/cls` | How to aggregate encoder output? |
| Context Length | `max_encoder_len: 512/1024/2048/8192` | Impact of context window |

### 6.5 Training Ablations

| Ablation | Config Change | Research Question |
|----------|---------------|-------------------|
| Learning Rate | `optim.lr: 1e-4/3e-4/1e-3` | Optimal LR |
| Warmup | `optim.warmup: 500/2000/5000` | Warmup duration impact |
| Batch Size | `data.batch_size × grad_acc_steps` | Effective batch size |
| Training Steps | `steps: 10k/50k/100k` | Training duration |

### 6.6 Suggested Ablation Experiment Matrix

```
Experiment Set 1: Encoder Type Comparison
├── trainable_6L_768D
├── pretrained_modernbert_base
├── pretrained_modernbert_large
└── embedding_only_768D

Experiment Set 2: Encoder-Decoder Balance
├── enc6_dec6 (balanced)
├── enc12_dec6 (encoder heavy)
├── enc6_dec12 (decoder heavy)
└── enc2_dec12 (minimal encoder)

Experiment Set 3: Attention Mechanism
├── mha_baseline (n_kv_heads = n_heads)
├── gqa_8kv (n_kv_heads = 8)
├── gqa_4kv (n_kv_heads = 4)
└── mqa (n_kv_heads = 1)

Experiment Set 4: Scale Study
├── small (256D, 2+2 layers)
├── base (512D, 6+6 layers)
├── large (768D, 12+12 layers)
└── xlarge (1024D, 24+24 layers)
```

### 6.7 Running Ablations

**Single ablation run:**
```bash
python -m apps.enc_dec.train \
    config=configs/enc_dec_base.yaml \
    name=ablation_enc_depth_12 \
    model.encoder.n_layers=12 \
    dump_dir=outputs/ablations/enc_depth_12
```

**Sweep with different configs:**
```bash
# Encoder type sweep
for enc_type in trainable pretrained embedding_only; do
    python -m apps.enc_dec.train \
        config=configs/enc_dec_base.yaml \
        name=ablation_${enc_type} \
        model.encoder_type=${enc_type} \
        dump_dir=outputs/ablations/${enc_type}
done
```

---

## 7. Practical Usage

### 7.1 Quick Start Commands

**Debug (local, <10 min):**
```bash
python -m apps.enc_dec.train config=apps/enc_dec/configs/debug.yaml
```

**MVP with Pretrained Encoder (single GPU, ~4 hours):**
```bash
sbatch apps/enc_dec/submit_mvp_modernbert.slurm
```

**Production Training (multi-GPU):**
```bash
torchrun --nproc-per-node 8 -m apps.enc_dec.train \
    config=apps/enc_dec/configs/enc_dec_base.yaml \
    data.dataset_name=your_dataset
```

### 7.2 CLI Override Examples

```bash
# Change model dimension
python -m apps.enc_dec.train config=... model.dim=1024

# Change encoder depth
python -m apps.enc_dec.train config=... model.encoder.n_layers=12

# Use different dataset
python -m apps.enc_dec.train config=... data.dataset_name=natural_questions

# Limit training samples for debugging
python -m apps.enc_dec.train config=... data.max_samples=1000

# Change output directory
python -m apps.enc_dec.train config=... dump_dir=/path/to/outputs

# Multiple overrides
python -m apps.enc_dec.train config=... \
    model.dim=512 \
    model.encoder.n_layers=4 \
    model.decoder.n_layers=8 \
    steps=10000
```

### 7.3 Evaluation

```bash
python -m apps.enc_dec.eval \
    checkpoint_path=outputs/enc_dec/checkpoints/step_10000 \
    data.dataset_split=validation \
    generation.max_new_tokens=128 \
    generation.temperature=0.0  # Greedy decoding
```

### 7.4 Output Structure

```
dump_dir/
├── config.yaml              # Saved configuration
├── train.log                # Detailed training logs
├── metrics.jsonl            # Per-step metrics (JSON lines)
├── checkpoints/
│   ├── step_1000/           # Training checkpoint
│   ├── step_2000/
│   └── step_3000/
└── eval_checkpoints/
    ├── step_2500/           # Evaluation checkpoint
    └── step_5000/
```

---

## 8. File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `enc_dec.py` | Model architecture (encoder, decoder, cross-attention) | ~1020 |
| `data.py` | Data loading, tokenization, collation | ~369 |
| `train.py` | Training loop, checkpointing, distributed setup | ~562 |
| `eval.py` | Evaluation, generation, metrics | ~368 |
| `configs/*.yaml` | Configuration files | Various |
| `submit_*.slurm` | SLURM job submission scripts | ~50 each |

---

## Appendix A: Parameter Count Reference

| Configuration | Encoder | Decoder | Total | Trainable |
|---------------|---------|---------|-------|-----------|
| debug.yaml | ~5M | ~5M | ~10M | ~10M |
| mvp_embedding_only | ~10M | ~40M | ~50M | ~50M |
| mvp_modernbert | ~149M | ~100M | ~250M | ~100M (encoder frozen) |
| enc_dec_base | ~150M | ~150M | ~300M | ~300M |
| smoke_test_1b | ~300M | ~1B | ~1.3B | ~1.3B |

---

## Appendix B: Dependencies

- Python 3.10+
- PyTorch 2.1+
- Lingua framework
- transformers (for pretrained encoders)
- datasets (HuggingFace)
- OmegaConf
- xformers (optional, for memory-efficient attention)

---

## Appendix C: Quick Ablation Checklist

- [ ] Define research question
- [ ] Select baseline configuration
- [ ] Identify variables to ablate
- [ ] Create config variants (or use CLI overrides)
- [ ] Set consistent seeds for reproducibility
- [ ] Run experiments
- [ ] Compare metrics: loss, perplexity, EM, F1
- [ ] Analyze throughput/memory tradeoffs

---

*Report generated from codebase analysis. For implementation details, refer to the source files in `apps/enc_dec/`.*
