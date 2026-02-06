# Encoder-Decoder Training Mechanism

This document describes the training mechanism for the encoder-decoder transformer model used for question-answering tasks.

## Overview

The training system implements a sequence-to-sequence model where:
- **Encoder**: Processes context/document tokens (bidirectional attention)
- **Decoder**: Generates answers autoregressively with cross-attention to encoder outputs

## Architecture

### Model Components

```
EncDecTransformer
├── Encoder (one of three types)
│   ├── Trainable: Full transformer encoder with bidirectional self-attention
│   ├── Pretrained: Frozen HuggingFace model (e.g., ModernBERT) with projection
│   └── EmbeddingOnly: Just embeddings, no transformer layers
│
└── Decoder
    ├── Token Embeddings
    ├── N x DecoderBlock
    │   ├── Causal Self-Attention (with RoPE)
    │   ├── Cross-Attention to Encoder
    │   └── Feed-Forward Network (SwiGLU)
    ├── RMSNorm
    └── Output Projection
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `EncDecTransformer` | `enc_dec.py:1061` | Main model combining encoder and decoder |
| `Encoder` | `enc_dec.py:474` | Trainable bidirectional transformer encoder |
| `PretrainedEncoder` | `enc_dec.py:580` | Frozen HuggingFace encoder with projection |
| `Decoder` | `enc_dec.py:784` | Autoregressive decoder with cross-attention |
| `DecoderBlock` | `enc_dec.py:388` | Self-attn + cross-attn + FFN |
| `CrossAttention` | `enc_dec.py:185` | Q from decoder, K/V from encoder |

## Training Pipeline

### 1. Data Flow

```
Input:
  - context (document text)
  - question
  - answer

Tokenization:
  encoder_input = tokenize(context)        # [B, enc_seq]
  decoder_input = tokenize(question + answer)[:-1]  # [B, dec_seq-1]
  labels = tokenize(question + answer)[1:]  # [B, dec_seq-1]
           with question tokens masked to -100

Forward Pass:
  encoder_output = encoder(encoder_input)  # [B, enc_seq, D]
  loss = decoder(decoder_input, encoder_output, labels)
```

### 2. Training Loop (`train.py:224`)

```python
while should_continue_training(step, epoch, args):
    # 1. Get batch
    batch = next(data_loader)

    # 2. Move to GPU
    encoder_input_ids = batch["encoder_input_ids"].cuda()
    decoder_input_ids = batch["decoder_input_ids"].cuda()
    labels = batch["labels"].cuda()
    encoder_padding_mask = batch["encoder_padding_mask"].cuda()

    # 3. Forward pass
    loss = model(
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        decoder_target=labels,
        encoder_padding_mask=encoder_padding_mask,
    )

    # 4. Backward pass with gradient accumulation
    loss = loss / args.grad_acc_steps
    loss.backward()

    # 5. Optimizer step (every grad_acc_steps)
    if train_state.acc_step == 0:
        grad_norm = clip_grad_norm_(model.parameters(), max_norm=args.optim.clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_state.step += 1
```

### 3. Loss Computation

The loss is computed using cross-entropy with label masking:

```python
# In Decoder.forward() at enc_dec.py:886
def forward(self, input_ids, encoder_output, target=None, ...):
    # ... transformer layers ...
    logits = self.output(self.norm(h))  # [B, dec_seq, vocab_size]

    if target is not None:
        # cross_entropy ignores positions where target == -100
        return cross_entropy(logits, target, ignore_index=-100)
    return logits
```

**Label Masking Strategy:**
- Question tokens: labeled as `-100` (ignored in loss)
- Answer tokens: actual token IDs (contribute to loss)
- This ensures the model only learns to predict the answer, not the question.

## Encoder Types

### 1. Trainable Encoder (`encoder_type: trainable`)

Full transformer encoder trained from scratch:
- Bidirectional self-attention (no causal mask)
- RoPE positional embeddings
- Shared embeddings with decoder (optional)

### 2. Pretrained Encoder (`encoder_type: pretrained`)

Frozen HuggingFace model (e.g., ModernBERT):
```yaml
model:
  encoder_type: pretrained
  pretrained_encoder:
    model_name: answerdotai/ModernBERT-base
    encoder_dim: 768
    freeze_encoder: true
    unfreeze_top_layers: 0  # Set > 0 to fine-tune top layers
```

Features:
- Adds projection layer if `encoder_dim != dim`
- Supports partial unfreezing of top layers
- Uses separate HuggingFace tokenizer for encoder

### 3. Embedding-Only Encoder (`encoder_type: embedding_only`)

Minimal encoder with just embeddings:
- Token embeddings + RMSNorm
- Can share embeddings with decoder
- Useful for ablation studies

## Cross-Attention Mechanism

Cross-attention (`enc_dec.py:185`) connects decoder to encoder:

```python
class CrossAttention(nn.Module):
    def forward(self, x, encoder_output, encoder_mask=None):
        # Q from decoder hidden states
        xq = self.wq(x)  # [B, dec_seq, n_heads * head_dim]

        # K, V from encoder output
        xk = self.wk(encoder_output)  # [B, enc_seq, n_kv_heads * head_dim]
        xv = self.wv(encoder_output)

        # No RoPE applied (positions are independent)
        # No causal mask (full attention over encoder)

        output = scaled_dot_product_attention(xq, xk, xv, attn_mask=encoder_mask)
        return self.wo(output)
```

Key differences from self-attention:
- No RoPE (encoder and decoder positions are independent)
- No causal masking (decoder attends to all encoder positions)
- Encoder padding mask applied to prevent attending to padding tokens

## Pretrained Decoder Support

Load decoder weights from HuggingFace causal LM models:

```yaml
model:
  pretrained_decoder:
    model_name: meta-llama/Llama-3-8B
    freeze_pretrained: false  # Cross-attention always trainable
    init_mode: copy           # Cross-attention initialization mode
```

Weight mapping (`enc_dec.py:998`):
- Self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` -> `wq`, `wk`, `wv`, `wo`
- FFN: `gate_proj`, `up_proj`, `down_proj` -> `w1`, `w3`, `w2`
- Cross-attention: initialized based on `init_mode` (see below)

### Cross-Attention Initialization Modes

The `init_mode` parameter controls how cross-attention layers are initialized from self-attention weights (`enc_dec.py:927`):

| Mode | Q, K, V | O Projection | LayerNorm | Description |
|------|---------|--------------|-----------|-------------|
| `none` | Random | Random | Random | Original behavior |
| `copy` | Copy from self-attn | Copy from self-attn | Copy from ffn_norm | Recommended |
| `zero` | Copy from self-attn | Zero-initialized | Copy from ffn_norm | Gradual integration |
| `normal` | Copy from self-attn | Kaiming normal | Copy from ffn_norm | Alternative |

**Rationale:**
- Q projection operates on the same decoder hidden states as self-attention
- K/V projections work on encoder outputs (different from self-attention), but copying provides reasonable initialization
- LayerNorm is copied from `ffn_norm` (which comes from HF's `post_attention_layernorm`)
- `init_mode: copy` is recommended for better training stability

## Configuration

### Training Arguments (`EncDecTrainArgs`)

```yaml
name: experiment_name
dump_dir: /path/to/output

# Training control
steps: 10000          # Total optimizer steps (or use max_epochs)
max_epochs: null      # Alternative: train for N epochs
grad_acc_steps: 2     # Gradient accumulation steps
seed: 42

# Optimizer
optim:
  lr: 1e-4
  weight_decay: 0.01
  warmup: 200
  lr_min_ratio: 0.01
  clip: 1.0

# Evaluation during training
eval:
  every: 100          # Evaluate every N steps
  max_steps: null     # Limit validation steps

# Checkpointing
checkpoint:
  dump:
    every: 500
    keep: -1          # Keep all (-1) or last N
  eval:
    every: 500
```

### Data Arguments (`EncDecDataArgs`)

```yaml
data:
  dataset_name: squad
  question_column: question
  answer_column: answers
  context_column: context

  max_encoder_len: 2048
  max_decoder_len: 512
  batch_size: 8

  # Optional: separate encoder tokenizer for pretrained encoder
  encoder_tokenizer_name: answerdotai/ModernBERT-base

  # Train/val split
  val_split_ratio: 0.1  # 10% for validation
```

## Distributed Training

Supports FSDP (Fully Sharded Data Parallel):

```yaml
distributed:
  fsdp_type: no_shard    # or full_shard, hybrid_shard
  dp_shard: 1
  dp_replicate: 1
  tp_size: 1
  compile: false
  model_dtype: bf16
  selective_activation_checkpointing: true
```

FSDP grouping plan (`enc_dec.py:1174`) creates sharding boundaries at:
- Encoder embeddings
- Each encoder layer
- Encoder norm
- Each decoder layer
- Decoder norm and output

## Validation During Training

Periodic validation (`train.py:574`):

```python
if val_loader is not None and every_n_steps(train_state, args.eval.every):
    val_metrics = evaluate_validation(model, val_loader, max_steps=args.eval.max_steps)
    # Logs: val_loss, val_perplexity
```

## Checkpointing

The training state includes:
- Model weights
- Optimizer state
- Scheduler state
- Training step
- Accumulation step
- Data loader state (epoch, step_in_epoch)

Preemption handling:
```python
if preemption_flag["flag"]:
    checkpoint.save(model, optimizer, train_state, args)
    requeue_slurm_job()
    sys.exit(0)
```

## Metrics Logged

| Metric | Description |
|--------|-------------|
| `loss/out` | Training loss (distributed mean) |
| `speed/wps` | Words per second |
| `speed/FLOPS` | Floating point operations per second |
| `optim/grad_norm` | Gradient norm before clipping |
| `optim/lr` | Current learning rate |
| `optim/total_tokens` | Total tokens processed |
| `memory/max_active_pct` | Peak GPU memory usage |
| `eval/val_loss` | Validation loss |
| `eval/val_perplexity` | Validation perplexity |

## Running Training

```bash
python -m apps.enc_dec.train config=configs/mvp_modernbert_base_300M.yaml
```

Override parameters via CLI:
```bash
python -m apps.enc_dec.train config=configs/base.yaml \
    name=my_experiment \
    optim.lr=5e-5 \
    data.batch_size=16
```

## Files Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training script and loop |
| `enc_dec.py` | Model architecture definitions |
| `data.py` | Dataset and dataloader implementations |
| `eval.py` | Evaluation and generation utilities |
| `infer.py` | Inference/generation script |
| `export_checkpoint.py` | Checkpoint consolidation |
