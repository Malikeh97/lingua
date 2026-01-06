# Encoder-Decoder Architecture for Question Answering

This app implements an encoder-decoder transformer architecture with cross-attention for question answering tasks, compatible with HuggingFace datasets.

## Overview

The encoder-decoder architecture processes:
- **Encoder**: Gold document/context (bidirectional attention)
- **Decoder**: Question + Answer generation with cross-attention to encoder output

### Encoder Types

Three encoder modes are supported:

| Type | Description | Use Case |
|------|-------------|----------|
| `trainable` | Full trainable transformer encoder | Baseline, full control |
| `pretrained` | Frozen HuggingFace model (e.g., ModernBERT) | Best quality embeddings, efficient |
| `embedding_only` | Just embeddings + layer norm | Quick debugging, ablations |

## File Structure

```
apps/enc_dec/
├── __init__.py                 # Package init
├── enc_dec.py                  # Model architecture
├── data.py                     # QA data loader for HuggingFace datasets
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── README.md                   # This file
├── MVP_GUIDE.txt               # Step-by-step guide for MVP testing
├── submit_smoke_test.slurm     # SLURM script for smoke test
├── submit_mvp_modernbert.slurm # SLURM script for MVP with ModernBERT
└── configs/
    ├── debug.yaml              # Small model for debugging
    ├── enc_dec_base.yaml       # Base configuration
    ├── smoke_test_1b.yaml      # 1.3B model smoke test
    ├── mvp_modernbert.yaml     # MVP: Frozen ModernBERT + trainable decoder
    └── mvp_embedding_only.yaml # MVP: Embedding-only encoder (quick debug)
```

## Architecture

### Configuration Design

Encoder and decoder have **separate but compatible** configurations:
- **Shared `dim`**: Required for cross-attention compatibility
- **Separate parameters**: `n_layers`, `n_heads`, `n_kv_heads`, etc. can differ for ablations
- **Encoder type**: Choose between trainable, pretrained, or embedding-only

```python
@dataclass
class EncDecTransformerArgs:
    dim: int = 512                      # Shared dimension
    max_encoder_seqlen: int = 2048
    max_decoder_seqlen: int = 512
    encoder_type: str = "trainable"     # "trainable", "pretrained", "embedding_only"
    encoder: EncoderArgs                # Encoder-specific config
    decoder: DecoderArgs                # Decoder-specific config
    pretrained_encoder: PretrainedEncoderArgs  # Config for pretrained encoder
    share_embeddings: bool = True       # Share encoder/decoder embeddings
```

### Key Components

| Class | Description |
|-------|-------------|
| `CrossAttention` | Q from decoder, K/V from encoder, no RoPE, supports padding masks |
| `EncoderBlock` | Bidirectional self-attention + FFN |
| `DecoderBlock` | Causal self-attention + cross-attention + FFN |
| `Encoder` | Trainable: Embedding + encoder blocks + final norm |
| `PretrainedEncoder` | Frozen HuggingFace model + optional projection layer |
| `EmbeddingOnlyEncoder` | Just embeddings + layer norm (no transformer layers) |
| `Decoder` | Embedding + decoder blocks + output projection |
| `EncDecTransformer` | Combined model with encoder type selection |

### Pretrained Encoder (ModernBERT)

The `PretrainedEncoder` class enables using frozen pretrained models:

```python
@dataclass
class PretrainedEncoderArgs:
    model_name: str = "answerdotai/ModernBERT-base"  # HuggingFace model
    encoder_dim: int = 768                            # Hidden dim of pretrained model
    pooling: str = "none"                             # "none", "mean", or "cls"
    use_flash_attention: bool = True                  # Use flash attention if available
```

Features:
- **Frozen weights**: Encoder parameters have `requires_grad=False`
- **Automatic projection**: Adds trainable linear layer if `encoder_dim != dim`
- **Flash attention**: Uses flash_attention_2 for efficiency when available
- **Separate tokenizer**: Supports using the pretrained model's tokenizer

### Cross-Attention Implementation

The `CrossAttention` class differs from self-attention:
- **No RoPE**: Encoder and decoder positions are independent
- **No causal mask**: Full attention over encoder sequence
- **Flexible backend**: Supports `sdpa`, `fmha`, `flex_attention`

```python
class CrossAttention(nn.Module):
    def forward(self, x, encoder_output, encoder_mask=None, attn_impl="sdpa"):
        # x: [B, dec_seq, D] - decoder hidden states
        # encoder_output: [B, enc_seq, D] - encoder outputs
        xq = self.wq(x)                    # Q from decoder
        xk = self.wk(encoder_output)       # K from encoder
        xv = self.wv(encoder_output)       # V from encoder
        # No RoPE applied
        # Apply attention with encoder_mask
```

### Decoder Block Structure

```python
def forward(self, x, encoder_output, freq_cis, self_attn_mask, encoder_mask):
    # 1. Causal self-attention
    h = x + self.self_attention(self.self_attn_norm(x), freq_cis, mask=self_attn_mask)
    # 2. Cross-attention to encoder
    h = h + self.cross_attention(self.cross_attn_norm(h), encoder_output, encoder_mask)
    # 3. Feed-forward
    return h + self.feed_forward(self.ffn_norm(h))
```

## Data Format

The data loader expects HuggingFace datasets with three columns:

| Column | Description |
|--------|-------------|
| `question` | The question to answer |
| `answer` | The ground truth answer |
| `gold_doc` / `context` | The document/context containing the answer |

### Separate Tokenizer Support

For pretrained encoders, you can use the model's native tokenizer:

```yaml
data:
  encoder_tokenizer_name: answerdotai/ModernBERT-base  # For encoder input
  tokenizer:
    name: bytes  # For decoder input
```

### Label Masking

- **Question tokens**: Masked with `-100` (ignored in loss)
- **Answer tokens**: Used for loss computation
- Supports next-token prediction with shifted labels

### Data Flow

```
1. Load HuggingFace dataset
2. Tokenize:
   - gold_doc → encoder_input (using encoder_tokenizer if specified)
   - question + answer → decoder_input (using decoder tokenizer)
3. Create labels: [-100]*question_len + answer_tokens
4. Shift: decoder_input[:-1], labels[1:]
5. Pad and batch
```

## Usage

### Quick Start with MVP

See `MVP_GUIDE.txt` for detailed step-by-step instructions.

```bash
# Submit MVP job with frozen ModernBERT encoder
sbatch apps/enc_dec/submit_mvp_modernbert.slurm

# Or run interactively
torchrun --nproc-per-node 1 -m apps.enc_dec.train \
    config=apps/enc_dec/configs/mvp_modernbert.yaml
```

### Training

```bash
# Single GPU (debugging)
python -m apps.enc_dec.train config=apps/enc_dec/configs/debug.yaml

# Multi-GPU with torchrun
torchrun --nproc-per-node 8 -m apps.enc_dec.train \
    config=apps/enc_dec/configs/enc_dec_base.yaml \
    dump_dir=/path/to/output \
    data.dataset_name=your_dataset

# SLURM with stool
python -m lingua.stool script=apps.enc_dec.train \
    config=apps/enc_dec/configs/enc_dec_base.yaml \
    nodes=1
```

### Evaluation

```bash
python -m apps.enc_dec.eval \
    config=apps/enc_dec/configs/eval.yaml \
    ckpt_dir=/path/to/checkpoint
```

### Configuration Override

Use dot notation to override nested parameters:

```bash
python -m apps.enc_dec.train config=config.yaml \
    model.dim=1024 \
    model.encoder_type=pretrained \
    model.pretrained_encoder.model_name=answerdotai/ModernBERT-large \
    model.decoder.n_layers=6 \
    data.batch_size=16
```

## Configuration Examples

### MVP with Pretrained Encoder (mvp_modernbert.yaml)

```yaml
model:
  dim: 768
  encoder_type: pretrained
  pretrained_encoder:
    model_name: answerdotai/ModernBERT-base
    encoder_dim: 768
    pooling: none
    use_flash_attention: true
  decoder:
    n_layers: 6
    n_heads: 12
data:
  encoder_tokenizer_name: answerdotai/ModernBERT-base
  tokenizer:
    name: bytes
```

### Embedding-Only Encoder (mvp_embedding_only.yaml)

```yaml
model:
  dim: 512
  encoder_type: embedding_only
  share_embeddings: true
  decoder:
    n_layers: 4
    n_heads: 8
```

### Trainable Encoder (smoke_test_1b.yaml)

```yaml
model:
  dim: 2048
  encoder_type: trainable  # default
  encoder:
    n_layers: 6
    n_heads: 16
  decoder:
    n_layers: 22
    n_heads: 16
```

### Debug Config (debug.yaml)

```yaml
model:
  dim: 256
  encoder:
    n_layers: 2
    n_heads: 4
  decoder:
    n_layers: 2
    n_heads: 4
data:
  dataset_name: squad
  max_encoder_len: 512
  max_decoder_len: 128
  batch_size: 4
```

## Ablation Support

The design enables various ablations:

| Ablation | Configuration |
|----------|---------------|
| Encoder type | `model.encoder_type` |
| Encoder depth | `model.encoder.n_layers` |
| Decoder depth | `model.decoder.n_layers` |
| Attention heads | `model.encoder.n_heads`, `model.decoder.n_heads` |
| GQA | `model.encoder.n_kv_heads`, `model.decoder.n_kv_heads` |
| Shared embeddings | `model.share_embeddings` |
| Weight tying | `model.weight_tying` |
| FFN size | `model.encoder.ffn_dim_multiplier` |
| Pretrained model | `model.pretrained_encoder.model_name` |
| Pooling strategy | `model.pretrained_encoder.pooling` |

## Supported Pretrained Encoders

Any HuggingFace encoder model can be used. Recommended options:

| Model | Params | Dim | Context | Notes |
|-------|--------|-----|---------|-------|
| `answerdotai/ModernBERT-base` | 149M | 768 | 8192 | Modern architecture, fast |
| `answerdotai/ModernBERT-large` | 395M | 1024 | 8192 | Larger, better quality |
| `nomic-ai/modernbert-embed-base` | 149M | 768 | 8192 | Optimized for embeddings |
| `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | 8192 | Good retrieval performance |

## Distributed Training

Supports:
- **FSDP**: Fully Sharded Data Parallel (`distributed.fsdp_type: full_shard`)
- **Gradient accumulation**: `grad_acc_steps`
- **Mixed precision**: `distributed.model_dtype: bf16`
- **Activation checkpointing**: `distributed.selective_activation_checkpointing: true`
- **Frozen encoder handling**: Pretrained encoder excluded from FSDP sharding

## Parameter Counting

Training logs show detailed parameter breakdown:

```
Total parameters: 250,000,000
Encoder parameters: 149,000,000
Decoder parameters: 100,000,000
Trainable parameters: 100,000,000  # Only decoder + projection
Frozen parameters: 149,000,000      # Pretrained encoder
```

## Dependencies

- PyTorch 2.0+
- HuggingFace `transformers` >= 4.48.0 (for ModernBERT)
- HuggingFace `datasets`
- xformers (for memory-efficient attention)
- OmegaConf (for configuration)
- flash-attn (optional, for faster attention)

## Future Work

- [ ] Subquadratic cross-attention (linear attention variants)
- [ ] KV cache for encoder output during generation
- [ ] Multi-document batching/packing
- [ ] Tensor Parallelism support

## References

- Based on lingua framework patterns from `apps/main/`
- Cross-attention follows standard encoder-decoder transformer design
- Compatible with HuggingFace QA datasets (SQuAD, Natural Questions, etc.)
- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base
