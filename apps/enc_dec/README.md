# Encoder-Decoder Architecture for Question Answering

This app implements an encoder-decoder transformer architecture with cross-attention for question answering tasks, compatible with HuggingFace datasets.

## Overview

The encoder-decoder architecture processes:
- **Encoder**: Gold document/context (bidirectional attention)
- **Decoder**: Question + Answer generation with cross-attention to encoder output

## File Structure

```
apps/enc_dec/
├── __init__.py           # Package init
├── enc_dec.py            # Model architecture
├── data.py               # QA data loader for HuggingFace datasets
├── train.py              # Training script
├── eval.py               # Evaluation script
├── README.md             # This file
└── configs/
    ├── debug.yaml        # Small model for debugging
    └── enc_dec_base.yaml # Base configuration
```

## Architecture

### Configuration Design

Encoder and decoder have **separate but compatible** configurations:
- **Shared `dim`**: Required for cross-attention compatibility
- **Separate parameters**: `n_layers`, `n_heads`, `n_kv_heads`, etc. can differ for ablations

```python
@dataclass
class EncDecTransformerArgs:
    dim: int = 512                      # Shared dimension
    max_encoder_seqlen: int = 2048
    max_decoder_seqlen: int = 512
    encoder: EncoderArgs                # Encoder-specific config
    decoder: DecoderArgs                # Decoder-specific config
    share_embeddings: bool = True       # Share encoder/decoder embeddings
```

### Key Components

| Class | Description |
|-------|-------------|
| `CrossAttention` | Q from decoder, K/V from encoder, no RoPE, supports padding masks |
| `EncoderBlock` | Bidirectional self-attention + FFN |
| `DecoderBlock` | Causal self-attention + cross-attention + FFN |
| `Encoder` | Embedding + encoder blocks + final norm |
| `Decoder` | Embedding + decoder blocks + output projection |
| `EncDecTransformer` | Combined model |

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
| `gold_doc` | The document/context containing the answer |

### Label Masking

- **Question tokens**: Masked with `-100` (ignored in loss)
- **Answer tokens**: Used for loss computation
- Supports next-token prediction with shifted labels

### Data Flow

```
1. Load HuggingFace dataset
2. Tokenize: gold_doc → encoder_input, question + answer → decoder_input
3. Create labels: [-100]*question_len + answer_tokens
4. Shift: decoder_input[:-1], labels[1:]
5. Pad and batch
```

## Usage

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
    model.encoder.n_layers=12 \
    model.decoder.n_layers=6 \
    data.batch_size=16
```

## Configuration Examples

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

### Base Config (enc_dec_base.yaml)

```yaml
model:
  dim: 768
  encoder:
    n_layers: 6
    n_heads: 12
  decoder:
    n_layers: 6
    n_heads: 12
data:
  max_encoder_len: 2048
  max_decoder_len: 512
  batch_size: 8
```

## Ablation Support

The design enables various ablations:

| Ablation | Configuration |
|----------|---------------|
| Encoder depth | `model.encoder.n_layers` |
| Decoder depth | `model.decoder.n_layers` |
| Attention heads | `model.encoder.n_heads`, `model.decoder.n_heads` |
| GQA | `model.encoder.n_kv_heads`, `model.decoder.n_kv_heads` |
| Shared embeddings | `model.share_embeddings` |
| Weight tying | `model.weight_tying` |
| FFN size | `model.encoder.ffn_dim_multiplier` |

## Distributed Training

Supports:
- **FSDP**: Fully Sharded Data Parallel (`distributed.fsdp_type: full_shard`)
- **Gradient accumulation**: `grad_acc_steps`
- **Mixed precision**: `distributed.model_dtype: bf16`
- **Activation checkpointing**: `distributed.selective_activation_checkpointing: true`

## Dependencies

- PyTorch 2.0+
- HuggingFace `datasets`
- xformers (for memory-efficient attention)
- OmegaConf (for configuration)

## References

- Based on lingua framework patterns from `apps/main/`
- Cross-attention follows standard encoder-decoder transformer design
- Compatible with HuggingFace QA datasets (SQuAD, Natural Questions, etc.)
