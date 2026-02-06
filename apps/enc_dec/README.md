# Encoder-Decoder Architecture for Question Answering

This app implements an encoder-decoder transformer architecture with cross-attention for question answering tasks, compatible with HuggingFace datasets.

## Installation

### Quick Start with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that provides reproducible installs.

**Local Development:**
```bash
cd apps/enc_dec

# Quick install (creates .venv and installs dependencies)
./scripts/install_uv.sh

# Activate
source .venv/bin/activate
```

**Compute Canada HPC:**
```bash
cd apps/enc_dec

# Full setup with CUDA support
./scripts/setup_cc.sh

# Activate (add to SLURM scripts)
module load python/3.11 cuda/12.2 cudnn/8.9 arrow/17
source $SCRATCH/envs/lingua_uv/bin/activate
```

### Manual Installation

If you prefer manual setup:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install PyTorch with CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install project
uv pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Project dependencies and metadata |
| `requirements.txt` | Direct dependencies (clean list) |
| `scripts/setup_cc.sh` | Compute Canada setup script |
| `scripts/install_uv.sh` | Quick local install script |
| `slurm/template_uv.slurm` | SLURM template using uv environment |

## Overview

The encoder-decoder architecture processes:
- **Encoder**: Gold document/context (bidirectional attention)
- **Decoder**: Question + Answer generation with cross-attention to encoder output

### Output Mechanisms

Four output mechanisms are supported for different use cases:

| Mechanism | Description | Best For | Expected SQuAD EM |
|-----------|-------------|----------|-------------------|
| **Standard** | Next-token prediction from vocabulary | Abstractive QA, general generation | ~20-30% |
| **Copy** | Blends generation with copying from input | Mixed extractive/abstractive | ~25-35% |
| **Pointer** | Autoregressively points to encoder positions | Pure extraction (token-by-token) | ~15% |
| **Span Pointer** | Predicts start/end positions in single pass | Pure extraction (BERT-style) | ~50-70% |

See [docs/](docs/) for detailed documentation on each mechanism.

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
├── README.md                   # This file
├── MVP_GUIDE.txt               # Step-by-step guide for MVP testing
│
├── # Core Architecture
├── enc_dec.py                  # Standard encoder-decoder model
├── enc_dec_copy.py             # Copy mechanism model
├── enc_dec_pointer.py          # Autoregressive pointer model
├── enc_dec_span_pointer.py     # Span pointer model (BERT-style)
│
├── # Data Loaders
├── data.py                     # Standard QA data loader
├── data_pointer.py             # Pointer mechanism data (position targets)
├── data_span_pointer.py        # Span pointer data (start/end targets)
│
├── # Training Scripts
├── train.py                    # Standard training
├── train_copy.py               # Copy mechanism training
├── train_pointer.py            # Pointer mechanism training
├── train_span_pointer.py       # Span pointer training
│
├── # Evaluation Scripts
├── eval.py                     # General evaluation
├── eval_squad.py               # SQuAD eval for standard model
├── eval_squad_copy.py          # SQuAD eval for copy mechanism
├── eval_squad_pointer.py       # SQuAD eval for pointer mechanism
├── eval_squad_span_pointer.py  # SQuAD eval for span pointer
│
├── # Inference Scripts
├── infer.py                    # Standard inference
├── infer_copy.py               # Copy mechanism inference
├── infer_pointer.py            # Pointer mechanism inference
├── infer_span_pointer.py       # Span pointer inference
│
├── # Utilities
├── export_checkpoint.py        # Checkpoint consolidation/export
│
├── # SLURM Scripts
├── submit_mvp_modernbert_300M.slurm          # Training with pretrained decoder
├── submit_eval_squad.slurm                   # Standard SQuAD evaluation
├── submit_extractive_copy.slurm              # Copy mechanism training
├── submit_extractive_pointer.slurm           # Pointer mechanism training
├── submit_extractive_span_pointer.slurm      # Span pointer training
├── submit_eval_copy.slurm                    # Copy mechanism evaluation
├── submit_eval_pointer.slurm                 # Pointer mechanism evaluation
├── submit_eval_span_pointer.slurm            # Span pointer evaluation
├── submit_infer_copy.slurm                   # Copy mechanism inference
├── submit_infer_pointer.slurm                # Pointer mechanism inference
├── submit_infer_span_pointer.slurm           # Span pointer inference
├── submit_infer_modernbert_300M.slurm        # Standard inference
├── submit_infer_custom.slurm                 # Custom inference template
│
├── # Documentation
├── docs/
│   ├── architecture_diagrams.md
│   ├── copy_mechanism.md
│   ├── pointer_mechanism.md
│   ├── span_pointer_mechanism.md
│   └── training_mechanism.md
│
├── # Weekly Reports
├── weekly-reports/
│   └── WEEKLY_REPORT_Jan27.md
│
└── configs/
    ├── debug.yaml                                              # Small model for debugging
    ├── enc_dec_base.yaml                                       # Base configuration
    ├── mvp_modernbert_base_300M.yaml                           # Frozen ModernBERT + 300M decoder
    ├── mvp_modernbert_pretrained_dec_300M.yaml                 # Frozen ModernBERT + pretrained 300M decoder
    ├── mvp_modernbert_scratch_dec_300M_frozen_Modern_BERT.yaml # Frozen encoder variants
    ├── mvp_modernbert_scratch_dec_300M_frozen_Modern_BERT_2layers.yaml
    ├── mvp_modernbert_scratch_dec_300M_frozen_Modern_BERT_4layers.yaml
    ├── extractive_copy.yaml                                    # Copy mechanism config
    ├── extractive_pointer.yaml                                 # Pointer mechanism config
    └── extractive_span_pointer.yaml                            # Span pointer config (recommended)
```

## Trainable Components

The following table shows what components are trainable based on configuration:

### With Pretrained Encoder + Pretrained Decoder

| Component | Config Setting | Trainable? |
|-----------|----------------|------------|
| **Encoder (ModernBERT)** | | |
| └─ Bottom layers | `freeze_encoder: true` | Frozen |
| └─ Top N layers | `unfreeze_top_layers: N` | Trainable |
| └─ Projection (768→dim) | Always | Trainable |
| **Decoder (from HF)** | | |
| └─ Embeddings | `freeze_pretrained: false` | Trainable |
| └─ Self-attention | `freeze_pretrained: false` | Trainable |
| └─ FFN layers | `freeze_pretrained: false` | Trainable |
| └─ Output projection | `freeze_pretrained: false` | Trainable |
| **Cross-Attention** | | |
| └─ All layers | Always (randomly init) | Trainable |

**Example config:**
```yaml
pretrained_encoder:
  freeze_encoder: true
  unfreeze_top_layers: 2    # Train top 2 encoder layers
pretrained_decoder:
  freeze_pretrained: false  # Train all decoder weights
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
    pretrained_decoder: PretrainedDecoderArgs  # Config for pretrained decoder
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

**Extractive Mechanism Components:**

| Class | Description |
|-------|-------------|
| `SpanPointerDecoder` | Predicts start/end positions for span extraction |
| `EncDecSpanPointerTransformer` | Encoder-decoder with span pointer mechanism |
| `CopyDecoder` | Blends generation with copying from encoder |
| `EncDecCopyTransformer` | Encoder-decoder with copy mechanism |
| `PointerDecoder` | Autoregressive pointer to encoder positions |
| `EncDecPointerTransformer` | Encoder-decoder with pointer mechanism |

### Pretrained Encoder (ModernBERT)

The `PretrainedEncoder` class enables using frozen pretrained models:

```python
@dataclass
class PretrainedEncoderArgs:
    model_name: str = "answerdotai/ModernBERT-base"  # HuggingFace model
    encoder_dim: int = 768                            # Hidden dim of pretrained model
    pooling: str = "none"                             # "none", "mean", or "cls"
    use_flash_attention: bool = True                  # Use flash attention if available
    freeze_encoder: bool = True                       # Whether to freeze encoder weights
    unfreeze_top_layers: int = 0                      # Number of top layers to unfreeze
```

Features:
- **Frozen weights**: Encoder parameters have `requires_grad=False` by default
- **Partial fine-tuning**: Unfreeze top N layers with `unfreeze_top_layers`
- **Automatic projection**: Adds trainable linear layer if `encoder_dim != dim`
- **Flash attention**: Uses flash_attention_2 for efficiency when available
- **Separate tokenizer**: Supports using the pretrained model's tokenizer

### Partial Encoder Fine-tuning

You can unfreeze the top N layers of the pretrained encoder for task-specific adaptation:

```yaml
pretrained_encoder:
  model_name: answerdotai/ModernBERT-base
  encoder_dim: 768
  freeze_encoder: true      # Freeze most layers
  unfreeze_top_layers: 2    # But train the top 2 layers
```

**Trainable components with this config:**
| Component | Trainable? |
|-----------|------------|
| Bottom encoder layers | Frozen |
| Top N encoder layers | Trainable |
| Projection layer (if dim mismatch) | Trainable |

This allows the encoder to adapt its representations to the downstream task while preserving most of the pretrained knowledge.

### Pretrained Decoder

The `PretrainedDecoderArgs` class enables initializing the decoder from a HuggingFace causal LM model (e.g., LLaMA-style models):

```python
@dataclass
class PretrainedDecoderArgs:
    model_name: str = ""           # HuggingFace model name/path (empty = no pretrained)
    freeze_pretrained: bool = False  # Whether to freeze loaded weights
    init_mode: str = "none"        # Cross-attention init: "none", "copy", "zero", "normal"
```

Features:
- **Partial weight loading**: Loads matching weights (embeddings, self-attention, FFN, norms) from the HF model
- **Cross-attention initialization**: Cross-attention layers can be initialized from self-attention weights using `init_mode`
- **Optional freezing**: Can freeze pretrained weights while keeping cross-attention trainable
- **Architecture matching**: Decoder config (`n_layers`, `n_heads`, etc.) must match the pretrained model

Weight Mapping (LLaMA-style HF model -> enc_dec decoder):

| HuggingFace Model | enc_dec Decoder |
|-------------------|-----------------|
| `model.embed_tokens.weight` | `decoder.tok_embeddings.weight` |
| `model.layers.{i}.self_attn.q_proj` | `decoder.layers.{i}.self_attention.wq` |
| `model.layers.{i}.self_attn.k_proj` | `decoder.layers.{i}.self_attention.wk` |
| `model.layers.{i}.self_attn.v_proj` | `decoder.layers.{i}.self_attention.wv` |
| `model.layers.{i}.self_attn.o_proj` | `decoder.layers.{i}.self_attention.wo` |
| `model.layers.{i}.mlp.gate_proj` | `decoder.layers.{i}.feed_forward.w1` |
| `model.layers.{i}.mlp.up_proj` | `decoder.layers.{i}.feed_forward.w3` |
| `model.layers.{i}.mlp.down_proj` | `decoder.layers.{i}.feed_forward.w2` |
| `model.layers.{i}.input_layernorm` | `decoder.layers.{i}.self_attention_norm` |
| `model.layers.{i}.post_attention_layernorm` | `decoder.layers.{i}.ffn_norm` |
| `model.norm` | `decoder.norm` |
| `lm_head.weight` | `decoder.output.weight` |
| Depends on `init_mode` | `decoder.layers.{i}.cross_attention.*` |
| Depends on `init_mode` | `decoder.layers.{i}.cross_attention_norm` |

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

### Cross-Attention Initialization

When using a pretrained decoder, cross-attention layers don't exist in standard causal LM models. The `init_mode` parameter controls how they are initialized:

| Mode | Q, K, V Projections | O Projection | LayerNorm | Use Case |
|------|---------------------|--------------|-----------|----------|
| `none` | Random (truncated normal) | Random | Random | Baseline |
| `copy` | Copy from self-attention | Copy from self-attention | Copy from ffn_norm | Best for training stability |
| `zero` | Copy from self-attention | Zero-initialized | Copy from ffn_norm | Gradual integration |
| `normal` | Copy from self-attention | Kaiming normal | Copy from ffn_norm | Alternative initialization |

**Configuration:**
```yaml
pretrained_decoder:
  model_name: Malikeh1375/nemotron_fineinstructions_1T_judged_exp_chat_300M
  freeze_pretrained: false
  init_mode: copy  # Recommended for better training dynamics
```

**How it works:**
- **Q projection**: Copied from self-attention (operates on same decoder hidden states)
- **K, V projections**: Copied from self-attention with dimension slicing if needed
- **O projection**: Based on mode - copy, zero, or kaiming_normal initialization
- **LayerNorm**: Copied from `ffn_norm` (corresponds to HF's `post_attention_layernorm`)

**Logs will show:**
```
Initializing cross-attention weights from self-attention (mode: copy)
Initialized 20,000,000 cross-attention parameters from self-attention
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

### Validation During Training

To evaluate on a held-out validation set during training:

1. Set `val_split_ratio` to hold out a fraction of training data:
```yaml
data:
  val_split_ratio: 0.1  # 10% for validation
```

2. Configure evaluation frequency:
```yaml
eval:
  every: 100  # Evaluate every 100 optimizer steps
  max_steps: null  # Use full validation set (or set a number to limit)
```

**Logged metrics:**
- `eval/val_loss`: Average cross-entropy loss on validation set
- `eval/val_perplexity`: Perplexity (exp of val_loss)

**Example log output:**
```
Running validation at step 100...
Validation: step=100  val_loss=2.3456  val_ppl=10.43
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

### SQuAD Evaluation

Evaluate on the SQuAD 2.0 dev set with official EM/F1 metrics:

```bash
# Full evaluation
python -m apps.enc_dec.eval_squad \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --data_file /path/to/dev-v2.0.json \
    --output_dir /path/to/results

# Quick test with limited examples
python -m apps.enc_dec.eval_squad \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --data_file /path/to/dev-v2.0.json \
    --max_examples 100

# Evaluate on a percentage of the data
python -m apps.enc_dec.eval_squad \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --data_file /path/to/dev-v2.0.json \
    --eval_percent 10
```

**SQuAD Evaluation Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | Path to consolidated `.pth` file | Required |
| `--config` | Path to `params.json` config | Required |
| `--data_file` | Path to SQuAD dev-v2.0.json | Required |
| `--output_dir` | Directory to save predictions/results | None |
| `--max_examples` | Limit number of examples | None (all) |
| `--eval_percent` | Percentage of data to evaluate | 100 |
| `--max_new_tokens` | Maximum tokens to generate | 64 |

**Output:**
- Live progress with per-example EM/F1 scores
- Running average metrics during evaluation
- Final results: `{"exact": X.X, "f1": X.X, "total": N}`
- Optional: `predictions.json` and `eval.json` saved to output_dir

### Inference

Run inference on a trained model to generate answers for questions:

```bash
# Using SQuAD validation examples
python -m apps.enc_dec.infer \
    --checkpoint /path/to/checkpoint/0000001000/consolidated/consolidated.pth \
    --config /path/to/checkpoint/0000001000/params.json \
    --num_examples 5

# Custom document and question
python -m apps.enc_dec.infer \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --document "Paris is the capital and largest city of France." \
    --question "What is the capital of France?"

# With sampling parameters
python -m apps.enc_dec.infer \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 100
```

**Inference Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | Path to consolidated `.pth` file | Required |
| `--config` | Path to `params.json` config | Required |
| `--document` | Custom document/context | Uses SQuAD |
| `--question` | Custom question | Uses SQuAD |
| `--num_examples` | Number of SQuAD examples | 3 |
| `--max_new_tokens` | Maximum tokens to generate | 64 |
| `--temperature` | Sampling temperature (1.0=greedy) | 1.0 |
| `--top_k` | Top-k sampling | None |
| `--top_p` | Nucleus sampling | None |
| `--device` | Device (cuda/cpu) | cuda |

## Extractive Mechanisms

For extractive QA where answers must be spans from the input context, we provide three specialized mechanisms. The **Span Pointer** mechanism is recommended for best performance.

### Span Pointer (Recommended)

BERT-style single-pass span extraction. Predicts start and end positions in one forward pass.

**Training:**
```bash
# Single GPU
torchrun --nproc-per-node 1 \
    -m apps.enc_dec.train_span_pointer \
    config=apps/enc_dec/configs/extractive_span_pointer.yaml

# SLURM
sbatch apps/enc_dec/submit_extractive_span_pointer.slurm
```

**Evaluation:**
```bash
python -m apps.enc_dec.eval_squad_span_pointer \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --data_file /path/to/dev-v2.0.json \
    --output_dir /path/to/results \
    --max_span_length 50
```

**Inference:**
```bash
python -m apps.enc_dec.infer_span_pointer \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --num_examples 10 \
    --visualize
```

### Copy Mechanism

Blends vocabulary generation with copying from encoder input.

**Training:**
```bash
torchrun --nproc-per-node 1 \
    -m apps.enc_dec.train_copy \
    config=apps/enc_dec/configs/extractive_copy.yaml
```

**Evaluation:**
```bash
python -m apps.enc_dec.eval_squad_copy \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --data_file /path/to/dev-v2.0.json
```

### Pointer Mechanism

Autoregressive pointing to encoder positions (token-by-token).

**Training:**
```bash
torchrun --nproc-per-node 1 \
    -m apps.enc_dec.train_pointer \
    config=apps/enc_dec/configs/extractive_pointer.yaml
```

**Evaluation:**
```bash
python -m apps.enc_dec.eval_squad_pointer \
    --checkpoint /path/to/consolidated.pth \
    --config /path/to/params.json \
    --data_file /path/to/dev-v2.0.json
```

### Mechanism Comparison

| Mechanism | Forward Passes | Output Guarantee | SQuAD EM | Use Case |
|-----------|----------------|------------------|----------|----------|
| Span Pointer | 1 | Contiguous span | ~50-70% | Best for extractive QA |
| Copy | N (autoregressive) | Tokens from vocab or input | ~25-35% | Mixed extractive/abstractive |
| Pointer | N (autoregressive) | Positions in input | ~15% | Pure extraction (experimental) |

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

### Pretrained Encoder + Pretrained Decoder (mvp_modernbert_pretrained_dec_300M.yaml)

This configuration uses both a frozen pretrained encoder (ModernBERT) and initializes the decoder from a pretrained HuggingFace causal LM model. Cross-attention layers are initialized from self-attention weights using `init_mode: copy`.

```yaml
model:
  dim: 960  # Must match the pretrained decoder's hidden dim
  encoder_type: pretrained
  pretrained_encoder:
    model_name: answerdotai/ModernBERT-base
    encoder_dim: 768
    pooling: none
    use_flash_attention: true
  pretrained_decoder:
    model_name: Malikeh1375/nemotron_fineinstructions_1T_judged_exp_chat_300M
    freeze_pretrained: false  # All weights trainable
    init_mode: copy  # Initialize cross-attention from self-attention weights
  decoder:
    # Architecture must match the pretrained model
    n_layers: 32
    n_heads: 15
    n_kv_heads: 5
    rope_theta: 100000.0
data:
  encoder_tokenizer_name: answerdotai/ModernBERT-base
  tokenizer:
    name: tiktoken
    path: /path/to/llama3_tokenizer.model
```

**Usage:**
```bash
python -m apps.enc_dec.train config=apps/enc_dec/configs/mvp_modernbert_pretrained_dec_300M.yaml
```

**Logs will show:**
```
Loading pretrained decoder weights from: Malikeh1375/nemotron_fineinstructions_1T_judged_exp_chat_300M
Loaded 280,000,000 parameters from pretrained model
Initializing cross-attention weights from self-attention (mode: copy)
Initialized 20,000,000 cross-attention parameters from self-attention
```

### Span Pointer for Extractive QA (extractive_span_pointer.yaml)

This configuration uses the span pointer mechanism for BERT-style extractive QA. The decoder predicts start and end positions in the encoder output instead of generating tokens.

```yaml
name: extractive_span_pointer_pretrained_dec_300M

model:
  dim: 960
  encoder_type: pretrained

  pretrained_encoder:
    model_name: answerdotai/ModernBERT-base
    encoder_dim: 768
    freeze_encoder: true
    unfreeze_top_layers: 0  # Fully frozen

  pretrained_decoder:
    model_name: Malikeh1375/nemotron_fineinstructions_1T_judged_exp_chat_300M
    freeze_pretrained: false  # Fine-tune for span prediction
    init_mode: copy  # Initialize cross-attention from self-attention weights

  decoder:
    n_layers: 32
    n_heads: 15
    n_kv_heads: 5
    rope_theta: 100000.0

# Span pointer mechanism configuration
span_pointer:
  use_projection: true
  projection_dim: null  # Use model dim (960)
  condition_end_on_start: false  # Independent start/end prediction
  question_pooling: last  # Use last token for question representation
  max_span_length: 50  # Maximum answer span length

data:
  dataset_name: squad
  max_encoder_len: 1024
  max_decoder_len: 128  # Just question (no answer tokens needed)
  batch_size: 8
```

**Key differences from standard encoder-decoder:**
- `span_pointer` section configures the span prediction mechanism
- Decoder outputs start/end logits instead of vocabulary logits
- Training uses (start_position, end_position) targets instead of token sequences
- Single forward pass for inference

**Usage:**
```bash
torchrun --nproc-per-node 1 \
    -m apps.enc_dec.train_span_pointer \
    config=apps/enc_dec/configs/extractive_span_pointer.yaml
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
| Pretrained encoder | `model.pretrained_encoder.model_name` |
| Pretrained decoder | `model.pretrained_decoder.model_name` |
| Freeze pretrained decoder | `model.pretrained_decoder.freeze_pretrained` |
| Cross-attention init | `model.pretrained_decoder.init_mode` |
| Pooling strategy | `model.pretrained_encoder.pooling` |

## Supported Pretrained Encoders

Any HuggingFace encoder model can be used. Recommended options:

| Model | Params | Dim | Context | Notes |
|-------|--------|-----|---------|-------|
| `answerdotai/ModernBERT-base` | 149M | 768 | 8192 | Modern architecture, fast |
| `answerdotai/ModernBERT-large` | 395M | 1024 | 8192 | Larger, better quality |
| `nomic-ai/modernbert-embed-base` | 149M | 768 | 8192 | Optimized for embeddings |
| `Alibaba-NLP/gte-modernbert-base` | 149M | 768 | 8192 | Good retrieval performance |

## Supported Pretrained Decoders

Any HuggingFace LLaMA-style causal LM model can be used. The decoder config must match the pretrained model architecture.

| Model | Params | Dim | Layers | Heads | Notes |
|-------|--------|-----|--------|-------|-------|
| `Malikeh1375/nemotron_fineinstructions_1T_judged_exp_chat_300M` | 300M | 960 | 32 | 15 | LLaMA-style, recommended for testing |
| `meta-llama/Llama-3.2-1B` | 1B | 2048 | 16 | 32 | Meta's small LLaMA model |
| `nvidia/Nemotron-Mini-4B-Instruct` | 4B | 3072 | 32 | 24 | NVIDIA's instruction-tuned model |

**Important:** When using a pretrained decoder, ensure your decoder config matches:
- `model.dim` = pretrained model's hidden dimension
- `model.decoder.n_layers` = pretrained model's number of layers
- `model.decoder.n_heads` = pretrained model's number of attention heads
- `model.decoder.n_kv_heads` = pretrained model's number of KV heads (for GQA)

## Distributed Training

Supports:
- **FSDP**: Fully Sharded Data Parallel (`distributed.fsdp_type: full_shard`)
- **Gradient accumulation**: `grad_acc_steps`
- **Mixed precision**: `distributed.model_dtype: bf16`
- **Activation checkpointing**: `distributed.selective_activation_checkpointing: true`
- **Frozen encoder handling**: Pretrained encoder excluded from FSDP sharding
- **Single-GPU mode**: Automatic FSDP bypass for single-GPU with `no_shard` (PyTorch 2.7+ compatibility)

## Checkpoint Consolidation

During distributed training, checkpoints are saved in PyTorch's Distributed Checkpoint (DCP) format, which consists of multiple sharded files. For easier model loading during inference, checkpoints can be automatically consolidated into a single `.pth` file.

### Automatic Consolidation (Enabled by Default)

Checkpoint consolidation is **enabled by default**. After each checkpoint save, a `consolidated/consolidated.pth` file is automatically created:

```
checkpoints/
└── 0000001000/
    ├── __0_0.distcp
    ├── __1_0.distcp
    ├── ...
    ├── params.json
    ├── train_state_00000.json
    └── consolidated/
        ├── consolidated.pth    # Single file with model + optimizer
        └── params.json
```

### Disabling Consolidation

To disable automatic consolidation (e.g., to save time during frequent checkpointing):

```yaml
checkpoint:
  consolidate: false
```

### Export Script for Existing Checkpoints

To consolidate existing DCP checkpoints that were saved without consolidation:

```bash
# Basic export
python -m apps.enc_dec.export_checkpoint --checkpoint_dir /path/to/checkpoint/0000001000

# Export model weights only (without optimizer state) - smaller file size
python -m apps.enc_dec.export_checkpoint --checkpoint_dir /path/to/checkpoint/0000001000 --model_only
```

The `--model_only` flag extracts just the model weights, creating a `model_weights.pth` file that's smaller and suitable for inference.

### Loading Consolidated Checkpoints

```python
import torch

# Load full checkpoint (model + optimizer)
checkpoint = torch.load("consolidated/consolidated.pth", map_location="cpu")
model_state_dict = checkpoint["model"]
optimizer_state_dict = checkpoint["optim"]

# Or load model-only weights
model_state_dict = torch.load("consolidated/model_weights.pth", map_location="cpu")
```

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

## Documentation

Detailed documentation for each mechanism is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [docs/span_pointer_mechanism.md](docs/span_pointer_mechanism.md) | Span pointer (BERT-style) for extractive QA |
| [docs/pointer_mechanism.md](docs/pointer_mechanism.md) | Autoregressive pointer mechanism |
| [docs/copy_mechanism.md](docs/copy_mechanism.md) | Copy mechanism with generation blend |
| [docs/architecture_diagrams.md](docs/architecture_diagrams.md) | Visual architecture diagrams |
| [docs/training_mechanism.md](docs/training_mechanism.md) | Training details and tips |

## Future Work

- [ ] Subquadratic cross-attention (linear attention variants)
- [ ] KV cache for encoder output during generation
- [ ] Multi-document batching/packing
- [ ] Tensor Parallelism support
- [x] Span pointer mechanism for BERT-style extractive QA
- [x] Pretrained decoder support with cross-attention initialization
- [x] Copy and pointer mechanisms for extractive QA

## References

**Framework:**
- Based on lingua framework patterns from `apps/main/`
- Cross-attention follows standard encoder-decoder transformer design
- Compatible with HuggingFace QA datasets (SQuAD, Natural Questions, etc.)

**Pretrained Models:**
- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base
- Nemotron 300M: https://huggingface.co/Malikeh1375/nemotron_fineinstructions_1T_judged_exp_chat_300M

**Papers:**
- [BERT for Question Answering](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [Pointer Networks](https://arxiv.org/abs/1506.03134) - Vinyals et al., 2015
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) - See et al., 2017
- [SQuAD: 100,000+ Questions for Machine Comprehension](https://arxiv.org/abs/1606.05250) - Rajpurkar et al., 2016
