# Minimal SQuAD: Overview & Running on Killarney

## Context

The `minimal_squad` app is a research framework for experimenting with **SQuAD question answering** using different model architectures (decoder-only vs encoder-decoder), data formats, and span extraction strategies. It lives inside the broader **Meta Lingua** framework on the `enc-dec-training` branch.

---

## What's Going On in Minimal SQuAD

### Files

| File | Purpose |
|------|---------|
| `main.py` (~2000 lines) | Unified training script supporting multiple architectures, data formats, and span modes |
| `bert_qa_baseline.py` (~450 lines) | Standard BERT-style extractive QA baseline for comparison |
| `exps.sh` (~220 lines) | Batch experiment submission script with 17 configured experiments (`cepe_exps.sh` for cepe replications) |

### Core Idea: A Format Grammar for QA

The key innovation is a **format string grammar** that controls how Question (Q), Context (C), and Answer (A) are arranged:

- `/` = concatenate in the same sequence
- `//` = split between encoder and decoder
- `S` = span extraction (instead of generative answer)

**Examples:**
| Format | Architecture | What happens |
|--------|-------------|-------------|
| `Q/C/A` | Decoder-only | Everything in one sequence, generate answer |
| `C/Q/A` | Decoder-only | Same, different ordering |
| `C//Q/A` | Enc-Dec | Context in encoder, Q+A in decoder (generate) |
| `C/Q//A` | Enc-Dec | Context+Question in encoder, generate A in decoder |
| `C/Q//S` | Enc-Dec | Context+Question in encoder, extract span in decoder |

### Models Supported

**Encoder-Decoder (encoder frozen, decoder trained from scratch):**
- `modernbert_150m` (answerdotai/ModernBERT-base)
- `modernbert_400m` (answerdotai/ModernBERT-large)
- `deberta_v3_300m`, `deberta_v2_900m`, `deberta_v2_1.5b`

**Encoder-Decoder with Pre-trained Decoder (CEPE-style):**
- Encoder: any of the above encoder models
- Decoder: `tinyllama_1b`, `llama3.2_1b`, `llama3.2_3b`, `llama3.1_8b` (via `--decoder_model_name`)
- Cross-attention adapters are injected between self-attention and FFN of each frozen decoder layer
- Only cross-attention adapters + encoder projection are trained; decoder self-attention/FFN are frozen
- Uses dual tokenizers (encoder tokenizer for Q/C, decoder tokenizer for A)

**Decoder-Only (fine-tuned):**
- `tinyllama_1b`, `llama3.2_1b`, `llama3.2_3b`, `llama3.1_8b`

### Span Extraction Modes

- `none` — Pure generation (decoder produces answer tokens)
- `bertlike` — Predict start/end positions over encoder tokens (like BERT QA)
- `first_last_hidden` — Generate anchor tokens, then match span via hidden states
- `first_last_attn` — Generate anchor tokens, then match span via cross-attention weights

### Key Results from `exps.sh`

| Model | Format | Span Mode | EM% |
|-------|--------|-----------|-----|
| TinyLlama 1B | Q/C/A | none (gen) | 90.0 |
| ModernBERT 150M | C//Q/A | none (gen) | 21.6 |
| ModernBERT 150M | C/Q//A | none (gen) | 64.4 |
| ModernBERT 150M | C/Q//S | bertlike | 75.6 |
| ModernBERT 400M | C/Q//A (6L dec) | none (gen) | 79.2 |
| ModernBERT 400M | C/Q//A (12L dec) | none (gen) | 82.2 |
| ModernBERT 400M | C/Q//S | bertlike | **93.8** |

**Takeaway:** Bert-like span extraction with ModernBERT 400M achieves the best results. Putting both C and Q in the encoder (`C/Q//`) consistently beats context-only (`C//Q`). See `reports/Feb20th_REPORT.md` for detailed CEPE pretrained-decoder results.

---

## How to Run on Killarney

### One-Time Setup

```bash
cd /home/ehghaghi/projects/aip-craffel/ehghaghi/lingua

# 1. Create the environment (only needed once)
bash setup/create_env_killarney_uv.sh

# 2. Place auth tokens in home directory (chmod 600 for safety)
echo "your_token" > ~/hf_token.txt && chmod 600 ~/hf_token.txt
echo "your_token" > ~/wandb_token.txt && chmod 600 ~/wandb_token.txt
```

### Interactive Mode (for debugging)

Start an interactive L40S session, source the env, then copy python commands from `exps.sh` directly into the command line.

```bash
# 1. Get an interactive GPU session
salloc --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64000M \
       --partition=gpubase_l40s_b3 --account=aip-craffel --time=3:00:00

# 2. Source the environment (loads modules, activates venv, sets up tokens)
cd /home/ehghaghi/projects/aip-craffel/ehghaghi/lingua
source setup/start_env.sh

# 3. Copy a python command from exps.sh or cepe_exps.sh and run it directly, e.g.:
python -m apps.minimal_squad.main \
    --data_format "C/Q//S" \
    --span_expr bertlike \
    --model_type encdec \
    --model_name modernbert_400m \
    --pretrained_weight_updating 0.333 \
    --epochs 10 \
    --batch_size 16
```

### Batch Mode (for running multiple experiments)

Source the env on the login node, then use the `submit` function defined in `start_env.sh` to submit jobs. The experiments and their `submit` calls are defined in `exps.sh`.

```bash
# 1. On the login node, source the environment
cd /home/ehghaghi/projects/aip-craffel/ehghaghi/lingua
source setup/start_env.sh

# 2. Submit all experiments as SLURM jobs
bash apps/minimal_squad/exps.sh or bash apps/minimal_squad/cepe_exps.sh
```

Each experiment is submitted as a separate SLURM job to L40S GPUs. The `submit` function handles:
- Loading modules and activating the venv on the compute node
- Setting cache dirs in `$SCRATCH`
- Deduplicating jobs (skips if already running/recently completed)

### Key Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_format` | `Q/C/A` | Format string (see grammar above) |
| `--model_type` | `dec` | `dec` (decoder-only) or `encdec` |
| `--model_name` | `tinyllama_1b` | Model alias or HF path |
| `--span_expr` | `none` | Span mode: none/bertlike/first_last_hidden/first_last_attn |
| `--epochs` | `2` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--pretrained_weight_updating` | `None` | If set (e.g. 0.333), scales pretrained LR; if None/0, freezes pretrained weights |
| `--encoder_weight_updating` | `None` | Override `pretrained_weight_updating` for encoder only (CEPE-style freeze) |
| `--num_decoder_layers` | `6` | Number of decoder layers (enc-dec only) |
| `--max_length` | `8192` | Max encoder sequence length |
| `--dec_max_length` | `8192` | Max decoder sequence length |
| `--enc_local_layer_ratio` | `0.0` | Fraction of encoder layers using local attention |
| `--decoder_model_name` | `None` | Pre-trained decoder model (e.g., `tinyllama_1b`). Enables CEPE-style cross-attention adapters |
| `--cross_attn_num_heads` | `16` | Number of attention heads for cross-attention adapters |

### Pre-trained Decoder (CEPE-style)

Instead of training a decoder from scratch, you can use a pre-trained causal LM (e.g., TinyLlama) as the decoder. Cross-attention adapters are injected between each layer's self-attention and FFN, following the [CEPE](https://github.com/princeton-nlp/CEPE) approach:

```
Encoder (ModernBERT): input -> encoder_hidden [B, enc_len, 1024]
Projection:           Linear(1024, 2048) -> projected [B, enc_len, 2048]
Decoder (TinyLlama + cross-attn adapters):
  For each of 22 layers:
    1. Self-Attention (frozen, RoPE, GQA)
    2. CrossAttentionAdapter(x, projected_encoder)  [trained]
    3. FFN (frozen, SwiGLU)
  Final: RMSNorm -> lm_head -> logits
```

Key design choices:
- Output projection of cross-attention adapters is **zero-initialized**, so the model starts as the original TinyLlama and gradually learns to use encoder information
- Dual tokenizers: encoder side uses ModernBERT tokenizer, decoder side uses TinyLlama tokenizer
- `--pretrained_weight_updating 0.0` freezes both encoder and decoder (only adapters + projection trained, ~370M params)
- `--pretrained_weight_updating 0.333` additionally fine-tunes encoder + decoder at 0.333x learning rate
- `--encoder_weight_updating 0.333` with `--pretrained_weight_updating 0.0` gives CEPE-style training: decoder frozen, encoder fine-tuned at 0.333x LR

**Cross-attention adapter initialization:**

| Component | Init Strategy | Why |
|-----------|--------------|-----|
| Q/K/V projections (`in_proj_weight`) | Truncated normal, `std = hidden_size^{-0.5}` (±3σ bounds) | Standard scaled init for attention projections |
| Output projection (`out_proj`) | **Zero-initialized** (weight and bias) | Adapter starts as no-op; model begins as vanilla TinyLlama |
| LayerNorm (`cross_attn_norm`) | weight=1, bias=0 | Standard LayerNorm default |
| Encoder projection (`Linear(1024, 2048)`) | Truncated normal, same std as decoder | Bridges encoder→decoder hidden size |

The zero-initialized output projection is the key design choice (from CEPE): since each adapter uses a residual connection (`output = residual + cross_attn(x)`), zeroing `out_proj` means the cross-attention contributes nothing at initialization. The model starts as the original TinyLlama and gradually learns to use encoder information during training.

```bash
# Example: ModernBERT 400M encoder + TinyLlama decoder (frozen)
python -m apps.minimal_squad.main \
    --data_format "C/Q//A" \
    --model_type encdec \
    --model_name modernbert_400m \
    --decoder_model_name tinyllama_1b \
    --pretrained_weight_updating 0.0 \
    --epochs 5 \
    --batch_size 8
```

### Monitoring

- **SLURM jobs**: `squeue -u ehghaghi`
- **W&B dashboard**: Check your wandb project for real-time metrics
- **Job logs**: Check SLURM output files in the submission directory

---

## Verification

After running an experiment, check:
1. Training completes without errors (check SLURM logs)
2. Validation EM/F1 metrics are printed at end of each epoch
3. W&B logs show training loss decreasing and eval metrics improving
4. Compare EM% against expected values in `exps.sh` comments
