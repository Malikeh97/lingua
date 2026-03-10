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

---

## Recent Code Changes

### `linear_kda` Performance Fix (`cepe.py`) — March 2026

The `linear_kda` cross-attention variant was 4–5x slower than full softmax attention (14–16h vs 3–4h on SQuAD). The root cause was ~512 sequential Python-controlled GPU kernel dispatches per forward pass, caused by two nested loops and excessive `.clone()` calls inside the inner one.

**What was changed:**

1. **Added `flash-linear-attention` library import** (lines 58–66). The bundled `fla` library at `apps/minimal_squad/flash-linear-attention/` is added to `sys.path` at import time, and `chunk_kda` is imported via a guarded `try/except`.

2. **Replaced the Python chunk loop with two paths** (lines 911–989 of `cepe.py`):

   - **Fast path** (`_FLA_AVAILABLE = True`): Calls `fla.ops.kda.chunk_kda` — a single fused Triton/CUDA kernel that handles the full encoder sequence at once. Tensor layout is permuted from `[B, H, T, d_k]` to `[B, T, H, d_k]` as expected by the library. The encoder key tensor is passed as Q (its output is discarded); only `final_state` is used. This reduces ~512 Python-level kernel dispatches to ~3.

   - **Fallback path** (`_FLA_AVAILABLE = False`): Keeps the outer `for t_start` chunk loop (8 iterations) but replaces the inner 63-iteration forward-substitution loop — and all 504 associated `.clone()` calls — with a single `torch.linalg.solve_triangular` call. This is a batched CUDA LAPACK kernel that computes the exact `(I - L)^{-1}` in one shot.

**Expected impact:**

| Path | Kernel launches (forward) | Expected training time |
|------|--------------------------|----------------------|
| Before (Python loops) | ~512 | 14–16 hours |
| Fallback (solve_triangular) | ~24 (8 outer × ~3) | ~3–5 hours |
| Fast path (chunk_kda) | ~3 | < 3 hours |

**Notes:**
- `scale=1.0` is passed to `chunk_kda` because Q and K are already L2-normalized by `_feature_map` before entering the KDA branch — no further scaling is needed.
- `use_qk_l2norm_in_kernel=False` for the same reason.
- `initial_state` is passed as `float32` zeros, as required by the `chunk_kda` API.
- Padding is handled the same way as before: K and V are zeroed for padded positions before the kernel call.

### `einops` Missing Dependency Fix — March 10, 2026

**Problem:** The `_FLA_AVAILABLE` flag was always `False` even when the `flash-linear-attention` library was bundled correctly. All `linear_kda_fla` experiments silently fell back to the PyTorch path, matching the `linear_kda_v2` speed (~0.67–0.82 it/s, ~4–5h/epoch).

**Root cause:** `from fla.ops.kda import chunk_kda` triggers `fla/__init__.py`, which imports `fla.layers.abc`, which requires `einops`. Since `einops` was not in the project dependencies, a `ModuleNotFoundError` was raised and silently caught:

```python
try:
    from fla.ops.kda import chunk_kda as _chunk_kda
    _FLA_AVAILABLE = True
except ImportError:          # ModuleNotFoundError is a subclass
    _FLA_AVAILABLE = False   # ← always landed here
```

**Fix:** Added `einops` to `pyproject.toml` and `requirements.txt`. Also added a startup print to make FLA status visible in job logs:

```python
print(f"[cepe] FLA available: {_FLA_AVAILABLE}", flush=True)
```

**Confirmed speedup** (L40S GPU, ModernBERT-large 400M + TinyLlama 1B, batch size 8, March 10 runs):

| Strategy | Fallback speed | FLA kernel speed | Speedup | Epoch time |
|----------|---------------|-----------------|---------|------------|
| Frozen (adapters only) | ~0.82 it/s | **~4.88 it/s** | **~6×** | ~37 min |
| CEPE 400M encoder | ~0.71 it/s | **~3.70 it/s** | **~5.2×** | ~49 min |
| CEPE 150M encoder | ~0.75 it/s | **~4.14 it/s** | **~5.5×** | ~44 min |
| Finetune (full model) | ~0.67 it/s | **~1.17 it/s** | **~1.75×** | ~2.6h |

The CEPE variants are now **faster than the softmax baseline** (~2.07 it/s, ~1.5h/epoch). The finetune speedup is smaller because the full encoder + decoder backward pass dominates over the adapter computation.
