# CEPE Architecture: Cross-Encoder Parallel Extension

## 1. Overview

CEPE grafts a frozen pre-trained encoder (e.g. ModernBERT) onto a frozen pre-trained
decoder (e.g. TinyLlama) by inserting a learned **cross-attention adapter** between
every decoder layer's self-attention and FFN blocks. A linear projection bridges the
encoder's hidden dimension to the decoder's. Because the adapter's output projection
is zero-initialized, the composite model starts as the original decoder and gradually
learns to consume encoder representations — no catastrophic forgetting, minimal new
parameters.

Implementation: `main.py`, class `UnifiedModel` with `decoder_model_name` set,
backed by `CrossAttentionAdapter`.

---

## 2. Architecture Diagram

```
                       ENCODER SIDE                           DECODER SIDE
                  ┌─────────────────────┐         ┌──────────────────────────────────┐
                  │  Encoder Tokenizer   │         │     Decoder Tokenizer            │
                  │  (ModernBERT vocab)  │         │     (LLaMA vocab)                │
                  └─────────┬───────────┘         └────────────┬─────────────────────┘
                            │                                  │
                            ▼                                  ▼
                  ┌─────────────────────┐         ┌──────────────────────────────────┐
                  │  encoder_input_ids   │         │     decoder_input_ids            │
                  │  [B, enc_len]        │         │     [B, dec_len]                 │
                  └─────────┬───────────┘         └────────────┬─────────────────────┘
                            │                                  │
                            ▼                                  ▼
                  ┌─────────────────────┐         ┌──────────────────────────────────┐
                  │  ModernBERT Encoder  │         │   embed_tokens (frozen)          │
                  │  (frozen or trained) │         │   [B, dec_len, dec_hidden]       │
                  │  N encoder layers    │         └────────────┬─────────────────────┘
                  └─────────┬───────────┘                      │
                            │                                  │ + rotary_emb(positions)
                            ▼                                  ▼
                  ┌─────────────────────┐
                  │  encoder hidden      │           ┌─── Decoder Layer i (×N) ───┐
                  │  [B, enc_len,        │           │                             │
                  │   enc_hidden]        │           │  ┌─────────────────────┐    │
                  │  (float32)           │           │  │ input_layernorm     │    │
                  └─────────┬───────────┘           │  │ self_attn + RoPE    │    │
                            │                        │  │ residual add        │    │
                            ▼                        │  │ (FROZEN)            │    │
                  ┌─────────────────────┐           │  └────────┬────────────┘    │
                  │  encoder_projection  │           │           │                 │
                  │  Linear(enc→dec,     │           │           ▼                 │
                  │         bias=False)  │           │  ┌─────────────────────┐    │
                  │  → bfloat16 cast     │           │  │ CrossAttentionAdapt │    │
                  └─────────┬───────────┘           │  │  ┌───────────────┐  │    │
                            │                        │  │  │ LayerNorm     │  │    │
                            ▼                        │  │  │ MHA(Q=dec,    │──┼────┼─── K,V from
                  ┌─────────────────────┐           │  │  │     K,V=enc)  │  │    │    projected
                  │  projected_encoder   │──────────▶│  │  │ residual add  │  │    │    encoder
                  │  [B, enc_len,        │           │  │  └───────────────┘  │    │
                  │   dec_hidden]        │           │  │  (TRAINED)          │    │
                  │  (bfloat16)          │           │  └────────┬────────────┘    │
                  └──────────────────────┘           │           │                 │
                                                     │           ▼                 │
                                                     │  ┌─────────────────────┐    │
                                                     │  │ post_attn_layernorm │    │
                                                     │  │ mlp (SwiGLU)        │    │
                                                     │  │ residual add        │    │
                                                     │  │ (FROZEN)            │    │
                                                     │  └────────┬────────────┘    │
                                                     │           │                 │
                                                     └───────────┼─────────────────┘
                                                                 │  (repeat N layers)
                                                                 ▼
                                                     ┌──────────────────────────┐
                                                     │  decoder_final_norm      │
                                                     │  (frozen RMSNorm)        │
                                                     └────────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                     ┌──────────────────────────┐
                                                     │  decoder_lm_head         │
                                                     │  Linear(dec_hidden,      │
                                                     │         dec_vocab)       │
                                                     │  (frozen)                │
                                                     └────────────┬─────────────┘
                                                                  │
                                                                  ▼
                                                          logits [B, dec_len,
                                                                  dec_vocab]
```

---

## 3. Components

### 3.1 Encoder — ModernBERT (or DeBERTa)

| Property | Value |
|----------|-------|
| Loaded via | `AutoModel.from_pretrained(encoder_name, torch_dtype=bfloat16)` |
| Output | `.last_hidden_state` cast to **float32** |
| Shape | `[B, enc_len, enc_hidden]` |
| Typical sizes | ModernBERT-base: 768, ModernBERT-large: 1024 |
| Default state | Frozen (`p.requires_grad = False`) |

**Reference:** `main.py:894-897` (loading), `main.py:1247-1288` (`_run_encoder`).

Optional: when `enc_local_layer_ratio > 0`, bottom encoder layers use segment-local
attention masks that block attention across `[SEP]` boundaries. Controlled per-layer
in `_run_encoder` slow path (`main.py:1258-1288`).

### 3.2 Encoder Projection

```python
self.encoder_projection = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)
```

| Property | Value |
|----------|-------|
| Input | `[B, enc_len, enc_hidden]` (float32) |
| Output | `[B, enc_len, dec_hidden]` cast to **bfloat16** |
| Init | Truncated normal, std = `dec_hidden ** -0.5`, bounds `[-3*std, 3*std]` |
| State | Always trained |

**Reference:** `main.py:1001-1004` (creation), `main.py:1055-1058` (forward), `main.py:1173-1180` (init).

### 3.3 Cross-Attention Adapter

One instance per decoder layer, stored in `self.cross_attn_adapters` (`nn.ModuleList`).

```
Architecture:  residual ──────────────────────────── (+) ── output
                  │                                   ▲
                  ▼                                   │
             LayerNorm(dec_hidden)                    │
                  │                                   │
                  ▼                                   │
             MultiheadAttention                       │
               Q = decoder hidden                     │
               K = projected encoder                  │
               V = projected encoder                  │
                  │                                   │
                  └───────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| Class | `CrossAttentionAdapter` (`main.py:722-771`) |
| Hidden size | `dec_hidden_size` (e.g. 2048 for TinyLlama) |
| Heads | Configurable via `--cross_attn_num_heads` (default 16) |
| dtype | bfloat16 (cast at `main.py:1036`) |
| Q/K/V init | Truncated normal, std = `dec_hidden ** -0.5` |
| Output proj init | **Zeros** (weight and bias) — adapter starts as identity/no-op |
| LayerNorm init | weight = 1, bias = 0 |

**Why zero-init output?** The zero output projection means the adapter contributes
nothing at the start of training. The model begins as the unmodified pre-trained
decoder and gradually learns to attend to encoder information. This prevents
catastrophic forgetting of the decoder's pre-trained capabilities.

**Reference:** `main.py:722-771` (class), `main.py:1006-1014` (creation), `main.py:1082-1096` (forward call).

### 3.4 Pre-trained Decoder Layers (Frozen)

Components extracted from a HuggingFace `AutoModelForCausalLM`:

| Component | Source | Attribute |
|-----------|--------|-----------|
| Token embeddings | `model.embed_tokens` | `self.decoder_tok_embeddings` |
| Transformer layers | `model.layers` | `self.pretrained_decoder_layers` |
| Final RMSNorm | `model.norm` | `self.decoder_final_norm` |
| LM head | `lm_head` | `self.decoder_lm_head` |
| Rotary embeddings | `model.rotary_emb` | `self.decoder_rotary_emb` |

All loaded in **bfloat16** and frozen by default.

**Reference:** `main.py:990-1039` (`_init_pretrained_decoder`), `main.py:934-944` (marking pretrained).

### 3.5 Final Norm + LM Head

```python
x = self.decoder_final_norm(x)       # RMSNorm, frozen
logits = self.decoder_lm_head(x)     # Linear(dec_hidden, dec_vocab), frozen
```

**Reference:** `main.py:1104-1106`.

---

## 4. Initialization Strategy

All new parameters use **Lingua-style initialization** (truncated normal with ±3-sigma clipping):

```python
std = hidden_size ** -0.5
factor = (3 * num_layers) ** 0.5

# Q/K/V projections:
nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)

# Output projection (for from-scratch decoder layers):
nn.init.trunc_normal_(weight, mean=0.0, std=std / factor, a=...)

# Cross-attention adapter output projection:
nn.init.zeros_(weight)   # <-- CEPE identity start
nn.init.zeros_(bias)
```

| Parameter | Init | Rationale |
|-----------|------|-----------|
| Adapter Q/K/V (`in_proj_weight`) | Truncated normal, std = `d^{-0.5}` | Standard transformer init, bounded |
| Adapter output proj | **Zeros** | Adapter = no-op at start, gradual learning |
| Adapter LayerNorm | weight=1, bias=0 | Identity transform |
| Encoder projection | Truncated normal, std = `d_dec^{-0.5}` | Matches decoder scale |
| From-scratch decoder output proj | Truncated normal, std = `d^{-0.5} / sqrt(3*L)` | Depth-scaled to prevent gradient explosion |

**Reference:** `main.py:1137-1193` (`init_weights`), `main.py:758-771` (`CrossAttentionAdapter.init_weights`).

---

## 5. Training Modes

Controlled by `--pretrained_weight_updating` (PWU) and `--encoder_weight_updating` (EWU):

| Mode | PWU | EWU | Encoder | Decoder | Adapters + Projection | Description |
|------|-----|-----|---------|---------|-----------------------|-------------|
| **Frozen** (default) | `None`/`0.0` | — | Frozen | Frozen | Full LR | Only adapters and projection are trained |
| **CEPE-style** | `0.0` | `0.333` | 0.333 × LR | Frozen | Full LR | Encoder fine-tuned at reduced LR, decoder frozen |
| **Full fine-tune** | `1.0` | — | Full LR | Full LR | Full LR | Everything trained at the same LR |
| **Mixed** | `0.1` | `0.5` | 0.5 × LR | 0.1 × LR | Full LR | Independent scaling for encoder vs decoder |

Parameter groups are assembled in `main()` at `main.py:2256-2314`:

```
optimizer_grouped_parameters = [
    {"params": pretrained_encoder_params, "lr": base_lr * encoder_updating},   # if > 0
    {"params": pretrained_decoder_params, "lr": base_lr * decoder_updating},   # if > 0
    {"params": other_params,              "lr": base_lr},                      # always
]
```

Parameters are tagged at model creation time:
- `p.is_pretrained = True` + `p.is_pretrained_encoder = True` for encoder params (`main.py:929-931`)
- `p.is_pretrained = True` + `p.is_pretrained_decoder = True` for decoder params (`main.py:934-944`)
- All other params (adapters, projection, span heads) are "other" and always trained

**Reference:** `main.py:2254-2330`.

---

## 6. Dual Tokenizer

When `--decoder_model_name` is set, the system uses two separate tokenizers:

| Tokenizer | Source | Used for | Vocab example |
|-----------|--------|----------|---------------|
| **Encoder** | `AutoTokenizer.from_pretrained(encoder_model)` | Encoder input sequence | ModernBERT: 50,368 tokens |
| **Decoder** | `AutoTokenizer.from_pretrained(decoder_model)` | Decoder input sequence | LLaMA: 32,000 tokens |

Both tokenizers have `infer_special_tokens()` applied to detect BOS/EOS/PAD when not
explicitly configured (needed for ModernBERT which uses CLS/SEP instead).

**Data preparation flow** (`DataPreparer`):

1. Each sample's Q, C, A fields are tokenized by **both** tokenizers (`_tokenize_segments`, `main.py:196-245`)
2. Encoder-side assembly uses encoder tokenizer tokens (e.g., `segments["C"]`)
3. Decoder-side assembly uses decoder tokenizer tokens (e.g., `segments["C_dec"]`)
4. Special token IDs (BOS, EOS) are selected per side:
   ```python
   bos = self.dec_bos_id if side == "decoder" else self.bos_id
   eos = self.dec_eos_id if side == "decoder" else self.eos_id
   ```

**Collation** uses separate pad token IDs: `pad_token_id` for encoder fields,
`dec_pad_token_id` for decoder fields (`main.py:479-536`).

**Reference:** `main.py:2091-2101` (tokenizer loading), `main.py:153-177` (`DataPreparer.__init__`), `main.py:292-296` (side selection).

---

## 7. Data Flow

Step-by-step trace through the forward pass for the CEPE configuration
(`data_format="C//Q/A"`, `model_type="encdec"`, `decoder_model_name="tinyllama_1b"`).

Example dimensions: ModernBERT-base encoder (hidden=768, 22 layers) + TinyLlama-1B
decoder (hidden=2048, 22 layers, 32 heads, vocab=32000).

### Step 1 — Encoder

```
encoder_input_ids    [B, enc_len]           # int64, ModernBERT token IDs
         │
         ▼
    ModernBERT(input_ids, attention_mask)
         │
         ▼
encoder_hidden       [B, enc_len, 768]      # float32 (.last_hidden_state.float())
```

**Reference:** `main.py:1393`, `main.py:1253-1256`.

### Step 2 — Projection

```
encoder_hidden       [B, enc_len, 768]      # float32
         │
         ▼
    encoder_projection (Linear 768→2048, no bias)
         │
         ▼
    .to(bfloat16)
         │
         ▼
projected_encoder    [B, enc_len, 2048]     # bfloat16
```

**Reference:** `main.py:1055-1058`.

### Step 3 — Decoder Embedding

```
decoder_input_ids    [B, dec_len]           # int64, LLaMA token IDs
         │
         ▼
    embed_tokens (frozen, bfloat16)
         │
         ▼
x                    [B, dec_len, 2048]     # bfloat16

position_ids         [1, dec_len]           # arange(dec_len)
         │
         ▼
    rotary_emb(x, position_ids)
         │
         ▼
position_embeddings  (cos, sin)             # for RoPE in self-attention
```

**Reference:** `main.py:1060-1069`.

### Step 4 — Decoder Layer Loop (×22 layers)

For each `(layer, adapter)` in `zip(pretrained_decoder_layers, cross_attn_adapters)`:

```
   ┌──────────────────────────────────────────────────────┐
   │  4a. Self-Attention (frozen)                          │
   │                                                       │
   │  residual = x                     [B, dec_len, 2048]  │
   │  x = input_layernorm(x)           [B, dec_len, 2048]  │
   │  x = self_attn(x, position_embeddings, use_cache=F)   │
   │  x = residual + x                [B, dec_len, 2048]   │
   ├───────────────────────────────────────────────────────┤
   │  4b. Cross-Attention Adapter (trained)                │
   │                                                       │
   │  residual = x                     [B, dec_len, 2048]  │
   │  x = cross_attn_norm(x)           [B, dec_len, 2048]  │
   │  x = MHA(Q=x, K=proj_enc, V=proj_enc)                │
   │       key_padding_mask = ~encoder_attention_mask       │
   │       16 heads, head_dim = 128                         │
   │  x = residual + x                [B, dec_len, 2048]   │
   ├───────────────────────────────────────────────────────┤
   │  4c. FFN (frozen, SwiGLU)                             │
   │                                                       │
   │  residual = x                     [B, dec_len, 2048]  │
   │  x = post_attention_layernorm(x)  [B, dec_len, 2048]  │
   │  x = mlp(x)                       [B, dec_len, 2048]  │
   │  x = residual + x                [B, dec_len, 2048]   │
   └───────────────────────────────────────────────────────┘
```

**Reference:** `main.py:1073-1102`.

### Step 5 — Output

```
x                    [B, dec_len, 2048]     # bfloat16
         │
         ▼
    decoder_final_norm (RMSNorm, frozen)
         │
         ▼
x                    [B, dec_len, 2048]
         │
         ▼
    decoder_lm_head (Linear 2048→32000, frozen)
         │
         ▼
logits               [B, dec_len, 32000]
```

**Reference:** `main.py:1104-1106`.

### Step 6 — Loss

```python
# Shifted cross-entropy (teacher forcing)
shift_logits = logits[:, :-1, :]           # [B, dec_len-1, vocab]
shift_labels = labels[:, 1:]               # [B, dec_len-1]
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

**Reference:** `main.py:1352-1362` (`_compute_gen_loss`).

---

## 8. Model Aliases

| Alias | HuggingFace ID | Role |
|-------|----------------|------|
| `modernbert_150m` | `answerdotai/ModernBERT-base` | Encoder |
| `modernbert_400m` | `answerdotai/ModernBERT-large` | Encoder |
| `deberta_v3_300m` | `microsoft/deberta-v3-large` | Encoder |
| `tinyllama_1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Decoder |
| `llama3.2_1b` | `meta-llama/Llama-3.2-1B` | Decoder |
| `llama3.2_3b` | `meta-llama/Llama-3.2-3B` | Decoder |
| `llama3.1_8b` | `meta-llama/Llama-3.1-8B` | Decoder |

**Reference:** `main.py:542-557`.
