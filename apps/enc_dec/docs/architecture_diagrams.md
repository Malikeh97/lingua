# Encoder-Decoder Architecture Diagrams

This document provides visual diagrams for the three encoder-decoder variants implemented in this codebase.

---

## 1. Standard Encoder-Decoder (`enc_dec.py`)

The base architecture for sequence-to-sequence tasks with vocabulary-based generation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STANDARD ENC-DEC                                   │
│                                                                              │
│  Input: "The capital of France is Paris"    Target: "What is the capital?"  │
└─────────────────────────────────────────────────────────────────────────────┘

                                    OUTPUT
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │    Vocabulary Softmax    │
                        │   P(token) over vocab    │
                        │     [B, seq, vocab]      │
                        └────────────┬────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │      Linear Output       │
                        │    (vocab projection)    │
                        └────────────┬────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │        RMSNorm           │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              │         ┌────────────┴────────────┐         │
              │         │                         │         │
              │         │    DECODER BLOCK x N    │         │
              │         │                         │         │
              │         │  ┌───────────────────┐  │         │
              │         │  │   Feed-Forward    │  │         │
              │         │  │   + RMSNorm       │  │         │
              │         │  └─────────┬─────────┘  │         │
              │         │            │            │         │
              │         │  ┌─────────┴─────────┐  │         │
              │         │  │  Cross-Attention  │◄─┼─────────┤
              │         │  │   + RMSNorm       │  │         │
              │         │  └─────────┬─────────┘  │         │
              │         │            │            │         │
              │         │  ┌─────────┴─────────┐  │         │
              │         │  │  Self-Attention   │  │         │
              │         │  │  (Causal + RoPE)  │  │         │
              │         │  │   + RMSNorm       │  │         │
              │         │  └─────────┬─────────┘  │         │
              │         │            │            │         │
              │         └────────────┴────────────┘         │
              │                      │                      │
              │         ┌────────────┴────────────┐         │
              │         │   Token Embeddings      │         │
              │         │   + RoPE Embeddings     │         │
              │         └────────────┬────────────┘         │
              │                      │                      │
              │                      ▲                      │
              │               DECODER INPUT                 │
              │           [question tokens]                 │
              │                                             │
              │                                             │
    ENCODER OUTPUT ─────────────────────────────────────────┘
    [B, enc_seq, D]
          │
          │
┌─────────┴─────────┐
│     RMSNorm       │
└─────────┬─────────┘
          │
┌─────────┴─────────┐
│                   │
│  ENCODER BLOCK    │
│      x N          │
│                   │
│ ┌───────────────┐ │
│ │ Feed-Forward  │ │
│ │  + RMSNorm    │ │
│ └───────┬───────┘ │
│         │         │
│ ┌───────┴───────┐ │
│ │Self-Attention │ │
│ │ (Bidir+RoPE)  │ │
│ │  + RMSNorm    │ │
│ └───────┬───────┘ │
│         │         │
└─────────┴─────────┘
          │
┌─────────┴─────────┐
│ Token Embeddings  │
│ + RoPE Embeddings │
└─────────┬─────────┘
          │
          ▲
    ENCODER INPUT
   [context tokens]
```

### Key Characteristics:
- **Output**: Probability distribution over vocabulary
- **Loss**: Cross-entropy over vocabulary tokens
- **Use case**: General seq2seq, abstractive generation

---

## 2. Pointer Network (`enc_dec_pointer.py`)

Pure extractive architecture that points to encoder positions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POINTER NETWORK                                    │
│                                                                              │
│  Input: "The capital of France is Paris"    Target: position indices [5]    │
└─────────────────────────────────────────────────────────────────────────────┘

                                    OUTPUT
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   Position Softmax       │
                        │  P(pos) over enc_seq     │
                        │    [B, dec, enc_seq]     │
                        └────────────┬────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │     Pointer Logits       │
                        │                          │
                        │  dec_proj @ enc_proj.T   │
                        │  ─────────────────────   │
                        │      sqrt(dim)           │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                │                ▼
        ┌───────────────────┐        │    ┌───────────────────┐
        │  Decoder Pointer  │        │    │  Encoder Pointer  │
        │    Projection     │        │    │    Projection     │
        │  [B,dec,ptr_dim]  │        │    │  [B,enc,ptr_dim]  │
        └─────────┬─────────┘        │    └─────────┬─────────┘
                  │                  │              │
                  │                  │              │
        ┌─────────┴─────────┐        │    ┌─────────┴─────────┐
        │     RMSNorm       │        │    │   Encoder Output  │
        └─────────┬─────────┘        │    │   [B,enc_seq,D]   │
                  │                  │    └─────────┬─────────┘
                  │                  │              │
    ┌─────────────┴─────────────┐    │              │
    │                           │    │              │
    │     DECODER BLOCK x N     │    │              │
    │                           │    │              │
    │  ┌─────────────────────┐  │    │              │
    │  │    Feed-Forward     │  │    │              │
    │  │     + RMSNorm       │  │    │              │
    │  └──────────┬──────────┘  │    │              │
    │             │             │    │              │
    │  ┌──────────┴──────────┐  │    │              │
    │  │   Cross-Attention   │◄─┼────┼──────────────┤
    │  │     + RMSNorm       │  │    │              │
    │  └──────────┬──────────┘  │    │              │
    │             │             │    │              │
    │  ┌──────────┴──────────┐  │    │              │
    │  │   Self-Attention    │  │    │              │
    │  │  (Causal + RoPE)    │  │    │              │
    │  │     + RMSNorm       │  │    │              │
    │  └──────────┬──────────┘  │    │              │
    │             │             │    │              │
    └─────────────┴─────────────┘    │              │
                  │                  │              │
    ┌─────────────┴─────────────┐    │              │
    │     Token Embeddings      │    │              │
    └─────────────┬─────────────┘    │              │
                  │                  │              │
                  ▲                  │              │
            DECODER INPUT            │         ENCODER
          [question tokens]          │         (same as
                                     │          standard)
                                     │              │
                                     └──────────────┘
```

### Pointer Computation Detail:
```
┌────────────────────────────────────────────────────────────────┐
│                     POINTER MECHANISM                          │
│                                                                │
│   Decoder Hidden:  h_dec  [B, dec_seq, D]                     │
│   Encoder Hidden:  h_enc  [B, enc_seq, D]                     │
│                                                                │
│   Step 1: Project (optional)                                   │
│           h_dec_proj = Linear(h_dec)  →  [B, dec_seq, ptr_dim]│
│           h_enc_proj = Linear(h_enc)  →  [B, enc_seq, ptr_dim]│
│                                                                │
│   Step 2: Dot Product                                          │
│           logits = h_dec_proj @ h_enc_proj.T                   │
│                  = [B, dec_seq, enc_seq]                       │
│                                                                │
│   Step 3: Scale & Mask                                         │
│           logits = logits / sqrt(ptr_dim)                      │
│           logits[padding] = -inf                               │
│                                                                │
│   Step 4: Softmax → P(position | decoder_state)               │
│           probs = softmax(logits, dim=-1)                      │
│                                                                │
│   Loss: CrossEntropy(logits, target_positions)                 │
└────────────────────────────────────────────────────────────────┘
```

### Key Characteristics:
- **Output**: Probability distribution over encoder positions
- **Loss**: Cross-entropy over position indices
- **Cannot generate OOV**: Output must exist in input
- **Use case**: Pure extraction (spans from context)

---

## 3. Copy Mechanism (`enc_dec_copy.py`)

Hybrid architecture that blends vocabulary generation with copying from encoder.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COPY MECHANISM                                     │
│                                                                              │
│  Input: "The capital of France is Paris"    Target: "Paris" (token or copy) │
└─────────────────────────────────────────────────────────────────────────────┘

                                    OUTPUT
                                      │
                                      ▼
              ┌───────────────────────────────────────────────┐
              │              FINAL DISTRIBUTION                │
              │                                                │
              │   P_final = (1 - g) * P_gen + g * P_copy       │
              │                                                │
              │            [B, dec_seq, vocab_size]            │
              └───────────────────────┬───────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │    P_gen        │      │   Copy Gate g   │      │    P_copy       │
   │   (Generate)    │      │                 │      │    (Copy)       │
   │                 │      │  g = σ(Linear)  │      │                 │
   │ softmax(logits) │      │    g ∈ [0,1]    │      │  scatter_add    │
   │ [B,dec,vocab]   │      │   [B,dec,1]     │      │ [B,dec,vocab]   │
   └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
            │                        │                        │
            │                        │                        │
   ┌────────┴────────┐      ┌────────┴────────┐      ┌────────┴────────┐
   │  Vocab Linear   │      │  Gate Linear    │      │  Cross-Attn     │
   │  [D → vocab]    │      │    [D → 1]      │      │   Weights       │
   └────────┬────────┘      └────────┬────────┘      │  [B,dec,enc]    │
            │                        │               └────────┬────────┘
            │                        │                        │
            └────────────────────────┴────────────────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │        RMSNorm          │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              │    ┌─────────────────┴─────────────────┐    │
              │    │                                   │    │
              │    │      COPY DECODER BLOCK x N       │    │
              │    │                                   │    │
              │    │    ┌───────────────────────┐      │    │
              │    │    │     Feed-Forward      │      │    │
              │    │    │      + RMSNorm        │      │    │
              │    │    └───────────┬───────────┘      │    │
              │    │                │                  │    │
              │    │    ┌───────────┴───────────┐      │    │
              │    │    │    Cross-Attention    │◄─────┼────┤
              │    │    │     + RMSNorm         │      │    │
              │    │    │                       │      │    │
              │    │    │  *** Returns Attn *** │──────┼────┼──► Attn Weights
              │    │    │  *** Weights for  *** │      │    │    (for P_copy)
              │    │    │  *** Copy Mech    *** │      │    │
              │    │    └───────────┬───────────┘      │    │
              │    │                │                  │    │
              │    │    ┌───────────┴───────────┐      │    │
              │    │    │    Self-Attention     │      │    │
              │    │    │   (Causal + RoPE)     │      │    │
              │    │    │      + RMSNorm        │      │    │
              │    │    └───────────┬───────────┘      │    │
              │    │                │                  │    │
              │    └────────────────┴──────────────────┘    │
              │                     │                       │
              │         ┌───────────┴───────────┐           │
              │         │    Token Embeddings   │           │
              │         └───────────┬───────────┘           │
              │                     │                       │
              │                     ▲                       │
              │              DECODER INPUT                  │
              │            [question tokens]                │
              │                                             │
    ENCODER OUTPUT ─────────────────────────────────────────┘
    [B, enc_seq, D]              +
                         ENCODER INPUT IDS
                         [B, enc_seq]
                         (for scatter_add)
```

### Copy Mechanism Detail:
```
┌────────────────────────────────────────────────────────────────────────────┐
│                         COPY DISTRIBUTION                                   │
│                                                                             │
│   Cross-Attention Weights:  attn  [B, dec_seq, enc_seq]                    │
│   Encoder Token IDs:        ids   [B, enc_seq]                             │
│                                                                             │
│   Step 1: Initialize copy distribution                                      │
│           P_copy = zeros([B, dec_seq, vocab_size])                         │
│                                                                             │
│   Step 2: Scatter-add attention weights to vocabulary positions            │
│           For each encoder position j:                                      │
│               token_id = ids[b, j]                                          │
│               P_copy[b, t, token_id] += attn[b, t, j]                      │
│                                                                             │
│   Example:                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Encoder: ["The", "capital", "of", "France", "is", "Paris"]         │  │
│   │  IDs:     [  100,      200,   50,     300,   75,    400  ]          │  │
│   │                                                                      │  │
│   │  Attn weights for decoder pos t: [0.1, 0.1, 0.05, 0.15, 0.1, 0.5]   │  │
│   │                                                                      │  │
│   │  After scatter_add:                                                  │  │
│   │    P_copy[t, 100] = 0.1   ("The")                                   │  │
│   │    P_copy[t, 200] = 0.1   ("capital")                               │  │
│   │    P_copy[t, 50]  = 0.05  ("of")                                    │  │
│   │    P_copy[t, 300] = 0.15  ("France")                                │  │
│   │    P_copy[t, 75]  = 0.1   ("is")                                    │  │
│   │    P_copy[t, 400] = 0.5   ("Paris")  ← highest probability          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Step 3: Blend with generation distribution                                │
│           g = sigmoid(gate_linear(decoder_hidden))                         │
│           P_final = (1 - g) * P_gen + g * P_copy                           │
│                                                                             │
│   Loss: -log(P_final[target_token])                                        │
└────────────────────────────────────────────────────────────────────────────┘
```

### Key Characteristics:
- **Output**: Blended probability over vocabulary
- **Loss**: NLL using blended distribution
- **Can generate OOV**: Through P_gen component
- **Use case**: Mixed extraction/generation (summarization, QA)

---

## Architecture Comparison

```
┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│     Aspect      │    Standard      │     Pointer      │      Copy        │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Output Space    │ Vocabulary       │ Encoder positions│ Vocabulary       │
│                 │ [vocab_size]     │ [enc_seq_len]    │ [vocab_size]     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Can Generate    │ Yes              │ No               │ Yes              │
│ New Tokens      │                  │                  │                  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Extraction      │ Implicit         │ Explicit         │ Explicit         │
│ Capability      │ (learned)        │ (position-based) │ (attention-based)│
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Output Head     │ Linear→vocab     │ Dot product      │ Linear + Gate    │
│                 │                  │ + projection     │ + Scatter        │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Loss Function   │ CE(vocab)        │ CE(positions)    │ NLL(blended)     │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Best For        │ Abstractive      │ Pure extractive  │ Mixed tasks      │
│                 │ generation       │ (spans)          │ (QA, summarize)  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Parameters      │ Base model       │ + Pointer proj   │ + Gate linear    │
│ Added           │                  │ (optional)       │                  │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## Shared Encoder Architecture

All three variants can use the same encoder options:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENCODER OPTIONS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Option 1: TRAINABLE ENCODER
┌─────────────────────────┐
│   Custom Transformer    │
│   - N encoder layers    │
│   - Bidirectional attn  │
│   - RoPE embeddings     │
│   - Trained from scratch│
└─────────────────────────┘

Option 2: PRETRAINED ENCODER (e.g., ModernBERT)
┌─────────────────────────┐
│   HuggingFace Model     │
│   - Frozen or partial   │
│   - Projection layer    │
│   - Own tokenizer       │
│   - Flash attention     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Dimension Projection  │
│   [encoder_dim → dim]   │
│   (if dims differ)      │
└─────────────────────────┘

Option 3: EMBEDDING ONLY
┌─────────────────────────┐
│   Just Embeddings       │
│   - No transformer      │
│   - Direct to decoder   │
│   - Lightweight         │
└─────────────────────────┘
```

---

## File Structure

```
apps/enc_dec/
├── enc_dec.py          # Standard encoder-decoder (base classes)
├── enc_dec_pointer.py  # Pointer network variant
├── enc_dec_copy.py     # Copy mechanism variant
├── train.py            # Training script (standard)
├── train_pointer.py    # Training script (pointer)
├── train_copy.py       # Training script (copy)
├── data.py             # Data loading (shared)
├── data_pointer.py     # Data loading (pointer-specific)
└── configs/
    ├── mvp_*.yaml           # Standard configs
    ├── extractive_pointer.yaml
    └── extractive_copy.yaml
```
