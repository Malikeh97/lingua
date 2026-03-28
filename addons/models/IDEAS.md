# Model Ideas Explored

Summary of architectural ideas explored in `apps/minimal_squad` and `apps/enc_dec`.

---

## 1. Model Types

| Idea | Source | Description |
|------|--------|-------------|
| **Encoder-Decoder** | both | Separate encoder and decoder with cross-attention |
| **Decoder-only** | minimal_squad | Standard autoregressive decoder (baseline) |

---

## 2. Encoder Variants

| Idea | Source | Description |
|------|--------|-------------|
| **Pretrained encoder** | both | HuggingFace models (ModernBERT, DeBERTa) |
| **From-scratch encoder** | enc_dec | Custom transformer encoder |
| **Embedding-only encoder** | enc_dec | Just embeddings, no transformer layers |
| **Frozen encoder** | both | Freeze pretrained weights |
| **Encoder LR scaling** | minimal_squad | Train encoder at fraction of main LR |

### Encoder Attention

| Idea | Source | Description |
|------|--------|-------------|
| **Global attention** | both | Full bidirectional attention |
| **Local attention** | minimal_squad | Segment-isolated attention for bottom layers |
| **Block-causal local** | minimal_squad | Segment 1 bidirectional, segment 2 causal |
| **Per-layer attention** | perlayer_attn.py | Extract per-layer hidden states for analysis |

---

## 3. Decoder Variants

| Idea | Source | Description |
|------|--------|-------------|
| **Pretrained decoder** | minimal_squad | HuggingFace LLM (TinyLlama, etc.) with adapters |
| **From-scratch decoder** | both | Custom transformer decoder |
| **Frozen decoder** | minimal_squad | Freeze pretrained, train only adapters |

---

## 4. Cross-Attention

### Attention Type

| Idea | Source | Description |
|------|--------|-------------|
| **Softmax** | both | Standard scaled dot-product attention |
| **Linear (parallel)** | minimal_squad | S = K^T V state, O(d²) memory |
| **Linear KDA** | minimal_squad | Delta-rule recurrence with FLA kernel |

### Adapter Design

| Idea | Source | Description |
|------|--------|-------------|
| **Zero-init output** | minimal_squad | out_proj starts at 0 → no-op at init |
| **Projection layer** | both | Linear(enc_dim → dec_dim) to bridge dimensions |
| **Per-layer injection** | both | Cross-attn between self-attn and FFN |
| **Selective layers** | enc_dec | Cross-attn only at specific layers |

### Feature Maps (Linear Attention)

| Idea | Source | Description |
|------|--------|-------------|
| **Swish + L2 norm** | minimal_squad | Feature map for Q, K |
| **Decay gates** | minimal_squad | Per-element log-space gates for KDA |
| **Write gates (beta)** | minimal_squad | Per-head gating for state updates |

---

## 5. Output Heads

| Idea | Source | Description |
|------|--------|-------------|
| **LM head** | both | Standard next-token prediction |
| **Span head (BERT-like)** | minimal_squad | Linear layers for start/end logits |
| **Span head (first_last_hidden)** | minimal_squad | Concat first+last hidden states |
| **Span head (attention-based)** | minimal_squad | Weighted attention across layers |
| **Pointer mechanism** | enc_dec | Copy from encoder positions |
| **Copy mechanism** | enc_dec | Blend generated vs copied tokens |

---

## 6. Data Handling

| Idea | Source | Description |
|------|--------|-------------|
| **Dual tokenizers** | both | Separate encoder/decoder vocabularies |
| **Data format DSL** | minimal_squad | "C/Q//A" format strings |
| **Segment separation** | minimal_squad | SEP tokens between parts |
| **Span extraction** | minimal_squad | Character→token offset mapping |

---

## 7. Training

| Idea | Source | Description |
|------|--------|-------------|
| **Pretrained weight scaling** | minimal_squad | Scale LR for pretrained params |
| **Encoder/decoder LR split** | minimal_squad | Different LR for each |
| **FSDP grouping** | enc_dec | Layer-wise sharding plan |
| **Activation recompute** | enc_dec | Selective recomputation |

---

## 8. What Matters (from experiments)

Based on the code structure and experiment scripts:

### Definitely important:
- Cross-attention type (softmax vs linear vs linear_kda)
- Encoder local attention ratio
- Zero-init adapters

### Maybe important:
- Encoder/decoder LR ratio
- Number of cross-attention layers
- Span expression method

### Probably not important:
- Specific feature map choice (swish + L2 seems standard)
- Pointer vs copy mechanism (task-specific)
