# Linear Cross-Attention Adapters for CEPE (March 3, 2026)

## Motivation

The existing `CrossAttentionAdapter` uses `nn.MultiheadAttention` (softmax), which computes full attention over all encoder tokens at each decoder step. This is O(T_enc × T_dec) in memory and time. For long encoder sequences (thousands of tokens) this becomes a bottleneck.

Inspired by the KDA (Kernel Delta-rule Attention) line of work, two linear-complexity alternatives are added under a new `--cross_attn_type` flag. Both compress the encoder sequence into a fixed-size matrix memory state **S** of shape `(B, H, d_k, d_k)`, making the cost O(T_enc) for encoding and O(T_dec) for decoding, independent of the other.

---

## New Flag

```
--cross_attn_type  {softmax, linear, linear_kda}   (default: softmax)
```

- `softmax`: existing behavior, unchanged.
- `linear`: new parallel variant.
- `linear_kda`: new KDA-style chunked delta-rule variant.

**Incompatibility**: `--cross_attn_type linear` or `linear_kda` cannot be combined with `--span_expr first_last_attn`, because linear attention produces no explicit weight matrix. An `AssertionError` is raised at model init if this combination is attempted. Use `first_last_hidden` instead.

---

## Architecture: `LinearCrossAttentionAdapter`

Both variants share the same projections and feature map, and follow the same insertion point as the softmax adapter: between self-attention and FFN of each frozen decoder layer.

### Projections (all `Linear(D, D, bias=False)`)
- `q_proj`, `k_proj`, `v_proj`, `out_proj`
- `cross_attn_norm`: `LayerNorm(D)`

### Feature Map
Q and K are passed through **Swish + L2-normalize** along `head_dim`:
```
φ(x) = normalize(silu(x), dim=-1)
```
This ensures the kernel approximation K^T @ V corresponds to a valid (positive semi-definite) similarity measure. V uses Swish only (no normalization needed for the value slot).

### Zero-Init (critical for CEPE)
`out_proj.weight` is zero-initialized so the adapter starts as a no-op, preserving the pretrained decoder's behavior at step 0. This is identical to the softmax adapter convention.

---

## Variant A: `linear` (Parallel)

The entire encoder sequence is compressed into one matrix-vector pair in a single pass:

```
K, V  ←  zero-out padded positions
S     =  K^T @ V            # (B, H, d_k, d_k)  — association matrix
z     =  K.sum(dim=seq)     # (B, H, d_k)        — normalizer

O     =  Q @ S              # (B, H, T_dec, d_k)
denom =  (Q @ z).clamp(1e-6)
O     =  O / denom
```

**Complexity**: O(T_enc · d_k²) to build S, O(T_dec · d_k²) to decode. No sequential dependency → fully parallelizable on GPU.

**Padding**: K and V are multiplied element-wise by the validity mask before the matmul, so padded encoder tokens contribute nothing to S or z.

---

## Variant B: `linear_kda` (Chunked Delta-Rule, FLA-inspired)

Instead of adding all encoder tokens uniformly, this variant processes them with a gated delta-rule write: each new token corrects the memory only for what it adds above what is already encoded. The implementation is adapted from the [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) KDA ops (`fla/ops/kda/naive.py`), rearchitected for an encoder-only state build followed by decoder cross-read.

### Additional Parameters

```
g_proj:    Linear(D, H * d_k, bias=True)    → per-element log-space decay gate g ≤ 0
beta_proj: Linear(D, H,        bias=True)   → per-head write gate β ∈ (0, 1)
```

Both are zero-initialized. At init: `logsigmoid(0) = −log 2 ≈ −0.693` → moderate decay; `sigmoid(0) = 0.5` → half-rate write. This is a neutral, symmetric starting point.

**Gate design (FLA convention):**
- `g_log = logsigmoid(g_proj(enc))` — shape `[B, T_enc, H×d_k]`, reshaped to `[B, H, T_enc, d_k]`
- `exp(g_log) ≤ 1` is mathematically guaranteed — no sigmoid ambiguity
- Per-element (one decay rate per head-dim slot), replacing the old rank-bottleneck `alpha_down/alpha_up` pair which had only `d_k/4 = 16` independent decay modes

### Chunked Recurrence (encoder state build)

The encoder sequence is processed in chunks of `CHUNK = 64` tokens. Within each chunk, the delta-rule is solved exactly via a `BT × BT` intra-chunk interaction matrix and forward substitution. Across chunks, the state `S` propagates sequentially (`NT = T_enc / 64 ≈ 8` steps for typical SQuAD sequences).

**Per-chunk steps** (for chunk tokens `t_start .. t_end`):

```
g_cum  = cumsum(g_c, dim=time)          # [B, H, BT, d_k] — cumulative decay within chunk
K_g    = K_c * exp(g_cum)              # gate-weighted keys

# Intra-chunk delta-rule matrix (strictly lower triangular)
A      = K_c @ K_g.T * beta           # [B, H, BT, BT]
A      = (-A) masked to lower-tri

# Exact forward substitution: A → (I − L)^{−1} − I
for i in 1..BT:
    A[i, :i] = A[i, :i] + (A[i, :] @ A)[:i]

A_inv  = (A + I) * beta               # (I − L)^{−1} scaled by beta (FLA convention)

# Inter-chunk aggregates
w      = A_inv @ K_g                  # effective key sum
u      = A_inv @ V_c                  # effective value sum

# Core delta: subtract what S already encodes
v_new  = u − w @ S

# State decay and write
g_last = g_cum[:, :, −1, :]
S      = S * exp(g_last)[..., None]   # decay along key rows
K_w    = K_c * exp(g_last − g_cum)   # back-weighted keys
S      = S + K_w.T @ v_new           # add new associations
```

**Decode** (all decoder positions read the same final state):
```
O = Q @ S    # [B, H, T_dec, d_k] — no explicit denominator
```
Q and K are L2-normalized by `_feature_map`, providing implicit scale normalization (FLA convention, no separate `z`).

**Complexity**: O(NT · BT² · d_k) for intra-chunk (matrix ops, GPU-friendly) + O(NT · d_k²) for state propagation. Sequential dependency is O(NT) ≈ 8, not O(T_enc) ≈ 512.

**Comparison with old token-by-token loop**:

| Property | Old (token loop + TBPTT) | New (chunked, FLA-style) |
|---|---|---|
| Python loop depth (autograd graph) | T_enc ≈ 512 | NT = T_enc/64 ≈ 8 |
| Gradient accuracy | Truncated at chunk boundaries | **Exact** |
| GPU memory for S graph | O(num_layers × T_enc × d_k²) ≈ 44 GB | O(num_layers × BT²) ≈ 1.4 GB |
| Gate parameterization | Bottleneck rank-16 sigmoid | Direct H×d_k logsigmoid |

---

## OOM Fix (March 3 update)

The first run of `linear_kda_cepe` (TinyLlama 1B, 22 decoder layers, B=8, T_enc≈512) hit a `CUDA OutOfMemoryError` at the very first training step:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 44.40 GiB of which 14.31 MiB is free.
```

**Root cause**: the original token-by-token for-loop kept O(num_layers × T_enc) copies of the state tensor `S` (each `[8, 32, 64, 64]`, ≈ 4 MB fp32) alive simultaneously for backward. With 22 adapter layers and T_enc = 512: 22 × 512 × 4 MB ≈ **44 GB** of saved autograd tensors — exactly filling the GPU before any activations were allocated.

**Fix applied**: replaced the token loop (+ TBPTT detach hack) with the FLA-inspired chunked recurrence described above. The autograd graph depth shrinks from 512 to 8, reducing the state-tensor memory footprint from ~44 GB to ~1.4 GB.

---

## New Experiments: C/Q//S span-extraction with `linear` + CEPE

Added to `cepe_exps.sh` after the `# --- Variant: linear ---` block:

| Run name | `--span_expr` | Notes |
|---|---|---|
| `modernbert400m_tinyllama1b_cq_s_linear_cepe_bertlike` | `bertlike` | Decoder not instantiated for `bertlike`; CEPE LR schedule |
| `modernbert400m_tinyllama1b_cq_s_linear_cepe_first_last_hidden` | `first_last_hidden` | Decoder frozen, instantiated for generation + anchors |

`first_last_attn` is intentionally omitted — incompatible with `linear` attention (raises `AssertionError` at model init). Both use `--data_format C/Q//S` (required when `span_expr != none`) and the CEPE training strategy (`--pretrained_weight_updating 0.0 --encoder_weight_updating 0.333`).

---

## Return API

Both variants match `CrossAttentionAdapter` exactly:

```python
# return_attn=False  (default)
return output                  # Tensor (B, T_dec, D)

# return_attn=True
return output, None            # (Tensor, None) — no explicit weight matrix
```

The `None` in the attention slot is intentional: there is no O(T_enc × T_dec) weight matrix to return.

---

## Parameter Count Comparison

For TinyLlama (D=2048, H=32, d_k=64, 22 layers):

| Adapter type | Extra gate params per layer | Total gate params (22 layers) | Total adapter params |
|---|---|---|---|
| `softmax` | 0 | 0 | ~184M |
| `linear` | 0 | 0 | ~370M |
| `linear_kda` (old) | 2× rank-16 bottleneck + H bias ≈ 68K | ~1.5M | ~372M |
| `linear_kda` (new) | `g_proj`: H×d_k×D = 4.2M + `beta_proj`: H×D = 65K | ~94M | ~464M |

The new `g_proj` is larger (full D → H×d_k matrix) but eliminates the rank bottleneck, giving each of the 32×64 = 2048 decay channels an independent learned rate. This is a deliberate trade: more expressive gates for the same architectural complexity.

---

## Implementation Notes

- **Dtype**: Projections are initialized in float32 and cast to bfloat16 by the existing `self.cross_attn_adapters.to(torch.bfloat16)` call in `_init_pretrained_decoder`. The KDA state `S` is initialized with `dtype=K.dtype` to inherit bfloat16 automatically.
- **Padding (linear)**: K and V are zeroed before the `K^T @ V` matmul; `z` normalizer absorbs the zeroing correctly.
- **Padding (linear_kda)**: K and V are zeroed before the chunked loop; zeroed tokens contribute `v_new ≈ 0` and do not corrupt `S` or `g_last`.
- **All-padding edge case**: S stays zero; `O = Q @ 0 = 0`; residual connection preserves decoder output — same behavior as softmax with fully-masked keys.
- **No ShortConv**: encoder is bidirectional; causal convolution over encoder positions would impose spurious left-to-right ordering — deliberately omitted.
- **FLA kernel upgrade path**: the new gate convention (`g = logsigmoid(g_proj(enc))`, shapes `[B, T, H, K]`) is pin-compatible with `fla.ops.kda.chunk_kda(output_final_state=True)`. Once the FLA library is installed, the chunked PyTorch loop can be swapped for the Triton fused kernel with no architectural changes.

---

## Next Steps

1. Run `linear_kda_cepe` (now fixed) and compare EM/F1 against `linear_cepe` and the softmax baseline (target: 77.6% EM).
2. Run `cq_s_linear_cepe_bertlike` and `cq_s_linear_cepe_first_last_hidden` (newly added) to see if span-extraction with linear attention is competitive with softmax span extraction.
3. Profile wall-clock step time: the chunked recurrence has O(NT × BT²) intra-chunk ops; for BT=64 this is a `[B, H, 64, 64]` matmul repeated 8 times — should be faster than the old 512-step Python loop despite being sequential.
4. Consider installing the FLA library (`pip install flash-linear-attention`) and swapping the chunked PyTorch loop for `chunk_kda(..., output_final_state=True)` to get the Triton-fused kernel with the full custom backward.
