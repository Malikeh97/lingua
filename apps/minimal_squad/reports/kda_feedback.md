# Analysis: Why KDA Linear Attention is Slower Than Softmax in the Current Setup

## Context

The experiments (2430231–2430236) show KDA linear attention is **1.2–1.4× slower** than softmax cross-attention at 12L/22L despite using the FLA Triton kernel library which claims 2.3–2.9× speedup over softmax. This document investigates what parallelizations from Section 3 of the Kimi Linear paper are and are not being realized.

---

## What the Paper Claims (Section 3)

The hardware-efficient chunkwise algorithm in the Kimi Linear paper (Section 3) is built around two levels of parallelism:

### 1. Chunkwise Parallelism
Sequence split into L/C chunks of size C. Intra-chunk operations (Aqk, Akk diagonal blocks) are computed in parallel across chunks. Inter-chunk recurrence (state propagation S[i] ← S[i-1]) is inherently sequential but has only L/C steps.

**Speedup claim**: Reduces non-matmul FLOPs via the UT transform, enabling TensorCore utilization.

### 2. Intra-Token Parallelism
The `chunk_kda_fwd_kernel_intra_token_parallel` kernel assigns **one token per thread block**, maximizing GPU SM occupancy and memory locality within each chunk.

**Speedup claim**: FLOPs(KDA) = 6Td² + 3Tcd + Tc² vs FLOPs(softmax) = 2T²d → linear in T.
At T=512k, this gives 2.3×; at T=1M, 2.9×.

---

## Current Setup — What's Actually Happening

### Sequence Lengths: The Core Problem

| Parameter | Value |
|-----------|-------|
| Encoder max_length (SQuAD context) | **512 tokens** (default, line 177 cepe.py) |
| Chunk size C | **64** (hard-coded, line 1074 cepe.py) |
| Number of chunks | **8** |
| Sequential inter-chunk steps | **8** |

The paper's speedup benchmarks are for **512k–1M token sequences** (8,000–15,000 chunks). At T=512, there are only **8 chunks** — the regime is completely wrong for linear attention to win.

### FLOP Count at T=512 Still Favors KDA

Even at T=512, d=64, C=64, the FLOP count favors KDA:
- **KDA**: 6·512·64² + 3·512·64·64 + 512·64² ≈ **21M FLOPs**
- **Softmax**: 2·512²·64 ≈ **33.6M FLOPs**

So KDA has ~35% fewer FLOPs, yet it's *slower*. This means **kernel launch overhead and synchronization dominate** over raw arithmetic at this sequence length — not a FLOP problem.

---

## What Parallelizations Are and Are Not Active

### ✅ Active (FLA IS being used)
- **Intra-token parallel kernel** — `chunk_kda_fwd_kernel_intra_token_parallel` fires: one thread block per token (512 thread blocks for T=512)
- **TensorCore path** — `safe_gate=True` enables the M=16 matmul path
- **Fused gate+cumsum** — `kda_gate_chunk_cumsum_vector_kernel` fuses gate activation and prefix-sum
- **Fused wy computation** — `recompute_w_u_fwd_kda_kernel` in `wy_fast.py` avoids separate W/U kernel
- **`disable_recompute=True`** — all intermediates (Aqk, Akk, w, u, qg, kg, v_new, h) saved to HBM for faster backward (no recomputation)

The PYTHONPATH in `cepe_exps.sh` includes `flash-linear-attention/`, so `_FLA_AVAILABLE = True` and `_chunk_kda` = the Triton kernel path, not the fallback Python loop.

### ❌ Not Realized — Why

**1. Long-sequence regime speedup is simply N/A at T=512**
The theoretical crossover where linear attention beats softmax in wall-clock time (not just FLOPs) is roughly T >> d²/C = 64²/64 = 64 tokens in FLOPs alone. But with real kernel constant factors (startup, sync, memory bandwidth), the actual crossover for this implementation vs FlashAttention2 is estimated around T=4k–8k tokens. At T=512 we are well below that.

**2. Multiple kernel launches vs. single fused kernel**
KDA forward requires **~6–8 distinct Triton kernel launches**:
```
kda_gate_chunk_cumsum_vector_kernel         [gate.py]
chunk_kda_fwd_kernel_intra_token_parallel   [chunk_intra.py]
chunk_kda_fwd_kernel_inter_solve_fused      [chunk_intra.py]
chunk_gated_delta_rule_fwd_h_kernel         [delta_h]
chunk_gla_fwd_o_gk_kernel                   [gla]
recompute_w_u_fwd_kda_kernel                [wy_fast.py]
```
FlashAttention2 softmax cross-attention: **1 kernel launch**.
At T=512, the kernel launch overhead (~5–10µs each) and inter-kernel HBM round-trips dominate the total time.

**3. disable_recompute=True is expensive at small T**
Saving all intermediates (Aqk, Akk, w, u, qg, kg, v_new, h) is a good trade when T is large (recompute cost is high). At T=512, recomputation is cheap but saving all tensors to HBM and reading them back in the backward pass is a constant per-tensor cost. This flag was designed for long-sequence training; at T=512, it may actually *hurt* by adding unnecessary memory bandwidth.

**4. Inter-chunk recurrence is inherently sequential**
The inter-chunk state propagation S[i] → S[i+1] cannot be parallelized — each chunk's state depends on the previous. At 8 chunks this is a minor cost, but the sequential dependency means the GPU must finish chunk i before starting the state update for chunk i+1, adding pipeline stalls.

**5. Cross-attention vs. self-attention mismatch**
The paper benchmarks KDA as **self-attention** in a language model decoder (long sequences). Here KDA is used as **cross-attention**: run `chunk_kda` on T_enc encoder tokens to produce a compressed state h=[B, H, 64, 64], then query with Q_dec. Softmax cross-attention (FlashAttention2) is heavily optimized for exactly this pattern and benefits from the highly tuned SDPA path in PyTorch/CUDA. KDA must run the full chunkwise machinery even though only the final state h is needed.

---

## Root Cause Summary

| Factor | Contribution to slowdown |
|--------|-------------------------|
| T=512 << crossover point for real wall-clock speedup | **Primary cause** |
| ~6–8 Triton kernel launches vs. 1 FlashAttention2 kernel | **Primary cause** |
| `disable_recompute=True` memory bandwidth at small T | **Secondary cause** |
| Cross-attention pattern vs. self-attention optimization | **Secondary cause** |
| Inter-chunk sequential recurrence (8 steps) | **Minor** |

**The hardware-efficient parallelizations in the FLA library ARE implemented and ARE active. The issue is not missing parallelizations but wrong sequence-length regime.** The Triton kernel overhead, synchronization, and HBM round-trips between multiple kernel launches are larger than the wall-clock savings from fewer FLOPs at T=512.

---

## When Would KDA Win?

At the current chunk size C=64 and head_dim d=64, KDA should become faster than softmax in wall-clock time at approximately **T ≥ 4k–8k encoder tokens** (accounting for kernel overhead). The SQuAD context is ~512 tokens — one order of magnitude too short.

For the KDA advantage to be realized in this cross-attention setting, the encoder would need to process **much longer documents** (e.g., full book chapters, long-form RAG contexts), which is exactly the use case the paper targets in its long-context experiments.

---

## Relevant Files

| File | Purpose |
|------|---------|
| `apps/minimal_squad/cepe.py:1031–1068` | `_chunk_kda` call with `disable_recompute=True`, `safe_gate=True` |
| `apps/minimal_squad/cepe.py:177, 1074` | `max_length=512`, `CHUNK=64` |
| `apps/minimal_squad/flash-linear-attention/fla/ops/kda/chunk_intra.py` | `chunk_kda_fwd_kernel_intra_token_parallel`, `inter_solve_fused` |
| `apps/minimal_squad/flash-linear-attention/fla/ops/kda/chunk_bwd.py` | `chunk_kda_bwd_kernel_dAv`, `chunk_kda_bwd_kernel_wy_dqkg_fused` |
| `apps/minimal_squad/flash-linear-attention/fla/ops/kda/wy_fast.py` | `recompute_w_u_fwd_kda_kernel` |
| `apps/minimal_squad/flash-linear-attention/fla/ops/kda/gate.py` | `kda_gate_chunk_cumsum_vector_kernel` |
| `apps/minimal_squad/papers/kimi_linear.pdf` | Section 3: chunkwise + intra-token parallel theory |
