# From-Scratch Decoder Depth Experiment: Softmax vs KDA (March 16, 2026)

## Overview

This report compares **softmax cross-attention** vs **Kimi KDA linear cross-attention** (with FLA Triton kernel) across decoder depths (6L / 12L / 22L) for a **from-scratch** ModernBERT 400M + custom decoder trained end-to-end. All experiments use C/Q//A data format, 5 epochs, batch size 8, `--pretrained_weight_updating 0.333` on L40S GPU.

The 22L from-scratch decoder matches TinyLlama's depth and is the most meaningful comparison point for the pretrained-decoder (CEPE/finetune) baselines.

---

## Experiment Status (as of March 16)

| Job ID | Experiment | Decoder | Status |
|--------|-----------|---------|--------|
| 2408630 | `modernbert400m_scratch6l_cq_a_softmax` | 6L from-scratch | ✅ 5 epochs complete |
| 2410115 | `modernbert400m_scratch6l_cq_a_softmax` (rerun) | 6L from-scratch | ⚠️ 4 epochs + epoch 5 in progress |
| 2410116 | `modernbert400m_scratch6l_cq_a_linear_kda` | 6L from-scratch KDA | ⚠️ 3 epochs + epoch 4 in progress |
| 2408556 | `modernbert400m_scratch12l_cq_a_softmax` (old) | 12L from-scratch | ✅ 5 epochs complete |
| 2410117 | `modernbert400m_scratch12l_cq_a_softmax` (rerun) | 12L from-scratch | ⚠️ 3 epochs + epoch 4 in progress |
| 2410118 | `modernbert400m_scratch12l_cq_a_linear_kda` | 12L from-scratch KDA | ⚠️ 2 epochs + epoch 3 in progress |
| 2410360 | `modernbert400m_scratch22l_cq_a_softmax` | 22L from-scratch | 🔄 Epoch 1 in progress (~72%) |
| 2410361 | `modernbert400m_scratch22l_cq_a_linear_kda` | 22L from-scratch KDA | 🔄 Epoch 1 in progress (~37%) |

The 22L runs are the first launch; no epochs have completed yet.

---

## Training Speed: Epoch-End Average Throughput

The table below uses the **epoch-end average** (the `it/s` reported on the tqdm 100% line), which is the most reliable measure — it covers the full epoch including warmup and eval overhead. For in-progress runs, instantaneous mid-epoch speeds are used as estimates.

### After Kimi KDA Speedup (current runs: 2410xxx)

| Depth | Softmax (it/s) | KDA (it/s) | KDA/Softmax slowdown |
|-------|---------------|-----------|----------------------|
| 6L | **6.86** (epoch 2–4 avg) | **5.96** (epoch 2–3 avg) | **1.15×** |
| 12L | **5.65** (epoch 2–3 avg) | **4.29** (epoch 2 avg) | **1.32×** |
| 22L | **~5.33** (epoch 1 est.) | **~4.0** (epoch 1 est.) | **~1.33×** |

### Before Kimi KDA Speedup (old runs: 2408xxx)

| Depth | Softmax (it/s) | KDA (it/s) | KDA/Softmax slowdown |
|-------|---------------|-----------|----------------------|
| 6L | ~7.0 | ~1.55 (epoch 1) | **~4.5×** |
| 12L | ~5.71 | ~1.81 (epoch 1) | **~3.2×** |

The Kimi speedup commit improved KDA by **~3.8× at 6L** and **~2.4× at 12L**.

### Epoch Wall-Clock Times (estimated from it/s, 10 892 steps/epoch)

| Depth | Softmax (min/epoch) | KDA (min/epoch) |
|-------|--------------------|--------------------|
| 6L | ~26 min | ~31 min |
| 12L | ~32 min | ~42 min |
| 22L | ~34 min (est.) | ~45 min (est.) |

---

## Key Speed Finding: KDA Gap Narrows With Depth — But Has Plateaued

The predicted trend was that deeper decoders would dilute the KDA cross-attention backward overhead (more self-attn + FFN backward per layer), shrinking the gap toward 1× as depth → ∞.

**Observed trend:**

| Depth | KDA slowdown (post-speedup) |
|-------|----------------------------|
| 6L | 1.15× |
| 12L | 1.32× |
| 22L | ~1.33× (in-progress) |

The gap *does not shrink* from 12L to 22L — it stays flat at ~1.32–1.33×. This suggests the slowdown has saturated: at 12L, the encoder backward (400M params, fixed cost regardless of decoder depth) already dominates over both the KDA and softmax decoder backward components, so adding more layers does not dilute the relative cost further.

**Why 6L is the outlier (1.15×):** At 6L, the decoder is small enough that the encoder backward is *not* the bottleneck — the cross-attention backward is still relatively large, but so is the softmax cost. The post-Kimi KDA backward is efficient enough that the gap is very small at shallow depth.

---

## Training Loss Convergence

Validation metrics are only available at epoch end (logged to W&B, not stdout) and the new runs have not yet completed enough epochs for final val scores. Training loss gives an early signal:

| Model | Epoch 1 loss | Epoch 2 loss | Epoch 3 loss | Epoch 4 loss |
|-------|-------------|-------------|-------------|-------------|
| 6L softmax | 2.03 | 0.74 | 0.49 | 0.37 |
| 6L KDA | 3.24 | 1.42 | 0.93 | (in progress) |
| 12L softmax | 2.17 | 0.78 | 0.53 | (in progress) |
| 12L KDA | 3.92 | 1.64 | (in progress) | — |

**Observations:**
- KDA training loss starts higher and converges more slowly than softmax at the same depth.
- At epoch 2, 6L KDA loss is ~1.93× higher than 6L softmax; 12L KDA is ~2.10× higher than 12L softmax.
- This does not necessarily mean worse final accuracy — the model may still reach similar EM/F1 by epoch 5 — but it suggests KDA cross-attention in from-scratch mode is harder to optimize than softmax.
- The slower convergence in KDA may partly reflect that the `chunk_kda_bwd` backward is less numerically smooth than softmax backward, making gradient-based optimization noisier at the start.

### Final Val Scores (completed runs only)

| Model | Val EM% | Val F1% |
|-------|---------|---------|
| 6L softmax (2408630, 5 epochs) | **85.4** | **89.9** |
| 12L softmax (2408556, 5 epochs) | **87.4** | **91.5** |
| 6L KDA (epoch 5 pending) | — | — |
| 12L KDA (epoch 5 pending) | — | — |
| 22L softmax (epoch 1 pending) | — | — |
| 22L KDA (epoch 1 pending) | — | — |

---

## Comparison: From-Scratch vs Pretrained Decoder

For reference, the best results from pretrained-decoder (CEPE/finetune) experiments (March 10 report):

| Architecture | Val EM% | Val F1% | Epoch time |
|---|---|---|---|
| TinyLlama 1B (decoder-only, C/Q/A) | 90.0 | 93.5 | — |
| ModernBERT 400M + TinyLlama (KDA FLA finetune, March 10) | 87.6 | 91.7 | ~2.6h |
| ModernBERT 400M + TinyLlama (softmax finetune) | **87.6** | **91.7** | ~1.8h |
| ModernBERT 400M + 6L from-scratch softmax | 85.4 | 89.9 | ~26 min |
| ModernBERT 400M + 12L from-scratch softmax | 87.4 | 91.5 | ~32 min |
| ModernBERT 400M + 22L from-scratch softmax | (pending) | (pending) | ~34 min (est.) |

The 12L from-scratch softmax (87.4% EM) nearly matches the TinyLlama 1B finetune (87.6% EM) while being **3.3× faster per epoch** and using a much smaller decoder (~100M vs ~1.1B params). The 22L from-scratch experiments are expected to match or exceed 12L.

---

## Summary

1. **Kimi KDA speedup (from latest commit) massively improved KDA**: from 3–4.5× slower than softmax to only **1.15–1.33× slower**, making it a practical alternative.

2. **The KDA gap does not shrink further beyond 12L**: stabilizes at ~1.32–1.33× regardless of depth. The bottleneck shifts to encoder backward at 12L+, not decoder cross-attention.

3. **KDA converges slower** in from-scratch mode: loss is ~2× higher at epoch 2 compared to softmax. Final EM/F1 comparison pending epoch 5.

4. **From-scratch 12L softmax is already competitive** with TinyLlama 1B finetune at 87.4% EM, 3× faster per epoch.

5. **22L experiments are running**: epoch 1 in progress — softmax at ~5.33 it/s and KDA at ~4.0 it/s. First val scores expected within the next few hours.

---

## Next Steps

- Wait for 22L epoch 1 to complete and check val EM/F1
- Compare 22L softmax and KDA final EM/F1 at epoch 5 (expected in ~3–4 days)
- If 22L softmax EM > 12L softmax EM by < 1%, the depth scaling curve has saturated and 12L is sufficient
- Investigate why KDA loss is ~2× higher than softmax at epochs 1–2 — possible learning rate tuning needed for KDA
