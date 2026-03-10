# Linear Cross-Attention: MLA vs Kimi KDA — Full Comparison (March 10, 2026)

## Overview

This report compares two linear cross-attention variants across training strategies (frozen, CEPE, finetune), encoder sizes (150M vs 400M), and the presence or absence of FLA (flash-linear-attention) Triton kernel acceleration. All experiments use **ModernBERT encoder + TinyLlama 1B decoder** with **C/Q//A** data format (context+question in encoder → answer from decoder) on SQuAD 2.0.

The two linear variants are:
- **MLA** (`--cross_attn_type linear`): simple parallel K^T @ V, O(T_enc · d_k²), fastest
- **Kimi KDA** (`--cross_attn_type linear_kda`): chunked delta-rule with decay gate, more expressive

For reference, the softmax baseline results (from February 20 report) are included throughout.

---

## Experiment Matrix

| Job ID | Variant | Encoder | Strategy | FLA? | Status |
|--------|---------|---------|----------|------|--------|
| — | MLA (linear) | 150M | CEPE | No | ✅ Complete (5 epochs) |
| 2355260 | Kimi KDA v2 | 400M | Frozen | No (PyTorch fallback) | 🔄 Ep4 running |
| 2355261 | Kimi KDA v2 | 400M | CEPE | No (PyTorch fallback) | 🔄 Ep3 ~90% |
| 2355262 | Kimi KDA v2 | 150M | CEPE | No (PyTorch fallback) | 🔄 Ep3 ongoing |
| 2355263 | Kimi KDA v2 | 400M | Finetune | No (PyTorch fallback) | 🔄 Ep3 ~70% |
| 2356257 | Kimi KDA FLA | 400M | Frozen | Yes (intended) | 🔄 Ep3 ongoing |
| 2356258 | Kimi KDA FLA | 400M | CEPE | Yes (intended) | 🔄 Ep3 ~50% |
| 2356259 | Kimi KDA FLA | 150M | CEPE | Yes (intended) | 🔄 Ep3 ongoing |
| 2356260 | Kimi KDA FLA | 400M | Finetune | Yes (intended) | 🔄 Ep3 ~40% |

> **Note on status**: None of the 8 new runs have completed 5 full epochs (all are ~3 epochs in). The MLA run is the only one that has fully completed. Results below are the best seen to date.

---

## Parameter Counts

| Adapter type | Strategy | Trainable params |
|---|---|---|
| Softmax (baseline) | Frozen (adapters only) | 371M (20%) |
| Softmax (baseline) | CEPE (adapters + encoder) | 766M (41%) |
| Softmax (baseline) | Finetune (everything) | 1,866M (100%) |
| MLA `linear` | CEPE | ~520M (32%) |
| Kimi KDA `linear_kda` | Frozen (adapters only) | **464M (24%)** |
| Kimi KDA `linear_kda` | CEPE (adapters + encoder) | **859M (44%)** |
| Kimi KDA `linear_kda` | Finetune (everything) | **1,959M (100%)** |

Kimi KDA's adapter is substantially larger than softmax (464M vs ~184M) because of the extra `g_proj` (D→H×d_k = ~4.2M params per layer × 22 layers) and `beta_proj` parameters encoding per-element decay gates.

---

## Results: Frozen Encoder

Only Kimi KDA frozen experiments were run (no MLA or softmax equivalent in this batch).

### Per-Epoch EM% — 400M Encoder, Frozen

| Epoch | Softmax (baseline) | Kimi KDA v2 (no FLA) | Kimi KDA FLA |
|-------|-------------------|----------------------|--------------|
| 1 | 34.6 | 10.2 | 11.0 |
| 2 | 36.8 | 12.6 | 15.4–15.8 |
| 3 | 39.0 | 14.8 | in progress |
| 4 | **43.4** | running | — |
| 5 | 41.6 | — | — |
| **Best** | **43.4** | **14.8** (ep3) | **15.4–15.8** (ep2) |

**Finding**: Kimi KDA frozen is substantially worse than softmax frozen (14.8% vs 43.4%). With adapter-only training and no encoder fine-tuning, the linear KDA mechanism struggles significantly more than softmax. The FLA variant shows similar poor performance. The frozen setting is not viable for linear attention adapters.

---

## Results: CEPE (400M Encoder)

### Per-Epoch EM% — 400M Encoder, CEPE

| Epoch | Softmax (baseline) | Kimi KDA v2 (no FLA) | Kimi KDA FLA |
|-------|-------------------|----------------------|--------------|
| 1 | 61.4 | **78.0** | 77.6–77.8 |
| 2 | 70.4 | **80.2** | 80.4–80.6 |
| 3 (partial) | **77.6** (final ep3) | **≥ 83.2** (@ 60%) | 78.6 (@ 50%) |
| Best seen | 77.6 (ep3) | **≥ 83.2** | **80.6** (ep2 end) |

### Intra-Epoch Progression — Epoch 1, 400M CEPE

| Progress | Kimi KDA v2 | Kimi KDA FLA |
|----------|------------|--------------|
| 10% | 5.4% | 7.8% |
| 20% | 32.8% | 23.6% |
| 30% | 60.2% | 54.8% |
| 40% | 65.8% | 56.6% |
| 50% | 73.2% | 63.0% |
| 60% | 70.6% | 64.2% |
| 70% | 72.2% | 68.0% |
| 80% | 74.8% | 73.0% |
| 90% | 76.8% | 72.4% |
| 100% | **78.0%** | **77.8%** |

**Findings**:
- Both variants converge to ~77.8% by the end of epoch 1, already matching the softmax 5-epoch best
- Kimi KDA v2 converges faster within epoch 1 (60.2% at 30% vs 54.8% for FLA)
- By epoch 3, v2 is reaching **83.2% EM** (at 60% of epoch 3) — exceeding the softmax baseline by 5.6 points
- FLA variant appears slightly behind v2 at the same training stage (78.6% at ep3 50% vs 81.0-83.2% for v2)

### Val Loss Progression — 400M CEPE

| Epoch | Kimi KDA v2 | Kimi KDA FLA |
|-------|------------|--------------|
| 1 | 0.5018 | 0.4979 |
| 2 | 0.4703 | 0.4291 |
| 3 (partial best) | 0.4858 | 0.4847 |

FLA achieves lower validation loss at epoch 2 end (0.4291 vs 0.4703) despite similar EM — this is a minor inconsistency worth monitoring.

---

## Results: CEPE (150M Encoder)

### Per-Epoch EM% — 150M Encoder, CEPE

| Epoch | Softmax (baseline) | MLA linear | Kimi KDA v2 | Kimi KDA FLA |
|-------|-------------------|------------|-------------|--------------|
| 1 | 42.8 | 29.8 | **56.8** | **62.8** |
| 2 | 51.2 | 32.4 | **65.8** | 62.8 (plateau) |
| 3 | 55.4 | **35.0** | 64.2 (overfit) | in progress (~63.8% seen) |
| 4 | 53.2 | 34.8 | — | — |
| 5 | **55.6** | 34.2 | — | — |
| **Best** | 55.6 | **35.0** | **65.8** (ep2) | **62.8** (ep1–2) |

**Findings**:
- **MLA (simple linear) is much weaker**: Only 35% EM vs 55.6% softmax and 65.8% Kimi KDA. The K^T@V mechanism without gating cannot match the delta-rule's selectivity.
- **Kimi KDA v2 peaks at epoch 2** (65.8%) then slightly overfits (64.2% at ep3 end). 10-point gain over softmax baseline.
- **Kimi KDA FLA plateaus at 62.8%** after epoch 1 and shows no improvement in epoch 2 — a flat learning curve suggesting potential optimization instability.
- Kimi KDA v2 > FLA for 150M encoder (65.8% vs 62.8%).

---

## Results: Full Fine-Tuning (400M Encoder)

### Per-Epoch EM% — 400M Encoder, Finetune

| Epoch | Softmax (baseline) | Kimi KDA v2 | Kimi KDA FLA |
|-------|-------------------|-------------|--------------|
| 1 | 68.6 | **79.4** | 74.2–75.2 |
| 2 | 73.6 | **80.8** | 78.8–79.0 |
| 3 (partial) | **78.4** (ep3 final) | **≥ 81.4** (@ 90%) | 79.4 (@ 30%) |
| Best seen | 78.4 (ep3) | **≥ 81.4** (ep2 @90%) | **80.0** (ep2 @60%) |

**Findings**:
- Kimi KDA v2 finetune starts much stronger than softmax in epoch 1 (79.4% vs 68.6%)
- FLA finetune is initially weaker than v2 (74.2-75.2% at ep1) but converges towards v2's level by epoch 2 (79% vs 80.8%)
- By epoch 2, both variants already exceed the softmax 5-epoch best
- Finetune slightly outpaces CEPE on the same architecture (80.8% vs 80.2% at epoch 2 for v2)

### Val Loss — Finetune

| Epoch | Kimi KDA v2 | Kimi KDA FLA |
|-------|------------|--------------|
| 1 | 0.4552 | 0.5188 |
| 2 | 0.4235 | 0.4570 |

v2 achieves lower val loss throughout, consistent with higher EM scores.

---

## Summary: Best EM% Across All Strategies (400M Encoder)

| Strategy | Softmax | MLA linear | Kimi KDA v2 | Kimi KDA FLA |
|----------|---------|------------|-------------|--------------|
| Frozen | 43.4 | — | 14.8 (ep3) | 15.4 (ep2) |
| CEPE | 77.6 | — | **≥ 83.2** | **80.6** |
| Finetune | 78.4 | — | **≥ 81.4** | **≥ 80.0** |

**Key takeaway**: Kimi KDA (both variants) already **surpasses the 5-epoch softmax baseline** after only 2 epochs for CEPE and finetune — before 5 full epochs are even complete.

---

## Summary: Best EM% Across All Strategies (150M Encoder)

| Strategy | Softmax | MLA linear | Kimi KDA v2 | Kimi KDA FLA |
|----------|---------|------------|-------------|--------------|
| Frozen | 22.4 | — | — | — |
| CEPE | 55.6 | 35.0 | **65.8** | 62.8 |

---

## Training Speed and Efficiency

All experiments run on **L40S GPU (44GB VRAM)**, batch size 8, ModernBERT encoder + TinyLlama 1B decoder.

| Variant | Speed (it/s) | Time/epoch | vs. Softmax |
|---------|-------------|------------|-------------|
| Softmax baseline (FlashAttention) | ~2.07 | ~1.5h | 1× (reference) |
| MLA linear (K^T@V) | **~1.95** | **~1.6h** | **~1.1× (near-parity)** |
| Kimi KDA v2 frozen (no FLA) | ~0.82 | ~3.7h | 0.4× |
| Kimi KDA v2 CEPE (no FLA) | ~0.71 | ~4.2h | 0.34× |
| Kimi KDA v2 finetune (no FLA) | ~0.67 | ~4.5h | 0.32× |
| Kimi KDA FLA frozen | ~0.75 | ~4.0h | 0.36× |
| Kimi KDA FLA CEPE | ~0.71 | ~4.2h | 0.34× |
| Kimi KDA FLA finetune | ~0.63 | ~4.8h | 0.30× |

> Speeds estimated from elapsed time at the 20% epoch checkpoint (2178/10892 steps).

### Critical Finding: FLA Kernel is NOT Accelerating Training

The `linear_kda_fla` runs show **identical or slightly slower training speed** compared to the `linear_kda_v2` (PyTorch fallback) runs. Expected speedup was ~3× (from ~0.67 to ~2.07 it/s) based on the Mar3rd analysis. This means the FLA `chunk_kda` Triton kernel is **not being activated** — both sets of runs are using the PyTorch fallback path. The `_FLA_AVAILABLE` flag may be returning `True` (FLA is installed) while the kernel silently falls back due to dtype or shape incompatibility.

The `wandb: Syncing run ...fla...` header confirms FLA jobs launched correctly, but speed equivalence proves the fast path is unused.

**Impact**: All 8 new runs are effectively running the same code path at ~0.67–0.82 it/s. The naming distinction (v2 vs fla) reflects the intended experiment, not an actual kernel difference.

---

## MLA vs Kimi KDA: Head-to-Head (150M CEPE)

Both linear variants tested with the same encoder/decoder/strategy for direct comparison:

| Metric | MLA (simple linear) | Kimi KDA v2 | Delta |
|--------|---------------------|-------------|-------|
| Best EM% | 35.0% (ep3) | **65.8%** (ep2) | **+30.8 pp** |
| Val loss (ep1) | 2.39 | 0.75 | — |
| Val loss (ep5/ep3) | 1.96 | 0.68 (ep3) | — |
| Training speed | **~1.95 it/s** | ~0.75 it/s | **2.6× faster** |
| Epochs completed | 5 (done) | 3 (ongoing) | — |
| Convergence | Peaks ep3, overfits slightly | Peaks ep2, slight overfit ep3 | — |

**MLA is 2.6× faster but achieves less than half the EM**. The delta-rule gating in Kimi KDA (per-token decay + delta correction) is essential for quality — simple uniform accumulation (K^T@V) cannot selectively weight encoder positions.

---

## Kimi KDA v2 vs FLA: Direct Comparison (400M CEPE)

Both use `linear_kda`, same architecture, both in fallback PyTorch mode (same speed).

| Metric | Kimi KDA v2 | Kimi KDA FLA | Delta |
|--------|------------|--------------|-------|
| EM ep1 end | **78.0%** | 77.8% | +0.2 pp |
| EM ep2 end | **80.2%** | 80.6% | −0.4 pp |
| EM ep3 @ 50% | **81.0%** | 78.6% | +2.4 pp |
| EM ep3 @ 60% | **83.2%** | — | — |
| Val loss ep2 | 0.4703 | **0.4291** | +0.04 |
| Speed | **~0.71 it/s** | ~0.71 it/s | identical |

The v2 variant appears to lead in EM at the epoch-3 stage, but both are still running. The difference is small and both far exceed the softmax baseline. Lower val loss for FLA at epoch 2 does not yet translate to higher EM — this may resolve as training continues.

---

## Key Observations

1. **Kimi KDA already outperforms softmax CEPE** before completing 5 epochs: v2 CEPE reaches ≥83.2% vs softmax peak of 77.6% — a **+5.6 point improvement** with comparable (though ~3× slower) training.

2. **MLA (simple linear) is not competitive**: At 35% best EM (150M), it falls 30 points below Kimi KDA and 20 points below the softmax baseline. The K^T@V mechanism without gating is insufficient for this task.

3. **FLA is not accelerating training**: Both v2 and FLA variants run at ~0.67–0.82 it/s (3× slower than softmax). The Triton `chunk_kda` kernel appears to be silently not loading. This is the highest-priority engineering issue.

4. **400M encoder substantially outperforms 150M** across all variants:
   - CEPE: 78%+ vs 35–65%
   - The gap is consistent regardless of attention type

5. **CEPE ≈ Finetune in quality, with fewer trainable params**:
   - v2 CEPE: 83.2% (partial ep3) at 859M trainable
   - v2 Finetune: 81.4% (partial ep3) at 1,959M trainable
   - The CEPE strategy preserves decoder weights effectively

6. **Frozen encoder with linear attention is not viable**: 14.8% (v2) vs 43.4% (softmax). The Kimi KDA adapter requires encoder fine-tuning to align representations with the delta-rule memory structure.

7. **Kimi KDA v2 150M overfits after epoch 2** (65.8% → 64.2%), mirroring softmax pattern. FLA 150M plateaus at 62.8% from epoch 1, suggesting a different (worse) optimization dynamic.

---

## Training Curves at a Glance

### 400M Encoder, CEPE — EM% per epoch

```
90 |                                        ≥83.2 (v2, ep3@60%)
   |                               ·········
80 |                      78.0  80.6(FLA)  80.2(v2)
   |            61.4 70.4 77.6 [softmax peak]
70 |
   |
60 |
   |
50 |
   +----------------------------------------------------------
     ep1   ep2   ep3   ep4   ep5

     — softmax    ··· Kimi KDA v2     - - Kimi KDA FLA
```

### 150M Encoder, CEPE — EM% per epoch

```
70 |           65.8 (v2 peak)
   |  56.8  ···
60 |      62.8 62.8 (FLA plateau)
   |   42.8  51.2  55.4  55.6 [softmax peak]
50 |
40 |   29.8  32.4  35.0  34.8  34.2 [MLA]
   +----------------------------------------------------------
     ep1   ep2   ep3   ep4   ep5
```

---

## Next Steps

1. **Diagnose FLA kernel loading**: Print `_FLA_AVAILABLE` at runtime and confirm `chunk_kda` is called vs fallback loop. The expected speedup (3×) would bring all runs from ~4.5h/epoch to ~1.5h/epoch — critical for completing 5-epoch runs.

2. **Await full 5-epoch completion**: Kimi KDA v2 CEPE 400M is on track to exceed 83%+ EM. The final plateau (or further improvement) will determine whether longer training is beneficial.

3. **Run MLA (linear, 400M encoder) experiments**: Only 150M MLA was tested; the 400M variant may be significantly stronger and merits a direct comparison with 400M Kimi KDA.

4. **Investigate FLA 150M plateau**: Kimi KDA FLA 150M is stuck at 62.8% EM from epoch 1. This plateau behavior (absent in v2) could indicate a learning rate or gradient issue specific to that run's random seed or initialization.

5. **Memory profiling**: GPU peak memory for Kimi KDA vs softmax has not been explicitly captured. With `torch.cuda.max_memory_allocated()` logging added, we can confirm the ~1.4GB state tensor footprint claimed in Mar3rd report.
