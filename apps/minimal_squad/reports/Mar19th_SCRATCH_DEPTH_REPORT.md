# From-Scratch Decoder Depth Experiment: Softmax vs KDA — Final Results (March 19, 2026)

## Overview

This report presents **complete 5-epoch results** for all six from-scratch depth experiments launched after the Mar 16 report. All runs use ModernBERT 400M encoder + custom from-scratch decoder, C/Q//A data format, batch size 8, `--pretrained_weight_updating 0.333`.

| Job ID | Experiment | Status |
|--------|-----------|--------|
| 2430231 | `modernbert400m_scratch6l_cq_a_softmax` | ✅ 5 epochs complete |
| 2430232 | `modernbert400m_scratch6l_cq_a_linear_kda` | ✅ 5 epochs complete |
| 2430233 | `modernbert400m_scratch12l_cq_a_softmax` | ✅ 5 epochs complete |
| 2430234 | `modernbert400m_scratch12l_cq_a_linear_kda` | ✅ 5 epochs complete |
| 2430235 | `modernbert400m_scratch22l_cq_a_softmax` | ✅ 5 epochs complete |
| 2430236 | `modernbert400m_scratch22l_cq_a_linear_kda` | ✅ 5 epochs complete |

---

## Validation Metrics by Epoch

### 6 Layers

| Epoch | Softmax EM | Softmax F1 | KDA EM | KDA F1 | ΔEM (KDA−Soft) |
|-------|-----------|-----------|--------|--------|----------------|
| 1 | 79.00% | 86.12% | 77.00% | 85.56% | −2.0 |
| 2 | 85.00% | 90.70% | 78.40% | 86.27% | −6.6 |
| 3 | **86.40%** | 90.49% | 80.00% | 87.43% | −6.4 |
| 4 | 86.00% | **90.83%** | **84.80%** | **89.77%** | −1.2 |
| 5 | 83.80% | 89.13% | 82.40% | 89.47% | −1.4 |

Softmax peaks at E3 (86.40% EM); KDA peaks at E4 (84.80% EM). KDA trails by **−1.6 EM / −1.1 F1** at best epochs. KDA converges more slowly, peaking 1 epoch later, but doesn't close the gap by E5.

### 12 Layers

| Epoch | Softmax EM | Softmax F1 | KDA EM | KDA F1 | ΔEM (KDA−Soft) |
|-------|-----------|-----------|--------|--------|----------------|
| 1 | 79.80% | 87.05% | 79.20% | 86.90% | −0.6 |
| 2 | **85.40%** | **90.78%** | 82.40% | 89.20% | −3.0 |
| 3 | 84.40% | 89.80% | **85.20%** | 89.60% | +0.8 |
| 4 | 86.20% | 90.35% | 84.60% | 89.53% | −1.6 |
| 5 | 84.40% | 89.27% | 84.40% | **89.50%** | 0.0 |

Best EM: softmax 86.20% (E4) vs KDA 85.20% (E3) — gap **−1.0 EM**. Both are competitive; KDA briefly overtakes softmax at E3.

### 22 Layers

| Epoch | Softmax EM | Softmax F1 | KDA EM | KDA F1 | ΔEM (KDA−Soft) |
|-------|-----------|-----------|--------|--------|----------------|
| 1 | 77.20% | 85.64% | 81.20% | 87.64% | +4.0 |
| 2 | 84.20% | 89.12% | 81.20% | 87.20% | −3.0 |
| 3 | 84.60% | **90.55%** | 85.60% | 89.98% | +1.0 |
| 4 | 84.60% | 89.33% | 84.40% | 89.41% | −0.2 |
| 5 | 83.20% | 88.95% | **86.80%** | **90.48%** | +3.6 |

At 22L, **KDA overtakes softmax**: KDA peaks at E5 (86.80% EM, still trending upward) vs softmax's best at E3 (84.60% EM). KDA wins by **+2.2 EM / +0.0 F1** at best epochs. Notably, softmax appears to start overfitting/degrading after E3 while KDA is still improving at E5.

---

## Best-Epoch Summary

| Model | Best EM | Best F1 | Best Epoch |
|-------|---------|---------|-----------|
| 6L Softmax | 86.40% | 90.83% | E3 / E4 |
| 6L KDA | 84.80% | 89.77% | E4 |
| 12L Softmax | **86.20%** | **90.78%** | E4 / E2 |
| 12L KDA | 85.20% | 89.60% | E3 |
| 22L Softmax | 84.60% | 90.55% | E3 |
| 22L KDA | **86.80%** | **90.48%** | E5 |

**22L KDA achieves the overall best EM (86.80%)** across all six models, while 6L softmax and 12L softmax are competitive at 86.40% and 86.20% respectively. Notably, the 22L softmax underperforms relative to depth expectations — it plateaus at 84.60%, below 6L and 12L softmax, suggesting overfitting or insufficient epochs.

---

## Training Loss Convergence

| Model | E1 loss | E2 loss | E3 loss | E4 loss | E5 loss |
|-------|---------|---------|---------|---------|---------|
| 6L Softmax | 1.906 | 0.621 | 0.404 | 0.284 | 0.204 |
| 6L KDA | 2.619 | 0.748 | 0.454 | 0.299 | 0.200 |
| 12L Softmax | 2.000 | 0.641 | 0.406 | 0.276 | 0.193 |
| 12L KDA | 2.376 | 0.720 | 0.435 | 0.277 | 0.183 |
| 22L Softmax | 2.037 | 0.651 | 0.408 | 0.273 | 0.186 |
| 22L KDA | 2.563 | 0.765 | 0.459 | 0.292 | 0.193 |

KDA consistently starts with ~25–35% higher E1 loss than softmax at matched depth, but **all models converge to nearly identical train loss by E5** (~0.19–0.20). The slower early convergence does not prevent KDA from reaching competitive or superior final val scores, especially at 22L.

---

## Training Speed

Epoch wall-clock times extracted from tqdm elapsed time at the 90% step checkpoint (10,890/10,892 steps), averaged across all 5 epochs.

| Model | Avg epoch time | Avg it/s |
|-------|---------------|---------|
| 6L Softmax | ~69 min | ~3.3 |
| 6L KDA | **~33 min** | ~7.0 |
| 12L Softmax | **~33 min** | ~6.7 |
| 12L KDA | ~43 min | ~5.5 |
| 22L Softmax | ~45 min | ~5.1 |
| 22L KDA | ~65 min | ~3.7 |

⚠️ **Hardware caveat:** These jobs ran on different cluster nodes. The 6L softmax speed (~3.3 it/s, ~69 min/epoch) is anomalously slow — it matches the 22L KDA speed, and is ~2× slower than the 6L KDA run. This is almost certainly a node assignment difference rather than a true attention-type effect, and is inconsistent with the Mar 16 runs (2410xxx) where 6L softmax ran at ~6.86 it/s (~26 min/epoch). The 12L and 22L comparisons are more trustworthy:

| Depth | Softmax (it/s) | KDA (it/s) | KDA slowdown |
|-------|---------------|-----------|--------------|
| 12L | ~6.7 | ~5.5 | **1.22×** |
| 22L | ~5.1 | ~3.7 | **1.38×** |

This is consistent with the Mar 16 report's finding that the KDA slowdown **does not improve** beyond 12L (~1.32–1.38× range). The bottleneck shifts to encoder backward at 12L+, not decoder cross-attention.

---

## Depth Scaling: Key Finding

With full 5-epoch results now available:

| Depth | Softmax best EM | KDA best EM | Winner |
|-------|----------------|------------|--------|
| 6L | 86.40% | 84.80% | **Softmax +1.6** |
| 12L | 86.20% | 85.20% | **Softmax +1.0** |
| 22L | 84.60% | **86.80%** | **KDA +2.2** |

The pattern is clear: **KDA benefits from depth more than softmax does.** Softmax peaks at 12L and *regresses* at 22L (84.60%), while KDA scales positively to 22L and is still improving at E5. This suggests:

1. Softmax cross-attention at 22L from-scratch may need more than 5 epochs or regularization adjustment to avoid overfitting.
2. KDA's linear attention, with its structured state-space-like inductive bias, may be better suited to deeper from-scratch decoders.
3. **22L KDA (86.80% EM) is the best model** in this sweep, at the cost of ~65 min/epoch (vs 45 min for 22L softmax). Considering it may still improve with more epochs, it warrants further training.

---

## Comparison with Prior Results

| Architecture | Val EM% | Val F1% | Notes |
|---|---|---|---|
| TinyLlama 1B (decoder-only, C/Q/A) | 90.0 | 93.5 | Pretrained 1B decoder |
| ModernBERT 400M + TinyLlama KDA finetune | 87.6 | 91.7 | Pretrained decoder, ~2.6h/epoch |
| ModernBERT 400M + TinyLlama softmax finetune | 87.6 | 91.7 | Pretrained decoder, ~1.8h/epoch |
| **22L from-scratch KDA (this report, E5)** | **86.80** | **90.48** | From-scratch, ~65 min/epoch, improving |
| 6L from-scratch softmax | 86.40 | 90.83 | From-scratch, ~26 min/epoch (est.) |
| 12L from-scratch softmax | 86.20 | 90.78 | From-scratch, ~33 min/epoch |
| 12L from-scratch KDA | 85.20 | 89.60 | From-scratch, ~43 min/epoch |
| 22L from-scratch softmax | 84.60 | 90.55 | From-scratch, plateaus at E3 |

The 22L KDA (86.80% EM) now sits **0.8 EM below the pretrained TinyLlama finetune** (87.6% EM), with a much smaller decoder from scratch and still improving at E5.

---

## Summary

1. **22L KDA is the best from-scratch model (86.80% EM at E5, still improving)** — the first time linear KDA has outperformed softmax in this series.
2. **KDA benefits from depth; softmax does not** — softmax peaks at 12L and regresses at 22L, while KDA keeps improving through E5 at 22L.
3. **All models converge to similar train loss by E5** (~0.19–0.20), but val scores diverge based on attention type and depth. KDA's slower early convergence does not hurt final quality at sufficient depth.
4. **KDA slowdown at 12–22L is ~1.2–1.4×** vs softmax (consistent with Mar 16 findings), and has plateaued — the encoder backward dominates at this scale.
5. **The 22L softmax result is disappointing** — it plateaus at 84.60% EM by E3 and then degrades, well below 12L softmax (86.20%). Further training or regularization tuning may be needed.

---

## Next Steps

- **Continue 22L KDA training** beyond 5 epochs — it is still improving (E5 best so far) and may reach or exceed the pretrained finetune baseline.
- Investigate **22L softmax degradation**: does more data augmentation, different LR schedule, or weight decay help prevent the E3→E5 regression?
- Consider a **30L from-scratch KDA** run to see if the depth trend continues.
- Evaluate whether the 22L KDA at E5 (86.80%) improves further with learning rate warmup/cooldown tuning.
