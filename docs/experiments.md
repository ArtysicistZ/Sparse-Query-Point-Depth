# SPD Experimental Results

## Setup
- **Dataset:** NYU Depth V2 (50K train, 654 val)
- **Resolution:** 256×320
- **GPU:** RTX 4060 Laptop (8GB VRAM)
- **Optimizer:** AdamW, encoder 1e-5 / decoder 1e-4, weight_decay=0.01
- **Scheduler:** CosineAnnealingLR, eta_min=1e-6
- **AMP:** bfloat16
- **K:** 256 query points

---

## Exp 1: B1-B3 + Calibration + L_point only
- **Architecture:** B1→B2→B3 + depth_head(MLP 74K) + global calibration (s, b)
- **Loss:** L_point (Huber on inverse depth)
- **Batch:** 8 → 12

| Epoch | AbsRel | Pred Range | GT Range | Notes |
|-------|--------|------------|----------|-------|
| 1     | 0.266  | ~[0.9, 5.0] | [0.7, 10] | |
| 2     | 0.270  | ~[0.9, 5.0] | [0.7, 10] | Stalled |
| 6     | 0.270  | ~[0.9, 5.0] | [0.7, 10] | Training loss still decreasing, val stuck |

**Conclusion:** L_point alone insufficient. Model compresses predictions to ~1-5m.

---

## Exp 2: B1-B3 + Calibration + L_point + 0.5*L_silog
- **Architecture:** Same as Exp 1
- **Loss:** L_point + 0.5 * L_silog (sqrt, λ_var=0.85)
- **Batch:** 12

| Epoch | AbsRel | Pred Range | GT Range | Notes |
|-------|--------|------------|----------|-------|
| 1     | ~0.26  | ~[0.9, 5.0] | [0.7, 10] | |
| 2     | 0.256  | [0.90, 5.61] | [0.90, 9.90] | Slight improvement |
| 3     | 0.259  | ~[0.9, 4.8] | [0.7, 10] | Converged, far depths still compressed |

**Conclusion:** SILog helps slightly (0.27→0.256). Far-depth compression persists. Global calibration (s, b) is the bottleneck.

---

## Exp 3: B1-B5 + Log-depth (no calibration) + L_point + 0.5*L_silog
- **Architecture:** B1→B2→B3→B4→B5, depth = exp(log_depth), no global s/b
- **Loss:** L_point(1/pred, gt) + 0.5 * L_silog(pred, gt), λ_var=0.85
- **Batch:** 8 (7.4GB VRAM)
- **Params:** ~36.5M total
- **Scheduler:** T_max=30 (mismatch with 10 epochs)

| Epoch | AbsRel | Pred Range | GT Range | Notes |
|-------|--------|------------|----------|-------|
| 1     | 0.255  | [0.77, 4.73] | [0.71, 9.91] | Already matches Exp 2 converged |
| 2     | 0.231  | [0.77, 5.85] | [0.71, 9.90] | Max pred +1.2m, AbsRel -9% |
| 3     | 0.260  | [~0.8, 5.75] | [0.71, 9.90] | Regression — L_point fighting SILog |

**Conclusion:** Log-depth + B4-B5 learns faster than calibration (0.231 vs 0.256 in 2 epochs). But L_point conflicts with SILog for far depths — causes oscillation at epoch 3. Pred range still capped at ~6m.

---

## Exp 4: B1-B5 + Log-depth + SILog only (λ=0.85)
- **Architecture:** Same as Exp 3
- **Loss:** L_silog only, λ_var=0.85 (stronger scale penalty for metric depth)
- **Batch:** 8 (7.4GB VRAM)
- **Scheduler:** T_max=10 (matches epochs)
- **Changes from Exp 3:** Dropped L_point, λ 0.85, T_max 30→10

| Epoch | AbsRel | Pred Range | GT Range | Notes |
|-------|--------|------------|----------|-------|
| 1     | 0.239  | [0.96, 6.03] | [0.71, 9.90] | Better than Exp 3 ep1 (0.255), max pred +1.3m |
| 2     | 0.251  | [0.89, 5.41] | [0.71, 9.96] | Regression again — max pred shrinking, same pattern as Exp 3 |

**Conclusion:** Dropping L_point didn't fix the regression pattern. Epoch 2 regresses like Exp 3 ep3. Loss function was NOT the root cause — the issue is architectural. Max pred shrinking (6.0→5.4) confirms the model can't sustain far-depth learning.

---

## Key Observations
1. Global calibration (softplus(r_q * s + b)) bottlenecks far-depth prediction — 2 scalars per image too restrictive
2. L_point biases toward near depths (gradient ∝ 1/depth²) and conflicts with SILog, causing oscillation (Exp 3 ep3 regression)
3. L_silog provides uniform gradient in log-space, essential for full depth range
4. λ_var: 0.85 used in Exp 1-4. Lower λ (e.g. 0.5) may be better for metric depth — TBD
5. **Regression pattern persists across Exp 3 and 4**: model peaks at epoch 1-2 then regresses regardless of loss function. This points to an architectural issue, not a training dynamics issue.
6. T_max must match actual epoch count for cosine schedule to decay properly

---

## Architectural Investigation (post-Exp 4)

### Recurring pattern
All experiments show: fast initial learning (epoch 1-2), then regression or plateau. Max pred consistently caps at ~5-6m against GT ~10m. This persists across loss functions (L_point+SILog, SILog-only) and LR schedules (T_max=30, T_max=10).

### Issue 1: B5 token imbalance (91 local vs 32 deformable)
- `fused_tokens = cat([query_tokens, deform_tokens])` → 91 + 32 = 123 tokens
- B2 already processed the same 91 local tokens — B5 re-reads them (74% of attention budget)
- The 32 deformable tokens from B4 are the only NEW information in B5, but outnumbered ~3:1
- B5 attention is biased toward familiar local tokens, underweighting multi-scale depth cues
- **Fix options:** Remove local tokens from B5 (rely on B2's encoding in h), or rebalance ratio

### Issue 2: B3→B4 bootstrapping problem
- B3 routing (top-32 L4 tokens) drives B4's deformable anchors
- Early training: B3 attention is near-random → B4 gets random anchors → noisy deformable tokens
- B5 learns to ignore noisy deformable tokens → less gradient to B3/B4 → they don't improve
- Self-reinforcing loop: model locks into relying on local features only
- **Fix options:** Init B4 offsets to 0 (sample anchor centers), or add L4 tokens directly to B5

### Issue 3: Depth head initialization bias
- Final `Linear(384→1)` in B5 has bias=0 → initial predictions ≈ exp(0) = 1.0m
- NYU depth range [0.7, 10m]: reaching 10m needs output=2.3 (+2.3 from init), reaching 0.7m needs -0.36
- Learning burden is 6× harder for far depths than near depths from initialization alone
- **Fix:** Init final bias to log(2.5) ≈ 0.92, centering predictions at median depth

### Issue 4: Low L4 resolution at 256×320
- L4 = 8×10 = 80 tokens. Top-32 routing selects 40% — barely selective
- Architecture designed for 480×640 where L4 = 15×20 = 300, top-32 = 11%
- B1's local L4 tokens (3×3=9) cover 11% of all L4 tokens — significant overlap with global set
- **Fix:** Scale to 480×640, or reduce top-K at low resolution

### Proposed fix priority
| Fix | Impact | Complexity | Risk |
|-----|--------|-----------|------|
| Depth head bias init (log 2.5) | Medium | 1 line | None |
| B5: use only deform_tokens (drop local) | High | Small edit | May lose local detail |
| B4: init offsets to 0 | Medium | 1 line | None |
| B3: increase to 3 layers | Medium | Small | +370K params |
| Scale to 480×640 | High | Config change | VRAM limit |

---

## Exp 5: v13.1 — B1→B2→B3a→B3b→B5, SILog only (λ=0.5), depth head bias
- **Architecture:** v13.1 redesign — separate B3a (L3, 20 routed) + B3b (L4, 10 routed), B5 fused (33 tokens: 3 central + 20 L3 + 10 L4), no B4
- **Depth head:** bias init log(2.5) = 0.916
- **Loss:** L_silog only, λ_var=0.5
- **Batch:** 8, K=256
- **Scheduler:** CosineAnnealingLR, T_max=10, eta_min=1e-6

| Epoch | Train Loss (end) | AbsRel | Pred Range | GT Range | Notes |
|-------|-----------------|--------|------------|----------|-------|
| 1     | ~0.13           | 0.267  | [0.70, 4.41] | [0.71, 9.90] | Pred max ~4.4m, similar to Exp 1-2 start |
| 2     | ~0.10           | 0.238  | [0.80, 5.53] | [0.71, 9.90] | Improving, max pred +1.1m |
| 3     | ~0.09           | 0.230  | [0.85, 6.17] | [0.71, 9.94] | **Best.** Max pred +0.6m, still growing |
| 4     | ~0.08           | 0.262  | [0.82, 5.32] | [0.71, 9.90] | **Overfit.** Train loss ↓ but val AbsRel ↑, max pred shrinking |

**Conclusion:** Best AbsRel 0.230 at epoch 3, matching Exp 3's best (0.231). Overfits after epoch 3 — train loss keeps decreasing (0.09→0.08) but val AbsRel regresses (0.230→0.262). The v13.1 architecture changes (separate B3a/B3b, type embeddings, bias init, no B4) did not break anything but also did not significantly improve over Exp 3. Pred max reached ~6.2m (vs ~5.9m in Exp 3), slightly better far-depth reach. Overfitting after 3-4 epochs is a consistent pattern across all experiments.

**Next steps to investigate:**
1. Add regularization (dropout in decoder, stronger weight decay)
2. Data augmentation (random crop, color jitter, horizontal flip)
3. Increase K (more query points per image for denser supervision)
4. Early stopping at epoch 3
5. Run with new detailed evaluation metrics to diagnose failure modes

---

## Exp 6: v13.1 + ImageNet norm + per-image SILog (λ=0.5) + augmentation
- **Architecture:** Same as Exp 5
- **Fixes:** ImageNet normalization, per-image SILog (was global), data augmentation (flip + color jitter)
- **Loss:** L_silog per-image, λ_var=0.5
- **Batch:** 8, K=256
- **Scheduler:** CosineAnnealingLR, T_max=10, eta_min=1e-6

| Epoch | AbsRel | Pred Range | GT Range | L3 Routing | Notes |
|-------|--------|------------|----------|------------|-------|
| 1     | 0.261  | [0.67, 5.24] | [0.71, 9.95] | 67.6 unique | Slower start than Exp 5 |
| 2     | 0.259  | [0.69, 5.24] | [0.71, 9.90] | 48.4 unique | Barely improving, pred max stuck |
| 3     | 0.254  | [0.74, 5.80] | [0.71, 9.94] | 43.4 unique | Slight improvement, routing collapsing |

**Conclusion:** Per-image SILog with λ=0.5 forgives too much scale error → 5× slower learning than Exp 5. Augmentation prevents overfitting (no regression at ep3) but learning is too slow. Routing diversity declining. Stopped after epoch 3.

---

## Exp 7: v13.1 + all fixes + SILog (λ=0.15)
- **Architecture:** Same as Exp 5
- **Fixes:** ImageNet normalization, per-image SILog, data augmentation (flip + color jitter)
- **Loss:** L_silog per-image, λ_var=0.15 (stronger metric signal)
- **Batch:** 8, K=256
- **Scheduler:** CosineAnnealingLR, T_max=10, eta_min=1e-6

| Epoch | AbsRel | Pred Range | s* | L3 Routing | 5-10m AbsRel | Notes |
|-------|--------|------------|-----|------------|-------------|-------|
| 1     | 0.297  | [0.61, 4.73] | 1.383 | 70.9 unique | 0.517 | Slow start, scale off (38%), routing healthy |
| 2     | **0.239** | [0.70, 4.73] | 1.165 | 57.2 unique | 0.476 | **Best.** Scale nearly correct, pred max stuck at 4.73 |
| 3     | 0.259  | [0.69, 5.94] | 1.260 | 49.1 unique | 0.479 | Overfit. Pred max broke through (5.94) but overall regressed |

**Conclusion:** λ=0.15 successfully fixed scale learning (s* 1.38→1.17 in one epoch). Best AbsRel 0.239 at epoch 2, comparable to Exp 5's 0.238. But pred max ceiling at 4.73m persists (log-depth head capped at ~1.55). Routing collapses by epoch 3 (70.9→49.1). 5-10m range remains catastrophic (~0.48 AbsRel). The ~0.23 AbsRel ceiling appears fundamental to the current architecture — consistent across Exp 3, 5, 6, 7 regardless of loss function.

**Key insight:** Dense models get spatial smoothness from convolutions for free — each pixel prediction is informed by neighboring predictions. SPD query points are isolated predictions with no spatial prior, making the task fundamentally harder per-point.

---

## Architecture Variants Tested
| Config | Params (decoder) | AbsRel (best) | Notes |
|--------|-----------------|---------------|-------|
| B1-B3 + calib + L_point | ~2.3M | 0.270 | Stalled without SILog |
| B1-B3 + calib + L_point+SILog | ~2.3M | 0.256 | Far depths still compressed |
| B1-B5 + log-depth + L_point+SILog | ~3.9M | 0.231 (ep2) | L_point caused ep3 regression |
| B1-B5 + log-depth + SILog(λ=0.85) | ~3.9M | 0.239 (ep1) | Ep2 regression, architectural issue confirmed |
| v13.1: SILog(λ=0.5), no aug | ~3.9M | 0.230 (ep3) | Overfits after ep3, max pred ~6.2m |
| v13.1: SILog(λ=0.5), per-img+aug | ~3.9M | 0.254 (ep3) | No overfit but 5× slower learning |
| **v13.1: SILog(λ=0.15), per-img+aug** | ~3.9M | **0.239 (ep2)** | Scale fixed, routing collapse + pred ceiling persist |

---

# v14 Experiments

## v14 Architecture Summary
- **Encoder:** ConvNeXt V2-T (FCMAE pre-trained), fine-tuned 0.1× LR, ~28.6M
- **Neck:** Conv(k1)+LN per level → uniform d=192, + 2× L4 Self-Attention (~1.2M)
- **Decoder:** Multi-scale seed → 3×MSDA (deformable cross-attn) → 2×B3b (L4 cross-attn) → depth head
- **Dual Spatial Canvas:** L2 (stride 8, DWConv5×5) + L3 (stride 16, DWConv3×3), write-smooth-read at every layer (~809K)
- **All 5 decoder layers:** cross-attn → dual canvas write-smooth-read → Q2Q self-attn → FFN
- **Depth head:** MLP(192→384→1), depth = exp(log_depth), learnable scale exp(s)
- **Total params:** ~33.6M (encoder 28.6M + neck/SA 1.2M + precompute 0.1M + decoder 3.7M)

### Key changes from v13.1
- Uniform d=192 neck (was [64,128,192,384])
- Multi-scale seed from all 4 levels (was single-level B1)
- Dual Spatial Canvas replaces no-canvas architecture
- 3×MSDA deformable cross-attn replaces B2+B5
- B3a removed (redundant with L3 canvas + MSDA)
- Q2Q self-attn inside every layer (was standalone)
- Dense auxiliary loss on L2/L3 neck features (new)

---

## Exp 8: v14 full pipeline — SILog + dense aux
- **Architecture:** v14 (see summary above)
- **Loss:** L_silog(λ=0.15) + 0.5 × (L_dense_silog(aux_L2) + L_dense_silog(aux_L3))
- **Batch:** 8, K=256
- **Optimizer:** AdamW, encoder 1e-5 / decoder 1e-4, weight_decay=0.01
- **Scheduler:** CosineAnnealingLR, T_max=10, eta_min=1e-6
- **AMP:** bfloat16

| Epoch | Train Loss | AbsRel | SILog | d<1.25 | Pred Range | GT Range | s* | Notes |
|-------|-----------|--------|-------|--------|------------|----------|----|-------|
| 1     | ~0.45     | 0.2743 | 0.3533 | 36.9% | [0.58, 4.81] | [0.71, 9.97] | 1.314 | Head barely learning (std=0.28), 5-10m: 0.526 |
| 2     | ~0.39     | 0.2601 | 0.3372 | 40.6% | [0.66, 5.00] | [0.71, 9.98] | 1.282 | Improving, no regression. 0-2m: 0.190, 2-5m: 0.271, 5-10m: 0.510 |
| 3     | ~0.35     | **0.2403** | 0.3171 | **47.1%** | [0.73, 6.03] | [0.71, 9.97] | 1.219 | Head OK (std=0.30). 0-2m: 0.194, 2-5m: 0.240, 5-10m: 0.474. Scale gap 5.8% |
| 4     | ~0.33     | 0.2409 | 0.3148 | 46.3% | [0.68, 5.89] | [0.71, 9.97] | 1.230 | **Plateau.** All metrics flat. Pred max shrank (6.03→5.89). Scale gap 7.3% |
| 5     | ~0.31     | 0.2651 | 0.3353 | 38.0% | [0.69, 5.67] | [0.71, 9.96] | 1.312 | **Overfit.** Scale regressed (s* 1.23→1.31). d<1.25 dropped 8pp. Scale gap 16.8% |

**Conclusion:** Plateau at epoch 3-4, overfit by epoch 5. Weak augmentation + low resolution + λ=0.15 insufficient for continued learning.

---

## Exp 9: v14.1 — 416×544, BTS augmentation, triple canvas, λ=0.50
- **Architecture:** v14 + triple canvas L2(DWConv5×5) + L3(DWConv5×5) + L4(DWConv3×3)
- **Resolution:** 416×544 (random crop from 480×640, BTS standard)
- **Augmentation:** BTS-style (rotation ±2.5°, gamma 0.9-1.1, brightness 0.75-1.25, per-channel color 0.9-1.1, hflip)
- **Loss:** L_silog(λ=0.50) + 0.5 × (L_dense_silog(aux_L2) + L_dense_silog(aux_L3)) — neck aux
- **Batch:** 4, K=256
- **Optimizer:** AdamW, encoder 1e-5 / decoder 1e-4, weight_decay=0.01
- **Scheduler:** CosineAnnealingLR, T_max=10, eta_min=1e-6

| Epoch | Train Loss | AbsRel | SILog | d<1.25 | Pred Range | GT Range | s* | Notes |
|-------|-----------|--------|-------|--------|------------|----------|----|-------|
| 1     | ~0.42     | 0.2535 | 0.3348 | 43.8% | [0.69, 5.53] | [0.71, 9.95] | 1.239 | Head barely learning (std=0.28). 0-2m: 0.206, 2-5m: 0.251, 5-10m: 0.506. Scale gap 4.9% |
| 2     |           | **0.2178** | 0.2876 | **56.6%** | [0.72, 6.42] | [0.71, 9.96] | 1.117 | Head OK (std=0.33). 0-2m: 0.238, 2-5m: 0.187, 5-10m: 0.402. **Scale nearly solved** (s*=1.12). Scaled AbsRel WORSE (-2.1%) — structure is the issue now, not scale |
| 3     |           | 0.2240 | 0.2959 | 53.3% | [0.71, 7.11] | [0.71, 9.96] | 1.152 | **Slight regression.** Pred max expanded (6.42→7.11) but AbsRel worsened. 0-2m: 0.218, 2-5m: 0.207, 5-10m: 0.411. Scale regressed (1.12→1.15) |

---

## Exp 10: v14 final — dual canvas, full NYU, large batch
- **Architecture:** v14 final (dual canvas L2 DWConv5×5 + L3 DWConv3×3, no triple canvas)
- **Loss:** L_silog(λ=0.15) + 0.5 × (L_dense_silog(aux_L2) + L_dense_silog(aux_L3))
- **Batch:** ~92 (inferred: 517 steps/epoch × 92 ≈ 47584 = full NYU train set), K=256
- **Optimizer:** AdamW, encoder 1e-5 / decoder 1e-4, weight_decay=0.01
- **Scheduler:** CosineAnnealingLR, T_max=10, eta_min=1e-6
- **AMP:** bfloat16

| Epoch | Train Loss | AbsRel | SILog | d<1.25 | Pred Range | GT Range | s* | Notes |
|-------|-----------|--------|-------|--------|------------|----------|----|-------|
| 1     | ~0.21     | 0.2551 | 0.3371 | 43.9% | [0.691, 5.281] | [0.714, 9.955] | 1.237 | Head barely learning (std=0.28). 0-2m: 0.209, 2-5m: 0.254, 5-10m: 0.497. Scale gap 4.2% |
| 2     | ~0.17     | 0.2591 | 0.3371 | 41.5% | [0.679, 5.578] | [0.713, 9.970] | 1.269 | **Regression.** Head OK (std=0.30). 0-2m: 0.202, 2-5m: 0.264, 5-10m: 0.493. Scale gap worsened to 9.1% (scaled AbsRel 0.2357). SILog unchanged from ep1 (0.3371) |

**Conclusion:** Epoch 1→2 regression despite head std improving (barely→OK). SILog metric flat across both epochs (0.3371) while AbsRel worsened — suggests scale shift rather than structural degradation. s* worsened (1.237→1.269): model is learning wrong scale direction. Pred max slightly expanded (5.28→5.58) but all-range accuracy dropped. Pattern consistent with prior v13.1 experiments: fast first-epoch gains, then regression. Large batch size may be reducing gradient noise too aggressively, hurting generalization.

---

# v15 Experiments

## v15 Architecture Summary
- **Key insight:** v14 canvas is causally backwards — queries write to dense maps, then read. DPT builds dense maps bottom-up from encoder (correct direction). v15 replaces the entire v14 decoder with a standard DPT refinenet hierarchy, split at the stride-8 boundary: global (dense) + local (per-query at inference, dense at training).
- **Global path:** DPT FPN fusion refinenet4(L4) + refinenet3(L3) → stride_8_map at D=64, identical to DAv2-S.
- **Local path:** Progressive refinenet: refinenet2 (fuses L2, stride 8→4) → refinenet1 (fuses L1, stride 4→2) → head (output_conv1 64→32 + output_conv2 Conv3×3(32→32)+ReLU+Conv1×1(32→1)).
- **Training:** Purely dense — same conv weights run with padded convolutions on full feature maps → H×W depth → dense SILog. No K queries at training.
- **Inference:** Per-query valid convolutions on small patches extracted via grid_sample. 3 grid_sample calls (stride_8_map 9×9, f2 9×9, L1 7×7).
- **Depth activation:** F.softplus (changed from F.relu after dead-ReLU bug in Exp 12).
- **All 4 encoder levels used:** L4+L3 (global refinenet4+3), L2 (local refinenet2), L1 (local refinenet1).

---

## Exp 11: v15.0 — ConvNeXt V2-T + DPT decoder, full NYU (47K), dense training
- **Architecture:** v15.0 — ConvNeXt V2-T (FCMAE pre-trained, ~28.6M) + DPT decoder (D=64, ~350K)
- **Encoder:** ConvNeXt V2-T, 4 stages [96,192,384,768], FCMAE pre-trained
- **Neck:** Conv1×1+LN per level → D=64 (projections for L1–L4)
- **Loss:** L_dense_silog, λ_var=0.50
- **Batch:** 12
- **Dataset:** Full NYU train (47,584 images) — **first experiment using full dataset** (Exp 1–10 used only ~4K images)
- **Optimizer:** AdamW, encoder 1e-5 / decoder 1e-4, weight_decay=0.01
- **Scheduler:** Linear warmup (500 steps) → CosineAnnealingLR, eta_min=1e-6
- **Total steps:** 19,830 (3,966 steps/epoch × 5 epochs planned)
- **AMP:** bfloat16
- **Depth activation:** F.relu + 1e-6 (later found to be buggy — see Exp 12)
- **Total params:** ~29.0M (encoder 28.6M + neck ~60K + decoder ~350K)

| Epoch | Train Loss | AbsRel | SILog | d<1.25 | s* | Scaled AR | Pred CV | Notes |
|-------|-----------|--------|-------|--------|-----|-----------|---------|-------|
| 1     | 0.158     | 0.2155 | 0.2274 | 65.6% | 0.873 | 0.1493 | 0.465 | Head barely learning (std=0.28). 0-2m: 0.266, 2-5m: 0.195, 5-10m: 0.159 |
| 2     | 0.107     | **0.2035** | **0.2177** | **69.0%** | 0.877 | 0.1399 | 0.459 | **Best.** Scale improved. 0-2m: 0.269, 2-5m: 0.174, 5-10m: 0.165. Scale gap 31% |

**Conclusion:** Strong results — AbsRel 0.2035 at epoch 2, already surpassing all v13/v14 experiments (best was 0.2178 in Exp 9). Full NYU dataset (47K vs ~4K) and dense training (100% pixel coverage vs K=256 sparse) are the main drivers. ConvNeXt V2-T + simple DPT decoder proves that the architecture works; the v13/v14 complexity was unnecessary. Scale bias persists (s*=0.877, scaled AR 0.14 vs raw 0.20) — model predicts ~14% too high. Training stopped at epoch 2 due to transition to v15.1.

**Key discovery:** Using full NYU (47K images) instead of ~4K subset made a massive difference. All Exp 1–10 results were handicapped by the small training set.

---

## Exp 12: v15.1 — DINOv2 ViT-S + DPT decoder, full NYU (47K), dense training
- **Architecture:** v15.1 — DINOv2 ViT-S (self-supervised pre-trained, ~22M) + DPT decoder (D=64, ~350K)
- **Encoder:** DINOv2 ViT-S/14, uniform 384-dim features, pre-trained on LVD-142M (purely self-supervised, zero depth data)
- **Neck:** Conv1×1+LN per level → D=64 (uniform 384→64 projection)
- **Loss:** L_dense_silog, λ_var=0.85 (changed from 0.50 — proven from-scratch standard per BTS/AdaBins/PixelFormer)
- **Batch:** 2 (limited by VRAM with ViT-S)
- **Dataset:** Full NYU train (47,584 images), 23,792 steps/epoch
- **Optimizer:** AdamW, encoder 5e-6 / decoder 1e-4, weight_decay=0.01
- **Scheduler:** Linear warmup (500 steps) → CosineAnnealingLR, eta_min=1e-6
- **Total steps:** 237,920 (23,792 steps/epoch × 10 epochs)
- **AMP:** bfloat16
- **Depth activation:** F.softplus (fixed from F.relu — see bug below)
- **Total params:** ~22.4M (encoder ~22M + neck ~100K + decoder ~350K)

### Attempt 1: F.relu depth activation (FAILED — dead ReLU)
- **Config:** encoder 1e-5, decoder 1e-4, λ_var=0.50
- **Bug:** `F.relu(out) + 1e-6` on final conv output. With random Kaiming init, ~50% of output_conv2 outputs are negative → ReLU kills them to 0 → zero gradient → model cannot learn.
- **Symptom:** Predictions stuck at 0.00±0.00 for 3000+ steps. Loss constant ~5.7 (= sqrt((log(1e-6) - log(gt))² × 0.15)).
- **Root cause:** DAv2 uses F.relu too, but fine-tunes from a pretrained model where output layer already produces positive values. Training from scratch (decoder randomly initialized) → dead ReLU.
- **Fix:** Replaced F.relu with F.softplus on both training and inference paths. softplus(0)=log(2)≈0.69, always has gradient.

### Attempt 2: Encoder LR too high (epoch 2 regression)
- **Config:** encoder 1e-5, decoder 1e-4, λ_var=0.50

| Epoch | Train Loss | AbsRel | SILog | d<1.25 | s* | Scaled AR | Notes |
|-------|-----------|--------|-------|--------|-----|-----------|-------|
| 1     | 0.164     | 0.2249 | 0.2209 | 64.1% | 0.850 | 0.1324 | Good structure (scaled AR 0.132). Scale bias: predicts 18% too high |
| 2     | 0.099     | 0.2745 | 0.2387 | 51.6% | 0.811 | 0.1329 | **Regression.** Scaled AR unchanged (0.133) — structure preserved, scale diverged (s* 0.85→0.81) |

- **Diagnosis:** Encoder LR 1e-5 too high for DINOv2 (whose features are already depth-relevant from self-supervised pretraining). Feature drift destabilized decoder's scale mapping. Structure survives (relative ordering robust) but absolute scale breaks.
- **Fix:** Reduced encoder LR to 5e-6, changed λ to 0.85 (proven from-scratch standard).

### Attempt 3: Corrected parameters (in progress)
- **Config:** encoder 5e-6, decoder 1e-4, λ_var=0.85, F.softplus activation
- **Status:** Training restarted with corrected parameters. Results pending.

**Key learnings:**
1. **Dead ReLU is fatal for from-scratch decoder training.** F.softplus is the safe default when output layer init is random.
2. **DINOv2 encoder LR must be lower than ConvNeXt.** DINOv2's self-supervised features are already depth-relevant; too-high LR causes feature drift that destabilizes decoder.
3. **λ_var=0.85 is the proven from-scratch standard** (BTS, AdaBins, PixelFormer all use it). λ=0.50 was DAv2's fine-tuning setting (different regime).
4. **Full NYU (47K) vs ~4K subset** used in Exp 1–10 is a critical difference. Exp 1–10 results are not directly comparable to Exp 11–12.

---

## Architecture Variants Tested (v15)
| Config | Encoder | Params | Dataset | AbsRel (best) | Notes |
|--------|---------|--------|---------|---------------|-------|
| v15.0: ConvNeXt V2-T + DPT, λ=0.50, bs=12 | ConvNeXt V2-T (28.6M) | ~29.0M | 47K NYU | **0.2035** (ep2) | Best overall. Scale bias 31% |
| v15.1: DINOv2 ViT-S + DPT, λ=0.50, bs=2, enc LR 1e-5 | DINOv2 ViT-S (22M) | ~22.4M | 47K NYU | 0.2249 (ep1) | Ep2 regression (feature drift). Lower params but worse result |
| v15.1: DINOv2 ViT-S + DPT, λ=0.85, bs=2, enc LR 5e-6 | DINOv2 ViT-S (22M) | ~22.4M | 47K NYU | *pending* | Corrected parameters |
