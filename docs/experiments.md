# SPD v13 Experimental Results

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

## Architecture Variants Tested
| Config | Params (decoder) | AbsRel (best) | Notes |
|--------|-----------------|---------------|-------|
| B1-B3 + calib + L_point | ~2.3M | 0.270 | Stalled without SILog |
| B1-B3 + calib + L_point+SILog | ~2.3M | 0.256 | Far depths still compressed |
| B1-B5 + log-depth + L_point+SILog | ~3.9M | 0.231 (ep2) | L_point caused ep3 regression |
| B1-B5 + log-depth + SILog(λ=0.85) | ~3.9M | 0.239 (ep1) | Ep2 regression, architectural issue confirmed |
