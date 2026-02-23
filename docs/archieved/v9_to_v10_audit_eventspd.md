# EventSPD v9 Formula Audit Report

**Date:** 2026-02-16
**Scope:** Sections 4A, 4B, and 5 of `research_plan_eventspd_v9.md`, with cross-checks against Sections 6 and 12.
**Goal:** Find correctness errors, stale numbers, inconsistencies, and improvement opportunities. Not to prove the design is correct — to find where it can be improved.

---

## Audit Methodology

Same framework as the v4.5→v5 audit:
1. **Correctness:** Is the formula written correctly? Any numerical inconsistency?
2. **Reference support:** Is this backed by published work? Any false claims?
3. **Logical meaning:** Is anything redundant or under-justified?
4. **Constants:** Are dimensions, counts, and FLOPs properly chosen?
5. **Cross-section consistency:** Do numbers in Section 4 match Sections 6 and 12?

Severity categories:
- **[BUG]** — Incorrect number, stale value, or factual error
- **[CONCERN]** — Design choice that may hurt performance or needs justification
- **[AMBIGUITY]** — Correct but could cause implementation confusion
- **[SUGGESTION]** — Improvement opportunity

---

## Summary of Findings

| # | Location | Severity | Summary |
|---|----------|----------|---------|
| V01 | B2 §4.4 line 475 | BUG | B2 per-query cost "~1.0M FLOPs" is stale — should be ~2.8M |
| V02 | §4.7 line 725 | BUG | Latency "~7.8 ms at K=256" is stale — should be 7.4 ms |
| V03 | §6.1 line 890 | BUG | DAv2 comparison uses stale narrow-pyramid numbers (5.3M, "12×") |
| V04 | §12 line 1376 | BUG | RGB-SPD L4 nonlinear stages: says 4, should be 8 (4 layers × 2 sub-layers) |
| V05 | A2 backbone table | BUG | L4 FLOPs (~6.2G) and L3 FLOPs (~3.3G) are numerically wrong |
| V06 | A2 line 220 | CONCERN | L3 at 80×45 with window_size=8: 45 not divisible by 8 |
| V07 | A3 line 294-299 | CONCERN | B2 KV projections shared across 2 cross-attention layers |
| V08 | B3 line 486 | CONCERN | B3 conditioning is single linear + LN (shallow for 3-way integration) |
| V09 | B3 line 524 | CONCERN | No per-level offset normalization (unlike Deformable DETR) |
| V10 | B1 line 573 | AMBIGUITY | l_q appears both in h_point AND as separate B4 context token (undocumented skip) |
| V11 | B1 line 365 | AMBIGUITY | Local MLP output dimension (192) not stated in B1 text |
| V12 | §5.4 line 817 | AMBIGUITY | L_ctr delayed to Stage 2 without justification |
| V13 | §5.3 line 798 | AMBIGUITY | Query sampling categories can overlap — not addressed |
| V14 | §6.2 line 917 | AMBIGUITY | Backbone timing: "rest ~0.3 ms" is unexplained, L2/L3 times differ from table |
| V15 | B2 | SUGGESTION | Add LN on K_t/V_t in B2 (B4 has LN_kv but B2 does not) |
| V16 | A2 | SUGGESTION | Consider window_size=5 for L3 (divides both 80 and 45) |

**Severity counts:** 5 BUGs, 4 CONCERNs, 4 AMBIGUITIEs, 2 SUGGESTIONs, 1 systemic issue (FLOPs convention)

---

## Phase A: Precompute

### A1. Backbone encoding (F^3 ds2)

**[OK]** — F^3 ds2 description is thorough and well-justified. The ds2 config (dsstrides=[2,1], dims=[64,64]) is clearly specified. Timing breakdown (Table at lines 900-911) is detailed. The roofline analysis (Section 8, ablation #35) provides solid evidence for the memory-bandwidth argument.

---

### A2. Wide Pyramid Backbone

#### Finding V05 [BUG] — L4 and L3 FLOPs are numerically incorrect

The backbone cost table (lines 267-278) claims:
- L3: 4× SwinBlock₁₉₂ → **~3.3G**
- L4: 2× FullSelfAttn₃₈₄ → **~6.2G**
- Backbone total: **~16.1G**

**Independent verification (using MACs, the standard "paper FLOPs" convention):**

**L4 (full self-attention, N=880, D=384, H=6, d_head=64, d_ff=1536):**

Per layer:
| Operation | MACs |
|---|---:|
| QKV projection (3×D²×N) | 389.3M |
| QK^T (N²×D, all heads) | 297.3M |
| attn×V (N²×D, all heads) | 297.3M |
| O projection (D²×N) | 129.8M |
| FFN (2×D×d_ff×N) | 1,038.6M |
| **Per layer total** | **2,152M** |

Two layers: **~4.3G MACs** (not 6.2G — the document overcounts by ~44%)

**L3 (windowed self-attention, N=3600, D=192, H=6, d_head=32, d_ff=768, W=8):**

Per block (assuming 60 windows of 64 tokens after padding 45→48):
| Operation | MACs |
|---|---:|
| QKV projection (3×D²×N) | 398.1M |
| QK^T (W²×d_head×H×N_win) | 47.2M |
| attn×V (same) | 47.2M |
| O projection (D²×N) | 132.7M |
| FFN (2×D×d_ff×N) | 1,061.7M |
| **Per block total** | **~1,687M** |

Four blocks: **~6.75G MACs** (not 3.3G — the document undercounts by ~2×)

**Impact:** The individual level FLOPs are wrong in opposite directions (L3 undercounted, L4 overcounted), so the total (~16.1G) may be approximately correct by coincidence. But the **per-component breakdown is unreliable** and cannot be cited in a paper. The relationship L4 > L3 (claimed) is actually reversed: L3 (~6.75G) > L4 (~4.3G) because L3 has 4× more blocks and 4× more tokens, which outweighs L4's wider channels.

**Systemic issue:** The FLOPs throughout the document appear to use an inconsistent convention (neither pure MACs nor strict 2×MACs). All FLOP estimates should be recalculated with a single consistent convention before implementation or publication.

**Recommendation:** Recalculate all backbone FLOPs using a consistent convention (recommend MACs = "paper FLOPs", as used by Swin, ConvNeXt, and all standard benchmarks). Update the backbone cost table and all downstream references.

---

#### Finding V06 [CONCERN] — L3 window_size=8 doesn't divide spatial height (45)

L3 operates at 80×45. The document specifies window_size=8 (line 178, 227).

- 80 ÷ 8 = 10 windows ✓
- 45 ÷ 8 = 5.625 ✗

Standard Swin implementations pad the feature map to the nearest multiple of window_size. This means 45 → 48 (adding 3 rows of zero-padding), creating 10×6 = 60 windows of 64 tokens each.

**Consequences:**
1. **6.7% wasted computation** (3,840 padded tokens vs 3,600 real tokens)
2. **Boundary artifacts** from shifted window attention on padded edges — the shift crosses real/padding boundaries, potentially corrupting features in the bottom ~3 rows
3. Standard Swin-T operates on spatial dimensions divisible by 7 (e.g., 56×56 from 224×224). The 80×45 grid is inherently awkward for any even window size.

**Why this happens:** 720 / 16 = 45 is odd. Any 2× factor window size produces a non-integer division.

---

#### Finding V16 [SUGGESTION] — Consider window_size=5 for L3

gcd(80, 45) = 5. With window_size=5:
- 80 ÷ 5 = 16 windows, 45 ÷ 5 = 9 windows → 144 windows × 25 tokens
- Zero padding needed ✓
- Clean shifted-window support (shift=2 divides both 80 and 45) ✓

**Trade-off:** Smaller window (25 vs 64 tokens per window) means less intra-window context, but more windows means better global coverage after shifted windows. The smaller window also makes each window's self-attention cheaper (25² vs 64² per head), potentially giving a net speedup.

**Alternative:** window_size=9 → 80/9 ≈ 8.9 (not clean), window_size=15 → 80/15 ≈ 5.3 (not clean). Only 5 works cleanly.

**Recommendation:** Add window_size ∈ {5, 7, 8} to ablation plan. The FLOPs and timing estimates should account for padding overhead if window_size=8 is retained.

---

### A3. Pre-compute KV and W_V projections

#### Finding V07 [CONCERN] — B2 shares K_t/V_t across both cross-attention layers

Line 294-299 defines a single $W_K$ and $W_V$ pair (148K params total) that produces $K_t, V_t$ once per frame. Line 440-444 shows both B2 layers using the same $K_t, V_t$.

**Standard practice comparison:**
- **DETR decoder:** Each cross-attention layer has its own K/V projections (separate W_K, W_V per layer). The source encoder output is the same, but each layer projects it differently.
- **Perceiver IO:** Shares KV across decoder layers (source features are fixed). Same rationale as EventSPD.
- **SAM:** Also shares KV across 2 decoder layers.

**Analysis:** Sharing KV means both layers attend to the same 192-dim projection of L4 features, differing only in the evolving query Q. This limits per-layer diversity — layer 1 might benefit from attending to broad spatial layout (needs one projection), while layer 2 might benefit from attending to fine-grained content (needs a different projection).

**Cost of separate projections:** +148K params (+1.7% of total), +0.003 ms precompute (negligible).

**Recommendation:** Keep shared as default (matches SAM), but add separate-KV as an ablation option. If depth accuracy improves by >0.5% AbsRel with separate KV, adopt it. Note: the param table in §6.1 correctly accounts for the shared projection (148K for A3 KV).

---

### A4. Global calibration heads

**[OK]** — Correct. MeanPool over 880 L4 tokens (at 384ch, post-self-attention) → $h_s, h_b$: R^384 → R^1. softplus ensures positive scale. Standard MiDaS/ZoeDepth pattern. The shift to 384ch input (from old 128ch) gives richer scene summary. ~0.8K params. ✓

---

## Phase B: Per-Query Sparse Depth Inference

### B1. Feature extraction and token construction

#### Finding V11 [AMBIGUITY] — Local MLP output dimension not stated

Line 365: $h_\delta = \text{GELU}(W_{\text{loc}} [f_\delta; \phi(\delta)] + b_{\text{loc}})$

Input dimension is stated (64 + 16 = 80), but the output dimension of $h_\delta$ is not. From the parameter table (line 868: "Local MLP (80→192)"), $h_\delta \in \mathbb{R}^{192}$. The same applies to the intermediate local MLP.

**Fix:** Add explicit output dimension: "where $h_\delta \in \mathbb{R}^d$ is the projected local feature."

---

#### Finding V10 [AMBIGUITY] — l_q serves dual role (h_point input + B4 context token)

$l_q$ appears in two places:
1. **h_point input** (line 423): MLP([f_q^(1); c_q^(2); pe_q; **l_q**; l_q^(int)]) → 608ch → 192
2. **B4 context token** (line 573): T_q = [**l_q + e_loc**; c_q^(2) + e_ms2; ...]

This means l_q's information enters the decoder through two paths:
- **Compressed path:** via h_point's 608→192 bottleneck, mixed with 5 other signals
- **Direct path:** as a raw 192-dim B4 context token

This is functionally a **skip connection** — the B4 cross-attention can directly access l_q's local gradient information without relying on h_point's compressed representation. This is good design (similar to DenseNet/U-Net skip connections), but the document never acknowledges this dual role or justifies why l_q gets a skip while l_q^(int) does not.

**Recommendation:** Add a brief note: "l_q appears both in h_point and as a separate B4 context token — a skip connection that lets the fusion decoder directly access local gradient information without h_point's 608→192 bottleneck."

Note: $l_q^{(\text{int})}$ does NOT get a skip to B4. Only l_q (the finest-resolution local) does. This asymmetry could be justified: L1 local captures the sharpest depth edges, which are most critical for the final prediction. Intermediate local provides supporting context that is adequately captured through h_point.

---

### B2. Global cross-attention into L4

#### Finding V01 [BUG] — B2 per-query cost "~1.0M FLOPs" is stale

Line 475: *"Per-query cost: ~1.0M FLOPs for 2-layer cross-attention"*

**But Section 6.2 (line 931) correctly states: B2 ~2,800K FLOPs.**

Verification with d=192, 6 heads, d_head=32, 880 tokens, 2 layers:
- Per layer: Q proj (73.7K) + QK^T (337.9K) + attn×V (337.9K) + O proj (73.7K) + FFN (589.8K) ≈ 1,413K FLOPs
- Two layers: **~2,826K ≈ 2.8M FLOPs** ✓ (matches Section 6.2)

The "~1.0M" figure appears to be from v8 when d=128 (4 heads). At d=128: per-layer ≈ 640K, two layers ≈ 1.28M ≈ "~1.0M". This was not updated when d increased to 192.

**Fix:** Change "~1.0M FLOPs" to "~2.8M FLOPs" in the B2 section text.

---

#### Finding V15 [SUGGESTION] — Add LayerNorm on B2's K_t/V_t

B4 applies $\text{LN}_{\text{kv}}(T_q)$ before using context tokens as KV (line 597). This normalizes heterogeneous token scales.

B2's $K_t$ and $V_t$ do NOT get normalized. They are raw linear projections of $G_t^{(4)}$ (line 296). The query gets $\text{LN}_q$ (line 440), creating a potential scale mismatch between normalized Q and un-normalized KV.

**Counter-argument:** B2's KV comes from a single homogeneous source ($G_t^{(4)}$ via shared $W_K$/$W_V$), unlike B4's heterogeneous context tokens from diverse paths. The scale is internally consistent.

**But:** L4 features after self-attention can have varying magnitudes across spatial positions (positions near scene boundaries vs uniform regions). Pre-normalizing K_t/V_t (once per frame, negligible cost) would stabilize attention logits.

**Recommendation:** Add $K_t \leftarrow \text{LN}_K(K_t)$, $V_t \leftarrow \text{LN}_V(V_t)$ as an optional improvement. Cost: +768 params, ~0 ms. Add to ablation plan.

---

### B3. Deformable multiscale read

#### Finding V08 [CONCERN] — B3 conditioning is shallow (single linear + LN)

Line 486: $u_r = \text{LN}(W_u [h_{\text{point}}';\; g_r;\; \phi_{\text{B3}}(\Delta\mathbf{p}_r)] + b_u)$

This is a single linear layer (416→192) + LayerNorm. It must integrate three qualitatively different signals:
1. $h_{\text{point}}'$ (192-dim): the query's globally-aware representation
2. $g_r$ (192-dim): the anchor's content from L4
3. $\phi_{\text{B3}}$ (32-dim): the spatial relationship between query and anchor

A single linear layer can only compute affine combinations of these inputs. It cannot model nonlinear interactions (e.g., "if the query is in a textured region AND the anchor is far away, sample more broadly"). LN adds normalization but not expressive nonlinearity.

**Comparison with Deformable DETR:** Deformable DETR's offset head is also a single linear from the query — but it doesn't have the 3-way conditioning problem. The query already carries all context.

**In EventSPD:** The conditioning explicitly merges three distinct information streams. A 2-layer MLP with GELU (416→192→192) would allow nonlinear cross-stream interaction at +37K params (+0.4% of total).

**Recommendation:** Add GELU between the linear and LN: $u_r = \text{LN}(\text{GELU}(W_{u1}[\ldots] + b_{u1}))$ or use a 2-layer MLP: $u_r = \text{LN}(W_{u2} \cdot \text{GELU}(W_{u1}[\ldots] + b_{u1}) + b_{u2})$. Add to ablation plan.

---

#### Finding V09 [CONCERN] — No per-level offset normalization

Line 524: $p_{\text{sample}} = \mathbf{p}_r^{(\ell)} + \Delta p_{r,h,\ell,m}$

Offsets $\Delta p$ are raw outputs of `nn.Linear(d, H*L*M*2)` reshaped into per-(h,ℓ,m) slices. All 144 output values come from one linear layer, but different level slices correspond to different spatial scales:
- L2 offset of 1.0 = 1 pixel at 160×90 = **8 original pixels**
- L3 offset of 1.0 = 1 pixel at 80×45 = **16 original pixels**
- L4 offset of 1.0 = 1 pixel at 40×22 = **32 original pixels**

The model must learn to output different magnitudes for different level slices from the same linear layer.

**Deformable DETR's approach:** Normalizes offsets by spatial dimensions: `sampling_locations = reference_points + offsets / spatial_shapes`. This makes offsets scale-invariant — a value of 1.0 means "one full spatial extent" regardless of level.

**EventSPD:** No normalization mentioned. The model can implicitly learn appropriate magnitudes, but explicit normalization would:
1. Make training easier (all offset slices have the same natural scale)
2. Enable weight sharing across levels (same offset magnitude = same physical displacement)
3. Follow the established Deformable DETR convention

**Recommendation:** Add per-level normalization:
$$p_{\text{sample}} = \mathbf{p}_r^{(\ell)} + \Delta p_{r,h,\ell,m} \cdot S_\ell$$
where $S_\ell$ is a per-level scaling factor (either fixed = spatial_shape, or learned). Document the choice explicitly and add to ablation.

---

### B4. Fusion decoder

**[OK with notes]** — B4 structure is sound. 2-layer cross-attention transformer over 36 static context tokens. Pre-LN, 6 heads, d_head=32, FFN 192→768→192. Standard design matching SAM and DETR decoder patterns.

$\text{LN}_{\text{kv}}$ on T_q (from v5 audit F14) is correctly included (line 597). ✓
Type embeddings (5 × d = 960 params) are appropriate. ✓
c_q^(3) at 192ch = d uses identity (no projection) — correct. ✓
Static T_q across layers — correct, standard practice. ✓

---

### B5. Depth prediction

**[OK]** — Depth head (192→192→1 with GELU) is well-specified. Calibration via (s_t, b_t) is standard. softplus + ε ensures positive inverse depth. Center auxiliary with shared calibration follows the v5 audit's F17 recommendation. ✓

---

## Section 5: Training

### 5.1 Loss Functions

**[OK]** — All three losses are correctly formulated:
- $L_{\text{point}}$: Huber on post-softplus inverse depth. $\hat{\rho}(q) = \text{softplus}(\rho_q) + \varepsilon$ is explicitly specified (v5 audit F18 fixed). ✓
- $L_{\text{silog}}$: sqrt() included (v5 audit F19 fixed). $\lambda_{\text{var}} = 0.5$ (v5 audit F20 fixed). ✓
- $L_{\text{ctr}}$: Calibrated by shared $(s_t, b_t)$ (v5 audit F17 fixed). ✓

---

### 5.3 Query Sampling

#### Finding V13 [AMBIGUITY] — Sampling categories overlap

Line 798-801:
- 40% LiDAR-valid pixels
- 20% DAv2 pseudo-labeled pixels (without LiDAR)
- 15% event-dense regions
- 25% high-gradient regions from depth maps

These categories are not mutually exclusive. A LiDAR-valid pixel can also be in a high-gradient region. The 100% total implies mutual exclusivity, but the descriptions don't enforce it.

**Two interpretations:**
(a) Multinomial sampling with priority: pick category first (40/20/15/25%), then sample a pixel from that category. Overlap doesn't matter.
(b) Mutually exclusive partition: each pixel belongs to exactly one category. Need explicit assignment rules.

**Recommendation:** Specify interpretation (a) or (b). If (a), note that some pixels may be sampled from multiple categories — this is fine for training. If (b), specify the priority order (e.g., LiDAR > event-dense > high-gradient > pseudo).

---

### 5.4 Training Schedule

#### Finding V12 [AMBIGUITY] — L_ctr delayed to Stage 2 without justification

Line 817-819: *"Stage 2 — Add center regularization: Enable L_ctr. Continue training (~10 epochs)."*

Stage 1 trains without $L_{\text{ctr}}$. This means the center token ($h_{\text{point}}$) has no independent depth supervision during the first ~15-20 epochs. It only learns through the full pipeline's $L_{\text{point}}$ and $L_{\text{silog}}$.

**Why this might be intentional:** Allow the full pipeline to stabilize before adding the auxiliary constraint. Adding $L_{\text{ctr}}$ too early could conflict with the main loss before the deformable sampling and cross-attention converge.

**Counter-argument:** PSPNet and DeepLabV3 enable auxiliary losses from the start. The auxiliary loss doesn't conflict — it provides additional gradient signal to shared layers.

**Recommendation:** Either (a) justify the delay: "Stage 1 establishes the decoder pipeline; $L_{\text{ctr}}$ is delayed to avoid interfering with cross-attention convergence" or (b) enable $L_{\text{ctr}}$ from Stage 1 with a warm-up coefficient: $\lambda_{\text{ctr}} = 0 \to 0.25$ over the first 5 epochs.

---

## Cross-Section Consistency Checks

### Finding V02 [BUG] — Section 4.7 latency is stale

Line 725 (§4.7 comparison table): *"Latency | ~23 ms (F^3 8.3ms + DAv2 ~15ms) | **~7.8 ms** at K=256"*

But Section 6.2 (line 954): $T(K=256) = 6.1 + 0.005 \times 256 = 7.38 \approx$ **7.4 ms**.

The ~7.8 ms figure is from the narrow pyramid era (T ≈ 6.9 + 0.006×256 ≈ 8.4... actually 7.8 doesn't match that either). Regardless, the correct v9 number is **7.4 ms**.

**Fix:** Update line 725 from "~7.8 ms" to "~7.4 ms".

---

### Finding V03 [BUG] — Section 6.1 DAv2 comparison uses stale numbers

Line 890:
> *"vs DAv2-S: Our decoder is **12× smaller** (2,163K vs 25M). Total params **4.7× smaller** (5.3M vs 25M). 2,356 lookups/query = 391× less than dense (921,600 pixels)."*

**Issues:**
1. **"2,163K"** → current decoder is **~2,206K** (line 878). Minor.
2. **"5.3M"** → current total is **~8,626K = ~8.6M** (line 886). This is the narrow-pyramid total.
3. **"4.7× smaller"** → should be 25M / 8.6M = **2.9×**.
4. **"12× smaller"** compares our DECODER (2.2M) to DAv2's TOTAL (25M) — apples-to-oranges. DAv2's DPT decoder at features=64 is ~1-3M params (not 25M). Fair comparison: decoder vs decoder = 2.2M vs ~3M = **1.4×**.
5. **"921,600 pixels"** → this is 1280×720, but F^3 ds2 operates at 640×360 = 230,400 pixels. The dense decoder processes at the encoder's output resolution. For F^3+DAv2: the DPT decoder upsamples to 518×518 = 268,324 pixels.

**Fix:** Recalculate all comparison numbers with current v9 architecture:
- Decoder: ~2.2M vs ~3M (DPT-S) = 1.4× smaller
- Total: ~8.6M (ours, excl. frozen F^3) vs ~25M (DAv2-S) = 2.9× smaller
- Lookups: 2,356 per query vs 268K pixels (DPT output resolution)

---

### Finding V04 [BUG] — RGB-SPD nonlinear stage count is wrong

Line 1376: *"Nonlinear depth per query: L1(3) + Int(6) + Stem-2(1) + L2(4) + L3(8) + **L4(4)** + decoder(14) = **~40 stages**"*

RGB-SPD has **4** self-attention layers at L4 (line 1334), each with 2 sub-layers (self-attn + FFN). Using the same counting convention as EventSPD (line 632: "L4 2× SelfAttn (2×2=4)"), RGB-SPD's L4 should be **4×2 = 8** sub-layers, not 4.

Corrected count: 3 + 6 + 1 + 4 + 8 + **8** + 14 = **44 stages** (not 40).

**Also:** The "Stem-2(1)" count includes LN but not GELU. If LN is counted as a nonlinear stage (which it is elsewhere), then Stem-1 should also be counted. The counting convention is inconsistent. The ~31 for EventSPD and ~44 for RGB-SPD are approximate anyway, but the L4 error is clear.

**Fix:** Change "L4(4)" to "L4(8)" and total to "~44 stages".

---

### Finding V14 [AMBIGUITY] — Backbone timing text doesn't match table

The backbone cost table (lines 267-278) gives per-component timings:
- L2 ConvNeXt: **~0.50 ms**
- L3 Swin: **~0.50 ms**

But line 917 says:
- L2 ConvNeXt at 160×90: **~0.3 ms**
- L3 Swin at 80×45: **~0.35 ms**
- **rest ~0.3 ms** (unexplained)

The "rest ~0.3 ms" accounts for the difference between text and table values (0.50-0.30 + 0.50-0.35 = 0.35 ms ≈ "rest ~0.3 ms"). But this redistribution is confusing and the "rest" is not attributed to any specific component.

**Recommendation:** Make line 917 match the table exactly, or explain what "rest" includes (kernel launch overhead? memory allocation? PyTorch dispatch?).

---

## Dimension Flow Verification (v9)

| Transition | In | Out | Verified? |
|---|---|---|---|
| $F_t$ → $F_t^{(1)}$ (LN+GELU) | 640×360×64 | 640×360×64 | ✓ identity |
| $F_t$ → $F_t^{(\text{int})}$ (Stem-1) | 640×360×64 | 320×180×64 | ✓ |
| $F_t^{(\text{int})}$ → Stem-2 | 320×180×64 | 160×90×128 | ✓ |
| Stem-2 → L2 ConvNeXt | 160×90×128 | 160×90×128 | ✓ |
| L2 → Down L3 | 160×90×128 | 80×45×192 | ✓ |
| Down → L3 Swin | 80×45×192 | 80×45×192 | ✓ |
| L3 → Down L4 | 80×45×192 | 40×22×384 | ✓ (45→22, see note) |
| Down → L4 SelfAttn | 40×22×384 | 40×22×384 | ✓ |
| $G_t^{(4)}$ → $K_t, V_t$ | 880×384 | 880×192 | ✓ |
| $[f_q^{(1)}; c_q^{(2)}; pe; l_q; l_q^{(\text{int})}]$ → $h_{\text{point}}$ | 608 | 192 | ✓ |
| $h_{\text{point}} + \text{B2}$ → $h_{\text{point}}'$ | 192 | 192 | ✓ |
| $[h_{\text{point}}'; g_r; \phi_{\text{B3}}]$ → $u_r$ | 416 | 192 | ✓ |
| $u_r$ → offsets | 192 | 144 | ✓ (6×3×4×2) |
| $u_r$ → weights | 192 | 72 | ✓ (6×3×4) |
| $T_q$ → B4 context | 36×192 | 36×192 | ✓ |
| $h_{\text{fuse}}$ → $r_q$ | 192 | 1 | ✓ (via 192→192→1) |

**Note on 45→22:** Conv(192→384, k2, s2) on 80×45 input: H_out = (45-2)/2 + 1 = 22. The bottom row (at position 44) is not fully covered. This is a 1-row loss at stride 16 = 16 original pixels out of 720 (~2.2%). Standard behavior with odd spatial dimensions.

---

## Feature Lookup Consistency (v9)

| Source | Count | Verified? |
|---|---|---|
| Center features (L1 + L2 + L3 + L4) | 4 | ✓ |
| L1 local samples | 32 | ✓ (24 fixed + 8 learned) |
| Stride-4 intermediate local | 16 | ✓ (8 fixed + 8 learned) |
| Deformable reads (B3) | 2,304 | ✓ (32 anchors × 6 heads × 3 levels × 4 samples) |
| **Total hard lookups** | **2,356** | ✓ |
| B2 soft attention over L4 | 880 | ✓ (100% L4 coverage) |

---

## Parameter Table Consistency (v9)

| Component | Claimed | Computed | Match? |
|---|---|---|---|
| Stem-1 Conv(64→64,k3s2)+LN | ~37K | 64×64×9 + 64 + 128 = 37,056 | ✓ |
| Stem-2 Conv(64→128,k2s2)+LN | ~33K | 64×128×4 + 128 + 256 = 33,024 | ✓ |
| L2 ConvNeXt (per block) | ~138.5K | DW(6.3K) + PW×2(131K) + GRN(1.0K) + LN(256) | ✓ |
| L3 Swin (per block) | ~444K | QKV(110.6K) + O(36.9K) + FFN(295K) + RPB(1.4K) + LN(0.8K) | ✓ |
| L4 SelfAttn (per layer) | ~1,772K | QKV(442K) + O(148K) + FFN(1,180K) + LN(2K) | ✓ |
| B1 h_point MLP | ~154K | 608×192 + 192 + 192×192 + 192 = 154,176 | ✓ |
| B2 (2 layers, no KV) | ~740K | 2 × (Q:36.9K + O:36.9K + FFN:295K + LN:1.5K) = 740.6K | ✓ |
| B3 Deformable | ~159K | Cond(80K) + Offset(28K) + Weight(14K) + W_O(37K) = 159K | ✓ |
| B4 Fusion (2 layers) | ~889K | 2 × (Q:36.9K + K:36.9K + V:36.9K + O:36.9K + FFN:295K + LN) ≈ 889K | ✓ |
| B5 Depth head | ~37K | 192×192 + 192 + 192×1 + 1 = 37,057 | ✓ |
| **Phase A total** | **~6,420K** | Sum of above | ✓ |
| **Phase B total** | **~2,206K** | Sum of above | ✓ |
| **Grand total** | **~8,626K** | | ✓ |

All parameter counts are internally consistent. ✓

---

## v5 Audit Findings: Resolution Status in v9

| v5 Finding | Severity | Status in v9 |
|---|---|---|
| F01: Symbol table inconsistency | AMBIGUITY | ✅ Fixed — symbol table matches formula (LN+GELU, not AvgPool) |
| F02: Redundant 128→128 projection | CONCERN | ✅ Eliminated — no latent bank in v9 |
| F03: Missing coordinate normalization | AMBIGUITY | ✅ Fixed — explicit `Normalize()` wrapper (line 326-327) |
| F04: Fourier encoding frequency mismatch | CONCERN | ✅ Fixed — B3 uses normalized encoding (line 493-498) |
| F05: Aggressive 544→128 bottleneck | CONCERN | ⚠️ Partially addressed — now 608→192 (ratio 3.17 vs 4.25), but still significant compression in one step |
| F06: Routing from level-1 only | CONCERN | ✅ Eliminated — v9 uses attention-based routing from B2 (conditioned on full h_point') |
| F07: Symbol overloading (c_r vs **c**_r) | AMBIGUITY | ✅ Fixed — uses p_r for position, g_r for content |
| F08: False tanh attribution to Deformable DETR | BUG | ✅ Fixed — v9 uses unbounded offsets (line 514), cites DAT/Deformable DETR correctly |
| F09: False reflective padding claim | BUG | ✅ Fixed — uses `GridSample_zeros` throughout (line 527) |
| F10: c_r^(ℓ) undefined | AMBIGUITY | ✅ Fixed — explicit p_r^(ℓ) = p_r / s_ℓ definition (line 530) |
| F11: No true multi-head deformable | CONCERN | ✅ Fixed — full multi-head W_V/W_O structure (line 534-552) |
| F12: Joint softmax across heads | CONCERN | ✅ Fixed — per-head softmax (line 543-546) |
| F13: Heterogeneous T_q normalization | CONCERN | ✅ Fixed — LN_kv on assembled T_q (line 597) |
| F14: Add LN on KV tokens | SUGGESTION | ✅ Adopted for B4 (line 597). Not applied to B2 (see V15). |
| F15: Depth head hidden dim unspecified | AMBIGUITY | ✅ Fixed — explicitly 192→192→1 (line 641) |
| F16: "Relative disparity" terminology | AMBIGUITY | ✅ Fixed — "Relative depth code" (line 636) |
| F17: Auxiliary calibration unspecified | AMBIGUITY | ✅ Fixed — calibrated by shared (s_t, b_t) (line 650) |
| F18: L_point ρ̂(q) undefined | AMBIGUITY | ✅ Fixed — explicitly $\hat{\rho}(q) = \text{softplus}(\rho_q) + \varepsilon$ (line 749) |
| F19: SiLog missing sqrt() | SUGGESTION | ✅ Fixed — sqrt included (line 756) |
| F20: λ_var = 0.85 vs F^3's 0.5 | SUGGESTION | ✅ Fixed — λ_var = 0.5 default, ablation plan included |

**19 of 20 v5 findings resolved. 1 partially addressed (F05: bottleneck ratio improved but still present).**

---

## Priority Action Items

### Must Fix (BUGs)

1. **V01:** Update B2 per-query cost from "~1.0M FLOPs" to "~2.8M FLOPs" (§4.4 line 475).
2. **V02:** Update §4.7 latency from "~7.8 ms" to "~7.4 ms" (line 725).
3. **V03:** Recalculate §6.1 DAv2 comparison with v9 numbers (8.6M total, 2.2M decoder).
4. **V04:** Fix RGB-SPD nonlinear count: L4(4)→L4(8), total ~40→~44 (§12 line 1376).
5. **V05:** Recalculate all backbone FLOPs with consistent convention; update table.

### Should Address (CONCERNs)

6. **V06/V16:** Acknowledge L3 window padding or change to window_size=5. Add to ablation.
7. **V07:** Document that B2 KV is shared across layers (intentional, matching SAM). Add separate-KV to ablation.
8. **V08:** Consider 2-layer MLP for B3 conditioning (+37K params). Add to ablation.
9. **V09:** Add per-level offset normalization or document why it's omitted.

### Should Clarify (AMBIGUITIEs)

10. **V10:** Document l_q's dual role (h_point input + B4 skip connection).
11. **V11:** State local MLP output dimension ($h_\delta \in \mathbb{R}^d$).
12. **V12:** Justify L_ctr delay or enable from Stage 1 with warm-up.
13. **V13:** Specify query sampling overlap behavior.
14. **V14:** Reconcile backbone timing text with table.
