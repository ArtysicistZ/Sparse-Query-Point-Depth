# Research Plan: EventSPD — Sparse Query-Point Depth from Events

Author: Claude
Date: 2026-02-16
Version: v10 (clean rewrite of v9, all audit fixes applied)

---

## 1. Problem Statement

**Input:**
- Raw events in a temporal window (e.g., 20 ms).
- A set of user-specified query pixel coordinates: $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:**
- Depth $\hat{d}_i$ only at each queried pixel.

**Hard constraint:**
- No dense $H \times W$ depth decoding at inference. One shared precompute, then sparse per-query decoding.

**Runtime decomposition:**
$$
T_{\text{total}}(K) = T_{\text{precompute}}(E_t) + T_{\text{query}}(K \mid \text{cache}_t)
$$

where $T_{\text{query}}$ scales with query count $K$, not with image resolution $HW$.

---

## 2. Why This Is Publishable

Every existing event-depth method produces dense $H \times W$ output. No published work targets sparse query-point depth from events as a first-class objective. The closest non-event work is InfiniDepth (Jan 2026), which applies LIIF-style continuous-coordinate depth queries to RGB — but without attention-guided spatial routing, deformable multi-scale sampling, or event-camera support.

The contribution is clear:
- Many downstream tasks (SLAM, grasping, obstacle avoidance) use sparse keypoints, not dense maps.
- Query budgets are often tiny ($K \in [1, 256]$).
- Latency and power matter on edge devices.
- If EventSPD achieves comparable accuracy to dense methods while being measurably faster for $K \ll HW$, this is a strong result.

---

## 3. Baseline Systems

### 3.1 Baseline A — Dense Best: F^3 + DepthAnythingV2

Existing pipeline:
- `src/f3/tasks/depth/utils/models/depth_anything_v2.py`
- Stage-1: `src/f3/tasks/depth/train_monocular_rel.py`
- Stage-2: `src/f3/tasks/depth/finetune.py`

Measure: full dense map latency, query extraction latency (dense map then sample $K$ points), query-point accuracy.

### 3.2 Baseline B — Naive Sparse: Dense Output + Bilinear Sample

Run Baseline A, then `Bilinear(dense_depth, q)` at each query. Establishes accuracy ceiling and speed floor.

### 3.3 Baseline C — Minimal Query Decoder

Bilinear lookup from F^3 features + small MLP depth head. No global context, no deformable sampling. Establishes "how far can pure local features go" baseline.

---

## 4. Proposed Algorithm: EventSPD

### 4.1 Design Principles

1. **Precompute once, decode per query** (SAM, Perceiver IO). F^3 features and globally-enriched L4 tokens are computed once. Each query runs a deep but lightweight decoder.

2. **Two-phase per-query reasoning: global then local** (novel for sparse depth). Phase 1: cross-attention into all 880 L4 tokens reveals scene layout and identifies important regions. Phase 2: deformable sampling gathers fine-grained multi-scale evidence at the attended locations.

3. **Attention-guided deformable sampling** (combining Deformable DETR + cross-attention). L4 cross-attention weights naturally identify task-relevant spatial regions (free routing). Deformable sampling at those regions provides multi-scale evidence from L2–L4.

4. **Deep per-query processing to compensate for shallow backbone** (sparse advantage). DAv2 processes every pixel through 12 ViT layers (~24 sub-layers). Our backbone is shallower (~17 stages to L4), but our per-query decoder adds ~14 sub-layers — affordable because we process only K points, not all pixels. Total depth per query (~31 stages) exceeds DAv2's encoder.

5. **Minimal viable complexity**. Every component must justify its existence via ablation.

### 4.2 Symbol Table

| Symbol | Meaning | Shape / Value |
|--------|---------|---------------|
| $E_t$ | Input event stream in $[t-\Delta, t)$ | — |
| $\mathcal{F}_{\text{F}^3}^{\text{ds2}}$ | Event-to-feature encoder (ds2 config) | Frozen / fine-tuned |
| $F_t$ | Dense shared feature field | 640×360×64 |
| $F_t^{(1)}$ | L1 fine features: $\text{GELU}(\text{LN}(F_t))$ | 640×360×64, stride 2 |
| $F_t^{(\text{int})}$ | Stride-4 intermediate (Stem-1 output) | 320×180×64, stride 4 |
| $F_t^{(2)}$ | L2 features (2× ConvNeXt₁₂₈) | 160×90×128, stride 8 |
| $F_t^{(3)}$ | L3 features (4× SwinBlock₁₉₂) | 80×45×192, stride 16 |
| $G_t^{(4, \text{pre})}$ | L4 pre-GRU features (2× FullSelfAttn₃₈₄) | 40×22×384, stride 32 |
| $G_t^{(4)}$ | L4 features (after ConvGRU temporal fusion) | 40×22×384, stride 32 |
| $h_t$ | ConvGRU hidden state at L4 (carried across windows) | 40×22×384 |
| $K_t^{(1:2)}, V_t^{(1:2)}$ | Pre-computed per-layer KV for B2 cross-attention | 2×880×192 |
| $\hat{F}_t^{(2:4)}$ | Pre-projected feature maps for B3 ($W_V$ applied) | ×192 per level |
| $s_t, b_t$ | Main depth scale / shift | scalars |
| $s_t^{\text{ctr}}, b_t^{\text{ctr}}$ | Auxiliary depth scale / shift (training only, separate from main) | scalars |
| $q = (u, v)$ | Query pixel coordinate | — |
| $f_q^{(1)}$ | Fine point feature: $\text{Bilinear}(F_t^{(1)}, q/s_1)$ | 64 |
| $c_q^{(\ell)}$ | Multi-scale center: $\text{Bilinear}(F_t^{(\ell)}, q/s_\ell)$ | 128 / 192 / 384 |
| $l_q$ | L1 local context (32 samples, max-pooled) | $d = 192$ |
| $l_q^{(\text{int})}$ | Stride-4 local context (16 samples, max-pooled) | $d = 192$ |
| $\text{pe}_q$ | Fourier positional encoding (8 freq × 2 trig × 2 dims) | 32 |
| $\mathcal{T}_{\text{ms}}$ | Multi-scale local tokens (9 L2 + 9 L3 + 9 L4 from 3×3 grids) | $27 \times d$ |
| $h_{\text{ms}}^{(0)}$ | MSLCA query seed: $W_q([f_q^{(1)}; \text{pe}_q])$ | $d = 192$ |
| $h_{\text{ms}}^{(\ell)}$ | MSLCA hidden state after layer $\ell$; $h_{\text{ms}} \equiv h_{\text{ms}}^{(2)}$ | $d = 192$ |
| $h_{\text{point}}$ | Center token: $h_{\text{ms}}$ + MLP$([h_{\text{ms}}; l_q; l_q^{(\text{int})}])$ | $d = 192$ |
| $h_{\text{point}}'$ | Globally-aware center token (after B2 cross-attn) | $d = 192$ |
| $\bar{\alpha}_q$ | Head-averaged B2 attention weights | 880 |
| $R_q$ | Top-$R$ attention-routed anchor set ($R = 32$) | 32 positions |
| $h_r$ | Per-anchor deformable evidence (B3 output) | $d = 192$ |
| $T_q$ | Context tokens: $[l_q{+}e_{\text{loc}}; c_q^{(2)}{+}e_{\text{ms2}}; c_q^{(3)}{+}e_{\text{ms3}}; c_q^{(4)}{+}e_{\text{ms4}}; h_{r_{1..32}}{+}e_{\text{deform}}]$ | 36×192 |
| $h_{\text{fuse}}$ | Fused query representation (B4 output) | $d = 192$ |
| $r_q$ | Relative depth code: MLP$(h_{\text{fuse}})$ | scalar |
| $\hat{d}_q$ | Final depth: $1 / (\text{softplus}(s_t r_q + b_t) + \varepsilon)$ | scalar |
| $\hat{d}_q^{\text{ctr}}$ | Auxiliary depth: $1 / (\text{softplus}(s_t^{\text{ctr}} r_q^{\text{ctr}} + b_t^{\text{ctr}}) + \varepsilon)$ | scalar (training only) |
| $\hat{\rho}_{\text{dense}}^{(\ell)}$ | Dense backbone auxiliary prediction at level $\ell$ (training only) | stride-$s_\ell$ maps |
| $s_t^{\text{d}\ell}, b_t^{\text{d}\ell}$ | Dense auxiliary calibration per level (training only) | scalars |
| $\mathcal{M}(u,v)$ | Spatial loss map for hard-example mining (stride 16, EMA-updated) | 80×45 |

Core dimension $d = 192$. Strides: [2, 8, 16, 32]. Channels: [64, 128, 192, 384] (ratio 1:2:3:6).

**Dual roles (skip connections):** $l_q$ appears both in $h_{\text{point}}$ (compressed via 576→192 MLP) and as a separate B4 context token ($l_q + e_{\text{loc}}$). This is an intentional skip connection — B4 cross-attention can directly access local gradient information without relying on $h_{\text{point}}$'s bottleneck. $l_q^{(\text{int})}$ enters only through $h_{\text{point}}$ (no skip) because L1 local captures the sharpest depth edges most critical for final prediction. Similarly, $c_q^{(2)}$, $c_q^{(3)}$, and $c_q^{(4)}$ — the center tokens of the MSLCA 3×3 grids at L2/L3/L4 — serve dual roles: absorbed into $h_{\text{ms}}$ via MSLCA cross-attention (compressed) AND passed as separate B4 context tokens (direct, uncompressed). This gives B4 both the MSLCA-processed multi-scale context (via $h_{\text{point}}$) and raw per-level features for independent reasoning.

### 4.3 Algorithm A: Precompute Once Per Event Window

**Input:** Event set $E_t$.
**Output:** $\text{cache}_t = \{F_t^{(1)}, F_t^{(\text{int})}, F_t^{(2)}, F_t^{(3)}, G_t^{(4)}, h_t, K_t^{(1:2)}, V_t^{(1:2)}, \hat{F}_t^{(2:4)}, s_t, b_t\}$

#### A1. F^3 ds2 backbone encoding

$$
F_t = \mathcal{F}_{\text{F}^3}^{\text{ds2}}(E_t) \quad \in \mathbb{R}^{640 \times 360 \times 64}
$$

F^3 ds2 configuration:
```yaml
dsstrides: [2, 1]      # Stride-2 in first stage → 640×360 output
dims: [64, 64]          # 2× wider channels (ds1 used [32, 32])
convdepths: [3, 3]      # 6 ConvNeXt blocks total
patch_size: 2
```

**Why ds2 over ds1:** ds1 processes 6 ConvNeXt blocks at full 1280×720 (~8.3 ms). ds2 applies stride-2 in the first stage, processing 5 of 6 blocks at 640×360. Wider channels (64 vs 32) compensate — each pixel carries 2× more information. Wall-clock drops from ~8.3 ms to ~4.5 ms (−46%), driven by memory bandwidth, not FLOPs: ds2 GFLOPs drop only 12% (57G vs 65G), but memory traffic halves and ds2's 59 MB feature maps fit the RTX 4090's L2 cache (72 MB) while ds1's 118 MB do not. Requires pre-training with the ds2 config.

#### A2. Wide pyramid backbone (Conv + Swin + SelfAttn)

```
F_t (640×360, 64ch)                                         ← F^3 ds2 output
├→ L1: LN + GELU                                            [640×360×64]   stride 2
└→ Stem-1: Conv(64→64, k3,s2) + LN                          [320×180×64]   stride 4 (INTERMEDIATE)
   → Stem-2: Conv(64→128, k2,s2) + LN                       [160×90×128]   stride 8
   → 3× ConvNeXt_128 (with GRN)                             → L2  [160×90×128]
   → Down: Conv(128→192, k2,s2) + LN                        [80×45×192]
   → 4× SwinBlock_192 (window=8, shifted)                   → L3  [80×45×192]
   → Down: Conv(192→384, k2,s2) + LN                        [40×22×384]
   → 2× FullSelfAttn_384 (6 heads, d_head=64, 880 tokens)  [40×22×384]
   → ConvGRU_384 (k=3, hidden=384)                          → L4 = G_t^{(4)}  [40×22×384]
```

**L1** — Fine features (identity from F^3 ds2, with nonlinearity):
$$F_t^{(1)} = \text{GELU}(\text{LN}(F_t)) \quad \in \mathbb{R}^{640 \times 360 \times 64}$$
No convolution needed — F^3 ds2 already outputs at 640×360×64 after 6 ConvNeXt blocks with ~37×37 receptive field. GELU enables nonlinear feature conjunctions. Params: ~0.1K (LN only).

**Two-step stem** (stride 2→4→8):
$$F_t^{(\text{int})} = \text{LN}(\text{Conv2d}(64, 64, k{=}3, s{=}2, p{=}1)(F_t)) \quad \in \mathbb{R}^{320 \times 180 \times 64}$$
$$\text{Stem-2} = \text{LN}(\text{Conv2d}(64, 128, k{=}2, s{=}2)(F_t^{(\text{int})})) \quad \in \mathbb{R}^{160 \times 90 \times 128}$$
$F_t^{(\text{int})}$ is retained for B1 local sampling — zero additional cost. Params: Stem-1 ~37K, Stem-2 ~33K.

**L2** — 3× ConvNeXt₁₂₈ blocks (ConvNeXt V2 with GRN, k=13):
```
Input → DW Conv 13×13 → LN → PW Conv 1×1 (C→4C) → GELU → GRN(4C) → PW Conv 1×1 (4C→C) → + Input
```
~145K/block, ~435K total. 13×13 depthwise convolutions at stride 8 cover ~104×104 original-pixel context per block (vs ~56×56 for 7×7), capturing surface-level depth patterns without attention overhead. DW conv cost scales with kernel area but is a small fraction of total block cost (pointwise 1×1 convolutions dominate); net per-block increase is only ~5% vs k=7. Ref: UniRepLKNet (Ding et al. CVPR 2024) showed large-kernel ConvNets match or exceed attention-based models. F^3 output is event-prediction features — not depth features — so L2 must extract depth-relevant mid-scale representations from scratch. 128ch gives the downstream $W_V$ projection 67% rank in $d{=}192$ space.

**L3** — 4× SwinBlock₁₉₂ (windowed self-attention, 6 heads, $d_{\text{head}} = 32$):
```
Input → LN → W-MSA (window_size=8) → + Input → LN → FFN (192→768→192) → + Input
```
Alternating regular/shifted windows (shift=4). ~444K/block, ~1,778K total. At 80×45 = 3,600 positions, shifted window attention propagates information ~16 positions after 4 blocks.

**Note on L3 window padding:** 45 is not divisible by 8, so the implementation pads 45→48 (3 zero rows), creating 10×6 = 60 windows of 64 tokens. This adds ~6.7% wasted computation and potential boundary artifacts from shifted attention crossing real/padding boundaries. Consider ablating window_size=5 (divides both 80 and 45 cleanly: 16×9 = 144 windows of 25 tokens).

**L4** — 2× FullSelfAttn₃₈₄ (6 heads, $d_{\text{head}} = 64$, 880 tokens):
```
Input → LN → MultiHeadSelfAttn → + Input → LN → FFN (384→1536→384) → + Input
```
At 40×22 = 880 positions, full self-attention is trivially cheap — no windowing needed. Every L4 token has direct access to all 880 tokens for true global scene understanding. ~1,772K/layer, ~3,542K total.

**ConvGRU₃₈₄** — Temporal state at L4 bottleneck:

$$z_t = \sigma(W_z * [G_t^{(4, \text{pre})}, h_{t-1}] + b_z)$$
$$r_t = \sigma(W_r * [G_t^{(4, \text{pre})}, h_{t-1}] + b_r)$$
$$\tilde{h}_t = \tanh(W_h * [G_t^{(4, \text{pre})}, r_t \odot h_{t-1}] + b_h)$$
$$G_t^{(4)} = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

where $G_t^{(4, \text{pre})}$ is the L4 output from self-attention (before GRU), $h_{t-1}$ is the hidden state from the previous window, and $*$ denotes 3×3 depthwise-separable convolution. The GRU integrates temporal evidence across windows — motion velocity, scene persistence, and depth stability — without requiring explicit pose. $h_0 = 0$ (zero initialization at sequence start).

Params: 3 gates × (DW Conv 3×3×384 + PW Conv 384×384×2) ≈ ~886K. Cost: ~0.8G MACs, ~0.1 ms (40×22 spatial, fully parallelizable). The ConvGRU operates at L4's compact spatial resolution (880 tokens, stride 32) — larger levels (L1–L3) remain stateless and fully parallelizable per window.

**Why ConvGRU at L4 only:** Event streams are continuous; independent window processing discards inter-window state (velocity, acceleration, scene persistence). All 2025 SOTA event-camera depth methods use recurrence: Depth AnyEvent-R (ICCV 2025) adds ConvLSTM at multiple scales; DERD-Net (NeurIPS 2025 Spotlight) uses GRU with only 70K params. Placing the GRU at L4 minimizes cost while maximizing impact — L4 tokens have global receptive fields (each sees all 880 tokens through self-attention), so the temporal state captures scene-level dynamics, not local jitter. L1–L3 features change per-window through the event content; L4 features change AND accumulate through the GRU.

**Note on 45→22:** Conv(192→384, k2, s2) on 80×45: $H_{\text{out}} = (45-2)/2 + 1 = 22$. Bottom row (position 44) is not fully covered — 1-row loss at stride 16 = 16 original pixels out of 720 (~2.2%). Standard behavior.

**No FPN/BiFPN:** Deformable DETR Table 2 shows identical AP with and without FPN when multi-scale deformable attention is used. Our B3 deformable read + B4 cross-attention provide equivalent cross-level exchange.

**Backbone cost (all FLOPs in MACs convention):**

| Component | Resolution | Channels | MACs | Time (est.) |
|-----------|:---:|:---:|---:|---:|
| L1: LN + GELU | 640×360 | 64 | ~0.03G | ~0.01 ms |
| Stem-1: Conv k3s2 + LN | 320×180 | 64→64 | ~2.12G | ~0.06 ms |
| Stem-2: Conv k2s2 + LN | 160×90 | 64→128 | ~0.47G | ~0.02 ms |
| L2: 3× ConvNeXt₁₂₈ (GRN, k=13) | 160×90 | 128 | ~6.60G | ~0.55 ms |
| Down L2→L3 | 80×45 | 128→192 | ~0.35G | ~0.02 ms |
| L3: 4× SwinBlock₁₉₂ | 80×45 | 192 | ~6.75G | ~0.50 ms |
| Down L3→L4 | 40×22 | 192→384 | ~0.26G | ~0.02 ms |
| L4: 2× FullSelfAttn₃₈₄ | 40×22 | 384 | ~4.30G | ~0.08 ms |
| L4: ConvGRU₃₈₄ (k=3) | 40×22 | 384 | ~0.8G | ~0.1 ms |
| **Backbone total** | | | **~21.7G** | **~1.35 ms** |

**FLOPs convention:** All numbers in this document use MACs (multiply-accumulate operations), the standard "paper FLOPs" convention used by Swin, ConvNeXt, and all standard benchmarks. 1 MAC = 1 multiply + 1 add.

Total backbone params: ~7,007K (6,121K CNN/Swin/SelfAttn + 886K ConvGRU).

#### A3. Pre-compute KV and $W_V$ projections

**KV for B2 cross-attention:** Each layer $\ell = 1, 2$ has its own $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times 384}$, projecting $G_t^{(4)}$ once per frame:
$$K_t^{(\ell)} = G_t^{(4)} (W_K^{(\ell)})^T, \quad V_t^{(\ell)} = G_t^{(4)} (W_V^{(\ell)})^T \quad \in \mathbb{R}^{880 \times d}, \quad \ell = 1, 2$$
Params: $4 \times 384 \times 192 = 296\text{K}$. Cost: ~0.02 ms. Per-layer KV projections follow the standard transformer decoder convention (DETR, SAM, Perceiver IO all use separate KV per layer on a shared source). Each layer can project L4 features into a different key-value space — layer 1 keys for coarse region finding, layer 2 keys for refined selection. Ablation: shared KV across layers (−148K params, −0.01 ms).

**$W_g$ anchor projection for B3:** $W_g \in \mathbb{R}^{d \times 384}$ projects L4 features for B3 conditioning. Pre-applied to all 880 positions. Params: ~74K.

**Per-level $W_V$ for B3 deformable reads:** Following Deformable DETR, value projections are pre-applied: L2 (128→192), L3 (192→192, square), L4 (384→192). Grid-sample reads already-projected 192ch features during per-query decoding. Params: ~136K. Cost: ~0.05 ms.

#### A4. Global calibration heads

**Main calibration:**
$$s_t = \text{softplus}(h_s(\text{MeanPool}(G_t^{(4)}))), \quad b_t = h_b(\text{MeanPool}(G_t^{(4)}))$$

**Auxiliary calibration (separate, training only):**
$$s_t^{\text{ctr}} = \text{softplus}(h_s^{\text{ctr}}(\text{MeanPool}(G_t^{(4)}))), \quad b_t^{\text{ctr}} = h_b^{\text{ctr}}(\text{MeanPool}(G_t^{(4)}))$$

Each pair is two independent linear layers $\mathbb{R}^{384} \to \mathbb{R}^1$. Main: ~0.8K params. Auxiliary: ~0.8K params (discarded at inference). softplus ensures positive scale. Separate calibration follows PSPNet / DeepLabV3 convention: auxiliary heads always use independent prediction parameters (Zhao et al. CVPR 2017; torchvision DeepLabV3). Sharing $(s_t, b_t)$ would create conflicting gradients — the main branch (after B2→B3→B4) and auxiliary branch (before B2) produce $r_q$ at different scales.

#### A5. Dense backbone auxiliary heads (training only)

**Motivation:** The sparse decoder provides gradients from only $K$ query points per batch (~0.1% spatial coverage at $K{=}256$). Backbone features at the other 99.9% of locations receive zero direct supervision. Dense auxiliary heads solve this by providing orders-of-magnitude more gradient signal through the backbone during training, discarded at inference. Ref: PSPNet (Zhao et al. CVPR 2017), GoogLeNet (Szegedy et al. CVPR 2015), Mask2Former (Cheng et al. CVPR 2022).

**L2 dense head (stride 8):**
$$\hat{\rho}_{\text{dense}}^{(2)}(p) = \text{Conv}_{1 \times 1}(F_t^{(2)}, p), \quad \hat{d}_{\text{dense}}^{(2)}(p) = \frac{1}{\text{softplus}(s_t^{\text{d2}} \cdot \hat{\rho}_{\text{dense}}^{(2)}(p) + b_t^{\text{d2}}) + \varepsilon}$$

Provides 14,400 loss terms (160×90) per batch — **56× more** than $K{=}256$ sparse queries. Params: Conv 1×1 (128→1) + calibration = ~0.3K. Cost: ~2.3M MACs (training only).

**L3 dense head (stride 16):**
$$\hat{\rho}_{\text{dense}}^{(3)}(p) = \text{Conv}_{1 \times 1}(F_t^{(3)}, p), \quad \hat{d}_{\text{dense}}^{(3)}(p) = \frac{1}{\text{softplus}(s_t^{\text{d3}} \cdot \hat{\rho}_{\text{dense}}^{(3)}(p) + b_t^{\text{d3}}) + \varepsilon}$$

Provides 3,600 loss terms (80×45). Params: Conv 1×1 (192→1) + calibration = ~0.4K. Cost: ~0.7M MACs (training only).

**Total dense auxiliary:** ~0.7K params, ~3M MACs, 18,000 gradient sources. All discarded at inference. Each head has independent calibration $(s_t^{\text{d}\ell}, b_t^{\text{d}\ell})$ following the same separate-calibration convention as A4.

---

### 4.4 Algorithm B: Per-Query Sparse Depth Inference

**Input:** $\text{cache}_t$ and query batch $Q = \{q_j\}_{j=1}^{K}$.
**Output:** $\{(\hat{d}_j, \sigma_j)\}_{j=1}^{K}$.

All steps are batched over $K$ queries in parallel.

**Coordinate convention:** All bilinear lookups use `F.grid_sample` with `align_corners=False`, `padding_mode='zeros'`. Pixel coordinates $p$ are normalized via $\text{grid} = 2p / \text{dim} - 1$ (Deformable DETR convention).

#### B1. Feature extraction and multi-scale local cross-attention (MSLCA)

**Fine center feature (L1):**
$$f_q^{(1)} = \text{Bilinear}(F_t^{(1)}, \text{Normalize}(q / s_1)) \quad \in \mathbb{R}^{64}$$

**Positional encoding:**
$$\text{pe}_q = [\sin(2\pi \sigma_l u/W); \cos(2\pi \sigma_l u/W); \sin(2\pi \sigma_l v/H); \cos(2\pi \sigma_l v/H)]_{l=0}^{7}$$
with $\sigma_l = 2^l$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

**L4 center feature:**
$$c_q^{(4)} = \text{Bilinear}(G_t^{(4)}, \text{Normalize}(q / s_4)) \quad \in \mathbb{R}^{384}$$
Dual role: (1) center of MSLCA L4 grid → projected via $W_{v4}$ into $\mathcal{T}_{\text{ms}}$, absorbed into $h_{\text{ms}}$; (2) B4 context token via $W_{\text{ms4}}$ (384→192). Single bilinear lookup, two separate projections.

**L1 local neighborhood sampling ($N_{\text{loc}} = 32$):**

Fixed grid (5×5 minus center, 24 points) + 8 learned offsets:
$$\Delta_m = r_{\max} \cdot \tanh(W_{\text{off}}^{(m)} f_q^{(1)} + b_{\text{off}}^{(m)}), \quad m = 1, \ldots, 8, \quad r_{\max} = 6$$

For each offset $\delta$:
$$f_\delta = \text{Bilinear}(F_t^{(1)}, \tilde{q} + \delta), \quad h_\delta = \text{GELU}(W_{\text{loc}} [f_\delta; \phi(\delta)] + b_{\text{loc}}), \quad h_\delta \in \mathbb{R}^d$$

where $\phi(\delta)$ is Fourier encoding of the offset (4 freq × 2 trig × 2 dims = 16 dims). Input: 64 + 16 = 80.

Aggregate via multi-head attention pooling ($H_{\text{loc}} = 4$, $d_{\text{head}} = 48$):
$$q_{\text{att}} = W_Q^{(\text{loc})} [f_q^{(1)}; \text{pe}_q] \in \mathbb{R}^d, \quad k_\delta = W_K^{(\text{loc})} h_\delta \in \mathbb{R}^d$$
$$\alpha_{\delta,h} = \frac{\exp(q_h^\top k_{\delta,h} / \sqrt{d_{\text{head}}})}{\sum_{\delta'} \exp(q_h^\top k_{\delta',h} / \sqrt{d_{\text{head}}})}, \quad \tilde{v}_h = \sum_\delta \alpha_{\delta,h} \, h_{\delta,h}$$
$$l_q = W_O^{(\text{loc})} [\tilde{v}_1; \ldots; \tilde{v}_{H_{\text{loc}}}], \quad l_q \in \mathbb{R}^d$$
The query conditions on both the center L1 feature and positional encoding — the model knows local texture AND where it is in the image. Four heads capture different relevance criteria (e.g., edge-aligned neighbors, same-texture neighbors, contrast-sensitive neighbors). No separate $W_V$ ($h_\delta$ is already feature-projected via $W_{\text{loc}}$, head-partitioned for values). Params: $W_Q^{(\text{loc})}$ (96→192, ~18.6K) + $W_K^{(\text{loc})}$ (192→192, ~37K) + $W_O^{(\text{loc})}$ (192→192, ~37K) = ~93K.

**Stride-4 intermediate local ($N_{\text{loc}}^{(\text{int})} = 16$):**

Same structure: 3×3 minus center (8 fixed) + 8 learned offsets ($r_{\max}^{(\text{int})} = 4$), Fourier-encoded, multi-head attention-pooled (4 heads, same mechanism as L1, query from $[f_q^{(\text{int})}; \text{pe}_q]$):
$$l_q^{(\text{int})} \in \mathbb{R}^d$$
Params: ~109K each (offset heads ~1K + local MLP ~16K + MH attention pooling ~93K).

**Multi-Scale Local Cross-Attention (MSLCA):**

Instead of naively concatenating center features and projecting via a flat MLP (608→192), MSLCA gathers 3×3 local grids at L2, L3, and L4, and uses 2-layer query-conditioned cross-attention (with per-layer KV and FFN) to aggregate multi-scale context. This forms a symmetric 3-level local pyramid — L2 (stride 8, ConvNeXt), L3 (stride 16, Swin windowed attention), and L4 (stride 32, full self-attention with global context) — each providing qualitatively different information: L2 captures local texture gradients, L3 captures surface structure, and L4 captures globally-enriched scene semantics. The 2-layer design enables iterative cross-scale refinement: layer 1 does coarse scale selection with a weak query, FFN provides nonlinear head mixing, layer 2 re-attends with an informed query that already carries multi-scale context. Architecturally consistent with B2 and B4 (all 2-layer cross-attention decoders). Having L4 in $\mathcal{T}_{\text{ms}}$ bootstraps B2: $h_{\text{ms}}$ already carries local L4 awareness, so B2's 2-layer cross-attention over all 880 L4 tokens can focus on global refinement rather than first discovering local context from scratch.

*Step 1 — Gather local multi-scale tokens:*

L2 grid (stride 8): Sample 3×3 grid centered at $q/s_2$ with unit spacing in L2 native coordinates (each cell covers 8×8 original pixels):
$$t_i^{(2)} = W_{v2} \, \text{Bilinear}(F_t^{(2)}, \text{Normalize}(q/s_2 + \delta_i)) + e_{L2} + \text{rpe}_i, \quad i = 1, \ldots, 9$$

$W_{v2} \in \mathbb{R}^{d \times 128}$ projects L2 features to $d = 192$. $e_{L2} \in \mathbb{R}^d$ is a learned scale embedding. $\text{rpe}_i \in \mathbb{R}^d$ is a learned relative position embedding for grid offset $\delta_i \in \{-1,0,1\}^2$ (9-entry table, shared across L2/L3/L4).

L3 grid (stride 16): Same 3×3 pattern at $q/s_3$ (each cell covers 16×16 original pixels):
$$t_i^{(3)} = \text{Bilinear}(F_t^{(3)}, \text{Normalize}(q/s_3 + \delta_i)) + e_{L3} + \text{rpe}_i, \quad i = 1, \ldots, 9$$

L3 features are 192ch = $d$, no projection needed (identity). $e_{L3} \in \mathbb{R}^d$ is a separate scale embedding. RPE table shared with L2 (same 3×3 spatial pattern).

L4 grid (stride 32): Same 3×3 pattern at $q/s_4$ (each cell covers 32×32 original pixels):
$$t_i^{(4)} = W_{v4} \, \text{Bilinear}(G_t^{(4)}, \text{Normalize}(q/s_4 + \delta_i)) + e_{L4} + \text{rpe}_i, \quad i = 1, \ldots, 9$$

$W_{v4} \in \mathbb{R}^{d \times 384}$ projects L4 features to $d = 192$. $e_{L4} \in \mathbb{R}^d$ is a separate scale embedding. RPE table shared with L2/L3 (same 3×3 spatial pattern). L4 features have passed through 2 full self-attention layers over 880 tokens — each token encodes globally-aggregated scene context, not just local appearance. The center token $t_5^{(4)}$ reuses the same bilinear lookup as $c_q^{(4)}$ (no extra lookup). Coverage: 96×96 original pixels.

Token set: $\mathcal{T}_{\text{ms}} = [t_1^{(2)}; \ldots; t_9^{(2)}; t_1^{(3)}; \ldots; t_9^{(3)}; t_1^{(4)}; \ldots; t_9^{(4)}] \in \mathbb{R}^{27 \times d}$.

*Step 2 — 2-layer query-conditioned cross-attention:*

Query seed from fine features:
$$h_{\text{ms}} = W_q \, [f_q^{(1)}; \text{pe}_q] \in \mathbb{R}^d, \quad W_q \in \mathbb{R}^{d \times 96}$$

Per-layer KV projections (each layer projects $\mathcal{T}_{\text{ms}}$ to its own key-value space), $\ell = 1, 2$:
$$K_{\text{ms}}^{(\ell)} = W_K^{(\ell)} \, \text{LN}_{\text{kv}}(\mathcal{T}_{\text{ms}}), \quad V_{\text{ms}}^{(\ell)} = W_V^{(\ell)} \, \text{LN}_{\text{kv}}(\mathcal{T}_{\text{ms}}) \quad \in \mathbb{R}^{27 \times d}$$

2-layer cross-attention decoder (Pre-LN, each layer = CrossAttn + FFN with residuals):
$$h_{\text{ms}} \leftarrow h_{\text{ms}} + \text{MHCrossAttn}^{(\ell)}(Q = \text{LN}_q^{(\ell)}(h_{\text{ms}}), \; K = K_{\text{ms}}^{(\ell)}, \; V = V_{\text{ms}}^{(\ell)})$$
$$h_{\text{ms}} \leftarrow h_{\text{ms}} + \text{FFN}^{(\ell)}(\text{LN}_{\text{ff}}^{(\ell)}(h_{\text{ms}}))$$

4 heads, $d_{\text{head}} = 48$, FFN: $192 \to 768 \to 192$. Each layer has its own $W_Q$, $W_K$, $W_V$, $W_O$ projections and FFN (standard transformer decoder convention — DETR, SAM, Perceiver IO all use per-layer KV projections on a shared source). $\text{LN}_{\text{kv}}$ is shared (normalizes heterogeneous token scales across L2, L3, and L4 once). The $1 \times 27$ attention matrix per head is trivially cheap. Layer 1: coarse scale selection conditioned on $h_{\text{ms}}^{(0)}$ (weak query from fine features + position only). FFN enables nonlinear head mixing — the 4 heads' independent scale-specialized summaries interact through GELU. Layer 2: refined re-attention with an informed query that already carries multi-scale context — "I've seen L4 global context, now re-weight L2/L3 tokens on the near side of this depth boundary." The 3-scale heterogeneity of $\mathcal{T}_{\text{ms}}$ (L2 ConvNeXt, L3 Swin, L4 self-attention — qualitatively different feature types) justifies 2 layers: B2 gets 2 layers for 880 homogeneous L4 tokens, so MSLCA with 27 *heterogeneous* 3-scale tokens deserves equal processing depth.

*Step 3 — Merge with local features:*
$$h_{\text{point}} = \text{LN}\big(h_{\text{ms}} + W_{p2} \cdot \text{GELU}(W_{p1} [h_{\text{ms}}; l_q; l_q^{(\text{int})}] + b_{p1}) + b_{p2}\big)$$
Input: $192 + 192 + 192 = 576$. Hidden/output: $d = 192$. Params: ~148K. The $h_{\text{ms}}$ residual ensures multi-scale context (L2/L3/L4, processed through 4 nonlinear stages in MSLCA's 2-layer decoder) is the base signal — the MLP learns a correction from local L1 features, preventing dilution of depth-rich $h_{\text{ms}}$ by shallow $l_q$/$l_q^{(\text{int})}$ (67% of concat input). With 2-layer MSLCA, the merge MLP's role is cleanly separated: MSLCA's FFNs handle cross-scale processing, the merge MLP handles $l_q$/$l_q^{(\text{int})}$ integration. Residual chain: $h_{\text{ms}}^{(0)} \xrightarrow{+\text{MSLCA 2L}} h_{\text{ms}} \xrightarrow{+\text{MLP}} h_{\text{point}} \xrightarrow{+\text{B2}} h_{\text{point}}'$.

**Why MSLCA over flat concat:**
- **Spatial selectivity:** Cross-attention lets the query attend to L2/L3 neighbors sharing the same depth surface, ignoring positions across depth boundaries. Flat concat treats all centers equally.
- **9 tokens per scale vs 1:** A single center feature captures only the value at the query coordinate. The 3×3 grids (covering 24×24px at L2, 48×48px at L3, 96×96px at L4) provide local receptive fields with gradients and context at each scale.
- **Scale-aware routing:** Scale embeddings enable the model to learn which scale is most informative per query (e.g., L3 for smooth regions, L2 for textured edges).
- **Attention everywhere:** B1 uses cross-attention over local multi-scale tokens, B2 over global L4 tokens, B4 over heterogeneous context tokens — uniform mechanism across the decoder.
- **L4 bootstrapping for B2:** Without L4 in $\mathcal{T}_{\text{ms}}$, $h_{\text{ms}}$ has zero L4 information — B2 must do all L4 integration from a cold start. With L4 3×3 in MSLCA, $h_{\text{point}}$ already carries local L4 awareness, so both B2 layers can focus on global refinement. Analogous to Swin's local windows followed by full self-attention — local first, then global.
- **Residual preservation:** The $h_{\text{ms}}$ residual in Step 3 ensures that multi-scale depth context survives the 576→192 merge bottleneck. Without it, $h_{\text{ms}}$ is only 33% of the concat input, competing with shallow L1-derived features. With the residual, the MLP's role shifts from "reconstruct from scratch" to "compute local correction" — matching how B2's cross-attention adds global context via residual.

$c_q^{(2)}$, $c_q^{(3)}$, and $c_q^{(4)}$ are the center tokens ($\delta = (0,0)$) of the L2/L3/L4 grids — dual roles: absorbed into $h_{\text{ms}}$ via MSLCA cross-attention (compressed) AND passed as separate B4 context tokens (uncompressed, via $W_{\text{ms2/4}}$). Single lookup each, two projections.

MSLCA params: $W_{v2}$ (~25K) + $W_{v4}$ (~74K) + $W_q$ (~19K) + per-layer $W_K/W_V$ (~148K) + L1 $W_Q/W_O$ (~74K) + L1 FFN (~296K) + L2 $W_Q/W_O$ (~74K) + L2 FFN (~296K) + LNs (~3K) + embeddings (~2K) = ~1,011K. 27 lookups (8 new L4 non-center + 1 shared with $c_q^{(4)}$). ~5.7M MACs per query (KV projected twice, once per layer).

#### B2. Global cross-attention into L4

**2-layer cross-attention into all 880 globally-enriched L4 tokens:**

Each layer $\ell = 1, 2$ (Pre-LN cross-attention + FFN with residuals):
$$h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}^{(\ell)}(Q = \text{LN}_q^{(\ell)}(h_{\text{point}}), \; K = K_t^{(\ell)}, \; V = V_t^{(\ell)})$$
$$h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}^{(\ell)}(\text{LN}^{(\ell)}(h_{\text{point}}))$$

6 heads, $d_{\text{head}} = 32$, FFN: $192 \to 768 \to 192$. The $1 \times 880$ attention matrix per head is trivially cheap. $K_t^{(\ell)}, V_t^{(\ell)}$ are pre-computed per layer in Phase A — per-query cost is only Q projection, attention, and FFN.

After 2 layers: $h_{\text{point}}' \in \mathbb{R}^d$ — center token enriched with global scene context.

**Attention-based routing (free — no extra parameters):**

Average attention weights from layer 2 across heads, select top-$R$:
$$\bar{\alpha}_q = \frac{1}{H} \sum_{h=1}^{H} \alpha_{q,h} \quad \in \mathbb{R}^{880}, \quad R_q = \text{TopR}(\bar{\alpha}_q, R{=}32)$$

Each $r \in R_q$ maps to pixel coordinate $\mathbf{p}_r$ via L4 grid geometry (40×22, stride 32). $R = 32$ from 880 = 3.6% selection ratio. Each L4 position covers 32×32 pixels. Straight-through routing: hard top-R forward, STE backward.

**Per-query cost:** ~1.4M MACs for 2-layer cross-attention 

#### B3. Deformable multiscale read

For each anchor $r \in R_q \cup \{q\}$ (32 from B2's attention-based routing + 1 query-local), predict sampling offsets and importance weights, then read from the multi-scale pyramid. The 33rd anchor at the query pixel itself ensures B3's adaptive deformable sampling covers the local neighborhood — not just the remote locations B2 attended to. Ref: Deformable DETR (Zhu et al. ICLR 2021) places reference points at the query's own location; our design extends this by combining self-referencing with attention-routed anchors.

**Conditioning:**
$$\Delta\mathbf{p}_r = \mathbf{p}_r - q \quad \text{(query-to-anchor offset in original pixel coordinates)}$$
$$u_r = \text{LN}(\text{GELU}(W_u [h_{\text{point}}';\; g_r;\; \phi_{\text{B3}}(\Delta\mathbf{p}_r)] + b_u)), \quad u_r \in \mathbb{R}^d$$

where $g_r = W_g \, G_t^{(4)}[\mathbf{p}_r^{(4)}] \in \mathbb{R}^d$ is the anchor's L4 feature (pre-projected from 384ch). Input: $d + d + 32 = 416$. The conditioning tells the offset head: "I'm this query ($h_{\text{point}}'$, globally aware from B2), looking at an anchor whose content is $g_r$, located at this spatial offset." Shared across all 33 anchors. For the query-local anchor ($r = q$): $\Delta\mathbf{p}_q = 0$, $g_q = W_g \, G_t^{(4)}[q/s_4]$, and $\phi_{\text{B3}}(0)$ is a fixed constant — the offset head learns a local sampling pattern conditioned purely on query content and L4 appearance. Zero extra parameters.

**v10 fix (audit V08):** Added GELU before LN — the conditioning now has a nonlinearity for cross-stream interaction between query, anchor content, and spatial offset. v9 used only linear + LN, which cannot model nonlinear interactions (e.g., "if textured region AND far anchor, sample more broadly"). Zero extra params.

$\phi_{\text{B3}}(\Delta\mathbf{p}_r)$: Fourier encoding of the normalized offset $[\Delta u/W, \Delta v/H] \in [-1, 1]$, same formula as $\text{pe}_q$ (8 frequencies, 32 dims).

**Offset and weight prediction:**
$$\Delta p_{r,h,\ell,m} = W^\Delta \, u_r + b^\Delta, \quad \beta_{r,h,\ell,m} = W^a \, u_r + b^a$$
Two shared linear layers: offsets (d → H×L×M×2 = 144), weights (d → H×L×M = 72). Offsets are unbounded (no tanh), following Deformable DETR. Training stability: zero initialization + 0.1× LR on offset heads.

**Sampling with per-level normalization:**
$$p_{\text{sample}} = \mathbf{p}_r^{(\ell)} + \frac{\Delta p_{r,h,\ell,m}}{S_\ell}$$
$$f_{r,h,\ell,m} = \text{GridSample}_{\text{zeros}}(\hat{F}_t^{(\ell)}, \text{Normalize}(p_{\text{sample}}))$$

where $\mathbf{p}_r^{(\ell)} = \mathbf{p}_r / s_\ell$ maps the anchor to level-$\ell$ native coordinates, and $S_\ell$ is the spatial extent of level $\ell$ (e.g., [160, 80, 40] for L2–L4).

**v10 fix (audit V09):** Added per-level offset normalization $\Delta p / S_\ell$, following Deformable DETR's convention (`sampling_locations = reference_points + offsets / spatial_shapes`). This makes offsets scale-invariant — the model outputs the same magnitude regardless of level, and the spatial shape scales it appropriately. Without normalization, the model must learn different offset magnitudes for each level from a single linear layer. Ablate: with vs without normalization.

**Multi-head per-anchor aggregation:** $H = 6$ heads, $L = 3$ levels (2–4), $M = 4$ samples per head per level.

Per-head softmax over levels and samples:
$$a_{r,h,\ell,m} = \frac{\exp(\beta_{r,h,\ell,m})}{\sum_{\ell',m'} \exp(\beta_{r,h,\ell',m'})}, \quad \tilde{h}_{r,h} = \sum_{\ell,m} a_{r,h,\ell,m} \, v_{r,h,\ell,m}$$

where $v_{r,h,\ell,m} \in \mathbb{R}^{d/H}$ is the head-partitioned pre-projected feature (from $\hat{F}_t^{(\ell)}$, already at 192ch). Concat heads + output projection:
$$h_r = W_O [\tilde{h}_{r,1}; \ldots; \tilde{h}_{r,H}] + b_O, \quad h_r \in \mathbb{R}^d$$

**Budget:** 33 anchors × 72 samples/anchor = **2,376 deformable lookups** (32 remote + 1 query-local). Total lookups per query: 1 (L1 center) + 27 (MSLCA L2/L3/L4 grids) + 32 (L1 local) + 16 (intermediate local) + 2,376 (deformable) = **2,452**. The query-local anchor adds 72 lookups (~3% increase) but these are cache-friendly (near the MSLCA sampling region) — negligible wall-clock impact.

B3 params: conditioning (416→192, ~80K) + offsets (192→144, ~28K) + weights (192→72, ~14K) + $W_O$ (192→192, ~37K) = ~159K.

#### B4. Fusion decoder (3-layer cross-attention transformer)

**Context token set (37 tokens with 5 type embeddings):**
$$T_q = [l_q{+}e_{\text{loc}};\; c_q^{(2)}{+}e_{\text{ms2}};\; c_q^{(3)}{+}e_{\text{ms3}};\; c_q^{(4)}{+}e_{\text{ms4}};\; h_{r_1..33}{+}e_{\text{deform}}]$$

- $l_q + e_{\text{loc}}$: aggregated L1 local neighborhood (from B1)
- $c_q^{(2)} + e_{\text{ms2}}$: L2 center (128ch → 192ch via $W_{\text{ms2}}$) — center of MSLCA L2 grid (no extra lookup)
- $c_q^{(3)} + e_{\text{ms3}}$: L3 center (192ch = $d$, identity) — center of MSLCA L3 grid (no extra lookup)
- $c_q^{(4)} + e_{\text{ms4}}$: L4 center (384ch → 192ch via $W_{\text{ms4}}$) — center of MSLCA L4 grid (no extra lookup)
- $h_{r_1..33} + e_{\text{deform}}$: deformable evidence from 33 anchors (32 attention-routed + 1 query-local)

Projection params: $W_{\text{ms2}}$ (128×192 ≈ 25K) + $W_{\text{ms4}}$ (384×192 ≈ 74K) ≈ ~99K. Type embeddings: 5 × $d$ = 960 params.

**KV normalization:** $T_q \leftarrow \text{LN}_{\text{kv}}(T_q)$ — normalizes heterogeneous token scales. Applied once (static $T_q$).

**3-layer transformer decoder (Pre-LN):**
$$h_{\text{point}}' \leftarrow h_{\text{point}}' + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}'), \; KV = T_q)$$
$$h_{\text{point}}' \leftarrow h_{\text{point}}' + \text{FFN}(\text{LN}(h_{\text{point}}'))$$

6 heads, $d_{\text{head}} = 32$, FFN $192 \to 768 \to 192$. Attention matrix per head: $1 \times 37$ — trivially cheap. After 3 layers: $h_{\text{fuse}} = h_{\text{point}}'$. The 3rd layer is essentially free in wall-clock time (B4 is compute-bound on 37 tokens, masked entirely by B3's memory latency) and enables higher-order cross-type reasoning: layer 1 attends to individual evidence types, layer 2 learns pairwise interactions, layer 3 resolves multi-way conflicts (e.g., deformable evidence from occluder vs occluded surface vs multi-scale center priors).

B4 params: 3 layers × (Q/K/V/O projections + FFN + LNs) ≈ ~1,334K.

**Why B4 after B2:** B2 provides global context via soft attention over all 880 L4 tokens — broad but at stride 32 only. B4 provides targeted multi-scale context via hard attention over 37 tokens carrying B3's deformable evidence at strides 8/16/32 — including the query-local anchor's adaptive local samples alongside remote contextual evidence. Complementary: B2 = "understand the scene", B4 = "fuse detailed evidence."

#### B5. Depth prediction

**Relative depth code:**
$$r_q = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2}$$
MLP $192 \to 384 \to 1$. The wider hidden layer (2× input dim) gives the final nonlinear mapping more capacity to separate depth-relevant features from noise in $h_{\text{fuse}}$, at negligible compute cost (~74K MACs). Params: ~74K.

**Calibration and depth conversion:**
$$\rho_q = s_t \cdot r_q + b_t, \quad \hat{d}_q = \frac{1}{\text{softplus}(\rho_q) + \varepsilon}, \quad \varepsilon = 10^{-6}$$

**Center-only auxiliary (training only):**
$$r_q^{\text{ctr}} = W_{\text{ctr},2} \cdot \text{GELU}(W_{\text{ctr},1} \, h_{\text{point}}^{(0)} + b_{\text{ctr},1}) + b_{\text{ctr},2}$$
MLP $192 \to 96 \to 1$ on $h_{\text{point}}^{(0)}$ (BEFORE B2). Calibrated by separate $(s_t^{\text{ctr}}, b_t^{\text{ctr}})$ — independent from the main branch's $(s_t, b_t)$. Forces center branch to remain independently informative. Params: ~19K (MLP) + ~0.8K (calibration, in A4).

**Uncertainty (optional, disabled by default):**
$$\sigma_q = \text{softplus}(W_\sigma \, h_{\text{fuse}} + b_\sigma) + \sigma_{\min}, \quad \sigma_{\min} = 0.01$$

#### Nonlinear depth per query

- Backbone: L1 GELU (1) + L2 3×ConvNeXt (3×2=6) + L3 4×Swin (4×2=8) + L4 2×SelfAttn (2×2=4) + ConvGRU (3 gates, ~3) = **~22 stages**
- Decoder: B1 MSLCA 2×(cross-attn+FFN) (2×2=4) + B1 merge MLP (2) + B2 cross-attn (2×2=4) + B3 conditioning (2) + B4 fusion (3×2=6) + B5 MLP (2) = **~20 stages**
- **Total: ~42 nonlinear stages per query**

**Comparison with DAv2-S (~24 encoder sub-layers):** Our total (~42) exceeds DAv2's depth, but the nature differs. DAv2's 24 sub-layers are all global self-attention (every token sees every other token at every layer). Our backbone stages are heterogeneous: L1 is pointwise, L2 is local (~112px RF), L3 is windowed (8×8), only L4 is global. The decoder adds 9 cross-attention stages (2×MSLCA + 2×B2 + 3×B4 = 7 cross-attention + 2 B4 FFN interaction stages total) plus local/conditioning stages. The sparse formulation trades uniform global context (DAv2) for deeper per-query processing that's affordable because only K queries traverse the full decoder, not all HW pixels.

---

## 5. Training

### 5.1 Loss Functions

**Data fit:**
$$L_{\text{point}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}(q) - \rho^*(q)), \quad \hat{\rho}(q) = \text{softplus}(\rho_q) + \varepsilon$$

**Scale-invariant structure:**
$$L_{\text{silog}} = \sqrt{\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q\right)^2}, \quad \delta_q = \log \hat{d}(q) - \log d^*(q)$$
Default $\lambda_{\text{var}} = 0.5$ (F^3 default). Ablate $\{0.5, 0.85, 1.0\}$.

**Center auxiliary (prevents center collapse):**
$$L_{\text{ctr}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}^{\text{ctr}}(q) - \rho^*(q)), \quad \hat{\rho}^{\text{ctr}}(q) = \text{softplus}(s_t^{\text{ctr}} \cdot r_q^{\text{ctr}} + b_t^{\text{ctr}}) + \varepsilon$$

**Dense backbone auxiliary (56× more backbone gradients, training only):**
$$L_{\text{dense}} = \sum_{\ell \in \{2,3\}} \frac{\lambda_{\text{d}\ell}}{|P_v^{(\ell)}|} \sum_{p \in P_v^{(\ell)}} \text{Huber}(\hat{\rho}_{\text{dense}}^{(\ell)}(p) - \rho^*(p))$$

where $P_v^{(\ell)}$ is the set of pixels at level $\ell$'s resolution with valid GT depth. Provides 18,000 gradient sources (14,400 at L2 + 3,600 at L3) vs $K{=}256$ from sparse queries. Gradients flow through the entire backbone (L2→L3→L4→stems), ensuring features are useful everywhere, not just at queried locations. See A5 for architecture. Ref: PSPNet (Zhao et al. CVPR 2017), Mask2Former (Cheng et al. CVPR 2022).

$\lambda_{\text{d2}} = 0.5$, $\lambda_{\text{d3}} = 0.25$. Warm-up schedule: $\lambda_{\text{dense}}$ scales from 1.0→0.25 over training (start with dense dominance when sparse routing is untrained, then fade as the query decoder matures).

**Temporal consistency via warped depth (exploits F^3's temporal advantage):**

Given consecutive F^3 windows $t$ and $t{+}1$ (each 20 ms, non-overlapping), reproject depth from window $t$ into window $t{+}1$'s coordinate frame using known camera pose, then compare with the depth predicted at the reprojected pixel:

$$q' = \pi(T_{t \to t+1} \cdot \pi^{-1}(q, \hat{d}_q^{(t)}))$$
$$L_{\text{temp}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} M_q \cdot |\hat{d}_{q'}^{(t+1)} - \hat{d}_q^{(t, \text{warped})}|$$

where $\pi$ / $\pi^{-1}$ are projection / back-projection using known camera intrinsics, $T_{t \to t+1} \in SE(3)$ is the relative pose between window centers (from dataset GT poses), $\hat{d}_q^{(t, \text{warped})}$ is $\hat{d}_q^{(t)}$ transformed to the new frame's depth, and $\hat{d}_{q'}^{(t+1)}$ is the model's prediction at the reprojected pixel $q'$ in window $t{+}1$.

**Validity mask** $M_q$: A query $q$ contributes to $L_{\text{temp}}$ only if: (1) $q'$ lands within the image bounds, (2) no occlusion — the reprojected depth $\hat{d}_q^{(t, \text{warped})}$ is within 5% of $\hat{d}_{q'}^{(t+1)}$ (forward-backward check), and (3) $q'$ is at least 0.5 pixels from the image border. This excludes disoccluded regions where the loss is undefined. Ref: MonoDepth2 auto-masking (Godard et al. ICCV 2019).

**Why warped, not same-pixel:** Depth at a fixed pixel $(u,v)$ changes between frames due to ego-motion — a wall at 5 m becomes 4.98 m after 2 cm forward motion. Same-pixel comparison penalizes correct depth changes and biases toward infinite depth. All major temporal depth methods (ManyDepth, Watson et al. CVPR 2021; TC-Depth, Ruhkamp et al. IROS 2021; MonoDepth2, Godard et al. ICCV 2019) use geometric reprojection. Our datasets (M3ED, DSEC, MVSEC) provide GT camera poses; TartanAir v2 provides exact synthetic poses. DAv2 has zero temporal signal — this loss is unique to our event-camera setting.

**Calibration smoothness (supplementary):**
$$L_{\text{cal}} = |s_t - s_{t+1}| + |b_t - b_{t+1}|$$

The global calibration parameters $(s_t, b_t)$ should not jump between consecutive 20 ms windows regardless of scene content. This is cheap (2 scalar comparisons) and geometrically valid without reprojection. $\lambda_{\text{cal}} = 0.05$.

$\lambda_{\text{temp}} = 0.1$, $\lambda_{\text{cal}} = 0.05$. Both enabled from Stage 1 (no warm-up needed). Requires loading consecutive F^3 windows and their GT relative poses per batch.

**Feature distillation from DAv2 (cross-modal knowledge transfer, training only):**

$$L_{\text{feat}} = \sum_{\ell \in \{3,4\}} \frac{\lambda_{\text{f}\ell}}{|P^{(\ell)}|} \sum_{p \in P^{(\ell)}} \left(1 - \frac{\hat{F}_{\text{event}}^{(\ell)}(p) \cdot \hat{F}_{\text{DAv2}}^{(\ell)}(p)}{\|\hat{F}_{\text{event}}^{(\ell)}(p)\| \, \|\hat{F}_{\text{DAv2}}^{(\ell)}(p)\|}\right)$$

where $\hat{F}_{\text{event}}^{(\ell)}$ are our L3/L4 pyramid features projected via a connector MLP ($W_c^{(\ell)} \in \mathbb{R}^{384 \times C_\ell}$), and $\hat{F}_{\text{DAv2}}^{(\ell)}$ are frozen DAv2 ViT-S intermediate features (layer 6 for L3, layer 12 for L4) bilinearly interpolated to match spatial resolution.

**Connector MLP (per level):** $\hat{F}_{\text{event}}^{(\ell)} = W_{c2}^{(\ell)} \cdot \text{GELU}(W_{c1}^{(\ell)} \cdot F_{\text{event}}^{(\ell)})$. L3: 192→384→384 (~148K), L4: 384→384→384 (~296K). Total connector params: ~444K (training only, discarded at inference).

**Why cosine similarity:** Cosine loss matches feature directions (structural semantics) without requiring magnitude alignment across the event-RGB modality gap. Events lack color/texture but share geometric structure (edges, surfaces, depth discontinuities) with RGB features. Cosine loss captures this structural alignment. Ref: Depth AnyEvent (ICCV 2025) uses cross-modal distillation from DAv2 for event depth; EventDAM (ICCV 2025) uses sparsity-aware feature distillation; Spike-Driven Transformer (2024) shows 49% Abs Rel improvement from DINOv2 feature alignment.

**Sparsity-aware weighting:** Event features are inherently sparse — textureless/static regions produce few events and weak features. Weight the cosine loss by event density at each spatial location: $w_p = \min(n_p / \bar{n}, 2.0)$ where $n_p$ is the event count in the patch at position $p$ and $\bar{n}$ is the mean. This focuses distillation on regions where the event backbone has sufficient signal. Ref: EventDAM's SFD module.

$\lambda_{\text{f3}} = 0.1$, $\lambda_{\text{f4}} = 0.1$. Requires running frozen DAv2 forward pass on aligned RGB frames during training — cost: ~8G MACs per frame (~0.5 ms on RTX 4090), amortized across $K$ queries. Enabled in Stage 1 only (pseudo-label datasets have aligned RGB). Disabled in Stage 2 (real-only, may lack synchronized RGB).

**Total:**
$$\mathcal{L} = L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + \lambda_{\text{ctr}} L_{\text{ctr}} + L_{\text{dense}} + \lambda_{\text{temp}} L_{\text{temp}} + \lambda_{\text{cal}} L_{\text{cal}} + L_{\text{feat}}$$

with $\lambda_{\text{si}} = 0.5$, $\lambda_{\text{ctr}} = 0.25$, $\lambda_{\text{temp}} = 0.1$, $\lambda_{\text{cal}} = 0.05$, $\lambda_{\text{f3}} = \lambda_{\text{f4}} = 0.1$. $L_{\text{dense}}$ has per-level weights (see above). All auxiliary losses ($L_{\text{ctr}}$, $L_{\text{dense}}$, $L_{\text{temp}}$, $L_{\text{cal}}$, $L_{\text{feat}}$) and connector MLPs are discarded at inference.

### 5.2 Training Data

| Dataset | GT type | Resolution | Density | Metric? | GT Pose? |
|---------|---------|------------|---------|---------|----------|
| **M3ED** | Real LiDAR (VLP-16) | 1280×720 | Sparse (~5–10%) | Yes | Yes (VIO) |
| **DSEC** | Real stereo + LiDAR | 640×480 | Semi-dense (~30–50%) | Yes | Yes (GPS/INS) |
| **MVSEC** | Real LiDAR + IMU + MoCap | 346×260 | Sparse | Yes | Yes (MoCap/IMU) |
| **TartanAir v2** | Synthetic (Unreal) | 640×640 | Dense (100%) | Yes | Yes (exact) |
| **M3ED pseudo** | DAv2 pseudo labels | 1280×720 | Dense (100%) | No (relative) | Yes (VIO) |

Real GT is primary — LiDAR sparsity is not a problem since we sample queries at LiDAR-valid locations. DAv2 pseudo labels provide supplementary dense coverage (used with scale-invariant losses only). All datasets provide GT camera poses, which are consumed for $L_{\text{temp}}$ (warped depth consistency) — F^3 itself does not use pose, but we load the relative pose $T_{t \to t+1}$ between consecutive windows for the temporal loss.

### 5.3 Query Sampling

Multinomial sampling with category priority — pick category first, then sample a pixel from that category:
- 40% LiDAR-valid pixels (real GT, highest quality)
- 20% DAv2 pseudo-labeled pixels without LiDAR (dense coverage)
- 15% event-dense regions (high-signal areas)
- 15% high-gradient regions from depth maps (boundary quality)
- 10% hard-example regions from spatial loss map (see below)

Categories may overlap (a LiDAR-valid pixel can also be high-gradient). With multinomial sampling, overlap is irrelevant — each draw selects a category, then a pixel within that category.

**Hard-example mining:** Maintain a spatial loss map $\mathcal{M}(u,v)$ at stride 16 (80×45 grid), updated via EMA after each batch:
$$\mathcal{M}_{t+1}(u,v) = 0.99 \cdot \mathcal{M}_t(u,v) + 0.01 \cdot \bar{L}_{\text{local}}(u,v)$$

where $\bar{L}_{\text{local}}$ is the mean loss of queries falling within each grid cell. The hard-example category samples from cells with probability $\propto \mathcal{M}(u,v)$. Initialized to uniform. This focuses training compute on regions where the model struggles (edges, textureless areas, far depths) — dense methods waste compute on easy flat surfaces uniformly. Ref: OHEM (Shrivastava et al. CVPR 2016), Focal Loss (Lin et al. ICCV 2017), PointRend uncertain-point oversampling (Kirillov et al. CVPR 2020).

**Train-large-K, infer-small-K:** Train with $K_{\text{train}} = 2048$, infer at $K_{\text{infer}} = 256$. The decoder has no inter-query coupling — $K$ is a free parameter. Larger $K$ provides: (1) 8× more loss terms per batch → lower gradient variance, (2) better spatial coverage → backbone sees more diverse locations, (3) more routing patterns → B2 learns richer selection. Training decoder cost increases from ~1.6ms to ~10.5ms; backbone cost unchanged. Ref: DETR trains 100 queries with set-based loss, InfiniDepth evaluates at 100K queries — both show per-query architectures generalize across $K$.

Vary $K_{\text{train}} \in \{256, 512, 1024, 2048\}$ across batches (query curriculum, with larger $K$ more frequent later in training).

### 5.4 Training Schedule

**Stage 1 — Relative depth (mixed supervision, ~15–20 epochs):**
- Sparse queries: $L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + \lambda_{\text{ctr}} L_{\text{ctr}}$ (LiDAR-valid) or $\lambda_{\text{si}} L_{\text{silog}}$ only (pseudo-label queries)
- Dense backbone auxiliary: $L_{\text{dense}}$ on all pixels with valid GT at L2/L3 (enabled from epoch 1)
- Temporal consistency: $L_{\text{temp}}$ (warped depth) + $L_{\text{cal}}$ (calibration smoothness) across consecutive windows (enabled from epoch 1)
- Feature distillation: $L_{\text{feat}}$ on L3/L4 features via frozen DAv2 teacher (enabled from epoch 1; requires aligned RGB frames, available in M3ED/DSEC/TartanAir)
- $L_{\text{ctr}}$ warm-up: $\lambda_{\text{ctr}} = 0 \to 0.25$ over first 5 epochs
- $L_{\text{dense}}$ warm-down: scale factor $1.0 \to 0.25$ over training (dense dominates early, sparse takes over)
- Hard-example mining: $\mathcal{M}$ initialized uniform, updated via EMA from epoch 1
- $K_{\text{train}} = 2048$ (8× inference budget for stronger supervision)
- ConvGRU hidden state: BPTT through 4 consecutive windows (truncated); $h_0 = 0$ per sequence
- Backbone frozen (except ConvGRU, which trains from scratch)
- Datasets: M3ED (LiDAR + pseudo + RGB), DSEC (stereo GT + RGB), TartanAir v2 (RGB available)

**v10 fix (audit V12):** $L_{\text{ctr}}$ is now enabled from Stage 1 with warm-up, rather than delayed to Stage 2. PSPNet and DeepLabV3 enable auxiliary losses from the start. The warm-up avoids interfering with cross-attention convergence in early training.

**Stage 2 — Metric fine-tuning (~10 epochs):**
- Real LiDAR / stereo GT only (no pseudo labels)
- Full loss: $L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + \lambda_{\text{ctr}} L_{\text{ctr}} + L_{\text{dense}} + \lambda_{\text{temp}} L_{\text{temp}} + \lambda_{\text{cal}} L_{\text{cal}}$ (no $L_{\text{feat}}$ — real datasets may lack synchronized RGB)
- $L_{\text{dense}}$ weight reduced to 0.1 (sparse decoder is primary, dense auxiliary is supplementary)
- $K_{\text{train}} = 2048$
- ConvGRU continues training; BPTT through 4 windows
- Optionally unfreeze F^3 backbone with 0.1× LR
- Evaluate on MVSEC outdoor_day1/day2

### 5.5 Regularization

- Attention dropout $p = 0.1$ on MSLCA, B2, and B4 cross-attention weights
- Weight decay 0.01, gradient clipping 1.0, mixed precision (bf16)

---

## 6. Runtime Analysis

### 6.1 Parameter Budget

**Phase A: Preprocessing (amortized, once per frame)**

| Component | Params |
|-----------|-------:|
| L1 LN + GELU | ~0.1K |
| Stem-1 Conv(64→64,k3s2) + LN | ~37K |
| Stem-2 Conv(64→128,k2s2) + LN | ~33K |
| L2: 3× ConvNeXt₁₂₈ (GRN, k=13) | ~435K |
| Down L2→L3 + LN | ~99K |
| L3: 4× SwinBlock₁₉₂ | ~1,778K |
| Down L3→L4 + LN | ~295K |
| L4: 2× FullSelfAttn₃₈₄ | ~3,542K |
| L4: ConvGRU₃₈₄ (3 gates, k=3, DW-sep) | ~886K |
| A3: Per-layer KV proj for B2 (2×384→192) | ~296K |
| A3: $W_g$ anchor proj (384→192) | ~74K |
| A3: Per-level $W_V$ pre-proj | ~136K |
| A4: Main calibration heads $(s_t, b_t)$ | ~0.8K |
| A4: Aux calibration heads $(s_t^{\text{ctr}}, b_t^{\text{ctr}})$ (train only) | ~0.8K |
| A5: Dense aux heads L2+L3 + calibration (train only) | ~0.7K |
| **Phase A total** | **~7,614K** |

**Phase B: Decoder (per-query, $d = 192$)**

| Component | Lookups | Params |
|-----------|:---:|-------:|
| B1: Center (L1) | 1 | — |
| B1: MSLCA (L2/L3/L4 3×3 grids, 2-layer cross-attn+FFN) | 27 | ~1,011K |
| B1: L1 local sampling (4-head attn-pool) | 32 | ~109K |
| B1: Stride-4 intermediate local (4-head attn-pool) | 16 | ~109K |
| B1: Merge MLP (576→192→192) | — | ~148K |
| B2: L4 cross-attn (2 layers, 880 tok) | — | ~740K |
| B3: Deformable read (H=6,L=3,M=4, 33 anchors) | 2,376 | ~159K |
| B4: $W_{\text{ms2/4}}$ proj + type emb + LN | — | ~100K |
| B4: Fusion (3 layers) | — | ~1,334K |
| B5: Depth head (192→384→1) + aux MLP | — | ~93K |
| **Phase B total** | **2,452** | **~3,803K** |

**Total trainable: ~11,417K (~11.4M)** (training-only components: A4 aux ~0.8K, A5 dense aux ~0.7K, B5 center aux MLP ~19K — all discarded at inference. Inference params: ~11,396K)

**vs DAv2-S:**
- Decoder: ~3.8M vs ~3M (DPT-S head) = **comparable** (3-layer B4 fusion + MSLCA 2-layer with per-layer KV + MH attention-pooled locals account for the increase; justified by heterogeneous token fusion and compute headroom)
- Total (excl. frozen F^3): ~11.4M vs ~25M (DAv2-S) = **2.2× smaller** (ConvGRU adds ~0.9M for temporal state — unique capability DAv2 lacks)
- Lookups: 2,452 per query vs 268K pixels (DPT output at 518×518)

### 6.2 Speed Estimate (RTX 4090, 1280×720)

**Precompute (once per window):**
$$T_{\text{precompute}} = T_{\text{F}^3}^{\text{ds2}} + T_{\text{backbone}} + T_{\text{misc}} \approx 4.5 + 1.35 + 0.1 = 5.95 \text{ ms}$$

**Per-query cost (MACs):**

| Stage | MACs | Memory reads | Est. time (μs) |
|-------|-----:|------------:|---:|
| B1: MSLCA (2-layer) + local + merge | ~6,600K | ~76 KB (76 lookups across L1/L2/L3/L4/int) | ~1.0 |
| B2: L4 cross-attn (2 layers) | ~1,410K | ~3 KB (KV cached across batch) | ~0.8 |
| B3: deformable (33 anchors: 32 remote + 1 local) | ~5,240K | ~3.5 MB (2,376 grid_samples; 72 local are cache-friendly) | ~2.5 |
| B4: projection + fusion (3 layers) | ~9,480K | ~30 KB (37 context tokens) | ~0.6 |
| B5: depth head (192→384→1) | ~74K | — | ~0.1 |
| **Total** | **~22.8M** | **~3.6 MB** (B3 = 97%) | **~5.0** |

**Per-query bottleneck — memory, not compute:** Arithmetic intensity: 22.8M MACs / 3.6 MB ≈ 6.3 MACs/byte — far below RTX 4090 FP16 roofline (~165 MACs/byte). Pure compute: 22.8M / 165T ≈ 0.14 μs, negligible vs ~5 μs memory latency. B3 dominates wall-clock (50%) despite only 23% of MACs — its 2,376 grid_samples from ~7 MB pre-projected feature maps cost ~2.5 μs (the 72 query-local samples are cache-friendly and add negligible latency vs the 2,304 remote scattered reads). B4 has 42% of MACs but only ~0.6 μs — 3 layers of cross-attention over 37 tokens is pure in-register compute. Consequence: compute-only changes (MSLCA depth, KV sharing, FFN width, B4 layers) have near-zero wall-clock impact; only grid_sample count (B3/local budget) or feature channels materially change $\beta$.

**Timing model:**
$$T_{\text{decoder}}(K) = \alpha + \beta K \approx 0.30 + 0.005K \text{ ms}$$

$\alpha \approx 0.30$ ms: kernel launch + dispatch overhead (independent of $K$).
$\beta \approx 0.005$ ms/query = 5.0 μs (see per-stage breakdown above). Memory-bandwidth-bound — B3's scattered reads dominate.

**Total:**
$$T_{\text{EventSPD}}(K) \approx 6.2 + 0.005K \text{ ms}$$

| $K$ | Decoder (ms) | Total (ms) | Hz | Dense (ms) | Speedup |
|-----|---:|---:|---:|---:|---:|
| 1 | 0.31 | 6.2 | 161 | ~23 | 3.7× |
| 64 | 0.62 | 6.5 | 154 | ~23 | 3.5× |
| 256 | 1.58 | 7.5 | 133 | ~23 | **3.1×** |
| 1024 | 5.42 | 11.3 | 88 | ~23 | 2.0× |

Dense baseline (ds1): F^3 ds1 (8.3 ms) + DAv2 decoder (~15 ms) = ~23 ms. Crossover: $(23 - 6.25) / 0.005 \approx K = 3{,}350$.

**Matched-resolution dense baseline (ds2):** F^3 ds2 (4.5 ms) + DAv2 decoder on 640×360 (~4 ms) = ~8.5 ms. Crossover: $(8.5 - 6.25) / 0.005 \approx K = 450$. This is the fairer comparison — same F^3 encoder, same resolution. Our speedup narrows (1.1× at K=256) but we process only K points vs 230K pixels, so accuracy-per-FLOP should favor sparse queries. Report both baselines.

**Key insight:** Precompute (~5.8 ms) = 78% of total at K=256. F^3 ds2 (~4.5 ms) = 78% of precompute. The event encoder is the bottleneck — further speedups require lighter event encoders or voxel grid representations.

---

## 7. Evaluation Protocol

### 7.1 Accuracy Metrics (at query points)

Standard monocular depth metrics at queried pixels:
- AbsRel, RMSE, RMSE_log, SiLog
- $\delta < 1.25$, $\delta < 1.25^2$, $\delta < 1.25^3$
- Report by depth range: near (0–10m), mid (10–30m), far (30m+)

### 7.2 Runtime Metrics

- $T_{\text{precompute}}$ (ms), $T_{\text{query}}(K)$ (ms), $T_{\text{total}}(K)$ (ms)
- Throughput (Hz): $1000 / T_{\text{total}}(K)$
- Speedup vs dense baseline
- GPU memory usage

### 7.3 Diagnostic Metrics

- **Center predictiveness:** Monitor $L_{\text{ctr}}$ — decreasing = center branch independently informative
- **Routing diversity:** Entropy of $\bar{\alpha}_q$ over 880 L4 positions. Low = collapse, high = diffuse, moderate = good spatial selection
- **B2 impact:** Cosine similarity between $h_{\text{point}}'$ (post-B2) and $h_{\text{point}}$ (pre-B2). Near 1.0 = B2 not contributing
- **B4 attention balance:** Average weight per token type. Monitor whether all 5 streams contribute

### 7.4 Robustness

Day / night evaluation, different platforms (car / spot / flying), event rate subsampling (25%, 50%, 75%, 100%).

---

## 8. Ablation Plan

### Tier 1 — Core validation (must have)

| # | Ablation | Question |
|---|----------|----------|
| 1 | Remove B2 (no global context) | Does L4 cross-attn help? Use random anchors for B3. |
| 2 | Remove B3 (no deformable) | Is targeted multi-scale evidence needed? $T_q$ = 4 tokens only. |
| 3 | Remove local sampling | Does neighborhood context help? Center + global + deformable only. |
| 4 | Query count scaling | Accuracy/speed at $K \in \{1, 4, 16, 64, 256, 1024\}$. |
| 5 | Freeze vs fine-tune backbone | Frozen / 0.1× LR / full LR. |
| 6 | Edge-pair loss ($L_{\text{edge}}$) | Sample P=32 edge-straddling query pairs from GT gradient map. $L_{\text{edge}} = \frac{1}{P}\sum\|(\hat{\rho}_a{-}\hat{\rho}_b) - (\rho^*_a{-}\rho^*_b)\|$. Tests whether per-query losses alone suffice for edge quality, or if explicit depth-difference supervision is needed. Adds 64 extra queries/batch. Ref: Xian et al. CVPR 2020 (structure-guided ranking); InfiniDepth shows ~25pp accuracy drop at edges. |

### Tier 2 — Design choices

| # | Ablation | Question |
|---|----------|----------|
| 7 | B2 layer count | 1, **2** (default), 3 layers. |
| 8 | Routing budget $R$ | $\{8, 16, \mathbf{32}, 64, 128\}$. |
| 9 | B4 fusion depth | 1, 2, **3** (default), 4 layers. |
| 10 | L4 self-attn depth | 0, 1, **2** (default), 4 layers. |
| 11 | Routing source | (a) **Attention-based**, (b) learned $W_r$, (c) random. |
| 12 | Local budget | $N_{\text{loc}} \in \{8, 16, \mathbf{32}, 48, 64\}$. |
| 12b | Local aggregation method | **4-head MH attention pooling with pe_q** (default) vs (a) single-head attention pooling without pe_q (−84K params), (b) max pooling (PointNet-style, −110K params), (c) max+mean concat (captures peaks + average), (d) multi-query attention (K=3 summary tokens into B4). |
| 13 | Deformable budget | $(H,L,M) \in \{(4,3,2)..\mathbf{(6,3,4)}...(6,3,6)\}$. |
| 13b | Query-local B3 anchor | **With** (default, 33 anchors) vs without (32 remote-only). Does adaptive local sampling improve edge/textureless regions? |
| 14b | ConvGRU temporal state | **With ConvGRU at L4** (default) vs (a) no GRU (independent windows), (b) ConvLSTM (adds cell state, +886K params), (c) GRU at L3+L4 (multi-scale state). Ref: Depth AnyEvent-R (ICCV 2025), DERD-Net (NeurIPS 2025). |
| 14 | B1 MSLCA configuration | **MSLCA L2+L3+L4, 2-layer, per-layer KV** (default) vs (a) L2+L3 only, 2-layer (no L4 grid, −74K params, −1.33M MACs), (b) L2+L3+L4, 1-layer no FFN (original design, −741K params, −2.7M MACs), (c) L2+L3+L4, 1-layer+FFN (single cross-attn with FFN), (d) flat concat MLP (608→192, no cross-attn), (e) shared KV across layers (−74K params, −2.0M MACs). Tests value of 3-level pyramid, L4 bootstrapping, MSLCA depth, per-layer KV, and cross-attention vs flat fusion. |

### Tier 3 — Training strategy

| # | Ablation | Question |
|---|----------|----------|
| 15 | Dense backbone auxiliary | Disable $L_{\text{dense}}$. Measures impact of 56× gradient coverage on backbone quality. |
| 16 | Train-large-K | $K_{\text{train}} \in \{256, 512, \mathbf{2048}, 4096\}$. Does 8× more queries per batch improve convergence? |
| 17 | Hard-example mining | Disable loss-map sampling (replace with random). Does focusing on hard regions help? |
| 18 | Temporal consistency | (a) Disable $L_{\text{temp}}$ + $L_{\text{cal}}$, (b) $L_{\text{cal}}$ only (no warped depth), (c) same-pixel $L_{\text{temp}}$ without warping (baseline), **(d) warped $L_{\text{temp}}$ + $L_{\text{cal}}$** (default). |
| 18b | Feature distillation | (a) Disable $L_{\text{feat}}$, (b) output distillation only (pseudo-labels, no feature matching), **(c) cosine feature distillation L3+L4** (default), (d) L4 only. Ref: Depth AnyEvent, EventDAM (ICCV 2025). |
| 19 | Dense aux warm-down | Fixed $\lambda_{\text{dense}}$ vs **warm-down** (default) vs disable after Stage 1. |

### Tier 4 — Extensions

| # | Ablation | Question |
|---|----------|----------|
| 20 | B2 only (no B3/B4) | Can L4 cross-attn alone suffice? Depth head on $h_{\text{point}}'$. |
| 21 | Hash encoding vs Fourier PE | Does learned position encoding help? |
| 22 | ~~Temporal memory~~ | *(Moved to Tier 2 as #14b — now default)* |
| 23 | Uncertainty head | $\sigma_q$ + $L_{\text{unc}}$. |
| 24 | Center auxiliary loss | Disable $L_{\text{ctr}}$. |
| 25 | Attention dropout rate | $p \in \{0, 0.05, \mathbf{0.1}, 0.2\}$. |

### Tier 5 — Backbone architecture

| # | Ablation | Question |
|---|----------|----------|
| 26 | Backbone architecture | (a) No backbone, (b) CNN-only, (c) CNN-light, (d) Hybrid-light, **(e) Hybrid** (default), (f) Heavy. |
| 27 | L3 window size | **8** (default, requires padding), 5 (divides 80×45 cleanly). |
| 28 | B2 KV sharing | **Separate per layer** (default, standard convention) vs shared KV across layers (−148K params). |
| 29 | B3 conditioning depth | **GELU+LN** (default) vs 2-layer MLP (+37K params). |
| 30 | B3 offset normalization | **Per-level** (default) vs unnormalized. |
| 31 | B2 KV normalization | Add LN on $K_t$/$V_t$ (+768 params). |
| 32 | Lightweight top-down FPN | Add L4→L3→L2 pathway: 1×1 conv + bilinear upsample + add (~98K params, ~0.05ms). Enriches L2/L3 features with L4 global context before B3 sampling. SAM2 added FPN over SAM1; DPT/DAv2 use top-down fusion. Our per-query B3 multi-level sampling may compensate (soft top-down), but raw L2 features at textureless regions could be uninformative. Trigger: if B4 attention on L2-sourced deformable tokens is consistently near-zero. |

### Tier 6 — Width and dimension

| # | Ablation | Question |
|---|----------|----------|
| 33 | **Option B** ($d=256$, channels [64,128,256,512]) | SOTA 1:2:4:8 ratio. ~14.1M params, ~47M FLOPs/query, $T(256) \approx 8.8$ ms, 2.6× vs dense. |
| 34 | Core dim $d$ (fixed backbone) | $d \in \{128, \mathbf{192}, 256\}$ with backbone [64,128,192,384]. |
| 35 | L2 channel width | $\{96, \mathbf{128}, 192\}$ch. |

### Tier 7 — F^3 resolution

| # | Ablation | Question |
|---|----------|----------|
| 36 | F^3 output resolution | (a) ds1: 1280×720×32 (~8.3ms), **(b) ds2: 640×360×64** (~4.5ms), (c) ds4: 320×180×64 (~3.0ms). |

---

## 9. Implementation

### 9.1 File Structure

```
src/f3/tasks/depth_sparse/
├── models/
│   ├── eventspd.py            # Main model: Algorithm A + B
│   ├── backbone.py             # A2: Wide pyramid backbone
│   ├── precompute.py          # A3: KV + W_V projections
│   ├── query_encoder.py       # B1: Local + center feature extraction
│   ├── l4_cross_attention.py  # B2: 2-layer cross-attn + routing
│   ├── deformable_read.py     # B3: Offset prediction + multiscale sampling
│   ├── fusion_decoder.py      # B4: 2-layer transformer (36 tokens)
│   └── depth_head.py          # B5: Depth prediction + calibration
├── utils/
│   ├── losses.py              # L_point, L_silog, L_ctr
│   ├── query_sampler.py       # Training query sampling
│   └── profiler.py            # Runtime benchmarking
├── train.py                   # Training loop
├── evaluate.py                # Accuracy evaluation
└── benchmark.py               # Speed benchmarking
```

### 9.2 Build Order

| Step | What | Validates |
|------|------|-----------|
| 1 | Dense baseline + query sampling | Accuracy/speed reference |
| 2 | Minimal query decoder (bilinear L1 + MLP) | Concept viability |
| 3 | Wide pyramid backbone (A2) | Feature quality |
| 4 | L4 cross-attention (B2) + routing | Global context + routing |
| 5 | Deformable sampling (B3) | Multi-scale evidence |
| 6 | Full pipeline (B4+B5) + training | Complete system |
| 7 | Ablations + paper figures | Publication readiness |

After each step: measure accuracy AND speed. Do not proceed if current step degrades accuracy without speed benefit.

---

## 10. Timeline (16 weeks)

- **Weeks 1–2:** Reproduce dense baseline, build query-level evaluation + profiling harness.
- **Weeks 3–4:** Minimal query decoder (steps 2–3), first accuracy/speed comparison.
- **Weeks 5–7:** Full EventSPD (steps 4–6), train stages 1–2, produce crossover plot.
- **Weeks 8–10:** Tier 1 ablations, identify key components, fix failure modes.
- **Weeks 11–12:** Tier 2–3 ablations.
- **Weeks 13–14:** Robustness evaluation, Stage 2 metric fine-tuning.
- **Weeks 15–16:** Paper writing.

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|:---:|------------|
| Accuracy drop at depth edges | Medium | Increase local budget; edge-focused query sampling |
| B2 attention collapse | Low | Monitor entropy; attention dropout |
| Routing collapse (same anchors) | Low | Monitor $\bar{\alpha}_q$ entropy; add $L_{\text{entropy}}$ if needed |
| Offsets collapse to zero | Low | 0.1× LR on offset heads (Deformable DETR) |
| Speed worse than dense | Very Low | Crossover at $K \approx 3{,}380$ |
| STE routing instability | Medium | Start without routing; add after convergence |
| Pseudo depth label noise | Medium | Two-stage: pseudo → metric LiDAR fine-tune |

---

## 12. RGB-SPD Extension

RGB-SPD applies the same decoder (B1–B5, identical) to RGB images by replacing F^3 with a lightweight conv stem and deepening the backbone.

### 12.1 Changes from EventSPD

| Component | EventSPD | RGB-SPD |
|-----------|----------|---------|
| A1 | F^3 ds2 → 640×360×64 (57G, 4.5 ms) | Conv(3→64,k7,s2)+LN+GELU (4.3G, 0.3 ms) |
| L1 | LN + GELU (identity) | +2× Conv(64→64,k3)+LN+GELU (3.2G, 0.2 ms) |
| Intermediate | Stem-1 passthrough | +3× ConvNeXt₆₄ with GRN (15.6G, 1.0 ms) |
| L4 self-attn | 2 layers | 4 layers (+0.6G, +0.05 ms) |
| L2, L3, A3, A4, B1–B5 | — | Identical |

### 12.2 RGB-SPD Backbone

```
RGB 1280×720×3
→ Conv(3→64, k7,s2,pad3) + LN + GELU                       640×360×64
→ 2× [Conv(64→64, k3,s1,pad1) + LN + GELU]                 → L1   640×360×64     stride 2
→ Stem-1: Conv(64→64, k3,s2) + LN                           320×180×64            stride 4
→ 3× ConvNeXt_64 (with GRN)                                 → Int  320×180×64     stride 4
→ Stem-2: Conv(64→128, k2,s2) + LN                          160×90×128            stride 8
→ 2× ConvNeXt_128 (with GRN)                                → L2   160×90×128     stride 8
→ Down: Conv(128→192, k2,s2) + LN                           80×45×192             stride 16
→ 4× SwinBlock_192 (window=8, shifted)                      → L3   80×45×192      stride 16
→ Down: Conv(192→384, k2,s2) + LN                           40×22×384             stride 32
→ 4× FullSelfAttn_384 (6 heads, d_head=64, 880 tokens)     → L4   40×22×384      stride 32
```

### 12.3 Cost

| | EventSPD | RGB-SPD |
|---|:---:|:---:|
| Encoder total | ~73G / 5.7 ms | ~40G / 2.8 ms |
| Decoder (K=256) | 1.6 ms | 1.6 ms (identical) |
| **System total (K=256)** | **~7.4 ms** | **~4.4 ms** |

$$T_{\text{RGB-SPD}}(K) \approx 3.1 + 0.005K \text{ ms}, \quad \text{Crossover: } K \approx 3{,}980$$

Nonlinear depth per query: L1(3) + Int(6) + Stem-2(1) + L2(4) + L3(8) + L4(8) + decoder(14) = **~44 stages**.

**v10 fix (audit V04):** L4 now correctly counted as 4 layers × 2 sub-layers = 8 (v9 had "L4(4)" which was wrong).

RGB-SPD is an extension / second contribution, not the primary paper. Demonstrates the sparse decoder generalizes beyond events.

---

## 13. References

**Core methods:** F^3 ([2509.25146](https://arxiv.org/abs/2509.25146)), Deformable DETR ([2010.04159](https://arxiv.org/abs/2010.04159)), SAM ([2304.02643](https://arxiv.org/abs/2304.02643)), Perceiver IO ([2107.14795](https://arxiv.org/abs/2107.14795)), LIIF ([2012.09161](https://arxiv.org/abs/2012.09161)), PointRend ([1912.08193](https://arxiv.org/abs/1912.08193))

**Depth estimation:** DAv2 ([2406.09414](https://arxiv.org/abs/2406.09414)), DPT ([2103.13413](https://arxiv.org/abs/2103.13413)), BTS ([1907.10326](https://arxiv.org/abs/1907.10326)), E2Depth ([2010.08350](https://arxiv.org/abs/2010.08350)), ManyDepth ([2104.14540](https://arxiv.org/abs/2104.14540))

**Training strategy:** PSPNet (Zhao et al. CVPR 2017), GoogLeNet ([1409.4842](https://arxiv.org/abs/1409.4842)), Mask2Former ([2112.01527](https://arxiv.org/abs/2112.01527)), OHEM ([1604.03540](https://arxiv.org/abs/1604.03540)), Focal Loss ([1708.02002](https://arxiv.org/abs/1708.02002)), MiDaS ([2103.13413](https://arxiv.org/abs/1907.01341))

**Architecture:** Swin ([2103.14030](https://arxiv.org/abs/2103.14030)), ConvNeXt ([2201.03545](https://arxiv.org/abs/2201.03545)), ConvNeXtV2 ([2301.00808](https://arxiv.org/abs/2301.00808)), DCNv2 ([1811.11168](https://arxiv.org/abs/1811.11168)), DAT ([2201.00520](https://arxiv.org/abs/2201.00520)), DAB-DETR ([2201.12329](https://arxiv.org/abs/2201.12329)), DCNv4 ([2401.06197](https://arxiv.org/abs/2401.06197))

**Datasets:** DSEC ([2103.06011](https://arxiv.org/abs/2103.06011)), MVSEC ([1801.10202](https://arxiv.org/abs/1801.10202)), M3ED ([2210.13093](https://arxiv.org/abs/2210.13093)), TartanAir ([2003.14338](https://arxiv.org/abs/2003.14338))

**Related work:** InfiniDepth ([2601.03252](https://arxiv.org/abs/2601.03252)), EventDAM (ICCV 2025), DepthAnyEvent (ICCV 2025). No existing work targets sparse query-point depth from events.
