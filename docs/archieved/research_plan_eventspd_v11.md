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

4. **Deep per-query processing to compensate for shallow backbone** (sparse advantage). DAv2 processes every pixel through 12 ViT layers (~24 sub-layers). Our backbone has ~25 nonlinear stages to L4, and the per-query decoder adds ~21 more — affordable because we process only K queries, not all pixels. Total depth per query (~46 stages) substantially exceeds DAv2's encoder.

5. **Minimal viable complexity**. Every component must justify its existence via ablation.

### 4.2 Symbol Table

| Symbol | Meaning | Shape / Value |
|--------|---------|---------------|
| $E_t$ | Input event stream in $[t-\Delta, t)$ | — |
| $\mathcal{F}_{\text{F}^3}^{\text{ds2}}$ | Event-to-feature encoder (ds2 config) | Frozen / fine-tuned |
| $F_t$ | Dense shared feature field | 640×360×64 |
| $F_t^{(\text{stem})}$ | Stem output: Conv(k3,s2)+LN on $F_t$ | 320×180×64, stride 4 |
| $F_t^{(1)}$ | L1 fine features: 2× ConvNeXt₆₄(k=7) on stem | 320×180×64, stride 4 |
| $F_t^{(2)}$ | L2 features (3× ConvNeXt₁₂₈) | 160×90×128, stride 8 |
| $F_t^{(3)}$ | L3 features (4× SwinBlock₁₉₂) | 80×45×192, stride 16 |
| $G_t^{(4, \text{pre})}$ | L4 pre-GRU features (2× FullSelfAttn₃₈₄) | 40×22×384, stride 32 |
| $G_t^{(4)}$ | L4 features (after ConvGRU temporal fusion) | 40×22×384, stride 32 |
| $h_t$ | ConvGRU hidden state at L4 (carried across windows) | 40×22×384 |
| $K_t^{(1:2)}, V_t^{(1:2)}$ | Pre-computed per-layer KV for B2 cross-attention | 2×880×192 |
| $\hat{F}_t^{(2:4)}$ | Pre-projected feature maps for B3 ($W_V$ applied) | ×192 per level |
| $s_t, b_t$ | Main depth scale / shift | scalars |
| $s_t^{\text{ctr}}, b_t^{\text{ctr}}$ | Auxiliary depth scale / shift (optional $L_{\text{ctr}}$, training only) | scalars |
| $q = (u, v)$ | Query pixel coordinate | — |
| $f_q^{(1)}$ | Fine point feature: $\text{Bilinear}(F_t^{(1)}, q/s_1)$ | 64 |
| $c_q^{(\ell)}$ | Multi-scale center: $\text{Bilinear}(F_t^{(\ell)}, q/s_\ell)$ | 128 / 192 / 384 |
| $h_{\text{center}}$ | B4 local token: $\text{GELU}(W_{\text{loc}}[f_q^{(1)}; \phi(\mathbf{0})])$ (reuses $W_{\text{loc}}$, zero extra params) | $d = 192$ |
| $\text{pe}_q$ | Fourier positional encoding (8 freq × 2 trig × 2 dims) | 32 |
| $\mathcal{T}_{\text{unified}}$ | Unified local tokens (32 L1 + 25 L2 + 25 L3 + 9 L4) | $91 \times d$ |
| $h_{\text{ms}}^{(0)}$ | Unified attention query seed: $W_q[f_q^{(1)}; \text{pe}_q]$ | $d = 192$ |
| $h_{\text{ms}}^{(\ell)}$ | Unified attention hidden state after layer $\ell$; $h_{\text{ms}} \equiv h_{\text{ms}}^{(3)}$ | $d = 192$ |
| $h_{\text{point}}$ | Center token: $h_{\text{ms}}^{(3)}$ (direct output of unified attention, no merge MLP) | $d = 192$ |
| $h_{\text{point}}'$ | Globally-aware center token (after B2 cross-attn) | $d = 192$ |
| $\bar{\alpha}_q$ | Head-averaged B2 attention weights | 880 |
| $R_q$ | Top-$R$ attention-routed anchor set ($R = 32$) | 32 positions |
| $h_r$ | Per-anchor deformable evidence (B3 output) | $d = 192$ |
| $T_q$ | Context tokens: $[h_{\text{center}}{+}e_{\text{loc}}; c_q^{(2)}{+}e_{\text{ms2}}; c_q^{(3)}{+}e_{\text{ms3}}; c_q^{(4)}{+}e_{\text{ms4}}; h_{r_{1..32}}{+}e_{\text{deform}}]$ | 36×192 |
| $h_{\text{fuse}}$ | Fused query representation (B4 output) | $d = 192$ |
| $r_q$ | Relative depth code: MLP$(h_{\text{fuse}})$ | scalar |
| $\hat{d}_q$ | Final depth: $1 / (\text{softplus}(s_t r_q + b_t) + \varepsilon)$ | scalar |
| $\hat{d}_q^{\text{ctr}}$ | Auxiliary depth: $1 / (\text{softplus}(s_t^{\text{ctr}} r_q^{\text{ctr}} + b_t^{\text{ctr}}) + \varepsilon)$ | scalar (optional $L_{\text{ctr}}$, training only) |
| $\hat{\rho}_{\text{dense}}^{(\ell)}$ | Dense backbone auxiliary prediction at level $\ell$ (training only) | stride-$s_\ell$ maps |
| $s_t^{\text{d}\ell}, b_t^{\text{d}\ell}$ | Dense auxiliary calibration per level (training only) | scalars |
| $\mathcal{M}(u,v)$ | Spatial loss map for hard-example mining (stride 16, EMA-updated) | 80×45 |

Core dimension $d = 192$. Strides: [4, 8, 16, 32]. Channels: [64, 128, 192, 384] (ratio 1:2:3:6).

**Dual roles (center tokens):** $c_q^{(2)}$, $c_q^{(3)}$, and $c_q^{(4)}$ — the center tokens of the L2/L3/L4 3×3 grids — serve dual roles: absorbed into $h_{\text{point}}$ via unified cross-attention (compressed) AND passed as separate B4 context tokens (direct, uncompressed via $W_{\text{ms2/4}}$). This gives B4 both the unified-attention-processed representation (via $h_{\text{point}}'$) and raw per-level features for independent reasoning. Note: the old $l_q$ skip connection (attention-pooled L1 → B4) is no longer needed — the unified attention eliminates the merge MLP bottleneck that the skip was designed to bypass. B4's local token is now $h_{\text{center}}$ (query's own processed L1 feature via $W_{\text{loc}}$), providing fine-scale depth at the query pixel.

### 4.3 Algorithm A: Precompute Once Per Event Window

**Input:** Event set $E_t$.
**Output:** $\text{cache}_t = \{F_t^{(1)}, F_t^{(2)}, F_t^{(3)}, G_t^{(4)}, h_t, K_t^{(1:2)}, V_t^{(1:2)}, \hat{F}_t^{(2:4)}, s_t, b_t\}$

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
└→ Stem: Conv(64→64, k3,s2) + LN                            [320×180×64]   stride 4
   → 2× ConvNeXt_64 (k=7, GRN)                              → L1  [320×180×64]
   → Down: Conv(64→128, k2,s2) + LN                         [160×90×128]   stride 8
   → 3× ConvNeXt_128 (k=13, GRN)                            → L2  [160×90×128]
   → Down: Conv(128→192, k2,s2) + LN                        [80×45×192]
   → 4× SwinBlock_192 (window=8, shifted)                   → L3  [80×45×192]
   → Down: Conv(192→384, k2,s2) + LN                        [40×22×384]
   → 2× FullSelfAttn_384 (6 heads, d_head=64, 880 tokens)  [40×22×384]
   → ConvGRU_384 (k=3, hidden=384)                          → L4 = G_t^{(4)}  [40×22×384]
```

**Stem** (stride 2→4):
$$F_t^{(\text{stem})} = \text{LN}(\text{Conv2d}(64, 64, k{=}3, s{=}2, p{=}1)(F_t)) \quad \in \mathbb{R}^{320 \times 180 \times 64}$$
Params: ~37K. The stem is a learned downsampler — not a simple strided subsample. The 3×3 convolution with learned weights acts as an anti-aliased decimation filter, combining 2×2 neighborhoods with overlap. This is strictly better than taking F^3's stride-4 output directly (which would require retraining F^3 at 320×180 and losing fine-grained event spatial information from the 640×360 voxel grid).

**L1** — Fine depth features (2× ConvNeXt₆₄, k=7, GRN):
```
Input → DW Conv 7×7 → LN → PW Conv 1×1 (C→4C) → GELU → GRN(4C) → PW Conv 1×1 (4C→C) → + Input
```
$$F_t^{(1)} = \text{ConvNeXt}_2(\text{ConvNeXt}_1(F_t^{(\text{stem})})) \quad \in \mathbb{R}^{320 \times 180 \times 64}$$
~37K/block, ~74K total. F^3 features are event-prediction features (trained to forecast future events), not depth features — L1 must extract depth-relevant fine-scale representations from scratch. Two ConvNeXt blocks provide 4 nonlinear stages (GELU × 2 + GRN × 2) with ~14×14 receptive field per block, transforming raw F^3 temporal features into local depth geometry (edges, texture gradients, surface normals). At stride 4, each L1 pixel covers 4×4 original pixels — sufficient for depth edge localization (standard depth methods operate at stride 4+; Deformable DETR, DAv2-DPT all use stride-4 finest features). The stride gap L1→L2 is now 2× (stride 4→8) vs the old 4× gap (stride 2→8), giving the pyramid more uniform scale progression.

**Down L1→L2:**
$$\text{Down} = \text{LN}(\text{Conv2d}(64, 128, k{=}2, s{=}2)(F_t^{(1)})) \quad \in \mathbb{R}^{160 \times 90 \times 128}$$
Params: ~33K.

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
| Stem: Conv k3s2 + LN | 320×180 | 64→64 | ~2.12G | ~0.06 ms |
| L1: 2× ConvNeXt₆₄ (GRN, k=7) | 320×180 | 64 | ~4.12G | ~0.20 ms |
| Down L1→L2: Conv k2s2 + LN | 160×90 | 64→128 | ~0.47G | ~0.02 ms |
| L2: 3× ConvNeXt₁₂₈ (GRN, k=13) | 160×90 | 128 | ~6.60G | ~0.55 ms |
| Down L2→L3 | 80×45 | 128→192 | ~0.35G | ~0.02 ms |
| L3: 4× SwinBlock₁₉₂ | 80×45 | 192 | ~6.75G | ~0.50 ms |
| Down L3→L4 | 40×22 | 192→384 | ~0.26G | ~0.02 ms |
| L4: 2× FullSelfAttn₃₈₄ | 40×22 | 384 | ~4.30G | ~0.08 ms |
| L4: ConvGRU₃₈₄ (k=3) | 40×22 | 384 | ~0.8G | ~0.1 ms |
| **Backbone total** | | | **~25.8G** | **~1.55 ms** |

**FLOPs convention:** All numbers in this document use MACs (multiply-accumulate operations), the standard "paper FLOPs" convention used by Swin, ConvNeXt, and all standard benchmarks. 1 MAC = 1 multiply + 1 add.

Total backbone params: ~7,179K (6,293K CNN/Swin/SelfAttn + 886K ConvGRU).

#### A3. Pre-compute KV and $W_V$ projections

**KV for B2 cross-attention:** Each layer $\ell = 1, 2$ has its own $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times 384}$, projecting $G_t^{(4)}$ once per frame:
$$K_t^{(\ell)} = G_t^{(4)} (W_K^{(\ell)})^T, \quad V_t^{(\ell)} = G_t^{(4)} (W_V^{(\ell)})^T \quad \in \mathbb{R}^{880 \times d}, \quad \ell = 1, 2$$
Params: $4 \times 384 \times 192 = 296\text{K}$. Cost: ~0.02 ms. Per-layer KV projections follow the standard transformer decoder convention (DETR, SAM, Perceiver IO all use separate KV per layer on a shared source). Each layer can project L4 features into a different key-value space — layer 1 keys for coarse region finding, layer 2 keys for refined selection. Ablation: shared KV across layers (−148K params, −0.01 ms).

**$W_g$ anchor projection for B3:** $W_g \in \mathbb{R}^{d \times 384}$ projects L4 features for B3 conditioning. Pre-applied to all 880 positions. Params: ~74K.

**Per-level $W_V$ for B3 deformable reads:** Following Deformable DETR, value projections are pre-applied: L2 (128→192), L3 (192→192, square), L4 (384→192). Grid-sample reads already-projected 192ch features during per-query decoding. Params: ~136K. Cost: ~0.05 ms.

#### A4. Global calibration heads

**Main calibration:**
$$s_t = \text{softplus}(h_s(\text{MeanPool}(G_t^{(4)}))), \quad b_t = h_b(\text{MeanPool}(G_t^{(4)}))$$

**Auxiliary calibration (for optional $L_{\text{ctr}}$, training only):**
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

#### B1. Feature extraction and unified local cross-attention

*Extract L1 local features and multi-scale grid tokens, then process them through a single cross-attention decoder producing $h_{\text{point}}$ — replacing the old L1 attention pool → MSLCA → merge MLP pipeline.*

**Per-query feature extraction:**

L1 center feature (stride 4):
$$f_q^{(1)} = \text{Bilinear}(F_t^{(1)}, \text{Normalize}(q / s_1)) \quad \in \mathbb{R}^{64}, \quad s_1 = 4$$

Positional encoding:
$$\text{pe}_q = [\sin(2\pi \sigma_l u/W); \cos(2\pi \sigma_l u/W); \sin(2\pi \sigma_l v/H); \cos(2\pi \sigma_l v/H)]_{l=0}^{7}$$
with $\sigma_l = 2^l$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

L4 center feature:
$$c_q^{(4)} = \text{Bilinear}(G_t^{(4)}, \text{Normalize}(q / s_4)) \quad \in \mathbb{R}^{384}$$
Dual role: (1) center of L4 3×3 grid → unified token set via $W_{v4}$; (2) B4 context token via $W_{\text{ms4}}$ (§B4). Single lookup, two projections.

L1 local neighborhood ($N_{\text{loc}} = 32$) — fixed grid (5×5 minus center, 24 points) + 8 learned offsets:
$$\Delta_m = r_{\max} \cdot \tanh(W_{\text{off}}^{(m)} f_q^{(1)} + b_{\text{off}}^{(m)}), \quad m = 1, \ldots, 8, \quad r_{\max} = 6$$

For each offset $\delta$:
$$f_\delta = \text{Bilinear}(F_t^{(1)}, \tilde{q} + \delta), \quad h_\delta = \text{GELU}(W_{\text{loc}} [f_\delta; \phi(\delta)] + b_{\text{loc}}), \quad h_\delta \in \mathbb{R}^d$$

where $\phi(\delta)$ is Fourier encoding of the offset (4 freq × 2 trig × 2 dims = 16 dims, input: 64 + 16 = 80). Each $h_\delta$ carries depth-processed L1 features (2× ConvNeXt₆₄ with ~14×14 RF per block) at a specific spatial offset. The 32 tokens are passed directly into the unified cross-attention — no attention pooling.

B4 local token (reuses $W_{\text{loc}}$, zero extra params):
$$h_{\text{center}} = \text{GELU}(W_{\text{loc}} [f_q^{(1)}; \phi(\mathbf{0})] + b_{\text{loc}}) \quad \in \mathbb{R}^d$$
The query's own L1 feature through the local MLP with zero offset. Used as B4's local context token (§B4).

Extraction params: offset heads ~1K + local MLP $W_{\text{loc}}$ (80→192) ~16K = ~17K. Lookups: 33 (1 center + 32 local).

**Unified token set (91 tokens):**

The 32 L1 tokens are combined with grid-sampled multi-scale tokens into a single set for cross-attention:

| Scale | Tokens | Grid | Projection | Coverage | Embedding |
|-------|--------|------|------------|----------|-----------|
| L1 (stride 4) | 32 | local neighborhood | identity ($h_\delta$ already $d$-dim) | ±8–24 px | $e_{L1}$ + spatial via $\phi(\delta)$ |
| L2 (stride 8) | 25 | 5×5 | $W_{v2} \in \mathbb{R}^{d \times 128}$ | 40×40 px | $e_{L2}$ + RPE (25-entry) |
| L3 (stride 16) | 25 | 5×5 | identity (192ch = $d$) | 80×80 px | $e_{L3}$ + RPE (25-entry shared) |
| L4 (stride 32) | 9 | 3×3 | $W_{v4} \in \mathbb{R}^{d \times 384}$ | 96×96 px | $e_{L4}$ + RPE (inner 9 of shared) |

Token formulas:
$$t_j^{(1)} = h_{\delta_j} + e_{L1}, \quad j = 1, \ldots, 32$$
$$t_i^{(2)} = W_{v2} \, \text{Bilinear}(F_t^{(2)}, \text{Normalize}(q/s_2 + \delta_i)) + e_{L2} + \text{rpe}_i, \quad i = 1, \ldots, 25$$
$$t_i^{(3)} = \text{Bilinear}(F_t^{(3)}, \text{Normalize}(q/s_3 + \delta_i)) + e_{L3} + \text{rpe}_i, \quad i = 1, \ldots, 25$$
$$t_i^{(4)} = W_{v4} \, \text{Bilinear}(G_t^{(4)}, \text{Normalize}(q/s_4 + \delta_i)) + e_{L4} + \text{rpe}_i, \quad i = 1, \ldots, 9$$

L2/L3 offsets: $\delta_i \in \{-2,-1,0,1,2\}^2$ (25-entry RPE table shared across L2/L3). L4 offsets: $\delta_i \in \{-1,0,1\}^2$ (inner 9 entries of the shared table). L4 stays at 3×3 because full self-attention gives each token global RF — wider grids yield diminishing spatial diversity. The L4 center ($\delta = 0$) reuses the same bilinear lookup as $c_q^{(4)}$ (no extra lookup). L1 tokens carry spatial info via $\phi(\delta)$ in the local MLP — RPE would be redundant.

Combined:
$$\mathcal{T}_{\text{unified}} = [\underbrace{t_1^{(1)}; \ldots; t_{32}^{(1)}}_{L1 \text{ local}};\; \underbrace{t_1^{(2)}; \ldots; t_{25}^{(2)}}_{L2};\; \underbrace{t_1^{(3)}; \ldots; t_{25}^{(3)}}_{L3};\; \underbrace{t_1^{(4)}; \ldots; t_9^{(4)}}_{L4}] \quad \in \mathbb{R}^{91 \times d}$$

Token balance: 32 L1 (35%) : 25 L2 (27%) : 25 L3 (27%) : 9 L4 (10%). Four scale embeddings ($e_{L1}, e_{L2}, e_{L3}, e_{L4}$) distinguish token provenance.

**3-layer cross-attention decoder → $h_{\text{point}}$:**

Query seed:
$$h_{\text{ms}}^{(0)} = W_q [f_q^{(1)}; \text{pe}_q] \quad \in \mathbb{R}^d$$

$W_q \in \mathbb{R}^{d \times 96}$ projects fine L1 texture + position. No L2 in the query — L2 is accessed only through KV-side tokens, avoiding the L2–L2 shortcut (if $c_q^{(2)}$ were in the query, attention to the L2 center would produce trivially high similarity).

Per-layer KV projections ($\ell = 1, 2, 3$):
$$K_{\text{u}}^{(\ell)} = W_K^{(\ell)} \, \text{LN}_{\text{kv}}(\mathcal{T}_{\text{unified}}), \quad V_{\text{u}}^{(\ell)} = W_V^{(\ell)} \, \text{LN}_{\text{kv}}(\mathcal{T}_{\text{unified}}) \quad \in \mathbb{R}^{91 \times d}$$

$\text{LN}_{\text{kv}}$ is shared across layers (normalizes heterogeneous token scales once). Per-layer $W_K, W_V$ follow DETR/SAM/Perceiver IO convention.

3-layer decoder (Pre-LN, each layer = cross-attn + FFN with residuals):
$$h_{\text{ms}} \leftarrow h_{\text{ms}} + \text{MHCrossAttn}^{(\ell)}(Q = \text{LN}_q^{(\ell)}(h_{\text{ms}}), \; K = K_{\text{u}}^{(\ell)}, \; V = V_{\text{u}}^{(\ell)})$$
$$h_{\text{ms}} \leftarrow h_{\text{ms}} + \text{FFN}^{(\ell)}(\text{LN}_{\text{ff}}^{(\ell)}(h_{\text{ms}}))$$

6 heads, $d_{\text{head}} = 32$, FFN: $192 \to 768 \to 192$. The $1 \times 91$ attention matrix per head is trivially cheap.

Layer semantics:
- **Layer 1 (multi-scale discovery):** L1-conditioned query selects relevant evidence across all 4 scales. Six heads specialize by scale and spatial relevance.
- **Layer 2 (cross-scale refinement):** Re-attention with multi-scale awareness — "seen L4 context, now re-weight L2/L3 near this depth boundary."
- **Layer 3 (conflict resolution):** Resolves remaining multi-scale ambiguities among 91 heterogeneous tokens (analogous to B4's 3rd layer for 36 tokens).

Three layers match B4's depth for heterogeneous token processing. 

Output:
$$h_{\text{point}} = h_{\text{ms}}^{(3)} \quad \in \mathbb{R}^d$$

No merge MLP. The old architecture computed L1 and multi-scale features separately, then compressed $[h_{\text{ms}}; l_q]$ (384→192) through a merge MLP. Unified attention eliminates this bottleneck — $h_{\text{point}}$ is produced directly by joint attention over all 91 tokens. Residual chain: $h_{\text{ms}}^{(0)} \xrightarrow{+\text{unified 3L}} h_{\text{point}} \xrightarrow{+\text{B2}} h_{\text{point}}'$.

**Why unified over separate modules (L1 attn-pool + MSLCA + merge MLP):**

| Aspect | Unified | Separate |
|--------|---------|----------|
| Cross-scale reasoning | From layer 1 (L1 edge + L3 boundary → depth discontinuity) | Only after merge MLP bottleneck |
| Information flow | All 91 KV tokens preserved across 3 layers | Merge MLP: $[h_{\text{ms}}; l_q]$ 384→192 (50% discarded) |
| B4 local token | $h_{\text{center}}$ (query's own L1 feature — targeted) | $l_q$ (attention-pooled 32 neighbors — diffuse) |
| L4 for B2 | $h_{\text{point}}$ carries local L4 → B2 starts warm | Cold start (zero L4 before B2) |
| Architecture | Consistent: B1 (91), B2 (880), B4 (36) — all cross-attn | Three distinct mechanisms |

Center tokens $c_q^{(2)}, c_q^{(3)}, c_q^{(4)}$ serve dual roles: (1) absorbed into $h_{\text{point}}$ via unified cross-attention (compressed), (2) passed as uncompressed B4 context tokens via $W_{\text{ms}}$ projections (§B4). Single lookup each, two projections.

**B1 params and compute:** local extraction ~17K + $W_{v2}$ (~25K) + $W_{v4}$ (~74K) + $W_q$ (~19K) + 3× per-layer $W_Q/W_K/W_V/W_O$ (~444K) + 3× FFN (~888K) + LNs (~5K) + scale embeddings (~0.8K) + RPE (~4.8K) = **~1,480K**. 92 lookups (1 center + 32 local + 25 L2 + 25 L3 + 9 L4). ~23.1M MACs/query (dominated by 3× KV projection of 91 tokens; compute-free in memory-bound regime). The 32 extra L2/L3 lookups (vs 3×3 grids) are spatial neighbors — cache-friendly, negligible wall-clock impact.

#### B2. Global L4 cross-attention and routing

*Enrich $h_{\text{point}}$ with global scene context from all 880 L4 tokens, then route to 32 anchors for B3's deformable sampling.*

**2-layer cross-attention into 880 L4 tokens:**

Each layer $\ell = 1, 2$ (Pre-LN cross-attention + FFN with residuals):
$$h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}^{(\ell)}(Q = \text{LN}_q^{(\ell)}(h_{\text{point}}), \; K = K_t^{(\ell)}, \; V = V_t^{(\ell)})$$
$$h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}^{(\ell)}(\text{LN}^{(\ell)}(h_{\text{point}}))$$

6 heads, $d_{\text{head}} = 32$, FFN: $192 \to 768 \to 192$. $K_t^{(\ell)}, V_t^{(\ell)}$ are pre-computed per layer in Phase A — per-query cost is only Q projection, attention, and FFN. The $1 \times 880$ attention matrix per head is trivially cheap.

After 2 layers: $h_{\text{point}}' \in \mathbb{R}^d$ — enriched with global scene context.

**Attention-based routing → B3 anchors (free — no extra parameters):**

Average attention weights from layer 2 across heads, select top-$R$:
$$\bar{\alpha}_q = \frac{1}{H} \sum_{h=1}^{H} \alpha_{q,h} \quad \in \mathbb{R}^{880}, \quad R_q = \text{TopR}(\bar{\alpha}_q, R{=}32)$$

Each $r \in R_q$ maps to pixel coordinate $\mathbf{p}_r$ via L4 grid geometry (40×22, stride 32). $R = 32$ from 880 = 3.6% selection ratio. Each L4 position covers 32×32 pixels. Straight-through routing: hard top-R forward, STE backward.

**B2 cost:** ~1.4M MACs per query (2-layer cross-attention).

#### B3. Deformable multiscale read

For each anchor $r \in R_q$ (32 from B2's attention-based routing), predict sampling offsets and importance weights, then read from the multi-scale pyramid. B3's role is purely remote adaptive sampling — local context is already thoroughly covered by B1's 91-token unified cross-attention (3 layers, 4 scales). Including a query-local anchor would create an attention shortcut in B4: the local deformable token would have high similarity with $h_{\text{point}}'$ (already saturated with local information from B1), biasing B4's cross-attention toward redundant local evidence instead of the novel remote features B3 was designed to bring in. Ref: Deformable DETR (Zhu et al. ICLR 2021).

**Conditioning:**
$$\Delta\mathbf{p}_r = \mathbf{p}_r - q \quad \text{(query-to-anchor offset in original pixel coordinates)}$$
$$u_r = \text{LN}(\text{GELU}(W_u [h_{\text{point}}';\; g_r;\; \phi_{\text{B3}}(\Delta\mathbf{p}_r)] + b_u)), \quad u_r \in \mathbb{R}^d$$

where $g_r = W_g \, G_t^{(4)}[\mathbf{p}_r^{(4)}] \in \mathbb{R}^d$ is the anchor's L4 feature (pre-projected from 384ch). Input: $d + d + 32 = 416$. The conditioning tells the offset head: "I'm this query ($h_{\text{point}}'$, globally aware from B2), looking at an anchor whose content is $g_r$, located at this spatial offset." Shared across all 32 anchors.

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

**Budget:** 32 anchors × 72 samples/anchor = **2,304 deformable lookups** (all remote). Total lookups per query: 1 (L1 center) + 32 (L1 local) + 59 (L2/L3/L4 grids: 25+25+9) + 2,304 (deformable) = **2,396**.

B3 params: conditioning (416→192, ~80K) + offsets (192→144, ~28K) + weights (192→72, ~14K) + $W_O$ (192→192, ~37K) = ~159K.

#### B4. Fusion decoder (3-layer cross-attention transformer)

**Context token set (36 tokens with 5 type embeddings):**
$$T_q = [h_{\text{center}}{+}e_{\text{loc}};\; c_q^{(2)}{+}e_{\text{ms2}};\; c_q^{(3)}{+}e_{\text{ms3}};\; c_q^{(4)}{+}e_{\text{ms4}};\; h_{r_1..32}{+}e_{\text{deform}}]$$

- $h_{\text{center}} + e_{\text{loc}}$: query's own L1 feature processed through $W_{\text{loc}}$ with zero offset (from B1, zero extra params)
- $c_q^{(2)} + e_{\text{ms2}}$: L2 center (128ch → 192ch via $W_{\text{ms2}}$) — center of L2 3×3 grid (no extra lookup)
- $c_q^{(3)} + e_{\text{ms3}}$: L3 center (192ch = $d$, identity) — center of L3 3×3 grid (no extra lookup)
- $c_q^{(4)} + e_{\text{ms4}}$: L4 center (384ch → 192ch via $W_{\text{ms4}}$) — center of L4 3×3 grid (no extra lookup)
- $h_{r_1..32} + e_{\text{deform}}$: deformable evidence from 32 attention-routed anchors

Projection params: $W_{\text{ms2}}$ (128×192 ≈ 25K) + $W_{\text{ms4}}$ (384×192 ≈ 74K) ≈ ~99K. Type embeddings: 5 × $d$ = 960 params.

**KV normalization:** $T_q \leftarrow \text{LN}_{\text{kv}}(T_q)$ — normalizes heterogeneous token scales. Applied once (static $T_q$).

**3-layer transformer decoder (Pre-LN):**
$$h_{\text{point}}' \leftarrow h_{\text{point}}' + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}'), \; KV = T_q)$$
$$h_{\text{point}}' \leftarrow h_{\text{point}}' + \text{FFN}(\text{LN}(h_{\text{point}}'))$$

6 heads, $d_{\text{head}} = 32$, FFN $192 \to 768 \to 192$. Attention matrix per head: $1 \times 36$ — trivially cheap. After 3 layers: $h_{\text{fuse}} = h_{\text{point}}'$. The 3rd layer is essentially free in wall-clock time (B4 is compute-bound on 36 tokens, masked entirely by B3's memory latency) and enables higher-order cross-type reasoning: layer 1 attends to individual evidence types, layer 2 learns pairwise interactions, layer 3 resolves multi-way conflicts (e.g., deformable evidence from occluder vs occluded surface vs multi-scale center priors).

B4 params: 3 layers × (Q/K/V/O projections + FFN + LNs) ≈ ~1,334K.

**Why B4 after B2:** B2 provides global context via soft attention over all 880 L4 tokens — broad but at stride 32 only. B4 provides targeted multi-scale context via hard attention over 36 tokens carrying B3's remote deformable evidence at strides 8/16/32. Complementary: B2 = "understand the scene", B4 = "fuse detailed evidence."

#### B5. Depth prediction

**Relative depth code:**
$$r_q = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2}$$
MLP $192 \to 384 \to 1$. The wider hidden layer (2× input dim) gives the final nonlinear mapping more capacity to separate depth-relevant features from noise in $h_{\text{fuse}}$, at negligible compute cost (~74K MACs). Params: ~74K.

**Calibration and depth conversion:**
$$\rho_q = s_t \cdot r_q + b_t, \quad \hat{d}_q = \frac{1}{\text{softplus}(\rho_q) + \varepsilon}, \quad \varepsilon = 10^{-6}$$

**Center-only auxiliary (for optional $L_{\text{ctr}}$, training only, ablation #24):**
$$r_q^{\text{ctr}} = W_{\text{ctr},2} \cdot \text{GELU}(W_{\text{ctr},1} \, h_{\text{point}}^{(0)} + b_{\text{ctr},1}) + b_{\text{ctr},2}$$
MLP $192 \to 96 \to 1$ on $h_{\text{point}}^{(0)}$ (BEFORE B2). Calibrated by separate $(s_t^{\text{ctr}}, b_t^{\text{ctr}})$ — independent from the main branch's $(s_t, b_t)$. Forces center branch to remain independently informative. Params: ~19K (MLP) + ~0.8K (calibration, in A4). Only instantiated if $L_{\text{ctr}}$ is enabled.

**Uncertainty (optional, disabled by default):**
$$\sigma_q = \text{softplus}(W_\sigma \, h_{\text{fuse}} + b_\sigma) + \sigma_{\min}, \quad \sigma_{\min} = 0.01$$

#### Nonlinear depth per query

- Backbone: L1 2×ConvNeXt (2×2=4) + L2 3×ConvNeXt (3×2=6) + L3 4×Swin (4×2=8) + L4 2×SelfAttn (2×2=4) + ConvGRU (3 gates, ~3) = **~25 stages**
- Decoder: B1 unified 3×(cross-attn+FFN) (3×2=6) + B1 local MLP (1) + B2 cross-attn (2×2=4) + B3 conditioning (2) + B4 fusion (3×2=6) + B5 MLP (2) = **~21 stages**
- **Total: ~46 nonlinear stages per query**

**Comparison with DAv2-S (~24 encoder sub-layers):** Our total (~46) substantially exceeds DAv2's depth, but the nature differs. DAv2's 24 sub-layers are all global self-attention (every token sees every other token at every layer). Our backbone stages are heterogeneous: L1 is local ConvNeXt (~14px RF per block), L2 is large-kernel local (~104px RF), L3 is windowed (8×8), only L4 is global. The decoder adds 11 cross-attention stages (3×unified + 2×B2 + 3×B4 = 8 cross-attention + 3 FFN interaction stages total) plus local/conditioning stages. The sparse formulation trades uniform global context (DAv2) for deeper per-query processing that's affordable because only K queries traverse the full decoder, not all HW pixels.

---

## 5. Training

### 5.1 Loss Functions

**Data fit:**
$$L_{\text{point}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}(q) - \rho^*(q)), \quad \hat{\rho}(q) = \text{softplus}(\rho_q) + \varepsilon$$

**Scale-invariant structure:**
$$L_{\text{silog}} = \sqrt{\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q\right)^2}, \quad \delta_q = \log \hat{d}(q) - \log d^*(q)$$
Default $\lambda_{\text{var}} = 0.5$ (F^3 default). Ablate $\{0.5, 0.85, 1.0\}$.

**Dense backbone auxiliary (56× more backbone gradients, training only):**
$$L_{\text{dense}} = \sum_{\ell \in \{2,3\}} \frac{\lambda_{\text{d}\ell}}{|P_v^{(\ell)}|} \sum_{p \in P_v^{(\ell)}} \text{Huber}(\hat{\rho}_{\text{dense}}^{(\ell)}(p) - \rho^*(p))$$

where $P_v^{(\ell)}$ is the set of pixels at level $\ell$'s resolution with valid GT depth. Provides 18,000 gradient sources (14,400 at L2 + 3,600 at L3) vs $K{=}256$ from sparse queries. Gradients flow through the entire backbone (L2→L3→L4→stems), ensuring features are useful everywhere, not just at queried locations. See A5 for architecture. Ref: PSPNet (Zhao et al. CVPR 2017), Mask2Former (Cheng et al. CVPR 2022).

$\lambda_{\text{d2}} = 0.5$, $\lambda_{\text{d3}} = 0.25$. Warm-up schedule: $\lambda_{\text{dense}}$ scales from 1.0→0.25 over training (start with dense dominance when sparse routing is untrained, then fade as the query decoder matures).

**Feature distillation from DAv2 (cross-modal knowledge transfer, training only):**

$$L_{\text{feat}} = \sum_{\ell \in \{3,4\}} \frac{\lambda_{\text{f}\ell}}{|P^{(\ell)}|} \sum_{p \in P^{(\ell)}} \left(1 - \frac{\hat{F}_{\text{event}}^{(\ell)}(p) \cdot \hat{F}_{\text{DAv2}}^{(\ell)}(p)}{\|\hat{F}_{\text{event}}^{(\ell)}(p)\| \, \|\hat{F}_{\text{DAv2}}^{(\ell)}(p)\|}\right)$$

where $\hat{F}_{\text{event}}^{(\ell)}$ are our L3/L4 pyramid features projected via a connector MLP ($W_c^{(\ell)} \in \mathbb{R}^{384 \times C_\ell}$), and $\hat{F}_{\text{DAv2}}^{(\ell)}$ are frozen DAv2 ViT-S intermediate features (layer 6 for L3, layer 12 for L4) bilinearly interpolated to match spatial resolution.

**Connector MLP (per level):** $\hat{F}_{\text{event}}^{(\ell)} = W_{c2}^{(\ell)} \cdot \text{GELU}(W_{c1}^{(\ell)} \cdot F_{\text{event}}^{(\ell)})$. L3: 192→384→384 (~148K), L4: 384→384→384 (~296K). Total connector params: ~444K (training only, discarded at inference).

**Why cosine similarity:** Cosine loss matches feature directions (structural semantics) without requiring magnitude alignment across the event-RGB modality gap. Events lack color/texture but share geometric structure (edges, surfaces, depth discontinuities) with RGB features. Cosine loss captures this structural alignment. Ref: Depth AnyEvent (ICCV 2025) uses cross-modal distillation from DAv2 for event depth; EventDAM (ICCV 2025) uses sparsity-aware feature distillation; Spike-Driven Transformer (2024) shows 49% Abs Rel improvement from DINOv2 feature alignment.

**Sparsity-aware weighting:** Event features are inherently sparse — textureless/static regions produce few events and weak features. Weight the cosine loss by event density at each spatial location: $w_p = \min(n_p / \bar{n}, 2.0)$ where $n_p$ is the event count in the patch at position $p$ and $\bar{n}$ is the mean. This focuses distillation on regions where the event backbone has sufficient signal. Ref: EventDAM's SFD module.

$\lambda_{\text{f3}} = 0.1$, $\lambda_{\text{f4}} = 0.1$. Requires running frozen DAv2 forward pass on aligned RGB frames during training — cost: ~8G MACs per frame (~0.5 ms on RTX 4090), amortized across $K$ queries. Enabled in Stage 1 only (pseudo-label datasets have aligned RGB). Disabled in Stage 2 (real-only, may lack synchronized RGB).

**Core total:**
$$\mathcal{L} = L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + L_{\text{dense}} + L_{\text{feat}}$$

with $\lambda_{\text{si}} = 0.5$, $\lambda_{\text{d2}} = 0.5$, $\lambda_{\text{d3}} = 0.25$, $\lambda_{\text{f3}} = \lambda_{\text{f4}} = 0.1$. Dense warm-down: $1.0 \to 0.25$ over training. All training-only components ($L_{\text{dense}}$ heads, $L_{\text{feat}}$ connectors) discarded at inference.

#### Optional losses (disabled by default — test via ablation)

*The 4 core losses provide a strong baseline comparable to leading monocular depth methods (DAv2: 1 loss; ZoeDepth: 2; Depth AnyEvent: 3). The following losses add hyperparameters, warm-up schedules, and data-loading complexity for uncertain marginal benefit. Enable selectively based on ablation results.*

**Center auxiliary (ablation #24):**
$$L_{\text{ctr}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}^{\text{ctr}}(q) - \rho^*(q)), \quad \hat{\rho}^{\text{ctr}}(q) = \text{softplus}(s_t^{\text{ctr}} \cdot r_q^{\text{ctr}} + b_t^{\text{ctr}}) + \varepsilon$$

$\lambda_{\text{ctr}} = 0.25$ with warm-up $0 \to 0.25$ over 5 epochs. Forces pre-B2 representation ($h_{\text{point}}^{(0)}$) to be independently predictive. Demoted: with $L_{\text{dense}}$ + $L_{\text{feat}}$ providing abundant backbone gradients, center collapse is unlikely.

**Temporal consistency via warped depth (ablation #18):**

Given consecutive F^3 windows $t$ and $t{+}1$ (each 20 ms, non-overlapping), reproject depth from window $t$ into window $t{+}1$'s coordinate frame using known camera pose:

$$q' = \pi(T_{t \to t+1} \cdot \pi^{-1}(q, \hat{d}_q^{(t)}))$$
$$L_{\text{temp}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} M_q \cdot |\hat{d}_{q'}^{(t+1)} - \hat{d}_q^{(t, \text{warped})}|$$

where $\pi$ / $\pi^{-1}$ are projection / back-projection using known camera intrinsics, $T_{t \to t+1} \in SE(3)$ is the relative pose between window centers (from dataset GT poses), $\hat{d}_q^{(t, \text{warped})}$ is $\hat{d}_q^{(t)}$ transformed to the new frame's depth, and $\hat{d}_{q'}^{(t+1)}$ is the model's prediction at the reprojected pixel $q'$ in window $t{+}1$.

**Validity mask** $M_q$: (1) $q'$ within image bounds, (2) no occlusion — forward-backward check within 5%, (3) $q'$ at least 0.5 px from border. Ref: MonoDepth2 auto-masking (Godard et al. ICCV 2019).

$\lambda_{\text{temp}} = 0.1$. Requires consecutive-window loading + GT poses per batch. Demoted: high implementation cost for uncertain benefit — temporal consistency tends to emerge from good per-frame features; most SOTA monocular depth methods achieve top results without temporal training losses. Ref: ManyDepth (Watson et al. CVPR 2021), TC-Depth (Ruhkamp et al. IROS 2021).

**Calibration smoothness (ablation #18):**
$$L_{\text{cal}} = |s_t - s_{t+1}| + |b_t - b_{t+1}|$$

$\lambda_{\text{cal}} = 0.05$. Demoted: ConvGRU already provides temporal smoothness in features that feed calibration heads — if features don't jump, $(s_t, b_t)$ won't jump. If $L_{\text{temp}}$ is enabled, depth consistency already implicitly constrains calibration.

### 5.2 Training Data

| Dataset | GT type | Resolution | Density | Metric? | GT Pose? |
|---------|---------|------------|---------|---------|----------|
| **M3ED** | Real LiDAR (VLP-16) | 1280×720 | Sparse (~5–10%) | Yes | Yes (VIO) |
| **DSEC** | Real stereo + LiDAR | 640×480 | Semi-dense (~30–50%) | Yes | Yes (GPS/INS) |
| **MVSEC** | Real LiDAR + IMU + MoCap | 346×260 | Sparse | Yes | Yes (MoCap/IMU) |
| **TartanAir v2** | Synthetic (Unreal) | 640×640 | Dense (100%) | Yes | Yes (exact) |
| **M3ED pseudo** | DAv2 pseudo labels | 1280×720 | Dense (100%) | No (relative) | Yes (VIO) |

Real GT is primary — LiDAR sparsity is not a problem since we sample queries at LiDAR-valid locations. DAv2 pseudo labels provide supplementary dense coverage (used with scale-invariant losses only). All datasets provide GT camera poses, available for optional $L_{\text{temp}}$ (warped depth consistency, ablation #18).

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
- Sparse queries: $L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}}$ (LiDAR-valid) or $\lambda_{\text{si}} L_{\text{silog}}$ only (pseudo-label queries)
- Dense backbone auxiliary: $L_{\text{dense}}$ on all pixels with valid GT at L2/L3 (enabled from epoch 1)
- Feature distillation: $L_{\text{feat}}$ on L3/L4 features via frozen DAv2 teacher (enabled from epoch 1; requires aligned RGB frames, available in M3ED/DSEC/TartanAir)
- $L_{\text{dense}}$ warm-down: scale factor $1.0 \to 0.25$ over training (dense dominates early, sparse takes over)
- Hard-example mining: $\mathcal{M}$ initialized uniform, updated via EMA from epoch 1
- $K_{\text{train}} = 2048$ (8× inference budget for stronger supervision)
- ConvGRU hidden state: BPTT through 4 consecutive windows (truncated); $h_0 = 0$ per sequence
- Backbone frozen (except ConvGRU, which trains from scratch)
- Datasets: M3ED (LiDAR + pseudo + RGB), DSEC (stereo GT + RGB), TartanAir v2 (RGB available)

**Stage 2 — Metric fine-tuning (~10 epochs):**
- Real LiDAR / stereo GT only (no pseudo labels)
- Core loss: $L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + L_{\text{dense}}$ (no $L_{\text{feat}}$ — real datasets may lack synchronized RGB)
- $L_{\text{dense}}$ weight reduced to 0.1 (sparse decoder is primary, dense auxiliary is supplementary)
- $K_{\text{train}} = 2048$
- ConvGRU continues training; BPTT through 4 windows
- Optionally unfreeze F^3 backbone with 0.1× LR
- Evaluate on MVSEC outdoor_day1/day2

### 5.5 Regularization

- Attention dropout $p = 0.1$ on B1 unified, B2, and B4 cross-attention weights
- Weight decay 0.01, gradient clipping 1.0, mixed precision (bf16)

---

## 6. Runtime Analysis

### 6.1 Parameter Budget

**Phase A: Preprocessing (amortized, once per frame)**

| Component | Params |
|-----------|-------:|
| Stem Conv(64→64,k3s2) + LN | ~37K |
| L1: 2× ConvNeXt₆₄ (GRN, k=7) | ~74K |
| Down L1→L2: Conv(64→128,k2s2) + LN | ~33K |
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
| A4: Aux calibration heads $(s_t^{\text{ctr}}, b_t^{\text{ctr}})$ (optional $L_{\text{ctr}}$, train only) | ~0.8K |
| A5: Dense aux heads L2+L3 + calibration (train only) | ~0.7K |
| **Phase A total** | **~7,688K** |

**Phase B: Decoder (per-query, $d = 192$)**

| Component | Lookups | Params |
|-----------|:---:|-------:|
| B1: Center (L1) + local MLP + $h_{\text{center}}$ | 33 | ~17K |
| B1: Unified local cross-attn (91 tok, 3-layer, 6-head, per-layer KV) | 59 | ~1,463K |
| B2: L4 cross-attn (2 layers, 880 tok) | — | ~740K |
| B3: Deformable read (H=6,L=3,M=4, 32 anchors) | 2,304 | ~159K |
| B4: $W_{\text{ms2/4}}$ proj + type emb + LN | — | ~100K |
| B4: Fusion (3 layers) | — | ~1,334K |
| B5: Depth head (192→384→1) + aux MLP | — | ~93K |
| **Phase B total** | **2,396** | **~3,906K** |

**Total trainable: ~11,594K (~11.6M)** (training-only components: A5 dense aux ~0.7K, $L_{\text{feat}}$ connectors ~444K — all discarded at inference. Optional $L_{\text{ctr}}$ adds A4 aux ~0.8K + B5 center aux MLP ~19K if enabled. Inference params: ~11,573K)

**vs DAv2-S:**
- Decoder: ~3.9M vs ~3M (DPT-S head) = **comparable** (3-layer B4 fusion + 3-layer unified attention with per-layer KV account for the increase; justified by heterogeneous token fusion and compute headroom in the memory-bound regime)
- Total (excl. frozen F^3): ~11.6M vs ~25M (DAv2-S) = **2.2× smaller** (ConvGRU adds ~0.9M for temporal state — unique capability DAv2 lacks)
- Lookups: 2,396 per query vs 268K pixels (DPT output at 518×518)

### 6.2 Speed Estimate (RTX 4090, 1280×720)

**Precompute (once per window):**
$$T_{\text{precompute}} = T_{\text{F}^3}^{\text{ds2}} + T_{\text{backbone}} + T_{\text{misc}} \approx 4.5 + 1.55 + 0.1 = 6.15 \text{ ms}$$

**Per-query cost (MACs):**

| Stage | MACs | Memory reads | Est. time (μs) |
|-------|-----:|------------:|---:|
| B1: Unified local cross-attn (3-layer, 91 tok) | ~23,100K | ~92 KB (92 lookups across L1/L2/L3/L4; extra L2/L3 are cache-friendly) | ~1.0 |
| B2: L4 cross-attn (2 layers) | ~1,410K | ~3 KB (KV cached across batch) | ~0.8 |
| B3: deformable (32 remote anchors) | ~5,080K | ~3.4 MB (2,304 scattered grid_samples) | ~2.5 |
| B4: projection + fusion (3 layers) | ~9,260K | ~28 KB (36 context tokens) | ~0.6 |
| B5: depth head (192→384→1) | ~74K | — | ~0.1 |
| **Total** | **~38.9M** | **~3.5 MB** (B3 = 97%) | **~5.0** |

**Per-query bottleneck — memory, not compute:** Arithmetic intensity: 38.9M MACs / 3.5 MB ≈ 11.1 MACs/byte — far below RTX 4090 FP16 roofline (~165 MACs/byte). Pure compute: 38.9M / 165T ≈ 0.24 μs, negligible vs ~5.0 μs memory latency. B1 has 59% of MACs but only ~1.0 μs — the 3-layer cross-attention over 91 tokens is pure in-register compute once the 92 bilinear lookups (~92 KB) are fetched; the extra 32 L2/L3 lookups (vs 3×3 grids) are spatial neighbors of the existing samples, making them cache-friendly with minimal latency increase. B3 dominates wall-clock (50%) despite only 13% of MACs — its 2,304 scattered grid_samples from ~7 MB pre-projected feature maps cost ~2.5 μs. B4 has 24% of MACs but only ~0.6 μs — 3 layers of cross-attention over 36 tokens is pure in-register compute. Consequence: compute-only changes (unified attention depth/heads, KV sharing, FFN width, B4 layers) have near-zero wall-clock impact; only grid_sample count (B3/local budget) or feature channels materially change $\beta$.

**Timing model:**
$$T_{\text{decoder}}(K) = \alpha + \beta K \approx 0.30 + 0.005K \text{ ms}$$

$\alpha \approx 0.30$ ms: kernel launch + dispatch overhead (independent of $K$).
$\beta \approx 0.005$ ms/query = 5.0 μs (see per-stage breakdown above). Memory-bandwidth-bound — B3's scattered reads dominate.

**Total:**
$$T_{\text{EventSPD}}(K) \approx 6.4 + 0.005K \text{ ms}$$

| $K$ | Decoder (ms) | Total (ms) | Hz | Dense (ms) | Speedup |
|-----|---:|---:|---:|---:|---:|
| 1 | 0.31 | 6.5 | 154 | ~23 | 3.5× |
| 64 | 0.62 | 6.8 | 147 | ~23 | 3.4× |
| 256 | 1.58 | 7.8 | 128 | ~23 | **2.9×** |
| 1024 | 5.42 | 11.6 | 86 | ~23 | 2.0× |

Dense baseline (ds1): F^3 ds1 (8.3 ms) + DAv2 decoder (~15 ms) = ~23 ms. Crossover: $(23 - 6.45) / 0.005 \approx K = 3{,}310$.

**Matched-resolution dense baseline (ds2):** F^3 ds2 (4.5 ms) + DAv2 decoder on 640×360 (~4 ms) = ~8.5 ms. Crossover: $(8.5 - 6.45) / 0.005 \approx K = 410$. This is the fairer comparison — same F^3 encoder, same resolution. Our speedup narrows (1.1× at K=256) but we process only K points vs 230K pixels, so accuracy-per-FLOP should favor sparse queries. Report both baselines.

**Key insight:** Precompute (~6.15 ms) = 79% of total at K=256. F^3 ds2 (~4.5 ms) = 73% of precompute. The event encoder is the bottleneck — further speedups require lighter event encoders or voxel grid representations.

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

- **Center predictiveness (if $L_{\text{ctr}}$ enabled):** Monitor $L_{\text{ctr}}$ — decreasing = center branch independently informative
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
| 3 | Remove L1 tokens from unified attention | Does L1 neighborhood context help? Unified attention over 27 multi-scale tokens only (no L1 local). |
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
| 12b | B1 architecture | **Unified 91-token cross-attn** (default) vs (a) separate modules: L1 4-head attn-pool + MSLCA 2-layer + merge MLP (old design, −254K params), (b) unified 59-token (multi-scale only, no L1 tokens, −17K params), (c) unified 91-token with L2-in-query ($h_{\text{ms}}^{(0)} = W_1[f_q^{(1)};\text{pe}_q] + \text{GELU}(W_2 c_q^{(2)})$, +25K params — tests L2-L2 shortcut). |
| 13 | Deformable budget | $(H,L,M) \in \{(4,3,2)..\mathbf{(6,3,4)}...(6,3,6)\}$. |
| 13b | Query-local B3 anchor | **Without** (default, 32 remote-only) vs with (33 anchors: +1 query-local). B1 already covers local thoroughly (91 tokens, 4 scales); query-local anchor creates B4 attention shortcut toward redundant local evidence. |
| 14b | ConvGRU temporal state | **With ConvGRU at L4** (default) vs (a) no GRU (independent windows), (b) ConvLSTM (adds cell state, +886K params), (c) GRU at L3+L4 (multi-scale state). Ref: Depth AnyEvent-R (ICCV 2025), DERD-Net (NeurIPS 2025). |
| 14 | Unified attention configuration | **3-layer, 6-head, 91-token (5×5 L2/L3, 3×3 L4), per-layer KV** (default) vs (a) 2-layer (−~490K params), (b) 4-layer (+~490K params), (c) 4 heads $d_{\text{head}} = 48$ (matches old MSLCA head config), (d) 3×3 grids at all levels (59 tokens — tests 5×5 vs 3×3 benefit at L2/L3), (e) 5×5 at all levels including L4 (107 tokens — tests whether L4 benefits), (f) shared KV across layers (−~221K params), (g) flat concat MLP (608→192, no cross-attn — baseline). Tests unified attention depth, head count, grid size, token composition, per-layer KV, and cross-attention vs flat fusion. |

### Tier 3 — Training strategy

| # | Ablation | Question |
|---|----------|----------|
| 15 | Dense backbone auxiliary | Disable $L_{\text{dense}}$. Measures impact of 56× gradient coverage on backbone quality. |
| 16 | Train-large-K | $K_{\text{train}} \in \{256, 512, \mathbf{2048}, 4096\}$. Does 8× more queries per batch improve convergence? |
| 17 | Hard-example mining | Disable loss-map sampling (replace with random). Does focusing on hard regions help? |
| 18 | Temporal consistency | **(a) Disabled** (default) vs (b) $L_{\text{cal}}$ only, (c) same-pixel $L_{\text{temp}}$ without warping, (d) warped $L_{\text{temp}}$ + $L_{\text{cal}}$. Optional losses — test whether temporal signal improves over core-only baseline. |
| 18b | Feature distillation | (a) Disable $L_{\text{feat}}$, (b) output distillation only (pseudo-labels, no feature matching), **(c) cosine feature distillation L3+L4** (default), (d) L4 only. Ref: Depth AnyEvent, EventDAM (ICCV 2025). |
| 19 | Dense aux warm-down | Fixed $\lambda_{\text{dense}}$ vs **warm-down** (default) vs disable after Stage 1. |

### Tier 4 — Extensions

| # | Ablation | Question |
|---|----------|----------|
| 20 | B2 only (no B3/B4) | Can L4 cross-attn alone suffice? Depth head on $h_{\text{point}}'$. |
| 21 | Hash encoding vs Fourier PE | Does learned position encoding help? |
| 22 | ~~Temporal memory~~ | *(Moved to Tier 2 as #14b — now default)* |
| 23 | Uncertainty head | $\sigma_q$ + $L_{\text{unc}}$. |
| 24 | Center auxiliary loss | **(a) Disabled** (default) vs (b) enable $L_{\text{ctr}}$ ($\lambda = 0.25$, warm-up 5 epochs). Optional loss — tests whether center collapse is a real problem when $L_{\text{dense}}$ + $L_{\text{feat}}$ are active. |
| 25 | Attention dropout rate | $p \in \{0, 0.05, \mathbf{0.1}, 0.2\}$. |

### Tier 5 — Backbone architecture

| # | Ablation | Question |
|---|----------|----------|
| 26 | Backbone architecture | (a) No backbone, (b) CNN-only, (c) CNN-light, (d) Hybrid-light, **(e) Hybrid** (default), (f) Heavy. |
| 27 | L3 window size | **8** (default, requires padding), 5 (divides 80×45 cleanly). |
| 27b | L2 kernel size | 7, **13** (default), 21. DW conv cost scales with k² but is <10% of block cost. Ref: UniRepLKNet. |
| 27c | L1 depth and stride | (a) Stride-2 identity (LN+GELU on raw F^3, 640×360, +intermediate local path, +37K params — original v9 design), **(b) Stride-4 2× ConvNeXt₆₄** (default), (c) stride-4 3× ConvNeXt₆₄ (+37K params, +2G MACs). Tests whether depth-processed L1 features justify the +4.1G backbone MACs vs raw F^3 features, and optimal L1 depth. |
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
| L1 | Stem(k3,s2)+2× ConvNeXt₆₄(k=7) at stride 4 | 2× Conv(64→64,k3)+LN+GELU at stride 2 (3.2G, 0.2 ms) |
| Intermediate | — (L1 is stride 4, no separate intermediate) | 3× ConvNeXt₆₄ with GRN at stride 4 (15.6G, 1.0 ms) |
| L4 self-attn | 2 layers | 4 layers (+0.6G, +0.05 ms) |
| L2, L3, A3, A4, B1–B5 | — | Identical (RGB-SPD keeps intermediate local path for stride-4 features) |

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
| Encoder total | ~83G / 6.2 ms | ~40G / 2.8 ms |
| Decoder (K=256) | 1.6 ms | 1.6 ms (identical) |
| **System total (K=256)** | **~7.8 ms** | **~4.4 ms** |

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
