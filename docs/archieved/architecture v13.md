# Research Plan: SPD v13 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-23
Version: v13 (first trial — ConvNeXt V2-T encoder + L4 self-attention, sparse B1–B5 decoder, NYU Depth V2)

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Depth $\hat{d}_i$ at each queried pixel.

**Constraint:** No dense $H \times W$ depth decoding. Shared encoder pass once, then sparse per-query decoding:

$$T_{\text{total}}(K) = T_{\text{encode}}(I) + T_{\text{decode}}(K \mid \text{features})$$

**Baseline:** DAv2-S (ViT-S encoder + DPT dense decoder) produces dense $H \times W$ depth (~49G MACs, ~5.0ms on RTX 4090).
**Ours:** ConvNeXt V2-T encoder + sparse B1–B5 decoder → depth at K points only (~42G MACs at K=256, ~3.9ms). Faster for $K < 490$.

---

## 2. Architecture Overview

> **v13.1 revision** (post-Exp 4): Removed B4 deformable read (caused co-adaptation instability and epoch 2–3 regression). Added B3a L3 cross-attention (fills mid-scale global gap). Redesigned B5 tokens: 3 central multi-scale + 20 routed L4 = 23 tokens. Added L3 self-attention (2× FullSelfAttn₃₈₄) between Stage 3 and Stage 4 — gives L3 global context for B3a and feeds globally-informed features into Stage 4. Kept L4 self-attention (2 layers) in neck for B3b routing. 4 total self-attention layers across two resolutions. See [experiments.md](experiments.md) for evidence.

```
RGB Image (256×320 current / 480×640 target)
  │
  ▼
ConvNeXt V2-T encoder (ImageNet pre-trained, fine-tuned with 0.1× LR)
  Stage 1: H/4  × W/4  × 96   (3× ConvNeXt block, stride 4)
  Stage 2: H/8  × W/8  × 192  (3× ConvNeXt block, stride 8)
  Stage 3: H/16 × W/16 × 384  (9× ConvNeXt block, stride 16)
  ├→ 2× FullSelfAttn₃₈₄ on L3  (global context, feeds B3a + Stage 4)
  Stage 4: H/32 × W/32 × 768  (3× ConvNeXt block, input = L3-enhanced)
  │
  ▼
Projection Neck (trainable, Conv 1×1 + LN per level)
  → L1 [H/4  × W/4  × 64]    stride 4
  → L2 [H/8  × W/8  × 128]   stride 8
  → L3 [H/16 × W/16 × 192]   stride 16  (from L3 self-attn output)
  → 2× FullSelfAttn₃₈₄ → L4 [H/32 × W/32 × 384]   stride 32
  │
  ▼
Pre-compute (once per image)
  B3a L3 KV, B3b L4 KV, wg (L4→d), L2/L3/L4 projections (→d)
  │
  ▼
Per-query decoder B1–B5 (×K queries in parallel)
  B1:  91 local tokens + query seed h⁰
  B2:  2L local cross-attn (91 tokens) → h²          [local-aware]
  B3a: 2L L3 global cross-attn (N_L3 tokens) → h³ᵃ   [mid-scale global]
  B3b: 2L L4 global cross-attn (N_L4 tokens) → h³ᵇ   [coarse global + top-20 routing]
  B5:  3L fused cross-attn (23 tokens: 3 central + 20 routed) + depth head → depth
```

At 256×320: N_L3 = 320, N_L4 = 80. At 480×640: N_L3 = 1200, N_L4 = 300.

Core dimension $d = 192$. Encoder channels: [96, 192, 384, 768]. Decoder pyramid channels: [64, 128, 192, 384]. 4× self-attention layers (2 L3 + 2 L4).

---

## 3. Encoder: ConvNeXt V2-T

ConvNeXt V2-Tiny (FCMAE pre-trained on ImageNet, 83.0% top-1). Hierarchical CNN with 4 stages producing a natural multi-scale pyramid at strides [4, 8, 16, 32]. Fine-tuned with 0.1× decoder learning rate.

| Parameter | Value |
|-----------|-------|
| Architecture | ConvNeXt V2-Tiny |
| Stages | 4 (depths: [3, 3, 9, 3]) |
| Channels | [96, 192, 384, 768] |
| Strides | [4, 8, 16, 32] |
| Input size | 480×640 (NYU native, no resize) |
| Stem | Conv(3→96, k4, s4) — patchify |
| Block type | ConvNeXt V2 (DW Conv k7 + GRN + PW Conv) |
| Params | ~28.6M |
| FLOPs (480×640) | ~27.5G MACs |

**Stage outputs (480×640 input):**

```
RGB 480×640×3
  → Stem: Conv(3→96, k4, s4)                                    120×160×96
  → Stage 1: 3× ConvNeXt V2 block (DW k7 + GRN + PW, 96ch)  → 120×160×96   stride 4
  → Down: LN + Conv(96→192, k2, s2)                              60×80×192
  → Stage 2: 3× ConvNeXt V2 block (192ch)                     →  60×80×192   stride 8
  → Down: LN + Conv(192→384, k2, s2)                             30×40×384
  → Stage 3: 9× ConvNeXt V2 block (384ch)                     →  30×40×384   stride 16
  → **L3 Self-Attn: 2× FullSelfAttn₃₈₄**                     →  30×40×384   (global context)
  → Down: LN + Conv(384→768, k2, s2)                             15×20×768
  → Stage 4: 3× ConvNeXt V2 block (768ch)                     →  15×20×768   stride 32
```

Note: The encoder forward pass is split — Stage 3 output goes through L3 self-attention (trainable, decoder LR) before the Stage 3→4 downsample. Stage 4 receives globally-informed L3 features, so L4 inherits global context.

**Why ConvNeXt V2-T over other backbones:**
- Natural 4-level pyramid at strides [4, 8, 16, 32] — true multi-resolution features (no fake upsampling from single-resolution ViT)
- 40–50% faster than Swin-T on GPU at same FLOPs (pure conv, no window partition overhead)
- Higher accuracy than Swin-T (83.0% vs 81.3% IN-1K)
- GRN (Global Response Normalization) — same component proven in our v12 design
- L1–L3 need strong **local** features (our decoder samples them locally); ConvNeXt excels at this

**Source:** Pre-trained weights from `timm` (`convnextv2_tiny.fcmae_ft_in22k_in1k` or `convnextv2_tiny.fcmae_ft_in1k`).

---

## 4. Self-Attention + Projection Neck (trainable)

Two stages of self-attention provide global spatial context that ConvNeXt V2's local 7×7 convolutions cannot:

- **L3 self-attention** (2 layers at d=384): Inserted between Stage 3 and Stage 4. Gives L3 tokens global context (for B3a) and feeds globally-informed features into Stage 4 (so L4 inherits global context).
- **L4 self-attention** (2 layers at d=384): Applied after projection in the neck. Re-establishes explicit global relationships at L4 level, which Stage 4's 3 local conv blocks partially fragment. Critical for B3b routing quality.

**Why 4 total layers (not 2):** ConvNeXt V2's GRN provides channel-wise but NOT spatial global context. 2 layers at a single level is shallow — layer 1 builds pairwise relationships, layer 2 refines, but there's no capacity for complex multi-hop reasoning. 4 layers across two resolutions (L3 stride 16 + L4 stride 32) capture both mid-scale and coarse-scale global spatial patterns. Cost is trivial: ~1.5G MACs at 256×320 (~6.4G at 480×640) for both, vs ~27.5G for the encoder alone.

### 4.1 L3 Self-Attention (between Stage 3 and Stage 4)

```
Stage 3 output [H/16 × W/16 × 384, N_L3 tokens]
  → FullSelfAttn₃₈₄ layer 1 (6 heads, d_head=64, Pre-LN + FFN 384→1536→384, dropout 0.1)
  → FullSelfAttn₃₈₄ layer 2 (same)
  → L3_enhanced [H/16 × W/16 × 384]   ← every L3 token sees all positions
  ├→ projected to L3 [H/16 × W/16 × 192] (via neck)
  └→ feeds into Stage 4 downsample + blocks → L4_raw [H/32 × W/32 × 768]
```

At 256×320: N_L3 = 320 tokens. At 480×640: N_L3 = 1,200 tokens. Full self-attention on 320 tokens is trivially cheap (~0.6G MACs per layer).

**L3 self-attention params:** 2× (4 × 384² + FFN 384→1536→384 + LayerNorms) = **~3,542K**.

### 4.2 Projection Neck

1×1 convolutions project to decoder channels [64, 128, 192, 384]:

```
Stage 1 [H/4  × W/4  × 96]         → Conv(96→64,   k1) + LN → L1 [H/4  × W/4  × 64]
Stage 2 [H/8  × W/8  × 192]        → Conv(192→128,  k1) + LN → L2 [H/8  × W/8  × 128]
L3_enhanced [H/16 × W/16 × 384]    → Conv(384→192,  k1) + LN → L3 [H/16 × W/16 × 192]
Stage 4 [H/32 × W/32 × 768]        → Conv(768→384,  k1) + LN → L4_pre [H/32 × W/32 × 384]
```

**Neck params:** ~6K + ~25K + ~74K + ~296K = **~401K**.

### 4.3 L4 Self-Attention (after projection)

```
L4_pre [H/32 × W/32 × 384, N_L4 tokens]
  → FullSelfAttn₃₈₄ layer 1 (6 heads, d_head=64, Pre-LN + FFN 384→1536→384, dropout 0.1)
  → FullSelfAttn₃₈₄ layer 2 (same)
  → L4 [H/32 × W/32 × 384]   ← every L4 token sees all positions
```

At 256×320: N_L4 = 80 tokens. At 480×640: N_L4 = 300 tokens.

**Why L4 self-attn is still needed despite L3 self-attn → Stage 4:** Stage 4 has only 3 blocks of local 7×7 depthwise conv. While L4 inherits global context from L3-enhanced input, the local conv processing fragments explicit global relationships. L4 self-attention cheaply re-establishes direct token-to-token communication (80 or 300 tokens), ensuring B3b routing selects anchors based on full scene structure, not just locally-informed content.

**L4 self-attention params:** 2× (4 × 384² + FFN 384→1536→384 + LayerNorms) = **~3,542K**.

### 4.4 Summary

| Level | Resolution (480×640) | Channels | Stride | Tokens | Self-Attention |
|-------|:---:|:---:|:---:|:---:|:---|
| L1 | 120×160 | 64 | 4 | 19,200 | — |
| L2 | 60×80 | 128 | 8 | 4,800 | — |
| L3 | 30×40 | 192 | 16 | 1,200 | 2× FullSelfAttn₃₈₄ (before Stage 4) |
| L4 | 15×20 | 384 | 32 | 300 | 2× FullSelfAttn₃₈₄ (after projection) |

**Section 4 total:** Neck ~401K + L3 self-attn ~3,542K + L4 self-attn ~3,542K = **~7,485K (~7.5M)**.

---

## 5. Pre-compute (once per image)

**Input:** Pyramid $\{L_1, L_2, L_3, L_4\}$.
**Output:** $\text{cache} = \{L_1, \ldots, L_4,\; K_{\text{L3}}^{(1:2)}, V_{\text{L3}}^{(1:2)},\; K_{\text{L4}}^{(1:2)}, V_{\text{L4}}^{(1:2)},\; g,\; \hat{L}_2, \hat{L}_3, \hat{L}_4\}$

### 5.1 KV projections for B3b (L4 cross-attention)

Each B3b layer $\ell = 1, 2$ has per-layer $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times 384}$, applied to L4:

$$K_{\text{L4}}^{(\ell)} = L_4 \, (W_K^{(\ell)})^T, \quad V_{\text{L4}}^{(\ell)} = L_4 \, (W_V^{(\ell)})^T \quad \in \mathbb{R}^{N_{L4} \times d}$$

$N_{L4} = 80$ at 256×320, $300$ at 480×640. Params: $4 \times 384 \times 192 =$ **~296K**.

### 5.1b KV projections for B3a (L3 cross-attention) [NEW]

Each B3a layer $\ell = 1, 2$ has per-layer $W_{K,\text{L3}}^{(\ell)}, W_{V,\text{L3}}^{(\ell)} \in \mathbb{R}^{d \times 192}$, applied to L3:

$$K_{\text{L3}}^{(\ell)} = L_3 \, (W_{K,\text{L3}}^{(\ell)})^T, \quad V_{\text{L3}}^{(\ell)} = L_3 \, (W_{V,\text{L3}}^{(\ell)})^T \quad \in \mathbb{R}^{N_{L3} \times d}$$

$N_{L3} = 320$ at 256×320, $1200$ at 480×640. L3 channels = 192 = $d$, so projections are square. Params: $4 \times 192 \times 192 =$ **~148K**.

### 5.2 L4 projection for B5 routed tokens

$W_g \in \mathbb{R}^{d \times 384}$ pre-projects all L4 features. B5 gathers from $g$ at routed indices:

$$g = L_4 \, W_g^T \quad \in \mathbb{R}^{N_{L4} \times d}$$

Params: **~74K**.

### 5.3 Per-level projections for B5 central tokens

Conv2d $1 \times 1$ projections to $d$-dim feature maps. B5 bilinear-samples these at query coordinates:

$$\hat{L}^{(\ell)} = \text{Conv}_{1 \times 1}(L^{(\ell)}) \quad \in \mathbb{R}^{d \times H_\ell \times W_\ell}$$

| Level | Projection | Params |
|-------|-----------|-------:|
| L2 | $128 \to 192$ | ~25K |
| L3 | $192 \to 192$ | ~37K |
| L4 | $384 \to 192$ | ~74K |
| **Total** | | **~136K** |

### ~~5.4 Calibration heads~~ (removed in v13.1)

Replaced by direct log-depth prediction: $\hat{d}_q = \exp(\text{MLP}(h_{\text{fuse}}))$. No per-image scale/bias.

**Pre-compute total:** ~296K + ~148K + ~74K + ~136K = **~654K**.

---

## 6. Per-Query Decoder (B1–B5)

All steps batched over $K$ queries in parallel. Coordinate convention: `F.grid_sample(align_corners=False, padding_mode='zeros')`. Pixel coords to grid: $\text{grid} = 2p / \text{dim} - 1$.

### 6.0 Symbol Table

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $d$ | Core dimension | 192 |
| $q = (u, v)$ | Query pixel coordinate | — |
| $s_\ell$ | Effective stride of level $\ell$: $s_1 {=} 4,\; s_2 {=} 8,\; s_3 {=} 16,\; s_4 {=} 32$ | scalar |
| $f_q^{(1)}$ | L1 center feature: Bilinear($L_1$, $q / s_1$) | 64 |
| $\text{pe}_q$ | Fourier positional encoding of $(u, v)$ | 32 |
| $h^{(0)}$ | Query seed: $W_q [f_q^{(1)};\; \text{pe}_q]$ | $d$ |
| $\mathcal{T}_{\text{unified}}$ | Local tokens (32 L1 + 25 L2 + 25 L3 + 9 L4) | $91 \times d$ |
| $h^{(2)}$ | B2 output (local-aware query) | $d$ |
| $h^{(3a)}$ | B3a output (mid-scale global-aware query) | $d$ |
| $h^{(3b)}$ | B3b output (coarse global-aware query) | $d$ |
| $\bar{\alpha}_q$ | Head-averaged B3b attention weights | $N_{L4}$ |
| $R_q$ | Top-20 L4 positions from B3b routing | 20 positions |
| $c_q^{(\ell)}$ | Central token: bilinear sample of $\hat{L}^{(\ell)}$ at $q$ | $d$ |
| $g_r$ | Routed L4 token: $g[r]$ for $r \in R_q$ | $d$ |
| $\mathcal{T}_{\text{B5}}$ | $[c_q^{(2)}{+}e_{c2};\; c_q^{(3)}{+}e_{c3};\; c_q^{(4)}{+}e_{c4};\; g_{r_1}{+}e_g;\; \ldots;\; g_{r_{20}}{+}e_g]$ | $23 \times d$ |
| $h^{(5)}$ | B5 output = $h_{\text{fuse}}$ (final query representation) | $d$ |
| $\hat{d}_q$ | Predicted depth: $\exp(\text{MLP}(h_{\text{fuse}}))$ | scalar |

### 6.1 B1: Feature Extraction + Token Construction

*Extract L1 local features, build multi-scale grid tokens, construct the 91-token set $\mathcal{T}_{\text{unified}}$ and query seed $h^{(0)}$. Pure feature gathering — no cross-attention.*

**L1 center feature:**

$$f_q^{(1)} = \text{Bilinear}(L_1, \text{Normalize}(q / s_1)) \quad \in \mathbb{R}^{64}$$

**Positional encoding:**

$$\text{pe}_q = [\sin(2\pi \sigma_l u/W);\; \cos(2\pi \sigma_l u/W);\; \sin(2\pi \sigma_l v/H);\; \cos(2\pi \sigma_l v/H)]_{l=0}^{7}$$

$\sigma_l = 2^l$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

**L1 local neighborhood ($N_{\text{loc}} = 32$):** 5×5 fixed grid minus center (24 points) + 8 learned offsets + 1 center = 32 tokens.

Learned offsets:
$$\Delta_m = r_{\max} \cdot \tanh(W_{\text{off}}^{(m)} f_q^{(1)} + b_{\text{off}}^{(m)}), \quad m = 1, \ldots, 8, \quad r_{\max} = 6$$

For each offset $\delta$ (fixed grid or learned):
$$f_\delta = \text{Bilinear}(L_1, \text{Normalize}(q/s_1 + \delta))$$
$$h_\delta = \text{GELU}(W_{\text{loc}} [f_\delta;\; \phi(\delta)] + b_{\text{loc}}) \quad \in \mathbb{R}^d$$

$\phi(\delta)$: Fourier encoding of offset (4 freq × 2 trig × 2 dims = 16 dims). MLP input: $64 + 16 = 80$.

Center token reuses $W_{\text{loc}}$: $h_{\text{center}} = \text{GELU}(W_{\text{loc}} [f_q^{(1)};\; \phi(\mathbf{0})])$.

**Multi-scale grid tokens:**

| Scale | Tokens | Grid | Projection | Embedding |
|-------|:---:|------|-----------|-----------|
| L1 (64ch) | 32 | local neighborhood | identity ($h_\delta$ already $d$) | $e_{L1}$ + spatial via $\phi(\delta)$ |
| L2 (128ch) | 25 | 5×5 at $q/s_2$ | $W_{v2} \in \mathbb{R}^{d \times 128}$ | $e_{L2}$ + RPE |
| L3 (192ch) | 25 | 5×5 at $q/s_3$ | identity ($192 = d$) | $e_{L3}$ + RPE |
| L4 (384ch) | 9 | 3×3 at $q/s_4$ | $W_{v4} \in \mathbb{R}^{d \times 384}$ | $e_{L4}$ + RPE |

Grid offsets: L2/L3 $\delta_i \in \{-2,-1,0,1,2\}^2$ (25-entry RPE table, shared). L4 $\delta_i \in \{-1,0,1\}^2$ (inner 9 of shared table).

Token formulas:
$$t_j^{(1)} = h_{\delta_j} + e_{L1}, \quad j = 1, \ldots, 32$$
$$t_i^{(2)} = W_{v2} \, \text{Bilinear}(L_2, \text{Normalize}(q/s_2 + \delta_i)) + e_{L2} + \text{rpe}_i$$
$$t_i^{(3)} = \text{Bilinear}(L_3, \text{Normalize}(q/s_3 + \delta_i)) + e_{L3} + \text{rpe}_i$$
$$t_i^{(4)} = W_{v4} \, \text{Bilinear}(L_4, \text{Normalize}(q/s_4 + \delta_i)) + e_{L4} + \text{rpe}_i$$

**Unified token set:**
$$\mathcal{T}_{\text{unified}} = [\underbrace{t_1^{(1)}; \ldots; t_{32}^{(1)}}_{L1};\; \underbrace{t_1^{(2)}; \ldots; t_{25}^{(2)}}_{L2};\; \underbrace{t_1^{(3)}; \ldots; t_{25}^{(3)}}_{L3};\; \underbrace{t_1^{(4)}; \ldots; t_9^{(4)}}_{L4}] \quad \in \mathbb{R}^{91 \times d}$$

**Query seed:**
$$h^{(0)} = W_q [f_q^{(1)};\; \text{pe}_q] \quad \in \mathbb{R}^d, \quad W_q \in \mathbb{R}^{d \times 96}$$

**B1 params:** offsets ~1K + $W_{\text{loc}}$ ~16K + $W_{v2}$ ~25K + $W_{v4}$ ~74K + $W_q$ ~19K + scale/type embeddings ~1K + RPE ~5K = **~142K**. Lookups: 92 (1 center + 32 local + 25 L2 + 25 L3 + 9 L4).

### 6.2 B2: Local Multi-Scale Cross-Attention (2 layers, 91 tokens)

*Process $h^{(0)}$ through 2 layers of cross-attention over $\mathcal{T}_{\text{unified}}$ (91 tokens).*

**Per-layer KV ($\ell = 1, 2$):**
$$K^{(\ell)} = W_K^{(\ell)} \, \text{LN}_{\text{kv}}(\mathcal{T}_{\text{unified}}), \quad V^{(\ell)} = W_V^{(\ell)} \, \text{LN}_{\text{kv}}(\mathcal{T}_{\text{unified}}) \quad \in \mathbb{R}^{91 \times d}$$

$\text{LN}_{\text{kv}}$ shared across B2 layers. Per-layer $W_K, W_V$ follow DETR/SAM convention.

**Pre-LN decoder layer ($\ell = 1, 2$):**
$$h^{(\ell)} \leftarrow h^{(\ell-1)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h^{(\ell-1)}),\; K{=}K^{(\ell)},\; V{=}V^{(\ell)})$$
$$h^{(\ell)} \leftarrow h^{(\ell)} + \text{FFN}^{(\ell)}(\text{LN}_{\text{ff}}^{(\ell)}(h^{(\ell)}))$$

6 heads, $d_{\text{head}} = 32$, FFN: $192 \to 768 \to 192$ (GELU activation).

**Output:** $h^{(2)} \in \mathbb{R}^d$ — local-aware query.

**B2 params:** 2× per-layer ($W_Q/W_K/W_V/W_O$ ~148K + FFN ~296K) + LNs ~3K = **~891K**.

### 6.3a B3a: L3 Mid-Scale Global Cross-Attention (2 layers) [NEW in v13.1]

*Enrich $h^{(2)}$ with mid-scale global context from all $N_{L3}$ L3 tokens. Fills the information gap between local features (B2, ~7px radius) and coarse global (B3b, stride 32). No routing — purely for enriching h.*

**Motivation:** After B2, h has only local multi-scale features. B3b reads L4 at stride 32 — very coarse. L3 (stride 16) carries mid-scale information: object extent, boundary patterns, relative positioning. These are critical for distinguishing mid-range (3–5m) from far (5–10m) depths.

**2-layer cross-attention into L3** using pre-computed KV from Section 5.1b:

$$h^{(3a)} \leftarrow h^{(2)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h),\; K{=}K_{\text{L3}}^{(\ell)},\; V{=}V_{\text{L3}}^{(\ell)})$$
$$h^{(3a)} \leftarrow h^{(3a)} + \text{FFN}^{(\ell)}(\text{LN}^{(\ell)}(h^{(3a)}))$$

6 heads, $d_{\text{head}} = 32$. All layers use SDPA (no attention weights needed — no routing). $N_{L3} = 320$ at 256×320, $1200$ at 480×640.

**B3a params:** 2× ($W_Q/W_O$ ~74K + FFN ~296K) + LNs ~2K = **~740K** (L3 KV counted in pre-compute).

> **Scalability note:** At 480×640, $N_{L3} = 1200$, which is 4× the L4 token count. Still feasible with shared-KV broadcast and SDPA, but if VRAM is tight, can apply spatial pooling (e.g., 2×2 avg pool → 300 tokens) or route a subset.

### 6.3b B3b: Global L4 Cross-Attention + Routing (2 layers)

*Enrich $h^{(3a)}$ with coarse global scene context from all $N_{L4}$ L4 tokens (each with global RF from L4 self-attention), then route top-20 anchors for B5.*

**2-layer cross-attention into L4** using pre-computed KV from Section 5.1:

$$h^{(3b)} \leftarrow h^{(3a)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h),\; K{=}K_{\text{L4}}^{(\ell)},\; V{=}V_{\text{L4}}^{(\ell)})$$
$$h^{(3b)} \leftarrow h^{(3b)} + \text{FFN}^{(\ell)}(\text{LN}^{(\ell)}(h^{(3b)}))$$

6 heads, $d_{\text{head}} = 32$. Per-query cost is only Q projection + attention + FFN (KV pre-computed).

**Output:** $h^{(3b)} \in \mathbb{R}^d$ — globally-aware query.

**Attention-based routing (zero extra params):** Head-average attention weights from B3b layer 2, select top-$R$:

$$\bar{\alpha}_q = \frac{1}{H} \sum_{h=1}^{H} \alpha_{q,h} \quad \in \mathbb{R}^{N_{L4}}, \quad R_q = \text{Top-}R(\bar{\alpha}_q, R{=}20)$$

At 256×320: $R = 20$ from $N_{L4} = 80$ = 25% selection. At 480×640: $R = 20$ from $300$ = 6.7%. Indices used by B5 to gather routed L4 tokens.

**B3b params:** 2× ($W_Q/W_O$ ~74K + FFN ~296K) + LNs ~2K = **~740K** (KV counted in pre-compute).

### ~~6.4 B4: Deformable Multi-Scale Read~~ (REMOVED in v13.1)

> **Removed.** B4 caused co-adaptation instability: B3 routing → B4 learned offsets → B5 attention formed a three-way moving target. Evidence:
> - Exp 3–4: regression at epoch 2–3 (not seen in B1–B3-only Exp 1–2)
> - B4 bootstrapping problem: early random routing → noisy deformable tokens → B5 learns to ignore them → gradient dies → self-reinforcing loop
> - B5 token imbalance: 91 local (redundant with B2) outnumbered 32 deformable 3:1
>
> Replaced by: B5 central tokens (multi-scale skip connection, stable) + B5 routed L4 tokens (simple gather, no learned offsets). See Section 6.5.
>
> **Lookups per query:** reduced from 2,396 (92 B1 + 2,304 B4) to **95** (92 B1 + 3 B5 central).

### 6.5 B5: Fused Cross-Attention + Depth Head (3 layers, 23 tokens) [v13.1]

*Refine $h^{(3b)}$ through 3 layers of cross-attention over 23 carefully-constructed tokens: 3 multi-scale central tokens (skip connection) + 20 routed L4 tokens (global structural context).*

#### B5 Token Construction (in SPD forward, no learned params except type embeddings)

**Central tokens (3):** Bilinear sample from pre-computed $\hat{L}^{(\ell)}$ (Section 5.3) at query coordinates. These act as **multi-scale skip connections** — giving B5 direct access to raw features at the query location, bypassing the information compression in B2→B3a→B3b.

$$c_q^{(2)} = \text{BilinearSample}(\hat{L}_2,\; q / s_2) + e_{c2} \quad \in \mathbb{R}^d$$
$$c_q^{(3)} = \text{BilinearSample}(\hat{L}_3,\; q / s_3) + e_{c3} \quad \in \mathbb{R}^d$$
$$c_q^{(4)} = \text{BilinearSample}(\hat{L}_4,\; q / s_4) + e_{c4} \quad \in \mathbb{R}^d$$

Note: these use **different projections** from B1's local tokens (PreCompute's Conv2d vs TokenConstructor's Linear). No redundancy.

**Routed L4 tokens (20):** Simple gather from pre-computed $g$ (Section 5.2) at B3b routing indices:

$$g_r = g[r] + e_g \quad \text{for each } r \in R_q, \quad \in \mathbb{R}^d$$

No learned offsets, no deformable sampling. Stable signal from the start of training.

**B5 token set:**
$$\mathcal{T}_{\text{B5}} = [c_q^{(2)};\; c_q^{(3)};\; c_q^{(4)};\; g_{r_1};\; \ldots;\; g_{r_{20}}] \quad \in \mathbb{R}^{23 \times d}$$

4 type embeddings: $e_{c2}, e_{c3}, e_{c4}, e_g \in \mathbb{R}^d$ (initialized to zero).

| Token type | Count | Source | Purpose |
|-----------|:---:|--------|---------|
| L2 central | 1 | bilinear from $\hat{L}_2$ at $q$ | Fine texture/edges at query |
| L3 central | 1 | bilinear from $\hat{L}_3$ at $q$ | Mid-scale features at query |
| L4 central | 1 | bilinear from $\hat{L}_4$ at $q$ | Coarse structure at query |
| Top-20 routed L4 | 20 | gather from $g$ | Global structural context |

#### B5 Cross-Attention (3 layers)

**Per-layer KV:**
$$K^{(\ell)} = W_K^{(\ell)} \, \text{LN}_{\text{kv2}}(\mathcal{T}_{\text{B5}}), \quad V^{(\ell)} = W_V^{(\ell)} \, \text{LN}_{\text{kv2}}(\mathcal{T}_{\text{B5}}) \quad \in \mathbb{R}^{23 \times d}$$

**3-layer Pre-LN decoder:**
$$h^{(\ell)} \leftarrow h^{(\ell-1)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h^{(\ell-1)}),\; K{=}K^{(\ell)},\; V{=}V^{(\ell)})$$
$$h^{(\ell)} \leftarrow h^{(\ell)} + \text{FFN}^{(\ell)}(\text{LN}_{\text{ff}}^{(\ell)}(h^{(\ell)}))$$

6 heads, $d_{\text{head}} = 32$, FFN: $192 \to 768 \to 192$. Input to layer 1 is $h^{(3b)}$ from B3b.

**Output:** $h_{\text{fuse}} = h^{(5)} \in \mathbb{R}^d$

**Residual chain:** $h^{(0)} \xrightarrow{+\text{B2: 2L, 91}} h^{(2)} \xrightarrow{+\text{B3a: 2L, L3}} h^{(3a)} \xrightarrow{+\text{B3b: 2L, L4}} h^{(3b)} \xrightarrow{+\text{B5: 3L, 23}} h^{(5)} \to \hat{d}_q$

#### Depth Head

$$\text{log\_depth} = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2} \quad (192 \to 384 \to 1)$$
$$\hat{d}_q = \exp(\text{log\_depth})$$

**Bias initialization:** $b_{r2}$ initialized to $\ln(2.5) \approx 0.916$, centering initial predictions at ~2.5m (NYU median). This equalizes the learning burden: reaching 10m requires $\Delta{=}+1.38$ from init, reaching 0.7m requires $\Delta{=}-1.27$ (ratio 1.09:1, was 6.4:1 with bias=0).

**B5 params:** 3× ($W_Q/W_K/W_V/W_O$ + FFN) ~1,332K + depth MLP ~74K + LNs ~4K + type embeddings ~1K = **~1,431K**.

---

## 7. Training

### 7.1 Dataset: NYU Depth V2

| Property | Value |
|----------|-------|
| Training | 795 scenes, ~24K image-depth pairs (Eigen split) |
| Test | 654 images (Eigen split) |
| Resolution | 480×640 (native, no resize) |
| GT depth | Dense (Kinect), max ~10m |
| Eval crop | Eigen center crop: rows [45:471], cols [41:601] |

### 7.2 Loss Functions (v13.1)

$$\mathcal{L} = L_{\text{silog}}$$

**Scale-invariant log loss (SILog):**
$$L_{\text{silog}} = \sqrt{\frac{1}{K} \sum_q \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{K} \sum_q \delta_q\right)^2}, \quad \delta_q = \log \hat{d}_q - \log d_q^*$$

$\lambda_{\text{var}} = 0.5$.

> **v13.1 changes:** Dropped $L_{\text{point}}$ (Huber on inverse depth). Exp 3 showed $L_{\text{point}}$ conflicts with SILog — gradient $\propto 1/\text{depth}^2$ biases toward near depths, causing oscillation at epoch 3. SILog alone provides uniform gradients in log-space across the full depth range. $\lambda_{\text{var}}$ lowered from 0.85 to 0.5 (less scale penalty, better for metric depth).

### 7.3 Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate (decoder + neck) | 1×10⁻⁴ (cosine decay to 1×10⁻⁶) |
| Learning rate (encoder) | 1×10⁻⁵ (0.1× decoder LR) |
| Weight decay | 0.01 |
| Batch size | 8 |
| $K_{\text{train}}$ (queries per image) | 256 |
| Epochs | 30 |
| Precision | bf16 mixed precision |
| Gradient clipping | 1.0 |
| Attention dropout | 0.1 (B2, B3, B5, L4 self-attn) |
| Encoder | ConvNeXt V2-T, fine-tuned with 0.1× LR |
| Trainable | All (~40.7M: encoder 28.6M + self-attn/neck 7.5M + precompute 0.7M + decoder 3.9M) |

**Query sampling:** Uniform random over all pixels (NYU has dense GT). Optional: 70% random + 30% high-gradient regions (depth edges) for better edge quality.

**Augmentation:** Random horizontal flip, random color jitter (brightness 0.2, contrast 0.2, saturation 0.2, hue 0.1), random resize crop (scale 0.8–1.2).

---

## 8. Evaluation

**Metrics (computed at K query points on Eigen test set):**

| Metric | Formula |
|--------|---------|
| AbsRel | $\frac{1}{K}\sum \|d^* - \hat{d}\| / d^*$ |
| RMSE | $\sqrt{\frac{1}{K}\sum (d^* - \hat{d})^2}$ |
| $\delta < 1.25^n$ | $\% \text{ of } \max(\hat{d}/d^*, d^*/\hat{d}) < 1.25^n, \; n{=}1,2,3$ |

**Baselines:**
- **DAv2-S dense:** ViT-S encoder + DPT decoder → dense depth → sample at query points (accuracy ceiling).
- **Bilinear + MLP:** Bilinear lookup from L1 features + 2-layer MLP depth head (simplest sparse baseline).

**Evaluation protocol:** For each test image, sample $K \in \{1, 64, 256, 1024\}$ query points uniformly. Report accuracy at each $K$. Compare accuracy vs DAv2-S dense baseline (which is $K$-independent). All evaluations use Eigen center crop and depth range $[10^{-3}, 10]$ m.

---

## 9. Parameter Budget (v13.1)

| Component | v13.0 | v13.1 | Delta |
|-----------|------:|------:|------:|
| **Encoder: ConvNeXt V2-T** (fine-tuned 0.1× LR) | **28,600K** | **28,600K** | — |
| Projection Neck (Section 4.2) | 401K | 401K | — |
| **L3 Self-Attention, 2× FullSelfAttn₃₈₄ (Section 4.1)** | — | **3,542K** | **+3,542K** |
| L4 Self-Attention, 2× FullSelfAttn₃₈₄ (Section 4.3) | 3,542K | 3,542K | — |
| B3b L4 KV projections (Section 5.1) | 296K | 296K | — |
| **B3a L3 KV projections (Section 5.1b)** | — | **148K** | **+148K** |
| $W_g$ L4 projection (Section 5.2) | 74K | 74K | — |
| Per-level projections (Section 5.3) | 136K | 136K | — |
| ~~Calibration heads (Section 5.4)~~ | 0.8K | — | -0.8K |
| B1: Feature extraction + tokens | 142K | 142K | — |
| B2: Local cross-attn (2 layers) | 891K | 891K | — |
| **B3a: L3 global cross-attn (2 layers)** | — | **740K** | **+740K** |
| B3b: L4 global cross-attn (2 layers) | 740K | 740K | — |
| ~~B4: Deformable read + fusion~~ | 159K | — | **-159K** |
| B5: Fused cross-attn (3L) + depth head + type emb | 1,430K | 1,431K | +1K |
| **Neck + self-attn + precompute + decoder** | **~7,812K** | **~12,083K** | **+4,271K** |
| **Total model** | **~36,412K (~36.5M)** | **~40,683K (~40.7M)** | **+4,271K** |

**What changed:** +L3 self-attn (3,542K) + B3a (L3 cross-attn, 888K) − B4 (DeformableRead, 159K) − Calib (0.8K) = net +4,271K.

**Efficiency comparison vs dense baseline (DAv2-S):**

$$T_{\text{ours}}(K) \approx 2.6 + 0.005K \;\text{ms}, \quad T_{\text{DAv2-S}} \approx 5.0 \;\text{ms}$$

| K | Ours (ms) | DAv2-S dense (ms) | Speedup |
|:---:|---:|---:|---:|
| 16 | 2.7 | 5.0 | 1.9× |
| 64 | 2.9 | 5.0 | 1.7× |
| 256 | 3.9 | 5.0 | **1.3×** |
| 490 | 5.0 | 5.0 | 1.0× (break-even) |

*Timings estimated for RTX 4090. B4 removal (no grid_sample loops) and B5 token reduction (123→23) should offset B3a addition. Net latency impact expected to be neutral or slightly faster.*

---

## 10. Implementation

### 10.1 File Structure

```
src/spd/
├── models/
│   ├── spd.py                  # Main model: encode + neck + pre-compute + decode
│   ├── encoder.py              # ConvNeXt V2-T wrapper (load pre-trained, extract stages)
│   ├── pyramid_neck.py         # Channel projection + L4 self-attention (Section 4)
│   ├── precompute.py           # KV projections, W_V projections, calibration (Section 5)
│   ├── query_encoder.py        # B1: Feature extraction + token construction (Section 6.1)
│   ├── local_cross_attn.py     # B2: 2-layer cross-attn over 91 tokens (Section 6.2)
│   ├── global_cross_attn.py    # B3a (no routing) + B3b (L4 + routing) (Sections 6.3a/6.3b)
│   ├── deformable_read.py      # ~~B4~~ (REMOVED in v13.1, kept for reference)
│   └── fused_decoder.py        # B5: 3-layer fused cross-attn + depth head (Section 6.5)
├── data/
│   ├── nyu_dataset.py          # NYU Depth V2 data loading + augmentation
│   └── query_sampler.py        # Query point sampling strategies
├── utils/
│   ├── losses.py               # L_point, L_silog
│   └── metrics.py              # AbsRel, RMSE, delta thresholds
├── train.py                    # Training loop
└── evaluate.py                 # Evaluation on Eigen test set
```

### 10.2 Dependencies

- PyTorch ≥ 2.0 (for `F.grid_sample`, `F.scaled_dot_product_attention`)
- `timm` — ConvNeXt V2-T pre-trained weights (`convnextv2_tiny.fcmae_ft_in1k`)
- NYU Depth V2 dataset (download via standard script)

### 10.3 Build Order (v13.1)

Steps 1–5 already implemented and validated. Steps 6–7 revised for v13.1 architecture.

| Step | What to build | Status |
|------|--------------|--------|
| 1 | ConvNeXt V2-T encoder + projection neck + L4 self-attn | ✅ Done |
| 2 | Pre-compute module (B3b L4 KV, wg, proj_l2/l3/l4) | ✅ Done |
| 3 | B1 token construction (91 tokens + seed) | ✅ Done |
| 4 | B2 local cross-attn | ✅ Done |
| 5 | B3b global L4 cross-attn + routing | ✅ Done (was B3) |

**v13.1 new steps:**

| Step | What to build | How to validate |
|------|--------------|-----------------|
| 6 | L3 Self-Attention: 2× FullSelfAttn₃₈₄ between Stage 3 and Stage 4 (Section 4.1) | Modify encoder to split forward (stages 1–3, then L3 self-attn, then stage 4). Verify L3_enhanced shape [B, 384, H/16, W/16]. Verify L4 shape unchanged. |
| 7 | Pre-compute: add L3 KV projections (Section 5.1b) | Verify `K_l3_0`, `V_l3_0` shapes [B, N_L3, 192]. At 256×320: N_L3=320. |
| 8 | B3a: L3 global cross-attn (2 layers, no routing) | Verify h shape unchanged [B, K, 192]. Use `GlobalCrossAttnNoRouting`. |
| 9 | B5 token construction: 3 central + 20 routed | Verify $\mathcal{T}_{\text{B5}}$ shape [B, K, 23, 192]. Central: bilinear from proj maps. Routed: gather from g. |
| 10 | B5 depth head bias init + FusedDecoder with 23 tokens | Verify initial pred mean ≈ 2.5m. Remove B4 from SPD. |
| 11 | Full pipeline: B1→B2→B3a→B3b→B5, SILog loss | Train 10 epochs. Target: AbsRel < 0.22 (beat Exp 3's 0.231), no regression. Monitor pred range — should reach >6m. |

**Key changes from v13.0 build order:**
- **L3 self-attention** (2 layers, d=384) inserted between Stage 3 and Stage 4
- L4 self-attention (2 layers, d=384) retained in neck — 4 total self-attn layers
- ~~B4 deformable sampling~~ → removed
- ~~B5 with 123 fused tokens~~ → 23 tokens (3 central + 20 routed)
- ~~$L_{\text{point}} + L_{\text{silog}}$~~ → $L_{\text{silog}}$ only ($\lambda_{\text{var}} = 0.5$)
- B3 routing: top-32 → top-20
- Depth head bias: 0 → $\ln(2.5)$
