# Research Plan: SPD v14 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-24
Version: v14 (v13.1 → MSDA decoder + Spatial Canvas + Q2Q + no B5 + dense auxiliary loss + learnable depth scale)

> **v14 revision** (post-Exp 7): Major decoder redesign addressing six bottlenecks:
> 1. **Rigid local sampling** — B1 used fixed 5×5/3×3 grids at L2/L3/L4. Fix: Multi-Scale Deformable Attention (MSDA) with learned per-head per-level offsets (Deformable DETR, Zhu et al. 2021).
> 2. **No inter-query communication** — queries were isolated predictions with no spatial consistency. Fix: Q2Q self-attention inside every decoder layer (Mask2Former pattern).
> 3. **No dense-like local neighborhood** — DPT gets spatial smoothness via Conv3×3 at every fusion stage; our queries predict with no local spatial prior. Fix: Spatial Canvas — an evolving dense feature map at L3 resolution with scatter-gather-smooth loop at every decoder layer, providing DPT-like neighborhood context.
> 4. **Sparse encoder gradient** — encoder gets gradient from only K=256 query points. Fix: dense auxiliary depth heads on L2/L3 features.
> 5. **Log-depth head output ceiling** — MLP output capped at ~1.55 (pred max ~4.7m vs GT ~10m). Fix: learnable output scale parameter.
> 6. **B5 redundancy** — B5's skip connections and routed-token re-reading are subsumed by MSDA (reads all levels every layer) + B3a/B3b (full global attention). Fix: remove B5 entirely, depth head directly after B3b.
>
> **Paper references:** MSDA follows Deformable DETR (Zhu et al. 2021): per-head per-level learned offsets with predicted attention weights. Q2Q follows DETR (Carion 2020) and Mask2Former (Cheng 2022) where cross-attn → self-attn → FFN is the standard decoder layer. Spatial Canvas is inspired by DPT's progressive spatial smoothing, adapted for sparse queries via scatter-gather operations. Dense aux losses follow DETR's intermediate supervision.

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Depth $\hat{d}_i$ at each queried pixel.

**Constraint:** No dense $H \times W$ depth decoding. Shared encoder pass once, then sparse per-query decoding:

$$T_{\text{total}}(K) = T_{\text{encode}}(I) + T_{\text{decode}}(K \mid \text{features})$$

**Baseline:** DAv2-S (ViT-S encoder + DPT dense decoder) produces dense $H \times W$ depth (~49G MACs, ~5.0ms on RTX 4090).
**Ours:** ConvNeXt V2-T encoder + sparse MSDA decoder → depth at K points only (~42G MACs at K=256, ~3.9ms). Faster for $K < 490$.

---

## 2. Architecture Overview

> **v13.1 base** (post-Exp 4): B4 removed, B3a L3 cross-attn added, B5 redesigned (23 tokens), L3 self-attn added. See [experiments.md](experiments.md).
>
> **v14 changes** (post-Exp 7): Replace B2+B5 with MSDA decoder (multi-scale deformable attention + Q2Q self-attention at every layer). Keep B3a/B3b for full global attention. Remove B5 (absorbed by MSDA + B3a/B3b). +Dense auxiliary loss, +learnable depth scale. Addresses 0.23 AbsRel ceiling.

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
  → L2 [H/8  × W/8  × 128]   stride 8    ──→ [Dense Aux L2: Conv(128→1), SILog on 32×40]
  → L3 [H/16 × W/16 × 192]   stride 16   ──→ [Dense Aux L3: Conv(192→1), SILog on 16×20]
  │
  ▼
2× FullSelfAttn₃₈₄ on L4_pre → L4 [H/32 × W/32 × 384]   stride 32
  │
  ▼
Pre-compute (once per image)
  MSDA value maps: V_L1(L1→d), V_L2(L2→d), V_L3(=L3), V_L4(L4→d)
  B3a L3 KV, B3b L4 KV
  │
  ▼
Per-query decoder (×K queries in parallel)
  B1:   multi-scale seed h⁰ from all 4 neck levels at center + Fourier PE
  │
  Spatial Canvas F⁰ = L3 [H/16 × W/16 × d]   ← shared dense context map
  │
  3 × MSDA DecoderLayer:
  │  ├── Multi-Scale Deformable Cross-Attn (h reads V_L1..V_L4 at learned offsets)
  │  ├── Canvas Read-Write-Smooth (h reads from F, writes back, DWConv3×3 smooths F)
  │  ├── Q2Q Self-Attn (K queries communicate, spatial PE on Q/K)
  │  └── FFN
  │
  B3a:  2 × [L3 cross-attn + Canvas + Q2Q + FFN]    [mid-scale global]
  B3b:  2 × [L4 cross-attn + Canvas + Q2Q + FFN]    [coarse global]
  │
  Final canvas read: h_final = h + W_final · bilinear(F⁷, pos)
  Depth head: MLP(h_final) × exp(s) → log_depth → exp → depth
```

At 256×320: N_L3 = 320, N_L4 = 80. At 480×640: N_L3 = 1200, N_L4 = 300.

Core dimension $d = 192$. Encoder channels: [96, 192, 384, 768]. Decoder pyramid channels: [64, 128, 192, 384]. ConvNeXt V2 blocks already provide extensive local spatial mixing (DW Conv k7 stacked 3–9× per stage). 4× self-attention layers (2 L3 + 2 L4) for global context. MSDA: 6 heads, 4 levels, 4 points per head per level = 96 sampling points per query per layer. Multi-scale seed: all 4 neck levels sampled at query center → 800-d → project to d. Spatial Canvas: dense $d$-dim map at L3 resolution, initialized from $L_3$, updated via scatter-gather-smooth at every decoder layer — provides DPT-like neighborhood context. All 7 decoder layers (3 MSDA + 2 B3a + 2 B3b) use the same 4-sublayer pattern: cross-attn → canvas read-write-smooth → Q2Q self-attn → FFN.

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

**Input:** Projected pyramid $\{L_1, L_2, L_3, L_4\}$ (from Sections 4.1–4.3: neck projection + L3/L4 self-attention).
**Output:** $\text{cache} = \{V_{L1}, V_{L2}, V_{L3}, V_{L4},\; K_{\text{L3}}^{(1:2)}, V_{\text{L3}}^{(1:2)},\; K_{\text{L4}}^{(1:2)}, V_{\text{L4}}^{(1:2)}\}$

### 5.1 MSDA value feature maps [NEW in v14]

Project neck feature levels to $d$-dim feature maps. MSDA decoder layers bilinear-sample from these at learned offset positions. ConvNeXt V2's DW Conv k7 (stacked 3–9× per stage) already provides extensive local spatial mixing in each feature position.

$$V_{L\ell} = \text{Conv}_{1 \times 1}(L_\ell) \quad \in \mathbb{R}^{d \times H_\ell \times W_\ell}$$

| Level | Input channels | Projection | Params |
|-------|:---:|-----------|-------:|
| L1 | 64 | Conv(64→192, k1) | ~12K |
| L2 | 128 | Conv(128→192, k1) | ~25K |
| L3 | 192 | identity ($C_{L3} = d$) | 0 |
| L4 | 384 | Conv(384→192, k1) | ~74K |
| **Total** | | | **~111K** |

> **Shared across MSDA layers:** Value projections are computed once and reused by all MSDA decoder layers. Per-layer differentiation comes from each layer's own offset predictions and attention weights. This is more efficient than per-layer value projections (which would cost $3 \times 111\text{K} = 333\text{K}$).

### 5.2 KV projections for B3b (L4 cross-attention)

Each B3b layer $\ell = 1, 2$ has per-layer $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times 384}$, applied to L4:

$$K_{\text{L4}}^{(\ell)} = L_4 \, (W_K^{(\ell)})^T, \quad V_{\text{L4}}^{(\ell)} = L_4 \, (W_V^{(\ell)})^T \quad \in \mathbb{R}^{N_{L4} \times d}$$

$N_{L4} = 80$ at 256×320, $300$ at 480×640. Params: $4 \times 384 \times 192 =$ **~296K**.

### 5.3 KV projections for B3a (L3 cross-attention)

Each B3a layer $\ell = 1, 2$ has per-layer $W_{K,\text{L3}}^{(\ell)}, W_{V,\text{L3}}^{(\ell)} \in \mathbb{R}^{d \times 192}$, applied to L3:

$$K_{\text{L3}}^{(\ell)} = L_3 \, (W_{K,\text{L3}}^{(\ell)})^T, \quad V_{\text{L3}}^{(\ell)} = L_3 \, (W_{V,\text{L3}}^{(\ell)})^T \quad \in \mathbb{R}^{N_{L3} \times d}$$

$N_{L3} = 320$ at 256×320, $1200$ at 480×640. L3 channels = 192 = $d$, so projections are square. Params: $4 \times 192 \times 192 =$ **~148K**.

### ~~5.4 W_g and per-level projections for B5~~ (REMOVED in v14)

> **Removed.** B5 is eliminated in v14 — its skip connections and routed-token re-reading are absorbed by MSDA (reads all levels every layer) and B3a/B3b (full global attention). The pre-computed $W_g$ (74K) and per-level B5 central token projections (136K) are no longer needed. MSDA value maps (Section 5.1) replace this functionality.

**Pre-compute total (v14):** ~111K + ~296K + ~148K = **~555K** (was ~654K in v13.1, saved ~99K by removing B5-related projections).

---

## 6. Per-Query Decoder [v14: MSDA + B3a/B3b]

All steps batched over $K$ queries in parallel. Coordinate convention: `F.grid_sample(align_corners=True)`. Pixel coords to grid: $\text{grid} = 2p / (\text{dim} - 1) - 1$.

### 6.0 Symbol Table

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $d$ | Core dimension | 192 |
| $H$ | Number of attention heads | 6 |
| $d_h$ | Per-head dimension ($d / H$) | 32 |
| $L$ | Number of feature levels for MSDA | 4 |
| $N_{\text{pts}}$ | Sampling points per head per level | 4 |
| $q = (u, v)$ | Query pixel coordinate | — |
| $s_\ell$ | Effective stride of level $\ell$: $s_1{=}4,\; s_2{=}8,\; s_3{=}16,\; s_4{=}32$ | scalar |
| $p_q^{(\ell)}$ | Reference point at level $\ell$: $q / s_\ell$, normalized to $[-1, 1]$ | 2 |
| $L_\ell$ | Projected feature map at level $\ell$ (Section 4.2) | $C_\ell \times H_\ell \times W_\ell$ |
| $V_{L\ell}$ | Pre-computed $d$-dim value map at level $\ell$ (Section 5.1), projected from $L_\ell$ | $d \times H_\ell \times W_\ell$ |
| $f_q^{(\ell)}$ | Center feature at level $\ell$: Bilinear($L_\ell$, $p_q^{(\ell)}$) | $C_\ell$ |
| $\text{pe}_q$ | Fourier positional encoding of $(u, v)$ | 32 |
| $h^{(0)}$ | Multi-scale query seed: $\text{LN}(W_{\text{seed}} [f_q^{(1)}; f_q^{(2)}; f_q^{(3)}; f_q^{(4)}; \text{pe}_q])$ | $d$ |
| $\text{pos}_q$ | Spatial PE for Q2Q: $W_{\text{pos}} \, \text{pe}_q$ | $d$ |
| $h^{(n)}$ | Query representation after MSDA layer $n$ | $d$ |
| $h^{(3a)}$ | Output after B3a (mid-scale global-aware) | $d$ |
| $h^{(3b)}$ | Output after B3b (coarse global-aware) | $d$ |
| $F^{(n)}$ | Spatial Canvas after decoder layer $n$ (Section 6.5). $F^{(0)} = L_3$ | $d \times H_c \times W_c$ |
| $r_q^{(n)}$ | Canvas read: $\text{bilinear}(F^{(n-1)}, p_q^{(L3)})$ — local neighborhood context | $d$ |
| $h_{\text{final}}$ | $h^{(3b)} + W_{\text{final}} \cdot \text{bilinear}(F^{(7)}, p_q^{(L3)})$ — canvas-enhanced final repr. | $d$ |
| $\hat{d}_q$ | Predicted depth: $\exp(\text{MLP}(h_{\text{final}}) \cdot \exp(s))$ | scalar |

### 6.1 B1: Multi-Scale Query Seed [REDESIGNED in v14]

*Sample features from ALL 4 pyramid levels at the query center, concatenate with Fourier PE, project to seed $h^{(0)}$. The seed is immediately meaningful — it carries coarse-to-fine context (L4 semantics → L1 fine detail) before any decoder processing. ConvNeXt V2's stacked DW Conv k7 blocks (3–9× per stage) already provide extensive local spatial context at every position.*

> **v14 change from v13.1:** B1 no longer constructs 91 multi-scale grid tokens, nor does it sample from a single level. Instead, it samples the center point from all 4 neck feature levels in their native channel dimensions, giving the seed multi-scale information. MSDA then focuses entirely on learning WHERE to sample (offsets), since WHAT's at center is already known.

**Multi-scale center features (from projected pyramid, Section 4.2):**
$$f_q^{(\ell)} = \text{Bilinear}(L_\ell,\; p_q^{(\ell)}) \quad \in \mathbb{R}^{C_\ell}$$

| Level | Channels | What it provides |
|-------|:---:|---|
| $f_q^{(1)}$ from L1 | 64 | Fine edges, textures, local geometry (DW k7 ×3 at stride 4) |
| $f_q^{(2)}$ from L2 | 128 | Mid-level patterns, small objects (DW k7 ×3 at stride 8) |
| $f_q^{(3)}$ from L3 | 192 | Object parts, spatial layout + global context (from L3 self-attn) |
| $f_q^{(4)}$ from L4 | 384 | Scene semantics, room structure + global context (from L4 self-attn) |

Each ConvNeXt V2 stage applies DW Conv k7 stacked 3–9 times, providing extensive local spatial mixing at every feature position before the decoder sees them.

**Positional encoding:**
$$\text{pe}_q = [\sin(2\pi \sigma_l u/W);\; \cos(2\pi \sigma_l u/W);\; \sin(2\pi \sigma_l v/H);\; \cos(2\pi \sigma_l v/H)]_{l=0}^{7}$$
$\sigma_l = 2^l$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

**Multi-scale query seed:**
$$h^{(0)} = \text{LN}\!\left(W_{\text{seed}} \,[f_q^{(1)};\; f_q^{(2)};\; f_q^{(3)};\; f_q^{(4)};\; \text{pe}_q]\right) \quad \in \mathbb{R}^d$$

$$W_{\text{seed}} \in \mathbb{R}^{d \times (64 + 128 + 192 + 384 + 32)} = \mathbb{R}^{192 \times 800}$$

Input is $64 + 128 + 192 + 384 + 32 = 800$ dimensions (native channel dims from all 4 neck levels + 32-dim Fourier PE). The projection compresses multi-scale information into a single $d$-dim vector. LayerNorm stabilizes the initial representation for downstream attention layers.

> **Why sample native-dim features (not $d$-dim value maps):** The MSDA value maps (Section 5.1) project all levels to $d = 192$ for multi-head attention. But the seed benefits from richer input — L4's full 384-dim representation carries more semantic information than its 192-dim projection. Sampling from neck features ($L_\ell$) before value projection preserves maximum information for seed construction.

**Spatial PE for Q2Q** (shared across all decoder layers):
$$\text{pos}_q = W_{\text{pos}} \, \text{pe}_q \in \mathbb{R}^d, \quad W_{\text{pos}} \in \mathbb{R}^{d \times 32}$$

**B1 params (v14):** $W_{\text{seed}}$ ~154K + LN ~0.4K + $W_{\text{pos}}$ ~6K = **~160K** (was ~142K in v13.1, ~49K in earlier v14 draft).

### 6.2 MSDA Decoder (3 layers) [NEW in v14 — replaces B2 + B5]

> **Replaces** B2 (local cross-attn over fixed tokens) and B5 (fused cross-attn over central + routed tokens).
>
> **Key idea from Deformable DETR** (Zhu et al. 2021): Instead of constructing tokens then cross-attending to them, the attention mechanism *itself* samples directly from feature maps at learned positions. Each attention head independently predicts WHERE to sample (offsets) and HOW MUCH to weight each sample (attention weights), then aggregates the sampled values. No QK dot-product — weights are directly predicted from the query representation.
>
> Each decoder layer follows the Mask2Former pattern: **deformable cross-attn → Q2Q self-attn → FFN** (Cheng 2022).

#### 6.2.1 MSDA cross-attention mechanism (step by step)

Given one query at pixel $(u, v)$ with current representation $h \in \mathbb{R}^d$:

**Step 1 — Reference points.** Compute where $(u,v)$ falls on each feature level:
$$p_q^{(\ell)} = \left(\frac{2u/s_\ell}{W_\ell - 1} - 1,\;\; \frac{2v/s_\ell}{H_\ell - 1} - 1\right) \quad \text{for } \ell = 1,\ldots,4$$

These are fixed grid-sample coordinates (computed once in B1, reused by all layers).

**Step 2 — Predict offsets.** Each head predicts $N_{\text{pts}}=4$ 2D offsets at each of the $L=4$ levels:
$$\Delta = W_{\text{off}}(\text{LN}(h)) \in \mathbb{R}^{H \times L \times N_{\text{pts}} \times 2}$$

$6 \times 4 \times 4 = 96$ sampling positions total. Offsets are in normalized $[-1, 1]$ coordinates. $W_{\text{off}}$ is initialized with small weights so initial offsets $\approx 0$ (samples near the reference point).

**Step 3 — Predict attention weights.** Each head predicts a scalar weight for each of its $L \times N_{\text{pts}} = 16$ sampling positions:
$$A = \text{softmax}\!\bigl(W_{\text{attn}}(\text{LN}(h))\bigr) \in \mathbb{R}^{H \times 16}$$

Softmax is per-head over the 16 positions — each head distributes attention across 4 levels × 4 points. This is NOT a QK dot-product; the weights are directly predicted from $h$.

**Step 4 — Sample and aggregate.** For each head $i$:

$$\text{out}_i = \sum_{\ell=1}^{4}\;\sum_{m=1}^{4} A_{i,(\ell,m)} \;\cdot\; \text{Bilinear}\!\bigl(V_{L\ell},\;\; p_q^{(\ell)} + \Delta_{i,\ell,m}\bigr)\bigl[i \cdot d_h : (i{+}1) \cdot d_h\bigr]$$

Each head reads only its own $d_h = 32$ channel slice from the $d$-dimensional value map. One bilinear sample from the full $d$-dim map, then slice.

**Step 5 — Output projection + residual:**
$$h \leftarrow h + W_O \;\text{concat}(\text{out}_1, \ldots, \text{out}_H)$$

**Per query per layer:**

| | |
|------|------:|
| Heads | $H = 6$ |
| Levels sampled | $L = 4$ (L1, L2, L3, L4) |
| Points per head per level | $N_{\text{pts}} = 4$ |
| **Total bilinear samples** | **96** |
| Attention weights per head | 16 (softmax over $L \times N_{\text{pts}}$) |

#### 6.2.2 Per-layer architecture: MSDA cross-attn → canvas → Q2Q self-attn → FFN

Each MSDA decoder layer $n = 1, 2, 3$:

```
h ──→ LN ──→ MSDA Cross-Attn (96 deformable samples from V_L1..V_L4) ──→ +residual
  ──→ LN ──→ Canvas Read-Write-Smooth (read F, fuse, scatter h back, DWConv3×3) ──→ +residual
  ──→ LN ──→ Q2Q Self-Attn (K queries attend to each other, spatial PE on Q/K) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

Canvas interaction (Section 6.5) is inserted between cross-attention and Q2Q. After MSDA reads from static value maps, the canvas provides complementary information: what nearby queries have decoded so far. Q2Q then reconciles local canvas context with global inter-query consistency.

**Q2Q self-attention** (inter-query communication):

$$Q = \text{LN}(h) + \text{pos}_q, \quad K = \text{LN}(h) + \text{pos}_q, \quad V = \text{LN}(h)$$
$$h \leftarrow h + W_O^{\text{q2q}} \cdot \text{MHA}(Q, K, V)$$

Spatial PE $\text{pos}_q = W_{\text{pos}} \, \text{pe}_q$ added to Q and K only (not V) — attention is position-aware, values are pure content. $W_{\text{pos}}$ is shared across all MSDA and B3a/B3b layers.

> **Why offsets evolve per layer:** Each layer predicts offsets from the *current* $h$, not fixed positions. Layer 1 explores broadly (h ≈ seed). Layer 2 focuses (h has multi-scale context). Layer 3 fine-tunes (h is nearly final).
>
> **Per-head specialization:** Some heads naturally focus on L1 fine detail, others on L3/L4 structure. The softmax over 16 positions per head allows each head to allocate its attention budget across levels independently.

#### 6.2.3 MSDA decoder params (per layer)

| Component | Params |
|-----------|-------:|
| $W_{\text{off}}$: $d \to H \cdot L \cdot N_{\text{pts}} \cdot 2 = 192$ | ~37K |
| $W_{\text{attn}}$: $d \to H \cdot L \cdot N_{\text{pts}} = 96$ | ~18K |
| $W_O$ (MSDA output): $d \to d$ | ~37K |
| LN (cross) | ~0.4K |
| Canvas: $W_{\text{read}}^{(n)}$ ($d \to d$) + $W_{\text{gate}}^{(n)}$ ($d \to 1$) + LN | ~37K |
| Q2Q $W_Q, W_K, W_V, W_O$: $4 \times d^2$ | ~148K |
| LN (Q2Q) | ~0.4K |
| FFN: $d \to 4d \to d$ | ~296K |
| LN (FFN) | ~0.4K |
| **Per layer total** | **~574K** |

**3 layers total:** ~1,722K. Plus shared $W_{\text{pos}}$ (~6K) from B1. Canvas shared params ($W_{\text{write}}$, DWConv, $W_{\text{final}}$) counted in Section 6.5.5.

---

### 6.3 B3a: L3 Mid-Scale Global Cross-Attention (2 layers)

*Full attention over ALL $N_{L3}$ L3 tokens. MSDA samples only 4 points per head at L3 — it cannot replace full attention over 320 (or 1,200) L3 tokens for scene-level relationships (object boundaries, room layout).*

Each B3a layer follows the same 4-sublayer pattern: **cross-attn → canvas → Q2Q self-attn → FFN**.

```
h ──→ LN ──→ Cross-Attn (Q=h, KV=pre-computed L3 KV, all N_L3 tokens) ──→ +residual
  ──→ LN ──→ Canvas Read-Write-Smooth (Section 6.5) ──→ +residual
  ──→ LN ──→ Q2Q Self-Attn (K queries communicate, shared spatial PE) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

**Cross-attention** uses pre-computed KV from Section 5.3 (per-layer $K_{\text{L3}}^{(\ell)}, V_{\text{L3}}^{(\ell)}$):

$$h \leftarrow h + W_O^{(\ell)} \cdot \text{MHA}(Q{=}\text{LN}(h),\; K{=}K_{\text{L3}}^{(\ell)},\; V{=}V_{\text{L3}}^{(\ell)})$$

**Q2Q self-attention** (same mechanism as MSDA layers, shared $W_{\text{pos}}$):

$$Q = \text{LN}(h) + \text{pos}_q, \quad K = \text{LN}(h) + \text{pos}_q, \quad V = \text{LN}(h)$$
$$h \leftarrow h + W_O^{\text{q2q}} \cdot \text{MHA}(Q, K, V)$$

After B3a reads all L3 tokens, Q2Q lets queries reconcile this new global context with each other — e.g., nearby queries that received conflicting mid-scale information can resolve inconsistencies.

6 heads, $d_{\text{head}} = 32$. All layers use SDPA. $N_{L3} = 320$ at 256×320, $1200$ at 480×640.

**B3a params:** 2× ($W_Q/W_O$ cross-attn ~74K + canvas $W_{\text{read}}/W_{\text{gate}}$/LN ~37K + Q2Q $W_Q/W_K/W_V/W_O$ ~148K + FFN ~296K + LNs ~1K) = **~1,112K** (L3 KV counted in pre-compute, canvas shared params in Section 6.5.5).

> **Scalability note:** At 480×640, $N_{L3} = 1200$. Still feasible with shared-KV broadcast and SDPA. If VRAM is tight, can apply spatial pooling (2×2 avg pool → 300 tokens).

### 6.4 B3b: Global L4 Cross-Attention (2 layers)

*Full attention over ALL $N_{L4}$ L4 tokens (each with global RF from L4 self-attention). Same layer pattern as B3a.*

```
h ──→ LN ──→ Cross-Attn (Q=h, KV=pre-computed L4 KV, all N_L4 tokens) ──→ +residual
  ──→ LN ──→ Canvas Read-Write-Smooth (Section 6.5) ──→ +residual
  ──→ LN ──→ Q2Q Self-Attn (K queries communicate, shared spatial PE) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

Uses pre-computed KV from Section 5.2 (per-layer $K_{\text{L4}}^{(\ell)}, V_{\text{L4}}^{(\ell)}$). Same 4-sublayer pattern as B3a: cross-attn → canvas → Q2Q → FFN.

6 heads, $d_{\text{head}} = 32$. Per-query cost is only Q projection + attention + FFN (KV pre-computed).

**Output:** $h^{(3b)} \in \mathbb{R}^d$ — the final query representation, fed to the depth head.

> **Routing removed in v14:** In v13.1, B3b extracted top-20 L4 routing indices for B5's routed tokens. With B5 removed, routing has no consumer. B3b now purely enriches h via full global attention.

**B3b params:** 2× ($W_Q/W_O$ cross-attn ~74K + canvas $W_{\text{read}}/W_{\text{gate}}$/LN ~37K + Q2Q $W_Q/W_K/W_V/W_O$ ~148K + FFN ~296K + LNs ~1K) = **~1,112K** (KV counted in pre-compute, canvas shared params in Section 6.5.5).

### 6.5 Spatial Canvas: Evolving Dense-Local Context [NEW in v14]

*DPT's depth predictions are spatially smooth because every fusion stage applies Conv3×3, making each pixel's representation a function of its neighbors' evolving, prediction-ready features. In SPD, queries predict in isolation — MSDA reads static feature maps, Q2Q is global (not local), and the depth head receives a single d-dim vector with no spatial smoothing. This is the root cause of the 0.23 AbsRel ceiling across all experiments (Exp 3: 0.231, Exp 5: 0.230, Exp 7: 0.239).*

*The Spatial Canvas provides the missing piece: a shared dense feature map that accumulates decoder-evolved query information and is spatially smoothed at every stage, giving each query access to its neighborhood's prediction-ready representations — exactly what DPT's Conv3×3 achieves, adapted for sparse queries.*

> **B4** was removed in v13.1. **B5** is removed in v14 — both absorbed by MSDA + B3a/B3b (see Section 6.2). The Spatial Canvas replaces them as the mechanism for dense spatial context.

#### 6.5.1 DPT ↔ SPD Correspondence

| DPT (dense) | SPD without canvas | SPD with Spatial Canvas |
|---|---|---|
| Dense feature map at each fusion stage | Static pre-computed value maps, never updated | Canvas $F$: initialized from $L_3$, updated every decoder layer |
| Conv3×3 at each fusion stage | No spatial smoothing between queries | DWConv3×3 on canvas after each scatter |
| Each pixel reads from 3×3 neighbors' evolved features | Each query reads from learned offsets (MSDA) or global tokens (B3a/B3b) | Each query reads from canvas (spatially-smoothed neighborhood of evolved decoder features) |
| Final Conv3×3 head smooths predictions | MLP on isolated $h$ per query | MLP on $h$ + final canvas read (spatially-smoothed prediction-ready features) |
| Features evolve: each stage's Conv3×3 output feeds the next | Static: all 7 layers read the same $V_{L\ell}$ maps | Evolving: scatter writes decoded $h$ back → DWConv propagates → next layer reads updated canvas |

**Critical distinction from encoder's local mixing:** ConvNeXt V2's DW Conv k7 operates on **raw encoder features** — it is static preprocessing. The Spatial Canvas operates on **decoded, prediction-ready query representations** that evolve at every layer. This mirrors DPT's Conv3×3 which smooths **prediction-ready fused features**, not raw encoder outputs.

#### 6.5.2 Architecture

**Canvas resolution: L3 (stride 16)**

At 256×320: $H_c \times W_c = 16 \times 20 = 320$ positions. With $K=256$ queries, ~80% of canvas positions are within one stride of a query — nearly dense. At 480×640: $30 \times 40 = 1200$ positions, 21% query coverage. L3 is the sweet spot: fine enough for local detail (16px spacing), coarse enough for effective scatter coverage.

**Canvas initialization:**

$$F^{(0)} = L_3 \in \mathbb{R}^{d \times H_c \times W_c}$$

Initialized from projected L3 features, which already carry extensive local context (from ConvNeXt's DW Conv k7 ×9) and global context (from L3 self-attention). The canvas starts meaningful, not random.

**Per decoder layer $n$ (all 7 layers: 3 MSDA + 2 B3a + 2 B3b):**

The canvas interaction is inserted as a sublayer between cross-attention and Q2Q self-attention:

```
Step 1: LN → Cross-Attn → +residual                          (existing)
Step 2: LN → Canvas Read-Write-Smooth → +residual             [NEW]
Step 3: LN → Q2Q Self-Attn → +residual                        (existing)
Step 4: LN → FFN → +residual                                  (existing)
```

**Step 2 detail — Canvas Read-Write-Smooth:**

```
Canvas Read:
  r_q = bilinear(F^(n-1), p_q^(L3))              ∈ R^d     ← gather neighborhood context

Gated Fusion:
  gate_q = sigmoid(W_gate^(n) · h)                ∈ R^1     ← per-query relevance gate
  h ← h + gate_q · (W_read^(n) · r_q)                       ← incorporate local context

Canvas Write (scatter):
  w_q = W_write · h                               ∈ R^d     ← project to canvas space
  F^(n) = scatter_add(F^(n-1), p_q^(L3), w_q)               ← write evolved h to canvas

Canvas Smooth:
  F^(n) ← F^(n) + DWConv3×3(GELU(DWConv3×3(F^(n))))        ← spatial propagation (residual)
```

**Scatter operation:** Nearest-neighbor assignment — each query writes to the closest L3 grid position. When multiple queries map to the same position, their contributions are added (effectively averaging via the residual). Positions with no queries retain their previous value from $F^{(n-1)}$. Implementation: flatten canvas to $[B, d, H_c W_c]$, compute flat indices $\lfloor v/s_3 \rfloor \cdot W_c + \lfloor u/s_3 \rfloor$, use `scatter_add_`.

**Canvas smooth:** 2× DWConv3×3 with GELU and residual. Effective RF = 5×5 per smooth step. After 7 decoder layers: accumulated effective RF = 5 + 2×6 = 17×17 at L3 resolution, covering $17 \times 16 = 272$ pixels at input resolution. Spatial propagation radiates information from scattered query positions to their neighborhoods — even positions with NO query receive propagated context from nearby queries.

**Why the canvas is truly evolving (not static):**
1. Layer 1: $h \approx$ seed → canvas gets seed-level representations → DWConv smooths coarse depth patterns
2. Layer 3: $h$ has multi-scale MSDA context → canvas gets richer representations → DWConv smooths more refined patterns
3. Layer 5: $h$ has global L3 context from B3a → canvas gets scene-aware representations
4. Layer 7: $h$ has global L4 context from B3b → canvas gets final prediction-ready representations
5. Depth head: reads from fully-evolved canvas → depth prediction incorporates spatially-smoothed, globally-informed neighborhood

Each layer reads a DIFFERENT canvas state (because prior layers scattered and smoothed). The same bilinear read position yields different values at each layer — analogous to how DPT's Conv3×3 receives different input features at each fusion stage.

**Why gate is important:** A query on a depth edge should NOT be smoothed toward the wrong side. The gate $\text{sigmoid}(W_{\text{gate}}^{(n)} \cdot h)$ allows the model to learn: "when $h$ indicates I'm on a depth discontinuity, gate down the canvas contribution." Per-layer gates let early layers be more open (coarse smoothing helpful) and later layers be more selective (fine edges matter).

#### 6.5.3 Final Canvas Read (before depth head)

After the last B3b layer writes to and smooths the canvas, one final read feeds into the depth head:

$$h_{\text{final}} = h^{(3b)} + W_{\text{final}} \cdot \text{bilinear}(F^{(7)},\; p_q^{(L3)})$$

$$\hat{d}_q = \text{depth\_head}(h_{\text{final}})$$

This final read is the SPD analog of DPT's final Conv3×3 output head. The depth prediction now incorporates:
- $h^{(3b)}$: the query's own decoded representation (multi-scale + global context)
- $F^{(7)}$ at $p_q$: the **spatially-smoothed, evolved** neighborhood — what nearby queries' decoded representations look like after 7 rounds of scatter + DWConv propagation

Without the canvas, $h^{(3b)}$ is an isolated prediction. With the canvas, it's a **neighborhood-informed** prediction.

#### 6.5.4 Interaction with Q2Q and MSDA

| Mechanism | What it provides | Spatial structure |
|---|---|---|
| MSDA cross-attn | Multi-scale features from value maps at **learned** positions | Deformable — points can be anywhere |
| Spatial Canvas | Evolved decoder features from **fixed local** neighborhood | Dense local grid (L3 resolution) |
| Q2Q self-attn | Inter-query consistency across **all** $K$ queries | Global — no spatial bias |

These three mechanisms are complementary, not redundant:
- **MSDA** tells the query "what features are at multi-scale positions around you" (from static encoder maps)
- **Canvas** tells the query "what your nearby queries have decoded so far" (from evolving decoder representations)
- **Q2Q** tells the query "what ALL queries across the whole image think" (global consistency)

Together, they give each query exactly what DPT gives each pixel: local spatial context + multi-scale features + scene-wide consistency.

#### 6.5.5 Spatial Canvas Parameters

| Component | Params | Notes |
|-----------|-------:|-------|
| $W_{\text{read}}^{(n)}$: $d \to d$, 7 layers | 7 × 36,864 = **258K** | Per-layer (each layer reads different aspects) |
| $W_{\text{gate}}^{(n)}$: $d \to 1$, 7 layers | 7 × 193 = **1.4K** | Per-layer scalar gate |
| LN (canvas), 7 layers | 7 × 384 = **2.7K** | Per-layer LayerNorm |
| $W_{\text{write}}$: $d \to d$, shared | **37K** | Shared across layers (always writing $h$ to canvas) |
| 2× DWConv3×3 ($d$ channels), shared | **3.8K** | Shared across layers (spatial smoothing is the same op) |
| $W_{\text{final}}$: $d \to d$ | **37K** | Final read before depth head |
| **Total** | **~340K** | ~8.5% of decoder, ~0.8% of total model |

**Compute per layer:** Canvas read (bilinear gather, $K \times d$) + gated fusion + write projection ($2 \times K \times d^2 \approx 19$M MACs) + DWConv smooth ($9 \times d \times H_c \times W_c \approx 0.6$M at 256×320). 7 layers + final read: ~140M MACs total — **1.5% of encoder MACs**.

**VRAM:** Canvas $F$ is $16 \times 20 \times 192 = 61$K values (~120KB in bf16) at 256×320. At 480×640: $30 \times 40 \times 192 = 230$K values (~460KB). Negligible.

#### 6.5.6 Connection to Experimental Failures

| Experiment failure | How Spatial Canvas addresses it |
|---|---|
| 0.23 AbsRel ceiling (Exp 3, 5, 7) | Spatially-smoothed predictions reduce per-query noise — each prediction is now a function of its neighborhood |
| Pred max capped at 5–6m | Far-depth queries reinforce each other via canvas propagation — one query reaching 8m helps neighbors reach 8m too |
| "Dense models get spatial smoothness for free" (experiments.md L186) | Canvas provides exactly this: DWConv3×3 at every stage, operating on evolved decoder features |
| Overfitting after 2–3 epochs | Canvas acts as implicit spatial regularization — isolated overfitted predictions get smoothed toward neighbors |
| No inter-query spatial consistency | Canvas + DWConv provides LOCAL consistency (complementing Q2Q's GLOBAL consistency) |

**Residual chain (v14 with canvas):**
$$h^{(0)} \xrightarrow[\text{cross + canvas + Q2Q + FFN}]{\text{3× MSDA}} h^{(3)} \xrightarrow[\text{cross + canvas + Q2Q + FFN}]{\text{B3a: 2L, L3}} h^{(3a)} \xrightarrow[\text{cross + canvas + Q2Q + FFN}]{\text{B3b: 2L, L4}} h^{(3b)} \xrightarrow[\text{+ canvas final read}]{} h_{\text{final}} \to \hat{d}_q$$

Every stage uses the same 4-sublayer pattern: **cross-attn → canvas → Q2Q → FFN**. Total: 7 cross-attn + 7 canvas + 7 Q2Q + 7 FFN = **28 sublayers** (was 21).

### 6.6 Depth Head [standalone in v14]

**Final canvas read** (Section 6.5.3) — incorporates spatially-smoothed neighborhood:
$$h_{\text{final}} = h^{(3b)} + W_{\text{final}} \cdot \text{bilinear}(F^{(7)},\; p_q^{(L3)})$$

**Depth prediction:**
$$\text{raw} = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{final}} + b_{r1}) + b_{r2} \quad (192 \to 384 \to 1)$$
$$\text{log\_depth} = \text{raw} \cdot \exp(s), \quad s = \text{nn.Parameter}(\mathbf{0})$$
$$\hat{d}_q = \exp(\text{log\_depth})$$

**Learnable output scale:** Exp 7 showed the MLP output capped at ~1.55 across all val images (pred max stuck at 4.73m, log(10m) = 2.30 needed). A single learnable scalar $s$ (initialized to 0, so $\exp(s) = 1.0$ at init) lets the optimizer expand the output range. When $s$ reaches ~0.4 ($\exp = 1.5$), the range expands from $[-0.5, 1.55]$ to $[-0.75, 2.33]$ — covering the full NYU [0.7m, 10m] range.

**Bias initialization:** $b_{r2}$ initialized to $\ln(2.5) \approx 0.916$, centering initial predictions at ~2.5m (NYU median).

**Depth head params:** MLP ~74K + scale ~0.001K = **~74K**.

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

### 7.2 Loss Functions (v14)

$$\mathcal{L} = L_{\text{silog}} + \lambda_{\text{aux}} \, L_{\text{dense\_aux}}$$

**Scale-invariant log loss (SILog) — primary, on K sparse query points:**
$$L_{\text{silog}} = \sqrt{\frac{1}{K} \sum_q \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{K} \sum_q \delta_q\right)^2}, \quad \delta_q = \log \hat{d}_q - \log d_q^*$$

$\lambda_{\text{var}} = 0.15$ (per-image computation). Computed per image, then averaged across batch.

> **v14 change from v13.1:** $\lambda_{\text{var}}$ lowered from 0.5 to 0.15 (AdaBins value). Exp 6 showed $\lambda = 0.5$ forgives too much scale error → 5× slower learning. Exp 7 with $\lambda = 0.15$ fixed scale learning (s* 1.38→1.17 in one epoch). Per-image computation (not global batch mean) is correct — prevents cross-image scale cancellation.

**Dense auxiliary loss [NEW in v14] — on full-resolution encoder features:**

$$L_{\text{dense\_aux}} = L_{\text{silog}}^{L2}(\hat{D}_{L2},\; D^*_{\downarrow 8}) + L_{\text{silog}}^{L3}(\hat{D}_{L3},\; D^*_{\downarrow 16})$$

where:
$$\hat{D}_{L2} = \exp(\text{Conv}_{1\times1}^{L2}(L_2)) \in \mathbb{R}^{H/8 \times W/8}, \quad \hat{D}_{L3} = \exp(\text{Conv}_{1\times1}^{L3}(L_3)) \in \mathbb{R}^{H/16 \times W/16}$$

$D^*_{\downarrow s}$: GT depth bilinear-downsampled to stride $s$, masked where GT > 0. $\lambda_{\text{aux}} = 0.5$.

**Why dense aux is critical:** The encoder (28.6M params) currently receives gradient through only K=256 query points — most spatial positions in L1/L2 get zero gradient per batch. Dense aux heads provide direct gradient to ALL spatial positions in L2 (32×40 = 1,280 positions) and L3 (16×20 = 320 positions) every batch. This forces the encoder to learn depth-aware features everywhere, not just where queries happen to land.

**Precedent:** DETR uses auxiliary prediction heads at every intermediate decoder layer to improve gradient flow. Multi-task auxiliary losses are standard in dense prediction (Mask2Former, UPerNet). The dense aux heads are discarded at inference — zero cost at test time.

**Dense aux params:** Conv2d(128→1) = 129 params + Conv2d(192→1) = 193 params = **~322 params** total. Negligible.

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
| Attention dropout | 0.1 (MSDA Q2Q, B3a, B3b, L3/L4 self-attn) |
| Encoder | ConvNeXt V2-T, fine-tuned with 0.1× LR |
| Trainable | All (~40.9M: encoder 28.6M + neck/self-attn 7.5M + precompute 0.6M + decoder+canvas 4.3M) |

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

## 9. Parameter Budget (v14)

| Component | v13.1 | v14 | Delta |
|-----------|------:|------:|------:|
| **Encoder: ConvNeXt V2-T** (fine-tuned 0.1× LR) | **28,600K** | **28,600K** | — |
| Projection Neck (Section 4.2) | 401K | 401K | — |
| L3 Self-Attention, 2× FullSelfAttn₃₈₄ (Section 4.1) | 3,542K | 3,542K | — |
| L4 Self-Attention, 2× FullSelfAttn₃₈₄ (Section 4.3) | 3,542K | 3,542K | — |
| **MSDA value projections (Section 5.1)** | ~~210K~~ | **111K** | **−99K** |
| B3b L4 KV projections (Section 5.2) | 296K | 296K | — |
| B3a L3 KV projections (Section 5.3) | 148K | 148K | — |
| ~~$W_g$ L4 projection (Section 5.2 old)~~ | ~~74K~~ | **—** | **−74K** |
| ~~Per-level B5 projections (Section 5.3 old)~~ | ~~136K~~ | **—** | **−136K** |
| **B1: Multi-scale seed (Section 6.1)** | ~~142K~~ | **160K** | **+18K** |
| ~~B2: Local cross-attn (2 layers)~~ | ~~891K~~ | **—** | **−891K** |
| **MSDA Decoder: 3× (MSDA + canvas + Q2Q + FFN) (Section 6.2)** | — | **1,722K** | **+1,722K** |
| **B3a: L3 cross-attn + canvas + Q2Q (2L) (Section 6.3)** | ~~740K~~ | **1,112K** | **+372K** |
| **B3b: L4 cross-attn + canvas + Q2Q (2L) (Section 6.4)** | ~~740K~~ | **1,112K** | **+372K** |
| **Spatial Canvas shared (Section 6.5)** | — | **78K** | **+78K** |
| ~~Q2Q₁ + Q2Q₂~~ | ~~896K~~ | **—** | **−896K** (Q2Q now inside every layer) |
| ~~B5: Fused cross-attn (3L) + type emb~~ | ~~1,431K~~ | **—** | **−1,431K** |
| **Depth Head (Section 6.6)** | (in B5) | **74K** | (moved out) |
| **Depth scale param** | — | **0.001K** | **+0.001K** |
| **Dense aux heads (training only)** | — | **0.3K** | **+0.3K** |
| | | | |
| **Neck + self-attn + precompute + decoder** | **~12,083K** | **~12,301K** | **+218K** |
| **Total model** | **~40,683K (~40.7M)** | **~40,901K (~40.9M)** | **+218K** |

**What changed in v14:**
- **Removed:** B1 token construction (−93K), B2 (−891K), standalone Q2Q₁/Q2Q₂ (−896K), B5 (−1,431K), B5 pre-compute (−210K) = **−3,521K**
- **Added:** Multi-scale seed (+160K), MSDA decoder 3L (+1,722K), MSDA value projections (+111K), B3a canvas+Q2Q (+372K), B3b canvas+Q2Q (+372K), Spatial Canvas shared (+78K), depth head (+74K), depth scale (+0.001K), dense aux (+0.3K) = **+2,890K**
- **Kept:** B3a cross-attn+FFN (740K), B3b cross-attn+FFN (740K), B3a/B3b KV pre-compute (444K)
- **Net:** **+218K** (model is ~0.5% larger, with major new capabilities: multi-scale seed, Q2Q at every layer, Spatial Canvas for DPT-like neighborhood context)

**VRAM estimate:** v13.1 used ~7.4GB at batch=8, K=256. MSDA replaces 91-token cross-attention (large KV per query) with 96 bilinear samples (tiny per query). Q2Q adds $K^2 \times d$ per layer (~100MB for 7 layers at K=256). Spatial Canvas adds ~120KB (canvas tensor) + ~140M MACs (negligible activation memory). Dense aux adds negligible overhead. Expected v14 VRAM: **~7.5–7.8GB** (fits in 8GB).

**Inference latency estimate (RTX 4060 Laptop, bf16, batch=1):**

RTX 4060 Laptop: ~16 TFLOPS bf16, ~256 GB/s memory bandwidth. Most operations are memory-bandwidth-bound, not compute-bound.

*At 256×320 (current training resolution):*

| Component | MACs | Estimated time | Notes |
|-----------|-----:|------:|-------|
| Encoder (ConvNeXt V2-T) | ~7.3G | ~6ms | CNN, memory-bound |
| L3 self-attn (2L, 320 tokens) | ~1.1G | ~1ms | Small sequence |
| Neck + L4 self-attn | ~0.3G | ~0.5ms | Conv k1 + self-attn trivial |
| Pre-compute (value maps + KV) | ~0.2G | ~0.5ms | Conv 1×1 |
| **Encoder total** | **~8.9G** | **~8ms** | |
| MSDA decoder (3L, K=256) | ~0.6G | ~3ms | 96 deformable + canvas per layer |
| B3a (2L, K=256, N_L3=320) | ~0.35G | ~2ms | Full attn + canvas + Q2Q |
| B3b (2L, K=256, N_L4=80) | ~0.35G | ~1.5ms | Small KV + canvas + Q2Q |
| Canvas DWConv smooth (7L) | ~0.004G | <0.1ms | 7× DWConv3×3 on 16×20 |
| Depth head + final canvas read | ~0.02G | <0.1ms | |
| **Decoder total** | **~1.3G** | **~7ms** | |
| **Total (256×320)** | **~10.2G** | **~15ms** | **~67 FPS** |

*At 480×640 (target resolution):*

| Component | MACs | Estimated time | Notes |
|-----------|-----:|------:|-------|
| Encoder (ConvNeXt V2-T) | ~27.5G | ~18ms | ~3.7× more pixels |
| L3 self-attn (2L, 1200 tokens) | ~4.2G | ~3ms | 1200² attention |
| Neck + L4 self-attn | ~1.5G | ~1.5ms | Conv k1 + self-attn |
| Pre-compute | ~0.5G | ~1ms | Larger feature maps |
| **Encoder total** | **~33.7G** | **~24ms** | |
| MSDA decoder (3L, K=256) | ~0.6G | ~3ms | Deformable + canvas per layer |
| B3a (2L, K=256, N_L3=1200) | ~0.65G | ~4ms | Full attn + canvas + Q2Q |
| B3b (2L, K=256, N_L4=300) | ~0.45G | ~2.5ms | 256×300 attn + canvas + Q2Q |
| Canvas DWConv smooth (7L) | ~0.015G | <0.1ms | 7× DWConv3×3 on 30×40 |
| Depth head + final canvas read | ~0.02G | <0.1ms | |
| **Decoder total** | **~1.7G** | **~10ms** | |
| **Total (480×640)** | **~35.4G** | **~34ms** | **~29 FPS** |

**DAv2-S baseline on RTX 4060 Laptop:** ~18ms at 518×518 (ViT-S well-optimized with flash attention).

**Key observations:**
- Encoder dominates (~60–70% of total time). Decoder is cheap.
- MSDA decoder is resolution-independent (bilinear samples don't scale with image size).
- B3a at 480×640 is the decoder bottleneck (1200 KV tokens × 256 queries × 2 layers).
- At 256×320 our model is slightly slower than DAv2-S (~14ms vs ~8ms for DAv2-S at 256×320). Speed advantage only appears at inference with small K or with optimized CUDA kernels for bilinear sampling.

---

## 10. Implementation

### 10.1 File Structure

```
src/
├── models/
│   ├── spd.py                  # Main model: encode + neck + pre-compute + decode
│   ├── encoder/
│   │   ├── convnext.py         # ConvNeXt V2-T wrapper (Section 3)
│   │   ├── pyramid_neck.py     # Channel projection + L3/L4 self-attention (Section 4)
│   │   └── precompute.py       # MSDA value maps, B3a/B3b KV projections (Section 5)
│   └── decoder/
│       ├── query_seed.py       # B1: multi-scale seed from all 4 neck levels + PE (Section 6.1)
│       ├── msda_decoder.py     # MSDA decoder: 3L deformable cross-attn + canvas + Q2Q + FFN (Section 6.2)
│       ├── spatial_canvas.py   # Spatial Canvas: scatter-gather-smooth loop + shared DWConv (Section 6.5)
│       ├── global_cross_attn.py # B3a (L3 full attn) + B3b (L4 full attn) + canvas (Sections 6.3/6.4)
│       └── depth_head.py       # MLP depth head + final canvas read + learnable scale (Section 6.6)
├── data/
│   └── nyu_dataset.py          # NYU Depth V2 data loading + augmentation
├── utils/
│   └── losses.py               # L_silog, dense aux loss
├── config.py                   # H_IMG, W_IMG, EPOCHS
├── train.py                    # Training loop
└── evaluate.py                 # Evaluation on Eigen test set
```

### 10.2 Dependencies

- PyTorch ≥ 2.0 (for `F.grid_sample`, `F.scaled_dot_product_attention`)
- `timm` — ConvNeXt V2-T pre-trained weights (`convnextv2_tiny.fcmae_ft_in1k`)
- NYU Depth V2 dataset (download via standard script)

### 10.3 Build Order (v14)

Steps 1–10 (v13.1) form the existing codebase. Steps 11–17 implement the v14 redesign.

| Step | What to build | Status |
|------|--------------|--------|
| 1 | ConvNeXt V2-T encoder + projection neck + L4 self-attn | ✅ Done |
| 2 | Pre-compute module (B3b L4 KV, B3a L3 KV) | ✅ Done |
| 3–4 | B1 token construction + B2 local cross-attn | ✅ Done (will be replaced) |
| 5 | B3b global L4 cross-attn | ✅ Done |
| 6–7 | L3 Self-Attention + pre-compute L3 KV | ✅ Done |
| 8 | B3a L3 global cross-attn | ✅ Done |
| 9–10 | B5 token construction + FusedDecoder | ✅ Done (will be removed) |

**v14 new steps:**

| Step | What to build | How to validate |
|------|--------------|-----------------|
| 11 | **Pre-compute MSDA value maps** (Section 5.1): Conv2d projections for V_L1, V_L2, V_L4 (L3=identity). Remove W_g and B5 per-level projections. | Shape test: V_L1 [B, 192, H/4, W/4], etc. Verify bilinear sampling from value maps. |
| 12 | **B1 → multi-scale seed** (Section 6.1): Sample all 4 neck levels at query center (64+128+192+384=768-d), concat PE (32-d), project 800→192. Replace TokenConstructor. | Shape test: input coords [B, K, 2] → seed [B, K, 192] + pos [B, K, 192]. Verify W_seed 800→192 = ~154K. |
| 13 | **Spatial Canvas** (Section 6.5): Build `spatial_canvas.py` with (a) canvas init from L3, (b) per-layer read-write-smooth: bilinear gather, gated W_read fusion, scatter_add_ write, shared DWConv3×3 smooth, (c) final read W_final. | Shape test: canvas [B, 192, H_c, W_c]. Verify scatter writes to correct positions. Verify DWConv residual (zero-init → identity). Verify gate ∈ (0,1). ~340K params. |
| 14 | **MSDA DecoderLayer** (Section 6.2): Build single layer with (a) offset prediction Linear, (b) attention weight prediction Linear, (c) bilinear sampling from value maps, (d) per-head aggregation, **(e) canvas read-write-smooth**, (f) Q2Q self-attn with spatial PE, (g) FFN. | Shape test: input h [B, K, 192] → output [B, K, 192]. Verify 96 sampling points (6×4×4). Verify canvas F evolves after each layer. Verify offsets near zero at init. |
| 15 | **MSDA Decoder stack + B3a/B3b with canvas** (Sections 6.2–6.4): Stack 3 MSDADecoderLayers + 2 B3a + 2 B3b, all with canvas sublayer. Update SPD forward to: B1(seed) → init canvas → 3×MSDA → B3a → B3b → final canvas read → depth_head. Remove B2, B5, routing. | End-to-end forward pass test. Compare param count to expected ~40.9M. Verify canvas state differs at each layer. |
| 16 | **Dense aux heads + depth head** (Sections 6.6, 7.2): Standalone depth_head.py (MLP + final canvas read + learnable scale). Dense aux Conv2d on L2/L3. Update train.py for combined loss. | Verify depth head bias = log(2.5). Verify scale param = 0 at init. Verify dense aux loss decreasing. |
| 17 | **Full v14 pipeline**: train 10 epochs with all changes | Target: AbsRel < 0.20 (break through 0.23 ceiling). Monitor: pred range reaching 8m+, no regression after epoch 3, canvas evolving meaningfully (not collapsing to uniform), Q2Q attention patterns. |

**Key changes from v13.1:**
- B1 (91-token constructor) → multi-scale seed (all 4 neck levels at center, 800→192)
- B2 (fixed-grid cross-attn) + B5 (fused cross-attn) → MSDA decoder (deformable cross-attn + canvas + Q2Q, 3 layers)
- +**Spatial Canvas**: evolving dense feature map at L3 resolution, scatter-gather-smooth at every decoder layer — DPT-like neighborhood context (~340K params)
- B3b routing removed (no B5 to route to)
- +Dense aux loss on L2/L3 (dense encoder gradient)
- +Learnable depth scale (break output ceiling)
- $\lambda_{\text{var}}$: 0.5 → 0.15 (stronger metric signal)
- Per-image SILog + data augmentation (from Exp 6–7 fixes)
