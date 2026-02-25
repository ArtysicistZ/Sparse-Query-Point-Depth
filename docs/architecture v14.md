# Research Plan: SPD v14 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-24
Version: v14 (v13.1 → MSDA decoder + Dual Spatial Canvas + Q2Q + no B3a/B5 + dense auxiliary loss + learnable depth scale)

> **v14 revision** (post-Exp 7): Major decoder redesign addressing six bottlenecks:
> 1. **Rigid local sampling** — B1 used fixed 5×5/3×3 grids at L2/L3/L4. Fix: Multi-Scale Deformable Attention (MSDA) with learned per-head per-level offsets (Deformable DETR, Zhu et al. 2021).
> 2. **No inter-query communication** — queries were isolated predictions with no spatial consistency. Fix: Q2Q self-attention inside every decoder layer (Mask2Former pattern).
> 3. **No dense-like local neighborhood** — DPT gets spatial smoothness via Conv3×3 at every fusion stage; our queries predict with no local spatial prior. Fix: Dual Spatial Canvas — evolving dense feature maps at L2 (stride 8) and L3 (stride 16) resolutions with scatter-gather-smooth loop at every decoder layer, providing DPT-like multi-resolution neighborhood context.
> 4. **Sparse encoder gradient** — encoder gets gradient from only K=256 query points. Fix: dense auxiliary depth heads on L2/L3 features.
> 5. **Log-depth head output ceiling** — MLP output capped at ~1.55 (pred max ~4.7m vs GT ~10m). Fix: learnable output scale parameter.
> 6. **B3a/B5 redundancy** — B3a's full L3 cross-attention is redundant: L3 canvas (initialized from L3, evolving) + MSDA deformable sampling already cover L3 access. B5's skip connections are subsumed by MSDA + B3b. Fix: remove both B3a and B5, decoder is MSDA → B3b → depth head.
>
> **Paper references:** MSDA follows Deformable DETR (Zhu et al. 2021): per-head per-level learned offsets with predicted attention weights. Q2Q follows DETR (Carion 2020) and Mask2Former (Cheng 2022) where cross-attn → self-attn → FFN is the standard decoder layer. Dual Spatial Canvas is inspired by DPT's multi-resolution progressive spatial smoothing and the SPN family (CSPN, NLSPN, DySPN) for iterative convolutional propagation, adapted for sparse queries via scatter-gather operations at L2+L3 resolutions. Canvas kernel sizes follow CSPN++ findings: L2 uses DWConv5×5 (sparser coverage needs faster propagation, stride 8 keeps physical area local ~40px) while L3 keeps DWConv3×3 (stride 16 already covers 48px per kernel, larger kernels cause mixed-depth boundary blurring per NLSPN). Dense aux losses follow DETR's intermediate supervision.

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
> **v14 changes** (post-Exp 7): Replace B2+B5 with MSDA decoder (multi-scale deformable attention + Q2Q self-attention at every layer). Remove B3a (redundant with L3 canvas + MSDA) and L3 self-attention (ConvNeXt 9× DW k7 provides sufficient spatial mixing). Keep B3b for global L4 attention. +Dense auxiliary loss, +learnable depth scale. Addresses 0.23 AbsRel ceiling.

```
RGB Image (256×320 current / 480×640 target)
  │
  ▼
ConvNeXt V2-T encoder (ImageNet pre-trained, fine-tuned with 0.1× LR)
  Stage 1: H/4  × W/4  × 96   (3× ConvNeXt block, stride 4)
  Stage 2: H/8  × W/8  × 192  (3× ConvNeXt block, stride 8)
  Stage 3: H/16 × W/16 × 384  (9× ConvNeXt block, stride 16)
  Stage 4: H/32 × W/32 × 768  (3× ConvNeXt block, stride 32)
  │
  ▼
Projection Neck (trainable, Conv 1×1 + LN per level → uniform d=192)
  → L1 [H/4  × W/4  × 192]   stride 4
  → L2 [H/8  × W/8  × 192]   stride 8    ──→ [Dense Aux L2: Conv(192→1), SILog on 32×40]
  → L3 [H/16 × W/16 × 192]   stride 16   ──→ [Dense Aux L3: Conv(192→1), SILog on 16×20]
  │
  ▼
2× FullSelfAttn₁₉₂ on L4_pre → L4 [H/32 × W/32 × 192]   stride 32
  │
  ▼
Pre-compute (once per image)
  MSDA value maps: V_Lℓ = Lℓ (all levels are d=192, identity)
  B3b L4 KV
  │
  ▼
Per-query decoder (×K queries in parallel)
  B1:   multi-scale seed h⁰ from all 4 neck levels at center + Fourier PE
  │
  Dual Spatial Canvas:
    F_L2⁰ = L2   [H/8  × W/8  × d]   ← fine canvas (stride 8, edge resolution)
    F_L3⁰ = L3   [H/16 × W/16 × d]   ← coarse canvas (stride 16, broad neighborhood)
  │
  3 × MSDA DecoderLayer:
  │  ├── Multi-Scale Deformable Cross-Attn (h reads V_L1..V_L4 at learned offsets)
  │  ├── Dual Canvas Read-Write-Smooth (h reads both F_L2 & F_L3, writes back, DWConv smooths each)
  │  ├── Q2Q Self-Attn (K queries communicate, spatial PE on Q/K)
  │  └── FFN
  │
  B3b:  2 × [L4 cross-attn + Dual Canvas + Q2Q + FFN]    [global semantic]
  │
  Final canvas read: h_final = h + W_final · [bilinear(F_L2⁵, pos); bilinear(F_L3⁵, pos)]
  Depth head: MLP(h_final) × exp(s) → log_depth → exp → depth
```

At 256×320: N_L4 = 80. At 480×640: N_L4 = 300.

Core dimension $d = 192$. Encoder channels: [96, 192, 384, 768]. Neck projects all levels to uniform $d = 192$. ConvNeXt V2 blocks already provide extensive local spatial mixing (DW Conv k7 stacked 3–9× per stage). 2× L4 self-attention layers at $d = 192$ for global context (L3 self-attention removed — ConvNeXt Stage 3's 9× DW k7 provides sufficient spatial mixing). MSDA: 6 heads, 4 levels, 4 points per head per level = 96 sampling points per query per layer. Multi-scale seed: all 4 neck levels (each $d$-dim) sampled at query center → 800-d → project to d. MSDA value maps are identity (all levels already $d$-dim). Dual Spatial Canvas: two dense $d$-dim maps at L2 (stride 8, DWConv5×5 smooth) and L3 (stride 16, DWConv3×3 smooth) resolutions, initialized from $L_2$ and $L_3$ respectively, updated via scatter-gather-smooth at every decoder layer — provides DPT-like multi-resolution neighborhood context. All 5 decoder layers (3 MSDA + 2 B3b) use the same 4-sublayer pattern: cross-attn → dual canvas read-write-smooth → Q2Q self-attn → FFN.

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
  → Down: LN + Conv(384→768, k2, s2)                             15×20×768
  → Stage 4: 3× ConvNeXt V2 block (768ch)                     →  15×20×768   stride 32
```

Note: Stage 3's 9× DW Conv k7 blocks provide near-image-spanning theoretical RF at stride 16, giving L3 features extensive spatial context without explicit self-attention. L4 self-attention (Section 4.2) re-establishes explicit global relationships at stride 32 for B3b.

**Why ConvNeXt V2-T over other backbones:**
- Natural 4-level pyramid at strides [4, 8, 16, 32] — true multi-resolution features (no fake upsampling from single-resolution ViT)
- 40–50% faster than Swin-T on GPU at same FLOPs (pure conv, no window partition overhead)
- Higher accuracy than Swin-T (83.0% vs 81.3% IN-1K)
- GRN (Global Response Normalization) — same component proven in our v12 design
- L1–L3 need strong **local** features (our decoder samples them locally); ConvNeXt excels at this

**Source:** Pre-trained weights from `timm` (`convnextv2_tiny.fcmae_ft_in22k_in1k` or `convnextv2_tiny.fcmae_ft_in1k`).

---

## 4. Projection Neck + L4 Self-Attention (trainable)

L4 self-attention provides global spatial context at the coarsest level. L3 self-attention is removed — ConvNeXt V2 Stage 3's 9× DW Conv k7 blocks give theoretical RF of ~55×55 at stride 16, exceeding the L3 feature map at both resolutions (16×20 at 256×320, 30×40 at 480×640). The L3 canvas (Section 6.5) provides dense L3-resolution context in the decoder. L4 self-attention is still needed because Stage 4 has only 3 local conv blocks, and B3b requires globally-coherent L4 features for scene-level understanding.

### ~~4.1 L3 Self-Attention~~ (REMOVED in v14)

> **Removed.** L3 self-attention (2× FullSelfAttn₃₈₄, 3,542K params) is eliminated. Its original purposes — providing global L3 context for B3a cross-attention and feeding globally-informed features into Stage 4 — are no longer needed: B3a is removed (Section 6.3), and ConvNeXt Stage 3's 9× DW Conv k7 already provides near-global spatial mixing at L3 resolution. The L3 canvas (initialized from L3, evolving at every decoder layer) provides decoded L3-resolution context. L4 self-attention (kept) provides explicit global context at the semantic level that matters most for B3b.

### 4.2 Projection Neck

1×1 convolutions project all encoder stages to uniform $d = 192$:

```
Stage 1 [H/4  × W/4  × 96]         → Conv(96→192,   k1) + LN → L1 [H/4  × W/4  × 192]
Stage 2 [H/8  × W/8  × 192]        → Conv(192→192,  k1) + LN → L2 [H/8  × W/8  × 192]
Stage 3 [H/16 × W/16 × 384]        → Conv(384→192,  k1) + LN → L3 [H/16 × W/16 × 192]
Stage 4 [H/32 × W/32 × 768]        → Conv(768→192,  k1) + LN → L4_pre [H/32 × W/32 × 192]
```

**Why uniform $d$:** All downstream consumers (MSDA value maps, B3b KV, B1 seed, dual canvas) operate at $d = 192$. The previous design [64, 128, 192, 384] created information bottlenecks — L1/L2 were downprojected then immediately reprojected back to $d$ for MSDA value maps. Uniform $d$ eliminates this waste and makes MSDA value maps identity (no learned V projections needed). L4 at $d$ instead of $2d$ also makes L4 self-attention 4× cheaper with minimal quality loss: 80–300 tokens need functional global communication, not high-dimensional expressiveness.

**Neck params:** ~19K + ~37K + ~74K + ~148K = **~278K**.

### 4.3 L4 Self-Attention (after projection)

```
L4_pre [H/32 × W/32 × 192, N_L4 tokens]
  → FullSelfAttn₁₉₂ layer 1 (6 heads, d_head=32, Pre-LN + FFN 192→768→192, dropout 0.1)
  → FullSelfAttn₁₉₂ layer 2 (same)
  → L4 [H/32 × W/32 × 192]   ← every L4 token sees all positions
```

At 256×320: N_L4 = 80 tokens. At 480×640: N_L4 = 300 tokens.

**Why L4 self-attn is needed:** Stage 4 has only 3 blocks of local 7×7 depthwise conv — insufficient for explicit global spatial relationships. L4 self-attention cheaply re-establishes direct token-to-token communication (80 or 300 tokens), ensuring B3b cross-attention accesses globally-coherent semantic features.

**Why $d = 192$ (not 384):** With uniform neck projection, L4 self-attention operates at the same dimension as all other attention modules (MSDA, B3b, Q2Q: all 6 heads × 32d/head). The progressive compression path 768→384→(SA)→384→(KV)→192 sounds appealing but a well-trained 768→192 projection preserves the essential information equally well — the decoder's capacity is bounded by $d = 192$ regardless. Savings: 2,656K params (3,542K → 886K).

**L4 self-attention params:** 2× (QKV 3×192² + O 192² + FFN 192→768→192 + 2×LN) = **~886K**.

### 4.4 Summary

| Level | Resolution (480×640) | Channels | Stride | Tokens | Self-Attention |
|-------|:---:|:---:|:---:|:---:|:---|
| L1 | 120×160 | 192 | 4 | 19,200 | — |
| L2 | 60×80 | 192 | 8 | 4,800 | — |
| L3 | 30×40 | 192 | 16 | 1,200 | — (9× DW k7 provides near-global RF) |
| L4 | 15×20 | 192 | 32 | 300 | 2× FullSelfAttn₁₉₂ (after projection) |

All levels output $d = 192$. MSDA value maps are identity (no learned projections needed).

**Section 4 total:** Neck ~278K + L4 self-attn ~886K = **~1,164K (~1.2M)**.

---

## 5. Pre-compute (once per image)

**Input:** Projected pyramid $\{L_1, L_2, L_3, L_4\}$ (from Section 4: neck projection + L4 self-attention).
**Output:** $\text{cache} = \{V_{L1}, V_{L2}, V_{L3}, V_{L4},\; K_{\text{L4}}^{(1:2)}, V_{\text{L4}}^{(1:2)}\}$

### 5.1 MSDA value feature maps [NEW in v14]

With uniform neck projection ($d = 192$ at all levels), MSDA value maps are identity — no learned projections needed. MSDA decoder layers bilinear-sample directly from neck features at learned offset positions. ConvNeXt V2's DW Conv k7 (stacked 3–9× per stage) already provides extensive local spatial mixing in each feature position.

$$V_{L\ell} = L_\ell \quad \in \mathbb{R}^{d \times H_\ell \times W_\ell}$$

| Level | Neck channels | Projection | Params |
|-------|:---:|-----------|-------:|
| L1 | 192 | identity ($C_{L1} = d$) | 0 |
| L2 | 192 | identity ($C_{L2} = d$) | 0 |
| L3 | 192 | identity ($C_{L3} = d$) | 0 |
| L4 | 192 | identity ($C_{L4} = d$, after self-attn) | 0 |
| **Total** | | | **0** |

> **Why identity is sufficient:** The previous design required V projections because neck channels varied ([64, 128, 192, 384]). With uniform $d = 192$, all levels are already in the attention dimension. Per-layer differentiation comes from each layer's own offset predictions and attention weights — this provides sufficient capacity without shared V projections.

### 5.2 KV projections for B3b (L4 cross-attention)

Each B3b layer $\ell = 1, 2$ has per-layer $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times d}$, applied to L4:

$$K_{\text{L4}}^{(\ell)} = L_4 \, (W_K^{(\ell)})^T, \quad V_{\text{L4}}^{(\ell)} = L_4 \, (W_V^{(\ell)})^T \quad \in \mathbb{R}^{N_{L4} \times d}$$

$N_{L4} = 80$ at 256×320, $300$ at 480×640. Square projections ($d \to d$) preserve full information. Params: $4 \times 192 \times 192 =$ **~148K**.

### ~~5.3 KV projections for B3a (L3 cross-attention)~~ (REMOVED in v14)

> **Removed.** B3a is eliminated in v14 — L3 access is covered by MSDA deformable sampling + L3 canvas (initialized from L3, evolving at every layer). The B3a L3 KV projections (148K) are no longer needed.

### ~~5.4 W_g and per-level projections for B5~~ (REMOVED in v14)

> **Removed.** B5 is eliminated in v14 — its skip connections and routed-token re-reading are absorbed by MSDA (reads all levels every layer) and B3a/B3b (full global attention). The pre-computed $W_g$ (74K) and per-level B5 central token projections (136K) are no longer needed. MSDA value maps (Section 5.1) replace this functionality.

**Pre-compute total (v14):** 0K (V maps) + ~148K (B3b KV) = **~148K** (was ~407K before uniform neck, ~654K in v13.1).

---

## 6. Per-Query Decoder [v14: MSDA + B3b]

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
| $h^{(3b)}$ | Output after B3b (global-aware) | $d$ |
| $F_{L2}^{(n)}, F_{L3}^{(n)}$ | Dual Spatial Canvas after decoder layer $n$ (Section 6.5). $F_{L2}^{(0)} = V_{L2}$, $F_{L3}^{(0)} = L_3$ | $d \times H_{c\ell} \times W_{c\ell}$ |
| $r_q^{(n)}$ | Dual canvas read: $[\text{bilinear}(F_{L2}^{(n-1)}, p_q^{(L2)});\; \text{bilinear}(F_{L3}^{(n-1)}, p_q^{(L3)})]$ | $2d$ |
| $h_{\text{final}}$ | $h^{(3b)} + W_{\text{final}} \cdot r_q^{(5)}$ — dual canvas-enhanced final repr. | $d$ |
| $\hat{d}_q$ | Predicted depth: $\exp(\text{MLP}(h_{\text{final}}) \cdot \exp(s))$ | scalar |

### 6.1 B1: Multi-Scale Query Seed [REDESIGNED in v14]

*Sample features from ALL 4 pyramid levels at the query center, concatenate with Fourier PE, project to seed $h^{(0)}$. The seed is immediately meaningful — it carries coarse-to-fine context (L4 semantics → L1 fine detail) before any decoder processing. ConvNeXt V2's stacked DW Conv k7 blocks (3–9× per stage) already provide extensive local spatial context at every position.*

> **v14 change from v13.1:** B1 no longer constructs 91 multi-scale grid tokens, nor does it sample from a single level. Instead, it samples the center point from all 4 neck feature levels in their native channel dimensions, giving the seed multi-scale information. MSDA then focuses entirely on learning WHERE to sample (offsets), since WHAT's at center is already known.

**Multi-scale center features (from projected pyramid, Section 4.2):**
$$f_q^{(\ell)} = \text{Bilinear}(L_\ell,\; p_q^{(\ell)}) \quad \in \mathbb{R}^{C_\ell}$$

| Level | Channels | What it provides |
|-------|:---:|---|
| $f_q^{(1)}$ from L1 | 192 | Fine edges, textures, local geometry (DW k7 ×3 at stride 4) |
| $f_q^{(2)}$ from L2 | 192 | Mid-level patterns, small objects (DW k7 ×3 at stride 8) |
| $f_q^{(3)}$ from L3 | 192 | Object parts, spatial layout (9× DW k7 near-global RF) |
| $f_q^{(4)}$ from L4 | 192 | Scene semantics, room structure + global context (from L4 self-attn) |

Each ConvNeXt V2 stage applies DW Conv k7 stacked 3–9 times, providing extensive local spatial mixing at every feature position before the decoder sees them.

**Positional encoding:**
$$\text{pe}_q = [\sin(2\pi \sigma_l u/W);\; \cos(2\pi \sigma_l u/W);\; \sin(2\pi \sigma_l v/H);\; \cos(2\pi \sigma_l v/H)]_{l=0}^{7}$$
$\sigma_l = 2^l$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

**Multi-scale query seed:**
$$h^{(0)} = \text{LN}\!\left(W_{\text{seed}} \,[f_q^{(1)};\; f_q^{(2)};\; f_q^{(3)};\; f_q^{(4)};\; \text{pe}_q]\right) \quad \in \mathbb{R}^d$$

$$W_{\text{seed}} \in \mathbb{R}^{d \times (4d + 32)} = \mathbb{R}^{192 \times 800}$$

Input is $4 \times 192 + 32 = 800$ dimensions (uniform $d$-dim from all 4 neck levels + 32-dim Fourier PE). With uniform neck projection, all levels contribute equally-sized but semantically-different feature vectors. The projection compresses multi-scale information into a single $d$-dim vector. LayerNorm stabilizes the initial representation for downstream attention layers.

> **Uniform neck simplifies seeding:** Since all neck levels output $d = 192$, the seed samples $d$-dim features at every level. MSDA value maps are also identity (Section 5.1). The seed and MSDA operate on the same feature representations — no information asymmetry between seed construction and deformable sampling.

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
  ──→ LN ──→ Dual Canvas Read-Write-Smooth (read F_L2 & F_L3, fuse, scatter h to both, DWConv smooths each) ──→ +residual
  ──→ LN ──→ Q2Q Self-Attn (K queries attend to each other, spatial PE on Q/K) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

Dual canvas interaction (Section 6.5) is inserted between cross-attention and Q2Q. After MSDA reads from static value maps, the dual canvas provides complementary information at two resolutions: what nearby queries have decoded so far (L2 for fine edges, L3 for broad neighborhood). Q2Q then reconciles local canvas context with global inter-query consistency.

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
| Dual Canvas: $W_{\text{read}}^{(n)}$ ($2d \to d$) + $W_{\text{gate}}^{(n)}$ ($d \to 1$) + LN | ~74K |
| Dual Canvas: $W_{\text{write\_L2}}^{(n)}$ ($d \to d$) + $W_{\text{write\_L3}}^{(n)}$ ($d \to d$) | ~74K |
| Q2Q $W_Q, W_K, W_V, W_O$: $4 \times d^2$ | ~148K |
| LN (Q2Q) | ~0.4K |
| FFN: $d \to 4d \to d$ | ~296K |
| LN (FFN) | ~0.4K |
| **Per layer total** | **~685K** |

**3 layers total:** ~2,055K. Plus shared $W_{\text{pos}}$ (~6K) from B1. Dual canvas per-layer DWConv smooth and $W_{\text{final}}$ counted in Section 6.5.5.

---

### ~~6.3 B3a: L3 Mid-Scale Global Cross-Attention~~ (REMOVED in v14)

> **Removed.** B3a (2L L3 cross-attention, ~1,186K params) is eliminated. L3 access is now redundantly covered by three other mechanisms:
> 1. **L3 canvas** — initialized from L3, evolving at every decoder layer (dense L3-resolution decoded features)
> 2. **MSDA** — deformable sampling from L3 at learned positions (72 total L3 samplings across 3 layers)
> 3. **Q2Q** — global inter-query communication provides indirect access to L3 information
>
> The SPN literature (CSPN, NLSPN, DySPN) shows that iterative scatter-gather-smooth on a dense canvas is an effective alternative to full cross-attention for spatial propagation. B3a's full cross-attention to 320–1,200 L3 tokens was the decoder bottleneck at 480×640 (~4ms, 40% of decoder time).

### 6.4 B3b: Global L4 Cross-Attention (2 layers)

*Full attention over ALL $N_{L4}$ L4 tokens (each with global RF from L4 self-attention). No L4 canvas exists, so B3b is the only mechanism for complete global L4 access.*

```
h ──→ LN ──→ Cross-Attn (Q=h, KV=pre-computed L4 KV, all N_L4 tokens) ──→ +residual
  ──→ LN ──→ Dual Canvas Read-Write-Smooth (Section 6.5) ──→ +residual
  ──→ LN ──→ Q2Q Self-Attn (K queries communicate, shared spatial PE) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

Uses pre-computed KV from Section 5.2 (per-layer $K_{\text{L4}}^{(\ell)}, V_{\text{L4}}^{(\ell)}$). Same 4-sublayer pattern as MSDA layers: cross-attn → dual canvas → Q2Q → FFN.

6 heads, $d_{\text{head}} = 32$. Per-query cost is only Q projection + attention + FFN (KV pre-computed).

**Output:** $h^{(3b)} \in \mathbb{R}^d$ — the final query representation, fed to the depth head.

> **Routing removed in v14:** In v13.1, B3b extracted top-20 L4 routing indices for B5's routed tokens. With B5 removed, routing has no consumer. B3b now purely enriches h via full global attention.

**B3b params:** 2× ($W_Q/W_O$ cross-attn ~74K + dual canvas $W_{\text{read}}/W_{\text{gate}}$/LN ~74K + dual canvas $W_{\text{write\_L2}}/W_{\text{write\_L3}}$ ~74K + Q2Q $W_Q/W_K/W_V/W_O$ ~148K + FFN ~296K + LNs ~1K) = **~1,334K** (KV counted in pre-compute, dual canvas per-layer DWConv smooth and $W_{\text{final}}$ in Section 6.5.5).

### 6.5 Dual Spatial Canvas: Evolving Dense-Local Context [NEW in v14]

*DPT's depth predictions are spatially smooth because every fusion stage applies Conv3×3 at multiple resolutions, making each pixel's representation a function of its neighbors' evolving, prediction-ready features. In SPD, queries predict in isolation — MSDA reads static feature maps, Q2Q is global (not local), and the depth head receives a single d-dim vector with no spatial smoothing. This is the root cause of the 0.23 AbsRel ceiling across all experiments (Exp 3: 0.231, Exp 5: 0.230, Exp 7: 0.239).*

*The Dual Spatial Canvas provides the missing piece: two shared dense feature maps at L2 (stride 8, fine edges) and L3 (stride 16, broad neighborhood) that accumulate decoder-evolved query information and are spatially smoothed at every stage, giving each query access to its multi-resolution neighborhood's prediction-ready representations — exactly what DPT's multi-scale Conv3×3 achieves, adapted for sparse queries.*

> **B4** was removed in v13.1. **B3a** and **B5** are removed in v14 — absorbed by MSDA + B3b + dual canvas. The Dual Spatial Canvas replaces B3a's L3 access role with evolved, prediction-ready features at L3 resolution.

#### 6.5.1 DPT ↔ SPD Correspondence

| DPT (dense) | SPD without canvas | SPD with Dual Spatial Canvas |
|---|---|---|
| Dense feature maps at multiple resolutions (stride 4, 8, 16, 32) | Static pre-computed value maps, never updated | Dual canvas: $F_{L2}$ (stride 8) + $F_{L3}$ (stride 16), updated every decoder layer |
| Conv3×3 at each fusion stage | No spatial smoothing between queries | DWConv3×3 on each canvas after each scatter |
| Each pixel reads from 3×3 neighbors' evolved features | Each query reads from learned offsets (MSDA) or global tokens (B3b) | Each query reads from both canvases — fine-grained (L2) + broad (L3) neighborhood |
| Final Conv3×3 head smooths predictions | MLP on isolated $h$ per query | MLP on $h$ + final dual canvas read (multi-resolution smoothed features) |
| Features evolve: each stage's Conv3×3 output feeds the next | Static: all 7 layers read the same $V_{L\ell}$ maps | Evolving: scatter writes decoded $h$ to both canvases → DWConv propagates → next layer reads updated state |

**Critical distinction from encoder's local mixing:** ConvNeXt V2's DW Conv k7 operates on **raw encoder features** — it is static preprocessing. The Spatial Canvas operates on **decoded, prediction-ready query representations** that evolve at every layer. This mirrors DPT's Conv3×3 which smooths **prediction-ready fused features**, not raw encoder outputs.

#### 6.5.2 Architecture

**Dual canvas resolutions: L2 (stride 8) + L3 (stride 16)**

DPT's finest Conv3×3 operates at stride 4. A single L3 canvas (stride 16) is 4× coarser — two adjacent canvas positions span 32 pixels, too wide to resolve sharp depth edges. Adding an L2 canvas (stride 8) provides edge-resolution spatial context: 8px spacing can distinguish which side of a depth boundary a query falls on.

| Canvas | Resolution (256×320) | Resolution (480×640) | K=256 coverage | Role |
|--------|:---:|:---:|:---:|---|
| $F_{L2}$ (stride 8) | 32×40 = 1,280 | 60×80 = 4,800 | 20% / 5.3% | Fine edges, depth boundaries |
| $F_{L3}$ (stride 16) | 16×20 = 320 | 30×40 = 1,200 | 80% / 21% | Broad neighborhood, surface regions |

**L2 canvas coverage is viable despite lower scatter density.** At 480×640 (worst case, 5.3% direct writes): average inter-query distance ≈ 35px = ~4.4 L2 cells. Our 2× DWConv5×5 smooth (9×9 effective RF) reaches ~4 cells per step — nearly the average inter-query distance in a single layer. After 5 layers, the accumulated RF of 37×37 at stride 8 covers 296px — well beyond average query spacing. The larger kernel (5×5 vs 3×3) is justified by L2's sparser scatter coverage: CSPN found k=5 optimal for balancing propagation speed and boundary preservation, and at stride 8 a 5×5 kernel covers only 40×40px (still local).

**Canvas initialization:**

$$F_{L2}^{(0)} = L_2 \in \mathbb{R}^{d \times H/8 \times W/8}$$
$$F_{L3}^{(0)} = L_3 \in \mathbb{R}^{d \times H/16 \times W/16}$$

Both canvases are initialized directly from neck features (all levels are $d = 192$, zero additional params). Both start meaningful — the canvases will diverge from their encoder-derived initializations after the first scatter+smooth.

**Per decoder layer $n$ (all 5 layers: 3 MSDA + 2 B3b):**

The canvas interaction is inserted as a sublayer between cross-attention and Q2Q self-attention:

```
Step 1: LN → Cross-Attn → +residual                          (existing)
Step 2: LN → Dual Canvas Read-Write-Smooth → +residual        [NEW]
Step 3: LN → Q2Q Self-Attn → +residual                        (existing)
Step 4: LN → FFN → +residual                                  (existing)
```

**Step 2 detail — Dual Canvas Read-Write-Smooth:**

```
Canvas Read (both levels):
  r_q_L2 = bilinear(F_L2^(n-1), p_q^(L2))       ∈ R^d     ← fine neighborhood (8px spacing)
  r_q_L3 = bilinear(F_L3^(n-1), p_q^(L3))       ∈ R^d     ← broad neighborhood (16px spacing)

Gated Fusion (concatenate both reads, single projection):
  gate_q = sigmoid(W_gate^(n) · h)                ∈ R^1     ← per-query relevance gate
  h ← h + gate_q · (W_read^(n) · [r_q_L2; r_q_L3])          ← W_read: 2d → d

Canvas Write (scatter, per-layer W_write, separate per canvas):
  w_q_L2 = W_write_L2^(n) · h                     ∈ R^d     ← per-layer, per-canvas projection
  w_q_L3 = W_write_L3^(n) · h                     ∈ R^d     ← per-layer, per-canvas projection
  F_L2^(n) = scatter_add(F_L2^(n-1), p_q^(L2), w_q_L2)     ← write to fine canvas
  F_L3^(n) = scatter_add(F_L3^(n-1), p_q^(L3), w_q_L3)     ← write to coarse canvas

Canvas Smooth (per-layer, independent per level):
  F_L2^(n) ← F_L2^(n) + DWConv5×5_L2^(n)(GELU(DWConv5×5_L2^(n)(F_L2^(n))))    ← 5×5 for sparser L2, per-layer
  F_L3^(n) ← F_L3^(n) + DWConv3×3_L3^(n)(GELU(DWConv3×3_L3^(n)(F_L3^(n))))    ← 3×3 for denser L3, per-layer
```

**Why concatenate-then-project (not separate additions):** $W_{\text{read}}^{(n)} \cdot [r_{L2}; r_{L3}]$ lets the model learn arbitrary linear combinations of the two canvas reads. At depth edges, it can emphasize L2 (finer resolution for edge-side disambiguation). On flat surfaces, it can emphasize L3 (broader, denser coverage). Separate additions would constrain the two reads to contribute independently.

**Why per-layer, per-canvas W_write:** Each decoder layer writes semantically different information (layer 1 writes seed-level features, layer 5 writes globally-informed features). Separate $W_{\text{write\_L2}}^{(n)}$ and $W_{\text{write\_L3}}^{(n)}$ let each layer decide independently what to store at each resolution — L2 may emphasize edge-relevant features while L3 emphasizes surface-region features. Cost: +37K per additional W_write per layer (5 layers × 2 canvases × 37K = 370K total, vs 37K shared).

**Why per-layer DWConv smooth (not shared):** The SPN literature evolved from shared propagation kernels (CSPN, NLSPN — same affinity iterated) to per-iteration adaptive kernels (DySPN, AAAI 2022). DySPN demonstrated that per-iteration affinity is strictly better in accuracy and convergence speed — fixed kernels limit representational power. In our architecture, the canvas content changes dramatically across layers: layer 1 receives sparse seed-level scatter writes (noisy, low coverage), while layer 5 receives globally-informed B3b outputs (refined, prediction-ready). Early layers benefit from more aggressive smoothing to fill coverage gaps (especially L2 at 5–20% direct writes), while later layers need conservative smoothing to preserve refined depth boundaries. Per-layer DWConv lets each layer learn its own optimal propagation pattern. Cost: +55K vs shared (69K total, vs 13.8K shared) — 0.17% of model, negligible.

**Why different kernel sizes per canvas:** L2 (stride 8) has sparser scatter coverage (20%/5.3%) and needs faster propagation — DWConv5×5 covers 40×40px, reaching average inter-query distance in ~1 step. L3 (stride 16) has denser coverage (80%/21%) and needs precision — DWConv3×3 covers 48×48px, already substantial at stride 16. Larger L3 kernels would cause the "mixed-depth problem" (NLSPN, DySPN): a 7×7 at stride 16 covers 112×112px, mixing features across depth boundaries. This asymmetric design follows CSPN++ findings that different regions benefit from different kernel sizes.

**Scatter operation:** Nearest-neighbor assignment — each query writes to the closest grid position at each level. For L2: flat index $\lfloor v/s_2 \rfloor \cdot W_{c2} + \lfloor u/s_2 \rfloor$. For L3: flat index $\lfloor v/s_3 \rfloor \cdot W_{c3} + \lfloor u/s_3 \rfloor$. Multiple queries at the same position are added. Positions with no queries retain their previous value.

**Accumulated effective RF after 5 layers:**
- L2 canvas (DWConv5×5): 37×37 at stride 8 = 296×296px — resolves depth edges within ±148px of any query
- L3 canvas (DWConv3×3): 21×21 at stride 16 = 336×336px — covers broad neighborhood around each query

**Why the canvas is truly evolving (not static):**
1. Layer 1: $h \approx$ seed → canvases get seed-level representations → DWConv smooths coarse patterns
2. Layer 3: $h$ has multi-scale MSDA context → canvases get richer representations
3. Layer 4–5: $h$ has global L4 context from B3b → canvases get final prediction-ready representations
4. Depth head: reads from fully-evolved canvases → multi-resolution neighborhood-informed prediction

Each layer reads a DIFFERENT canvas state (because prior layers scattered and smoothed). Analogous to how DPT's Conv3×3 receives different input features at each fusion stage.

**Why gate is important:** A query on a depth edge should NOT be smoothed toward the wrong side. The gate $\text{sigmoid}(W_{\text{gate}}^{(n)} \cdot h)$ allows the model to learn: "when $h$ indicates I'm on a depth discontinuity, gate down the canvas contribution." The L2 canvas's finer resolution means the gate needs to be less aggressive — L2 can resolve which side of the edge the query is on — but the gate still protects against cross-boundary contamination at sub-stride distances.

#### 6.5.3 Final Canvas Read (before depth head)

After the last B3b layer writes to and smooths both canvases, one final dual read feeds into the depth head:

$$h_{\text{final}} = h^{(3b)} + W_{\text{final}} \cdot [\text{bilinear}(F_{L2}^{(5)},\; p_q^{(L2)});\; \text{bilinear}(F_{L3}^{(5)},\; p_q^{(L3)})]$$

$$\hat{d}_q = \text{depth\_head}(h_{\text{final}})$$

$W_{\text{final}} \in \mathbb{R}^{d \times 2d}$ projects the concatenated dual canvas read to $d$-dim, same structure as the per-layer $W_{\text{read}}$.

This final read is the SPD analog of DPT's final Conv3×3 output head. The depth prediction now incorporates:
- $h^{(3b)}$: the query's own decoded representation (multi-scale + global context)
- $F_{L2}^{(5)}$ at $p_q$: fine-grained evolved neighborhood (8px spacing, sharp edge info)
- $F_{L3}^{(5)}$ at $p_q$: broad evolved neighborhood (16px spacing, surface-level smoothness)

Without the canvas, $h^{(3b)}$ is an isolated prediction. With the dual canvas, it's a **multi-resolution neighborhood-informed** prediction.

#### 6.5.4 Interaction with Q2Q and MSDA

| Mechanism | What it provides | Spatial structure |
|---|---|---|
| MSDA cross-attn | Multi-scale features from value maps at **learned** positions | Deformable — points can be anywhere |
| B3b cross-attn | Full access to all L4 tokens for global semantic context | Global — all N_L4 tokens |
| Dual Spatial Canvas | Evolved decoder features from **fixed local** neighborhood at two resolutions | Dense local grids: L2 (stride 8, DWConv5×5) + L3 (stride 16, DWConv3×3) |
| Q2Q self-attn | Inter-query consistency across **all** $K$ queries | Global — no spatial bias |

These three mechanisms are complementary, not redundant:
- **MSDA** tells the query "what features are at multi-scale positions around you" (from static encoder maps)
- **Dual Canvas** tells the query "what your nearby queries have decoded so far" at two resolutions (fine edges via L2, broad surface via L3)
- **Q2Q** tells the query "what ALL queries across the whole image think" (global consistency)

Together, they give each query exactly what DPT gives each pixel: multi-resolution local spatial context + multi-scale features + scene-wide consistency.

#### 6.5.5 Dual Spatial Canvas Parameters

| Component | Params | Notes |
|-----------|-------:|-------|
| $W_{\text{read}}^{(n)}$: $2d \to d$, 5 layers | 5 × 73,728 = **369K** | Per-layer (concatenated L2+L3 reads → h) |
| $W_{\text{gate}}^{(n)}$: $d \to 1$, 5 layers | 5 × 193 = **1.0K** | Per-layer scalar gate |
| LN (canvas), 5 layers | 5 × 384 = **1.9K** | Per-layer LayerNorm |
| $W_{\text{write\_L2}}^{(n)}$: $d \to d$, 5 layers | 5 × 36,864 = **184K** | Per-layer, writes to L2 canvas |
| $W_{\text{write\_L3}}^{(n)}$: $d \to d$, 5 layers | 5 × 36,864 = **184K** | Per-layer, writes to L3 canvas |
| 2× DWConv5×5 ($d$ ch), L2 smooth, 5 layers | 5 × 9,984 = **50K** | Per-layer L2 canvas smooth (5×5 for sparser coverage) |
| 2× DWConv3×3 ($d$ ch), L3 smooth, 5 layers | 5 × 3,840 = **19K** | Per-layer L3 canvas smooth (3×3 to preserve boundaries) |
| $W_{\text{final}}$: $2d \to d$ | **74K** | Final dual read before depth head |
| **Total** | **~883K** | ~25% of decoder, ~2.6% of total model |

**Compute per layer:** Dual canvas read (2× bilinear gather, $2 \times K \times d$) + gated fusion ($K \times 2d \times d \approx 19$M) + write projection ($K \times d^2 \approx 9.4$M) + L2 DWConv5×5 smooth ($2 \times 25 \times d \times H_{c2} \times W_{c2} \approx 12$M at 256×320) + L3 DWConv3×3 smooth ($\approx 0.6$M). 5 layers + final read: ~210M MACs total — **2.3% of encoder MACs**.

**VRAM:** L2 canvas $32 \times 40 \times 192 = 245$K values (~480KB in bf16) + L3 canvas $16 \times 20 \times 192 = 61$K values (~120KB). Total ~600KB at 256×320. At 480×640: L2 $60 \times 80 \times 192 = 922$K (~1.8MB) + L3 ~460KB = ~2.3MB. Still negligible vs 8GB VRAM.

#### 6.5.6 Connection to Experimental Failures

| Experiment failure | How Spatial Canvas addresses it |
|---|---|
| 0.23 AbsRel ceiling (Exp 3, 5, 7) | Spatially-smoothed predictions reduce per-query noise — each prediction is now a function of its neighborhood |
| Pred max capped at 5–6m | Far-depth queries reinforce each other via canvas propagation — one query reaching 8m helps neighbors reach 8m too |
| "Dense models get spatial smoothness for free" (experiments.md L186) | Canvas provides exactly this: DWConv3×3 at every stage, operating on evolved decoder features |
| Overfitting after 2–3 epochs | Canvas acts as implicit spatial regularization — isolated overfitted predictions get smoothed toward neighbors |
| No inter-query spatial consistency | Canvas + DWConv provides LOCAL consistency (complementing Q2Q's GLOBAL consistency) |

**Residual chain (v14 with dual canvas):**
$$h^{(0)} \xrightarrow[\text{cross + dual canvas + Q2Q + FFN}]{\text{3× MSDA}} h^{(3)} \xrightarrow[\text{cross + dual canvas + Q2Q + FFN}]{\text{B3b: 2L, L4}} h^{(3b)} \xrightarrow[\text{+ dual canvas final read}]{} h_{\text{final}} \to \hat{d}_q$$

Every stage uses the same 4-sublayer pattern: **cross-attn → dual canvas → Q2Q → FFN**. Total: 5 cross-attn + 5 dual canvas + 5 Q2Q + 5 FFN = **20 sublayers**.

### 6.6 Depth Head [standalone in v14]

**Final dual canvas read** (Section 6.5.3) — incorporates multi-resolution spatially-smoothed neighborhood:
$$h_{\text{final}} = h^{(3b)} + W_{\text{final}} \cdot [\text{bilinear}(F_{L2}^{(5)},\; p_q^{(L2)});\; \text{bilinear}(F_{L3}^{(5)},\; p_q^{(L3)})]$$

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

**Dense aux params:** Conv2d(192→1) = 193 params + Conv2d(192→1) = 193 params = **~386 params** total. Negligible.

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
| Attention dropout | 0.1 (MSDA Q2Q, B3b, L4 self-attn) |
| Encoder | ConvNeXt V2-T, fine-tuned with 0.1× LR |
| Trainable | All (~33.6M: encoder 28.6M + neck/self-attn 1.2M + precompute 0.1M + decoder+dual canvas 3.7M) |

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
| Projection Neck (Section 4.2, uniform $d = 192$) | 401K | **278K** | **−123K** |
| ~~L3 Self-Attention, 2× FullSelfAttn₃₈₄~~ | ~~3,542K~~ | **—** | **−3,542K** |
| L4 Self-Attention, 2× FullSelfAttn₁₉₂ (Section 4.3) | ~~3,542K~~ | **886K** | **−2,656K** |
| **MSDA value projections (Section 5.1)** | ~~210K~~ | **0** (identity) | **−210K** |
| B3b L4 KV projections (Section 5.2, $d \to d$) | ~~296K~~ | **148K** | **−148K** |
| ~~B3a L3 KV projections~~ | ~~148K~~ | **—** | **−148K** |
| ~~$W_g$ L4 projection (Section 5.2 old)~~ | ~~74K~~ | **—** | **−74K** |
| ~~Per-level B5 projections (Section 5.3 old)~~ | ~~136K~~ | **—** | **−136K** |
| **B1: Multi-scale seed (Section 6.1)** | ~~142K~~ | **160K** | **+18K** |
| ~~B2: Local cross-attn (2 layers)~~ | ~~891K~~ | **—** | **−891K** |
| **MSDA Decoder: 3× (MSDA + dual canvas + Q2Q + FFN) (Section 6.2)** | — | **2,055K** | **+2,055K** |
| ~~B3a: L3 cross-attn + dual canvas + Q2Q (2L)~~ | ~~740K~~ | **—** | **−740K** |
| **B3b: L4 cross-attn + dual canvas + Q2Q (2L) (Section 6.4)** | ~~740K~~ | **1,334K** | **+594K** |
| **Dual Spatial Canvas (Section 6.5)** | — | **143K** | **+143K** |
| ~~Q2Q₁ + Q2Q₂~~ | ~~896K~~ | **—** | **−896K** (Q2Q now inside every layer) |
| ~~B5: Fused cross-attn (3L) + type emb~~ | ~~1,431K~~ | **—** | **−1,431K** |
| **Depth Head (Section 6.6)** | (in B5) | **74K** | (moved out) |
| **Depth scale param** | — | **0.001K** | **+0.001K** |
| **Dense aux heads (training only)** | — | **0.4K** | **+0.4K** |
| | | | |
| **Neck + self-attn + precompute + decoder** | **~12,083K** | **~5,078K** | **−7,005K** |
| **Total model** | **~40,683K (~40.7M)** | **~33,678K (~33.7M)** | **−7,005K** |

**What changed in v14:**
- **Removed:** L3 self-attention (−3,542K), L4 SA dimension 384→192 (−2,656K), B3a decoder+KV (−1,334K), B1 token construction (−93K), B2 (−891K), standalone Q2Q₁/Q2Q₂ (−896K), B5 (−1,431K), B5 pre-compute (−210K), MSDA V projections (−111K, now identity), neck bottleneck channels (−123K), B3b KV fan-in (−148K) = **−11,435K**
- **Added:** Multi-scale seed (+160K), MSDA decoder 3L (+2,055K), B3b dual canvas+Q2Q (+594K), Dual Canvas per-layer DWConv+W_final (+143K), depth head (+74K), depth scale (+0.001K), dense aux (+0.4K) = **+3,026K**
- **Kept:** B3b cross-attn+FFN (740K, now with $d \to d$ KV)
- **Net:** **−7,005K** (model is ~17.2% smaller than v13.1, with major new capabilities: uniform $d$ neck, multi-scale seed, Q2Q at every layer, Dual Spatial Canvas with per-layer W_write + per-layer asymmetric DWConv smooth for DPT-like multi-resolution neighborhood context)

**VRAM estimate:** v13.1 used ~7.4GB at batch=8, K=256. v14 removes L3 self-attention, B3a, L4 SA dimension halved (384→192), and V map projections eliminated (identity). MSDA replaces 91-token cross-attention with 96 bilinear samples. Q2Q adds $K^2 \times d$ per layer (~70MB for 5 layers at K=256). Dual Spatial Canvas adds ~600KB (two canvas tensors at 256×320). With 7.0M fewer params than v13.1, expected v14 VRAM: **~5.5–6.0GB** (comfortable in 8GB).

**Inference latency estimate (RTX 4060 Laptop, bf16, batch=1):**

RTX 4060 Laptop: ~16 TFLOPS bf16, ~256 GB/s memory bandwidth. Most operations are memory-bandwidth-bound, not compute-bound.

*At 256×320 (current training resolution):*

| Component | MACs | Estimated time | Notes |
|-----------|-----:|------:|-------|
| Encoder (ConvNeXt V2-T) | ~7.3G | ~6ms | CNN, memory-bound |
| Neck + L4 self-attn (192d) | ~0.15G | ~0.3ms | Conv k1 + SA at $d$, 4× cheaper SA |
| Pre-compute (B3b KV only) | ~0.03G | ~0.1ms | V maps identity, B3b KV $d \to d$ |
| **Encoder total** | **~7.5G** | **~6.5ms** | |
| MSDA decoder (3L, K=256) | ~0.65G | ~3ms | 96 deformable + dual canvas per layer |
| B3b (2L, K=256, N_L4=80) | ~0.4G | ~1.5ms | Small KV + dual canvas + Q2Q |
| Canvas DWConv smooth (5L, L2 k5 + L3 k3) | ~0.065G | <0.2ms | 5× DWConv5×5 on 32×40 + DWConv3×3 on 16×20 |
| Depth head + final canvas read | ~0.02G | <0.1ms | |
| **Decoder total** | **~0.9G** | **~5ms** | |
| **Total (256×320)** | **~8.4G** | **~11.5ms** | **~87 FPS** |

*At 480×640 (target resolution):*

| Component | MACs | Estimated time | Notes |
|-----------|-----:|------:|-------|
| Encoder (ConvNeXt V2-T) | ~27.5G | ~18ms | ~3.7× more pixels |
| Neck + L4 self-attn (192d) | ~0.6G | ~0.8ms | Conv k1 + SA at $d$, 4× cheaper SA |
| Pre-compute (B3b KV only) | ~0.1G | ~0.2ms | V maps identity, B3b KV $d \to d$ |
| **Encoder total** | **~28.2G** | **~19ms** | |
| MSDA decoder (3L, K=256) | ~0.65G | ~3ms | Deformable + dual canvas per layer |
| B3b (2L, K=256, N_L4=300) | ~0.5G | ~2.5ms | 256×300 attn + dual canvas + Q2Q |
| Canvas DWConv smooth (5L, L2 k5 + L3 k3) | ~0.15G | ~0.2ms | 5× DWConv5×5 on 60×80 + DWConv3×3 on 30×40 |
| Depth head + final canvas read | ~0.02G | <0.1ms | |
| **Decoder total** | **~1.1G** | **~6ms** | |
| **Total (480×640)** | **~29.3G** | **~25ms** | **~40 FPS** |

**DAv2-S baseline on RTX 4060 Laptop:** ~18ms at 518×518 (ViT-S well-optimized with flash attention).

**Key observations:**
- Encoder dominates (~75–80% of total time). Decoder is very cheap.
- MSDA decoder is resolution-independent (bilinear samples don't scale with image size).
- Removing B3a eliminated the decoder bottleneck (was 1200 KV tokens × 256 queries × 2 layers at 480×640, ~4ms).
- Removing L3 self-attention saved ~3ms at 480×640 (1200² attention eliminated).
- Uniform $d = 192$ neck eliminates V map computation and halves L4 self-attention FLOPs.
- At 480×640, our model is now competitive with DAv2-S (~25ms vs ~18ms for DAv2-S at 518×518).

---

## 10. Implementation

### 10.1 File Structure

```
src/
├── models/
│   ├── spd.py                  # Main model: encode + neck + pre-compute + decode
│   ├── encoder/
│   │   ├── convnext.py         # ConvNeXt V2-T wrapper (Section 3)
│   │   ├── pyramid_neck.py     # Channel projection + L4 self-attention (Section 4)
│   │   └── precompute.py       # MSDA value maps, B3b KV projections (Section 5)
│   └── decoder/
│       ├── query_seed.py       # B1: multi-scale seed from all 4 neck levels + PE (Section 6.1)
│       ├── msda_decoder.py     # MSDA decoder: 3L deformable cross-attn + dual canvas + Q2Q + FFN (Section 6.2)
│       ├── spatial_canvas.py   # Dual Spatial Canvas: L2 (DWConv5×5) + L3 (DWConv3×3) scatter-gather-smooth (Section 6.5)
│       ├── global_cross_attn.py # B3b (L4 full attn) + canvas (Section 6.4)
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
| 2 | Pre-compute module (B3b L4 KV) | ✅ Done |
| 3–4 | B1 token construction + B2 local cross-attn | ✅ Done (will be replaced) |
| 5 | B3b global L4 cross-attn | ✅ Done |
| 6–7 | ~~L3 Self-Attention + pre-compute L3 KV~~ | ✅ Done (will be removed) |
| 8 | ~~B3a L3 global cross-attn~~ | ✅ Done (will be removed) |
| 9–10 | B5 token construction + FusedDecoder | ✅ Done (will be removed) |

**v14 new steps:**

| Step | What to build | How to validate |
|------|--------------|-----------------|
| 11 | **Uniform neck + L4 self-attn** (Section 4): Update pyramid_neck.py to project all levels to $d = 192$. L4 self-attention at 192d (6 heads × 32d). Remove V map projections (identity). Remove L3 self-attention, B3a L3 KV. | Shape test: L1–L4 all [B, 192, H/s, W/s]. L4 SA at 192d. Verify neck ~278K, L4 SA ~886K. |
| 12 | **B1 → multi-scale seed** (Section 6.1): Sample all 4 neck levels at query center (all 192-d), concat PE (32-d), project 800→192. Replace TokenConstructor. | Shape test: input coords [B, K, 2] → seed [B, K, 192] + pos [B, K, 192]. Verify W_seed 800→192 = ~154K. |
| 13 | **Dual Spatial Canvas** (Section 6.5): Build `spatial_canvas.py` with (a) dual canvas init from L2 and L3 (both $d$-dim, no V projection needed), (b) per-layer dual read-write-smooth: bilinear gather from both, concat+project via W_read (2d→d), gated fusion, scatter_add_ to both, asymmetric DWConv smooth (L2: DWConv5×5, L3: DWConv3×3), (c) final dual read via W_final (2d→d). | Shape test: F_L2 [B, 192, H/8, W/8], F_L3 [B, 192, H/16, W/16]. Verify scatter at correct positions for each level. Verify both DWConv residuals with correct kernel sizes. Verify gate ∈ (0,1). ~497K params. |
| 14 | **MSDA DecoderLayer** (Section 6.2): Build single layer with (a) offset prediction Linear, (b) attention weight prediction Linear, (c) bilinear sampling from neck features (identity V maps), (d) per-head aggregation, **(e) dual canvas read-write-smooth**, (f) Q2Q self-attn with spatial PE, (g) FFN. | Shape test: input h [B, K, 192] → output [B, K, 192]. Verify 96 sampling points (6×4×4). Verify both canvases evolve after each layer. Verify offsets near zero at init. |
| 15 | **MSDA Decoder stack + B3b with dual canvas** (Sections 6.2, 6.4): Stack 3 MSDADecoderLayers + 2 B3b, all with dual canvas sublayer. B3b KV at $d \to d$ (148K). Update SPD forward to: B1(seed) → init dual canvas → 3×MSDA → B3b → final dual canvas read → depth_head. Remove B2, B3a, B5, routing. | End-to-end forward pass test. Compare param count to expected ~33.3M. Verify both canvas states differ at each layer. |
| 16 | **Dense aux heads + depth head** (Sections 6.6, 7.2): Standalone depth_head.py (MLP + final canvas read + learnable scale). Dense aux Conv2d(192→1) on L2/L3. Update train.py for combined loss. | Verify depth head bias = log(2.5). Verify scale param = 0 at init. Verify dense aux loss decreasing. |
| 17 | **Full v14 pipeline**: train 10 epochs with all changes | Target: AbsRel < 0.20 (break through 0.23 ceiling). Monitor: pred range reaching 8m+, no regression after epoch 3, canvas evolving meaningfully (not collapsing to uniform), Q2Q attention patterns. |

**Key changes from v13.1:**
- **Uniform $d = 192$ neck** — all 4 levels project to $d$, eliminating bottleneck channels [64,128] and making MSDA V maps identity (−123K neck, −111K V maps, −2,656K L4 SA, −148K B3b KV)
- B1 (91-token constructor) → multi-scale seed (all 4 neck levels at center, each $d$-dim, + PE → 800→192)
- B2 (fixed-grid cross-attn) + B5 (fused cross-attn) → MSDA decoder (deformable cross-attn + dual canvas + Q2Q, 3 layers)
- **Removed B3a** (L3 cross-attn, 1,186K) — redundant with L3 canvas + MSDA deformable L3 sampling
- **Removed L3 Self-Attention** (3,542K) — ConvNeXt Stage 3's 9× DW k7 provides sufficient spatial mixing; L3 canvas provides decoded L3-resolution context
- **L4 Self-Attention at $d = 192$** (886K, was 3,542K at 384d) — uniform dimension across all attention modules
- +**Dual Spatial Canvas**: evolving dense feature maps at L2 (stride 8, DWConv5×5) + L3 (stride 16, DWConv3×3) resolutions, scatter-gather-smooth at every decoder layer — per-layer DWConv smooth (DySPN showed per-iteration kernels outperform shared), asymmetric kernels: larger for sparser L2, smaller for denser L3 to preserve depth boundaries (~883K params)
- B3b routing removed (no B5 to route to), KV projections now $d \to d$ (square, information-preserving)
- +Dense aux loss on L2/L3 (dense encoder gradient)
- +Learnable depth scale (break output ceiling)
- $\lambda_{\text{var}}$: 0.5 → 0.15 (stronger metric signal)
- Per-image SILog + data augmentation (from Exp 6–7 fixes)

### 10.4 Next Step: Inter-Canvas Communication (post-baseline)

**Problem:** L2 and L3 canvases evolve independently — no direct cross-resolution information flow. Queries provide indirect relay (read both → process → write both), but this is sparse (K=256 points) and bottlenecked through the query vector. DPT has explicit multi-scale fusion at every stage.

**Proposed fix:** Bidirectional L2↔L3 exchange after DWConv smooth each layer:

```
# Top-down: coarse semantic context helps fine level
F_L2 += gate_down × Conv1×1(bilinear_upsample(F_L3))    # L3 stride 16 → L2 stride 8

# Bottom-up: fine edge details help coarse level
F_L3 += gate_up × Conv1×1(bilinear_downsample(F_L2))     # L2 stride 8 → L3 stride 16
```

**Why it helps:**
- **L3→L2 (top-down):** L2 positions in flat surface regions (~5% scatter coverage at 256×320) get dense contextual guidance from L3 (80% coverage), instead of waiting for a query to land nearby.
- **L2→L3 (bottom-up):** L3 positions near depth edges get fine-grained correction from L2 — currently L3 only knows about edges if a query at the edge writes to it.

**Cost:** 2 × Conv1×1($d \to d$) = ~74K params (shared across layers) + 2 scalar gates. Compute negligible (bilinear resample on small maps).

**When to add:** After Step 17 baseline validation. If base v14 hits AbsRel < 0.20, this is an incremental improvement. If it plateaus above 0.20, this is the cheapest way to get DPT-like multi-scale fusion in our sparse framework.

**L1 canvas (stride 4) was considered and rejected:** At K=256, L1 scatter coverage is ~1.3% at 256×320 and ~0.3% at 480×640 — too sparse for meaningful canvas evolution. DWConv7×7 smooth on L1 adds ~880M MACs (nearly doubles decoder cost). L2 at stride 8 is the sweet spot between edge resolution and scatter coverage.
