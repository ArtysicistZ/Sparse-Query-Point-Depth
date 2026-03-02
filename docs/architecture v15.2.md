# SPD v15.2 Architecture — RCU + Cross-Attention Local Path

Author: Claude
Date: 2026-03-01
Version: v15.2 (change from v15.1: replace convolutional local path with RCU spatial refinement + hierarchical cross-attention fusion)

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Metric depth $\hat{d}_i$ at each queried pixel.

**Core idea:** The encoder runs once per image. The decoder is split into a **global path** (dense, $O(HW)$, runs once) and a **local path** (per-query, $O(K)$ at inference). v15.1's convolutional local path uses spatially-invariant operations (Conv3×3, bilinear upsample) that blur depth at boundaries. v15.2 replaces the local path with a two-phase design:

1. **RCU phase (spatial refinement):** Three independent RCUs process 9×9 patches from each feature source (s8, L2, L1) via valid Conv3×3 → 5×5 refined patches. This adds local spatial context that plain ViT features lack.
2. **Cross-attention phase (multi-scale fusion):** A single query token (initialized from s8 center) reads from the three refined feature sources via cross-attention, one source per layer — content-adaptive fusion instead of fixed-weight addition.

This mirrors DPT's RCU-before-refinenet pattern: RCU processes features, then refinenet fuses them. v15.2 replaces the fusion (addition → cross-attention) while retaining the spatial processing (RCU).

**Baseline:** v15.1 (DINOv2 ViT-S + DPT decoder with convolutional local path).

**Precedent:** PixelFormer (WACV 2023) poses depth estimation as pixel query refinement via cross-attention to encoder features. BinsFormer uses a transformer decoder where bin queries cross-attend to multi-scale features (L=3 layers per scale, 9 layers total). Mask2Former uses 9 decoder layers cycling through 3 scales. All demonstrate: **one source per cross-attention layer, never mixed.**

---

## 2. Architecture Overview

```
RGB Image [H × W × 3]
  │
  ▼
DINOv2 ViT-S/14  (pretrained, fine-tuned)
  Features tapped at layers {2, 5, 8, 11}
  All: [B, 384, H/14, W/14]
  │
  ▼
Projection Neck  (Conv1×1 + LN + spatial resize per level)
  L1: [B, D, 4H/14, 4W/14]     ≈ stride 3.5
  L2: [B, D, 2H/14, 2W/14]     ≈ stride 7
  L3: [B, D,  H/14,  W/14]     = stride 14     ← global refinenet3
  L4: [B, D,  H/28,  W/28]     ≈ stride 28     ← global refinenet4
  │
  ▼ ────────────────────────────────────────────
  GLOBAL PATH  (dense, once per image)       [unchanged from v15.1]
  │
  refinenet4:  RCU(L4) → ×2 bilinear → Conv1×1
  refinenet3:  RCU(L3) + add → RCU → ×2 bilinear → Conv1×1
               = stride_8_map  [B, D, 2H/14, 2W/14]
  │
  ▼ ────────────────────────────────────────────
  LOCAL PATH  (RCU + cross-attention)        [NEW in v15.2]
  │
  For each query position (u, v):
  ┌─ Phase 1: Spatial Refinement ──────────────────────┐
  │  grid_sample 9×9 from s8, L2, L1                   │
  │  RCU_s8(9×9) → 5×5   (25 tokens, valid conv)       │
  │  RCU_L2(9×9) → 5×5   (25 tokens)                   │
  │  RCU_L1(9×9) → 5×5   (25 tokens)                   │
  └─────────────────────────────────────────────────────┘
  ┌─ Phase 2: Cross-Attention Fusion ──────────────────┐
  │  q₀ = center of s8 RCU output           [1 × D]    │
  │  Layer 1: q ← CrossAttn(q, s8_rcu)  + FFN          │
  │  Layer 2: q ← CrossAttn(q, L2_rcu)  + FFN          │
  │  Layer 3: q ← CrossAttn(q, L1_rcu)  + FFN          │
  └─────────────────────────────────────────────────────┘
     MLP(q) → softplus → depth              [scalar]
```

Where $D = 64$.

---

## 3. Encoder: DINOv2 ViT-S/14

| Parameter | Value |
|---|---|
| Architecture | Vision Transformer Small (ViT-S/14) |
| Patch size | $14 \times 14$ |
| Embedding dim | 384 |
| Transformer blocks | 12, each with 6 attention heads ($384 / 6 = 64$ dim/head) |
| Pretraining | DINOv2 self-supervised (DINO + iBOT) on LVD-142M — zero depth labels |
| Parameters | ~22M |

### 3.1 Feature Extraction

DINOv2 is a plain (non-hierarchical) ViT. Given input $I \in \mathbb{R}^{B \times 3 \times H \times W}$, the image is split into $14 \times 14$ patches, yielding a token grid of size:

$$h = \lfloor H / 14 \rfloor, \quad w = \lfloor W / 14 \rfloor$$

All transformer blocks operate at the same spatial resolution $h \times w$ with uniform channel dimension 384. We tap intermediate features at blocks $\{2, 5, 8, 11\}$ (0-indexed), creating a pseudo-hierarchy:

$$F_l \in \mathbb{R}^{B \times 384 \times h \times w}, \quad l \in \{L1, L2, L3, L4\}$$

### 3.2 Concrete Dimensions at $350 \times 476$

$$h = 350 / 14 = 25, \quad w = 476 / 14 = 34$$

All four feature maps: $[B, 384, 25, 34]$.

---

## 4. Projection Neck

The neck projects all four levels from encoder dimension 384 to decoder dimension $D = 64$, and resizes each to a different spatial scale.

### 4.1 Per-Level Projection

$$\hat{F}_l = \text{LN}\!\left(\text{Conv}_{1 \times 1}^{384 \to D}(F_l)\right)$$

### 4.2 Spatial Resize

| Level | Resize Operation | Output Size ($350 \times 476$ input) | Effective Stride |
|-------|-----------------|--------------------------------------|-----------------|
| L1 | $\text{ConvTranspose}_{4 \times 4}^{D \to D}(\text{stride}=4)$ | $[B, 64, 100, 136]$ | ~3.5 |
| L2 | $\text{ConvTranspose}_{2 \times 2}^{D \to D}(\text{stride}=2)$ | $[B, 64, 50, 68]$ | ~7 |
| L3 | Identity | $[B, 64, 25, 34]$ | 14 |
| L4 | $\text{Conv}_{3 \times 3}^{D \to D}(\text{stride}=2, \text{pad}=1)$ | $[B, 64, 13, 17]$ | ~28 |

### 4.3 Parameters

| Component | Count |
|---|---|
| 4× Conv1×1(384→64) | 98,560 |
| 4× LayerNorm(64) | 512 |
| ConvTranspose(64→64, k=4, s=4) | 65,600 |
| ConvTranspose(64→64, k=2, s=2) | 16,448 |
| Conv2d(64→64, k=3, s=2, p=1) | 36,928 |
| **Total Neck** | **218,048** |

---

## 5. Global Path: DPT FPN Fusion

**Unchanged from v15.1.** Fuses L4, L3 into a stride-8 dense feature map.

### 5.1 Residual Convolutional Unit (RCU)

$$\text{RCU}(x) = x + \text{Conv}_{3 \times 3}\!\left(\text{GELU}\!\left(\text{Conv}_{3 \times 3}\!\left(\text{GELU}(x)\right)\right)\right)$$

Parameters per RCU: $2 \times (D^2 \cdot 9 + D) = 73{,}856$.

### 5.2 Refinenet4 + Refinenet3

$$r_4 = \text{Conv}_{1 \times 1}\!\left(\text{Upsample}_{\times 2}\!\left(\text{RCU}(L4)\right)\right)$$

$$\text{stride\_8\_map} = \text{Conv}_{1 \times 1}\!\left(\text{Upsample}_{\times 2}\!\left(\text{RCU}\!\left(\text{RCU}(L3) + r_4\right)\right)\right) \in \mathbb{R}^{B \times D \times 50 \times 68}$$

### 5.3 Parameters

| Component | Count |
|---|---|
| rcu_L4 + rcu_L4_out | 78,016 |
| rcu_L3 + rcu_L3_merge + rcu_L3_out | 151,872 |
| **Total Global** | **229,888** |

---

## 6. Local Path: RCU + Hierarchical Cross-Attention

### 6.1 Why Two Phases

**Phase 1 (RCU) solves a ViT problem.** DINOv2 is a plain ViT — global self-attention on 14×14 patches, zero local convolution. The features are semantically rich but spatially coarse: each token is a patch-level embedding with no local edge or texture processing. Conv3×3 (via RCU) adds the local spatial refinement that ViT entirely lacks. DPT applies RCU to features before fusion; we do the same.

**Phase 2 (cross-attention) solves a DPT problem.** DPT's refinenet fuses multi-scale features via addition — content-blind, applies the same operation everywhere. At a depth boundary, addition averages across surfaces. Cross-attention computes content-dependent weights: the query attends to same-surface tokens and ignores cross-boundary ones. This is the proven pattern in PixelFormer (per-pixel cross-attention to encoder features), BinsFormer (3 layers per scale), and Mask2Former (cycling through scales).

**Why one source per cross-attention layer.** BinsFormer, Mask2Former, and PixelFormer all feed one scale to one decoder layer — never mixing scales. Each feature source (s8, L2, L1) carries fundamentally different information: s8 is the processed global path output, L2 is a raw mid-level encoder feature, L1 is a raw low-level encoder feature. Separate layers let each cross-attention specialize.

### 6.2 Phase 1: Per-Source RCU (Spatial Refinement)

For each query pixel $(u, v)$, extract $9 \times 9$ patches from three feature sources and apply independent RCUs with valid convolution:

| Source | Resolution | Grid Sample | After RCU (valid) | Physical Coverage |
|--------|-----------|-------------|-------------------|-------------------|
| stride_8_map | $50 \times 68$ | $9 \times 9$ | $5 \times 5$ (25 tokens) | ~63 × 63 px |
| L2 | $50 \times 68$ | $9 \times 9$ | $5 \times 5$ (25 tokens) | ~63 × 63 px |
| L1 | $100 \times 136$ | $9 \times 9$ | $5 \times 5$ (25 tokens) | ~31.5 × 31.5 px |

Each RCU has independent weights:

$$\text{RCU}_s(x) = x + \text{Conv}_{3 \times 3}\!\left(\text{GELU}\!\left(\text{Conv}_{3 \times 3}\!\left(\text{GELU}(x)\right)\right)\right), \quad s \in \{s8, L2, L1\}$$

Valid convolution spatial trace: $9 \xrightarrow{\text{Conv3×3}} 7 \xrightarrow{\text{Conv3×3}} 5$ (each Conv3×3 shrinks by 2).

**Why s8 needs its own RCU despite global path processing:** The global path's RCUs serve a different purpose — fusing L4+L3 into the stride-8 map. The local RCU_s8 adapts the s8 features specifically for per-query reading, processing the local $9 \times 9$ neighborhood centered at the query position. At 74K parameters, the cost is negligible.

### 6.3 Phase 2: Per-Source Cross-Attention (Multi-Scale Fusion)

**Query initialization.** $q_0$ is the center pixel (position $(2,2)$) of the $\text{RCU}_{s8}$ output:

$$q_0 = \text{RCU}_{s8}(\text{patch}_{s8})[2, 2] \in \mathbb{R}^D$$

This token carries global path context (fused L4+L3 through refinenet) plus local spatial context from the RCU. Better than a raw point sample.

**Key embeddings.** Each of the 25 tokens per source receives additive embeddings:

$$k_{s,i,j} = f_{s,i,j}^{\text{rcu}} + e_s + P_{i,j}$$

- **Scale embedding:** $e_s \in \mathbb{R}^D$ per source ($s \in \{s8, L2, L1\}$), distinguishes which feature source the keys come from.
- **Positional embedding:** Learned 2D grid $P \in \mathbb{R}^{5 \times 5 \times D}$, shared across sources. Encodes relative spatial position within the $5 \times 5$ neighborhood.

**Cross-attention structure.** Each layer uses pre-norm multi-head cross-attention with $H = 4$ heads ($d_h = D/H = 16$), followed by a pre-norm FFN:

$$z = \text{LN}(q), \quad q' = q + \text{MHCrossAttn}(Q\!=\!z,\ K\!=\!\mathbf{k}_s,\ V\!=\!\mathbf{k}_s)$$
$$z' = \text{LN}(q'), \quad q_{\text{out}} = q' + \text{FFN}(z')$$

where $\text{FFN}(x) = \text{Linear}_{4D \to D}(\text{GELU}(\text{Linear}_{D \to 4D}(x)))$.

**Layer 1 — s8 cross-attention:**

$$q_0 \xrightarrow{\text{CrossAttn}(\cdot,\ \mathbf{k}_{s8}) + \text{FFN}} q_1$$

"Given my coarse depth from the global path, what does the local s8 neighborhood add?" The s8 RCU tokens carry the global path's structural understanding, refined with local spatial context. The cross-attention reads the 5×5 s8 neighborhood with content-adaptive weights.

**Layer 2 — L2 cross-attention:**

$$q_1 \xrightarrow{\text{CrossAttn}(\cdot,\ \mathbf{k}_{L2}) + \text{FFN}} q_2$$

"What mid-level edges and textures does L2 reveal?" L2 (ViT block 5) captures intermediate representations — edges, texture gradients — that the global path (L3+L4 only) never saw.

**Layer 3 — L1 cross-attention:**

$$q_2 \xrightarrow{\text{CrossAttn}(\cdot,\ \mathbf{k}_{L1}) + \text{FFN}} q_3$$

"What fine boundary detail does L1 provide?" L1 (ViT block 2) has 2× the spatial density of s8/L2, resolving edges that stride-8 features blur.

This coarse-to-fine order (s8 → L2 → L1) mirrors the DPT refinenet hierarchy: refinenet2 fuses s8+L2, refinenet1 fuses with L1.

### 6.4 Depth Prediction

$$\hat{d} = \text{softplus}\!\left(\text{Linear}_{128 \to 1}\!\left(\text{ReLU}\!\left(\text{Linear}_{D \to 128}(q_3)\right)\right)\right)$$

**Sub-pixel positioning** is handled by `grid_sample`. The 9×9 patches are extracted at the exact fractional coordinate, and the RCU/cross-attention processes them. No additional coordinate conditioning is needed.

### 6.5 Nonlinear Depth

The query passes through 12 nonlinear transformations from feature extraction to depth:

| Phase | Operations | Nonlinear Layers |
|-------|-----------|-----------------|
| RCU_s8 | 2× (GELU + Conv3×3) | 2 |
| RCU_L2 | 2× (GELU + Conv3×3) | 2 |
| RCU_L1 | 2× (GELU + Conv3×3) | 2 |
| CrossAttn Layer 1 | attention + GELU-FFN | 2 |
| CrossAttn Layer 2 | attention + GELU-FFN | 2 |
| CrossAttn Layer 3 | attention + GELU-FFN | 2 |
| **Total** | | **12** |

Compared to v15.1's 8 Conv3×3 layers (all content-blind), v15.2 has 50% more depth and 3 of the 6 fusion operations are content-adaptive.

### 6.6 Dense-Sparse Equivalence

**RCU phase.** Valid Conv3×3 on 9×9 patches (sparse) = padded Conv3×3 on full feature maps (dense) for interior pixels. The 9×9 input fully contains the 5×5 receptive field of 2× Conv3×3 — all 25 output positions use identical computation in both modes. $\checkmark$

**Cross-attention phase.** Both modes apply identical operations per position: Q projection on 1 query token, KV projection on 25 keys, multi-head attention, output projection, FFN. The keys come from the same RCU outputs (padded-dense or valid-sparse), which are equivalent at interior positions. $\checkmark$

**Dense evaluation resolution.** Dense training evaluates at stride-4 positions ($100 \times 136 = 13{,}600$ positions). The s8 and L2 RCU outputs ($50 \times 68$) are bilinearly upsampled to stride-4 before KV projection and unfold. This parallels v15.1 where refinenet2 processes at stride-8 and upsamples to stride-4. $\square$

### 6.7 Parameters

| Component | Count |
|---|---|
| **RCU Phase** | |
| $\quad$ RCU_s8 (2× Conv3×3($D$, $D$)) | 73,856 |
| $\quad$ RCU_L2 (2× Conv3×3($D$, $D$)) | 73,856 |
| $\quad$ RCU_L1 (2× Conv3×3($D$, $D$)) | 73,856 |
| **Cross-Attention Phase** | |
| $\quad$ Scale embeddings ($3 \times D$) | 192 |
| $\quad$ Positional embeddings ($25 \times D$) | 1,600 |
| $\quad$ Layer 1: Q,K,V,O + LN×2 + FFN | 49,984 |
| $\quad$ Layer 2: (same structure) | 49,984 |
| $\quad$ Layer 3: (same structure) | 49,984 |
| **Depth MLP** ($D \to 128 \to 1$) | 8,449 |
| **Total Local** | **381,761** |

---

## 7. Training

### 7.1 Dense Execution

At training time, the local path evaluates at all stride-4 positions ($100 \times 136 = 13{,}600$ positions):

1. **RCU on full maps** (padded Conv3×3, amortized):
   - $\text{RCU}_{s8}(\text{stride\_8\_map})$: $[B, D, 50, 68] \to [B, D, 50, 68]$
   - $\text{RCU}_{L2}(L2)$: $[B, D, 50, 68] \to [B, D, 50, 68]$
   - $\text{RCU}_{L1}(L1)$: $[B, D, 100, 136] \to [B, D, 100, 136]$

2. **Upsample** s8\_rcu, L2\_rcu from $50 \times 68$ to $100 \times 136$ (bilinear ×2)

3. **KV projections** as Conv1×1 on all 3 maps at $100 \times 136$ (amortized across positions)

4. **Unfold** $5 \times 5$ neighborhoods from each KV map $\to$ per-position keys

5. $q_0$ = upsampled s8\_rcu values $\to$ $[B, D, 100, 136]$, one query per position

6. **Per-position cross-attention:** 3 layers (s8 → L2 → L1), each with 25 keys

7. **Depth MLP** $\to$ $[B, 1, 100, 136]$ stride-4 depth map, resize to $H \times W$

The RCU phase is standard dense convolution (amortized). The KV projections are Conv1×1 on full maps (amortized). Only Q projection, attention computation, and FFN are per-position — and with 1 query token per position, these are tiny.

### 7.2 Loss Function: SILog

$$\mathcal{L}_{\text{SILog}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \delta_i^2 - \lambda_{\text{var}} \left(\frac{1}{N}\sum_{i=1}^{N} \delta_i\right)^2 + \epsilon}$$

where $\delta_i = \ln \hat{d}_i - \ln d_i^*$.

**$\lambda_{\text{var}} = 0.50$**, $\epsilon = 10^{-8}$. Reduced from v15.1's 0.85 to allow 3.3× more scale gradient — fixes the scale convergence stall observed in v15.1 training (optimal_scale stuck at ~0.82).

### 7.3 Optimizer and Schedule

| Parameter | Value |
|---|---|
| Optimizer | AdamW, weight decay $0.01$ |
| Encoder LR | $5 \times 10^{-6}$ |
| Decoder LR | $1 \times 10^{-4}$ |
| Warmup | Linear, 500 steps, start factor $0.01$ |
| Schedule | Cosine annealing after warmup, $\eta_{\min} = 10^{-6}$ |
| AMP | bfloat16 |
| Batch size | 2 |
| Gradient clipping | Max norm $1.0$ |
| Epochs | 10 (237,920 total steps) |

### 7.4 Dataset

| Parameter | Value |
|---|---|
| Dataset | NYU Depth V2 |
| Train split | 47,584 images |
| Val split | 654 images |
| Resolution | $350 \times 476$ |
| Depth format | Float32 meters, range $[0, 10]$ m |
| Augmentation | BTS-style (rotation, gamma, brightness, color jitter, horizontal flip) |

---

## 8. Inference: Per-Query RCU + Cross-Attention

| Step | Operation | Shape |
|------|-----------|-------|
| 0 | `grid_sample` 9×9 from s8, L2, L1 | 3× $[K, D, 9, 9]$ |
| 1 | RCU_s8 valid (9→5), RCU_L2 valid (9→5), RCU_L1 valid (9→5) | 3× $[K, D, 5, 5]$ |
| 2 | Flatten to tokens, add scale + positional embeddings | 3× $[K, 25, D]$ |
| 3 | $q_0$ = center of s8\_rcu output (index $[2,2]$) | $[K, D]$ |
| 4 | **Layer 1:** CrossAttn($q$, s8\_rcu) + FFN | $[K, D]$ |
| 5 | **Layer 2:** CrossAttn($q$, L2\_rcu) + FFN | $[K, D]$ |
| 6 | **Layer 3:** CrossAttn($q$, L1\_rcu) + FFN | $[K, D]$ |
| 7 | MLP($D \to 128 \to 1$) + softplus | $[K]$ → **K depth values** |

Three `grid_sample` calls (9×9 each), three valid-conv RCUs, three cross-attention layers, one MLP. No bilinear upsamples, no recenter crops.

---

## 9. Parameter Summary

| Component | Parameters |
|---|---|
| **Encoder** (DINOv2 ViT-S/14) | ~22,000,000 |
| **Projection Neck** | 218,048 |
| **Global DPT** (refinenet4 + refinenet3) | 229,888 |
| **Local RCU + Cross-Attention** (3 RCU + 3 CA + MLP) | 381,761 |
| **Total** | **~22,829,697** |

Decoder total (neck + global + local): **829,697** (~0.83M).

| Metric | v15.1 (conv) | v15.2 (RCU + CA) | Change |
|---|---|---|---|
| Local params | 331,489 | 381,761 | $+15\%$ |
| Decoder params | 779,425 | 829,697 | $+6\%$ |

The parameter increase comes entirely from the 3 local RCUs (222K) replacing v15.1's 4 RCUs shared between refinenet2/refinenet1 (296K) — net +15%. The cross-attention layers (150K) are cheaper than v15.1's output head (28K) plus the rn_out projections (8K), offset by the depth MLP (8K) and embeddings (2K).

---

## 10. Computational Cost: Dense vs Sparse

All MACs computed at input resolution $350 \times 476$, $D = 64$.

### 10.1 Shared Cost (Encoder + Neck + Global)

$$C_{\text{shared}} = 24.9\text{G} + 0.16\text{G} + 0.16\text{G} = \mathbf{25.22\text{G}}$$

(Identical to v15.1 — see v15.1 doc for breakdown.)

### 10.2 Local Path: Per-Query MACs (Sparse)

| Operation | MACs |
|---|---|
| **Grid sample** (3× $9 \times 9$, bilinear) | 62K |
| **RCU_s8** (Conv3×3 valid $9 \to 7$: $49 \times D^2 \times 9$, Conv3×3 valid $7 \to 5$: $25 \times D^2 \times 9$) | 2,728K |
| **RCU_L2** (same) | 2,728K |
| **RCU_L1** (same) | 2,728K |
| **Layer 1** (s8, 25 keys) | |
| $\quad$ Q projection ($1 \times D^2$) | 4K |
| $\quad$ KV projection ($25 \times 2D^2$) | 205K |
| $\quad$ Attention ($2 \times H \times d_h \times 25$) | 3K |
| $\quad$ Output projection ($1 \times D^2$) | 4K |
| $\quad$ FFN ($D \times 4D \times 2$) | 33K |
| **Layer 2** (L2, 25 keys, same structure) | 249K |
| **Layer 3** (L1, 25 keys, same structure) | 249K |
| **Depth MLP** ($D \to 128 \to 1$) | 8K |
| **Per-query total** | **9,001K ≈ 9.0M** |

RCU dominates at 91% of per-query cost. The 3 cross-attention layers together cost only 747K (8%) — the attention computation itself is tiny because there is only 1 query token.

### 10.3 Local Path: Dense Training MACs

| Operation | MACs |
|---|---|
| **RCU on full maps** | |
| $\quad$ RCU_s8 on $50 \times 68$ (2× Conv3×3) | 250.7M |
| $\quad$ RCU_L2 on $50 \times 68$ (2× Conv3×3) | 250.7M |
| $\quad$ RCU_L1 on $100 \times 136$ (2× Conv3×3) | 1,002.7M |
| **Upsample** s8\_rcu, L2\_rcu to $100 \times 136$ | 7.0M |
| **Amortized KV projections** (Conv1×1 on $100 \times 136$) | |
| $\quad$ 3 layers × K,V projections ($6 \times D^2 \times 13{,}600$) | 334.2M |
| **Per-position compute** (× 13,600 positions) | |
| $\quad$ Q proj + attention + out proj + FFN, Layer 1 | 44,160 each → 600.6M |
| $\quad$ Layer 2 (same) | 600.6M |
| $\quad$ Layer 3 (same) | 600.6M |
| $\quad$ Depth MLP | 8,320 each → 113.2M |
| Resize to $H \times W$ | 0.7M |
| **Dense local total** | **3,761M ≈ 3.76G** |

### 10.4 Total MACs Comparison

$$C_{\text{dense}} = 25.22\text{G} + 3.76\text{G} = \mathbf{28.98\text{G}}$$

$$C_{\text{sparse}}(K) = 25.22\text{G} + K \times 9.0\text{M}$$

| $K$ | Local MACs | Total MACs | vs Dense Total |
|---|---|---|---|
| 64 | 0.58G | 25.80G | 89.0% |
| 128 | 1.15G | 26.37G | 91.0% |
| 256 | 2.30G | 27.52G | 95.0% |
| **418** | **3.76G** | **28.98G** | **100%** |
| 512 | 4.61G | 29.83G | 102.9% |

**Break-even point:**

$$K^* = \frac{C_{\text{local,dense}}}{C_{\text{local,query}}} = \frac{3{,}761\text{M}}{9.0\text{M}} \approx 418 \text{ queries}$$

### 10.5 Comparison with v15.1

| Metric | v15.1 (conv) | v15.2 (RCU + CA) | Change |
|---|---|---|---|
| Local params | 331,489 | 381,761 | $+15\%$ |
| Sparse per-query MACs | 8.48M | 9.0M | $+6\%$ |
| Dense local MACs | 5,846M | 3,761M | $\mathbf{-36\%}$ |
| Dense total MACs | 31.07G | 28.98G | $-7\%$ |
| Break-even $K^*$ | 689 | 418 | $-39\%$ |
| Nonlinear depth | 8 (all conv) | 12 (6 conv + 6 attn) | $+50\%$ |

Per-query cost is comparable ($+6\%$). Dense training is significantly cheaper ($-36\%$ local) because v15.1 must run RCUs at stride-4 ($100 \times 136$) and the full head at stride-2/stride-1 ($200 \times 272$, $400 \times 544$), while v15.2's cross-attention operates on single query tokens — the per-position cost is dominated by the tiny FFN ($33\text{K}$), not spatial convolutions.

The lower break-even ($K^* = 418$) means sparse inference becomes advantageous at fewer queries.

### 10.6 Training Memory Estimate

At batch size 2 with AMP (bfloat16):

| Component | Memory |
|---|---|
| Encoder features + gradients | ~1.5G |
| Neck + global path | ~0.3G |
| RCU feature maps + gradients (3 maps at 100×136) | ~0.1G |
| KV projected maps (6 maps × 100×136 × 64, bf16) | ~0.01G |
| Per-position query states + gradients | ~0.2G |
| Optimizer states (AdamW, FP32) | ~0.5G |
| Remaining (activations, buffers) | ~1.5G |
| **Estimated total** | **~4.1G** |

Fits comfortably in 8GB VRAM.

---

## 11. Optional: Multi-Round Cycling

Following Mask2Former's pattern (3 rounds × 3 scales = 9 layers), the cross-attention phase can be extended to **2 rounds** of cycling through all three sources:

```
Round 1: CA(q, s8) → CA(q, L2) → CA(q, L1)
Round 2: CA(q, s8) → CA(q, L2) → CA(q, L1)
```

6 cross-attention layers total, each with its own weights. In round 2, the query re-reads s8 with full multi-scale knowledge from round 1, asking content-adaptive questions informed by L1's boundary detail.

| Metric | 1 round (3 CA) | 2 rounds (6 CA) |
|---|---|---|
| Local params | 381,761 | 531,713 |
| Per-query MACs | 9.0M | 9.75M |
| Dense local MACs | 3,761M | 5,897M |
| Dense total MACs | 28.98G | 31.12G |

2 rounds matches v15.1's dense budget almost exactly (31.12G vs 31.07G). Start with 1 round; add round 2 if the model underfits.

---

## 12. File Structure

| File | Component |
|---|---|
| `src/models/spd.py` | Top-level model, routes train/infer |
| `src/models/encoder_vits/vit_s.py` | DINOv2 ViT-S encoder |
| `src/models/encoder_vits/pyramid_neck.py` | Projection neck |
| `src/models/decoder/global_dpt.py` | Global path (refinenet4 + refinenet3) |
| `src/models/decoder/rcu.py` | Residual Convolutional Unit |
| `src/models/decoder/local_attn.py` | **Local RCU + cross-attention path (NEW)** |
| `src/config.py` | Hyperparameters |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Dense + sparse evaluation |
| `src/utils/losses.py` | SILog loss |
| `src/data/nyu_dataset.py` | NYU Depth V2 dataloader |
