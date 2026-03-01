# SPD v15.2 Architecture — Cross-Attention Local Path

Author: Claude
Date: 2026-03-01
Version: v15.2 (change from v15.1: replace convolutional local path with hierarchical cross-attention)

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Metric depth $\hat{d}_i$ at each queried pixel.

**Core idea:** The encoder runs once per image. The decoder is split into a **global path** (dense, $O(HW)$, runs once) and a **local path** (per-query, $O(K)$ at inference). v15.1's convolutional local path uses spatially-invariant operations (Conv3×3, bilinear upsample) that blur depth at boundaries. v15.2 replaces the entire convolutional local path with **two-stage cross-attention**: a depth query initialized from the global path progressively reads from multi-scale encoder features, mirroring the DPT refinenet hierarchy but with content-adaptive attention instead of fixed-weight convolution.

**Baseline:** v15.1 (DINOv2 ViT-S + DPT decoder with convolutional local path).

**Precedent:** PixelFormer (WACV 2023) poses depth estimation as pixel query refinement via cross-attention to encoder features. BinsFormer uses a transformer decoder where bin queries cross-attend to multi-scale features. Both demonstrate that the correct pattern is **query reads from features** — features are read-only context, not things that self-attend.

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
  L1: [B, D, 4H/14, 4W/14]     ≈ stride 3.5   ← Stage 2 keys
  L2: [B, D, 2H/14, 2W/14]     ≈ stride 7     ← Stage 1 keys
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
  LOCAL PATH  (per-query cross-attention)    [NEW in v15.2]
  │
  For each query position (u, v):
    0. Init query  q = grid_sample(stride_8_map, center)    [1 × D]
    1. Extract 5×5 from s8 and L2                           [50 × D]
    2. Stage 1:  q ← CrossAttn(q, s8+L2) + FFN             [1 × D]
    3. Extract 5×5 from L1                                  [25 × D]
    4. Stage 2:  q ← CrossAttn(q, L1) + FFN                [1 × D]
    5. MLP(q) → softplus → depth                            [scalar]
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

## 6. Local Path: Hierarchical Cross-Attention

### 6.1 Why Cross-Attention, Not Convolution or Self-Attention

**Why not convolution (v15.1's problem):** Conv3×3 and bilinear upsampling apply the same fixed weights everywhere. At a depth boundary, the kernel straddles both surfaces and averages them. The operation is content-blind.

**Why not self-attention (previous v15.2 draft):** Self-attention among all 75 feature tokens treats features as things to be mutually refined. But s8 is already refined by the global path; L1 and L2 are encoder outputs. None of them need modification by each other. They are **read-only context**. Self-attention wastes compute on token-to-token interactions (s8↔L1) that serve no purpose — the encoder (ViT with global self-attention) already performed cross-scale fusion.

**Why cross-attention is correct:** What needs refinement is the **depth prediction**, not the features. A single query token (initialized from the global path's coarse depth estimate) should **read from** the multi-scale features to gather the information it needs. Cross-attention captures this asymmetry: the query asks, the features answer.

This matches the proven pattern in depth estimation:
- **PixelFormer** (WACV 2023): pixel queries cross-attend to encoder features via Skip Attention Modules, progressively coarse-to-fine
- **BinsFormer**: bin queries cross-attend to multi-scale features through a transformer decoder
- **DPT itself**: the refinenet reads from encoder features at each scale (conv fusion is a fixed-weight form of cross-attention)

### 6.2 Architecture

The local path is a **two-stage transformer decoder** that mirrors the DPT refinenet hierarchy:

| | DPT refinenet (v15.1) | Cross-attention (v15.2) |
|---|---|---|
| **Stage 1** | RCU(s8) + RCU(L2) → add → ×2 → Conv1×1 | query ← CrossAttn(query, s8+L2) + FFN |
| **Stage 2** | RCU(rn2) + RCU(L1) → add → ×2 → Conv1×1 | query ← CrossAttn(query, L1) + FFN |
| **Head** | Conv3×3 → ×2 → Conv3×3+ReLU → Conv1×1 | MLP(D→128→1) → softplus |

Both follow the same coarse-to-fine flow: fuse stride-8 information first (s8+L2), then fuse stride-4 information (L1). The difference is fixed-weight convolution vs content-adaptive attention.

### 6.3 Query Initialization

The query is initialized as the **center token of the stride-8 map** at the query position:

$$q_0 = \text{grid\_sample}(\text{stride\_8\_map}, (u/s_8, v/s_8)) \in \mathbb{R}^D$$

This token already carries the global path's coarse depth estimate (fused L3+L4 through refinenet4 and refinenet3). The cross-attention stages refine it with local multi-scale detail.

### 6.4 Patch Extraction

For each query pixel $(u, v)$, extract $5 \times 5$ patches via bilinear `grid_sample`:

| Source Map | Stage | Patch Size | Tokens | Coverage |
|---|---|---|---|---|
| stride_8_map | 1 | $5 \times 5$ | 25 | ~35 × 35 px |
| L2 | 1 | $5 \times 5$ | 25 | ~35 × 35 px |
| L1 | 2 | $5 \times 5$ | 25 | ~17.5 × 17.5 px |

Stage 1 keys: 50 tokens (s8 + L2). Stage 2 keys: 25 tokens (L1).

Each key token receives additive embeddings:

**Scale embedding:** $e_s \in \mathbb{R}^D$ per source ($s \in \{s8, L2, L1\}$). In Stage 1, this distinguishes s8 tokens from L2 tokens.

**Positional embedding:** Learned 2D grid $P \in \mathbb{R}^{5 \times 5 \times D}$, shared across scales. Encodes relative spatial position within the $5 \times 5$ neighborhood.

$$k_{s,i,j} = f_{s,i,j} + e_s + P_{i,j}$$

### 6.5 Stage 1: Stride-8 Cross-Attention (s8 + L2)

The query cross-attends to the 50 stride-8 neighborhood tokens:

$$z = \text{LN}(q)$$
$$q' = q + \text{MHCrossAttn}(Q\!=\!z,\ K\!=\!\mathbf{k}_{s8{+}L2},\ V\!=\!\mathbf{k}_{s8{+}L2})$$
$$z' = \text{LN}(q')$$
$$q_1 = q' + \text{FFN}(z')$$

where $\text{MHCrossAttn}$ has $H = 4$ heads with $d_h = 16$, and $\text{FFN}$ is $\text{Linear}(D, 4D) \to \text{GELU} \to \text{Linear}(4D, D)$.

**What this computes:** "Given my coarse depth estimate ($q_0$ from the global path), what does the local stride-8 neighborhood add?" The attention weights are content-adaptive — at a depth boundary, the query attends to same-surface s8/L2 tokens and ignores cross-boundary ones.

### 6.6 Stage 2: Stride-4 Cross-Attention (L1)

The refined query cross-attends to the 25 stride-4 L1 tokens:

$$z = \text{LN}(q_1)$$
$$q' = q_1 + \text{MHCrossAttn}(Q\!=\!z,\ K\!=\!\mathbf{k}_{L1},\ V\!=\!\mathbf{k}_{L1})$$
$$z' = \text{LN}(q')$$
$$q_2 = q' + \text{FFN}(z')$$

**What this computes:** "Given my stride-8 understanding, what fine boundary detail does L1 provide?" L1 has 2× the spatial resolution of s8/L2, so it resolves edges that stride-8 features blur. The cross-attention reads this selectively.

### 6.7 Depth Prediction

$$\hat{d} = \text{softplus}\!\left(\text{Linear}_{128 \to 1}\!\left(\text{ReLU}\!\left(\text{Linear}_{D \to 128}(q_2)\right)\right)\right)$$

**Sub-pixel positioning** is handled by `grid_sample`. The query token $q_0$ is sampled at the exact fractional coordinate $(u/s_8, v/s_8)$, and all key patches are centered at the corresponding positions in their respective feature maps. No additional coordinate conditioning is needed.

### 6.8 Dense-Sparse Equivalence

**Proof:** Both modes perform identical steps per position:

1. `grid_sample` the center s8 token → query $q_0$
2. `grid_sample` $5 \times 5$ patches from s8, L2, L1 → keys
3. Two cross-attention stages with identical weights → $q_2$
4. MLP → depth

Same inputs, same function, same output. No padding/valid distinction exists. $\square$

### 6.9 Parameters

| Component | Count |
|---|---|
| Scale embeddings ($3 \times D$) | 192 |
| Positional embeddings ($25 \times D$) | 1,600 |
| **Stage 1 Cross-Attention** | |
| $\quad$ Q projection ($D \times D + D$) | 4,160 |
| $\quad$ K projection ($D \times D + D$) | 4,160 |
| $\quad$ V projection ($D \times D + D$) | 4,160 |
| $\quad$ Output projection ($D \times D + D$) | 4,160 |
| $\quad$ LayerNorm × 2 | 256 |
| $\quad$ FFN ($D \to 4D \to D$) | 33,088 |
| **Stage 2 Cross-Attention** | |
| $\quad$ (same structure as Stage 1) | 49,984 |
| **Depth MLP** ($D \to 128 \to 1$) | 8,449 |
| **Total Local** | **110,209** |

Compared to v15.1's 331,489 params: $-67\%$. The two cross-attention stages have comparable depth (4 nonlinear layers: 2× attention + 2× FFN) to v15.1's pipeline, but share weights across all spatial positions.

---

## 7. Training

### 7.1 Stride-4 Dense Training

At training time, the local path evaluates at **all stride-4 positions** ($100 \times 136 = 13{,}600$ positions). The KV projections are **amortized** across positions:

1. Upsample s8 and L2 to L1 resolution: $[B, D, 100, 136]$
2. Apply K and V projections as dense Conv1×1 on full maps (amortized)
3. Unfold into $5 \times 5$ neighborhoods → per-position KV patches
4. For each position: `grid_sample` center s8 → $q_0$, Q projection, cross-attention, FFN (Stage 1)
5. Unfold L1 KV patches, cross-attention + FFN (Stage 2)
6. Depth MLP → $[B, 1, 100, 136]$ stride-4 depth map
7. Resize to $H \times W$ for loss computation

The amortized KV projection makes dense training efficient — the expensive per-token projections are done once on the full feature maps, then neighborhoods are extracted via unfold (a free reshape).

### 7.2 Loss Function: SILog

$$\mathcal{L}_{\text{SILog}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \delta_i^2 - \lambda_{\text{var}} \left(\frac{1}{N}\sum_{i=1}^{N} \delta_i\right)^2 + \epsilon}$$

**$\lambda_{\text{var}} = 0.50$**, $\epsilon = 10^{-8}$.

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

## 8. Inference: Per-Query Cross-Attention

| Step | Operation | Shape |
|------|-----------|-------|
| 0 | `grid_sample` center from stride_8_map → $q_0$ | $[K, D]$ |
| 1 | `grid_sample` 5×5 from s8 + 5×5 from L2 | $[K, 50, D]$ |
| 2 | Add scale + positional embeddings to keys | $[K, 50, D]$ |
| 3 | **Stage 1:** CrossAttn($q_0$, s8+L2) + FFN → $q_1$ | $[K, D]$ |
| 4 | `grid_sample` 5×5 from L1 | $[K, 25, D]$ |
| 5 | Add scale + positional embeddings to keys | $[K, 25, D]$ |
| 6 | **Stage 2:** CrossAttn($q_1$, L1) + FFN → $q_2$ | $[K, D]$ |
| 7 | MLP(D→128→1) + softplus | $[K]$ → **K depth values** |

No convolutions, no recenter crops, no bilinear upsamples. Three `grid_sample` calls, two cross-attention stages, one MLP.

---

## 9. Parameter Summary

| Component | Parameters |
|---|---|
| **Encoder** (DINOv2 ViT-S/14) | ~22,000,000 |
| **Projection Neck** | 218,048 |
| **Global DPT** (refinenet4 + refinenet3) | 229,888 |
| **Local Cross-Attention** (2 stages + depth MLP) | 110,209 |
| **Total** | **~22,558,145** |

Decoder total (neck + global + local): **558,145** (~0.56M).

| Metric | v15.1 (conv) | v15.2 (cross-attn) | Change |
|---|---|---|---|
| Local params | 331,489 | 110,209 | $-67\%$ |
| Decoder params | 779,425 | 558,145 | $-28\%$ |

---

## 10. Computational Cost: Dense vs Sparse

All MACs computed at input resolution $350 \times 476$, $D = 64$.

### 10.1 Shared Cost (Encoder + Neck + Global)

$$C_{\text{shared}} = 24.9\text{G} + 0.16\text{G} + 0.16\text{G} = \mathbf{25.22\text{G}}$$

(Identical to v15.1 — see v15.1 doc for breakdown.)

### 10.2 Local Path: Per-Query MACs (Sparse)

Each query processes 1 query token against 50 + 25 key tokens:

| Operation | MACs |
|---|---|
| **Stage 1** (1 query, 50 keys) | |
| $\quad$ Q projection ($1 \times D^2$) | 4,096 |
| $\quad$ KV projection ($50 \times D \times 2D$) | 409,600 |
| $\quad$ Attention $QK^\top + \alpha V$ ($2 \times H \times 1 \times d_h \times 50$) | 6,400 |
| $\quad$ Output projection ($1 \times D^2$) | 4,096 |
| $\quad$ FFN ($1 \times D \times 4D \times 2$) | 32,768 |
| **Stage 2** (1 query, 25 keys) | |
| $\quad$ Q projection | 4,096 |
| $\quad$ KV projection ($25 \times D \times 2D$) | 204,800 |
| $\quad$ Attention | 3,200 |
| $\quad$ Output projection | 4,096 |
| $\quad$ FFN | 32,768 |
| **Depth MLP** | 8,320 |
| **Grid sample** (3× $5 \times 5$, bilinear) | 19,200 |
| **Per-query total** | **733K ≈ 0.73M** |

### 10.3 Local Path: Dense Training MACs

In dense mode, KV projections are **amortized** as Conv1×1 on full feature maps:

| Operation | MACs |
|---|---|
| Upsample s8, L2 to stride-4 | 7.0M |
| **Amortized KV projections** (Conv1×1 on full maps) | |
| $\quad$ Stage 1: K,V on s8\_up + L2\_up ($4 \times D^2 \times 100 \times 136$) | 222.8M |
| $\quad$ Stage 2: K,V on L1 ($2 \times D^2 \times 100 \times 136$) | 111.4M |
| **Per-position compute** (× 13,600 positions) | |
| $\quad$ Q proj + attention + out proj + FFN, Stage 1 | 47,360 each → 644M |
| $\quad$ Q proj + attention + out proj + FFN, Stage 2 | 44,160 each → 601M |
| $\quad$ Depth MLP | 8,320 each → 113M |
| Resize depth to $H \times W$ | 0.7M |
| **Dense local total** | **1,700M ≈ 1.70G** |

The amortization is the key insight: KV projections (which dominate sparse per-query cost at 84%) become shared Conv1×1 operations in dense mode. The per-position attention and FFN costs are tiny because only 1 query token is processed.

### 10.4 Total MACs Comparison

$$C_{\text{dense}} = 25.22\text{G} + 1.70\text{G} = \mathbf{26.92\text{G}}$$

$$C_{\text{sparse}}(K) = 25.22\text{G} + K \times 0.73\text{M}$$

| $K$ | Local MACs | Total MACs | vs Dense Total |
|---|---|---|---|
| 64 | 0.047G | 25.27G | 93.9% |
| 128 | 0.094G | 25.31G | 94.0% |
| 256 | 0.187G | 25.41G | 94.4% |
| 512 | 0.374G | 25.59G | 95.1% |
| 1024 | 0.748G | 25.97G | 96.5% |
| **2,329** | **1.70G** | **26.92G** | **100%** |

**Break-even point:**

$$K^* = \frac{C_{\text{local,dense}}}{C_{\text{local,query}}} = \frac{1{,}700\text{M}}{0.73\text{M}} \approx 2{,}329 \text{ queries}$$

### 10.5 Comparison with v15.1

| Metric | v15.1 (conv) | v15.2 (cross-attn) | Change |
|---|---|---|---|
| Local params | 331,489 | 110,209 | $-67\%$ |
| Sparse per-query MACs | 8.48M | 0.73M | $\mathbf{-91\%}$ |
| Dense local MACs | 5,846M | 1,700M | $-71\%$ |
| Dense total MACs | 31.07G | 26.92G | $-13\%$ |
| Break-even $K^*$ | 689 | 2,329 | $+3.4\times$ |

The cross-attention approach is cheaper than v15.1 in **both** dense and sparse:

- **Sparse: $11.6\times$ cheaper per query.** v15.1 runs 4 RCUs (8× Conv3×3) on per-query patches. v15.2 runs 2 cross-attention stages on 1 query token — the only expensive part is KV projection of 75 key tokens, which is far less than 8 convolutional layers.

- **Dense: 13% cheaper overall.** The KV projections amortize as simple Conv1×1 on full feature maps. The per-position cost (Q proj + tiny attention + FFN) is much less than v15.1's 4 RCUs on full maps.

- **Break-even is higher** (2,329 vs 689) because sparse-to-dense cost ratio is lower — cross-attention benefits more from amortization than convolution does. But K* = 2,329 still means sparse wins for any practical query count.

### 10.6 Training Memory Estimate

At batch size 2 with AMP (bfloat16):

| Component | Memory |
|---|---|
| Encoder features + gradients | ~1.5G |
| Neck + global path | ~0.3G |
| KV projected maps (6 maps × 100×136 × 64, bf16) | ~0.01G |
| Per-position query states + gradients | ~0.1G |
| Optimizer states (AdamW, FP32) | ~0.4G |
| Remaining (activations, buffers) | ~1.5G |
| **Estimated total** | **~3.8G** |

Fits comfortably in 8GB VRAM with significant headroom.

---

## 11. File Structure

| File | Component |
|---|---|
| `src/models/spd.py` | Top-level model, routes train/infer |
| `src/models/encoder_vits/vit_s.py` | DINOv2 ViT-S encoder |
| `src/models/encoder_vits/pyramid_neck.py` | Projection neck |
| `src/models/decoder/global_dpt.py` | Global path (refinenet4 + refinenet3) |
| `src/models/decoder/local_attn.py` | **Local cross-attention path (NEW)** |
| `src/config.py` | Hyperparameters |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Dense + sparse evaluation |
| `src/utils/losses.py` | SILog loss |
| `src/data/nyu_dataset.py` | NYU Depth V2 dataloader |
