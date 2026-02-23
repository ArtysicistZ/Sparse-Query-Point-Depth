# Research Plan: SPD v13 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-23
Version: v13 (first trial — frozen DAv2-S encoder, sparse B1–B5 decoder, NYU Depth V2)

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Depth $\hat{d}_i$ at each queried pixel.

**Constraint:** No dense $H \times W$ depth decoding. Shared encoder pass once, then sparse per-query decoding:

$$T_{\text{total}}(K) = T_{\text{encode}}(I) + T_{\text{decode}}(K \mid \text{features})$$

**Baseline:** DAv2-S (ViT-S + DPT decoder) produces dense $H \times W$ depth, then sample at query points.
**Ours:** Same ViT-S encoder (frozen) + lightweight sparse B1–B5 decoder → depth at K points only.

---

## 2. Architecture Overview

```
RGB Image (H×W)
  │
  ▼
Frozen DAv2-S ViT-S encoder
  Extract at layers [2, 5, 8, 11] → 4 × [H_p×W_p×384]
  │
  ▼
Pyramid Neck (trainable)
  L1 [4H_p × 4W_p × 64]    (bilinear ↑4×, Conv 1×1)
  L2 [2H_p × 2W_p × 128]   (bilinear ↑2×, Conv 1×1)
  L3 [H_p × W_p × 192]     (identity,     Conv 1×1)
  L4 [H_p/2 × W_p/2 × 384] (AvgPool  ↓2×, Conv 1×1)
  │
  ▼
Pre-compute (once per image)
  B3 KV projections, B4 W_V projections, calibration (s, b)
  │
  ▼
Per-query decoder B1–B5 (×K queries in parallel)
  B1: 91 local tokens + query seed h^(0)
  B2: 2-layer cross-attn (91 tokens) → h^(2)
  B3: 2-layer L4 cross-attn (324 tokens) → h^(2)', 32 anchors
  B4: Deformable read (32 anchors × 72 samples) → 123 fused tokens
  B5: 3-layer cross-attn (123 tokens) → depth
```

Core dimension $d = 192$. Pyramid channels: [64, 128, 192, 384].

---

## 3. Encoder: Frozen DAv2-S

Frozen Depth Anything V2 Small encoder (DINOv2-pretrained, depth-finetuned ViT-S). The DPT decoder head is discarded — we use only the ViT backbone.

| Parameter | Value |
|-----------|-------|
| Architecture | ViT-S (DINOv2) |
| embed_dim | 384 |
| Layers | 12 |
| Heads | 6 |
| patch_size | 14 |
| Input size | 518×518 (resize from NYU 480×640) |
| Patch grid $(H_p \times W_p)$ | 37×37 = 1,369 tokens |
| Intermediate layers | [2, 5, 8, 11] (0-indexed) |
| Params | ~24.8M (all frozen) |

Features at each intermediate layer: reshape from $(N, 384)$ sequence to $(H_p, W_p, 384)$ spatial grid, excluding CLS token.

**Source:** `src/f3/tasks/depth/depth_anything_v2/dinov2.py` (ViT-S definition), `src/f3/tasks/depth/depth_anything_v2/dpt.py` (intermediate layer indices).

---

## 4. Pyramid Neck (trainable)

Converts 4 ViT intermediate feature maps into a 4-level pyramid via bilinear resize + 1×1 convolution + LayerNorm. All sizes shown for 518×518 input ($H_p = W_p = 37$):

```
ViT Layer 2  (37×37×384) → bilinear ↑4× [148×148] → Conv(384→64,  k1) + LN → L1
ViT Layer 5  (37×37×384) → bilinear ↑2× [ 74×74 ] → Conv(384→128, k1) + LN → L2
ViT Layer 8  (37×37×384) → identity      [ 37×37 ] → Conv(384→192, k1) + LN → L3
ViT Layer 11 (37×37×384) → AvgPool  ↓2× [ 18×18 ] → Conv(384→384, k1) + LN → L4
```

General formula: $L_1 = 4H_p$, $L_2 = 2H_p$, $L_3 = H_p$, $L_4 = \lfloor H_p/2 \rfloor$. Effective stride ratios between levels ≈ 1:2:4:8.

| Level | Resolution | Channels | Eff. stride | Tokens |
|-------|:---:|:---:|:---:|:---:|
| L1 | 148×148 | 64 | ~3.5 | 21,904 |
| L2 | 74×74 | 128 | ~7.0 | 5,476 |
| L3 | 37×37 | 192 | ~14.0 | 1,369 |
| L4 | 18×18 | 384 | ~28.8 | 324 |

**Neck params:** ~25K + ~50K + ~74K + ~149K ≈ **~298K**.

---

## 5. Pre-compute (once per image)

**Input:** Pyramid $\{L_1, L_2, L_3, L_4\}$.
**Output:** $\text{cache} = \{L_1, L_2, L_3, L_4,\; K^{(1:2)}, V^{(1:2)},\; \hat{L}_2, \hat{L}_3, \hat{L}_4,\; g,\; s, b\}$

### 5.1 KV projections for B3

Each B3 layer $\ell = 1, 2$ has per-layer $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times 384}$, applied to L4:

$$K^{(\ell)} = L_4 \, (W_K^{(\ell)})^T, \quad V^{(\ell)} = L_4 \, (W_V^{(\ell)})^T \quad \in \mathbb{R}^{324 \times d}$$

Params: $4 \times 384 \times 192 =$ **~296K**.

### 5.2 Anchor projection for B4

$W_g \in \mathbb{R}^{d \times 384}$ pre-projects all L4 features for B4 anchor conditioning:

$$g = L_4 \, W_g^T \quad \in \mathbb{R}^{324 \times d}$$

Params: **~74K**.

### 5.3 Per-level $W_V$ for B4 deformable reads

Value projections pre-applied to feature maps so `grid_sample` reads already-projected $d$-dim features:

$$\hat{L}^{(\ell)} = L^{(\ell)} \, (W_V^{(\ell)})^T \quad \in \mathbb{R}^{H_\ell \times W_\ell \times d}$$

| Level | Projection | Params |
|-------|-----------|-------:|
| L2 | $128 \to 192$ | ~25K |
| L3 | $192 \to 192$ | ~37K |
| L4 | $384 \to 192$ | ~74K |
| **Total** | | **~136K** |

### 5.4 Calibration heads

$$s = \text{softplus}(W_s \cdot \text{MeanPool}(L_4) + b_s), \quad b = W_b \cdot \text{MeanPool}(L_4) + b_b$$

Two linear $\mathbb{R}^{384} \to \mathbb{R}^1$. $\text{softplus}$ ensures positive scale. Params: **~0.8K**.

**Pre-compute total:** ~296K + ~74K + ~136K + ~0.8K = **~507K**.

---

## 6. Per-Query Decoder (B1–B5)

All steps batched over $K$ queries in parallel. Coordinate convention: `F.grid_sample(align_corners=False, padding_mode='zeros')`. Pixel coords to grid: $\text{grid} = 2p / \text{dim} - 1$.

### 6.0 Symbol Table

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $d$ | Core dimension | 192 |
| $q = (u, v)$ | Query pixel coordinate | — |
| $s_\ell$ | Effective stride of level $\ell$: $s_1 {\approx} 3.5,\; s_2 {=} 7,\; s_3 {=} 14,\; s_4 {\approx} 29$ | scalar |
| $f_q^{(1)}$ | L1 center feature: Bilinear($L_1$, $q / s_1$) | 64 |
| $\text{pe}_q$ | Fourier positional encoding of $(u, v)$ | 32 |
| $h^{(0)}$ | Query seed: $W_q [f_q^{(1)};\; \text{pe}_q]$ | $d$ |
| $\mathcal{T}_{\text{unified}}$ | Local tokens (32 L1 + 25 L2 + 25 L3 + 9 L4) | $91 \times d$ |
| $h^{(2)}$ | B2 output (local-aware query) | $d$ |
| $h^{(2)'}$ | B3 output (globally-aware query) | $d$ |
| $\bar{\alpha}_q$ | Head-averaged B3 attention weights | 324 |
| $R_q$ | Top-32 L4 anchor positions from B3 routing | 32 positions |
| $h_r$ | Per-anchor deformable evidence (B4 output) | $d$ |
| $\mathcal{T}_{\text{fused}}$ | $[\mathcal{T}_{\text{unified}};\; h_{r_1}{+}e_{\text{deform}};\; \ldots;\; h_{r_{32}}{+}e_{\text{deform}}]$ | $123 \times d$ |
| $h^{(5)}$ | B5 output = $h_{\text{fuse}}$ (final query representation) | $d$ |
| $r_q$ | Relative depth code: MLP($h_{\text{fuse}}$) | scalar |
| $\hat{d}_q$ | Predicted depth: $1 / (\text{softplus}(s \cdot r_q + b) + \varepsilon)$ | scalar |

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

### 6.3 B3: Global L4 Cross-Attention + Routing (2 layers, 324 tokens)

*Enrich $h^{(2)}$ with global scene context from all 324 L4 tokens, then route to 32 anchors.*

**2-layer cross-attention into L4** using pre-computed KV from Section 5.1:

$$h^{(2)} \leftarrow h^{(2)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h^{(2)}),\; K{=}K^{(\ell)},\; V{=}V^{(\ell)})$$
$$h^{(2)} \leftarrow h^{(2)} + \text{FFN}^{(\ell)}(\text{LN}^{(\ell)}(h^{(2)}))$$

6 heads, $d_{\text{head}} = 32$. Per-query cost is only Q projection + attention + FFN (KV pre-computed).

**Output:** $h^{(2)'} \in \mathbb{R}^d$ — globally-aware query.

**Attention-based routing (zero extra params):** Head-average attention weights from B3 layer 2, select top-$R$:

$$\bar{\alpha}_q = \frac{1}{H} \sum_{h=1}^{H} \alpha_{q,h} \quad \in \mathbb{R}^{324}, \quad R_q = \text{Top-}R(\bar{\alpha}_q, R{=}32)$$

Each $r \in R_q$ maps to pixel coordinate $\mathbf{p}_r$ via L4 grid geometry (18×18, stride ~29). $R = 32$ from 324 = 9.9% selection. Straight-through routing: hard top-R forward, STE backward.

**B3 params:** 2× ($W_Q/W_O$ ~74K + FFN ~296K) + LNs ~2K = **~740K** (KV counted in pre-compute).

### 6.4 B4: Deformable Multi-Scale Read + Token Fusion

*For each of 32 anchors from B3, predict offsets and importance weights, read from multi-scale pyramid.*

**Conditioning:**

$$\Delta\mathbf{p}_r = \mathbf{p}_r - q \quad \text{(anchor-to-query offset in original pixel coords)}$$
$$u_r = \text{LN}(\text{GELU}(W_u [h^{(2)'};\; g_r;\; \phi(\Delta\mathbf{p}_r)] + b_u)) \quad \in \mathbb{R}^d$$

$g_r = g[\mathbf{p}_r^{(4)}] \in \mathbb{R}^d$ (pre-projected L4 anchor feature from Section 5.2). $\phi$: Fourier encoding (8 freq, 32 dims). Input: $d + d + 32 = 416$.

**Offset and weight prediction (shared across anchors):**

$$\Delta p_{r,h,\ell,m} = W^\Delta \, u_r + b^\Delta \quad (d \to H \times L \times M \times 2 = 144)$$
$$\beta_{r,h,\ell,m} = W^a \, u_r + b^a \quad (d \to H \times L \times M = 72)$$

$H = 6$ heads, $L = 3$ levels (L2–L4), $M = 4$ samples per head per level.

**Sampling with per-level normalization:**

$$p_{\text{sample}} = \mathbf{p}_r^{(\ell)} + \frac{\Delta p_{r,h,\ell,m}}{S_\ell}$$
$$f_{r,h,\ell,m} = \text{GridSample}(\hat{L}^{(\ell)}, \text{Normalize}(p_{\text{sample}}))$$

$S_\ell$: spatial extent of level $\ell$ (e.g., [74, 37, 18] for L2–L4).

**Per-head aggregation:**

$$a_{r,h,\ell,m} = \frac{\exp(\beta_{r,h,\ell,m})}{\sum_{\ell',m'} \exp(\beta_{r,h,\ell',m'})}, \quad \tilde{h}_{r,h} = \sum_{\ell,m} a_{r,h,\ell,m} \, v_{r,h,\ell,m}$$

$v_{r,h,\ell,m} \in \mathbb{R}^{d/H}$: head-partitioned pre-projected feature. Concat + output projection:

$$h_r = W_O [\tilde{h}_{r,1}; \ldots; \tilde{h}_{r,H}] + b_O \quad \in \mathbb{R}^d$$

**Budget:** 32 anchors × 72 samples = **2,304 deformable lookups**.
**Total lookups per query:** 92 (B1) + 2,304 (B4) = **2,396**.

**Token fusion:** Append 32 deformable tokens to local tokens:

$$\mathcal{T}_{\text{fused}} = [\mathcal{T}_{\text{unified}};\; h_{r_1}{+}e_{\text{deform}};\; \ldots;\; h_{r_{32}}{+}e_{\text{deform}}] \quad \in \mathbb{R}^{123 \times d}$$

5 type embeddings: $e_{L1}, e_{L2}, e_{L3}, e_{L4}, e_{\text{deform}} \in \mathbb{R}^d$.

**B4 params:** conditioning ~80K + offsets ~28K + weights ~14K + $W_O$ ~37K = **~159K**.

### 6.5 B5: Fused Cross-Attention + Depth Head (3 layers, 123 tokens)

*Fuse local + deformable evidence through 3 layers of cross-attention over $\mathcal{T}_{\text{fused}}$.*

**Per-layer KV ($\ell = 3, 4, 5$):**
$$K^{(\ell)} = W_K^{(\ell)} \, \text{LN}_{\text{kv2}}(\mathcal{T}_{\text{fused}}), \quad V^{(\ell)} = W_V^{(\ell)} \, \text{LN}_{\text{kv2}}(\mathcal{T}_{\text{fused}}) \quad \in \mathbb{R}^{123 \times d}$$

$\text{LN}_{\text{kv2}}$ shared across B5 layers, independent from B2's $\text{LN}_{\text{kv}}$.

**3-layer Pre-LN decoder (same structure as B2):**
$$h^{(\ell)} \leftarrow h^{(\ell-1)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h^{(\ell-1)}),\; K{=}K^{(\ell)},\; V{=}V^{(\ell)})$$
$$h^{(\ell)} \leftarrow h^{(\ell)} + \text{FFN}^{(\ell)}(\text{LN}_{\text{ff}}^{(\ell)}(h^{(\ell)}))$$

6 heads, $d_{\text{head}} = 32$, FFN: $192 \to 768 \to 192$. Input to layer 3 is $h^{(2)'}$ from B3.

**Output:** $h_{\text{fuse}} = h^{(5)} \in \mathbb{R}^d$

**Residual chain:** $h^{(0)} \xrightarrow{+\text{B2: 2L, 91 tok}} h^{(2)} \xrightarrow{+\text{B3: 2L, 324 tok}} h^{(2)'} \xrightarrow{+\text{B5: 3L, 123 tok}} h^{(5)} \to \hat{d}_q$

**Depth prediction:**
$$r_q = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2} \quad (192 \to 384 \to 1)$$
$$\hat{d}_q = \frac{1}{\text{softplus}(s \cdot r_q + b) + \varepsilon}, \quad \varepsilon = 10^{-6}$$

**B5 params:** 3× ($W_Q/W_K/W_V/W_O$ + FFN) ~1,332K + depth MLP ~74K + LNs ~4K = **~1,430K**.

---

## 7. Training

### 7.1 Dataset: NYU Depth V2

| Property | Value |
|----------|-------|
| Training | 795 scenes, ~24K image-depth pairs (Eigen split) |
| Test | 654 images (Eigen split) |
| Resolution | 480×640 → resize to 518×518 |
| GT depth | Dense (Kinect), max ~10m |
| Eval crop | Eigen center crop: rows [45:471], cols [41:601] |

### 7.2 Loss Functions

$$\mathcal{L} = L_{\text{point}} + \lambda_{\text{si}} \, L_{\text{silog}}$$

**Data fit (Huber on inverse depth):**
$$L_{\text{point}} = \frac{1}{K} \sum_{q \in Q} \text{Huber}(\hat{\rho}(q) - \rho^*(q)), \quad \hat{\rho} = \text{softplus}(s \cdot r_q + b) + \varepsilon, \quad \rho^* = 1/d^*$$

**Scale-invariant structure:**
$$L_{\text{silog}} = \sqrt{\frac{1}{K} \sum_q \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{K} \sum_q \delta_q\right)^2}, \quad \delta_q = \log \hat{d}_q - \log d_q^*$$

$\lambda_{\text{si}} = 0.5$, $\lambda_{\text{var}} = 0.85$.

### 7.3 Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 1×10⁻⁴ (cosine decay to 1×10⁻⁶) |
| Weight decay | 0.01 |
| Batch size | 8 |
| $K_{\text{train}}$ (queries per image) | 256 |
| Epochs | 30 |
| Precision | bf16 mixed precision |
| Gradient clipping | 1.0 |
| Attention dropout | 0.1 (B2, B3, B5) |
| Encoder | Frozen (no gradients) |
| Trainable | Pyramid neck + pre-compute + decoder (~4.2M params) |

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

## 9. Parameter Budget

**Trainable parameters:**

| Component | Params |
|-----------|-------:|
| Pyramid Neck (Section 4) | ~298K |
| B3 KV projections (Section 5.1) | ~296K |
| $W_g$ anchor projection (Section 5.2) | ~74K |
| B4 $W_V$ pre-projections (Section 5.3) | ~136K |
| Calibration heads (Section 5.4) | ~0.8K |
| B1: Feature extraction + tokens | ~142K |
| B2: Local cross-attn (2 layers) | ~891K |
| B3: Global cross-attn (2 layers) | ~740K |
| B4: Deformable read + fusion | ~159K |
| B5: Fused cross-attn (3 layers) + depth head | ~1,430K |
| **Total trainable** | **~4,167K (~4.2M)** |

| Component | Params |
|-----------|-------:|
| Frozen DAv2-S encoder | ~24.8M |
| **Total model** | **~29.0M** |

---

## 10. Implementation

### 10.1 File Structure

```
src/spd/
├── models/
│   ├── spd.py                  # Main model: encode + pre-compute + decode
│   ├── pyramid_neck.py         # ViT features → 4-level pyramid (Section 4)
│   ├── precompute.py           # KV projections, W_V projections, calibration (Section 5)
│   ├── query_encoder.py        # B1: Feature extraction + token construction (Section 6.1)
│   ├── local_cross_attn.py     # B2: 2-layer cross-attn over 91 tokens (Section 6.2)
│   ├── global_cross_attn.py    # B3: 2-layer L4 cross-attn + routing (Section 6.3)
│   ├── deformable_read.py      # B4: Offset prediction + multi-scale sampling (Section 6.4)
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
- `depth_anything_v2` (from `src/f3/tasks/depth/depth_anything_v2/`) — ViT-S encoder weights
- NYU Depth V2 dataset (download via standard script)

### 10.3 Build Order

Each step is independently testable. Do not proceed if the current step fails.

| Step | What to build | How to validate |
|------|--------------|-----------------|
| 1 | DAv2-S encoder + pyramid neck | Verify output shapes: L1 [B,64,148,148], L2 [B,128,74,74], L3 [B,192,37,37], L4 [B,384,18,18]. Visual sanity check: PCA of L3 features should show semantic structure. |
| 2 | Pre-compute module | Verify KV shapes [324, 192], $\hat{L}$ shapes, calibration scalars. |
| 3 | B1 token construction | Verify $\mathcal{T}_{\text{unified}}$ shape [B, K, 91, 192], $h^{(0)}$ shape [B, K, 192]. Test with random queries. |
| 4 | B2 local cross-attn + simple depth head | Train with $L_{\text{point}}$ only. Should achieve non-trivial AbsRel (< 0.3) on NYU. This validates the local-only decoder. |
| 5 | B3 global cross-attn + routing | Add B3. Verify routing selects diverse L4 positions (monitor attention entropy). |
| 6 | B4 deformable sampling | Add B4. Verify 2,304 grid_sample lookups produce [B, K, 32, 192] anchor features. |
| 7 | B5 fused cross-attn + full training | Full B1–B5 pipeline. Train with $L_{\text{point}} + L_{\text{silog}}$. Target: AbsRel close to DAv2-S dense baseline. |
