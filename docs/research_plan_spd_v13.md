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

```
RGB Image (480×640, NYU native)
  │
  ▼
ConvNeXt V2-T encoder (ImageNet pre-trained, fine-tuned with 0.1× LR)
  Stage 1: 120×160×96   (3× ConvNeXt block, stride 4)
  Stage 2:  60×80×192   (3× ConvNeXt block, stride 8)
  Stage 3:  30×40×384   (9× ConvNeXt block, stride 16)
  Stage 4:  15×20×768   (3× ConvNeXt block, stride 32)
  │
  ▼
Projection Neck (trainable, Conv 1×1 + LN per level)
  Stage 1 [120×160×96]  → Conv(96→64, k1)  + LN → L1 [120×160×64]
  Stage 2 [ 60×80×192]  → Conv(192→128, k1) + LN → L2 [ 60×80×128]
  Stage 3 [ 30×40×384]  → Conv(384→192, k1) + LN → L3 [ 30×40×192]
  Stage 4 [ 15×20×768]  → Conv(768→384, k1) + LN → 2× FullSelfAttn₃₈₄ → L4 [15×20×384]
  │
  ▼
Pre-compute (once per image)
  B3 KV projections, B4 W_V projections, calibration (s, b)
  │
  ▼
Per-query decoder B1–B5 (×K queries in parallel)
  B1: 91 local tokens + query seed h^(0)
  B2: 2-layer cross-attn (91 tokens) → h^(2)
  B3: 2-layer L4 cross-attn (300 tokens) → h^(2)', 32 anchors
  B4: Deformable read (32 anchors × 72 samples) → 123 fused tokens
  B5: 3-layer cross-attn (123 tokens) → depth
```

Core dimension $d = 192$. Encoder channels: [96, 192, 384, 768]. Decoder pyramid channels: [64, 128, 192, 384].

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

**Why ConvNeXt V2-T over other backbones:**
- Natural 4-level pyramid at strides [4, 8, 16, 32] — true multi-resolution features (no fake upsampling from single-resolution ViT)
- 40–50% faster than Swin-T on GPU at same FLOPs (pure conv, no window partition overhead)
- Higher accuracy than Swin-T (83.0% vs 81.3% IN-1K)
- GRN (Global Response Normalization) — same component proven in our v12 design
- L1–L3 need strong **local** features (our decoder samples them locally); ConvNeXt excels at this

**Source:** Pre-trained weights from `timm` (`convnextv2_tiny.fcmae_ft_in22k_in1k` or `convnextv2_tiny.fcmae_ft_in1k`).

---

## 4. Projection Neck + L4 Self-Attention (trainable)

**Projection neck:** 1×1 convolutions project ConvNeXt channels [96, 192, 384, 768] to decoder channels [64, 128, 192, 384]:

```
Stage 1 [120×160×96]  → Conv(96→64,   k1) + LN → L1 [120×160×64]    stride 4
Stage 2 [ 60×80×192]  → Conv(192→128,  k1) + LN → L2 [ 60×80×128]   stride 8
Stage 3 [ 30×40×384]  → Conv(384→192,  k1) + LN → L3 [ 30×40×192]   stride 16
Stage 4 [ 15×20×768]  → Conv(768→384,  k1) + LN → L4_pre [15×20×384] stride 32
```

**L4 self-attention:** 2× FullSelfAttn₃₈₄ on top of L4_pre. Required for B3 routing — every L4 token must have global scene context.

```
L4_pre [15×20×384, 300 tokens]
  → FullSelfAttn₃₈₄ layer 1 (6 heads, d_head=64, Pre-LN + FFN 384→1536→384)
  → FullSelfAttn₃₈₄ layer 2 (same)
  → L4 [15×20×384, 300 tokens]    ← every token sees all 300 positions
```

At 300 tokens, full self-attention is trivially cheap (~0.8G MACs, ~0.03ms). ConvNeXt's L4 has only local receptive field from k=7 convolutions — without self-attention, B3 routing would select anchors based on local content only, missing global scene structure.

| Level | Resolution | Channels | Stride | Tokens |
|-------|:---:|:---:|:---:|:---:|
| L1 | 120×160 | 64 | 4 | 19,200 |
| L2 | 60×80 | 128 | 8 | 4,800 |
| L3 | 30×40 | 192 | 16 | 1,200 |
| L4 | 15×20 | 384 | 32 | 300 |

**Neck params:** ~6K + ~25K + ~74K + ~296K = **~401K**.
**L4 self-attention params:** 2× (4 × 384² + FFN 384→1536→384) = **~3,542K**.
**Neck + L4 self-attn total: ~3,943K (~3.9M)**.

---

## 5. Pre-compute (once per image)

**Input:** Pyramid $\{L_1, L_2, L_3, L_4\}$.
**Output:** $\text{cache} = \{L_1, L_2, L_3, L_4,\; K^{(1:2)}, V^{(1:2)},\; \hat{L}_2, \hat{L}_3, \hat{L}_4,\; g,\; s, b\}$

### 5.1 KV projections for B3

Each B3 layer $\ell = 1, 2$ has per-layer $W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d \times 384}$, applied to L4:

$$K^{(\ell)} = L_4 \, (W_K^{(\ell)})^T, \quad V^{(\ell)} = L_4 \, (W_V^{(\ell)})^T \quad \in \mathbb{R}^{300 \times d}$$

Params: $4 \times 384 \times 192 =$ **~296K**.

### 5.2 Anchor projection for B4

$W_g \in \mathbb{R}^{d \times 384}$ pre-projects all L4 features for B4 anchor conditioning:

$$g = L_4 \, W_g^T \quad \in \mathbb{R}^{300 \times d}$$

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
| $s_\ell$ | Effective stride of level $\ell$: $s_1 {=} 4,\; s_2 {=} 8,\; s_3 {=} 16,\; s_4 {=} 32$ | scalar |
| $f_q^{(1)}$ | L1 center feature: Bilinear($L_1$, $q / s_1$) | 64 |
| $\text{pe}_q$ | Fourier positional encoding of $(u, v)$ | 32 |
| $h^{(0)}$ | Query seed: $W_q [f_q^{(1)};\; \text{pe}_q]$ | $d$ |
| $\mathcal{T}_{\text{unified}}$ | Local tokens (32 L1 + 25 L2 + 25 L3 + 9 L4) | $91 \times d$ |
| $h^{(2)}$ | B2 output (local-aware query) | $d$ |
| $h^{(2)'}$ | B3 output (globally-aware query) | $d$ |
| $\bar{\alpha}_q$ | Head-averaged B3 attention weights | 300 |
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

### 6.3 B3: Global L4 Cross-Attention + Routing (2 layers, 300 tokens)

*Enrich $h^{(2)}$ with global scene context from all 300 L4 tokens (each with global RF from L4 self-attention), then route to 32 anchors.*

**2-layer cross-attention into L4** using pre-computed KV from Section 5.1:

$$h^{(2)} \leftarrow h^{(2)} + \text{MHCrossAttn}^{(\ell)}(Q{=}\text{LN}_q^{(\ell)}(h^{(2)}),\; K{=}K^{(\ell)},\; V{=}V^{(\ell)})$$
$$h^{(2)} \leftarrow h^{(2)} + \text{FFN}^{(\ell)}(\text{LN}^{(\ell)}(h^{(2)}))$$

6 heads, $d_{\text{head}} = 32$. Per-query cost is only Q projection + attention + FFN (KV pre-computed).

**Output:** $h^{(2)'} \in \mathbb{R}^d$ — globally-aware query.

**Attention-based routing (zero extra params):** Head-average attention weights from B3 layer 2, select top-$R$:

$$\bar{\alpha}_q = \frac{1}{H} \sum_{h=1}^{H} \alpha_{q,h} \quad \in \mathbb{R}^{300}, \quad R_q = \text{Top-}R(\bar{\alpha}_q, R{=}32)$$

Each $r \in R_q$ maps to pixel coordinate $\mathbf{p}_r$ via L4 grid geometry (15×20, stride 32). $R = 32$ from 300 = 10.7% selection. Straight-through routing: hard top-R forward, STE backward.

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

$S_\ell$: spatial extent of level $\ell$ (e.g., [80, 40, 20] for L2–L4, using $W$ dimension).

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

**Residual chain:** $h^{(0)} \xrightarrow{+\text{B2: 2L, 91 tok}} h^{(2)} \xrightarrow{+\text{B3: 2L, 300 tok}} h^{(2)'} \xrightarrow{+\text{B5: 3L, 123 tok}} h^{(5)} \to \hat{d}_q$

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
| Resolution | 480×640 (native, no resize) |
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
| Trainable | All (~36.5M: encoder 28.6M + neck/self-attn 3.9M + precompute 0.5M + decoder 3.4M) |

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

| Component | Params |
|-----------|-------:|
| **Encoder: ConvNeXt V2-T** (fine-tuned 0.1× LR) | **~28,600K** |
| Projection Neck (Section 4) | ~401K |
| L4 Self-Attention, 2× FullSelfAttn₃₈₄ (Section 4) | ~3,542K |
| B3 KV projections (Section 5.1) | ~296K |
| $W_g$ anchor projection (Section 5.2) | ~74K |
| B4 $W_V$ pre-projections (Section 5.3) | ~136K |
| Calibration heads (Section 5.4) | ~0.8K |
| B1: Feature extraction + tokens | ~142K |
| B2: Local cross-attn (2 layers) | ~891K |
| B3: Global cross-attn (2 layers) | ~740K |
| B4: Deformable read + fusion | ~159K |
| B5: Fused cross-attn (3 layers) + depth head | ~1,430K |
| **Neck + L4 self-attn + precompute + decoder** | **~7,812K (~7.8M)** |
| **Total model** | **~36,412K (~36.5M)** |

**Efficiency comparison vs dense baseline (DAv2-S):**

$$T_{\text{ours}}(K) \approx 2.6 + 0.005K \;\text{ms}, \quad T_{\text{DAv2-S}} \approx 5.0 \;\text{ms}$$

| K | Ours (ms) | DAv2-S dense (ms) | Speedup |
|:---:|---:|---:|---:|
| 16 | 2.7 | 5.0 | 1.9× |
| 64 | 2.9 | 5.0 | 1.7× |
| 256 | 3.9 | 5.0 | **1.3×** |
| 490 | 5.0 | 5.0 | 1.0× (break-even) |

*Timings estimated for RTX 4090. On RTX 4060 laptop (training hardware), absolute times scale proportionally but relative speedup is preserved.*

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
- `timm` — ConvNeXt V2-T pre-trained weights (`convnextv2_tiny.fcmae_ft_in1k`)
- NYU Depth V2 dataset (download via standard script)

### 10.3 Build Order

Each step is independently testable. Do not proceed if the current step fails.

| Step | What to build | How to validate |
|------|--------------|-----------------|
| 1 | ConvNeXt V2-T encoder + projection neck + L4 self-attn | Verify output shapes: L1 [B,64,120,160], L2 [B,128,60,80], L3 [B,192,30,40], L4 [B,384,15,20]. PCA of L3/L4 features should show semantic structure. |
| 2 | Pre-compute module | Verify KV shapes [300, 192], $\hat{L}$ shapes, calibration scalars. |
| 3 | B1 token construction | Verify $\mathcal{T}_{\text{unified}}$ shape [B, K, 91, 192], $h^{(0)}$ shape [B, K, 192]. Test with random queries. |
| 4 | B2 local cross-attn + simple depth head | Train with $L_{\text{point}}$ only. Should achieve non-trivial AbsRel (< 0.3) on NYU. This validates the local-only decoder. |
| 5 | B3 global cross-attn + routing | Add B3. Verify routing selects diverse L4 positions (monitor attention entropy). |
| 6 | B4 deformable sampling | Add B4. Verify 2,304 grid_sample lookups produce [B, K, 32, 192] anchor features. |
| 7 | B5 fused cross-attn + full training | Full B1–B5 pipeline. Train with $L_{\text{point}} + L_{\text{silog}}$. Target: AbsRel close to DAv2-S dense baseline. |
