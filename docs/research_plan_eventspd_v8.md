# Research Plan: Streamlined EventSPD — Sparse Query-Point Depth from Events

Author: Claude (redesigned from Codex v4, audited v5, enriched pyramid v6, hybrid backbone v7, widened d=192 v8)
Date: 2026-02-12 to 2026-02-16

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

Every existing event-depth method produces dense $H \times W$ output. No published work targets sparse query-point depth from events as a first-class objective. The closest non-event work is InfiniDepth (Jan 2026), which applies LIIF-style continuous-coordinate depth queries to RGB — but without sparse routing, deformable attention, or event-camera support.

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

Measure:
- Full dense map latency.
- Query extraction latency: compute dense map, then sample $K$ points.
- Query-point accuracy at sampled locations.

### 3.2 Baseline B — Naive Sparse: Dense Output + Bilinear Sample

Run Baseline A, then `Bilinear(dense_depth, q)` at each query. This is the "do nothing clever" approach. It establishes the accuracy ceiling and the speed floor.

### 3.3 Baseline C — Minimal Query Decoder

Bilinear lookup from F^3 features + small MLP depth head. No global context, no deformable sampling. This establishes the "how far can pure local features go" baseline.

---

## 4. Proposed Algorithm: Streamlined EventSPD

### 4.1 Design Principles

This architecture is built on five principles, each grounded in proven methods:

1. **Precompute once, decode per query** (from SAM, Perceiver IO).
   F^3 features and a compact latent bank are computed once. Each query runs only a lightweight decoder.

2. **Three streams of evidence** (from DPT, monocular depth literature).
   Depth at a point requires: (a) exact-point identity, (b) local neighborhood gradients, (c) global scene context. These are separate streams merged via attention.

3. **Sparse spatial reads via deformable sampling** (from Deformable DETR).
   Instead of dense decoding, we sample features at learned offsets around content-routed spatial anchors. This gives precise non-local evidence at sublinear cost.

4. **Attention-based fusion with residual identity preservation** (from SAM's decoder, standard transformers).
   A 2-layer cross-attention transformer fuses the three streams. Residual connections naturally preserve center-point identity without explicit gating.

5. **Minimal viable complexity**.
   Every component must justify its existence via ablation. Optional extensions are documented separately and disabled by default.

### 4.2 Symbol Table

| Symbol | Computed from | Meaning | Trainable |
|--------|--------------|---------|-----------|
| $E_t$ | Raw events in $[t-\Delta, t)$ | Input event stream | No |
| $\mathcal{F}_{\text{F}^3}^{\text{ds2}}$ | Backbone network (ds2 config) | Event-to-feature encoder (640×360×64) | Frozen / fine-tuned |
| $F_t$ | $\mathcal{F}_{\text{F}^3}^{\text{ds2}}(E_t)$ | Dense shared feature field (640×360, 64ch) | Via backbone |
| $F_t^{(1)}$ | $\text{LN}(F_t)$ + GELU | Fine features (= F^3 ds2 output, 640×360, 64ch) | LN only |
| $F_t^{(2..4)}$ | Hybrid Conv+Swin backbone from $F_t$ | Multi-scale pyramid (64/96/192ch) | Yes |
| $s_\ell$ | $s_1{=}2, s_2{=}4, s_3{=}8, s_4{=}16$ | Stride of level $\ell$ | No |
| $C_t$ | $\text{Proj}(\text{GridPool}(F_t^{(4)}))$ | Latent bank from L4 (192→192) | Proj: Yes |
| $s_t, b_t$ | $\text{Heads}(\text{Pool}(C_t))$ | Global depth scale / shift | Yes |
| $q = (u, v)$ | User input | Query pixel coordinate | No |
| $f_q^{(1)}$ | $\text{Bilinear}(F_t^{(1)}, q)$ | Fine point feature (64ch) | No |
| $c_q^{(\ell)}$ | $\text{Bilinear}(F_t^{(\ell)}, q)$, $\ell{=}2,3,4$ | Multi-scale center (64/96/192ch) | No |
| $l_q$ | MaxPool(Local) from $F_t^{(1)}$ | L1 local context (32 samples) | MLP: Yes |
| $l_q^{(2)}$ | MaxPool(Local) from $F_t^{(2)}$ | L2 local (16 samples) | MLP: Yes |
| $\text{pe}_q$ | $\text{Fourier}(u/W, v/H)$ | Positional encoding (32d) | No |
| $h_{\text{point}}$ | MLP$([f_q^{(1)}; c_q^{(2)}; \text{pe}; l_q; l_q^{(2)}])$ | Center token (L1 + L2 context, 544ch→192) | Yes |
| $z_q$ | MLP$([f_q^{(1)}; l_q; \text{pe}_q])$ | Routing / retrieval token | Yes |
| $\bar{c}_q$ | $\text{MHCrossAttn}(z_q, C_t)$ | Global summary | Yes |
| $\alpha_q$ | $\text{softmax}(W_r z_q \cdot C_t^\top / \sqrt{d})$ | Routing scores | Yes |
| $R_q$ | $\text{TopR}(\alpha_q)$ | Routed coarse tokens | No |
| $S_q$ | $R_q \cup \{i_q^{\text{loc}}\}$ | Anchor set (33 total) | No |
| $h_{r}$ | DeformRead per $r \in S_q$ | Per-anchor evidence | Yes |
| $T_q$ | See below | 38 context tokens | No |
| $e_{\text{loc..glob}}$ | Learned | 7 type embeddings ($\in \mathbb{R}^d$) | Yes |
| $h_{\text{fuse}}$ | TransformerDec$(h_{\text{point}}, T_q)$ | Fused query representation | Yes |
| $r_q$ | MLP$(h_{\text{fuse}})$ | Relative depth code | Yes |
| $\rho_q$ | $s_t \cdot r_q + b_t$ | Calibrated inverse depth | No |
| $\hat{d}_q$ | $1 / (\text{softplus}(\rho_q) + \varepsilon)$ | Final predicted depth | No |
| $\sigma_q$ | $\text{softplus}(\text{Linear}(h_{\text{fuse}}))$ | Uncertainty (optional) | Yes |

**$T_q$ expansion (38 context tokens with type embeddings):**

$$
T_q = [l_q{+}e_{\text{loc}};\; c_q^{(2)}{+}e_{\text{ms2}};\; c_q^{(3)}{+}e_{\text{ms3}};\; c_q^{(4)}{+}e_{\text{ms4}};\; h_{r_1}{+}e_{\text{near}};\; h_{r_2..33}{+}e_{\text{route}};\; \bar{c}_q{+}e_{\text{glob}}]
$$

Core dimension: $d = 192$. Positional encoding: $d_{\text{pe}} = 32$ (8 Fourier frequencies × 2 trig functions × 2 spatial dims).

### 4.3 Algorithm A: Precompute Once Per Event Window

**Input:** Event set $E_t$.
**Output:** $\text{cache}_t = \{F_t^{(1)}, F_t^{(2)}, F_t^{(3)}, F_t^{(4)}, C_t, s_t, b_t\}$ where $F_t^{(1)}$ is the F^3 ds2 output with LN+GELU (64ch, 640×360) and $F_t^{(2:4)}$ are produced by a hybrid Conv+Swin backbone with GRN-enhanced ConvNeXt blocks.

#### A1. Backbone encoding (F^3 ds2)

$$
F_t = \mathcal{F}_{\text{F}^3}^{\text{ds2}}(E_t) \quad \in \mathbb{R}^{640 \times 360 \times 64}
$$

Shared event-derived feature field at half resolution. F^3 uses a Multi-Resolution Hash Encoder (MRHE) to convert sparse events into dense features, followed by 2 stages of ConvNeXt blocks.

**F^3 ds2 configuration** (new, replacing ds1):
```yaml
dsstrides: [2, 1]     # Stride-2 in first stage → 640×360 output
dims: [64, 64]         # 2× wider channels (ds1 used [32, 32])
convdepths: [3, 3]     # 6 ConvNeXt blocks total
patch_size: 2          # For F^3 event prediction training
```

**Why ds2 over ds1:** The F^3 ds1 config processes all 6 ConvNeXt blocks at full 1280×720 resolution (921,600 pixels), taking ~8.3 ms — 80% of total precompute. But our backbone immediately downsamples to 640×360 (L1) anyway. With ds2, F^3 applies stride-2 in the first stage, processing 5 of 6 blocks at 640×360 (230,400 pixels). The MRHE scatter still operates at full resolution (events are hashed at their native pixel coordinates), preserving fine-grained spatial structure in the hash features. Wider channels (64 vs 32) compensate: each pixel carries 2× more information, and the total information volume (pixels × channels) is comparable.

**Estimated cost:** ~4.5 ms (vs ~8.3 ms for ds1), saving ~3.8 ms. The speedup is driven by **memory bandwidth, not FLOPs**: ds2's GFLOPs drop only ~12% (57G vs 65G) because the pointwise convolutions have identical cost (4× fewer pixels × 4× wider channels = same MACs), but memory traffic halves (linear in C, not quadratic). On RTX 4090, all F^3 operations are memory-bound (arithmetic intensity 10–21 FLOP/byte, well below the 82 FLOP/byte ridge point). The L2 cache (72 MB) fits ds2's 59 MB feature maps but not ds1's 118 MB, giving ds2 an additional ~15% inter-kernel reuse bonus. The MRHE scatter overhead (~1.5 ms, always at full resolution) is fixed and identical for both. F^3 ds2 requires pre-training with the ds2 config — existing ds1 checkpoints do not transfer (different dims and strides).

#### A2. Multi-scale feature pyramid (Hybrid Conv + Swin backbone)

**Motivation (v7→v8 changes):** v6 used a 4-stage enrichment pipeline — effective but complex. v7 replaced this with a unified ConvNeXt+Swin backbone on F^3 ds1 (1280×720×32). v8 makes four improvements: (1) **F^3 ds2**: half-resolution output (640×360×64) saving ~3.8 ms precompute (memory-bandwidth driven, not FLOPs — see Section 6.2), (2) **GELU at L1** (zero-cost nonlinearity on the terminal fine-scale features), (3) **L2 widened to 64ch** with **GRN** in ConvNeXt blocks (ConvNeXt V2-style), and (4) **core dimension $d = 192$** for richer per-query representations.

**No FPN/BiFPN (justified):** Deformable DETR's ablation (Zhu et al., Table 2) shows that multi-scale deformable attention achieves identical AP with and without FPN (43.8 AP in both cases). The paper states: *"Because the cross-level feature exchange is already adopted, adding FPNs will not improve the performance."* Our B3 deformable read + B4 cross-attention provide the same cross-level exchange, making a top-down feature path unnecessary.

**Key design changes (v7→v8):**
- **L1 GELU activation:** L1 is a terminal feature map (used directly by the decoder, unlike ConvNeXt stems which feed into subsequent blocks). Adding GELU gives L1 nonlinear representation power at zero parameter cost — important because F^3 features are only ~4 CNN layers deep (much shallower than ResNet-50 features used in Deformable DETR).
- **L2 widened to 64ch:** v7's 48ch L2 had a ConvNeXt hidden dimension of only 192 (4× expansion). Widening to 64ch raises this to 256, matching ConvNeXt V2 Pico's stage-1. L2 feeds both $h_{\text{point}}$ and B4 tokens — a bottleneck here constrains two critical paths.
- **GRN in ConvNeXt blocks:** From ConvNeXt V2 (Woo et al., CVPR 2023). GRN normalizes each channel by the global L2 norm across all channels, adding nonlinear inter-channel competition. Near-zero cost (~2.6K params total).
- **Core dimension $d = 192$:** Wider per-query representations. 6 attention heads × 32-dim (standard config). B4 FFN expands to 768. Latent bank: 512 × 192 values (+50% scene capacity). Total system: ~4.0M params (was ~3.0M).

**Level 1** — fine-grained features (identity from F^3 ds2, with nonlinearity):
$$
F_t^{(1)} = \text{GELU}(\text{LN}(F_t)) \quad \in \mathbb{R}^{640 \times 360 \times 64}
$$

LayerNorm + GELU applied directly to the F^3 ds2 output. No convolution needed — F^3 ds2 already outputs at 640×360×64, matching the target L1 resolution and channel width. The GELU ensures L1 features are nonlinear, enabling representation of feature conjunctions and edge detectors. L1 is the most heavily sampled level (32 local + center + routing) and receives no further block-level processing — the per-query MLPs in B1 add one more nonlinear layer, giving the L1→decoder path at least 2 nonlinear operations.

**Why L1 = F^3 ds2 output:** With F^3 ds2, the backbone already processes 6 ConvNeXt blocks to produce 640×360×64 features. These are *richer* than the previous L1 (a single stride-2 conv on 32ch ds1 features). No additional conv is needed — just normalization and activation. Parameters: ~0.1K (LN only, GELU is parameter-free).

**Levels 2–4** — hierarchical backbone with GRN-enhanced ConvNeXt and Swin at L4:

```
F_t (640×360, 64ch)                                         ← F^3 ds2 output
├→ L1: LN + GELU                                            [640×360×64]
└→ Stem: Conv(64→64, k3s2) + LN                             [320×180×64]
   → 2× ConvNeXt_64 (with GRN)                              → L2  [320×180×64]
   → Down: Conv(64→96, k2s2) + LN                           [160×90×96]
   → 2× ConvNeXt_96 (with GRN)                              → L3  [160×90×96]
   → Down: Conv(96→192, k2s2) + LN                          [80×45×192]
   → 4× SwinBlock_192 (window=8, shifted)                    → L4  [80×45×192]
```

Four scales at strides 2×, 4×, 8×, 16× relative to the original HD image (1280×720). $F_t^{(1)}$ is 640×360 (= F^3 ds2 output), $F_t^{(2)}$ is 320×180, $F_t^{(3)}$ is 160×90, $F_t^{(4)}$ is 80×45.

**Stem (stride-2× projection from F^3 ds2):**
$$
\text{Stem}(F_t) = \text{LN}(\text{Conv2d}(64, 64, k{=}3, s{=}2, p{=}1)(F_t)) \quad \in \mathbb{R}^{320 \times 180 \times 64}
$$

A stride-2 convolution from the F^3 ds2 output (640×360) to L2 resolution (320×180). With ds2, the stem input is already 64ch (not 32ch as with ds1), so this is a same-width projection that halves spatial resolution. Parameters: $64 \times 64 \times 3 \times 3 + 64 \approx 37\text{K}$.

**ConvNeXt blocks (L2, L3) — with GRN:**

Each ConvNeXt block follows the ConvNeXt V2 design (Woo et al., CVPR 2023), with GRN after GELU:
```
Input → DW Conv 7×7 → LN → PW Conv 1×1 (C→4C) → GELU → GRN(4C) → PW Conv 1×1 (4C→C) → + Input
```

GRN (Global Response Normalization) computes per-channel $\ell_2$ norms globally, then normalizes and rescales: $\text{GRN}(x_i) = \gamma_i \cdot x_i \cdot \|X\|_2 / (\|x_i\|_2 + \epsilon) + \beta_i + x_i$. This adds nonlinear inter-channel competition, improving feature diversity. Parameters per block: $2 \times 4C$ (learnable $\gamma, \beta$).

- L2: 2× ConvNeXt$_{64}$ blocks. Each block: DW 7×7 (64×7×7 = 3.1K) + expansion 64→256→64 (16.4K + 16.4K) + GRN(256: 0.5K) + LN ≈ 36.7K/block. Total: ~73.3K.
- L3: 2× ConvNeXt$_{96}$ blocks. Each block: DW 7×7 (96×7×7 = 4.7K) + expansion 96→384→96 (36.9K + 36.9K) + GRN(384: 0.8K) + LN ≈ 79.5K/block. Total: ~159K.

ConvNeXt blocks build local context incrementally through 7×7 depthwise convolutions. At L2 (320×180), 2 blocks provide ~14px receptive field = ~56px original-coordinate context. At L3 (160×90), 2 blocks provide ~14px × stride 8 = ~112px context.

**Downsampling layers:**
$$
\text{Down}_{L2 \to L3} = \text{LN}(\text{Conv2d}(64, 96, k{=}2, s{=}2)) \quad \text{(~24.9K params)}
$$
$$
\text{Down}_{L3 \to L4} = \text{LN}(\text{Conv2d}(96, 192, k{=}2, s{=}2)) \quad \text{(~73.9K params)}
$$

Strided convolutions with LayerNorm. The first downsampling now takes 64ch input (widened L2) instead of 48ch.

**Swin Transformer blocks at L4 (the key depth context component):**

At 80×45 = 3,600 positions, self-attention within 8×8 windows (64 tokens per window, ~70 windows) is affordable and provides the global receptive field that depth estimation demands. After 4 blocks with alternating shifted windows, every position has attended to the full feature map.

Each Swin block:
```
Input → LN → W-MSA (window_size=8) → + Input → LN → FFN (192→768→192) → + Input
```

where W-MSA = Window Multi-Head Self-Attention with 6 heads ($d_{\text{head}} = 32$). Blocks alternate between regular windows and shifted windows (shift = window_size/2 = 4), following Swin Transformer v1.

Per-block parameters:
- QKV projection: $3 \times 192 \times 192 = 110.6\text{K}$
- Output projection: $192 \times 192 = 36.9\text{K}$
- FFN: $192 \times 768 + 768 \times 192 = 295\text{K}$
- Relative position bias: $(2 \times 8 - 1)^2 \times 6 = 1.4\text{K}$
- LayerNorms: $2 \times 2 \times 192 = 0.8\text{K}$
- Per-block total: ~444K
- 4 blocks: ~1,778K

**Why Swin at L4, ConvNeXt at L2–L3:**
- At 80×45, shifted window attention is affordable (~56M FLOPs/layer) and provides global receptive field — critical for scene-level depth (ground plane, walls, sky/foreground). CNN blocks cannot achieve this with local 7×7 kernels.
- At 320×180 (L2), window partitioning overhead dominates — ConvNeXt is simpler and faster at high resolutions. L2-L3 need local context (edges, textures), which 7×7 depthwise convolutions capture efficiently.

**Why this over v6's enrichment pipeline:** Simpler (3 module types vs 4 separate stages), natural cross-level flow (L2→L3→L4 through progressive downsampling — Deformable DETR proves FPN/BiFPN adds no accuracy when multi-scale deformable attention provides cross-level exchange), true global attention at L4, and faster (~2.0 ms vs ~3.2 ms).

**Backbone cost (amortized over all queries):**

| Component | Resolution | Channels | FLOPs | Time (est.) |
|-----------|:---:|:---:|---:|---:|
| L1: LN + GELU | 640×360 | 64 (identity) | ~0.01G | ~0.01 ms |
| Stem: Conv k3s2 + LN | 320×180 | 64→64 | ~0.43G | ~0.06 ms |
| L2: 2× ConvNeXt$_{64}$ (GRN) | 320×180 | 64 | ~6.2G | ~0.85 ms |
| Down L2→L3 | 160×90 | 64→96 | ~0.18G | ~0.02 ms |
| L3: 2× ConvNeXt$_{96}$ (GRN) | 160×90 | 96 | ~3.5G | ~0.5 ms |
| Down L3→L4 | 80×45 | 96→192 | ~0.13G | ~0.02 ms |
| L4: 4× SwinBlock$_{192}$ | 80×45 | 192 | ~3.3G | ~0.5 ms |
| **Backbone total** | | | **~13.8G** | **~2.0 ms** |

Total: ~2,148K params, ~2.0 ms. L1 is now nearly free (LN+GELU only) since F^3 ds2 already provides 640×360×64 features. The backbone's job reduces to building L2-L4 from the ds2 output.

#### A3. Compact latent bank (spatial grid pooling)

Divide $F_t^{(4)}$ into a $P_h \times P_w$ spatial grid. Each cell is adaptive-average-pooled and projected to core dimension:

$$
C_t = \text{Proj}_{192 \to d}(\text{AdaptiveAvgPool}_{P_h \times P_w}(F_t^{(4)})) \in \mathbb{R}^{P_c \times d}
$$

where $\text{Proj}_{192 \to d}$ is a learned linear projection $192 \to 192$ (~37K parameters). Since $d = 192$ matches L4's channel dimension, this projection is a square linear map — it can be replaced by an identity if ablations show no benefit, but the learned projection allows the latent bank to occupy a different subspace than raw L4 features.

Default: $P_h = 16, P_w = 32, P_c = 512$.

For HD input: $F_t^{(4)}$ is $80 \times 45$ (stride 16×), so each cell covers $5 \times 2.8$ level-4 positions ($\approx 80 \times 45$ original pixels). Compression ratio: $3{,}600 / 512 \approx 7 : 1$.

Each token $c_i$ has:
- **Content**: pooled and projected features from its spatial cell. $F_t^{(4)}$ is the output of 4× SwinBlock$_{192}$ with shifted window attention, providing global context. The 192→192 projection maps to core dimension $d = 192$.
- **Location**: the known center coordinate $\mathbf{p}_i$ of its cell in pixel coordinates (not learned — deterministic from grid geometry). We write $\mathbf{p}_i^{(\ell)} = \mathbf{p}_i / s_\ell$ for the position in level-$\ell$ native coordinates.

**Why pool from $F_t^{(4)}$:**
- L4's Swin features carry global context — each position has attended to the full feature map through shifted windows. This makes L4 the richest source for the latent bank's role as a coarse global scene description.
- 7:1 compression (3,600 → 512) preserves per-cell fidelity well. Each cell averages only ~7 features. With $d = 192$, the latent bank stores $512 \times 192 = 98{,}304$ values — 50% more scene capacity than at $d = 128$.
- Fine-grained spatial detail is recovered via direct bilinear lookups (B1) and deformable sampling (B3), not from the latent bank.
- Anchor positions in original-resolution coordinates are deterministic from grid geometry.

**Why $P_c = 512$:** Each cell covers ~40×45 original pixels — comparable to F^3's 63px receptive field, preserving discriminative content. $R = 32$ out of 512 (6.25%) gives generous anchor coverage within MoE sparsity norms. Cross-attention cost over 512 tokens is trivial. Ablate $P_c \in \{128, 256, 512, 1024\}$.

**Why spatial grid pooling:** Spatial tokens have explicit locations for deformable sampling in B3 without extra position learning. Simpler than Perceiver-style learned queries — swap in if ablations show benefit.

#### A4. Global calibration heads

$$
s_t = \text{softplus}(h_s(\text{MeanPool}(C_t))), \quad b_t = h_b(\text{MeanPool}(C_t))
$$

Window-level scale and shift for depth calibration. All queries in the same window share the same $(s_t, b_t)$, ensuring global consistency. $h_s$ and $h_b$ are single linear layers.

Standard practice for monocular depth scale ambiguity (MiDaS, ZoeDepth, F^3).

---

### 4.4 Algorithm B: Per-Query Sparse Depth Inference

**Input:** $\text{cache}_t$ and query batch $Q = \{q_j\}_{j=1}^{K}$.
**Output:** $\{(\hat{d}_j, \sigma_j)\}_{j=1}^{K}$.

All steps below are batched over $K$ queries in parallel.

#### B1. Feature extraction and token construction

**Implementation note — coordinate normalization:** All bilinear lookups throughout B1–B3 use `F.grid_sample` with `align_corners=False` and `padding_mode='zeros'`. Pixel coordinates $p$ are normalized to $[-1, 1]$ via $\text{grid} = 2p / \text{dim} - 1$ before sampling, following the Deformable DETR convention.

**Fine center feature (L1, 2× stride):**

$$
f_q^{(1)} = \text{Bilinear}(F_t^{(1)}, \text{Normalize}(q / s_1)) \quad \in \mathbb{R}^{64}
$$

A single feature vector at the finest scale (640×360, 64ch), capturing the query point's precise spatial identity. This feeds into $h_{\text{point}}$ (the fusion query token).

**Multi-scale center features (L2 for h_point, L3-L4 as B4 context tokens):**

$$
c_q^{(\ell)} = \text{Bilinear}(F_t^{(\ell)}, \text{Normalize}(q / s_\ell)), \quad \ell = 2, 3, 4
$$

Three feature vectors capturing the query's context at medium (L2, 64ch), broad (L3, 96ch), and scene-level (L4, 192ch) scales. Each level's features carry progressively deeper context from the hierarchical backbone. $c_q^{(2)}$ feeds into $h_{\text{point}}$ (giving it medium-range context from GRN-enhanced ConvNeXt blocks), while $c_q^{(3)}$ and $c_q^{(4)}$ become **separate context tokens in B4** after projection to $d = 192$, preserving their broader information for adaptive cross-attention fusion. $c_q^{(4)}$ is particularly valuable as it carries global context from Swin Transformer's shifted window attention.

**Tiered information flow:** $h_{\text{point}}$ = fine identity (L1) + medium context (L2); B4 cross-attention = broad context (L3-L4) + non-local evidence (deformable). Clean separation: MLP for local, attention for global.

**Local neighborhood sampling:**

Sample $N_{\text{loc}} = 32$ points around $q$ in $F_t^{(1)}$ coordinates:

Fixed grid (5×5 minus center):
$$
\Omega_{\text{fixed}} = \{(\delta_x, \delta_y) : \delta_x, \delta_y \in \{-2, -1, 0, 1, 2\}\} \setminus \{(0, 0)\}, \quad |\Omega_{\text{fixed}}| = 24
$$

Learned offsets (8 additional, predicted from center feature):
$$
\Delta_m = r_{\max} \cdot \tanh(W_{\text{off}}^{(m)} f_q^{(1)} + b_{\text{off}}^{(m)}), \quad m = 1, \ldots, 8
$$

with $r_{\max} = 6$ (maximum reach in $F_t^{(1)}$ coordinates).

For each offset $\delta \in \Omega_{\text{fixed}} \cup \{\Delta_m\}$:
$$
f_\delta = \text{Bilinear}(F_t^{(1)}, \tilde{q} + \delta), \quad
h_\delta = \text{GELU}(W_{\text{loc}} [f_\delta; \phi(\delta)] + b_{\text{loc}})
$$

where $\phi(\delta)$ is a small Fourier encoding of the offset (4 frequencies × 2 trig × 2 dims = 16 dims).

Aggregate via PointNet-style max pooling:
$$
l_q = W_{\text{agg}} \cdot \text{MaxPool}(\{h_\delta\}_\delta) + b_{\text{agg}}, \quad l_q \in \mathbb{R}^d
$$

**Why 32 local samples:** A 5×5 grid captures local gradients and edges (sufficient for depth discontinuity detection). 8 learned offsets can reach further for elongated structures. This is much smaller than the original plan's 121 samples — PointRend shows that even 3-4 adaptive points suffice for boundary refinement. We start conservative and can increase via ablation.

**Why max pooling:** Selects the strongest activation per dimension (PointNet), capturing the most discriminative local feature. Mean pooling dilutes sharp responses (e.g. a depth edge within a smooth region).

**Positional encoding:**
$$
\text{pe}_q = [\sin(2\pi \sigma_l u/W); \cos(2\pi \sigma_l u/W); \sin(2\pi \sigma_l v/H); \cos(2\pi \sigma_l v/H)]_{l=0}^{L_{\text{pe}}-1}
$$

with $\sigma_l = 2^l$ and $L_{\text{pe}} = 8$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

**Why Fourier instead of hash encoding:** Parameter-free and sufficient when combined with spatially-varying bilinear features. If ablations show hash encoding helps, add it.

**L2 local neighborhood sampling:**

Sample $N_{\text{loc}}^{(2)} = 16$ points around $q$ in $F_t^{(2)}$ coordinates (stride-4×):

Fixed grid (3×3 minus center):
$$
\Omega_{\text{fixed}}^{(2)} = \{(\delta_x, \delta_y) : \delta_x, \delta_y \in \{-1, 0, 1\}\} \setminus \{(0, 0)\}, \quad |\Omega_{\text{fixed}}^{(2)}| = 8
$$

Learned offsets (8 additional, predicted from L2 center feature):
$$
\Delta_m^{(2)} = r_{\max}^{(2)} \cdot \tanh(W_{\text{off}}^{(2,m)} c_q^{(2)} + b_{\text{off}}^{(2,m)}), \quad m = 1, \ldots, 8
$$

with $r_{\max}^{(2)} = 4$ (maximum reach in $F_t^{(2)}$ coordinates = 16 original pixels).

For each offset $\delta \in \Omega_{\text{fixed}}^{(2)} \cup \{\Delta_m^{(2)}\}$:
$$
f_\delta^{(2)} = \text{Bilinear}(F_t^{(2)}, \tilde{q}^{(2)} + \delta), \quad
h_\delta^{(2)} = \text{GELU}(W_{\text{loc}}^{(2)} [f_\delta^{(2)}; \phi^{(2)}(\delta)] + b_{\text{loc}}^{(2)})
$$

where $\tilde{q}^{(2)} = q / s_2$ is the query in L2 coordinates and $\phi^{(2)}(\delta)$ is a small Fourier encoding of the offset (4 frequencies × 2 trig × 2 dims = 16 dims).

Aggregate via max pooling:
$$
l_q^{(2)} = W_{\text{agg}}^{(2)} \cdot \text{MaxPool}(\{h_\delta^{(2)}\}_\delta) + b_{\text{agg}}^{(2)}, \quad l_q^{(2)} \in \mathbb{R}^d
$$

**Why L2 local sampling:** Adds medium-range context (~56px via ConvNeXt processing) beyond L1's fine-grained ~12px reach. 16 samples suffice because L2 features are deeper — each already carries ConvNeXt context. Total: 16 + 32 (L1) = 48 local samples per query.

Parameters: offset heads ($64 \times 2 \times 8 + 16 = 1.0K$), per-sample MLP ($80 \times 192 + 192 \approx 15.6K$), aggregation ($192 \times 192 + 192 \approx 37.1K$). Total: ~53.7K.

**Center token (L1 identity + L2 context):**
$$
h_{\text{point}} = \text{LN}(W_{p2} \cdot \text{GELU}(W_{p1} [f_q^{(1)}; c_q^{(2)}; \text{pe}_q; l_q; l_q^{(2)}] + b_{p1}) + b_{p2})
$$

A 2-layer MLP with GELU activation and **no skip connection**. Input dimension: $64 + 64 + 32 + 192 + 192 = 544$. Hidden dimension: $d = 192$. Output: $h_{\text{point}} \in \mathbb{R}^d$. Parameters: $544 \times 192 + 192 + 192 \times 192 + 192 \approx 141.7\text{K}$.

**h_point design:** $f_q^{(1)}$ (64ch, precise identity from GELU-activated L1), $c_q^{(2)}$ (64ch, ~56px GRN-enhanced ConvNeXt context), $l_q$ and $l_q^{(2)}$ (192ch each, max-pooled local gradients). L3-L4 broader context enters via B4 cross-attention instead — adaptive weighting beats fixed MLP mixing.

**No skip connection:** Depth is a nonlinear function of appearance features — $h_{\text{point}}$ should be fully nonlinear. Matches DAv2 and F^3's `EventPixelFF` patterns.

**Routing token:**
$$
z_q = \text{LN}(W_z [f_q^{(1)}; l_q; \text{pe}_q] + b_z), \quad z_q \in \mathbb{R}^d
$$

Input dimension: $64 + 192 + 32 = 288$. Parameters: $288 \times 192 + 192 \approx 55.5\text{K}$. This token sees the center feature plus local context. It drives global retrieval (B2) and deformable conditioning (B3). It is separate from $h_{\text{point}}$ because routing and depth prediction have different objectives.

#### B2. Global context retrieval

**Global summary via multi-head cross-attention:**
$$
\bar{c}_q = \text{MHCrossAttn}(Q = W_Q z_q, \; K = W_K C_t, \; V = W_V C_t)
$$

with 6 attention heads, $d_{\text{head}} = 32$. Output: $\bar{c}_q \in \mathbb{R}^d$.

This attends to ALL $P_c = 512$ tokens with soft weights. No information is lost. The summary captures the full global scene context relevant to this query. The $1 \times 512$ attention matrix per head is computationally trivial. 6 heads (up from 4 at $d = 128$) allow more diverse attention patterns over the latent bank.

**Spatial routing (for deformable sampling anchors):**
$$
\alpha_q = \text{softmax}\left(\frac{W_r z_q \cdot (W_k^r C_t)^\top}{\sqrt{d}}\right), \quad
R_q = \text{TopR}(\alpha_q, R)
$$

Selects $R = 32$ out of 512 tokens (6.25%) whose spatial regions are most relevant for deformable sampling. Decoder cost is dominated by precompute, so anchor budget is accuracy-driven.

**Nearest coarse anchor:**
$$
i_q^{\text{loc}} = \arg\min_{i \in \{1, \ldots, P_c\}} \|\mathbf{p}_i - q\|_2^2
$$

where $q = (u, v)$ is the query in pixel coordinates and $\mathbf{p}_i$ is the grid center of token $i$ in pixel coordinates. Since the grid geometry is deterministic, this is equivalent to finding the nearest grid cell center — independent of source level.

If $i_q^{\text{loc}} \in R_q$, replace the lowest-scoring token in $R_q$ with the next-highest-scoring token, ensuring no duplication.

**Spatial anchor set:**
$$
S_q = R_q \cup \{i_q^{\text{loc}}\}, \quad |S_q| = R + 1 = 33 \text{ (always)}
$$

**Separate routing and summary:** Different objectives — summary optimizes for information aggregation, routing for spatial anchor selection. Nearest anchor guarantees local neighborhood is always represented.

**Straight-through routing:** Hard top-R forward, gradients via STE backward (`selected = top_mask - probs.detach() + probs`).

#### B3. Deformable multiscale read

For each anchor $r \in S_q$, predict sampling offsets and importance weights, then read features from the pyramid.

**Conditioning:**
$$
\Delta\mathbf{p}_r = \mathbf{p}_r - q \quad \text{(query-to-anchor offset in original pixel coordinates)}
$$
$$
u_r = \text{LN}(W_u [z_q;\; c_r;\; \phi_{\text{B3}}(\Delta\mathbf{p}_r)] + b_u), \quad u_r \in \mathbb{R}^d
$$

where $c_r \in \mathbb{R}^d$ is the anchor's content vector from the latent bank $C_t$, $\mathbf{p}_r$ is the anchor's spatial position in pixel coordinates, and $\phi_{\text{B3}}$ is a normalized Fourier encoding (see below). Input dimension: $d + d + d_{\text{pe}} = 192 + 192 + 32 = 416$. This conditioning vector tells the offset head three things: "I'm this query ($z_q$), looking at an anchor **whose content is** $c_r$, **located at** this spatial offset." The conditioning and offset/weight heads are **shared across all 33 anchors** — each anchor produces a different $u_r$ because its inputs ($c_r$, $\Delta\mathbf{p}_r$) differ.

**B3 Fourier encoding (normalized, 8 frequencies):**
$$
\phi_{\text{B3}}(\Delta\mathbf{p}_r) = [\sin(2\pi \sigma_l \Delta u/W); \cos(2\pi \sigma_l \Delta u/W); \sin(2\pi \sigma_l \Delta v/H); \cos(2\pi \sigma_l \Delta v/H)]_{l=0}^{L_{\text{pe}}-1}
$$

with $\sigma_l = 2^l$, $L_{\text{pe}} = 8$, giving $\phi_{\text{B3}} \in \mathbb{R}^{32}$ — the same formula as $\text{pe}_q$ but applied to the normalized offset $[\Delta u/W, \Delta v/H] \in [-1, 1]$.

**Why normalized encoding:** B3's offsets span the full image (up to 1280px). Normalizing by image dimensions maps to $[-1, 1]$ (standard practice: NeRF, DAB-DETR). Finest resolved period is ~5px, finer than the ~40px grid cell spacing.

**Why include anchor content $c_r$:** Enables content-adaptive sampling — the offset head asks "given what this region looks like ($c_r$), where should I sample?" Following DCNv2's approach of predicting offsets from local features.

**Offset and weight prediction (shared across anchors):**

For each head $h \in [1, H]$, level $\ell \in [1, L]$, sample $m \in [1, M]$:
$$
\Delta p_{r,h,\ell,m} = W_{h,\ell,m}^\Delta \, u_r + b_{h,\ell,m}^\Delta
$$
$$
\beta_{r,h,\ell,m} = W_{h,\ell,m}^a \, u_r + b_{h,\ell,m}^a
$$

Offsets are **unbounded** (no tanh), following Deformable DETR, Mask2Former, DCNv3/v4, and DAB-DETR. Training stability: **zero initialization** + **0.1× learning rate** on offset heads.

In practice, all offsets and weights are predicted by two shared linear layers (following Deformable DETR):
```python
self.sampling_offsets = nn.Linear(d, H * L * M * 2)   # -> 8*3*4*2 = 192
self.attention_weights = nn.Linear(d, H * L * M)       # -> 8*3*4   = 96
```

**Sampling:**
$$
p_{\text{sample}} = \mathbf{p}_r^{(\ell)} + \Delta p_{r,h,\ell,m}
$$
$$
f_{r,h,\ell,m} = \text{GridSample}_{\text{zeros}}(F_t^{(\ell)}, \text{Normalize}(p_{\text{sample}}))
$$

where $\mathbf{p}_r^{(\ell)} = \mathbf{p}_r / s_\ell$ maps the anchor's pixel-coordinate position to level-$\ell$'s native coordinate system. Zero padding for out-of-bound samples is the universal standard across all deformable attention implementations (Deformable DETR, DCNv3, DCNv4, Mask2Former — verified exhaustively). With unbounded offsets and 0.1× LR, out-of-bound samples are rare (offsets stay moderate) but handled gracefully by zero padding.

**Multi-head per-anchor aggregation (standard Deformable DETR pattern):**

First, project sampled features into head-partitioned value space:
$$
v_{r,h,\ell,m} = W_V^{(h)} f_{r,h,\ell,m}, \quad v_{r,h,\ell,m} \in \mathbb{R}^{d/H}
$$

where $W_V^{(h)} \in \mathbb{R}^{(d/H) \times C_\ell}$ is the value projection for head $h$ ($C_\ell$ is the channel dimension of level $\ell$: 64 for L2, 96 for L3, 192 for L4). In practice, a per-level $W_V^{(\ell)} \in \mathbb{R}^{d \times C_\ell}$ is used (with output reshaped to $H = 8$ heads of $d/H = 24$ dims). **Implementation optimization:** Following Deformable DETR, $W_V$ is pre-applied to each feature map $F_t^{(\ell)}$ once per frame during precompute, so grid_sample reads already-projected 192-ch features. This eliminates the per-sample projection cost from the per-query path.

**Mixed channel dimensions:** Per-level $W_V$ pre-projection normalizes 64/96/192ch → 192ch before storage (~68K total, ~0.05 ms).

Per-head softmax over levels and samples:
$$
a_{r,h,\ell,m} = \frac{\exp(\beta_{r,h,\ell,m})}{\sum_{\ell',m'} \exp(\beta_{r,h,\ell',m'})}, \quad
\tilde{h}_{r,h} = \sum_{\ell,m} a_{r,h,\ell,m} \, v_{r,h,\ell,m}
$$

Concatenate heads and project:
$$
h_r = W_O [\tilde{h}_{r,1}; \ldots; \tilde{h}_{r,H}] + b_O, \quad h_r \in \mathbb{R}^d
$$

Each $h_r$ is a self-contained spatial evidence summary from one anchor region.

**Why full multi-head structure:** $H=8$ independent 16-dim subspaces learn complementary sampling patterns. Per-head softmax preserves each head's weight distribution (standard in Deformable DETR, DCNv3). Concat + $W_O$ enables representations outside the convex hull of sampled features.

**Default budget:** $H = 8$ heads, $L = 3$ levels (2–4), $M = 4$ samples.
- Per anchor: $8 \times 3 \times 4 = 96$ samples.
- 33 anchors: $33 \times 96 = 3{,}168$ deformable lookups total.

**Why $H = 8$, $L = 3$, $M = 4$:** Standard Deformable DETR defaults. L1 dropped from deformable (stride 2×, no deep processing — local detail handled by B1). L2–L4 carry progressively deeper features (ConvNeXt → Swin). 3,168 total lookups spread across 33 diverse anchor regions. Decoder cost is dominated by F^3 precompute (~95% of total time), so per-anchor sample budget is essentially free.

**Per-anchor aggregation** (not global softmax) — preserves each region's contribution. B4 learns inter-region weighting via cross-attention.

#### B4. Fusion decoder (2-layer cross-attention transformer)

**Context token set (with type embeddings):**

$$
T_q = [l_q + e_{\text{loc}}; \; c_q^{(2)} + e_{\text{ms2}}; \; c_q^{(3)} + e_{\text{ms3}}; \; c_q^{(4)} + e_{\text{ms4}}; \; h_{r_1} + e_{\text{near}}; \; h_{r_2..33} + e_{\text{route}}; \; \bar{c}_q + e_{\text{glob}}]
$$

38 context tokens total, each with a learned type embedding:
- $l_q + e_{\text{loc}}$: aggregated local neighborhood (from B1).
- $c_q^{(2)} + e_{\text{ms2}}$: medium-range context (L2 feature at query, 64ch → 192ch via $W_{\text{ms2}}$, ~56px GRN-enhanced ConvNeXt context).
- $c_q^{(3)} + e_{\text{ms3}}$: broad context (L3 feature at query, 96ch → 192ch via $W_{\text{ms3}}$, ~112px context + information from L2 via downsampling).
- $c_q^{(4)} + e_{\text{ms4}}$: scene-level context (L4 feature at query, 192ch → 192ch via $W_{\text{ms4}}$, global context from Swin Transformer shifted window attention).
- $h_{r_1} + e_{\text{near}}$: deformable evidence from the nearest anchor (medium-range non-local context).
- $h_{r_2} + e_{\text{route}}, \ldots, h_{r_{33}} + e_{\text{route}}$: deformable evidence from 32 routed anchors (non-local context).
- $\bar{c}_q + e_{\text{glob}}$: compressed global scene summary (from B2).

**Per-level projection:** $c_q^{(2:4)}$ have different channels (64/96/192) — each projected to $d = 192$ via $W_{\text{ms}\ell}$. $c_q^{(2)}$ has **dual role**: raw 64ch feeds $h_{\text{point}}$, projected 192ch is a B4 token. $c_q^{(3:4)}$ are B4-only. $W_{\text{ms4}}$ is a square 192→192 projection (L4 channel width matches $d$). Projection parameters: ~68K total.

**Type embeddings:**

Each context token receives a learned type embedding identifying its role:

$e_{\text{loc}}, e_{\text{ms2}}, e_{\text{ms3}}, e_{\text{ms4}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}} \in \mathbb{R}^d$ — 7 learned embeddings (7 × 192 = 1,344 parameters). The three multi-scale embeddings ($e_{\text{ms2}}, e_{\text{ms3}}, e_{\text{ms4}}$) let the attention heads distinguish between context at different spatial scales. This is standard practice in DETR-family architectures where different token roles receive distinct learned embeddings.

**KV normalization:**

Before entering the transformer, apply a shared LayerNorm to the assembled context tokens:
$$
T_q \leftarrow \text{LN}_{\text{kv}}(T_q)
$$

Normalizes heterogeneous token scales from different computational paths. Applied once since $T_q$ is static — essentially free.

**2-layer transformer decoder (standard Pre-LN):**

Each layer applies cross-attention with residual, then FFN with residual (standard Pre-LN transformer):
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}), \; KV = T_q)
$$
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}(\text{LN}(h_{\text{point}}))
$$

where $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ with expansion ratio 4 ($d \to 4d \to d$, i.e., $192 \to 768 \to 192$).

Cross-attention uses 6 heads with $d_{\text{head}} = 32$. Attention matrix per head is $1 \times 38$ — trivially cheap. 6 heads (matching B2) allow diverse attention patterns over the 7 token types.

After 2 layers: $h_{\text{fuse}} = h_{\text{point}} \in \mathbb{R}^d$.

**Standard Pre-LN residuals** in both sub-layers (SAM, DETR, all modern transformers). Cross-attention is data-dependent: at depth edges, upweights local detail ($l_q$, $h_{r_1}$); in textureless regions, upweights global context ($c_q^{(3:4)}$, $\bar{c}_q$). The 7 type embeddings help distinguish token roles. Residuals preserve $h_{\text{point}}$'s identity while cross-attention adds context. If center collapse is observed, $L_{\text{ctr}}$ addresses it directly.

**Why 2 layers:** SAM's decoder uses 2 layers and achieves excellent point-prompt results. Our context set is small (38 tokens vs thousands of image tokens in SAM), so 2 layers provide sufficient mixing depth. Ablate with 1 and 3 layers.

#### B5. Depth prediction

**Relative depth code:**
$$
r_q = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2}
$$

A 2-layer MLP ($d \to d \to 1$, i.e., $192 \to 192 \to 1$) with GELU activation. Output: scalar $r_q \in \mathbb{R}$. Parameters: $192 \times 192 + 192 + 192 \times 1 + 1 \approx 37\text{K}$.

**Center-only auxiliary code (training only):**
$$
r_q^{\text{ctr}} = W_{\text{ctr},2} \cdot \text{GELU}(W_{\text{ctr},1} \, h_{\text{point}}^{(0)} + b_{\text{ctr},1}) + b_{\text{ctr},2}
$$

A 2-layer MLP ($192 \to 96 \to 1$) applied to $h_{\text{point}}^{(0)}$, the center token BEFORE fusion (saved from B1). This forces the center branch to remain independently informative.

**Auxiliary calibration:** Calibrated by the same $(s_t, b_t)$: $\rho_q^{\text{ctr}} = s_t \cdot r_q^{\text{ctr}} + b_t$. Shared calibration lets the auxiliary focus purely on depth structure (PSPNet/DeepLabV3 principle).

**Why a 2-layer MLP:** Following PSPNet/DeepLabV3 auxiliary head practice. A linear probe would constrain gradients to rank-1 updates — insufficient for training-time representation learning.

**Calibration:**
$$
\rho_q = s_t \cdot r_q + b_t
$$

**Depth conversion:**
$$
\hat{d}_q = \frac{1}{\text{softplus}(\rho_q) + \varepsilon}
$$

with $\varepsilon = 10^{-6}$.

**Uncertainty (optional, disabled in default profile):**
$$
\sigma_q = \text{softplus}(W_\sigma \, h_{\text{fuse}} + b_\sigma) + \sigma_{\min}
$$

with $\sigma_{\min} = 0.01$.

#### B6. Return sparse outputs

$$
\{(\hat{d}_j, \sigma_j)\}_{j=1}^K
$$

---

### 4.5 Merged Runtime Implementation (EventSPD-lite)

The B1-B5 decomposition is for clarity. For efficient implementation, merge into 4 CUDA-friendly stages:

| Stage | Merges | Computation |
|-------|--------|-------------|
| `M1_Precompute` | A1–A4 | $E_t \to F_t^{(1)}(\text{L1 2× stride + GELU}), F_t^{(2:4)}(\text{GRN-Conv+Swin backbone}), C_t, s_t, b_t, W_V\text{-projected maps}$ |
| `M2_QueryEncode` | B1+B2 | $q \to h_{\text{point}}, z_q, l_q, l_q^{(2)}, c_q^{(2:4)}, \bar{c}_q, S_q$ |
| `M3_DeformRead` | B3 | $S_q, z_q, C_t, F_t^{(2:4)} \to h_{r_1}, \ldots, h_{r_{33}}$ |
| `M4_FuseDecode` | B4+B5 | $h_{\text{point}}, T_q(38\text{ tokens}), s_t, b_t \to \hat{d}_q$ |

---

### 4.6 Why This Design Will Work — Confidence Arguments

1. **Precompute-then-query is proven.** SAM and Perceiver IO achieve SOTA with lightweight decoders on precomputed features.
2. **F^3 features + hybrid backbone provide rich depth cues.** F^3 already matches DAv2 for dense depth. The v8 backbone adds GELU-activated L1, GRN-enhanced ConvNeXt local context (widened to 64ch at L2), and Swin global attention at L4.
3. **Local + global context is standard in depth.** MiDaS, DPT, ZoeDepth all use multi-scale local+global features. Our multi-stream B4 fusion replicates this sparsely.
4. **Deformable attention is battle-tested.** Deformable DETR, Mask2Former, DAB-DETR — robust and well-understood.
5. **Speed advantage is structural.** Precompute dominates (~83% at $K=256$). Crossover at $K \approx 1{,}800$. Sampling budgets are free in wall-clock time.
6. **Nonlinearity at every stage.** GELU at L1, GRN+GELU in ConvNeXt, Swin attention, GELU MLPs, softmax attention, softplus depth — no linear shortcuts from raw features to output. Verified: Deformable DETR shows FPN is unnecessary when multi-scale deformable attention provides cross-level exchange.

---

### 4.7 Comparison: DepthAnythingV2 (DAv2) vs EventSPD

DAv2 is the dense depth baseline used in the F^3 pipeline. Understanding its architecture clarifies what EventSPD replaces and why.

**Architecture comparison:**

| Aspect | DAv2-S (F^3 baseline) | EventSPD v8 |
|--------|----------------------|-------------|
| **Encoder** | DINOv2 ViT-S (384-dim, 12 layers, 6 heads, patch=14) | Hybrid Conv+Swin backbone (64/96/192ch, 2+2+4 blocks, GRN) |
| **Encoder input** | 518×518 RGB (resized) | 640×360 F^3 ds2 features (64ch) |
| **Encoder output** | 37×37 tokens (1,369 positions) | 4-level pyramid: 640×360 / 320×180 / 160×90 / 80×45 |
| **Decoder** | DPT head (4 reassemble + 4 fusion stages, features=64) | Sparse per-query: routing + deformable + 2-layer transformer ($d = 192$) |
| **Output** | Dense 518×518 depth map | K sparse depth values at query points |
| **Encoder FLOPs** | ~93G (ViT-S self-attention + FFN, 12 layers) | ~14G (Conv+Swin backbone) |
| **Decoder FLOPs** | ~8G (DPT, 4-stage upsampling + fusion) | ~25M per query × K |
| **Total FLOPs** | ~101G per frame | ~71G (F^3 ds2 ~57G + backbone ~14G) + 25M×K |
| **Encoder params** | ~22M (ViT-S) | ~2.3M (backbone) |
| **Decoder params** | ~3M (DPT-S head) | ~1.7M (per-query decoder) |
| **Latency** | ~23 ms (F^3 8.3ms + DAv2 ~15ms) | ~8.7 ms at K=256 |
| **Depth source** | DINOv2 pretrained features (142M images) | F^3 event features + learned backbone |
| **Receptive field** | Global (ViT self-attention at every layer) | L1-L3: local (CNN+GRN); L4: global (Swin shifted windows) |

**DAv2 advantages:** Dense depth maps (all pixels), massive DINOv2 pretraining (142M images), works with RGB cameras.

**EventSPD advantages:** Sublinear query scaling ($O(1) + O(K)$, 2.6× faster at K=256), adaptive spatial sampling (deformable attention at content-routed anchors), native event camera support, per-query uncertainty.

**Key trade-off:** DAv2 spends ~93G FLOPs building a universally rich representation (ViT-S, 12 self-attention layers over 1,369 tokens), then reads densely. EventSPD spends ~71G on F^3 ds2 (~57G) + hybrid backbone (~14G) — comparable FLOPs but **much faster** because F^3 is memory-bound (4.5 ms + 2.0 ms = 6.6 ms vs DAv2's ~15 ms). It then reads sparsely via routing + deformable sampling. The wider $d = 192$ core dimension gives the per-query decoder more capacity to compensate for the lighter backbone.

**F^3 + DAv2 dense baseline cost:** F^3 ~65G (8.3ms) + DAv2-S encoder ~93G (12ms) + DPT decoder ~8G (3ms) = **~166G, ~23ms** (verified from F^3 codebase).

---

## 5. Training

### 5.1 Loss Functions

Three loss terms. That's it.

**Data fit:**
$$
L_{\text{point}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}(q) - \rho^*(q)), \quad \hat{\rho}(q) = \text{softplus}(\rho_q) + \varepsilon
$$

Pointwise Huber loss on predicted inverse depth $\hat{\rho}(q)$ vs ground truth $\rho^*(q) = 1/d^*(q)$. Both quantities are positive (softplus ensures $\hat{\rho} > 0$), so the Huber loss operates in a consistent positive domain. Robust to outliers. $Q_v$ is the set of queries with valid ground truth.

**Scale-invariant structure:**
$$
L_{\text{silog}} = \sqrt{\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q\right)^2}, \quad \delta_q = \log \hat{d}(q) - \log d^*(q)
$$

Standard SiLog loss with sqrt (matching F^3, BTS, AdaBins, DAv2). Default $\lambda_{\text{var}} = 0.5$ (F^3 codebase default). Ablate $\lambda_{\text{var}} \in \{0.5, 0.85, 1.0\}$.

**Center auxiliary (prevents center collapse):**
$$
L_{\text{ctr}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}^{\text{ctr}}(q) - \rho^*(q)), \quad \hat{\rho}^{\text{ctr}}(q) = \text{softplus}(\rho_q^{\text{ctr}}) + \varepsilon
$$

where $\rho_q^{\text{ctr}} = s_t \cdot r_q^{\text{ctr}} + b_t$ (calibrated auxiliary prediction, see B5). Forces the pre-fusion center token to predict depth independently. If the center branch is informative on its own, the fused output will be at least as good. Gradients from $L_{\text{ctr}}$ flow to both the auxiliary MLP and the shared calibration heads $(s_t, b_t)$.

**Total objective:**
$$
\mathcal{L} = L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + \lambda_{\text{ctr}} L_{\text{ctr}}
$$

Default weights: $\lambda_{\text{si}} = 0.5$, $\lambda_{\text{ctr}} = 0.25$.

**Optional additions:** $L_{\text{rank}}$ (pairwise ranking), $L_{\text{unc}}$ (Gaussian NLL for uncertainty), $L_{\text{entropy}}$ (routing regularizer). Add only for specific failure modes.

### 5.2 Training Data

EventSPD predicts sparse query points, not dense depth maps. This is a structural advantage for supervision: we only need ground truth at the K query locations per batch, not at every pixel.

**Available depth ground truth for event cameras:**

| Dataset | GT type | Source | Resolution | Density | Metric? |
|---------|---------|--------|------------|---------|---------|
| **M3ED** | Real LiDAR | Velodyne VLP-16 | 1280×720 | Sparse (~5-10% of pixels) | Yes |
| **DSEC** | Real stereo + LiDAR | LiDAR + SGM filtering | 640×480 | Semi-dense (~30-50%) | Yes |
| **MVSEC** | Real LiDAR + IMU + MoCap | Fused multi-sensor | 346×260 | Sparse | Yes |
| **TartanAir v2** | Synthetic rendered | Unreal Engine | 640×640 | Dense (100%) | Yes |
| **M3ED pseudo** | DAv2 pseudo labels | DepthAnythingV2 on RGB | 1280×720 | Dense (100%) | No (relative) |

**Real GT is primary:** We predict sparse points — LiDAR sparsity is not a problem (sample queries at LiDAR-valid locations). LiDAR measures actual distance; pseudo labels inherit DAv2's biases.

**DAv2 pseudo labels (supplementary):** Dense supervision where no LiDAR exists, relative depth structure between sparse points. Use with scale-invariant losses only ($L_{\text{silog}}$).

### 5.3 Query Sampling During Training

Mixed sampling policy per batch:
- 40% from LiDAR-valid pixels (real GT, highest quality supervision).
- 20% from DAv2 pseudo-labeled pixels without LiDAR (dense coverage augmentation).
- 15% event-dense regions (high-signal areas where event cameras are most reliable).
- 25% high-gradient regions from depth maps (boundary quality — sample near depth edges).

Vary $K \in \{32, 64, 128, 256\}$ across batches (query curriculum) for robustness to different query loads.

When a query falls on a LiDAR-valid pixel, use the real LiDAR depth as $\rho^*(q)$. When it falls on a pseudo-labeled pixel, use the DAv2 prediction (with scale-invariant loss only, not $L_{\text{point}}$ which assumes metric accuracy).

### 5.4 Training Schedule

**Stage 1 — Relative depth (mixed supervision):**
- Labels: Real LiDAR GT (primary) + DAv2 pseudo labels (supplementary).
- For LiDAR-valid queries: $L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}}$.
- For pseudo-label queries: $\lambda_{\text{si}} L_{\text{silog}}$ only (no pointwise Huber — pseudo labels lack metric accuracy).
- Backbone: Frozen.
- Datasets: M3ED (LiDAR + pseudo), DSEC (stereo GT), TartanAir v2 (synthetic GT).
- Duration: Until convergence (~15-20 epochs).

**Stage 2 — Add center regularization:**
- Enable $L_{\text{ctr}}$.
- Continue training (~10 epochs).

**Stage 3 — Metric fine-tuning:**
- Labels: Real LiDAR / stereo GT only (no pseudo labels).
- Loss at valid GT pixels only: full $L_{\text{point}} + \lambda_{\text{si}} L_{\text{silog}} + \lambda_{\text{ctr}} L_{\text{ctr}}$.
- Optionally unfreeze F^3 backbone with 0.1× learning rate.
- Evaluate on MVSEC outdoor_day1/day2 (standard benchmark).

### 5.5 Regularization

**Attention dropout:** Standard dropout ($p = 0.1$) on cross-attention weights in B4, following standard transformer practice. This mildly regularizes which context tokens dominate per query.

**Center collapse prevention** relies on $L_{\text{ctr}}$ (auxiliary loss on pre-fusion center token), following DETR's auxiliary loss approach.

**Standard:** Weight decay 0.01, gradient clipping 1.0, mixed precision (bf16).

---

## 6. Runtime Analysis

### 6.1 Parameter Budget

**v8 separates parameters into two categories:** Phase A preprocessing (computed once per frame, amortized over all queries) and Phase B decoder (per-query cost). The v8 backbone refines v7 with L1 GELU, L2 widened to 64ch, GRN in ConvNeXt blocks, and core dimension widened to $d = 192$.

**Phase A: Preprocessing parameters (amortized, once per frame)**

| Component | Parameters |
|-----------|------------|
| A2: L1 LN + GELU (identity from F^3 ds2) | ~0.1K |
| A2: Stem Conv (64→64, k3s2) + LN | ~37K |
| A2: L2 — 2× ConvNeXt$_{64}$ (GRN) | ~73.3K |
| A2: Down L2→L3 (64→96, k2s2) + LN | ~24.9K |
| A2: L3 — 2× ConvNeXt$_{96}$ (GRN) | ~159K |
| A2: Down L3→L4 (96→192, k2s2) + LN | ~73.9K |
| A2: L4 — 4× SwinBlock$_{192}$ (w=8, shifted) | ~1,778K |
| A2: Per-level $W_V$ pre-proj (64/96/192→192) | ~68K |
| A3: Latent bank projection (192→192) | ~37K |
| A4: Calibration heads ($s_t$, $b_t$) | ~0.4K |
| **Phase A total** | **~2,252K** |

L4's Swin blocks dominate Phase A (~1,778K, 79%) but are computationally cheap at 80×45 (~0.5 ms for all 4 blocks). With F^3 ds2, L1 reduces from ~18.5K (conv) to ~0.1K (LN only); Stem increases from ~33K to ~37K (wider input). Net Phase A change from ds1: -14K (negligible).

**Phase B: Decoder parameters (per-query cost, $d = 192$)**

| Component | Lookups | Attn ops | Params | Detail |
|-----------|:---:|:---:|---:|--------|
| B1: Centers (L1 + L2–L4) | 4 | — | ~142K | MLP 544→192→192 |
| B1: L1 local sampling | 32 | — | ~54K | Local MLP (80→192) + offsets |
| B1: L2 local | 16 | — | ~54K | Local MLP (80→192) + offsets |
| B1: Routing token | — | — | ~56K | MLP 288→192 |
| B2: Global summary | — | 512 | ~148K | 6-head cross-attn |
| B2: Routing scores | — | 512 | ~74K | Routing proj |
| B3: Deformable read | 3,168 | — | ~173K | Cond + offsets + $W_O$ (no $W_V$, pre-applied) |
| B4: $W_{\text{ms2/3/4}}$ proj | — | — | ~68K | 64/96/192→192 |
| B4: Type embeddings | — | — | ~1.3K | 7 × $d$ |
| B4: KV norm | — | — | ~0.4K | $\text{LN}_{\text{kv}}$ |
| B4: Fusion (2 layers) | — | 38×2 | ~889K | 6-head transformer |
| B5: Depth head | — | — | ~37K | MLP 192→192→1 |
| B5: $L_{\text{ctr}}$ aux | — | — | ~19K | MLP 192→96→1 |
| **Phase B total** | **3,220** | **~1,100** | **~1,714K** | |

B3 parameter breakdown: conditioning MLP (416→192, ~80K) + offset head (192→192, ~37K) + weight head (192→96, ~18.5K) + $W_O$ (192→192, ~37K). $W_V$ pre-projection (68K) is in Phase A.

B4 fusion dominates Phase B (~889K, 52%): 2 layers × (6-head cross-attn $4d^2$ + FFN $8d^2$ + LNs) = $24d^2 + \text{linear terms}$. This is the single largest $d$-dependent component and the main cost of widening from $d = 128$ to $d = 192$.

**Total trainable parameters: ~3,966K (~4.0M)**

**v8 vs v7:** Phase A: 2,180K → 2,252K (+3%, L2 widening + GRN + ds2 stem adjustment). Phase B: 808K → 1,714K (+112%, $d$ widening). Net: 2,988K → 3,966K (+33%). F^3 ds2 + backbone refinements + richer per-query representations ($d = 192$).

**vs DAv2-S:** Our decoder is **15× smaller** (1,714K vs 25M). Total params **6.3× smaller** (4.0M vs 25M). 3,220 lookups/query = 286× less than dense (921,600 pixels).

### 6.2 Speed Estimate (RTX 4090, 1280×720 events)

**Precompute (once per window):**
$$
T_{\text{precompute}} = T_{\text{F}^3}^{\text{ds2}} + T_{\text{backbone}} + T_{\text{pool+cal}} + T_{W_V\text{-proj}} \approx 4.5 + 2.0 + 0.06 + 0.05 = 6.6 \text{ ms}
$$

**F^3 ds2 timing breakdown (RTX 4090, estimated):**

| Component | ds1 (ms) | ds2 (ms) | Ratio | Notes |
|-----------|:--------:|:--------:|:-----:|-------|
| MRHE encoding | 0.15 | 0.15 | 1.0× | Hash tables (16 MB) fit L2; same N events |
| Scatter (`index_put_`) | 0.80 | 0.80 | 1.0× | Random atomics to 1280×720×8 tensor |
| `forward_variable` overhead | 0.25 | 0.25 | 1.0× | `repeat_interleave`, `.clone()`, coord scaling |
| ds\_layer\[0\] + LN | 0.35 | 0.20 | 0.57× | ds2: Conv k3s2 on full-res → half-res |
| 3× ConvNeXt blocks (stage 0) | 2.85 | 1.20 | 0.42× | 0.50× traffic × 0.85 L2 cache bonus |
| ds\_layer\[1\] + LN | 0.30 | 0.15 | 0.50× | |
| 3× ConvNeXt blocks (stage 1) | 2.85 | 1.20 | 0.42× | 0.50× traffic × 0.85 L2 cache bonus |
| Kernel launch overhead | 0.20 | 0.20 | 1.0× | ~40 kernels × 5 μs |
| **F^3 total** | **~8.3** [^1] | **~4.5** | **0.54×** | |

[^1]: Calibration anchor: ds1 = ~8.3 ms from F^3 codebase benchmarks.

**Why FLOPs mislead here:** F^3 ds2 has ~57 GFLOPs vs ds1's ~65G — only 12% fewer. But the **wall-clock drops 46%** because F^3 is **memory-bandwidth-bound** on RTX 4090. The roofline analysis (see Section 8, ablation #34) shows all operations have arithmetic intensity 10–21 FLOP/byte, well below the 82 FLOP/byte ridge point. The 50% memory traffic reduction (from halving spatial resolution) and L2 cache fit (ds2's 59 MB feature maps fit the 72 MB L2; ds1's 118 MB do not) together account for the ~2× speedup on the conv backbone portion.

Our backbone adds ~2.0 ms (see A2 cost table: L1 LN+GELU ~0.01 ms, stem ~0.06 ms, L2 ConvNeXt$_{64}$ ~0.85 ms, L3 ConvNeXt$_{96}$ ~0.5 ms, L4 Swin ~0.5 ms, downsampling ~0.06 ms). The 32×16 latent bank pool from $F_t^{(4)}$ (80×45, 7:1 compression) is negligible; the 192→192 projection over 512 tokens adds ~0.01 ms. Following the Deformable DETR implementation pattern, B3's per-level $W_V$ value projection is pre-applied to the feature maps $F_t^{(2:4)}$ once per frame (~0.05 ms for mixed-channel projection: L2 64→192, L3 96→192, L4 192→192), so grid_sample reads already-projected 192ch features during per-query decoding.

**Per-query cost analysis:**

The decoder cost has two components: a **fixed overhead** (kernel launches and PyTorch dispatch, independent of $K$) and a **marginal cost** per query (compute + memory bandwidth, scaling with $K$):
$$
T_{\text{decoder}}(K) = \alpha + \beta K
$$

Per-query FLOPs breakdown (~27M FLOPs/query with pre-projected $W_V$, $d = 192$):

| Stage | Dominant operations | FLOPs |
|-------|-------------------|---:|
| B1: center + local + routing | MLP(544→192→192), L1/L2 local MLPs, 52 grid_samples | ~1,100K |
| B2: summary + routing | 6-head cross-attn over 512 tokens, scoring | ~1,050K |
| B3: deformable | 33× cond MLP(416→192), offset/weight heads, $W_O$, 3,168 grid_samples | ~11,600K |
| B4: projection + fusion | $W_{\text{ms}}$(64/96/192→192), 2× [6-head cross-attn + FFN(192→768→192)] | ~12,900K |
| B5: depth head | MLP(192→192→1) | ~74K |
| **Total** | | **~26.7M** |

For reference, the F^3 ds2 backbone processes ~57 GFLOPs per frame (vs ~65G for ds1, only −12% — see "Why FLOPs mislead" above). Our decoder at ~27M FLOPs/query is **~2,100× less compute** than one F^3 ds2 pass. The v8 widening to $d = 192$ doubles per-query FLOPs vs v7 ($d = 128$, ~13.1M), but the decoder remains negligible compared to precompute.

**Conservative estimates** (standard PyTorch, no custom CUDA, no torch.compile):
- $\alpha \approx 0.30$ ms (fixed kernel launch + dispatch overhead).
- $\beta \approx 0.007$ ms/query (compute scales ~2× with $d^2$, but grid_sample bandwidth scales only ~1.5× since the 3,220 lookups read 192ch vs 128ch pre-projected features; at these small matrix sizes, memory bandwidth dominates over compute).

**Total:**
$$
T_{\text{EventSPD}}(K) \approx 6.9 + 0.007K \text{ ms}
$$

| Query count $K$ | Decoder (ms) | EventSPD total (ms) | Throughput (Hz) | Dense baseline (ms) | Speedup |
|-----|---:|---:|---:|---:|---:|
| 1 | 0.31 | 6.9 | 145 | ~23 | 3.3× |
| 64 | 0.75 | 7.3 | 137 | ~23 | 3.1× |
| 256 | 2.09 | 8.7 | 115 | ~23 | 2.6× |
| 1024 | 7.47 | 14.1 | 71 | ~23 | 1.6× |

Dense baseline: F^3 ds1 (8.3 ms) + DepthAnythingV2 decoder (~15 ms) = ~23 ms. The crossover point where EventSPD matches dense cost: $(23 - 6.9) / 0.007 \approx K = 2{,}300$ — far beyond any practical query count. Even at $K = 1024$, EventSPD is **1.6× faster** than dense.

**v8 vs v7 speed impact:** F^3 ds2 saves ~3.8 ms (8.3→4.5 ms), the single largest speed improvement in v8. The $d = 192$ widening adds ~0.4 ms precompute and ~0.002 ms/query. Net effect at $K = 256$: 8.7 ms (v8) vs 11.6 ms (v7) — **25% faster** despite richer representations. The ds2 switch dominates; the $d$ widening is nearly free in wall-clock terms.

**Key insight:** Precompute (~6.6 ms) accounts for ~76% of total time at $K = 256$. F^3 ds2 (~4.5 ms) is still 68% of precompute — the event encoder remains the bottleneck. Our backbone (~2.0 ms) and decoder (~2.1 ms) together are less than F^3. This means **further speedups require either lighter event encoders or event voxel grid representations** (see Section 8, ablation #34).

**The key figure for the paper:** Plot $T_{\text{total}}(K)$ vs $K$ for EventSPD and dense-then-sample. The nearly flat EventSPD curve (dominated by precompute) contrasts sharply with the constant dense cost, demonstrating that sparse querying is structurally faster for all practical $K$.

---

## 7. Evaluation Protocol

### 7.1 Accuracy Metrics (at query points)

Standard monocular depth metrics computed only at queried pixels:
- AbsRel, RMSE, RMSE_log, SiLog.
- $\delta < 1.25$, $\delta < 1.25^2$, $\delta < 1.25^3$ thresholds.
- Report by depth range buckets (near: 0-10m, mid: 10-30m, far: 30m+).

### 7.2 Runtime Metrics (the key contribution)

- $T_{\text{precompute}}$ (ms): one-time cost per event window.
- $T_{\text{query}}(K)$ (ms): per-batch query cost for $K \in \{1, 4, 16, 64, 256, 1024\}$.
- $T_{\text{total}}(K)$ (ms): end-to-end latency.
- Throughput (Hz): $1000 / T_{\text{total}}(K)$.
- Speedup vs dense baseline.
- GPU memory usage.

### 7.3 Diagnostic Metrics

**Center predictiveness:**
- Monitor $L_{\text{ctr}}$ (the auxiliary loss from the 2-layer MLP on $h_{\text{point}}^{(0)}$). Decreasing $L_{\text{ctr}}$ = center branch is becoming independently informative.

**Routing diversity:**
- Entropy of $\alpha_q$ averaged over queries. Low entropy = potential router collapse.

**Attention balance:**
- Average cross-attention weight per context token type ($l_q$, $c_q^{(2)}$, $c_q^{(3)}$, $c_q^{(4)}$, $h_{\text{near}}$, $h_{\text{route}}$, $\bar{c}_q$) across queries. Shows whether the model uses all 7 information streams. Particularly monitor the multi-scale center tokens ($c_q^{(2:4)}$) — if their attention weights are near-zero, the backbone's multi-scale features are not contributing to fusion and may need stronger features or different type embeddings.

### 7.4 Robustness

- Day / night evaluation.
- Different platforms (car / spot / flying if available).
- Event rate subsampling ($25\%$, $50\%$, $75\%$, $100\%$).

---

## 8. Ablation Plan

Organized by priority. Each ablation answers one question.

### Tier 1 — Core validation (must have for paper)

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 1 | Remove global context | Does non-local information help depth? | Remove B2, B3 far anchors. Keep only $l_q$ and nearest anchor. |
| 2 | Remove local sampling | Does neighborhood context help? | Set $N_{\text{loc}} = 0$. Center + global only. |
| 3 | Remove deformable sampling | Is precise non-local evidence needed beyond $\bar{c}_q$? | Remove B3. Keep $T_q = [l_q; \bar{c}_q]$ (2 tokens). |
| 4 | Query count scaling | How do accuracy and speed scale with $K$? | $K \in \{1, 4, 16, 64, 256, 1024\}$. |
| 5 | Freeze vs fine-tune backbone | Does backbone adaptation help? | Frozen / 0.1× LR / full LR. |

### Tier 2 — Design choices

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 6 | No routing | Does sparse routing help? | All 512 tokens as anchors (49,152 grid_samples). |
| 7 | Residual design | Asymmetric residual? | (a) Both (current). (b) Drop FFN. (c) None. (d) +center skip. |
| 8 | Fusion depth | How many layers? | 1, 2, 3 layers. |
| 9 | Latent bank size | How many tokens? | $P_c \in \{128, 256, 512, 1024\}$. Default: 512. |
| 10 | Routing budget | How many routed? | $R \in \{8, 16, 32, 64, 128\}$. Default: 32. |
| 11 | Local budget | How many local? | $N_{\text{loc}} \in \{8, 16, 32, 48, 64\}$. |
| 12 | Deformable budget | Samples per anchor? | $(H,L,M) \in \{(4,3,2)..(8,3,6)\}$. Default: $(8,3,4)$. |

### Tier 3 — Extensions and alternatives

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 13 | Perceiver-style LatentPool | Do learned queries outperform spatial grid? | Replace A3 with MHA-based LatentPool. |
| 14 | Hash encoding vs Fourier PE | Does learned position encoding help? | Replace $\text{pe}_q$ with Instant-NGP hash. |
| 15 | Temporal memory | Does GRU state across windows help? | Add $H_t$ to cache and to $T_q$. |
| 16 | Uncertainty head | Does uncertainty improve hard queries? | Enable $\sigma_q$ + $L_{\text{unc}}$. |
| 17 | Center auxiliary loss | Does $L_{\text{ctr}}$ prevent center collapse? | Disable $L_{\text{ctr}}$ and compare. |
| 18 | Attention dropout rate | What rate is best for cross-attention regularization? | $p_{\text{attn}} \in \{0, 0.05, 0.1, 0.2\}$. |

### Tier 4 — Architecture audit ablations (backbone design)

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 19 | L1 stride | 2× vs 1× vs 4× stride? | (a) Conv k3s2 (current, 640×360). (b) 1×1 proj (1280×720, v6). (c) Conv k3s4 (320×180). |
| 20 | Swin block count | How many Swin blocks at L4? | (a) 4 blocks (current, ~1,778K). (b) 2 blocks (~889K). (c) 6 blocks (~2,667K). |
| 21 | L4 channel dim | 192ch vs 128ch vs 256ch? | (a) 192ch (current, ~1,778K). (b) 128ch (~800K). (c) 256ch (~3,150K). |
| 22 | Center token paths | Separate L3-L4 from $h_{\text{point}}$? | (a) L1+L2 in $h_{\text{point}}$ (current, 544ch). (b) All compressed (v5). (c) L1-only (288ch). |
| 23 | Offset bounding | Unbounded vs bounded? | (a) Unbounded+0.1×LR (current). (b) $\tanh$. (c) Clamp. |
| 24 | Multi-head deformable | Full $W_V$/$W_O$ vs simple? | (a) Full (91K). (b) Weighted sum (60K). |
| 25 | KV normalization | $\text{LN}_{\text{kv}}$ on context? | (a) With (current). (b) Without. |
| 26 | Auxiliary calibration | Calibrate $r_q^{\text{ctr}}$? | (a) Calibrated (current). (b) Uncalibrated. |
| 27 | L2 local in h_point | L2 local contribution? | (a) Full 544ch (current). (b) No $l_q^{(2)}$ (352ch). (c) L1-only (288ch). |
| 28 | Latent bank source | Which level for $C_t$? | (a) L4, 7:1 (current). (b) L3, 28:1. (c) L2, 112:1. |
| 29 | Backbone architecture | CNN-only vs Hybrid vs full Swin? | See configs below. |

**Ablation #29 — Backbone architecture variants:**

This is the key ablation for justifying the hybrid backbone design. Tests whether Swin Transformer at L4 contributes over CNN-only, and how much backbone depth matters. All configs use L2=64ch with GRN and L1 GELU (v8 defaults):

| Config | Architecture | A2 Params | Precompute | Global RF? |
|:---:|----------|---:|---:|:---:|
| (a) | AvgPool + 1×1 proj only (no backbone) | ~50K | ~4.7 ms | No |
| (b) | CNN-only: 2+2+4 ConvNeXt (no Swin) | ~1,200K | ~6.1 ms | No (~28px at L4) |
| (c) | CNN-light: 2+2+2 ConvNeXt (no Swin) | ~750K | ~5.8 ms | No (~14px at L4) |
| (d) | Hybrid-light: 2+2+2 Swin (current minus 2 Swin) | ~1,300K | ~5.8 ms | Partial |
| (e) | **Hybrid (default): 2+2+4 Swin** | **~2,161K** | **~6.6 ms** | **Yes** |
| (f) | Heavy: 2+4+6 Swin | ~3,500K | ~7.2 ms | Yes |

Config (a) establishes the accuracy floor without any backbone processing. Config (b) replaces L4's Swin blocks with ConvNeXt blocks (same count) — tests whether global attention matters vs local-only context. Config (c) reduces backbone depth uniformly. Config (d) uses fewer Swin blocks. Config (e) is the default. Config (f) tests whether more depth improves accuracy enough to justify the cost. The key comparison is (b) vs (e): does Swin's global attention at L4 measurably improve depth accuracy over CNN-only?

### Tier 5 — v8 widening ablations

These ablations justify the v8-specific changes: core dimension widening, L1 activation, L2 channel width, and GRN. Each isolates one variable against the v8 default configuration.

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 30 | Core dimension $d$ | Does $d = 192$ justify the cost over $d = 128$? | (a) $d = 128$: B4 FFN 128→512→128, 4-head attention. ~808K decoder, ~3.0M total. (b) **$d = 192$ (default)**: B4 FFN 192→768→192, 6-head. ~1,714K decoder, ~4.0M total. (c) $d = 256$: B4 FFN 256→1024→256, 8-head. ~2,940K decoder, ~5.2M total. |
| 31 | L1 activation | Does GELU on L1 help depth accuracy? | (a) **L1 GELU (default)**: Conv(32→64, k3s2) + LN + GELU. Zero extra params. (b) L1 GELU + extra conv: + Conv(64→64, k3, pad1) + LN + GELU (~37K). (c) L1 linear: Conv + LN only (no activation, v7 design). |
| 32 | L2 channel width | Is 64ch the right width for L2? | (a) L2 = 48ch (v7 default): 2× ConvNeXt$_{48}$. ~42K L2 blocks. $h_{\text{point}}$ input 528ch. (b) **L2 = 64ch (default)**: 2× ConvNeXt$_{64}$. ~73K L2 blocks. $h_{\text{point}}$ input 544ch. (c) L2 = 96ch: 2× ConvNeXt$_{96}$. ~159K L2 blocks. $h_{\text{point}}$ input 576ch. |
| 33 | GRN in ConvNeXt | Does Global Response Normalization help? | (a) **With GRN (default)**: ConvNeXt V2-style blocks (+2.6K total). (b) Without GRN: ConvNeXt V1-style blocks (v7 design). |

**Ablation #30 — Core dimension $d$:** This is the single most impactful v8 change. At $d = 128$ (v7), B4's 2-layer transformer has only $4 \times 32$-head attention, limiting capacity. At $d = 192$, we get $6 \times 32$-head attention (+50% heads) and wider FFN (768 vs 512 hidden). The cost is +906K decoder params (+112%). At $d = 256$, diminishing returns: +2,132K (+264%) for another +33% heads. The key question: does the richer per-query representation improve depth accuracy enough to justify the budget? Monitor B4 attention entropy — if $d = 128$ heads show saturation (high entropy / uniform attention), $d = 192$ is justified.

**Ablation #31 — L1 activation:** The v7 L1 was Conv + LN (linear output), following the ConvNeXt stem convention. But unlike a stem (which feeds into nonlinear blocks), L1 is terminal — it feeds directly into $h_{\text{point}}$ MLP and local sampling. GELU adds nonlinearity at zero parameter cost. The extra conv variant tests whether a second 3×3 conv (capturing more local patterns) helps beyond what the $h_{\text{point}}$ MLP already provides.

### Tier 6 — F^3 backbone resolution ablation

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 34 | F^3 output resolution | Does F^3 ds2 lose accuracy vs ds1? | See configs below. |

**Ablation #34 — F^3 output resolution:**

This ablation tests whether compressing the F^3 backbone from full-resolution (ds1: 1280×720×32) to half-resolution (ds2: 640×360×64) preserves depth accuracy while delivering significant speed gains. The ds2 config is the default in v8.

| Config | F^3 output | F^3 time | L1 op | Backbone time | Precompute | Total params |
|:---:|----------|---:|----------|---:|---:|---:|
| (a) | ds1: 1280×720×32 | ~8.3 ms | Conv(32→64, k3s2) + LN + GELU | ~2.0 ms | ~10.4 ms | ~4.0M |
| (b) | **ds2: 640×360×64 (default)** | **~4.5 ms** | **LN + GELU (identity)** | **~2.0 ms** | **~6.6 ms** | **~4.0M** |
| (c) | ds4: 320×180×64 | ~3.0 ms | — (skip L1) | ~1.5 ms | ~4.6 ms | ~3.9M |

**F^3 ds2 config** (proposed `1280x720x20_patchff_ds2_small.yml`):
```yaml
model: EventPatchFF
T: 20
convbtlncks: [2, 2]
convdepths: [3, 3]
convdilations: [1, 1]
convkernels: [7, 7]
dims: [64, 64]
dskernels: [3, 1]
dsstrides: [2, 1]
frame_sizes: [1280, 720, 20]
multi_hash_encoder:
  coarsest_resolution: [8, 8, 1]
  feature_size: 2
  finest_resolution: [320, 180, 8]
  levels: 4
  log2_entries_per_level: 19
patch_size: 2
use_decoder_block: false
variable_mode: true
# Receptive field: ~63x63 (in ds2 output coords)
# Feature downsample: 2x2
```

Config (a) is the ds1 baseline used in v7 — full 1280×720 output, Conv L1 stride-2 compress. Config (b) is the v8 default — F^3 directly outputs 640×360×64, making L1 a trivial LN+GELU. This saves ~3.8 ms precompute (37% faster). Config (c) pushes further to ds4 — F^3 outputs 320×180×64, skipping L1 entirely and feeding directly into the Stem. This saves ~5.8 ms total but loses half-resolution spatial detail for L1 local sampling. The key question: does the information lost in ds2's 2× spatial compression (partially compensated by 2× channel widening to 64ch) degrade depth accuracy at query points?

**Roofline analysis (RTX 4090): why FLOPs mislead for F^3**

All F^3 operations are **memory-bandwidth-bound** — their arithmetic intensity (AI) is far below the RTX 4090's ridge point:

| Operation | FLOPs | Bytes (R+W) | AI (FLOP/B) | Ridge = 82 |
|-----------|------:|------------:|:---:|:---:|
| dwconv(32, k7, g=32) @ 1280×720 | 2.89G | 236 MB | 12.2 | Memory |
| Linear(32→64) @ 1280×720 | 3.77G | 354 MB | 10.7 | Memory |
| dwconv(64, k7, g=64) @ 640×360 | 1.44G | 118 MB | 12.2 | Memory |
| Linear(64→128) @ 640×360 | 3.78G | 177 MB | 21.3 | Memory |

The pointwise convolutions dominate FLOPs (~83% of each block) and have **identical MACs** between ds1 and ds2 (4× fewer pixels × 4× wider channels = same compute), yet their memory traffic **halves** (proportional to $C \times H \times W$, linear in $C$, not quadratic).

**Memory traffic per ConvNeXt block** (b = bottleneck factor = 2):
$$
\text{Bytes}_{\text{block}} = (9 + 6b) \cdot C \cdot H \cdot W \cdot 4 = 21 \cdot C \cdot P \cdot 4
$$

| | ds1 (C=32, P=922K) | ds2 (C=64, P=230K) | Ratio |
|---|---:|---:|:---:|
| Per block | 2,478 MB | 1,239 MB | 0.50× |
| 6 blocks total | 14,868 MB | 7,434 MB | 0.50× |
| Feature map size | 118 MB | 59 MB | — |
| Fits RTX 4090 L2 (72 MB)? | No | **Yes** | — |

ds2's base feature maps (59 MB) fit in the RTX 4090's 72 MB L2 cache, enabling **inter-kernel data reuse** for ~40% of block operations (dwconv→LN→pwconv1 and pwconv2→residual chains). ds1's 118 MB feature maps thrash L2, forcing every operation through DRAM. Combined effect: conv backbone runs at ~0.43× ds1 time (0.50× traffic × ~0.85 L2 bonus).

**Total GFLOPs (feature output, excluding pred head):**
- ds1: ~65G — ds2: ~57G (**only −12%**)
- But wall-clock: ~8.3 ms → ~4.5 ms (**−46%**), because timing follows memory traffic, not FLOPs.

Note: MRHE hash scatter always creates a full 1280×720 intermediate tensor (~1.5 ms fixed), consuming ~1.2 ms for scatter + 0.3 ms for encoding — identical for both ds variants.

---

## 9. Implementation Structure

### 9.1 New files

```
src/f3/tasks/depth_sparse/
├── models/
│   ├── eventspd.py            # Main model: Algorithm A + B pipeline
│   ├── backbone.py             # A2: Hybrid Conv+Swin backbone (L1 conv, Stem, ConvNeXt L2-L3, Swin L4)
│   ├── latent_bank.py         # A3: spatial grid pool from L4 (192→192 proj) + A4 calibration heads
│   ├── query_encoder.py       # B1: L1 center + L2 local + multi-scale center reads + routing token
│   ├── context_retrieval.py   # B2: cross-attention summary + routing
│   ├── deformable_read.py     # B3: offset prediction + multiscale sampling (per-level W_V)
│   ├── fusion_decoder.py      # B4: 2-layer cross-attention transformer (38 tokens, 7 type embeddings)
│   └── depth_head.py          # B5: depth prediction + uncertainty
├── utils/
│   ├── losses.py              # L_point, L_silog, L_ctr
│   ├── query_sampler.py       # Training query sampling strategies
│   └── profiler.py            # Runtime benchmarking utilities
├── train.py                   # Training loop (stage 1 + 2)
├── evaluate.py                # Accuracy evaluation
└── benchmark.py               # Speed benchmarking
```

### 9.2 Reuse from existing codebase

- F^3 backbone init/load from `init_event_model` and existing checkpoints.
- Existing depth losses in `src/f3/tasks/depth/utils/losses.py` as templates for $L_{\text{silog}}$.
- Benchmarking patterns from `test_speed.py`.
- Config system from `confs/`.

### 9.3 Recommended build order

| Step | What | Validates | Est. effort |
|------|------|-----------|-------------|
| 1 | Dense baseline + query sampling | Accuracy/speed reference | 1 week |
| 2 | Minimal query decoder (bilinear L1 + MLP, no global) | Concept viability | 1 week |
| 3 | Add hybrid backbone (A2: L1 conv + ConvNeXt L2-L3 + Swin L4) | Feature quality improvement, multi-scale pyramid | 1-2 weeks |
| 4 | Add latent bank from L4 + cross-attention summary | Global context benefit | 1 week |
| 5 | Add deformable sampling (small budget, per-level $W_V$) | Precise non-local evidence | 1-2 weeks |
| 6 | Add routing (top-R) | Efficiency of sparse reads | 1 week |
| 7 | Full EventSPD pipeline + training (38 tokens, 7 types) | Complete system | 2 weeks |
| 8 | Ablations + paper figures | Publication readiness | 3-4 weeks |

**Critical rule:** After each step, measure accuracy AND speed. Do not proceed to the next step if the current one degrades accuracy without clear speed benefit.

---

## 10. Timeline (16 weeks)

### Weeks 1-2: Foundations
- Reproduce dense F^3 + DA-V2 baseline. Verify metrics and runtime.
- Build query-level evaluation harness (sample from dense output, compare).
- Build profiling harness for latency decomposition.

### Weeks 3-4: Minimal query decoder
- Implement build steps 2-3 (bilinear + MLP, then add latent bank).
- First accuracy comparison: minimal decoder vs dense baseline at query points.
- First speed comparison: latency curves.

### Weeks 5-7: Full EventSPD implementation
- Add deformable sampling, routing, fusion transformer (build steps 4-6).
- Train stages 1 and 2.
- Produce crossover plot: $T_{\text{total}}(K)$ for EventSPD vs dense.

### Weeks 8-10: Core ablations
- Tier 1 ablations (global context, local sampling, deformable, query scaling, backbone).
- Identify which components contribute most.
- Fix any failure modes discovered.

### Weeks 11-12: Design ablations
- Tier 2 ablations (routing budget, local budget, deformable budget, fusion depth).
- Tier 3 extensions if time permits (Perceiver pooling, hash encoding, temporal memory).

### Weeks 13-14: Robustness + metric fine-tuning
- Day/night evaluation, cross-platform, event subsampling.
- Stage 3 metric fine-tuning on LiDAR.

### Weeks 15-16: Paper
- Main tables and figures.
- Failure case analysis.
- Draft submission-ready method + experiments sections.

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|:---:|------------|
| Accuracy drop at depth edges | Medium | Increase local budget; edge-focused query sampling. |
| Router collapse | Low | Monitor entropy; add entropy regularizer. |
| Offsets collapse to zero | Low | 0.1× LR on offset heads (Deformable DETR). |
| Speed worse than dense | Very Low | ~27M FLOPs/query (~2,100× < F^3 ds2). Crossover at $K \approx 2{,}300$. |
| Backbone latency too high | Very Low | ~2.0 ms amortized (30% of precompute). Ablation #29 tests CNN-only and lighter configs. |
| Swin window boundary artifacts | Low | Shifted windows (alternating blocks) cover all boundary positions. Standard Swin practice. |
| Multi-scale tokens ignored | Low | Monitor attention weights on $c_q^{(2:4)}$. Ablation #22. |
| STE routing instability | Medium | Start without routing; add after convergence. |
| Pseudo depth label noise | Medium | Two-stage: pseudo → metric LiDAR fine-tune. |

---

## 12. References

**Core method references:**
- F^3 (Fast Feature Fields): https://arxiv.org/abs/2509.25146
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Segment Anything (SAM): https://arxiv.org/abs/2304.02643
- Perceiver IO: https://arxiv.org/abs/2107.14795
- LIIF: https://arxiv.org/abs/2012.09161
- PointRend: https://arxiv.org/abs/1912.08193
- Instant-NGP: https://arxiv.org/abs/2201.05989

**Depth estimation references:**
- DepthAnythingV2: https://arxiv.org/abs/2406.09414
- DPT (Vision Transformers for Dense Prediction): https://arxiv.org/abs/2103.13413
- BTS (Big to Small, ASPP for depth): https://arxiv.org/abs/1907.10326
- AdaBins (ASPP for depth): https://arxiv.org/abs/2011.14141
- PixelFormer (ASPP for depth): https://arxiv.org/abs/2301.01255
- E2Depth: https://arxiv.org/abs/2010.08350
- RAM-Net: https://arxiv.org/abs/2102.09320
- Feature Pyramid Networks: https://arxiv.org/abs/1612.03144

**Architecture references:**
- Swin Transformer (Liu et al., ICCV 2021): https://arxiv.org/abs/2103.14030
- ConvNeXt (Liu et al., CVPR 2022): https://arxiv.org/abs/2201.03545
- ConvNeXtV2 (Woo et al., CVPR 2023): https://arxiv.org/abs/2301.00808
- EfficientDet / BiFPN (Tan et al., 2020): https://arxiv.org/abs/1911.09070
- DeepLabV3 (ASPP, Chen et al., 2017): https://arxiv.org/abs/1706.05587
- Deformable ConvNets v2 (DCNv2): https://arxiv.org/abs/1811.11168
- DAT (Deformable Attention Transformer, CVPR 2022): https://arxiv.org/abs/2201.00520
- DAT++ (2023): https://arxiv.org/abs/2309.01430
- DAB-DETR (Dynamic Anchor Boxes, ICLR 2022): https://arxiv.org/abs/2201.12329
- DCNv4 (CVPR 2024): https://arxiv.org/abs/2401.06197
- GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- Differentiable Top-k: https://arxiv.org/abs/2002.06504
- Fourier Features (Tancik et al., NeurIPS 2020): https://arxiv.org/abs/2006.10739

**Datasets:**
- DSEC (stereo events + LiDAR): https://arxiv.org/abs/2103.06011
- MVSEC (multi-vehicle stereo events): https://arxiv.org/abs/1801.10202
- M3ED (multi-robot multi-sensor): https://arxiv.org/abs/2210.13093
- TartanAir (synthetic, Unreal Engine): https://arxiv.org/abs/2003.14338

**Competing / related work:**
- InfiniDepth (LIIF-style depth, Jan 2026): https://arxiv.org/abs/2601.03252
- EventDAM (ICCV 2025): Dense event depth via distillation from DAM.
- DepthAnyEvent (ICCV 2025): DAv2 teacher-student for event depth.
- No existing work targets sparse query-point depth from events.
