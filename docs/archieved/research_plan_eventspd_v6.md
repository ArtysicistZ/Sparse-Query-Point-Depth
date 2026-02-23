# Research Plan: Streamlined EventSPD — Sparse Query-Point Depth from Events

Author: Claude (redesigned from Codex v4, audited v5, enriched pyramid v6)
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
| $\mathcal{F}_{\text{F}^3}$ | Backbone network | Event-to-feature encoder | Frozen / fine-tuned |
| $F_t$ | $\mathcal{F}_{\text{F}^3}(E_t)$ | Dense shared feature field | Via backbone |
| $F_t^{(1)}$ | $\text{Proj}_{1\times1}(F_t)$ | Fine features (raw, 128ch) | Proj: Yes |
| $F_t^{(2..4)}$ | StridedConv/ProgConv → ASPP → BiFPN → ConvNeXt from $F_t$ | Enriched pyramid (128/128/256ch) | Yes |
| $s_\ell$ | $s_1{=}1, s_2{=}4, s_3{=}8, s_4{=}16$ | Stride of level $\ell$ | No |
| $C_t$ | $\text{Proj}(\text{GridPool}(F_t^{(4)}))$ | Latent bank from fully-enriched L4 (256→128) | Proj: Yes |
| $s_t, b_t$ | $\text{Heads}(\text{Pool}(C_t))$ | Global depth scale / shift | Yes |
| $q = (u, v)$ | User input | Query pixel coordinate | No |
| $f_q^{(1)}$ | $\text{Bilinear}(F_t^{(1)}, q)$ | Fine point feature (128ch) | No |
| $c_q^{(\ell)}$ | $\text{Bilinear}(F_t^{(\ell)}, q)$, $\ell{=}2,3,4$ | Enriched center (128/128/256ch) | No |
| $l_q$ | MaxPool(Local) from $F_t^{(1)}$ | L1 local context (32 samples) | MLP: Yes |
| $l_q^{(2)}$ | MaxPool(Local) from $F_t^{(2)}$ | L2 enriched local (16 samples) | MLP: Yes |
| $\text{pe}_q$ | $\text{Fourier}(u/W, v/H)$ | Positional encoding (32d) | No |
| $h_{\text{point}}$ | MLP$([f_q^{(1)}; c_q^{(2)}; \text{pe}; l_q; l_q^{(2)}])$ | Center token (L1 + L2 context) | Yes |
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

Core dimension: $d = 128$. Positional encoding: $d_{\text{pe}} = 32$ (8 Fourier frequencies × 2 trig functions × 2 spatial dims).

### 4.3 Algorithm A: Precompute Once Per Event Window

**Input:** Event set $E_t$.
**Output:** $\text{cache}_t = \{F_t^{(1)}, F_t^{(2)}, F_t^{(3)}, F_t^{(4)}, C_t, s_t, b_t\}$ where $F_t^{(2:4)}$ are fully enriched (StridedConv/ProgConv → ASPP → BiFPN → ConvNeXt).

#### A1. Backbone encoding

$$
F_t = \mathcal{F}_{\text{F}^3}(E_t)
$$

Shared event-derived feature field for the entire window. This is the dominant cost (~8.3 ms at 120 Hz HD on RTX 4090).

#### A2. Enriched feature pyramid (ASPP preprocessing)

**Motivation (v6 change):** The F^3 backbone produces 32-channel features trained for event prediction, not depth. DAv2's DPT decoder achieves depth understanding through ~16 sequential conv layers at 256ch that teach each feature about its spatial neighborhood (texture, edges, depth discontinuities). Our v5 design read raw features — each sample carried limited contextual information. The v6 enriched pyramid addresses this gap through a four-stage pipeline per level: (1) **strided convolution projections** that learn spatial patterns within each stride cell (replacing lossy AvgPool), (2) **ASPP** (Atrous Spatial Pyramid Pooling) for multi-scale local context, (3) **BiFPN** (Bidirectional Feature Pyramid Network) for cross-level communication, and (4) **ConvNeXt blocks** for deep nonlinear refinement. This pipeline achieves 9–15 sequential layers of depth per level, approaching DPT's 16 layers, while remaining efficient through depthwise-separable operations and amortization over all queries.

**Level 1** — raw fine-grained features (unchanged):
$$
F_t^{(1)} = \text{Proj}_{1 \times 1}(F_t) \quad \in \mathbb{R}^{1280 \times 720 \times d}
$$

Level 1 preserves the full spatial resolution for precise center and local reads. No enrichment is applied — L1 at 1280×720 is too large for affordable 128ch convolution (~127 GFLOPs per block). Fine-grained L1 information is preserved via $h_{\text{point}}$ and the residual connection in B4.

**Levels 2–4** — fully enriched context features:
$$
F_t^{(2)} = \text{BiFPN}_{L2}(\text{ASPP}_{128}(\text{StridedConv}_{32 \to 128}^{k4,s4}(F_t)), \ldots) \to \text{ConvNeXt}_{128} \quad \in \mathbb{R}^{320 \times 180 \times d}
$$
$$
F_t^{(3)} = \text{BiFPN}_{L3}(\text{ASPP}_{256 \to 128}(\text{ProgConv}_{32 \to 128}^{s8}(F_t)), \ldots) \to \text{ConvNeXt}_{128} \quad \in \mathbb{R}^{160 \times 90 \times d}
$$
$$
F_t^{(4)} = \text{BiFPN}_{L4}(\text{ASPP}_{256}(\text{ProgConv}_{32 \to 256}^{s16}(F_t)), \ldots) \to 2 \times \text{ConvNeXt}_{256} \quad \in \mathbb{R}^{80 \times 45 \times d_4}
$$

where $d = 128$ and $d_4 = 256$. The full per-level pipeline is:
```
F_t (1280×720, 32ch)
  ├→ L1: Proj_1×1 (32→128)                                              [1280×720×128, raw]
  ├→ L2: StridedConv(32→128, k4s4) → ASPP_128 → BiFPN → 1×ConvNeXt_128  [320×180×128]
  ├→ L3: ProgConv(32→128, s8)     → ASPP_256→128 → BiFPN → 1×ConvNeXt_128  [160×90×128]
  └→ L4: ProgConv(32→256, s16)    → ASPP_256 → BiFPN → 2×ConvNeXt_256  [80×45×256]
```

Four scales at strides 1×, 4×, 8×, 16× relative to $F_t$. For HD input (1280×720): $F_t^{(1)}$ is 1280×720, $F_t^{(2)}$ is 320×180, $F_t^{(3)}$ is 160×90, $F_t^{(4)}$ is 80×45.

**Stage 1 — Strided convolution projections (replacing AvgPool):**

The v5 design used AvgPool + 1×1 projection to create coarse levels. At stride 16×, AvgPool averages 256 pixels (16×16 cell) into a single value — a 256:1 spatial compression that destroys within-cell spatial arrangements (textures, edges, gradients). This is the same limitation that ViT's patch embedding and ConvNeXt's stem address with strided convolution: let the network learn which spatial patterns within each stride cell are informative.

For L2 (stride 4×), a single strided convolution suffices:
$$
\text{StridedConv}_{L2}(F_t) = \text{GELU}(\text{BN}(\text{Conv2d}(32, 128, k{=}4, s{=}4)(F_t))) \quad \in \mathbb{R}^{320 \times 180 \times 128}
$$

Parameters: $32 \times 128 \times 4 \times 4 + 128 \approx 65\text{K}$.

For L3 (stride 8×) and L4 (stride 16×), progressive strided convolutions maintain intermediate representations at each halving:
$$
\text{ProgConv}_{L3}(F_t) = \text{Conv}(64, 128, k{=}2, s{=}2)(\text{GELU}(\text{BN}(\text{Conv}(32, 64, k{=}4, s{=}4)(F_t)))) \quad \in \mathbb{R}^{160 \times 90 \times 128}
$$
$$
\text{ProgConv}_{L4}(F_t) = \text{Conv}(128, 256, k{=}2, s{=}2)(\text{Conv}(64, 128, k{=}2, s{=}2)(\text{GELU}(\text{BN}(\text{Conv}(32, 64, k{=}4, s{=}4)(F_t))))) \quad \in \mathbb{R}^{80 \times 45 \times 256}
$$

Parameters: L3 ~66K, L4 ~197K. Total projection parameters: ~328K (vs ~16K for the old AvgPool + 1×1 projections). The progressive convolutions also contribute 2–3 sequential layers of free depth (each conv is one layer of learned transformation), partially closing the gap with DPT's 16-layer depth.

**Why strided conv over AvgPool:** AvgPool at stride 16× compresses a 16×16 = 256-pixel cell into a single mean, losing all within-cell spatial structure. A strided convolution with kernel size matching the stride learns a weighted spatial combination — it can detect edges, textures, and gradients within each cell. This is the standard approach in modern architectures: ViT's patch embedding (Conv2d with k=16, s=16), ConvNeXt's stem (two Conv2d layers with s=4 and s=4), and Swin Transformer's patch merging all use learned projections rather than pooling. Progressive convolutions for larger strides (L3, L4) maintain intermediate channel dimensions, avoiding the extreme compression of a single 32→256 strided conv at stride 16.

**Stage 2 — ASPP module (depthwise separable, per level):**

Each ASPP module applies parallel dilated convolution branches at different rates, capturing multi-scale spatial context in a single pass:

```
Input (C_in channels)
├── 1×1 conv (C_in → C_branch)                      # point-level features
├── DW 3×3 dilation=r₁ → PW 1×1 (C_in → C_branch)  # medium dilated
├── DW 3×3 dilation=r₂ → PW 1×1 (C_in → C_branch)  # large dilated
├── DW 3×3 dilation=r₃ → PW 1×1 (C_in → C_branch)  # very large dilated
└── GlobalAvgPool → 1×1 (C_in → C_branch)           # full scene context
Concat (5 × C_branch) → 1×1 conv → C_out + BatchNorm + GELU
```

Per-level configuration:

| Level | Stride | Resolution | $C_{\text{branch}}$ | $C_{\text{out}}$ | Dilation rates |
|:---:|:---:|:---:|:---:|:---:|:---:|
| L2 | 4× | 320×180 | 128 | 128 | (6, 12, 18) |
| L3 | 8× | 160×90 | 256 | 128 | (6, 12, 18) |
| L4 | 16× | 80×45 | 256 | 256 | (3, 6, 12) |

L3 uses 256ch internal branches with 128ch output (wider intermediate representations capture richer context at 2× lower cost than full 256ch). L4 uses full 256ch throughout — at 80×45, the cost is negligible. L4's reduced dilation rates (3, 6, 12) are adapted for its smaller spatial dimensions (effective kernel 25 < feature map size 45). ConvNeXt refinement blocks are applied **after BiFPN** (see Stage 4 below), not directly after ASPP — BiFPN first exchanges cross-level information, then ConvNeXt deepens the representation.

**Context coverage in original pixel coordinates (per ASPP module):**

| Level | Stride | Dilated RFs | Original-coord context | + Backbone 63px RF |
|:---:|:---:|:---:|:---:|:---:|
| L2 | 4× | 13, 25, 37 px | 52, 100, **148 px** | ~211 px |
| L3 | 8× | 13, 25, 37 px | 104, 200, **296 px** | ~359 px |
| L4 | 16× | 7, 13, 25 px | 112, 208, **400 px** | ~463 px |

One ASPP pass at L2 covers 148px context radius — equivalent to 4+ ConvNeXt blocks (each with 7×7 DW conv covering ~7px per layer). The global average pooling branch additionally provides full scene context at each level.

**Why ASPP over stacked ConvNeXt blocks:** ASPP (DeepLabV3, Chen et al. 2017) captures multi-scale context in a **single parallel pass** via dilated convolutions, while ConvNeXt blocks build context incrementally (~7px per layer, sequentially). For depth estimation specifically, ASPP is proven: BTS, AdaBins, and PixelFormer all use ASPP-style modules. The key advantage is that dilated 3×3 convolutions achieve large effective receptive fields (up to 37×37 at dilation 18) with only 9 weights per channel — the same cost as a standard 3×3 conv.

**Why depthwise separable:** Following MobileNetV3, using depthwise convolutions for the spatial dilated branches and pointwise 1×1 convolutions for channel mixing reduces FLOPs by ~8× compared to standard ASPP. The spatial context aggregation (DW branch) and channel mixing (PW branch) are factored independently.

**Why strides 1×, 4×, 8×, 16× (v6 change from 1×, 2×, 4×, 8×):** The stride-2× level (v5's L2 at 640×360) was the most expensive level to enrich — at 230,400 pixels, 128ch convolution costs ~32 GFLOPs per block. Shifting L2 to stride-4× (320×180) makes 128ch ASPP affordable (~8.7 GFLOPs). The stride gap from L1 (1×) to L2 (4×) is acceptable because these serve different roles: L1 provides precise spatial detail for the query point (center reads + local samples), while L2-L4 provide enriched contextual information for the fusion transformer. Deformable attention at L2 with unbounded offsets still covers a wide spatial range.

**Why level 1 stays raw (32ch backbone features):** Applying ASPP at 1280×720 would cost ~127 GFLOPs per 128ch block — 2× the entire backbone. L1's role is precise spatial reads at the query point (center bilinear lookup + 32 local samples). The per-query MLP in B1 already transforms these 32ch features to 128ch. Backbone fine-tuning (ablation #5) is the primary lever for L1 feature quality.

**Why independent projections into ASPP (from v5):** Each level is projected independently from raw $F_t$: strided conv → ASPP. Independent projection avoids cascaded error accumulation, following the ViTDet Simple Feature Pyramid principle. With only 32 backbone channels, each channel carries significant information and is sensitive to projection noise. Cross-level communication happens **after** per-level ASPP enrichment, via BiFPN (Stage 3).

**Stage 3 — BiFPN cross-level fusion (2 rounds):**

After per-level ASPP enrichment, BiFPN (Bidirectional Feature Pyramid Network, from EfficientDet — Tan et al., 2020) adds bidirectional cross-level communication. This is the component that closes the biggest gap with DPT: DPT's FPN-style decoder propagates information top-down and bottom-up across 4 levels, while our v5 design had zero cross-level flow — each level was processed in isolation. BiFPN fixes this.

Each BiFPN round consists of top-down and bottom-up passes across L2–L4:

```
Round structure (top-down then bottom-up):
  Top-down:
    Step 1: L3' = DW_Conv(w1·L3 + w2·Upsample(Proj_256→128(L4)))
    Step 2: L2' = DW_Conv(w1·L2 + w2·Upsample(L3'))
  Bottom-up:
    Step 3: L3'' = DW_Conv(w1·L3 + w2·L3' + w3·Downsample(L2'))
    Step 4: L4'  = DW_Conv(w1·L4 + w2·Downsample(Proj_128→256(L3'')))
```

Fusion weights ($w_1, w_2, w_3$) are learned per-node with fast normalized fusion (softmax over positive-valued weights, following EfficientDet). Each DW_Conv is a depthwise separable 3×3 conv with BN + GELU. Two rounds are applied sequentially.

**Mixed-channel handling:** L2 and L3 operate at 128ch; L4 at 256ch. Channel adapters (1×1 conv) are applied at each cross-level connection: $\text{Proj}_{256 \to 128}$ for L4→L3 top-down, $\text{Proj}_{128 \to 256}$ for L3→L4 bottom-up. These are shared across rounds.

**BiFPN parameters (2 rounds, 3 levels):**
- Per-round DW-sep convs: 4 nodes × DW-sep conv (L2/L3 at 128ch, L4 at 256ch)
- Channel adapters: 2 × (256×128 + 128×256) = ~66K
- Fusion weights: 4 nodes × 2-3 weights × 2 rounds ≈ negligible
- Total: ~272K

**BiFPN effective depth (per level, 2 rounds):**
- L2: 2 DW-sep convs (1 per round in bottom-up only at first round, then both passes) → ~2–3 effective layers
- L3: 4 DW-sep convs (2 per round, receives from both L2 and L4) → ~4–6 effective layers
- L4: 2 DW-sep convs (1 per round) → ~2–3 effective layers

Note: BiFPN's primary value is **cross-level information flow** (not raw sequential depth). After 2 rounds, each level has seen information from both adjacent levels — L3 has the richest context (receives from both L2 and L4 in both passes). This addresses the 10% gap identified in our DPT comparison: DPT's FPN provides bidirectional cross-level flow; BiFPN is the direct modern equivalent.

**Why BiFPN over FPN:** Standard FPN is top-down only (one-way information flow). BiFPN adds a bottom-up pass, enabling fine-grained features (L2) to inform coarser levels. EfficientDet showed that BiFPN consistently outperforms FPN, PANet, and NAS-FPN across detection scales. The cost difference is minimal (2 additional DW-sep convs per round for bottom-up). With learned weighted fusion (not just addition), each node controls how much to incorporate from each input based on feature informativeness.

**Stage 4 — ConvNeXt refinement blocks:**

After BiFPN, ConvNeXtV2 blocks provide additional sequential depth with deep nonlinear processing. The block counts are chosen to bring each level's total sequential depth close to DPT's 16 layers:

| Level | StridedConv | ASPP | BiFPN (2 rnds) | ConvNeXt | **Total depth** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| L2 | 1 layer | 3 layers | 2–3 layers | 1 block (3 layers) | **~9–10 layers** |
| L3 | 2 layers | 3 layers | 4–6 layers | 1 block (3 layers) | **~12–14 layers** |
| L4 | 3 layers | 3 layers | 2–3 layers | 2 blocks (6 layers) | **~14–15 layers** |

L4 gets 2 ConvNeXt blocks (vs 1 in the old design) to compensate for BiFPN's lower depth contribution at the endpoint levels. L2 gains its first ConvNeXt block — the ASPP-only design had zero post-processing at L2. The per-query decoder (B4) adds ~8 more layers, but only at query points; the stored features carry the depth shown above.

**Preprocessing cost (amortized over all queries):**

| Component | L2 (320×180) | L3 (160×90) | L4 (80×45) | Total |
|-----------|:---:|:---:|:---:|:---:|
| StridedConv/ProgConv | ~0.05 ms | ~0.03 ms | ~0.02 ms | ~0.10 ms |
| ASPP | ~1.1 ms | ~0.6 ms | ~0.4 ms | ~2.1 ms |
| BiFPN (2 rounds) | — | — | — | ~0.50 ms |
| ConvNeXt | ~0.30 ms | ~0.12 ms | ~0.20 ms | ~0.62 ms |
| **Per-level total** | **~1.45 ms** | **~0.75 ms** | **~0.62 ms** | **~3.22 ms** |

Note: BiFPN cost is shared across levels (cross-level operation). Total enrichment adds ~3.2 ms to Phase A, representing 39% of backbone time (8.3 ms). This is fully amortized over all queries — at K=256, the per-query amortized cost is 0.013 ms. The enrichment benefits cascade through the entire pipeline: center reads, deformable reads, latent bank quality, and routing accuracy all improve from deeply context-aware, cross-level-informed features.

#### A3. Compact latent bank (spatial grid pooling)

Divide the enriched $F_t^{(4)}$ into a $P_h \times P_w$ spatial grid. Each cell is adaptive-average-pooled and projected to core dimension:

$$
C_t = \text{Proj}_{d_4 \to d}(\text{AdaptiveAvgPool}_{P_h \times P_w}(F_t^{(4)})) \in \mathbb{R}^{P_c \times d}
$$

where $\text{Proj}_{d_4 \to d}$ is a learned linear projection $256 \to 128$ (~33K parameters).

Default: $P_h = 16, P_w = 32, P_c = 512$.

For HD input: $F_t^{(4)}$ is $80 \times 45$ (stride 16×), so each cell covers $2.5 \times 2.8125$ level-4 positions ($\approx 40 \times 45$ original pixels). Compression ratio: $3{,}600 / 512 \approx 7 : 1$.

Each token $c_i$ has:
- **Content**: pooled and projected features from its spatial cell. Since $F_t^{(4)}$ is fully enriched — strided conv projection + ASPP + BiFPN cross-level fusion + 2×ConvNeXt refinement (256ch, with 112-400px context per feature, plus cross-level information from L2/L3 via BiFPN) — each latent token carries the richest, most context-aware information in the pyramid. Even after 256→128 projection, these tokens retain more discriminative content than L3-sourced tokens (128ch, 28:1 compression).
- **Location**: the known center coordinate $\mathbf{p}_i$ of its cell in pixel coordinates (not learned — deterministic from grid geometry). We write $\mathbf{p}_i^{(\ell)} = \mathbf{p}_i / s_\ell$ for the position in level-$\ell$ native coordinates.

**Why pool from enriched $F_t^{(4)}$ (v6 design choice):**
- **Coarse global purpose:** The latent bank serves as a compact global scene description for routing, summary, and anchor conditioning. L4 at stride-16× (80×45) with ASPP context of 112-400px provides the broadest spatial context per position in the pyramid — each L4 feature already encodes a large spatial neighborhood, well-matched to the bank's role as a coarse scene map.
- **Lowest compression ratio:** 7:1 compression (3,600 → 512) vs 28:1 from L3 (14,400 → 512) or 112:1 from L2 (57,600 → 512). Each pooled cell averages only ~7 features instead of ~28, preserving far more discriminative content per token. This directly improves routing quality in B2 (more distinctive tokens → finer routing discrimination) and B3 anchor conditioning (richer $c_r$ → better offset prediction).
- **Richer source features (256ch):** L4 features are 256-dimensional after ASPP + ConvNeXt processing. Even after the learned 256→128 projection, each token carries information distilled from a richer representation than L3's native 128ch. The projection learns to preserve the most discriminative dimensions for routing and conditioning.
- **Full enrichment pipeline:** L4 features are processed through strided conv projection → ASPP → BiFPN (receiving cross-level information from L2/L3) → 2×ConvNeXt$_{256}$ — the deepest enrichment in the pyramid (~14-15 sequential layers). The BiFPN cross-level fusion means L4 tokens carry information not just from their local receptive field, but from finer-scale features at L2 and L3.
- Fine-grained spatial detail is recovered via direct bilinear lookups (B1) and deformable sampling (B3), not from the latent bank.
- Anchor positions in original-resolution coordinates are identical regardless of source level (32×16 grid cells map to the same physical locations).

**Why $P_c = 512$:**
- Each F^3 backbone feature has a 63×63 pixel receptive field. With $P_c = 512$ (32×16 grid), each cell covers ~40×45 original pixels — comparable to the receptive field size, preserving discriminative content. Going finer ($P_c = 1024$, 32×32) risks top-R anchor clustering: large uniform regions (road, sky) produce many cells with near-identical content, and top-R over-selects from the same region, wasting anchor budget. With $P_c = 512$, each region produces fewer candidate cells, naturally promoting spatial diversity in anchor selection.
- Routing precision: $R = 32$ out of 512 (6.25%) gives generous anchor coverage for complex outdoor scenes (forest, city roads) where depth structure spans many distinct regions (10-20+ depth layers). Still within MoE sparsity norms (DeepSeek-V3 routes 8/256 = 3.1%), while keeping the candidate pool concentrated enough to avoid clustering.
- Cross-attention cost over 512 tokens remains trivial ($1 \times 512$ attention matrix per head, ~130K ops per query — negligible vs B3's 3,168 grid_sample calls).
- Parameters are unchanged: $W_Q$, $W_K$, $W_V$ are all $d \times d$ regardless of $P_c$.
- Ablate with $P_c \in \{128, 256, 512, 1024, 2048, 4096\}$ to find the accuracy/efficiency sweet spot.

**Why spatial grid pooling instead of Perceiver-style learned queries:**
- Spatial tokens have explicit locations, enabling deformable sampling in B3 without extra position learning.
- Simpler to implement and debug.
- The query-level cross-attention in B2 already provides semantic aggregation across all tokens.
- If ablations show learned queries significantly outperform spatial grid, swap in Perceiver-style LatentPool.

#### A4. Global calibration heads

$$
s_t = \text{softplus}(h_s(\text{MeanPool}(C_t))), \quad b_t = h_b(\text{MeanPool}(C_t))
$$

Window-level scale and shift for depth calibration. All queries in the same window share the same $(s_t, b_t)$, ensuring global consistency. $h_s$ and $h_b$ are single linear layers.

**Why this is needed:** Monocular depth from a single event window has scale ambiguity. $(s_t, b_t)$ injects scene-level calibration so per-query predictions are globally consistent. This follows standard practice in MiDaS, ZoeDepth, and F^3's own depth pipeline.

---

### 4.4 Algorithm B: Per-Query Sparse Depth Inference

**Input:** $\text{cache}_t$ and query batch $Q = \{q_j\}_{j=1}^{K}$.
**Output:** $\{(\hat{d}_j, \sigma_j)\}_{j=1}^{K}$.

All steps below are batched over $K$ queries in parallel.

#### B1. Feature extraction and token construction

**Implementation note — coordinate normalization:** All bilinear lookups throughout B1–B3 use `F.grid_sample` with `align_corners=False` and `padding_mode='zeros'`. Pixel coordinates $p$ are normalized to $[-1, 1]$ via $\text{grid} = 2p / \text{dim} - 1$ before sampling, following the Deformable DETR convention.

**Fine center feature (L1 only):**

$$
f_q^{(1)} = \text{Bilinear}(F_t^{(1)}, \text{Normalize}(q / s_1)) \quad \in \mathbb{R}^d
$$

A single feature vector at the finest scale, capturing the query point's precise spatial identity. This feeds into $h_{\text{point}}$ (the fusion query token).

**Enriched multi-scale center features (L2 for h_point, L3-L4 as B4 context tokens):**

$$
c_q^{(\ell)} = \text{Bilinear}(F_t^{(\ell)}, \text{Normalize}(q / s_\ell)), \quad \ell = 2, 3, 4
$$

Three enriched feature vectors capturing the query's neighborhood context at medium (L2, 128ch), broad (L3, 128ch), and scene-level (L4, 256ch → projected to 128ch) scales. These are fully enriched — each vector encodes multi-scale spatial context (52-400px radius from ASPP) plus cross-level information (from BiFPN) and deep nonlinear refinement (from ConvNeXt) from the preprocessing in A2. $c_q^{(2)}$ feeds into $h_{\text{point}}$ (giving it depth-aware context), while $c_q^{(3)}$ and $c_q^{(4)}$ become **separate context tokens in B4**, preserving their broader information for adaptive cross-attention fusion.

**v6 design rationale — tiered information flow:** In v5, all four center features were concatenated into a single $h_{\text{point}}$ MLP (448ch → 128ch), creating a 3.5:1 compression bottleneck where the MLP applied fixed mixing weights regardless of query context. In v6, $h_{\text{point}}$ combines L1 fine identity with L2 enriched depth context (center + local), giving the query token depth-awareness while preserving fine detail. L3-L4 broader context lives in separate B4 tokens. B4's cross-attention adaptively weights each source: at depth edges it can upweight $h_{\text{point}}$'s fine detail, in flat regions it can upweight L3-L4 context. The hierarchy is clean: $h_{\text{point}}$ = fine identity + medium context; B4 cross-attention = broad context + non-local evidence.

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

**Why max pooling:** PointNet (Qi et al., 2017) demonstrated that max pooling over point features significantly outperforms mean pooling (80.3 vs 73.2 on ShapeNet classification) because it selects the strongest activation per dimension, capturing the most discriminative local feature. Mean pooling dilutes sharp responses (e.g. a single strong depth edge within a smooth region). Max pooling acts as a channel-wise "most informative" selector.

**Positional encoding:**
$$
\text{pe}_q = [\sin(2\pi \sigma_l u/W); \cos(2\pi \sigma_l u/W); \sin(2\pi \sigma_l v/H); \cos(2\pi \sigma_l v/H)]_{l=0}^{L_{\text{pe}}-1}
$$

with $\sigma_l = 2^l$ and $L_{\text{pe}} = 8$, giving $\text{pe}_q \in \mathbb{R}^{32}$.

**Why Fourier instead of hash encoding:** The bilinear feature lookup $f_q^{(\ell)}$ already provides spatially-varying information. The positional encoding only needs to provide a smooth address fingerprint. Fourier features are parameter-free, well-understood, and sufficient. If ablations show hash encoding helps, it can be added.

**Enriched L2 local neighborhood sampling:**

Sample $N_{\text{loc}}^{(2)} = 16$ points around $q$ in $F_t^{(2)}$ coordinates (stride-4×):

Fixed grid (3×3 minus center):
$$
\Omega_{\text{fixed}}^{(2)} = \{(\delta_x, \delta_y) : \delta_x, \delta_y \in \{-1, 0, 1\}\} \setminus \{(0, 0)\}, \quad |\Omega_{\text{fixed}}^{(2)}| = 8
$$

Learned offsets (8 additional, predicted from enriched center feature):
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

**Why L2 enriched local sampling:** The L1 local aggregate ($l_q$) captures fine-grained edge/gradient information from raw 32ch features within a 5×5 grid (~6px reach). The L2 enriched local aggregate ($l_q^{(2)}$) adds depth-aware neighborhood context: each L2 feature already encodes 52-148px ASPP context, so even nearby L2 samples carry information about the query's medium-range depth structure. This gives $h_{\text{point}}$ access to depth-specific information that raw L1 features lack, without relying solely on the L2 center read ($c_q^{(2)}$). The 3×3 fixed grid at stride-4× covers ~32px original radius, complementing L1's ~6px reach.

**Why 16 samples (not 32):** L2 features are enriched — each sample already carries 52-148px context via ASPP, so fewer samples are needed to capture the neighborhood structure. The 3×3 grid (8 samples) provides systematic coverage, and 8 learned offsets add adaptive reach. Total: 16 + 32 (L1) = 48 local samples per query.

Parameters: offset heads ($128 \times 2 \times 8 + 16 = 2K$), per-sample MLP ($144 \times 128 + 128 \approx 18.6K$), aggregation ($128 \times 128 + 128 \approx 16.5K$). Total: ~37K.

**Center token (L1 identity + L2 enriched depth context):**
$$
h_{\text{point}} = \text{LN}(W_{p2} \cdot \text{GELU}(W_{p1} [f_q^{(1)}; c_q^{(2)}; \text{pe}_q; l_q; l_q^{(2)}] + b_{p1}) + b_{p2})
$$

A 2-layer MLP with GELU activation and **no skip connection**. Input dimension: $d + d + d_{\text{pe}} + d + d = 128 + 128 + 32 + 128 + 128 = 544$. Hidden dimension: $d = 128$. Output: $h_{\text{point}} \in \mathbb{R}^d$. Parameters: $544 \times 128 + 128 + 128 \times 128 + 128 \approx 86\text{K}$.

**v6 h_point design — L1 identity + L2 enriched depth context:** In v5, $h_{\text{point}}$ concatenated all 4 center features (544ch → 128ch). The initial v6 design used L1-only input (288ch), but this left h_point without any depth-specific information — L1 features are raw backbone outputs with no depth-aware enrichment. The final v6 design adds two L2-enriched components:

1. **$c_q^{(2)}$ (enriched L2 center):** A single bilinear read from the ASPP-enriched L2 map. Each L2 feature encodes 52-148px spatial context with depth-relevant structure learned through dilated convolutions. This gives h_point awareness of the query's medium-range depth context.
2. **$l_q^{(2)}$ (enriched L2 local aggregate):** 16 max-pooled samples from L2, capturing depth-aware neighborhood gradients. Complements the L1 local aggregate ($l_q$) which sees only raw features.

**Why not include L3 in h_point:** L2's ASPP context (52-148px) is sufficient for h_point's identity role — capturing the query's local depth structure and immediate surroundings. L3's broader context (104-296px) is better accessed through cross-attention in B4 (via $c_q^{(3)} + e_{\text{ms3}}$), where the model can adaptively weight it based on scene content. Including L3 would increase input to 672ch (5.25:1 compression), diluting all streams. The clean hierarchy is: h_point = fine identity + medium context; B4 cross-attention = broad context + non-local evidence.

**Why no skip connection here:** A skip connection would introduce a direct linear path from raw features to the depth head (through the residual stream of the fusion decoder). Since depth is a fundamentally nonlinear function of appearance features, we want $h_{\text{point}}$ to be a fully nonlinear encoding. This matches the pattern in both DepthAnythingV2 and F^3's own `EventPixelFF`, where prediction-facing layers are always fully nonlinear with no additive identity path.

**Routing token:**
$$
z_q = \text{LN}(W_z [f_q^{(1)}; l_q; \text{pe}_q] + b_z), \quad z_q \in \mathbb{R}^d
$$

This token sees the center feature plus local context. It drives global retrieval (B2) and deformable conditioning (B3). It is separate from $h_{\text{point}}$ because routing and depth prediction have different objectives.

#### B2. Global context retrieval

**Global summary via multi-head cross-attention:**
$$
\bar{c}_q = \text{MHCrossAttn}(Q = W_Q z_q, \; K = W_K C_t, \; V = W_V C_t)
$$

with 4 attention heads, $d_{\text{head}} = 32$. Output: $\bar{c}_q \in \mathbb{R}^d$.

This attends to ALL $P_c = 512$ tokens with soft weights. No information is lost. The summary captures the full global scene context relevant to this query. The $1 \times 512$ attention matrix per head is computationally trivial.

**Spatial routing (for deformable sampling anchors):**
$$
\alpha_q = \text{softmax}\left(\frac{W_r z_q \cdot (W_k^r C_t)^\top}{\sqrt{d}}\right), \quad
R_q = \text{TopR}(\alpha_q, R)
$$

Separate routing projection from the summary attention. This selects the $R = 32$ coarse tokens whose spatial regions are most relevant for precise evidence gathering. With $P_c = 512$, routing selects 32 out of 512 tokens (6.25%). The generous $R = 32$ default targets complex outdoor scenes (forest, city roads) where depth structure spans many distinct regions (10-20+ depth layers). Speed analysis shows the decoder is dominated by the F^3 precompute (~95% of total time), so anchor budget should be accuracy-driven. Using $P_c = 512$ instead of 1024 naturally promotes spatial diversity in anchor selection by reducing per-region candidate density.

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

**Why separate routing and summary:**
- The summary cross-attention optimizes for global information aggregation (what context to mix).
- The routing head optimizes for spatial anchor selection (where to look precisely).
- These are different objectives. A sky token may be important for global calibration (high summary weight) but useless as a deformable sampling anchor.

**Why the nearest anchor:**
- Guarantees that the query's own neighborhood is always represented in the deformable read.
- Prevents disconnection between center-point evidence and non-local evidence.
- This is the simplest connectivity guarantee — no complex landmark banks or coverage mechanisms needed.

**Straight-through routing during training:**

Forward pass: hard top-R selection (binary mask).
Backward pass: gradients flow through the softmax logits.

```
# PyTorch implementation sketch
probs = softmax(routing_logits)
top_mask = top_k_mask(probs, R)  # binary
selected = top_mask - probs.detach() + probs  # STE
```

At inference: hard top-R only.

#### B3. Deformable multiscale read

For each anchor $r \in S_q$, predict sampling offsets and importance weights, then read features from the pyramid.

**Conditioning:**
$$
\Delta\mathbf{p}_r = \mathbf{p}_r - q \quad \text{(query-to-anchor offset in original pixel coordinates)}
$$
$$
u_r = \text{LN}(W_u [z_q;\; c_r;\; \phi_{\text{B3}}(\Delta\mathbf{p}_r)] + b_u), \quad u_r \in \mathbb{R}^d
$$

where $c_r \in \mathbb{R}^d$ is the anchor's content vector from the latent bank $C_t$, $\mathbf{p}_r$ is the anchor's spatial position in pixel coordinates, and $\phi_{\text{B3}}$ is a normalized Fourier encoding (see below). Input dimension: $d + d + d_{\text{pe}} = 128 + 128 + 32 = 288$. This conditioning vector tells the offset head three things: "I'm this query ($z_q$), looking at an anchor **whose content is** $c_r$, **located at** this spatial offset." The conditioning and offset/weight heads are **shared across all 33 anchors** — each anchor produces a different $u_r$ because its inputs ($c_r$, $\Delta\mathbf{p}_r$) differ.

**B3 Fourier encoding (normalized, 8 frequencies):**
$$
\phi_{\text{B3}}(\Delta\mathbf{p}_r) = [\sin(2\pi \sigma_l \Delta u/W); \cos(2\pi \sigma_l \Delta u/W); \sin(2\pi \sigma_l \Delta v/H); \cos(2\pi \sigma_l \Delta v/H)]_{l=0}^{L_{\text{pe}}-1}
$$

with $\sigma_l = 2^l$, $L_{\text{pe}} = 8$, giving $\phi_{\text{B3}} \in \mathbb{R}^{32}$ — the same formula as $\text{pe}_q$ but applied to the normalized offset $[\Delta u/W, \Delta v/H] \in [-1, 1]$.

**Why a separate encoding from B1.2:** B1.2's $\phi(\delta)$ uses 4 frequencies on raw pixel offsets $|\delta| \leq 6$, which is well-matched to that small range. B3's offsets span the full image ($|\Delta\mathbf{p}_r|$ up to 1280 pixels). Applying B1.2's raw-pixel frequencies to such large offsets would produce meaningless high-frequency oscillations ($\sin(2\pi \cdot 8 \cdot 1280) = \sin(64{,}339)$). Normalizing by image dimensions maps the offset to $[-1, 1]$, matching the standard practice from NeRF (Mildenhall et al., 2020), Tancik et al. (2020), and DAB-DETR (Liu et al., 2022) — none of which apply Fourier/sinusoidal encoding to raw pixel coordinates. With 8 frequencies, the finest resolved period is $\sim 5$ pixels, comfortably finer than the ~40-pixel latent grid cell spacing in original coordinates.

**Why include anchor content $c_r$:** Without $c_r$, the offset MLP receives $z_q$ (shared across all 33 anchors) plus only a 32-dim spatial offset — insufficient for content-adaptive sampling. DCNv2 (Zhu et al., 2019) predicts offsets from the input feature map at the reference location (via convolution over local features). Similarly, we provide anchor content so the offset head can ask: "given what this region looks like ($c_r$), where should I sample?" This enables structure-aware offsets (e.g. sampling along edges detected in $c_r$) rather than purely geometry-driven offsets.

**Offset and weight prediction (shared across anchors):**

For each head $h \in [1, H]$, level $\ell \in [1, L]$, sample $m \in [1, M]$:
$$
\Delta p_{r,h,\ell,m} = W_{h,\ell,m}^\Delta \, u_r + b_{h,\ell,m}^\Delta
$$
$$
\beta_{r,h,\ell,m} = W_{h,\ell,m}^a \, u_r + b_{h,\ell,m}^a
$$

Offsets are **unbounded** (no tanh), following the dominant convention from Deformable DETR, Mask2Former, DCNv3/v4, and DAB-DETR. Training stability is ensured by **zero initialization** of offset parameters and **0.1× learning rate** on the offset head — offsets start at the anchor center and grow gradually as the model learns where to look. DAT (CVPR 2022) was the only method to use tanh bounding; its follow-up DAT++ (2023) showed the bound is dispensable at only -0.3% accuracy cost. With $P_c = 512$ (32×16 grid, ~40×45 pixel cells), unbounded offsets allow the model to freely sample within and across cell boundaries at all levels, which is critical for capturing depth discontinuities near cell edges.

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

where $W_V^{(h)} \in \mathbb{R}^{(d/H) \times C_\ell}$ is the value projection for head $h$ ($C_\ell$ is the channel dimension of level $\ell$: 128 for L2-L3, 256 for L4). In practice, a per-level $W_V^{(\ell)} \in \mathbb{R}^{d \times C_\ell}$ is used (with output reshaped to $H = 8$ heads of $d/H = 16$ dims). **Implementation optimization:** Following Deformable DETR, $W_V$ is pre-applied to each feature map $F_t^{(\ell)}$ once per frame during precompute, so grid_sample reads already-projected 128-ch features. This eliminates the per-sample projection cost from the per-query path.

**Mixed channel dimensions:** L2 and L3 are 128ch (after full enrichment), L4 is 256ch. The per-level $W_V$ pre-projection normalizes all levels to 128ch before storage, so grid_sample always reads 128ch features regardless of original level dimension. Pre-projection costs: L2 ($320 \times 180 \times 128 \times 128$) ~0.06 ms, L3 ($160 \times 90 \times 128 \times 128$) ~0.01 ms, L4 ($80 \times 45 \times 256 \times 128$) ~0.01 ms — total ~0.08 ms.

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

**Why full multi-head structure:** Per-head value projection partitions features into $H=8$ independent 16-dim subspaces, allowing different heads to learn complementary sampling patterns (e.g., one head for fine boundary evidence, another for broad structural context, others for texture gradients or occlusion boundaries). Per-head softmax (over $(\ell, m)$ instead of jointly over $(h, \ell, m)$) preserves each head's independent weight distribution — Deformable DETR, DCNv3, and all published variants use per-head or per-group normalization; joint softmax has zero published precedent and allows one dominant head to suppress all others. The concat + $W_O$ projection enables representations outside the convex hull of sampled features, which a simple weighted average cannot produce. Cost: $W_V$ (128×128) + $W_O$ (128×128) = ~33K params (shared across all 33 anchors).

**Default budget:** $H = 8$ heads, $L = 3$ levels (2–4), $M = 4$ samples.
- Per anchor: $8 \times 3 \times 4 = 96$ samples.
- 33 anchors: $33 \times 96 = 3{,}168$ deformable lookups total.

**Why $H = 8$ heads:** Every major deformable attention method (Deformable DETR, Mask2Former, DAB-DETR) uses 8 heads. $H = 8$ matches the published standard, providing maximum sampling diversity — 8 independent offset sets per anchor, each specializing in different spatial patterns (edges, textures, depth boundaries, occlusion). The parameter cost is unchanged ($W_V$ and $W_O$ remain 128×128).

**Why $L = 3$ levels (2–4, dropping level 1):** The deformable anchors gather **non-local** evidence from distant scene regions. Level 1 (stride 1×, raw 32ch) has no enrichment and provides no contextual advantage over the anchor's content vector $c_r$. Levels 2–4 are fully enriched at strides 4×, 8×, 16× (9-15 layers of processing each), providing progressively broader spatial reach with deeply context-aware, cross-level-informed features — each deformable sample carries rich depth-relevant information from the full enrichment pipeline. Fine-grained local detail at the query point is handled by B1's local sampling (32 samples from level 1) and the enriched center reads $c_q^{(2:4)}$ which enter B4 as separate context tokens.

**Why $M = 4$ samples per head per level:** $M = 4$ matches Deformable DETR's standard default — the most widely validated configuration across Deformable DETR, Mask2Former, DAB-DETR, and all major deformable attention implementations. Each head samples 4 locations per level × 3 levels = 12 samples per anchor — a sufficient local attention window for capturing depth gradients, boundaries, and occlusion edges within each anchor region. Speed analysis confirms the decoder cost is dominated by the F^3 precompute (~95% of total time), so the per-anchor sample budget is essentially free in wall-clock time. The total deformable budget (3,168) is spread across 33 diverse anchor regions.

**Why per-anchor aggregation instead of a single global softmax:**
- Each anchor represents a distinct spatial region. Per-anchor normalization preserves the relative contribution of each region.
- The fusion transformer (B4) then learns how to weigh different regions via cross-attention.
- A single global softmax would allow one dominant region to suppress all others.

#### B4. Fusion decoder (2-layer cross-attention transformer)

**Context token set (with type embeddings):**

$$
T_q = [l_q + e_{\text{loc}}; \; c_q^{(2)} + e_{\text{ms2}}; \; c_q^{(3)} + e_{\text{ms3}}; \; c_q^{(4)} + e_{\text{ms4}}; \; h_{r_1} + e_{\text{near}}; \; h_{r_2..33} + e_{\text{route}}; \; \bar{c}_q + e_{\text{glob}}]
$$

38 context tokens total, each with a learned type embedding:
- $l_q + e_{\text{loc}}$: aggregated local neighborhood (from B1).
- $c_q^{(2)} + e_{\text{ms2}}$: enriched medium-range context (fully-enriched L2 feature at query, 52-148px ASPP context + BiFPN cross-level info).
- $c_q^{(3)} + e_{\text{ms3}}$: enriched broad context (fully-enriched L3 feature at query, 104-296px ASPP context + BiFPN cross-level info).
- $c_q^{(4)} + e_{\text{ms4}}$: enriched scene-level context (fully-enriched L4 feature at query, 112-400px ASPP context + BiFPN + 2×ConvNeXt). L4 features are 256ch, projected to 128ch via $W_{\text{ms4}} \in \mathbb{R}^{d \times d_4}$ before adding the type embedding.
- $h_{r_1} + e_{\text{near}}$: deformable evidence from the nearest anchor (medium-range non-local context).
- $h_{r_2} + e_{\text{route}}, \ldots, h_{r_{33}} + e_{\text{route}}$: deformable evidence from 32 routed anchors (non-local context).
- $\bar{c}_q + e_{\text{glob}}$: compressed global scene summary (from B2).

**v6 design — enriched multi-scale center tokens:** The three enriched center features $c_q^{(2:4)}$ enter B4 as separate context tokens, giving cross-attention direct access to fully-enriched multi-scale context at the query point. Each feature has been processed through strided conv + ASPP + BiFPN + ConvNeXt (9-15 layers of depth-aware processing). Note that $c_q^{(2)}$ serves a **dual role**: it feeds into $h_{\text{point}}$'s MLP (giving the center token depth-aware context) AND appears as a separate B4 token (preserving the full uncompressed L2 feature for the cross-attention to access directly). This is analogous to DPT/ViTDet where features are used at multiple stages — the compressed MLP encoding and the raw token carry complementary information. $c_q^{(3:4)}$ are B4-only tokens, providing broader context that the cross-attention can adaptively weight. Each enriched center is a bilinear read from the fully-processed feature map — 3 additional grid_sample calls per query (negligible vs 3,168 in B3).

**Type embeddings:**

Each context token receives a learned type embedding identifying its role:

$e_{\text{loc}}, e_{\text{ms2}}, e_{\text{ms3}}, e_{\text{ms4}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}} \in \mathbb{R}^d$ — 7 learned embeddings (7 × 128 = 896 parameters). The three multi-scale embeddings ($e_{\text{ms2}}, e_{\text{ms3}}, e_{\text{ms4}}$) let the attention heads distinguish between context at different spatial scales. This is standard practice in DETR-family architectures where different token roles receive distinct learned embeddings.

**KV normalization:**

Before entering the transformer, apply a shared LayerNorm to the assembled context tokens:
$$
T_q \leftarrow \text{LN}_{\text{kv}}(T_q)
$$

This normalizes the heterogeneous token scales: $l_q$ (local aggregate), $c_q^{(2:4)}$ (ASPP-enriched bilinear reads), $h_{r_1..33}$ (multi-head deformable read output), and $\bar{c}_q$ (cross-attention output) come from different computational paths with potentially different magnitude distributions. Without normalization, attention logits ($Q \cdot K^\top$) can be dominated by whichever token type has the largest norm, creating systematic bias unrelated to content relevance. SAM's decoder applies separate LayerNorm to both queries and keys (via `norm2` and `norm3` in TwoWayAttentionBlock). Since $T_q$ is static (not updated across layers), this LN is applied once — essentially free.

**2-layer transformer decoder (standard Pre-LN):**

Each layer applies cross-attention with residual, then FFN with residual (standard Pre-LN transformer):
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}), \; KV = T_q)
$$
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}(\text{LN}(h_{\text{point}}))
$$

where $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ with expansion ratio 4 ($d \to 4d \to d$, i.e., $128 \to 512 \to 128$).

Cross-attention uses 4 heads with $d_{\text{head}} = 32$. Attention matrix per head is $1 \times 38$ — trivially cheap.

After 2 layers: $h_{\text{fuse}} = h_{\text{point}} \in \mathbb{R}^d$.

**Why standard residuals in both sub-layers:**

Every published transformer architecture — from the original Transformer (Vaswani et al., 2017) through DETR, SAM, Mask2Former, and DepthAnythingV2 — uses residual connections in **both** the attention and FFN sub-layers. There is zero published precedent for dropping the FFN residual while keeping the attention residual:

- **SAM's decoder** (our direct reference): the code confirms `queries = queries + mlp_out` — full residual in FFN.
- **Pre-LN Transformer** (Xiong et al., 2020): adds residual BEFORE LayerNorm in both sub-layers. This is the modern default.
- **ReZero, SkipInit, Fixup** all MODIFY residual magnitude (via learned scalar gates) but never REMOVE residuals entirely.

The FFN residual does create a linear path from $h_{\text{point}}$ to the depth head. But this is not harmful — the center MLP ($h_{\text{point}}$) already applies GELU nonlinearity, so $h_{\text{point}}$ is not a raw feature. The residual path carries a nonlinearly-transformed representation plus the nonlinear FFN correction. The depth head (a separate 2-layer MLP with GELU) provides the final nonlinear transform.

**Nonlinearity is still sufficient:** $h_{\text{point}}$ = GELU(MLP(raw features)), fusion adds cross-attention (softmax = nonlinear) + FFN (GELU), depth head = GELU(MLP). Multiple nonlinear stages exist regardless of residual connections. The concern about "linear shortcuts" applies only if the input to the residual path were raw features, which it is not.

**Why this works for center-near-global balance:**

The cross-attention is inherently data-dependent. At each layer:
- If the query is at a depth edge, the model can upweight $l_q$ (local detail), $c_q^{(2)}$ (medium context), and $h_{r_1}$ (nearest anchor).
- If the query is in a textureless region, the model can upweight $c_q^{(3:4)}$ (broad context), $\bar{c}_q$ (global summary), and $h_{r_2..33}$ (non-local evidence).
- The 7 type embeddings help the attention heads distinguish token roles and spatial scales without relying on content differences alone.
- The residual connections preserve $h_{\text{point}}$'s core identity (L1 fine detail + L2 enriched depth context) through both sub-layers, while cross-attention adds broader context (L3-L4) and non-local evidence.

This replaces the original plan's complex bounded-carry gates, explicit BCR/NUR monitoring, and balance losses. The transformer learns the right balance from data. If center collapse is observed, the center auxiliary loss $L_{\text{ctr}}$ addresses it directly.

**Why 2 layers:** SAM's decoder uses 2 layers and achieves excellent point-prompt results. Our context set is small (38 tokens vs thousands of image tokens in SAM), so 2 layers provide sufficient mixing depth. Ablate with 1 and 3 layers.

#### B5. Depth prediction

**Relative depth code:**
$$
r_q = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2}
$$

A 2-layer MLP ($d \to d \to 1$, i.e., $128 \to 128 \to 1$) with GELU activation. Output: scalar $r_q \in \mathbb{R}$. Parameters: $128 \times 128 + 128 + 128 \times 1 + 1 \approx 17\text{K}$.

**Center-only auxiliary code (training only):**
$$
r_q^{\text{ctr}} = W_{\text{ctr},2} \cdot \text{GELU}(W_{\text{ctr},1} \, h_{\text{point}}^{(0)} + b_{\text{ctr},1}) + b_{\text{ctr},2}
$$

A 2-layer MLP ($128 \to 64 \to 1$) applied to $h_{\text{point}}^{(0)}$, the center token BEFORE fusion (saved from B1). This forces the center branch to remain independently informative.

**Auxiliary calibration:** The auxiliary prediction is calibrated by the same $(s_t, b_t)$ as the main head before computing $L_{\text{ctr}}$:
$$
\rho_q^{\text{ctr}} = s_t \cdot r_q^{\text{ctr}} + b_t
$$
Without calibration, the auxiliary MLP must simultaneously learn depth estimation AND implicit scale/shift, producing noisy gradients. With shared $(s_t, b_t)$, the auxiliary focuses purely on depth structure, and the gradient from $L_{\text{ctr}}$ provides additional supervision for the calibration heads — matching the PSPNet/DeepLabV3 principle that auxiliary and main heads should predict in the same output space.

**Why a 2-layer MLP instead of a linear probe:** Linear probes are designed for frozen-feature **evaluation** (DINO, MAE) where the goal is to measure representation quality without modifying it. During **training**, the auxiliary head's gradients flow back into the center MLP — a linear probe constrains these gradients to rank-1 updates, limiting the representation learning signal. PSPNet and DeepLabV3 both use 2-layer FCNHead auxiliary heads for their training-time auxiliary losses. A shallow MLP provides just enough capacity to produce a meaningful training signal without overpowering the main depth head.

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
| `M1_Precompute` | A1–A4 | $E_t \to F_t^{(1)}(\text{raw}), F_t^{(2:4)}(\text{enriched: StridedConv+ASPP+BiFPN+CNX}), C_t, s_t, b_t, W_V\text{-projected maps}$ |
| `M2_QueryEncode` | B1+B2 | $q \to h_{\text{point}}, z_q, l_q, l_q^{(2)}, c_q^{(2:4)}, \bar{c}_q, S_q$ |
| `M3_DeformRead` | B3 | $S_q, z_q, C_t, F_t^{(2:4)} \to h_{r_1}, \ldots, h_{r_{33}}$ |
| `M4_FuseDecode` | B4+B5 | $h_{\text{point}}, T_q(38\text{ tokens}), s_t, b_t \to \hat{d}_q$ |

---

### 4.6 Why This Design Will Work — Confidence Arguments

1. **The precompute-then-query paradigm is proven.** SAM does exactly this for segmentation. Perceiver IO does it for arbitrary structured outputs. Both achieve state-of-the-art results with lightweight decoders on precomputed features.

2. **F^3 features contain sufficient depth information, now deeply enriched for depth.** F^3 already achieves dense depth comparable to DepthAnythingV2. The v6 enrichment pipeline (strided conv + ASPP + BiFPN + ConvNeXt) provides 9–15 sequential layers of depth-aware processing per pyramid level — approaching DPT's 16 layers. ASPP teaches features about textures, edges, and depth structure; BiFPN adds cross-level communication (fine→coarse and coarse→fine); ConvNeXt blocks add deep nonlinear refinement.

3. **Local + global context is standard in depth estimation.** MiDaS, DPT, ZoeDepth, and Metric3D all use multi-scale features capturing both local and global context. Our multi-stream design (L1 center + local + enriched L2-L4 context + deformable non-local + global) replicates this in a sparse decoder, with each stream entering B4 as a separate context token for adaptive fusion.

4. **Deformable attention is battle-tested.** Used in Deformable DETR, Mask2Former, DAB-DETR, and many more. Learned offsets + importance weights for spatial sampling is robust and well-understood.

5. **The speed advantage is structural and massive.** Dense decoders process $HW = 921{,}600$ pixels. Our precompute (~12.0 ms including full enrichment pipeline) accounts for ~88% of total time at $K=256$; the decoder adds only ~1.6 ms. The conservative crossover is at $K \approx 2{,}200$ — far beyond any practical query count. This means **sampling budget choices (R, M, L) are free in wall-clock time** and should be driven purely by accuracy.

6. **Non-linearity is enforced at every stage.** ASPP preprocessing: GELU + dilated convolutions (each enriched feature is deeply nonlinear). Center MLP: GELU without skip connection (L1 fine identity + L2 enriched context → fully nonlinear $h_{\text{point}}$). Local aggregates (L1 + L2): per-sample GELU + max pooling. Cross-attention: softmax (nonlinear). FFN: GELU expansion with standard residual. Depth head: 2-layer GELU MLP + softplus conversion. The enriched multi-scale center features ($c_q^{(3:4)}$) enter B4 as separate context tokens — each already a deeply nonlinear transform of backbone features through ASPP.

---

### 4.7 What Was Removed From v4 and Why

| Removed component | Original section | Reason |
|-------------------|-----------------|--------|
| Temporal memory $H_t$ (GRU) | A6 | F^3's 20ms window already encodes temporal structure. GRU adds sequential dependency and training complexity. Add as ablation if needed. |
| Ring pooling $l_q^{\text{ring}}$ | B1.1 | The 32-sample local aggregate with learned offsets already captures near-vs-far structure. Ring pooling adds complexity without clear benefit. |
| Local reliability score $m_q$ | B1.1 | The fusion transformer's attention weights implicitly learn when to trust local vs global evidence. Explicit reliability scoring is redundant. |
| Landmark coverage tokens $U_q$ | B4 | Content routing + nearest anchor provides sufficient spatial coverage. Coverage tokens add a second learned compression on top of $C_t$. Add if routing-only ablation shows far-field failures. |
| Global anchor tokens $G_t$ | A5 | The global summary $\bar{c}_q$ already attends to all $C_t$ tokens. Separate always-visible anchors are redundant. |
| Bounded carry gates $\lambda_{\min}, \lambda_{\max}$ | B8 | Transformer residual connections naturally preserve center identity. Explicit bounded carry constrains learning. |
| Balance loss $L_{\text{bal}}$ | 4.5A | Designed to prevent failure modes of the complex bounded-carry fusion. With simpler transformer fusion, not needed. |
| Alignment loss $L_{\text{align}}$ | 4.5A | Forces center/near/global branches to be similar — contradicts the goal of carrying different information. |
| Route stability loss $L_{\text{route-stab}}$ | 4.5A | STE with R=32 out of 512 tokens (6.25%) is within the range where STE works (MoE models route 6-8 out of 64-256 experts). If routing collapses, use entropy regularization (simpler). |
| Coverage loss $L_{\text{cov}}$ | 4.5A | Removed with coverage tokens. |
| Second-hop rerouting | B10 | Significant complexity for marginal gain on hard queries. Address hard queries via local sampling expansion in ablation. |
| Reciprocal write-back | B8 | Two-way coupling adds compute. The unidirectional center-reads-context pattern (as in SAM) is sufficient. |
| Hash-grid coordinate encoding | B2 | Fourier PE is parameter-free and sufficient when combined with spatially-varying features from $F_t^{(1)}$. |
| Event-recency map $R_t$ | A6 | Removed with hash encoding. The backbone features already encode temporal information. |
| Importance map $M_t$ | A6 | Was used for adaptive local expansion. Fixed 32-sample budget makes this unnecessary. |

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

Standard SiLog loss with sqrt (matching F^3, BTS, AdaBins, Depth Anything v2 convention). The sqrt provides L1-type gradient scaling — more stable than the raw quadratic form, especially near zero. Default $\lambda_{\text{var}} = 0.5$ (matching F^3 codebase default). Handles scale ambiguity; $\lambda_{\text{var}}$ controls mean-shift invariance (higher = more scale-invariant). Ablate $\lambda_{\text{var}} \in \{0.5, 0.85, 1.0\}$.

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

**Why only 3 losses:** F^3's depth training uses 2-3 terms. E2Depth and RAM-Net use 2 terms. Monodepth2 uses 2 terms. Adding more losses requires more hyperparameter tuning and risks optimization conflicts. Start simple; add terms only for specific documented failure modes.

**Optional additions (if needed, via ablation):**
- $L_{\text{rank}}$: pairwise ranking loss if ordinal consistency is empirically poor.
- $L_{\text{unc}}$: Gaussian NLL if uncertainty head is enabled.
- $L_{\text{entropy}}$: routing entropy regularizer if router collapse is observed.

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

**Why real ground truth is primary, not pseudo labels:**
- **We predict sparse points** — LiDAR sparsity is not a problem. We can sample query points $q$ at locations where LiDAR depth exists, giving exact metric supervision at every query.
- **LiDAR is ground truth** — it measures actual distance. Pseudo labels from DAv2 inherit DAv2's biases, scale ambiguity, and failure modes (reflective surfaces, thin structures, domain gap from event features).
- **F^3's own codebase** supports both: configs like `full_dsec_gt_train.yml` (real GT) and `dav2b_fullm3ed_pseudo_518x518x20.yml` (pseudo labels) exist side by side.

**Role of DAv2 pseudo labels (supplementary, not primary):**
- Provide dense supervision at locations where no LiDAR coverage exists (useful for query sampling diversity).
- Provide relative depth structure in regions between sparse LiDAR points.
- Best used with scale-invariant losses ($L_{\text{silog}}$) since pseudo labels lack metric accuracy.
- DAv2's training itself traces back to real GT: the DAv2 teacher was trained on synthetic data with pixel-perfect rendered depth (Hypersim, TartanAir, VKITTI2), then fine-tuned on real LiDAR data (KITTI, NYUv2). Pseudo labels are a distillation of real GT, not a substitute.

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

**Center collapse prevention** relies on $L_{\text{ctr}}$ (the auxiliary loss on the pre-fusion center token), following DETR's approach of using intermediate auxiliary losses rather than input masking. Full context masking (zeroing all $T_q$) has no published precedent in the transformer literature — DETR, SAM, Mask2Former, and DepthAnythingV2 all use standard dropout instead.

**Standard:** Weight decay 0.01, gradient clipping 1.0, mixed precision (bf16).

---

## 6. Runtime Analysis

### 6.1 Parameter Budget

**v6 separates parameters into two categories:** Phase A preprocessing (computed once per frame, amortized over all queries) and Phase B decoder (per-query cost). This reflects the v6 enrichment pipeline — strided conv projections + ASPP + BiFPN + ConvNeXt — a significant parameter investment that pays for itself through improved feature quality at every downstream stage.

**Phase A: Preprocessing parameters (amortized, once per frame)**

| Component | Parameters |
|-----------|------------|
| A2: L1 projection (32→128, 1×1 conv) | ~4K |
| A2: L2 StridedConv (32→128, k4s4) + ASPP$_{128}$ + 1×ConvNeXt$_{128}$ | ~271K |
| A2: L3 ProgConv (32→128, s8) + ASPP$_{256 \to 128}$ + 1×ConvNeXt$_{128}$ | ~576K |
| A2: L4 ProgConv (32→256, s16) + ASPP$_{256}$ + 2×ConvNeXt$_{256}$ | ~1,370K |
| A2: BiFPN (2 rounds, L2–L4, mixed channels) | ~272K |
| A3: Latent bank projection (256→128) | ~33K |
| A4: Calibration heads ($s_t$, $b_t$) | ~0.3K |
| **Phase A total** | **~2,526K** |

The A2 enrichment pipeline is the main parameter investment. L4 dominates (~1,370K) due to the progressive strided conv projection (32→64→128→256, ~197K) + full 256ch ASPP branches (~846K) + 2×ConvNeXt$_{256}$ (~260K) — but at 80×45 spatial resolution, this is computationally cheap (~0.62 ms). L3 uses wider internal ASPP branches (256ch) compressed to 128ch output (~477K), plus progressive conv projection (~66K) and 1×ConvNeXt$_{128}$ (~33K). BiFPN (2 rounds, ~272K) provides cross-level communication across L2–L4 with channel adapters for L4's 256ch. A3 adds a learned 256→128 projection (~33K) to map L4's 256ch features to the core dimension $d = 128$ for routing and cross-attention.

**Phase B: Decoder parameters (per-query cost)**

| Component | Lookups | Attn ops | Params | Detail |
|-----------|:---:|:---:|---:|--------|
| B1: Centers (L1 + L2–L4) | 4 | — | ~86K | MLP 544→128→128 |
| B1: L1 local sampling | 32 | — | ~37K | Local MLP + offsets |
| B1: L2 enriched local | 16 | — | ~37K | Local MLP + offsets |
| B1: Routing token | — | — | ~37K | Routing MLP |
| B2: Global summary | — | 512 | ~65K | Cross-attn |
| B2: Routing scores | — | 512 | ~33K | Routing proj |
| B3: Deformable read | 3,168 | — | ~157K | Cond + offsets + $W_V$ + $W_O$ |
| B4: $W_{\text{ms4}}$ proj | — | — | ~33K | 256→128 |
| B4: Type embeddings | — | — | ~0.9K | 7 × $d$ |
| B4: KV norm | — | — | ~0.3K | $\text{LN}_{\text{kv}}$ |
| B4: Fusion (2 layers) | — | 38×2 | ~400K | Transformer |
| B5: Depth head | — | — | ~17K | MLP 128→128→1 |
| B5: $L_{\text{ctr}}$ aux | — | — | ~8K | MLP 128→64→1 |
| **Phase B total** | **3,220** | **~1,100** | **~911K** | |

B3 parameter breakdown: conditioning MLP (288→128, ~37K) + offset head (128→192, ~25K) + weight head (128→96, ~12K) + per-level $W_V$ (L2/L3 128→128, L4 256→128, ~66K) + $W_O$ (128→128, ~17K).

**Total trainable parameters: ~3,437K (~3.4M)**

**v6 vs v5 changes:**
- **Phase A projections:** AvgPool + 1×1 proj (~16K) → strided/progressive conv (~328K). Learns within-cell spatial patterns instead of destructive averaging.
- **Phase A ASPP:** +1,680K from ASPP preprocessing (unchanged from initial v6).
- **Phase A BiFPN:** +272K (new). 2 rounds of bidirectional cross-level fusion with channel adapters for L4 256ch.
- **Phase A ConvNeXt:** L2 gains 1×ConvNeXt$_{128}$ (+33K, new). L4 gains 1 additional ConvNeXt$_{256}$ (+130K). Deeper enrichment to approach DPT's 16-layer depth.
- **Phase A latent bank:** Now pools from fully-enriched L4 (strided conv + ASPP + BiFPN + 2×CNX). 256→128 projection (+33K).
- **B1 center MLP:** 86K (same total, different composition: L1+L2 enriched vs v5's 4 raw centers).
- **B1 L2 local:** +37K (new). 16 enriched L2 samples with MLP + max-pool.
- **B3:** 148K → 157K (+9K). $M{=}4$ (was 3), lookups 2,376 → 3,168.
- **B3 $W_V$:** 16K → 66K (+50K). Per-level mixed-channel projection. Pre-applied in Phase A.
- **B4:** +$W_{\text{ms4}}$ (33K), type embeddings 4→7, context tokens 35→38.
- **Net Phase A:** ~50K → ~2,526K (+2,476K). Dominates total model size but fully amortized.
- **Net decoder:** 816K → 911K (+95K, ~12%). Per-query cost barely changes.

**Comparison:** DepthAnythingV2-Small has ~25M decoder parameters. Our per-query decoder is **27× smaller** (911K vs 25M). Including the full enrichment pipeline + latent bank, total trainable parameters are **7.3× smaller** (3.4M vs 25M). The 3,220 feature lookups per query remain 286× less than dense decoding (921,600 pixels). The enrichment pipeline investment improves feature quality at every downstream stage — center reads, deformable samples, latent bank tokens, and routing accuracy all benefit from deeply context-aware, cross-level-informed features. The effective sequential depth per level (9–15 layers) approaches DPT's 16 layers while remaining 7× more parameter-efficient.

### 6.2 Speed Estimate (RTX 4090, 1280×720)

**Precompute (once per window):**
$$
T_{\text{precompute}} = T_{\text{F}^3} + T_{\text{enrich}} + T_{\text{pool+cal}} + T_{W_V\text{-proj}} \approx 8.33 + 3.22 + 0.06 + 0.08 = 11.69 \text{ ms}
$$

The enrichment pipeline adds ~3.2 ms (see A2 cost table: strided conv ~0.10 ms, ASPP ~2.1 ms, BiFPN ~0.50 ms, ConvNeXt ~0.62 ms), representing 39% of backbone time. This is fully amortized over all queries. The 32×16 latent bank pool from fully-enriched $F_t^{(4)}$ (80×45, 7:1 compression) is negligible; the 256→128 projection over 512 tokens adds ~0.01 ms. Following the Deformable DETR implementation pattern, B3's per-level $W_V$ value projection is pre-applied to the feature maps $F_t^{(2:4)}$ once per frame (~0.08 ms for mixed-channel projection: L2/L3 128→128, L4 256→128), so grid_sample reads already-projected 128ch features during per-query decoding.

**Per-query cost analysis:**

The decoder cost has two components: a **fixed overhead** (kernel launches and PyTorch dispatch, independent of $K$) and a **marginal cost** per query (compute + memory bandwidth, scaling with $K$):
$$
T_{\text{decoder}}(K) = \alpha + \beta K
$$

Per-query FLOPs breakdown (~13.3M FLOPs/query with pre-projected $W_V$):

| Stage | Dominant operations | FLOPs |
|-------|-------------------|---:|
| B1: center + local + routing | MLP(544→128→128), L1/L2 local MLPs, 55 grid_samples | ~610K |
| B2: summary + routing | Cross-attn over 512 tokens, scoring | ~528K |
| B3: deformable | 33× cond MLP, offset/weight heads, $W_O$, 3,168 grid_samples | ~6,530K |
| B4: projection + fusion | $W_{\text{ms4}}$, KV proj on 38 tokens, FFN(128→512→128) | ~5,580K |
| B5: depth head | MLP(128→128→1) | ~33K |
| **Total** | | **~13.3M** |

v6 changes from initial design: B1 adds L2 enriched local sampling (16 samples, +315K FLOPs) and larger center MLP (544ch vs 288ch, +33K). B3 uses $M=4$ instead of $M=3$ (+600K from larger offset/weight heads and 33% more grid_samples). B4 unchanged. Net increase from initial v6: +1.5M FLOPs/query (~13%).

For reference, the F^3 backbone (2-stage ConvNeXtV2, 6 blocks at full 1280×720, 32 channels) processes ~65 GFLOPs per frame in 8.33 ms. Our decoder at 13.3M FLOPs/query is **~4,900× less compute** than one backbone pass.

**Three cost components:**

1. **Fixed overhead** ($\alpha$): ~30–40 sequential CUDA kernel launches × ~5–10 μs each = 0.20–0.35 ms per batch. Independent of $K$ — same number of kernel launches whether processing 1 or 1024 queries.
2. **Compute** ($\beta_{\text{compute}}$): Matrix multiplications (B3 conditioning, B4 projections/FFN). GPU utilization scales with $K$: ~2% at $K=1$, ~25% at $K=256$, ~40% at $K=1024$.
3. **Memory** ($\beta_{\text{mem}}$): grid_sample reads from pre-projected 128ch feature maps (bilinear, 4 neighbors × 128 channels per sample). Level 4 (0.9 MB at 80×45×128) and level 3 (3.7 MB at 160×90×128) fit in L2 cache (72 MB); level 2 (15 MB at 320×180×128 after $W_V$ projection) mostly fits. Per-query memory: ~1 KB per sample × 3,220 samples ≈ 3.2 MB. Each read is channel-contiguous (128 channels stored sequentially), so the "random" part is only the 2D spatial index — actual data transfer is coalesced.

**Conservative estimates** (standard PyTorch, no custom CUDA, no torch.compile):
- $\alpha \approx 0.30$ ms (fixed kernel launch + dispatch overhead).
- $\beta \approx 0.005$ ms/query (combined compute + memory bandwidth at typical utilization; slightly higher than initial v6 estimate due to L2 local sampling [+16 grid_samples] and $M=4$ deformable [+792 grid_samples]).

**Total:**
$$
T_{\text{EventSPD}}(K) \approx 12.0 + 0.005K \text{ ms}
$$

| Query count $K$ | Decoder (ms) | EventSPD total (ms) | Throughput (Hz) | Dense baseline (ms) | Speedup |
|-----|---:|---:|---:|---:|---:|
| 1 | 0.31 | 12.0 | 83 | ~23 | 1.9× |
| 64 | 0.62 | 12.3 | 81 | ~23 | 1.9× |
| 256 | 1.58 | 13.3 | 75 | ~23 | 1.7× |
| 1024 | 5.42 | 17.1 | 58 | ~23 | 1.3× |

Dense baseline: F^3 (8.3 ms) + DepthAnythingV2 decoder (~15 ms) = ~23 ms. The crossover point where EventSPD matches dense cost: $(23 - 12.0) / 0.005 \approx K = 2{,}200$ — far beyond any practical query count. Even at $K = 1024$, EventSPD is **1.3× faster** than dense.

**v6 speed impact:** The full enrichment pipeline (strided conv + ASPP + BiFPN + ConvNeXt) adds ~3.2 ms to precompute (vs ~2.1 ms for ASPP-only), shifting the total curve upward by a constant offset. The enriched h_point (L2 local sampling) and $M=4$ deformable increase $\beta$ from ~0.004 to ~0.005 ms/query. The speedup at $K = 256$ decreases from 2.3× (v5) to 1.7× (v6), and the crossover drops from ~3,500 to ~2,200. Both remain well within comfortable margins — the crossover at $K = 2{,}200$ is ~9× beyond any practical query count. The additional ~1.1 ms (BiFPN + extra ConvNeXt blocks + strided conv) buys cross-level communication and 3-6 additional sequential layers of depth, closing the gap with DPT from ~6 layers to ~9-15 layers.

**Why the decoder is still cheap:** The precompute (11.7 ms, dominated by F^3 backbone + enrichment pipeline) accounts for ~88% of total time at $K = 256$. The decoder adds only ~1.6 ms. This means **sampling budget decisions should be driven purely by accuracy, not speed** — even pushing to R=64, M=6 (~6,200 lookups) would add only ~1.5 ms beyond the current budget, keeping the total under 15 ms (1.5× faster than dense).

**Key insight:** The $\alpha + \beta K$ model reveals that the decoder's fixed overhead (~0.3 ms kernel launches) dominates for small $K$, while the marginal cost (~0.005 ms/query) is negligible because: (a) the 13.3M FLOPs/query is tiny for an RTX 4090 GPU, (b) all $K$ queries are batched into a single forward pass, (c) per-level $W_V$ is pre-applied to feature maps, eliminating the costliest per-sample operation, and (d) grid_sample memory reads are channel-contiguous (128 channels per burst), with smaller pyramid levels fitting in L2 cache.

**Design philosophy:** The decoder cost is dominated by the precompute phase, not the per-query sampling budget. The v6 enrichment pipeline increases precompute by ~3.2 ms but provides deeply context-aware, cross-level-informed features at every downstream read — a favorable trade when amortized over all queries. This validates the accuracy-first approach: R=32, M=4, H=8 are essentially free in wall-clock time. Once accuracy is validated, ablation targets should focus on **parameter efficiency and training speed** (fewer parameters = faster convergence), not inference speed.

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
- Average cross-attention weight per context token type ($l_q$, $c_q^{(2)}$, $c_q^{(3)}$, $c_q^{(4)}$, $h_{\text{near}}$, $h_{\text{route}}$, $\bar{c}_q$) across queries. Shows whether the model uses all 7 information streams. Particularly monitor the enriched center tokens ($c_q^{(2:4)}$) — if their attention weights are near-zero, the ASPP enrichment is not contributing to fusion and may need stronger features or different type embeddings.

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

### Tier 4 — Architecture audit ablations (v5/v6 changes)

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 19 | Projection method | StridedConv vs AvgPool? | (a) StridedConv/ProgConv (current). (b) AvgPool + 1×1 proj (v5). |
| 20 | ASPP dilation rates | Optimal rates? | (a) Current. (b) Smaller. (c) Larger. See note below. |
| 21 | L4 channel dim | 256ch vs 128ch L4? | (a) 256ch (~1,370K). (b) 128ch (~250K). |
| 22 | Center token paths | Separate L3-L4 from $h_{\text{point}}$? | (a) L1+L2 in $h_{\text{point}}$ (current). (b) All compressed (v5). (c) L1-only. |
| 23 | Offset bounding | Unbounded vs bounded? | (a) Unbounded+0.1×LR (current). (b) $\tanh$. (c) Clamp. |
| 24 | Multi-head deformable | Full $W_V$/$W_O$ vs simple? | (a) Full (157K). (b) Weighted sum (60K). |
| 25 | KV normalization | $\text{LN}_{\text{kv}}$ on context? | (a) With (current). (b) Without. |
| 26 | Auxiliary calibration | Calibrate $r_q^{\text{ctr}}$? | (a) Calibrated (current). (b) Uncalibrated. |
| 27 | L2 enriched local | L2 local in $h_{\text{point}}$? | (a) Full 544ch (current). (b) No $l_q^{(2)}$ (416ch). (c) L1-only (288ch). |
| 28 | Latent bank source | Which level for $C_t$? | (a) L4, 7:1 (current). (b) L3, 28:1. (c) L2, 112:1. |
| 29 | Enrichment pipeline | Which components contribute? | See configs below. |

**Ablation #20 dilation rate configs:** (a) L2 (6,12,18), L3 (6,12,18), L4 (3,6,12) — current. (b) Smaller: L2 (3,6,12), L3 (3,6,12), L4 (2,3,6). (c) Larger: L2 (12,18,24), L3 (12,18,24), L4 (6,12,18).

**Ablation #29 — Enrichment pipeline variants:**

This is the key ablation for justifying the v6 enrichment pipeline. Tests which components (strided conv, ASPP, BiFPN, ConvNeXt) contribute and in what combination:

| Config | Pipeline | Params | Precompute | Depth (L2/L3/L4) |
|:---:|----------|---:|---:|:---:|
| (a) | AvgPool + 1×1 proj only (v5 baseline) | ~50K | ~8.5 ms | 0/0/0 layers |
| (b) | StridedConv + ASPP only (no BiFPN, no CNX) | ~545K | ~10.5 ms | ~4/5/6 layers |
| (c) | StridedConv + ASPP + ConvNeXt (no BiFPN) | ~708K | ~11.1 ms | ~7/8/9 layers |
| (d) | StridedConv + ASPP + BiFPN (no extra CNX) | ~817K | ~11.0 ms | ~6/9/8 layers |
| (e) | **Full: StridedConv + ASPP + BiFPN + ConvNeXt (default)** | **~2,493K** | **~11.7 ms** | **~9-10/12-14/14-15 layers** |

Config (a) establishes the accuracy floor without enrichment. Configs (b)–(d) isolate each component's contribution: (b) tests ASPP alone, (c) adds depth via ConvNeXt, (d) adds cross-level flow via BiFPN. Config (e) is the full default. This progression reveals whether the ~+3.2 ms enrichment cost translates to measurable accuracy gains, and which components are most important.

---

## 9. Implementation Structure

### 9.1 New files

```
src/f3/tasks/depth_sparse/
├── models/
│   ├── eventspd.py            # Main model: Algorithm A + B pipeline
│   ├── enriched_pyramid.py     # A2: Full enrichment pipeline (StridedConv/ProgConv + ASPP + BiFPN + ConvNeXt)
│   ├── latent_bank.py         # A3: spatial grid pool from fully-enriched L4 (256→128 proj) + A4 calibration heads
│   ├── query_encoder.py       # B1: L1 center + L2 enriched local + enriched center reads + routing token
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
| 3 | Add enrichment pipeline (A2: StridedConv + ASPP + BiFPN + ConvNeXt) | Feature quality improvement, enriched center reads | 1-2 weeks |
| 4 | Add latent bank from fully-enriched L4 + cross-attention summary | Global context benefit | 1 week |
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
| Speed worse than dense | Very Low | ~13.3M FLOPs/query (~4,900× < backbone). Crossover at $K \approx 2{,}200$. |
| Enrichment latency too high | Low | ~3.2 ms amortized. Ablation #29 tests ASPP-only (2.1 ms) and minimal configs. |
| ASPP overfitting | Low-Med | DW-separable + 100K+ training windows. Reduce $C_{\text{branch}}$ if needed. |
| Enriched tokens ignored | Low | Monitor attention weights. Ablation #22. |
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
