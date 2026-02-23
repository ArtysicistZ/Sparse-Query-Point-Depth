# Research Plan: Streamlined EventSPD — Sparse Query-Point Depth from Events

Author: Claude (redesigned from Codex v4, audited v5, enriched pyramid v6, hybrid backbone v7, widened d=192 v8, attention-guided hybrid decoder v9)
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
   F^3 features and globally-enriched L4 tokens are computed once. Each query runs a deep but lightweight decoder.

2. **Two-phase per-query reasoning: global then local** (novel for sparse depth).
   Phase 1: cross-attention into all 880 L4 tokens (40×22 at stride 32) reveals scene layout and identifies important regions. Phase 2: deformable sampling gathers fine-grained multi-scale evidence at the attended locations. This mirrors human depth estimation — first understand the scene, then examine reference surfaces.

3. **Attention-guided deformable sampling** (combining Deformable DETR + cross-attention).
   L4 cross-attention weights naturally identify task-relevant spatial regions (free routing). Deformable sampling at those regions provides multi-scale evidence from L2-L4, enriching the query point's representation beyond what the shallower backbone provides. Dense methods process all pixels equally; our sparse approach concentrates budget on the most informative evidence for each query.

4. **Deep per-query processing to compensate for shallow backbone** (sparse advantage).
   DAv2 processes every pixel through 12 ViT layers (24 nonlinear sub-layers). Our backbone is shallower (~14 stages to L4), but our per-query decoder adds ~12 nonlinear sub-layers — affordable because we process only K points, not all 921,600 pixels. Total depth per query (~26 stages) matches DAv2.

5. **Minimal viable complexity**.
   Every component must justify its existence via ablation. Optional extensions are documented separately and disabled by default.

### 4.2 Symbol Table

| Symbol | Computed from | Meaning | Trainable |
|--------|--------------|---------|-----------|
| $E_t$ | Raw events in $[t-\Delta, t)$ | Input event stream | No |
| $\mathcal{F}_{\text{F}^3}^{\text{ds2}}$ | Backbone network (ds2 config) | Event-to-feature encoder (640×360×64) | Frozen / fine-tuned |
| $F_t$ | $\mathcal{F}_{\text{F}^3}^{\text{ds2}}(E_t)$ | Dense shared feature field (640×360, 64ch) | Via backbone |
| $F_t^{(1)}$ | $\text{LN}(F_t)$ + GELU | Fine features (= F^3 ds2 output, 640×360, 64ch) | LN only |
| $F_t^{(\text{int})}$ | Stem-1 conv from $F_t$ | Stride-4 intermediate features (320×180, 64ch) — not a pyramid level | No (conv: Yes) |
| $F_t^{(2..4)}$ | Hybrid Conv+Swin+SelfAttn backbone from $F_t$ | Multi-scale pyramid (128/192/384ch) at strides 8/16/32 | Yes |
| $s_\ell$ | $s_1{=}2, s_2{=}8, s_3{=}16, s_4{=}32$ | Stride of level $\ell$ | No |
| $G_t^{(4)}$ | L4 backbone output (includes full self-attn) | Globally-enriched L4 features (40×22, 384ch) | Yes |
| $K_t, V_t$ | $W_K G_t^{(4)}, W_V G_t^{(4)}$ | Pre-computed KV for per-query cross-attention (880 tokens) | Yes |
| $s_t, b_t$ | $\text{Heads}(\text{MeanPool}(G_t^{(4)}))$ | Global depth scale / shift | Yes |
| $q = (u, v)$ | User input | Query pixel coordinate | No |
| $f_q^{(1)}$ | $\text{Bilinear}(F_t^{(1)}, q)$ | Fine point feature (64ch) | No |
| $c_q^{(\ell)}$ | $\text{Bilinear}(F_t^{(\ell)}, q)$, $\ell{=}2,3,4$ | Multi-scale center (128/192/384ch) | No |
| $l_q$ | MaxPool(Local) from $F_t^{(1)}$ | L1 local context (32 samples, stride 2) | MLP: Yes |
| $l_q^{(\text{int})}$ | MaxPool(Local) from $F_t^{(\text{int})}$ | Stride-4 intermediate local (16 samples) | MLP: Yes |
| $\text{pe}_q$ | $\text{Fourier}(u/W, v/H)$ | Positional encoding (32d) | No |
| $h_{\text{point}}$ | MLP$([f_q^{(1)}; c_q^{(2)}; \text{pe}; l_q; l_q^{(\text{int})}])$ | Center token (L1 + L2 context, 608ch→192) | Yes |
| $h_{\text{point}}'$ | $\text{CrossAttn}^{2\times}(h_{\text{point}}, G_t^{(4)})$ | Globally-aware center token (after B2) | Yes |
| $\alpha_q$ | Attn weights from B2 layer 2 | Attention-based routing scores over 880 L4 positions | No |
| $R_q$ | $\text{TopR}(\bar{\alpha}_q)$, $R{=}32$ | Attention-routed anchor set | No |
| $h_{r}$ | DeformRead per $r \in R_q$ | Per-anchor multi-scale evidence | Yes |
| $T_q$ | See below | 36 context tokens | No |
| $e_{\text{loc..ms4,deform}}$ | Learned | 5 type embeddings ($\in \mathbb{R}^d$) | Yes |
| $h_{\text{fuse}}$ | TransformerDec$(h_{\text{point}}', T_q)$ | Fused query representation | Yes |
| $r_q$ | MLP$(h_{\text{fuse}})$ | Relative depth code | Yes |
| $\rho_q$ | $s_t \cdot r_q + b_t$ | Calibrated inverse depth | No |
| $\hat{d}_q$ | $1 / (\text{softplus}(\rho_q) + \varepsilon)$ | Final predicted depth | No |
| $\sigma_q$ | $\text{softplus}(\text{Linear}(h_{\text{fuse}}))$ | Uncertainty (optional) | Yes |

**$T_q$ expansion (36 context tokens with type embeddings):**

$$
T_q = [l_q{+}e_{\text{loc}};\; c_q^{(2)}{+}e_{\text{ms2}};\; c_q^{(3)}{+}e_{\text{ms3}};\; c_q^{(4)}{+}e_{\text{ms4}};\; h_{r_1..32}{+}e_{\text{deform}}]
$$

Core dimension: $d = 192$. Positional encoding: $d_{\text{pe}} = 32$ (8 Fourier frequencies × 2 trig functions × 2 spatial dims).

**v8→v9 key changes:** Removed $z_q$ (routing token), $C_t$ (latent bank), $\bar{c}_q$ (global summary). Added $G_t^{(4)}$ (self-attention-enriched L4), $K_t/V_t$ (pre-computed KV), attention-based routing from B2 weights. Global context now comes from direct cross-attention into all 880 L4 tokens (replacing 512-token compressed latent bank). Routing is free — derived from B2 attention weights.

### 4.3 Algorithm A: Precompute Once Per Event Window

**Input:** Event set $E_t$.
**Output:** $\text{cache}_t = \{F_t^{(1)}, F_t^{(\text{int})}, F_t^{(2)}, F_t^{(3)}, G_t^{(4)}, K_t, V_t, \hat{F}_t^{(2:4)}, s_t, b_t\}$ where $F_t^{(1)}$ is the F^3 ds2 output with LN+GELU (64ch, 640×360), $F_t^{(\text{int})}$ is the stride-4 intermediate (64ch, 320×180, for B1 local sampling), $F_t^{(2:4)}$ are produced by the wide pyramid backbone (128/192/384ch at strides 8/16/32), $G_t^{(4)}$ is L4 after full self-attention (384ch, 40×22, 880 tokens), $K_t/V_t$ are pre-computed KV projections for B2 cross-attention, and $\hat{F}_t^{(2:4)}$ are $W_V$-projected feature maps for B3 deformable sampling.

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

#### A2. Multi-scale feature pyramid (Wide Pyramid: Conv + Swin + SelfAttn backbone)

**Motivation (v9 "wide pyramid"):** SOTA dense prediction backbones (Swin-T, ConvNeXt-T, InternImage-T) universally use channel ratios of 1:2:4:8, doubling width at each downsample to preserve information capacity. Our previous pyramid (strides 2/4/8/16, channels 64/64/96/192, ratio 1:1:1.5:3) was anomalously narrow — L2 at 64ch had only 33% rank when projected to $d{=}192$ for deformable reads, and L4 at 192ch was narrower than even the smallest SOTA coarsest levels (Swin-T: 768, InternImage-T: 512). The wide pyramid architecture pushes to strides 2/8/16/32 with channels 64/128/192/384 (ratio 1:2:3:6), much closer to SOTA conventions. The stride-4 spatial level is sacrificed as a pyramid level but retained as an intermediate for B1 local sampling.

**No FPN/BiFPN (justified):** Deformable DETR's ablation (Zhu et al., Table 2) shows that multi-scale deformable attention achieves identical AP with and without FPN (43.8 AP in both cases). The paper states: *"Because the cross-level feature exchange is already adopted, adding FPNs will not improve the performance."* Our B3 deformable read + B4 cross-attention provide the same cross-level exchange, making a top-down feature path unnecessary.

**Level 1** — fine-grained features (identity from F^3 ds2, with nonlinearity):
$$
F_t^{(1)} = \text{GELU}(\text{LN}(F_t)) \quad \in \mathbb{R}^{640 \times 360 \times 64}
$$

LayerNorm + GELU applied directly to the F^3 ds2 output. No convolution needed — F^3 ds2 already outputs at 640×360×64, matching the target L1 resolution and channel width. The GELU ensures L1 features are nonlinear, enabling representation of feature conjunctions and edge detectors. L1 is the most heavily sampled level (32 local + center) and receives no further block-level processing — the per-query MLPs in B1 add one more nonlinear layer, giving the L1→decoder path at least 2 nonlinear operations.

**Why L1 = F^3 ds2 output:** With F^3 ds2, the backbone already processes 6 ConvNeXt blocks to produce 640×360×64 features. These are *richer* than the previous L1 (a single stride-2 conv on 32ch ds1 features). No additional conv is needed — just normalization and activation. Parameters: ~0.1K (LN only, GELU is parameter-free).

**Levels 2–4** — wide hierarchical backbone with ConvNeXt at L2, Swin at L3, full self-attention at L4:

```
F_t (640×360, 64ch)                                         ← F^3 ds2 output
├→ L1: LN + GELU                                            [640×360×64]   stride 2
└→ Stem-1: Conv(64→64, k3,s2) + LN                          [320×180×64]   stride 4 (INTERMEDIATE)
   → Stem-2: Conv(64→128, k2,s2) + LN                       [160×90×128]   stride 8
   → 2× ConvNeXt_128 (with GRN)                             → L2  [160×90×128]
   → Down: Conv(128→192, k2,s2) + LN                        [80×45×192]
   → 4× SwinBlock_192 (window=8, shifted)                   → L3  [80×45×192]
   → Down: Conv(192→384, k2,s2) + LN                        [40×22×384]
   → 2× FullSelfAttn_384 (6 heads, d_head=64, 880 tokens)  → L4 = G_t^{(4)}  [40×22×384]
```

Four pyramid scales at strides 2×, 8×, 16×, 32× relative to the original HD image (1280×720), plus a stride-4× intermediate. $F_t^{(1)}$ is 640×360 (= F^3 ds2 output), $F_t^{(\text{int})}$ is 320×180 (used for B1 local sampling only), $F_t^{(2)}$ is 160×90, $F_t^{(3)}$ is 80×45, $F_t^{(4)} = G_t^{(4)}$ is 40×22.

**Two-step stem (stride 2→4→8):**
$$
F_t^{(\text{int})} = \text{LN}(\text{Conv2d}(64, 64, k{=}3, s{=}2, p{=}1)(F_t)) \quad \in \mathbb{R}^{320 \times 180 \times 64}
$$
$$
\text{Stem-2}(F_t^{(\text{int})}) = \text{LN}(\text{Conv2d}(64, 128, k{=}2, s{=}2)(F_t^{(\text{int})})) \quad \in \mathbb{R}^{160 \times 90 \times 128}
$$

Two 2× downsampling steps with channel widening at the second step (64→128). The stride-4 intermediate $F_t^{(\text{int})}$ (320×180×64) is **retained as a side feature** for B1's local neighborhood sampling — zero additional cost since it's a natural byproduct of the stem. Parameters: Stem-1 $64 \times 64 \times 3 \times 3 + 64 \approx 37\text{K}$, Stem-2 $64 \times 128 \times 2 \times 2 + 128 \approx 33\text{K}$.

**ConvNeXt blocks (L2) — with GRN:**

Each ConvNeXt block follows the ConvNeXt V2 design (Woo et al., CVPR 2023), with GRN after GELU:
```
Input → DW Conv 7×7 → LN → PW Conv 1×1 (C→4C) → GELU → GRN(4C) → PW Conv 1×1 (4C→C) → + Input
```

GRN (Global Response Normalization) computes per-channel $\ell_2$ norms globally, then normalizes and rescales: $\text{GRN}(x_i) = \gamma_i \cdot x_i \cdot \|X\|_2 / (\|x_i\|_2 + \epsilon) + \beta_i + x_i$. This adds nonlinear inter-channel competition, improving feature diversity. Parameters per block: $2 \times 4C$ (learnable $\gamma, \beta$).

- L2: 2× ConvNeXt$_{128}$ blocks. Each block: DW 7×7 (128×7×7 = 6.3K) + expansion 128→512→128 (65.5K + 65.5K) + GRN(512: 1.0K) + LN ≈ 138.5K/block. Total: ~277K.

ConvNeXt blocks build local context through 7×7 depthwise convolutions. At L2 (160×90), 2 blocks provide ~14px receptive field = ~112px original-coordinate context (stride 8 × 14px). 128ch at L2 gives the downstream $W_V$ projection 67% rank in $d{=}192$ space (vs previous 33% at 64ch) — a key improvement for B3 deformable read quality.

**Downsampling layers:**
$$
\text{Down}_{L2 \to L3} = \text{LN}(\text{Conv2d}(128, 192, k{=}2, s{=}2)) \quad \text{(~98.5K params)}
$$
$$
\text{Down}_{L3 \to L4} = \text{LN}(\text{Conv2d}(192, 384, k{=}2, s{=}2)) \quad \text{(~295K params)}
$$

Strided convolutions with LayerNorm and channel widening.

**Swin Transformer blocks at L3 (medium-to-global context):**

At 80×45 = 3,600 positions, self-attention within 8×8 windows (64 tokens per window, ~70 windows) provides rich local-to-medium context. After 4 blocks with alternating shifted windows, information propagates ~16 positions (~20% of the 80-wide map).

Each Swin block:
```
Input → LN → W-MSA (window_size=8) → + Input → LN → FFN (192→768→192) → + Input
```

where W-MSA = Window Multi-Head Self-Attention with 6 heads ($d_{\text{head}} = 32$). Blocks alternate between regular and shifted windows (shift = window_size/2 = 4), following Swin Transformer v1.

Per-block parameters:
- QKV projection: $3 \times 192 \times 192 = 110.6\text{K}$
- Output projection: $192 \times 192 = 36.9\text{K}$
- FFN: $192 \times 768 + 768 \times 192 = 295\text{K}$
- Relative position bias: $(2 \times 8 - 1)^2 \times 6 = 1.4\text{K}$
- LayerNorms: $2 \times 2 \times 192 = 0.8\text{K}$
- Per-block total: ~444K
- 4 blocks: ~1,778K

**Why Swin at L3 (80×45), ConvNeXt at L2 (160×90):**
- At 80×45, shifted window attention is affordable (~56M FLOPs/layer) and builds toward global receptive field through shifted windows. CNN blocks cannot achieve this coverage with local 7×7 kernels.
- At 160×90 (L2), self-attention is still expensive. ConvNeXt captures local context (edges, textures) efficiently with 7×7 depthwise convolutions.

**Full self-attention at L4 (true global context — replaces separate A3 stage):**

At 40×22 = 880 positions, full self-attention is trivially cheap. No windowing needed. Two layers of full self-attention give every L4 token **direct access to all 880 tokens**, providing true global scene understanding (e.g., relating foreground objects to a distant ground plane).

Each self-attention layer:
```
Input → LN → MultiHeadSelfAttn (6 heads, d_head=64) → + Input → LN → FFN (384→1536→384) → + Input
```

Note $d_{\text{head}} = 64$ at L4 (384ch / 6 heads), vs $d_{\text{head}} = 32$ at L3 (192ch / 6 heads). The wider head dimension gives each head more expressive capacity for global reasoning.

Per-layer parameters:
- QKV: $3 \times 384 \times 384 = 442\text{K}$
- Output: $384 \times 384 = 148\text{K}$
- FFN: $384 \times 1536 + 1536 \times 384 = 1{,}180\text{K}$
- LayerNorms + bias: ~2K
- Per-layer total: ~1,772K
- 2 layers: **~3,542K**

**Cost:** ~6.2G FLOPs (2 layers). 880 tokens at 384ch is well within efficient self-attention range. The 880×880 attention matrix (0.77M entries per head) is tiny. With flash attention, estimated wall-clock: **~0.08 ms**.

**Why L4 self-attention is not a separate A3 stage:** In the previous architecture, the L4 Swin blocks (80×45×192) needed a separate self-attention stage (A3) because Swin's 8×8 windows only propagated information ~16 positions. With the wide pyramid, L4 sits at 40×22 — small enough for full self-attention directly in the backbone. The Swin blocks at L3 (80×45) provide local-to-medium context, and L4's full self-attention provides global context. The architecture is cleaner: L3 Swin = medium range, L4 self-attn = global.

**Backbone cost (amortized over all queries):**

| Component | Resolution | Channels | FLOPs | Time (est.) |
|-----------|:---:|:---:|---:|---:|
| L1: LN + GELU | 640×360 | 64 (identity) | ~0.01G | ~0.01 ms |
| Stem-1: Conv k3s2 + LN | 320×180 | 64→64 | ~0.43G | ~0.06 ms |
| Stem-2: Conv k2s2 + LN | 160×90 | 64→128 | ~0.12G | ~0.02 ms |
| L2: 2× ConvNeXt$_{128}$ (GRN) | 160×90 | 128 | ~5.8G | ~0.50 ms |
| Down L2→L3 | 80×45 | 128→192 | ~0.09G | ~0.02 ms |
| L3: 4× SwinBlock$_{192}$ | 80×45 | 192 | ~3.3G | ~0.50 ms |
| Down L3→L4 | 40×22 | 192→384 | ~0.13G | ~0.02 ms |
| L4: 2× FullSelfAttn$_{384}$ | 40×22 | 384 | ~6.2G | ~0.08 ms |
| **Backbone total** | | | **~16.1G** | **~1.2 ms** |

Total: ~5,963K params, ~1.2 ms. The L2 ConvNeXt is cheaper than before (160×90 vs 320×180, 4× fewer positions, despite 128 vs 64 channels — net ~6% less FLOPs). L4 self-attention at 880 tokens is very fast (~0.08 ms) despite being 384ch wide. Overall backbone is **faster** than the previous architecture (~1.2 ms vs ~2.0 ms + ~0.3 ms A3 = ~2.3 ms) because the spatial reduction at L2 (4× fewer positions) dominates the cost.

**Wide pyramid vs SOTA channel widths:**

| | Swin-T | InternImage-T | SegFormer-B3 | DAv2-S DPT | **EventSPD** |
|---|:---:|:---:|:---:|:---:|:---:|
| Strides | 4/8/16/32 | 4/8/16/32 | 4/8/16/32 | ~3.5/7/14/28 | **2/8/16/32** |
| Channels | 96/192/384/768 | 64/128/256/512 | 64/128/320/512 | 48/96/192/384 | **64/128/192/384** |
| Ratio | 1:2:4:8 | 1:2:4:8 | 1:2:5:8 | 1:2:4:8 | **1:2:3:6** |
| Finest res | 1/4 | 1/4 | 1/4 | ~1/3.5 | **1/2** |

Our channel profile closely matches DAv2-S's DPT decoder (48/96/192/384 → 64/128/192/384). With L1 at stride 2 (finer than any SOTA), our pyramid offers both wider channels and higher spatial resolution at the fine end.

#### A3. Pre-compute KV and W_V projections

**Pre-compute KV for per-query cross-attention (B2):** B2 cross-attends into L4's 880 tokens. The learned $W_K \in \mathbb{R}^{d \times 384}$ and $W_V \in \mathbb{R}^{d \times 384}$ projections are pre-applied to $G_t^{(4)}$ once per frame:
$$
K_t = G_t^{(4)} W_K^T, \quad V_t = G_t^{(4)} W_V^T \quad \in \mathbb{R}^{880 \times d}
$$

This eliminates the KV projection cost from the per-query path. Both $K_t$ and $V_t$ are stored as part of the cache (~1.3 MB total at $d{=}192$, fits L2 easily). Cost: ~0.01 ms. Parameters: $2 \times 384 \times 192 = 148\text{K}$.

Each L4 position $i$ has a known spatial coordinate $\mathbf{p}_i$ in original pixel space (deterministic from the 40×22 grid geometry). We write $\mathbf{p}_i^{(\ell)} = \mathbf{p}_i / s_\ell$ for the position in level-$\ell$ native coordinates. These positions are used for attention-based routing in B2 and as deformable anchor locations in B3.

**Pre-apply per-level $W_V$ for B3 deformable reads:** Following the Deformable DETR implementation pattern, B3's value projections are pre-applied to feature maps $\hat{F}_t^{(2:4)}$ once per frame. Each level's features are projected from their native channel width to $d{=}192$: L2 ($128 \to 192$), L3 ($192 \to 192$, square), L4 ($384 \to 192$). This means grid_sample reads already-projected 192ch features during per-query decoding. Parameters: ~136K. Cost: ~0.05 ms.

#### A4. Global calibration heads

$$
s_t = \text{softplus}(h_s(\text{MeanPool}(G_t^{(4)}))), \quad b_t = h_b(\text{MeanPool}(G_t^{(4)}))
$$

Window-level scale and shift for depth calibration. Mean-pooling over globally-enriched L4 tokens (880 tokens at 384ch, after full self-attention) provides a rich scene summary. $h_s, h_b: \mathbb{R}^{384} \to \mathbb{R}^1$ are single linear layers (~0.8K params total). All queries in the same window share the same $(s_t, b_t)$, ensuring global consistency.

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

Three feature vectors capturing the query's context at medium (L2, 128ch), broad (L3, 192ch), and scene-level (L4, 384ch) scales. Each level's features carry progressively deeper context from the wide hierarchical backbone. $c_q^{(2)}$ feeds into $h_{\text{point}}$ (giving it medium-range context from 128ch GRN-enhanced ConvNeXt blocks), while $c_q^{(3)}$ and $c_q^{(4)}$ become **separate context tokens in B4** after projection to $d = 192$. With the wide pyramid, $c_q^{(3)}$ is already at $d{=}192$ (no projection needed), and $c_q^{(4)}$ at 384ch provides an over-specified source that is projected down to $d$.

**Tiered information flow (v9 two-phase):** $h_{\text{point}}$ = fine identity (L1) + medium context (L2) via MLP; B2 cross-attention into L4 = true global context + attention-based routing; B3 deformable = multi-scale evidence at attended locations; B4 fusion = merge all streams. Phase 1 (B2): *understand the scene globally*. Phase 2 (B3-B4): *gather and fuse detailed evidence*.

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

**Stride-4 intermediate local neighborhood sampling:**

Sample $N_{\text{loc}}^{(\text{int})} = 16$ points around $q$ in $F_t^{(\text{int})}$ coordinates (stride-4×, 320×180×64):

Fixed grid (3×3 minus center):
$$
\Omega_{\text{fixed}}^{(\text{int})} = \{(\delta_x, \delta_y) : \delta_x, \delta_y \in \{-1, 0, 1\}\} \setminus \{(0, 0)\}, \quad |\Omega_{\text{fixed}}^{(\text{int})}| = 8
$$

Learned offsets (8 additional, predicted from intermediate center feature):
$$
\Delta_m^{(\text{int})} = r_{\max}^{(\text{int})} \cdot \tanh(W_{\text{off}}^{(\text{int},m)} f_q^{(\text{int})} + b_{\text{off}}^{(\text{int},m)}), \quad m = 1, \ldots, 8
$$

where $f_q^{(\text{int})} = \text{Bilinear}(F_t^{(\text{int})}, q/4) \in \mathbb{R}^{64}$ is the center feature at the intermediate level, and $r_{\max}^{(\text{int})} = 4$ (maximum reach in stride-4 coordinates = 16 original pixels).

For each offset $\delta \in \Omega_{\text{fixed}}^{(\text{int})} \cup \{\Delta_m^{(\text{int})}\}$:
$$
f_\delta^{(\text{int})} = \text{Bilinear}(F_t^{(\text{int})}, \tilde{q}^{(\text{int})} + \delta), \quad
h_\delta^{(\text{int})} = \text{GELU}(W_{\text{loc}}^{(\text{int})} [f_\delta^{(\text{int})}; \phi^{(\text{int})}(\delta)] + b_{\text{loc}}^{(\text{int})})
$$

where $\tilde{q}^{(\text{int})} = q / 4$ and $\phi^{(\text{int})}(\delta)$ is a small Fourier encoding of the offset (4 frequencies × 2 trig × 2 dims = 16 dims).

Aggregate via max pooling:
$$
l_q^{(\text{int})} = W_{\text{agg}}^{(\text{int})} \cdot \text{MaxPool}(\{h_\delta^{(\text{int})}\}_\delta) + b_{\text{agg}}^{(\text{int})}, \quad l_q^{(\text{int})} \in \mathbb{R}^d
$$

**Why stride-4 intermediate local sampling:** The stride-4 intermediate (320×180×64) is a natural byproduct of the two-step stem downsampling — zero backbone cost. Sampling from it adds medium-range context (~16px reach in stride-4 coords = ~64px original) beyond L1's fine-grained ~12px reach. The 64ch features have had one conv processing step from F^3 output. 16 samples suffice because the wider L2 ConvNeXt features (stride 8, 128ch) are available via B3 deformable reads. Total: 16 + 32 (L1) = 48 local samples per query.

Parameters: offset heads ($64 \times 2 \times 8 + 16 = 1.0K$), per-sample MLP ($80 \times 192 + 192 \approx 15.6K$), aggregation ($192 \times 192 + 192 \approx 37.1K$). Total: ~53.7K.

**Center token (L1 identity + L2 context):**
$$
h_{\text{point}} = \text{LN}(W_{p2} \cdot \text{GELU}(W_{p1} [f_q^{(1)}; c_q^{(2)}; \text{pe}_q; l_q; l_q^{(\text{int})}] + b_{p1}) + b_{p2})
$$

A 2-layer MLP with GELU activation and **no skip connection**. Input dimension: $64 + 128 + 32 + 192 + 192 = 608$. Hidden dimension: $d = 192$. Output: $h_{\text{point}} \in \mathbb{R}^d$. Parameters: $608 \times 192 + 192 + 192 \times 192 + 192 \approx 154\text{K}$.

**h_point design:** $f_q^{(1)}$ (64ch, precise identity from GELU-activated L1), $c_q^{(2)}$ (128ch, ~112px GRN-enhanced ConvNeXt context from stride-8 L2), $l_q$ and $l_q^{(\text{int})}$ (192ch each, max-pooled local gradients from L1 and stride-4 intermediate). L3-L4 broader context enters via B4 cross-attention instead — adaptive weighting beats fixed MLP mixing.

**No skip connection:** Depth is a nonlinear function of appearance features — $h_{\text{point}}$ should be fully nonlinear. Matches DAv2 and F^3's `EventPixelFF` patterns.

**No separate routing token (v9 simplification):** v8 used a separate $z_q$ token to drive routing and global retrieval. In v9, $h_{\text{point}}$ itself drives B2 cross-attention — the attention weights naturally perform routing (selecting relevant L4 regions), and the cross-attention output updates $h_{\text{point}}$ with global context. This eliminates ~56K parameters and one redundant MLP, while producing *better-informed* routing: B2's routing is conditioned on the full $h_{\text{point}}$ (608ch → 192 → 192, all local evidence), not just a subset.

#### B2. Global cross-attention into L4 (v9: replaces latent bank retrieval + routing)

**2-layer cross-attention into all 880 globally-enriched L4 tokens:**

Each layer applies standard Pre-LN cross-attention + FFN with residuals:
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}), \; K = K_t, \; V = V_t)
$$
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}(\text{LN}(h_{\text{point}}))
$$

with 6 attention heads, $d_{\text{head}} = 32$, $\text{FFN}: 192 \to 768 \to 192$. The $1 \times 880$ attention matrix per head is computationally trivial — **4× cheaper than the previous 3,600-token design.** $K_t$ and $V_t$ are **pre-computed** during Phase A (projecting L4's 384ch → $d{=}192$, shared across all $K$ queries), so per-query cost is only the Q projection, attention computation, and FFN.

After 2 layers: $h_{\text{point}}' \in \mathbb{R}^d$ — the center token enriched with true global scene context.

**Why 2 layers:** Layer 1 gathers initial global context. Layer 2 refines — the query's attention pattern can adapt based on what it learned in layer 1. Total nonlinear depth added: 4 sub-layers (2× cross-attn + 2× FFN). SAM's decoder uses 2 cross-attention layers with similar motivation.

**Attention-based routing (free — no extra parameters):**

Extract attention weights from the **last cross-attention layer** (layer 2), average across heads to get per-position relevance:
$$
\bar{\alpha}_q = \frac{1}{H} \sum_{h=1}^{H} \alpha_{q,h} \quad \in \mathbb{R}^{880}
$$

where $\alpha_{q,h}$ is the softmax attention weight vector for head $h$. Select the top-$R$ positions as deformable anchors:
$$
R_q = \text{TopR}(\bar{\alpha}_q, R), \quad |R_q| = R = 32
$$

Each selected position $r \in R_q$ maps to a known pixel coordinate $\mathbf{p}_r$ via the L4 grid geometry (40×22, stride 32). These become the spatial anchors for B3's multi-scale deformable read.

**Routing budget analysis:** $R = 32$ from 880 positions = **3.6% selection ratio**. Each L4 position covers 32×32 pixels in the original image, so 32 anchors cover ~32,768 px² (~3.6% of the 1280×720 image). Deformable offsets from each anchor refine within its region across L2-L3-L4 pyramid levels. The 3.6% selection ratio is well-calibrated — selective enough to focus on informative regions, broad enough to cover the relevant scene structure.

**Straight-through routing:** Hard top-R forward, gradients via STE backward (`selected = top_mask - probs.detach() + probs`), following v8's approach.

**Why this replaces v8's routing:** v8 used a separate routing token $z_q$ (built from local features only) to select 32+1 anchors from 512 compressed latent bank tokens. v9 routing has three advantages:
1. **Better-informed:** routing is conditioned on $h_{\text{point}}'$, which already carries global context from the L4 cross-attention. v8's $z_q$ had only local features.
2. **Good selection ratio:** R=32 from 880 positions (3.6%) vs v8's 32 from 512 (6.25%). Slightly more selective but each position represents one L4 cell directly (no 7:1 compression).
3. **Zero extra parameters:** falls out of the attention weights that are already computed for global context retrieval. No separate $W_r$, $W_k^r$ projection heads.

**Per-query cost:** ~1.0M FLOPs for 2-layer cross-attention (dominated by $Q \times K_t^T$ and $\text{attn} \times V_t$ operations, each $1 \times 880 \times 32$ per head). These are **batched GEMM operations** across $K$ queries — highly efficient on GPU.

#### B3. Deformable multiscale read (v9: attention-guided anchors, globally-conditioned)

For each anchor $r \in R_q$ (selected by B2's attention weights), predict sampling offsets and importance weights, then read features from the multi-scale pyramid.

**Conditioning (v9: h_point' carries global context):**
$$
\Delta\mathbf{p}_r = \mathbf{p}_r - q \quad \text{(query-to-anchor offset in original pixel coordinates)}
$$
$$
u_r = \text{LN}(W_u [h_{\text{point}}';\; g_r;\; \phi_{\text{B3}}(\Delta\mathbf{p}_r)] + b_u), \quad u_r \in \mathbb{R}^d
$$

where $g_r = W_g \, G_t^{(4)}[\mathbf{p}_r^{(4)}] \in \mathbb{R}^d$ is the anchor's feature from the globally-enriched L4 map (direct index lookup at the 40×22 grid, since anchor positions align with L4), projected from 384ch to $d{=}192$ via a shared linear $W_g$. $\mathbf{p}_r$ is the anchor's spatial position in pixel coordinates, and $\phi_{\text{B3}}$ is a normalized Fourier encoding (see below). Input dimension: $d + d + d_{\text{pe}} = 192 + 192 + 32 = 416$. This conditioning vector tells the offset head three things: "I'm this query ($h_{\text{point}}'$, **already globally aware from B2**), looking at an anchor **whose content is** $g_r$, **located at** this spatial offset." The conditioning and offset/weight heads are **shared across all 32 anchors** — each anchor produces a different $u_r$ because its inputs ($g_r$, $\Delta\mathbf{p}_r$) differ. The $W_g$ projection (~74K params) can be pre-applied once per frame to all 880 L4 positions.

**v9 improvement over v8:** In v8, the conditioning used $z_q$ (built from local features only — no global context). In v9, $h_{\text{point}}'$ carries **full scene context** from B2's cross-attention over 880 L4 tokens. This means the offset head can make globally-informed sampling decisions — e.g., "I know this is an indoor scene with a wall here, so sample along the wall edge in L2." The anchor content $g_r$ at 384ch (projected to $d$) is richer than v8's compressed latent bank tokens, benefiting from both Swin (L3) and full self-attention (L4) processing.

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
self.sampling_offsets = nn.Linear(d, H * L * M * 2)   # -> 6*3*4*2 = 144
self.attention_weights = nn.Linear(d, H * L * M)       # -> 6*3*4   = 72
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

where $W_V^{(h)} \in \mathbb{R}^{(d/H) \times C_\ell}$ is the value projection for head $h$ ($C_\ell$ is the channel dimension of level $\ell$: 128 for L2, 192 for L3, 384 for L4). In practice, a per-level $W_V^{(\ell)} \in \mathbb{R}^{d \times C_\ell}$ is used (with output reshaped to $H = 6$ heads of $d/H = 32$ dims). **Implementation optimization:** Following Deformable DETR, $W_V$ is pre-applied to each feature map $F_t^{(\ell)}$ once per frame during precompute, so grid_sample reads already-projected 192-ch features. This eliminates the per-sample projection cost from the per-query path.

**Wide pyramid channel ranks:** Per-level $W_V$ pre-projection normalizes 128/192/384ch → 192ch before storage (~136K total, ~0.05 ms). L2 at 128ch → rank 128/192 = **67%** (was 33% at 64ch). L3 at 192ch = $d$ → **full rank** (was 50% at 96ch). L4 at 384ch → over-specified (was 100% at 192ch). Every deformable read is significantly more informative than in the narrow pyramid.

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

**Why full multi-head structure:** $H=6$ independent 32-dim subspaces (matching B2 and B4 head count) learn complementary sampling patterns. Per-head softmax preserves each head's weight distribution (standard in Deformable DETR, DCNv3). Concat + $W_O$ enables representations outside the convex hull of sampled features.

**Default budget:** $H = 6$ heads, $L = 3$ levels (2–4), $M = 4$ samples.
- Per anchor: $6 \times 3 \times 4 = 72$ samples.
- 32 anchors: $32 \times 72 = 2{,}304$ deformable lookups total.

**Sampling budget analysis:** Total feature lookups per query: 4 (center) + 32 (L1 local) + 16 (intermediate local) + 2,304 (deformable) = **2,356**. Total multi-scale positions: L1(230,400) + L2(14,400) + L3(3,600) + L4(880) = **249,280**. Hard sampling fraction: 2,356 / 249,280 ≈ **0.95%** (~1%). Additionally, B2's soft cross-attention covers 100% of L4 (880 tokens) — the 1% applies only to hard multi-scale reads.

**Why $H = 6$, $L = 3$, $M = 4$:** $H = 6$ matches the attention head count in B2 and B4 ($d = 192$, $d_{\text{head}} = 32$). L1 and the stride-4 intermediate are covered by B1 local sampling. L2–L4 carry progressively deeper features: L2 (128ch, GRN-enhanced ConvNeXt), L3 (192ch, Swin windowed attention), L4 (384ch, full self-attention). 2,304 total lookups spread across 32 attention-guided anchor regions.

**Per-anchor aggregation** (not global softmax) — preserves each region's contribution. B4 learns inter-region weighting via cross-attention.

#### B4. Fusion decoder (2-layer cross-attention transformer)

**Context token set (with type embeddings):**

$$
T_q = [l_q + e_{\text{loc}}; \; c_q^{(2)} + e_{\text{ms2}}; \; c_q^{(3)} + e_{\text{ms3}}; \; c_q^{(4)} + e_{\text{ms4}}; \; h_{r_1..32} + e_{\text{deform}}]
$$

36 context tokens total, each with a learned type embedding:
- $l_q + e_{\text{loc}}$: aggregated local neighborhood (from B1).
- $c_q^{(2)} + e_{\text{ms2}}$: medium-range context (L2 feature at query, 128ch → 192ch via $W_{\text{ms2}}$, ~112px GRN-enhanced ConvNeXt context).
- $c_q^{(3)} + e_{\text{ms3}}$: broad context (L3 feature at query, 192ch = $d$, **no projection needed** — identity, Swin-level context with shifted window attention).
- $c_q^{(4)} + e_{\text{ms4}}$: scene-level context (L4 feature at query, 384ch → 192ch via $W_{\text{ms4}}$, global context from full self-attention).
- $h_{r_1} + e_{\text{deform}}, \ldots, h_{r_{32}} + e_{\text{deform}}$: deformable evidence from 32 attention-routed anchors (multi-scale non-local context from B3).

**v9 simplification from v8:** Removed $\bar{c}_q + e_{\text{glob}}$ (global summary) because $h_{\text{point}}'$ already carries global context from B2's cross-attention — a separate summary token would be redundant. Removed the $e_{\text{near}}$ / $e_{\text{route}}$ distinction (v8 had separate embeddings for nearest vs routed anchors) because attention-based routing has no special "nearest" anchor — all 32 are attention-selected. Net: 36 tokens, 5 type embeddings (was 38 tokens, 7 types).

**Per-level projection:** $c_q^{(2)}$ at 128ch and $c_q^{(4)}$ at 384ch are projected to $d = 192$ via $W_{\text{ms2}}$ and $W_{\text{ms4}}$ respectively. $c_q^{(3)}$ at 192ch = $d$ needs **no projection** (identity). $c_q^{(2)}$ has **dual role**: raw 128ch feeds $h_{\text{point}}$, projected 192ch is a B4 token. Projection parameters: $W_{\text{ms2}}$ (128×192 ≈ 25K) + $W_{\text{ms4}}$ (384×192 ≈ 74K) ≈ ~99K total.

**Type embeddings:**

Each context token receives a learned type embedding identifying its role:

$e_{\text{loc}}, e_{\text{ms2}}, e_{\text{ms3}}, e_{\text{ms4}}, e_{\text{deform}} \in \mathbb{R}^d$ — 5 learned embeddings (5 × 192 = 960 parameters). The three multi-scale embeddings ($e_{\text{ms2}}, e_{\text{ms3}}, e_{\text{ms4}}$) let the attention heads distinguish between context at different spatial scales. The single $e_{\text{deform}}$ for all 32 deformable tokens is appropriate because each token already carries distinct anchor content from B3's conditioning.

**KV normalization:**

Before entering the transformer, apply a shared LayerNorm to the assembled context tokens:
$$
T_q \leftarrow \text{LN}_{\text{kv}}(T_q)
$$

Normalizes heterogeneous token scales from different computational paths. Applied once since $T_q$ is static — essentially free.

**2-layer transformer decoder (standard Pre-LN):**

Each layer applies cross-attention with residual, then FFN with residual (standard Pre-LN transformer):
$$
h_{\text{point}}' \leftarrow h_{\text{point}}' + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}'), \; KV = T_q)
$$
$$
h_{\text{point}}' \leftarrow h_{\text{point}}' + \text{FFN}(\text{LN}(h_{\text{point}}'))
$$

where $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ with expansion ratio 4 ($d \to 4d \to d$, i.e., $192 \to 768 \to 192$). Note: the input to B4 is $h_{\text{point}}'$ (the globally-aware center token from B2), not the original $h_{\text{point}}$.

Cross-attention uses 6 heads with $d_{\text{head}} = 32$. Attention matrix per head is $1 \times 36$ — trivially cheap. 6 heads (matching B2 and B3) allow diverse attention patterns over the 5 token types.

After 2 layers: $h_{\text{fuse}} = h_{\text{point}}' \in \mathbb{R}^d$.

**Standard Pre-LN residuals** in both sub-layers (SAM, DETR, all modern transformers). Cross-attention is data-dependent: at depth edges, upweights local detail ($l_q$, deformable evidence near the query); in textureless regions, upweights broader multi-scale context ($c_q^{(3:4)}$). The 5 type embeddings help distinguish token roles. Residuals preserve $h_{\text{point}}'$'s global+local identity while cross-attention adds targeted multi-scale evidence.

**Why B4 is still needed after B2:** B2 provides *global* context via soft attention over all 880 L4 tokens — broad but at L4 resolution only (stride 32). B4 provides *targeted multi-scale* context via hard attention over 36 tokens carrying B3's deformable evidence at L2-L4 resolution (strides 8/16/32). The two are complementary: B2 = "understand the scene", B4 = "fuse detailed evidence for this specific depth prediction."

**Why 2 layers:** SAM's decoder uses 2 layers and achieves excellent point-prompt results. Our context set is small (36 tokens vs thousands of image tokens in SAM), so 2 layers provide sufficient mixing depth. Ablate with 1 and 3 layers.

**Total nonlinear depth per query (v9):**
- B1: h_point MLP (2 sub-layers)
- B2: L4 cross-attention (2 layers × 2 sub-layers = 4)
- B3: conditioning MLP (2 sub-layers)
- B4: fusion transformer (2 layers × 2 sub-layers = 4)
- B5: depth head MLP (2 sub-layers)
- **Decoder total: ~14 nonlinear sub-layers.**

Backbone nonlinear stages: L1 GELU (1) + L2 2× ConvNeXt (2×2=4) + L3 4× Swin (4×2=8) + L4 2× SelfAttn (2×2=4) = **~17 stages**. Combined with the decoder's ~14, each query passes through **~31 total nonlinear stages** — exceeding DAv2-S's ~24 encoder sub-layers (12 ViT layers × 2). Note: DAv2's DPT decoder adds ~18 more sub-layers for a total of ~42 per pixel; our advantage is that the decoder's 14 stages are applied only to K queries, not all 921,600 pixels.

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

A 2-layer MLP ($192 \to 96 \to 1$) applied to $h_{\text{point}}^{(0)}$, the center token BEFORE B2 global cross-attention (saved from B1). This forces the center branch to remain independently informative — it sees only local features, not global context from B2 or deformable evidence from B3.

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
| `M1_Precompute` | A1–A4 | $E_t \to F_t^{(1)}, F_t^{(\text{int})}, F_t^{(2:4)}, G_t^{(4)}, K_t, V_t, \hat{F}_t^{(2:4)}, s_t, b_t$ |
| `M2_QueryEncode` | B1+B2 | $q \to h_{\text{point}}, l_q, l_q^{(\text{int})}, c_q^{(2:4)} \to h_{\text{point}}' \text{ (via L4 cross-attn, 880 tok)}, R_q$ |
| `M3_DeformRead` | B3 | $R_q, h_{\text{point}}', G_t^{(4)}, \hat{F}_t^{(2:4)} \to h_{r_1}, \ldots, h_{r_{32}}$ |
| `M4_FuseDecode` | B4+B5 | $h_{\text{point}}', T_q(36\text{ tokens}), s_t, b_t \to \hat{d}_q$ |

**v9 note:** M2 now includes B2 (L4 cross-attention + routing) since it produces both the globally-aware $h_{\text{point}}'$ and the routing set $R_q$. B2's cross-attention is batched across all $K$ queries using pre-computed $K_t, V_t$.

---

### 4.6 Why This Design Will Work — Confidence Arguments

1. **Precompute-then-query is proven.** SAM and Perceiver IO achieve SOTA with lightweight decoders on precomputed features. v9's L4 self-attention + KV pre-computation fits this paradigm.
2. **Wide pyramid with SOTA-calibrated channels provides rich depth cues.** F^3 already matches DAv2 for dense depth. The wide pyramid (64/128/192/384ch) matches DAv2-S's DPT decoder channel profile. L3 Swin gives medium-range context, L4 full self-attention (880 tokens at 384ch) gives true global scene understanding.
3. **Two-phase reasoning (global then local) is well-motivated.** B2's cross-attention into 880 globally-enriched L4 tokens first understands the scene layout, then B3's deformable read gathers targeted multi-scale evidence at strides 8/16/32. This mirrors how DPT uses global ViT features followed by local reassembly.
4. **Deformable attention is battle-tested.** Deformable DETR, Mask2Former, DAB-DETR — robust and well-understood. v9's attention-based routing (R=32 from 880 tokens, 3.6% selection ratio) is well-calibrated.
5. **Speed advantage is structural.** Precompute dominates (~84% at $K=256$). Crossover at $K \approx 3{,}000$+. B2 over 880 tokens is 4× cheaper than a 3,600-token design.
6. **Deep per-query processing compensates for backbone.** ~31 total nonlinear stages per query (backbone ~17 + decoder ~14) exceed DAv2-S's encoder depth (~24). Dense methods can't afford this depth per pixel.

---

### 4.7 Comparison: DepthAnythingV2 (DAv2) vs EventSPD

DAv2 is the dense depth baseline used in the F^3 pipeline. Understanding its architecture clarifies what EventSPD replaces and why.

**Architecture comparison:**

| Aspect | DAv2-S (F^3 baseline) | EventSPD v9 |
|--------|----------------------|-------------|
| **Encoder** | DINOv2 ViT-S (384-dim, 12 layers, 6 heads, patch=14) | Wide pyramid (64/128/192/384ch, strides 2/8/16/32) + L4 full self-attn (2 layers, 880 tokens) |
| **Encoder input** | 518×518 RGB (resized) | 640×360 F^3 ds2 features (64ch) |
| **Encoder output** | 37×37 tokens (1,369 positions) | 4-level pyramid: 640×360 / 160×90 / 80×45 / 40×22 + enriched L4 ($G_t^{(4)}$, 384ch) |
| **Decoder** | DPT head (4 reassemble + 4 fusion stages, features=64) | Sparse per-query: L4 cross-attn (2 layers, 880 tok) + attention-guided deformable + 2-layer transformer ($d = 192$) |
| **Output** | Dense 518×518 depth map | K sparse depth values at query points |
| **Encoder FLOPs** | ~93G (ViT-S self-attention + FFN, 12 layers) | ~16G (Conv+Swin+SelfAttn backbone) |
| **Decoder FLOPs** | ~8G (DPT, 4-stage upsampling + fusion) | ~26M per query × K |
| **Total FLOPs** | ~101G per frame | ~73G (F^3 ds2 ~57G + backbone ~16G) + 26M×K |
| **Encoder params** | ~22M (ViT-S) | ~6.3M (wide backbone + KV proj) |
| **Decoder params** | ~3M (DPT-S head) | ~2.1M (per-query decoder) |
| **Latency** | ~23 ms (F^3 8.3ms + DAv2 ~15ms) | ~7.8 ms at K=256 |
| **Depth source** | DINOv2 pretrained features (142M images) | F^3 event features + learned backbone |
| **Receptive field** | Global (ViT self-attention at every layer) | L1-L2: local (CNN+GRN); L3: medium (Swin); L4: global (full self-attn) |
| **Nonlinear depth** | ~42 sub-layers (24 encoder + 18 DPT decoder) | ~31 sub-layers (backbone ~17 + decoder ~14) |
| **Channel profile** | DPT: 48/96/192/384 | **64/128/192/384** (SOTA-calibrated) |

**DAv2 advantages:** Dense depth maps (all pixels), massive DINOv2 pretraining (142M images), works with RGB cameras.

**EventSPD advantages:** Sublinear query scaling ($O(1) + O(K)$, ~3× faster at K=256), two-phase per-query reasoning (global cross-attn + targeted deformable), attention-guided routing (3.6% selection ratio, zero extra parameters), SOTA-calibrated channel widths (matching DAv2-S's DPT profile), native event camera support, per-query uncertainty.

**Key trade-off:** DAv2 spends ~93G FLOPs building a universally rich representation (ViT-S, 12 self-attention layers over 1,369 tokens), then reads densely. EventSPD spends ~73G on F^3 ds2 (~57G) + wide pyramid backbone (~16G) — 24% fewer encoder FLOPs — then reads sparsely via attention-guided deformable sampling. The wide pyramid's channel profile (64/128/192/384) closely matches DAv2-S's DPT decoder (48/96/192/384), while the backbone runs **much faster** because F^3 is memory-bound (4.5 ms + 1.2 ms = 5.7 ms vs DAv2's ~15 ms). The per-query decoder adds ~14 nonlinear stages on top of the backbone's ~17, giving ~31 total per query.

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

**Optional additions:** $L_{\text{rank}}$ (pairwise ranking), $L_{\text{unc}}$ (Gaussian NLL for uncertainty), $L_{\text{entropy}}$ (attention entropy regularizer on B2 routing). Add only for specific failure modes.

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

**Attention dropout:** Standard dropout ($p = 0.1$) on cross-attention weights in B2 and B4, following standard transformer practice. This mildly regularizes which L4 tokens (B2) and context tokens (B4) dominate per query.

**Center collapse prevention** relies on $L_{\text{ctr}}$ (auxiliary loss on pre-fusion center token), following DETR's auxiliary loss approach.

**Standard:** Weight decay 0.01, gradient clipping 1.0, mixed precision (bf16).

---

## 6. Runtime Analysis

### 6.1 Parameter Budget

**v9 separates parameters into two categories:** Phase A preprocessing (computed once per frame, amortized over all queries) and Phase B decoder (per-query cost). The wide pyramid architecture (strides 2/8/16/32, channels 64/128/192/384) invests more parameters in the backbone to match SOTA channel profiles, while the decoder dimension $d{=}192$ is unchanged.

**Phase A: Preprocessing parameters (amortized, once per frame)**

| Component | Parameters |
|-----------|------------|
| A2: L1 LN + GELU (identity from F^3 ds2) | ~0.1K |
| A2: Stem-1 Conv(64→64, k3s2) + LN | ~37K |
| A2: Stem-2 Conv(64→128, k2s2) + LN | ~33K |
| A2: L2 — 2× ConvNeXt$_{128}$ (GRN) | ~277K |
| A2: Down L2→L3 (128→192, k2s2) + LN | ~99K |
| A2: L3 — 4× SwinBlock$_{192}$ (w=8, shifted) | ~1,778K |
| A2: Down L3→L4 (192→384, k2s2) + LN | ~295K |
| A2: L4 — 2× FullSelfAttn$_{384}$ (880 tokens) | ~3,542K |
| A3: $K_t, V_t$ projection for B2 ($384 \to 192$) | ~148K |
| A3: $W_g$ anchor projection for B3 ($384 \to 192$) | ~74K |
| Per-level $W_V$ pre-proj (128/192/384→192) | ~136K |
| A4: Calibration heads ($s_t$, $b_t$, from 384ch) | ~0.8K |
| **Phase A total** | **~6,420K** |

The L4 full self-attention at 384ch dominates (~3,542K, 55%). This is the cost of SOTA-width channels at the global level — each self-attention layer requires $12 \times 384^2$ params. L3 Swin at 192ch (~1,778K, 28%) is identical to the previous L4 Swin. L2 ConvNeXt at 128ch (~277K) is 3.8× the previous 64ch cost but still modest.

**Phase B: Decoder parameters (per-query cost, $d = 192$)**

| Component | Lookups | Attn ops | Params | Detail |
|-----------|:---:|:---:|---:|--------|
| B1: Centers (L1 + L2–L4) | 4 | — | ~154K | MLP 608→192→192 |
| B1: L1 local sampling | 32 | — | ~54K | Local MLP (80→192) + offsets |
| B1: Stride-4 intermediate local | 16 | — | ~54K | Local MLP (80→192) + offsets |
| B2: L4 cross-attention (2 layers) | — | 880×2 | ~740K | Q proj + O proj + FFN per layer ($K_t, V_t$ pre-computed) |
| B3: Deformable read | 2,304 | — | ~159K | Cond + offsets + $W_O$ (H=6, no $W_V$, pre-applied) |
| B4: $W_{\text{ms2/4}}$ proj | — | — | ~99K | 128/384→192 (L3=identity) |
| B4: Type embeddings | — | — | ~1.0K | 5 × $d$ |
| B4: KV norm | — | — | ~0.4K | $\text{LN}_{\text{kv}}$ |
| B4: Fusion (2 layers) | — | 36×2 | ~889K | 6-head transformer |
| B5: Depth head | — | — | ~37K | MLP 192→192→1 |
| B5: $L_{\text{ctr}}$ aux | — | — | ~19K | MLP 192→96→1 |
| **Phase B total** | **2,356** | **~1,832** | **~2,206K** | |

The decoder is essentially unchanged from the narrow-pyramid design (~2.2M vs ~2.2M). B2 now attends to only 880 tokens (was 3,600) — 4× fewer attention operations per query, but same parameter count since the learned weights are independent of token count. B1's h_point MLP is slightly wider (608→192 vs 544→192, +12K). B4 saves ~31K by not needing a $W_{\text{ms3}}$ projection (L3 at 192ch = $d$).

B3 parameter breakdown: conditioning MLP (416→192, ~80K) + offset head (192→144, ~28K) + weight head (192→72, ~14K) + $W_O$ (192→192, ~37K). Same as before (d unchanged).

B4 fusion remains the largest decoder component (~889K, 40%): 2 layers × (6-head cross-attn $4d^2$ + FFN $8d^2$ + LNs).

**Total trainable parameters: ~8,626K (~8.6M)**

**Wide pyramid vs narrow pyramid:** Phase A: 3,177K → 6,420K (+102%, L4 self-attn at 384ch). Phase B: 2,163K → 2,206K (+2%, negligible). Net: 5,340K → 8,626K (+62%). The increase is entirely in the backbone. The decoder cost is unchanged because $d{=}192$ is preserved.

**vs DAv2-S:** Our decoder is **12× smaller** (2,163K vs 25M). Total params **4.7× smaller** (5.3M vs 25M). 2,356 lookups/query = 391× less than dense (921,600 pixels).

### 6.2 Speed Estimate (RTX 4090, 1280×720 events)

**Precompute (once per window):**
$$
T_{\text{precompute}} = T_{\text{F}^3}^{\text{ds2}} + T_{\text{backbone}} + T_{\text{KV+}W_V} + T_{\text{cal}} \approx 4.5 + 1.2 + 0.05 + 0.06 = 5.8 \text{ ms}
$$

**F^3 ds2 timing breakdown (RTX 4090, estimated):**

| Component | ds1 (ms) | ds2 (ms) | Ratio | Notes |
|-----------|:--------:|:--------:|:-----:|-------|
| MRHE encoding | 0.15 | 0.15 | 1.0× | Hash tables (16 MB) fit L2; same N events |
| Scatter (`index_put_`) | 0.80 | 0.80 | 1.0× | Random atomics to 1280×720×8 tensor |
| `forward_variable` overhead | 0.25 | 0.25 | 1.0× | `repeat_interleave`, `.clone()`, coord scaling |
| $\text{ds}_{\text{layer}}[0]$ + LN | 0.35 | 0.20 | 0.57× | ds2: Conv k3s2 on full-res → half-res |
| 3× ConvNeXt blocks (stage 0) | 2.85 | 1.20 | 0.42× | 0.50× traffic × 0.85 L2 cache bonus |
| $\text{ds}_{\text{layer}}[1]$ + LN | 0.30 | 0.15 | 0.50× | |
| 3× ConvNeXt blocks (stage 1) | 2.85 | 1.20 | 0.42× | 0.50× traffic × 0.85 L2 cache bonus |
| Kernel launch overhead | 0.20 | 0.20 | 1.0× | ~40 kernels × 5 μs |
| **F^3 total** | **~8.3** [^1] | **~4.5** | **0.54×** | |

[^1]: Calibration anchor: ds1 = ~8.3 ms from F^3 codebase benchmarks.

**Why FLOPs mislead here:** F^3 ds2 has ~57 GFLOPs vs ds1's ~65G — only 12% fewer. But the **wall-clock drops 46%** because F^3 is **memory-bandwidth-bound** on RTX 4090. The roofline analysis (see Section 8, ablation #35) shows all operations have arithmetic intensity 10–21 FLOP/byte, well below the 82 FLOP/byte ridge point. The 50% memory traffic reduction (from halving spatial resolution) and L2 cache fit (ds2's 59 MB feature maps fit the 72 MB L2; ds1's 118 MB do not) together account for the ~2× speedup on the conv backbone portion.

Our backbone adds ~1.2 ms (see A2 cost table: L1 LN+GELU ~0.01 ms, Stem-1+Stem-2 ~0.08 ms, L2 ConvNeXt$_{128}$ at 160×90 ~0.3 ms, L3 Swin$_{192}$ at 80×45 ~0.35 ms, L4 FullSelfAttn$_{384}$ at 40×22 ~0.1 ms, downsampling ~0.06 ms, rest ~0.3 ms). Note: L4's full self-attention over 880 tokens is integrated into the backbone — no separate A3 stage. The wider channels (128/192/384) cost more FLOPs per position but operate on far fewer positions (14,400/3,600/880 vs old 57,600/14,400/3,600), yielding a net speedup. The $K_t, V_t$ projection for B2 is negligible (~0.003 ms). Following the Deformable DETR implementation pattern, B3's per-level $W_V$ value projection is pre-applied to the feature maps $\hat{F}_t^{(2:4)}$ once per frame (~0.05 ms for mixed-channel projection: L2 128→192, L3 192→192, L4 384→192), so grid_sample reads already-projected 192ch features during per-query decoding.

**Per-query cost analysis:**

The decoder cost has two components: a **fixed overhead** (kernel launches and PyTorch dispatch, independent of $K$) and a **marginal cost** per query (compute + memory bandwidth, scaling with $K$):
$$
T_{\text{decoder}}(K) = \alpha + \beta K
$$

Per-query FLOPs breakdown (~26.5M FLOPs/query with pre-projected $W_V$ and $K_t$/$V_t$, $d = 192$):

| Stage | Dominant operations | FLOPs |
|-------|-------------------|---:|
| B1: center + local | MLP(608→192→192), L1/stride-4-int local MLPs, 52 grid_samples | ~1,100K |
| B2: L4 cross-attention | 2× [Q proj + QK$^T$(880) + attn×V + O proj + FFN(192→768→192)] | ~2,800K |
| B3: deformable | 32× cond MLP(416→192), offset/weight heads, $W_O$ (H=6), 2,304 grid_samples | ~10,300K |
| B4: projection + fusion | $W_{\text{ms}}$(128/192/384→192), 2× [6-head cross-attn + FFN(192→768→192)], 36 tokens | ~12,200K |
| B5: depth head | MLP(192→192→1) | ~74K |
| **Total** | | **~26.5M** |

**Why per-query FLOPs drop with the wide pyramid:** The wide pyramid's L4 has only 880 tokens (40×22) vs the narrow pyramid's 3,600 tokens (80×45). B2's cross-attention — which scales linearly with L4 token count — drops from ~7,000K to ~2,800K FLOPs (−60%). B2 is batched across all $K$ queries as a single $(K \times 192) \times (192 \times 880)$ matrix multiplication — **extremely GPU-efficient**. B3's grid_sample count is unchanged at 2,304 (same R=32, H=6, L=3, M=4), so the memory-bound bottleneck is identical. The wider channel features (128/192/384ch) read by grid_sample are pre-projected to 192ch, so read bandwidth is the same.

For reference, the F^3 ds2 backbone processes ~57 GFLOPs per frame. Our decoder at ~26.5M FLOPs/query is **~2,150× less compute** than one F^3 ds2 pass.

**Conservative estimates** (standard PyTorch, no custom CUDA, no torch.compile):
- $\alpha \approx 0.30$ ms (fixed kernel launch + dispatch overhead).
- $\beta \approx 0.005$ ms/query (880-token B2 cross-attention is 4× cheaper than a 3,600-token variant; B3's grid_samples unchanged at 2,304).

**Total:**
$$
T_{\text{EventSPD}}(K) \approx 6.1 + 0.005K \text{ ms}
$$

| Query count $K$ | Decoder (ms) | EventSPD total (ms) | Throughput (Hz) | Dense baseline (ms) | Speedup |
|-----|---:|---:|---:|---:|---:|
| 1 | 0.31 | 6.1 | 164 | ~23 | 3.8× |
| 64 | 0.62 | 6.4 | 156 | ~23 | 3.6× |
| 256 | 1.58 | 7.4 | 135 | ~23 | 3.1× |
| 1024 | 5.42 | 11.2 | 89 | ~23 | 2.1× |

Dense baseline: F^3 ds1 (8.3 ms) + DepthAnythingV2 decoder (~15 ms) = ~23 ms. The crossover point where EventSPD matches dense cost: $(23 - 6.1) / 0.005 \approx K = 3{,}380$ — far beyond any practical query count. Even at $K = 1024$, EventSPD is **2.1× faster** than dense.

**Wide pyramid speed impact:** The wide pyramid (strides [2,8,16,32]) delivers speed improvements in both precompute and per-query phases. Precompute drops from ~6.9 ms (narrow) to ~5.8 ms (wide, −16%) because L2 ConvNeXt operates at 160×90 (14,400 positions) instead of 320×180 (57,600 positions), and L4 self-attention runs over 880 tokens instead of 3,600. Per-query: $\beta$ drops from 0.006 to 0.005 ms/query thanks to B2's 4× smaller cross-attention (880 vs 3,600 tokens). Net effect at $K = 256$: **7.4 ms (wide) vs 8.5 ms (narrow) — 13% faster**. The speed gain comes at the cost of +3.3M backbone parameters (see Section 6.1), but all additional params are in the amortized precompute — the per-query decoder is essentially unchanged.

**Key insight:** Precompute (~5.8 ms) accounts for ~78% of total time at $K = 256$. F^3 ds2 (~4.5 ms) is still 78% of precompute — the event encoder remains the bottleneck. Our backbone (~1.2 ms) and decoder (~1.6 ms) together are less than F^3. This means **further speedups require either lighter event encoders or event voxel grid representations** (see Section 8, ablation #35).

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

**Routing diversity (v9: from B2 attention weights):**
- Entropy of $\bar{\alpha}_q$ (B2's averaged attention weights over 880 L4 positions) averaged over queries. Low entropy = potential routing collapse (all queries attend to same positions). High entropy = diffuse attention (no spatial focus). Target: moderate entropy indicating query-dependent spatial selection. With 880 tokens, the selection ratio is 3.6% ($R = 32$ out of 880), making routing more selective than the narrow pyramid's 0.89% ($R = 32$ out of 3,600). v9's attention-based routing provides this metric for free — no separate routing head to diagnose.

**B2 global context quality:**
- Compare $h_{\text{point}}'$ (post-B2) vs $h_{\text{point}}$ (pre-B2): measure cosine similarity. Low similarity = B2 cross-attention is making significant changes (good — global context is informative). Near-1.0 similarity = B2 is not contributing (bad — cross-attention may be collapsing to identity).

**B4 attention balance:**
- Average cross-attention weight per context token type ($l_q$, $c_q^{(2)}$, $c_q^{(3)}$, $c_q^{(4)}$, $h_{\text{deform}}$) across queries. Shows whether the model uses all 5 information streams. Particularly monitor the multi-scale center tokens ($c_q^{(2:4)}$) — if their attention weights are near-zero, the backbone's multi-scale features are not contributing to fusion. Also compare deformable token weights vs local/center weights: if deformable tokens dominate, the global + local contributions may be insufficient.

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
| 1 | Remove global context (B2) | Does L4 cross-attention help depth? | Remove B2. $h_{\text{point}}' = h_{\text{point}}$ (no global context). Use random anchors for B3. |
| 2 | Remove deformable sampling (B3) | Is targeted multi-scale evidence needed beyond B2? | Remove B3. $T_q = [l_q; c_q^{(2:4)}]$ (4 tokens, no deformable). |
| 3 | Remove local sampling | Does neighborhood context help? | Set $N_{\text{loc}} = 0$. Center + global + deformable only. |
| 4 | Query count scaling | How do accuracy and speed scale with $K$? | $K \in \{1, 4, 16, 64, 256, 1024\}$. |
| 5 | Freeze vs fine-tune backbone | Does backbone adaptation help? | Frozen / 0.1× LR / full LR. |

### Tier 2 — Design choices

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 6 | B2 layer count | How many cross-attention layers over L4? | 1, 2 (default), 3 layers. |
| 7 | Routing budget $R$ | How many attention-guided anchors? | $R \in \{8, 16, 32, 64, 128\}$. Default: 32. |
| 8 | Fusion depth (B4) | How many fusion layers? | 1, 2 (default), 3 layers. |
| 9 | L4 self-attention depth | How many full self-attn layers at L4? | 0 (Swin-only at L4), 1, 2 (default), 4 layers. |
| 10 | Routing source | Attention weights vs learned routing? | (a) Attention-based (default). (b) v8-style $W_r$ routing head. (c) Random anchors. |
| 11 | Local budget | How many local? | $N_{\text{loc}} \in \{8, 16, 32, 48, 64\}$. |
| 12 | Deformable budget | Samples per anchor? | $(H,L,M) \in \{(4,3,2)..(6,3,6)\}$. Default: $(6,3,4)$. |

### Tier 3 — Extensions and alternatives

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 13 | B2 only (no B3/B4) | Can L4 cross-attention alone suffice? | Remove B3+B4. Depth head directly on $h_{\text{point}}'$. Tests if global context alone predicts depth. |
| 14 | Hash encoding vs Fourier PE | Does learned position encoding help? | Replace $\text{pe}_q$ with Instant-NGP hash. |
| 15 | Temporal memory | Does GRU state across windows help? | Add $H_t$ to cache and to $T_q$. |
| 16 | Uncertainty head | Does uncertainty improve hard queries? | Enable $\sigma_q$ + $L_{\text{unc}}$. |
| 17 | Center auxiliary loss | Does $L_{\text{ctr}}$ prevent center collapse? | Disable $L_{\text{ctr}}$ and compare. |
| 18 | Attention dropout rate | What rate is best for B2/B4 regularization? | $p_{\text{attn}} \in \{0, 0.05, 0.1, 0.2\}$. |

### Tier 4 — Architecture audit ablations (backbone design)

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 19 | L1 stride | 2× vs 1× vs 4× stride? | (a) Conv k3s2 (current, 640×360). (b) 1×1 proj (1280×720, v6). (c) Conv k3s4 (320×180). |
| 20 | L3 Swin block count | How many Swin blocks at L3? | (a) 4 blocks (current). (b) 2 blocks. (c) 6 blocks. |
| 21 | L4 self-attn block count | How many full self-attn layers at L4? | (a) 2 layers (current). (b) 1 layer. (c) 4 layers. See also ablation #9. |
| 22 | Center token paths | Separate L3-L4 from $h_{\text{point}}$? | (a) L1+L2 in $h_{\text{point}}$ (current, 608ch). (b) All compressed (v5). (c) L1-only (288ch). |
| 23 | Offset bounding | Unbounded vs bounded? | (a) Unbounded+0.1×LR (current). (b) $\tanh$. (c) Clamp. |
| 24 | Multi-head deformable | Full $W_V$/$W_O$ vs simple? | (a) Full (current). (b) Weighted sum (fewer params). |
| 25 | KV normalization | $\text{LN}_{\text{kv}}$ on B4 context? | (a) With (current). (b) Without. |
| 26 | Auxiliary calibration | Calibrate $r_q^{\text{ctr}}$? | (a) Calibrated (current). (b) Uncalibrated. |
| 27 | Stride-4 intermediate local in h_point | Stride-4 intermediate local contribution? | (a) Full 608ch (current). (b) No $l_q^{(\text{int})}$ (416ch). (c) L1-only (288ch). |
| 28 | Deformable head count | H=6 vs H=4 vs H=8? | (a) H=4 (fewer subspaces). (b) **H=6 (default)**. (c) H=8 (more subspaces). |
| 29 | Backbone architecture | CNN-only vs Hybrid vs full Swin? | See configs below. |

**Ablation #29 — Backbone architecture variants:**

This is the key ablation for justifying the hybrid backbone design. Tests whether Swin at L3 and full self-attention at L4 contribute over CNN-only, and how much backbone depth matters. All configs use the wide pyramid strides [2,8,16,32] with channels [64,128,192,384] and L1 GELU. L4 full self-attention is applied regardless of L3 block type.

| Config | Architecture | A2 Params | Precompute | Global RF? |
|:---:|----------|---:|---:|:---:|
| (a) | AvgPool + 1×1 proj only (no backbone) | ~150K | ~5.0 ms | Via L4 self-attn only |
| (b) | CNN-only: L2 2×ConvNeXt$_{128}$, L3 4×ConvNeXt$_{192}$, L4 2×SelfAttn$_{384}$ | ~4,500K | ~5.5 ms | Via L4 self-attn only |
| (c) | CNN-light: L2 2×ConvNeXt$_{128}$, L3 2×ConvNeXt$_{192}$, L4 2×SelfAttn$_{384}$ | ~3,800K | ~5.3 ms | Via L4 self-attn only |
| (d) | Hybrid-light: L2 2×ConvNeXt$_{128}$, L3 2×Swin$_{192}$, L4 2×SelfAttn$_{384}$ | ~4,200K | ~5.4 ms | Partial + L4 |
| (e) | **Hybrid (default): L2 2×ConvNeXt$_{128}$, L3 4×Swin$_{192}$, L4 2×SelfAttn$_{384}$** | **~5,963K** | **~5.8 ms** | **Yes + L4** |
| (f) | Heavy: L2 4×ConvNeXt$_{128}$, L3 6×Swin$_{192}$, L4 4×SelfAttn$_{384}$ | ~10,200K | ~6.5 ms | Yes + L4 |

Config (a) establishes the accuracy floor without backbone processing (L4 self-attention still provides global context on projected F^3 features). Config (b) replaces L3's Swin blocks with ConvNeXt blocks — tests whether Swin's shifted-window attention at L3 matters when L4 self-attention already provides full global context. The key question: with L4's 880-token full self-attention, does Swin at L3 still contribute over CNN-only? If (b) ≈ (e), the Swin blocks may be redundant.

### Tier 5 — Widening and architecture ablations

These ablations justify design choices: pyramid width, core dimension, L1 activation, L2 channel width, and GRN. Each isolates one variable against the default configuration.

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 30 | **Pyramid width (Option B)** | Does widening the core dim to $d = 256$ with channels [64,128,256,512] improve accuracy enough to justify the cost? | See detailed config below. |
| 31 | Core dimension $d$ (fixed backbone) | Does $d = 192$ justify the cost over $d = 128$? | (a) $d = 128$: 4-head attention, FFN 128→512→128. ~1.1M decoder. (b) **$d = 192$ (default)**: 6-head, FFN 192→768→192. ~2.2M decoder. (c) $d = 256$: 8-head, FFN 256→1024→256. ~3.6M decoder. Backbone channels unchanged [64,128,192,384]. |
| 32 | L1 activation | Does GELU on L1 help depth accuracy? | (a) **L1 GELU (default)**: LN + GELU (identity from ds2). Zero extra params. (b) L1 GELU + extra conv: + Conv(64→64, k3, pad1) + LN + GELU (~37K). (c) L1 linear: LN only (no activation). |
| 33 | L2 channel width | Is 128ch the right width for L2? | (a) L2 = 96ch: 2× ConvNeXt$_{96}$ at 160×90. $h_{\text{point}}$ input 576ch. (b) **L2 = 128ch (default)**: 2× ConvNeXt$_{128}$ at 160×90. $h_{\text{point}}$ input 608ch. (c) L2 = 192ch: 2× ConvNeXt$_{192}$ at 160×90. $h_{\text{point}}$ input 672ch. |
| 34 | GRN in ConvNeXt | Does Global Response Normalization help? | (a) **With GRN (default)**: ConvNeXt V2-style blocks. (b) Without GRN: ConvNeXt V1-style blocks (v7 design). |

**Ablation #30 — Option B: Wide pyramid with $d = 256$, channels [64,128,256,512]:**

This is the key alternative architecture ablation. Option B widens both the backbone and the decoder core dimension, bringing the channel profile closer to the SOTA 1:2:4:8 doubling pattern (Option A uses 1:2:3:6). The comparison isolates whether wider channels and richer per-query capacity improve depth accuracy enough to justify the parameter and compute cost.

| | Option A (default) | Option B (ablation) | Ratio |
|---|:---:|:---:|:---:|
| **Pyramid channels** | 64 / 128 / 192 / 384 | 64 / 128 / 256 / 512 | — |
| **Channel ratio** | 1 : 2 : 3 : 6 | 1 : 2 : 4 : 8 | SOTA standard |
| **Core dim $d$** | 192 | 256 | 1.33× |
| **B2/B4 heads** | 6 (d_head=32) | 8 (d_head=32) | 1.33× |
| **FFN hidden** | 768 | 1024 | 1.33× |
| **L4 tokens** | 880 (40×22) | 880 (40×22) | 1.0× |
| **L4 self-attn d_head** | 64 (384/6) | 64 (512/8) | 1.0× |
| **Backbone params** | ~5,963K | ~10,200K | 1.71× |
| **Decoder params** | ~2,206K | ~3,900K | 1.77× |
| **Total params** | ~8.6M | ~14.1M | 1.64× |
| **Per-query FLOPs** | ~26.5M | ~47M | 1.78× |
| **$\beta$ (ms/query)** | ~0.005 | ~0.009 | 1.8× |
| **Precompute (ms)** | ~5.8 | ~6.5 | 1.12× |
| **$T(K{=}256)$ (ms)** | ~7.4 | ~8.8 | 1.19× |
| **Speedup vs dense** | 3.1× | 2.6× | — |

Option B's per-query cost increases by ~1.78× due to wider attention and FFN, raising $\beta$ from ~0.005 to ~0.009 ms/query. Precompute also increases (~6.5 ms vs ~5.8 ms) from wider L3 Swin blocks and L4 self-attention. The key question: does the richer representation (+33% heads, +33% FFN width, SOTA channel profile) improve depth accuracy enough to justify 1.64× parameters and 1.19× latency? If Option B shows meaningful accuracy gains (>1% AbsRel improvement), it may be the preferred configuration for accuracy-first deployments. If the gains are marginal, Option A's speed advantage dominates.

**Ablation #31 — Core dimension $d$:** This isolates the decoder width from the backbone. All three configs use the same backbone channels [64,128,192,384]; only the decoder's core dim, attention heads, and FFN width change. At $d = 128$, B2+B4 transformers have only 4 heads, limiting multi-pattern attention. At $d = 192$, we get 6 heads (+50%) and wider FFN (768 vs 512 hidden). At $d = 256$, +33% more heads but with the same backbone — tests whether decoder capacity alone (without wider backbone features) improves depth.

**Ablation #32 — L1 activation:** L1 is terminal — it feeds directly into $h_{\text{point}}$ MLP and local sampling (not into further backbone blocks). GELU adds nonlinearity at zero parameter cost. The extra conv variant tests whether a second 3×3 conv (capturing more local patterns) helps beyond what the $h_{\text{point}}$ MLP already provides.

### Tier 6 — F^3 backbone resolution ablation

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 35 | F^3 output resolution | Does F^3 ds2 lose accuracy vs ds1? | See configs below. |

**Ablation #35 — F^3 output resolution:**

This ablation tests whether compressing the F^3 backbone from full-resolution (ds1: 1280×720×32) to half-resolution (ds2: 640×360×64) preserves depth accuracy while delivering significant speed gains. The ds2 config is the default.

| Config | F^3 output | F^3 time | L1 op | Backbone time | Precompute | Total params |
|:---:|----------|---:|----------|---:|---:|---:|
| (a) | ds1: 1280×720×32 | ~8.3 ms | Conv(32→64, k3s2) + LN + GELU | ~1.2 ms | ~9.6 ms | ~8.6M |
| (b) | **ds2: 640×360×64 (default)** | **~4.5 ms** | **LN + GELU (identity)** | **~1.2 ms** | **~5.8 ms** | **~8.6M** |
| (c) | ds4: 320×180×64 | ~3.0 ms | — (skip L1) | ~1.0 ms | ~4.1 ms | ~8.5M |

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

Config (a) is the ds1 baseline — full 1280×720 output, Conv L1 stride-2 compress. Config (b) is the default — F^3 directly outputs 640×360×64, making L1 a trivial LN+GELU. This saves ~3.8 ms in F^3 alone (backbone time is similar since L2+ operates at the same resolution). Config (c) pushes further to ds4 — F^3 outputs 320×180×64, skipping L1 entirely and feeding directly into Stem-1. This saves ~5.5 ms total but loses half-resolution spatial detail for L1 local sampling. The key question: does the information lost in ds2's 2× spatial compression (partially compensated by 2× channel widening to 64ch) degrade depth accuracy at query points?

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
│   ├── backbone.py             # A2: Wide pyramid backbone (L1 GELU, Stem-1/2, ConvNeXt L2, Swin L3, SelfAttn L4)
│   ├── precompute.py          # A3: KV projection for B2, W_g anchor projection, W_V pre-projection
│   ├── query_encoder.py       # B1: L1 center + stride-4 intermediate local + multi-scale center reads
│   ├── l4_cross_attention.py  # B2: 2-layer L4 cross-attention (880 tokens) + attention-based routing
│   ├── deformable_read.py     # B3: offset prediction + multiscale sampling (L2-L4 at s8/s16/s32)
│   ├── fusion_decoder.py      # B4: 2-layer cross-attention transformer (36 tokens, 5 type embeddings)
│   └── depth_head.py          # B5: depth prediction + calibration + uncertainty
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
| 3 | Add wide pyramid backbone (A2: L1 GELU + Stem-1/2 + ConvNeXt L2 + Swin L3 + SelfAttn L4) | Feature quality improvement, multi-scale pyramid | 1-2 weeks |
| 4 | Add L4 cross-attention (B2, 880 tokens) + attention-based routing | Global context benefit + routing | 1-2 weeks |
| 5 | Add deformable sampling (B3, s8/s16/s32, per-level $W_V$) | Targeted multi-scale evidence | 1-2 weeks |
| 6 | Full EventSPD pipeline + training (36 tokens, 5 types) | Complete system | 2 weeks |
| 7 | Ablations (incl. Option B) + paper figures | Publication readiness | 3-4 weeks |

**Critical rule:** After each step, measure accuracy AND speed. Do not proceed to the next step if the current one degrades accuracy without clear speed benefit.

---

## 10. Timeline (16 weeks)

### Weeks 1-2: Foundations
- Reproduce dense F^3 + DA-V2 baseline. Verify metrics and runtime.
- Build query-level evaluation harness (sample from dense output, compare).
- Build profiling harness for latency decomposition.

### Weeks 3-4: Minimal query decoder
- Implement build steps 2-3 (bilinear + MLP, then add backbone).
- First accuracy comparison: minimal decoder vs dense baseline at query points.
- First speed comparison: latency curves.

### Weeks 5-7: Full EventSPD implementation
- Add L4 cross-attention (B2, 880 tokens), deformable sampling (B3, s8/s16/s32), fusion transformer (B4) (build steps 4-6).
- Train stages 1 and 2.
- Produce crossover plot: $T_{\text{total}}(K)$ for EventSPD vs dense.

### Weeks 8-10: Core ablations
- Tier 1 ablations (B2 removal, B3 removal, local sampling, query scaling, backbone).
- Identify which components contribute most.
- Fix any failure modes discovered.

### Weeks 11-12: Design ablations
- Tier 2 ablations (B2 depth, routing budget, A3 depth, deformable budget, fusion depth).
- Tier 3 extensions if time permits (B2-only, hash encoding, temporal memory).

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
| B2 attention collapse (uniform over L4) | Low | Monitor B2 attention entropy; attention dropout ($p=0.1$). |
| Routing collapse (all queries select same anchors) | Low | Monitor $\bar{\alpha}_q$ entropy; if needed, add entropy regularizer $L_{\text{entropy}}$. |
| Offsets collapse to zero | Low | 0.1× LR on offset heads (Deformable DETR). |
| Speed worse than dense | Very Low | ~26.5M FLOPs/query (~2,150× < F^3 ds2). Crossover at $K \approx 3{,}380$. |
| L4 self-attn latency too high | Very Low | ~0.1 ms for 2 layers over 880 tokens (integrated into backbone). Ablation #9 tests 0-4 layers. |
| Backbone latency too high | Very Low | ~1.2 ms amortized (21% of precompute). Ablation #29 tests lighter configs. |
| Swin window boundary artifacts at L3 | Low | Shifted windows (alternating blocks). L4 full self-attention covers all 880 positions globally. |
| Multi-scale tokens ignored in B4 | Low | Monitor B4 attention weights on $c_q^{(2:4)}$. Ablation #22. |
| B2 cross-attn makes B3 redundant | Low | Ablation #2 (remove B3) tests this. If so, simplify to B2-only decoder. |
| STE routing instability | Medium | Start without routing; add after convergence. Attention-based routing from 880 L4 tokens has smoother gradients than v8's learned routing. |
| Pseudo depth label noise | Medium | Two-stage: pseudo → metric LiDAR fine-tune. |

---

## 12. RGB-SPD Variant (Extension)

RGB-SPD applies the same sparse query-point decoder (B1–B5) to standard RGB images by replacing F^3 with a lightweight conv stem and deepening the backbone where features are cheapest. **All decoder stages (B1–B5) are identical to EventSPD.** Only the encoder (Phase A) changes.

### 12.1 Differences from EventSPD

**Replace F^3 with RGB stem (A1):**

| | EventSPD | RGB-SPD |
|---|---|---|
| Input | Raw events (1280×720, 20 ms window) | RGB image (1280×720×3) |
| A1 | F^3 ds2 → 640×360×64 (~57G, 4.5 ms) | Conv(3→64, k7,s2) + LN + GELU → 640×360×64 (~4.3G, 0.3 ms) |

**Enrich L1 features (new):**

F^3 output has passed through 6 ConvNeXt blocks with ~37×37 receptive field. A single conv produces only edge/color filters (7×7 RF). Add 2 lightweight conv layers at L1 to build texture features for B1 local sampling:

$$
\text{L1}: \quad F_{\text{stem}} \xrightarrow{2 \times [\text{Conv}(64 \to 64, k3, \text{pad1}) + \text{LN} + \text{GELU}]} F_t^{(1)} \quad [640 \times 360 \times 64]
$$

Effective RF after stem + 2 convs: 13×13. Cost: ~3.2G FLOPs, ~0.2 ms, ~74K params.

**Deepen stride-4 intermediate (A2 modified):**

EventSPD's stride-4 intermediate is a single conv passthrough (Stem-1). For RGB, this level must build mid-level features that flow into L2→L3→L4 and into B1's intermediate local sampling. Add 3 ConvNeXt$_{64}$ blocks:

```
                                         EventSPD                          RGB-SPD
Stem-1: Conv(64→64, k3,s2) + LN         320×180×64  (passthrough)         320×180×64
                                         —                                 → 3× ConvNeXt_64 (with GRN)
Stem-2: Conv(64→128, k2,s2) + LN        160×90×128                        160×90×128
```

Cost: ~15.6G FLOPs, ~1.0 ms, ~110K params. This partially replaces F^3's 6 ConvNeXt blocks (which operated at 640×360) at 4× fewer positions.

**Deepen L4 self-attention (A2 modified):**

With weaker input features, L4 needs more global reasoning. Increase from 2 to 4 full self-attention layers:

| | EventSPD | RGB-SPD |
|---|---|---|
| L4 self-attn layers | 2 | 4 |
| L4 FLOPs | ~0.6G | ~1.2G |
| L4 params | ~3,542K | ~7,084K |
| L4 wall-clock | ~0.1 ms | ~0.15 ms |

Cost increase: ~0.6G FLOPs, ~0.05 ms — negligible at 880 tokens.

**Everything else is identical:** L2 (2× ConvNeXt$_{128}$), L3 (4× Swin$_{192}$), A3 projections, A4 calibration, B1–B5 decoder.

### 12.2 Full RGB-SPD backbone diagram

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

### 12.3 Cost comparison

| | EventSPD | RGB-SPD | Δ |
|---|:---:|:---:|:---:|
| **A1 (F^3 / RGB stem)** | 57G / 4.5 ms | 4.3G / 0.3 ms | −53G / −4.2 ms |
| **L1 enrichment** | — | 3.2G / 0.2 ms | +3.2G / +0.2 ms |
| **Intermediate ConvNeXt** | — | 15.6G / 1.0 ms | +15.6G / +1.0 ms |
| **Backbone (L2–L4)** | 16.1G / 1.2 ms | 16.7G / 1.25 ms | +0.6G / +0.05 ms |
| **Encoder total** | ~73G / 5.7 ms | ~40G / 2.8 ms | −33G / −2.9 ms |
| **Encoder params** | ~6.4M + F^3 | ~10.2M | — |
| **Decoder (K=256)** | 6.8G / 1.6 ms | 6.8G / 1.6 ms | identical |
| **System total (K=256)** | **~80G / 7.4 ms** | **~47G / 4.4 ms** | **−33G / −3.0 ms** |

Nonlinear depth per query: L1(3) + Int(6) + Stem-2(1) + L2(4) + L3(8) + L4(4) + decoder(14) = **~40 stages** (vs EventSPD ~31, DAv2 ~32).

$$
T_{\text{RGB-SPD}}(K) \approx 3.1 + 0.005K \text{ ms}
$$

| K | Total (ms) | Hz | vs dense (23 ms) |
|---|:---:|:---:|:---:|
| 1 | 3.1 | 323 | 7.4× |
| 64 | 3.4 | 294 | 6.8× |
| 256 | 4.4 | 227 | 5.2× |
| 1024 | 8.2 | 122 | 2.8× |

Crossover vs dense: $(23 - 3.1) / 0.005 \approx K = 3{,}980$.

### 12.4 Why this works without ImageNet pretraining

The sparse decoder (B1–B5) does 14 nonlinear stages of per-query reasoning — cross-attention, deformable sampling, multi-head fusion. Dense decoders (DPT) do 4–8 stages of per-pixel conv upsampling. Our decoder compensates for a shallower encoder by concentrating compute on each query. The total per-query depth (~40 stages) exceeds DAv2 (~32) despite having a much lighter encoder.

Training: end-to-end from random init (same as EventSPD — F^3 features aren't pretrained on depth either). The RGB stem + intermediate ConvNeXt learns general visual features jointly with the depth objective.

### 12.5 Scope

RGB-SPD is an **extension / second contribution**, not the primary paper. Priority: demonstrate EventSPD on event cameras first. If successful, RGB-SPD shows the sparse decoder architecture generalizes beyond events — strengthening the claim that the decoder design (not F^3) is the core contribution.

---

## 13. References

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
