# Research Plan: Streamlined EventSPD — Sparse Query-Point Depth from Events

Author: Claude (redesigned from Codex v4, audited and revised v5)
Date: 2026-02-12 to 2026-02-15

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
| $\mathcal{F}_{\text{F}^3}$ | Backbone network | Event-to-feature encoder | Frozen or fine-tuned |
| $F_t$ | $\mathcal{F}_{\text{F}^3}(E_t)$ | Dense shared feature field | Via backbone |
| $F_t^{(1)}, F_t^{(2)}, F_t^{(3)}, F_t^{(4)}$ | Independent AvgPool + $1\times1$ Proj from $F_t$ | Fine / mid / mid-coarse / coarse features | Projection layers: Yes |
| $s_\ell$ | $s_1{=}1, s_2{=}2, s_3{=}4, s_4{=}8$ | Stride of pyramid level $\ell$ relative to $F_t$ | No |
| $C_t \in \mathbb{R}^{P_c \times d}$ | $\text{SpatialGridPool}(F_t^{(3)})$ | Compact coarse latent bank | Projection: Yes |
| $s_t, b_t$ | $\text{Heads}(\text{Pool}(C_t))$ | Global depth scale / shift | Yes |
| $q = (u, v)$ | User input | Query pixel coordinate | No |
| $f_q^{(\ell)}$ | $\text{Bilinear}(F_t^{(\ell)}, q)$ | Point feature at scale $\ell$ | No |
| $l_q$ | MaxPool(LocalAggregate) from $F_t^{(1)}$ | Neighborhood context around $q$ | MLP: Yes |
| $\text{pe}_q$ | $\text{Fourier}(u/W, v/H)$ | Positional encoding | No |
| $h_{\text{point}}$ | $\text{MLP}([f_q^{(1:4)}; \text{pe}_q])$ | Center token (multi-scale point identity) | Yes |
| $z_q$ | $\text{MLP}([f_q^{(1)}; l_q; \text{pe}_q])$ | Routing / retrieval token | Yes |
| $\bar{c}_q$ | $\text{MHCrossAttn}(z_q, C_t)$ | Query-conditioned global summary | Yes |
| $\alpha_q$ | $\text{softmax}(W_r z_q \cdot C_t^\top / \sqrt{d})$ | Routing scores over latent bank | Yes |
| $R_q$ | $\text{TopR}(\alpha_q)$ | Content-routed coarse tokens | No (selection) |
| $S_q$ | $R_q \cup \{i_q^{\text{loc}}\}$ | Anchor set for deformable reads | No |
| $h_{r}$ | DeformRead per anchor $r \in S_q$ | Per-anchor spatial evidence | Offset/weight heads: Yes |
| $T_q$ | $[l_q + e_{\text{loc}}; \; h_{r_1} + e_{\text{near}}; \; h_{r_2..33} + e_{\text{route}}; \; \bar{c}_q + e_{\text{glob}}]$ | 35 context tokens with type embeddings | No (assembly) |
| $e_{\text{loc}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}}$ | Learned | Type embeddings for context token roles ($\in \mathbb{R}^d$) | Yes |
| $h_{\text{fuse}}$ | TransformerDec($h_{\text{point}}, T_q$) | Fused query representation | Yes |
| $r_q$ | $\text{MLP}(h_{\text{fuse}})$ | Relative depth code | Yes |
| $\rho_q$ | $s_t \cdot r_q + b_t$ | Calibrated inverse depth | No (algebraic) |
| $\hat{d}_q$ | $1 / (\text{softplus}(\rho_q) + \varepsilon)$ | Final predicted depth | No (conversion) |
| $\sigma_q$ | $\text{softplus}(\text{Linear}(h_{\text{fuse}})) + \sigma_{\min}$ | Uncertainty estimate (optional) | Yes |

Core dimension: $d = 128$. Positional encoding: $d_{\text{pe}} = 32$ (8 Fourier frequencies × 2 trig functions × 2 spatial dims).

### 4.3 Algorithm A: Precompute Once Per Event Window

**Input:** Event set $E_t$.
**Output:** $\text{cache}_t = \{F_t^{(1)}, F_t^{(2)}, F_t^{(3)}, F_t^{(4)}, C_t, s_t, b_t\}$.

#### A1. Backbone encoding

$$
F_t = \mathcal{F}_{\text{F}^3}(E_t)
$$

Shared event-derived feature field for the entire window. This is the dominant cost (~8.3 ms at 120 Hz HD on RTX 4090).

#### A2. Feature pyramid

$$
F_t^{(1)} = \text{Proj}_{1 \times 1}(F_t), \quad
F_t^{(2)} = \text{Proj}_{1 \times 1}(\text{AvgPool}_{2 \times 2}(F_t))，
$$
$$
F_t^{(3)} = \text{Proj}_{1 \times 1}(\text{AvgPool}_{4 \times 4}(F_t)), \quad
F_t^{(4)} = \text{Proj}_{1 \times 1}(\text{AvgPool}_{8 \times 8}(F_t))
$$

Each level is built **independently from the raw backbone output** $F_t$: pool to the target resolution, then project $32 \to d = 128$ via a $1 \times 1$ convolution. This avoids cascaded error accumulation where quantization artifacts at level 2 propagate to levels 3 and 4, following the ViTDet (Li et al., 2022) Simple Feature Pyramid principle of building all FPN levels from a single backbone feature map using independent operations.

Four scales at strides 1×, 2×, 4×, 8× relative to $F_t$. For HD input (1280×720): $F_t^{(1)}$ is 1280×720, $F_t^{(2)}$ is 640×360, $F_t^{(3)}$ is 320×180, $F_t^{(4)}$ is 160×90. Each $\text{Proj}_{1 \times 1}$ maps backbone channels (32) to $d = 128$.

**Why 4 levels:** DPT (the gold standard for ViT-based depth) uses 4 levels and shows that removing the coarsest causes the largest accuracy drop — global scale and scene layout information is lost. Our coarsest level (160×90) has 14,400 positions, giving each feature effective coverage of ~8×8 original pixels. Combined with the backbone's 63×63 receptive field, level-4 features summarize ~70×70 pixel regions — genuinely "coarse" context.

**Why independent projections over cascaded pooling:** Cascaded pooling (level $\ell+1$ = Pool(Proj(level $\ell$))) creates a serial dependency where any projection error at level 2 is inherited by levels 3 and 4. Independent pooling from raw $F_t$ ensures each level starts from the same unprocessed 32-channel features. With only 32 backbone channels (vs. 96–768 in Swin Transformer's cascaded PatchMerging), each channel carries more information and is more sensitive to projection noise. If ablations show that strided convolutions (learned downsampling) help over average pooling, they can replace AvgPool at levels 2–4.

#### A3. Compact latent bank (spatial grid pooling)

Divide $F_t^{(3)}$ into a $P_h \times P_w$ spatial grid. Each cell is adaptive-average-pooled and projected:

$$
C_t = \text{Proj}(\text{AdaptiveAvgPool}_{P_h \times P_w}(F_t^{(3)})) \in \mathbb{R}^{P_c \times d}
$$

Default: $P_h = 16, P_w = 32, P_c = 512$.

For HD input: $F_t^{(3)}$ is $320 \times 180$, so each cell covers $10 \times 11.25$ level-3 positions ($\approx 40 \times 45$ original pixels). Compression ratio: $57{,}600 / 512 \approx 112 : 1$.

Each token $c_i$ has:
- **Content**: pooled features from its spatial cell.
- **Location**: the known center coordinate $\mathbf{p}_i$ of its cell in pixel coordinates (not learned — deterministic from grid geometry). We write $\mathbf{p}_i^{(\ell)} = \mathbf{p}_i / s_\ell$ for the position in level-$\ell$ native coordinates.

**Why pool from $F_t^{(3)}$ (mid-coarse) instead of $F_t^{(2)}$ (mid):**
- The latent bank represents a **coarse global description** of the scene. Level-3 features already carry broader context per position, so the adaptive-average-pool preserves more useful information per cell.
- Pooling from $F_t^{(2)}$ (640×360) with $P_c = 512$ gives ~450:1 compression — each cell averages 450 features, losing significant per-cell fidelity. Pooling from $F_t^{(3)}$ (320×180) gives ~112:1, a 4× improvement.
- Fine-grained spatial detail is recovered via direct bilinear lookups (B1) and deformable sampling (B3), not from the latent bank.
- Anchor positions in original-resolution coordinates are identical regardless of source level (32×16 grid cells map to the same physical locations).

**Why $P_c = 512$:**
- Each F^3 backbone feature has a 63×63 pixel receptive field. With $P_c = 512$ (32×16 grid), each cell covers ~40×45 original pixels — comparable to the receptive field size, preserving discriminative content. Going finer ($P_c = 1024$, 32×32) risks top-R anchor clustering: large uniform regions (road, sky) produce many cells with near-identical content, and top-R over-selects from the same region, wasting anchor budget. With $P_c = 512$, each region produces fewer candidate cells, naturally promoting spatial diversity in anchor selection.
- Routing precision: $R = 32$ out of 512 (6.25%) gives generous anchor coverage for complex outdoor scenes (forest, city roads) where depth structure spans many distinct regions (10-20+ depth layers). Still within MoE sparsity norms (DeepSeek-V3 routes 8/256 = 3.1%), while keeping the candidate pool concentrated enough to avoid clustering.
- Cross-attention cost over 512 tokens remains trivial ($1 \times 512$ attention matrix per head, ~130K ops per query — negligible vs B3's 2,376 grid_sample calls).
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

**Multi-scale center features:**

$$
f_q^{(\ell)} = \text{Bilinear}(F_t^{(\ell)}, \text{Normalize}(q / s_\ell)), \quad \ell = 1, 2, 3, 4
$$

Four feature vectors capturing the query point at fine (level 1), mid (level 2), mid-coarse (level 3), and coarse (level 4) scales.

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

**Center token (multi-scale point identity):**
$$
h_{\text{point}} = \text{LN}(W_{p2} \cdot \text{GELU}(W_{p1} [f_q^{(1)}; f_q^{(2)}; f_q^{(3)}; f_q^{(4)}; \text{pe}_q] + b_{p1}) + b_{p2})
$$

A 2-layer MLP with GELU activation and **no skip connection**. Input dimension: $4d + d_{\text{pe}} = 544$. Hidden dimension: $d = 128$. Output: $h_{\text{point}} \in \mathbb{R}^d$. Parameters: $544 \times 128 + 128 + 128 \times 128 + 128 \approx 86\text{K}$.

Ablate with hidden=256 (~172K params): the MobileNetV2 principle suggests wider intermediates preserve information through nonlinearities; PointNet++ FP1 uses 1280→256 (5:1) compression. However, start with the simpler hidden=128 default and validate whether the bottleneck is a limiting factor.

**Why no skip connection here:** A skip connection would introduce a direct linear path from raw features to the depth head (through the residual stream of the fusion decoder). Since depth is a fundamentally nonlinear function of appearance features, we want $h_{\text{point}}$ to be a fully nonlinear encoding. This matches the pattern in both DepthAnythingV2 and F^3's own `EventPixelFF`, where prediction-facing layers are always fully nonlinear with no additive identity path.

**Why 4-scale center features:** Depth at a point depends on fine details (edges at level 1), medium context (structural elements at levels 2-3), and global context (scene layout at level 4). Including all four scales makes the center token informative on its own, reducing dependency on the global summary for coarse depth estimation. DPT shows that each pyramid level contributes non-redundant information to depth prediction.

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
i_q^{\text{loc}} = \arg\min_{i \in \{1, \ldots, P_c\}} \|\mathbf{p}_i^{(3)} - q^{(3)}\|_2^2
$$

where $q^{(3)} = (u/s_3, v/s_3)$ is the query in level-3 coordinates and $\mathbf{p}_i^{(3)} = \mathbf{p}_i / s_3$ is the grid center of token $i$ in level-3 coordinates.

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

**Why a separate encoding from B1.2:** B1.2's $\phi(\delta)$ uses 4 frequencies on raw pixel offsets $|\delta| \leq 6$, which is well-matched to that small range. B3's offsets span the full image ($|\Delta\mathbf{p}_r|$ up to 1280 pixels). Applying B1.2's raw-pixel frequencies to such large offsets would produce meaningless high-frequency oscillations ($\sin(2\pi \cdot 8 \cdot 1280) = \sin(64{,}339)$). Normalizing by image dimensions maps the offset to $[-1, 1]$, matching the standard practice from NeRF (Mildenhall et al., 2020), Tancik et al. (2020), and DAB-DETR (Liu et al., 2022) — none of which apply Fourier/sinusoidal encoding to raw pixel coordinates. With 8 frequencies, the finest resolved period is $\sim 5$ pixels, comfortably finer than the $\sim 20$-pixel latent grid cell spacing at level-3.

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
self.sampling_offsets = nn.Linear(d, H * L * M * 2)   # -> 8*3*3*2 = 144
self.attention_weights = nn.Linear(d, H * L * M)       # -> 8*3*3   = 72
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

where $W_V^{(h)} \in \mathbb{R}^{(d/H) \times d}$ is the value projection for head $h$ (in practice, a single $W_V \in \mathbb{R}^{d \times d}$ with output reshaped to $H = 8$ heads of $d/H = 16$ dims). **Implementation optimization:** Following Deformable DETR, $W_V$ is pre-applied to the feature maps $F_t^{(2:4)}$ once per frame during precompute (~0.06 ms), so grid_sample reads already-projected 128-ch features. This eliminates the per-sample projection cost from the per-query path.

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

**Default budget:** $H = 8$ heads, $L = 3$ levels (2–4), $M = 3$ samples.
- Per anchor: $8 \times 3 \times 3 = 72$ samples.
- 33 anchors: $33 \times 72 = 2{,}376$ deformable lookups total.

**Why $H = 8$ heads:** Every major deformable attention method (Deformable DETR, Mask2Former, DAB-DETR) uses 8 heads. $H = 8$ matches the published standard, providing maximum sampling diversity — 8 independent offset sets per anchor, each specializing in different spatial patterns (edges, textures, depth boundaries, occlusion). The parameter cost is unchanged ($W_V$ and $W_O$ remain 128×128).

**Why $L = 3$ levels (2–4, dropping level 1):** The deformable anchors gather **non-local** evidence from distant scene regions. At level 1 (stride 1×), offsets cover a tiny window around the anchor center — redundant with the anchor's content vector $c_r$ which already summarizes that region. Levels 2–4 provide progressively broader spatial reach, probing structure within and around the anchor's region at meaningful scales. Fine-grained local detail at the query point is already handled by B1's local sampling (32 samples from level 1). Note: the 4-level pyramid is still built for B1's center features ($f_q^{(1:4)}$); only B3's deformable sampling skips level 1.

**Why $M = 3$ samples per head per level:** $M = 3$ is the closest to Deformable DETR's default ($M = 4$) while being slightly more conservative. Each head samples 3 locations per level × 3 levels = 9 samples per anchor — a small but sufficient local attention window for capturing depth gradients, boundaries, and occlusion edges within each anchor region. Speed analysis confirms the decoder cost is dominated by the F^3 precompute (~95% of total time), so the per-anchor sample budget is essentially free in wall-clock time. The total deformable budget (2,376) is spread across 33 diverse anchor regions.

**Why per-anchor aggregation instead of a single global softmax:**
- Each anchor represents a distinct spatial region. Per-anchor normalization preserves the relative contribution of each region.
- The fusion transformer (B4) then learns how to weigh different regions via cross-attention.
- A single global softmax would allow one dominant region to suppress all others.

#### B4. Fusion decoder (2-layer cross-attention transformer)

**Context token set (with type embeddings):**

$$
T_q = [l_q + e_{\text{loc}}; \; h_{r_1} + e_{\text{near}}; \; h_{r_2} + e_{\text{route}}; \; \ldots; \; h_{r_{33}} + e_{\text{route}}; \; \bar{c}_q + e_{\text{glob}}]
$$

35 context tokens total, each with a learned type embedding:
- $l_q + e_{\text{loc}}$: aggregated local neighborhood (from B1).
- $h_{r_1} + e_{\text{near}}$: deformable evidence from the nearest anchor (medium-range context).
- $h_{r_2} + e_{\text{route}}, \ldots, h_{r_{33}} + e_{\text{route}}$: deformable evidence from 32 routed anchors (non-local context).
- $\bar{c}_q + e_{\text{glob}}$: compressed global scene summary (from B2).

**Type embeddings:**

Each context token receives a learned type embedding ($e_{\text{loc}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}} \in \mathbb{R}^d$) identifying its role.

where $e_{\text{loc}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}} \in \mathbb{R}^d$ are learned embeddings (4 × 128 = 512 parameters). This is standard practice in DETR-family architectures where different token roles (object queries, encoder tokens, positional queries) receive distinct learned embeddings. Without type embeddings, the cross-attention must infer token roles purely from content, requiring extra capacity.

**KV normalization:**

Before entering the transformer, apply a shared LayerNorm to the assembled context tokens:
$$
T_q \leftarrow \text{LN}_{\text{kv}}(T_q)
$$

This normalizes the heterogeneous token scales: $l_q$ (raw linear output), $h_{r_1..33}$ (multi-head deformable read output), and $\bar{c}_q$ (cross-attention output) come from different computational paths with potentially different magnitude distributions. Without normalization, attention logits ($Q \cdot K^\top$) can be dominated by whichever token type has the largest norm, creating systematic bias unrelated to content relevance. SAM's decoder applies separate LayerNorm to both queries and keys (via `norm2` and `norm3` in TwoWayAttentionBlock). Since $T_q$ is static (not updated across layers), this LN is applied once — essentially free.

**2-layer transformer decoder (standard Pre-LN):**

Each layer applies cross-attention with residual, then FFN with residual (standard Pre-LN transformer):
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}(Q = \text{LN}_q(h_{\text{point}}), \; KV = T_q)
$$
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}(\text{LN}(h_{\text{point}}))
$$

where $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ with expansion ratio 4 ($d \to 4d \to d$, i.e., $128 \to 512 \to 128$).

Cross-attention uses 4 heads with $d_{\text{head}} = 32$. Attention matrix per head is $1 \times 35$ — trivially cheap.

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
- If the query is at a depth edge, the model can upweight $l_q$ (local context) and $h_{r_1}$ (nearest anchor).
- If the query is in a textureless region, the model can upweight $\bar{c}_q$ (global summary) and $h_{r_2..33}$ (non-local evidence).
- The type embeddings help the attention heads distinguish token roles without relying on content differences alone.
- The residual connections preserve $h_{\text{point}}$'s identity through both sub-layers, while cross-attention and FFN add contextual and nonlinear refinements.

This replaces the original plan's complex bounded-carry gates, explicit BCR/NUR monitoring, and balance losses. The transformer learns the right balance from data. If center collapse is observed, the center auxiliary loss $L_{\text{ctr}}$ addresses it directly.

**Why 2 layers:** SAM's decoder uses 2 layers and achieves excellent point-prompt results. Our context set is small (35 tokens vs thousands of image tokens in SAM), so 2 layers provide sufficient mixing depth. Ablate with 1 and 3 layers.

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
| `M1_Precompute` | A1–A4 | $E_t \to F_t^{(1:4)}, C_t, s_t, b_t$ |
| `M2_QueryEncode` | B1+B2 | $q \to h_{\text{point}}, z_q, l_q, \bar{c}_q, S_q$ |
| `M3_DeformRead` | B3 | $S_q, z_q, C_t, F_t^{(2:4)} \to h_{r_1}, \ldots, h_{r_{33}}$ |
| `M4_FuseDecode` | B4+B5 | $h_{\text{point}}, T_q, s_t, b_t \to \hat{d}_q$ |

---

### 4.6 Why This Design Will Work — Confidence Arguments

1. **The precompute-then-query paradigm is proven.** SAM does exactly this for segmentation. Perceiver IO does it for arbitrary structured outputs. Both achieve state-of-the-art results with lightweight decoders on precomputed features.

2. **F^3 features contain sufficient depth information.** F^3 already achieves dense depth comparable to DepthAnythingV2. We use the same features — only the decoder changes.

3. **Local + global context is standard in depth estimation.** MiDaS, DPT, ZoeDepth, and Metric3D all use multi-scale features capturing both local and global context. Our three-stream design (center + local + global) replicates this in a sparse decoder.

4. **Deformable attention is battle-tested.** Used in Deformable DETR, Mask2Former, DAB-DETR, and many more. Learned offsets + importance weights for spatial sampling is robust and well-understood.

5. **The speed advantage is structural and massive.** Dense decoders process $HW = 921{,}600$ pixels. Our decoder costs ~11.4M FLOPs per query (~5,700× less than one F^3 backbone pass of ~65 GFLOPs). Bottom-up analysis shows precompute (8.8 ms) accounts for ~87% of total time at $K=256$; the decoder adds only ~1.3 ms. The conservative crossover is at $K \approx 3{,}500$ — far beyond any practical query count. This means **sampling budget choices (R, M, L) are free in wall-clock time** and should be driven purely by accuracy.

6. **Non-linearity is enforced at every stage.** Center MLP: GELU without skip connection (raw features → fully nonlinear $h_{\text{point}}$). Local aggregate: per-sample GELU + max pooling. Cross-attention: softmax (nonlinear). FFN: GELU expansion with standard residual. Depth head: 2-layer GELU MLP + softplus conversion. While standard residual connections exist in the fusion transformer, the input to those residuals ($h_{\text{point}}$) is already a nonlinear transform of raw features. Multiple nonlinear stages (center MLP → cross-attention softmax → FFN GELU → depth MLP GELU → softplus) ensure sufficient representational capacity.

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

### 6.1 Per-Query Complexity

| Component | Feature lookups | Attention ops | Parameters |
|-----------|----------------|---------------|------------|
| A2: Pyramid projections | — | — | ~17K (4 × Proj 32→128) |
| A3: Latent bank | — | — | ~17K (Proj) |
| B1: Multi-scale center | 4 | — | ~86K (center MLP: 544→128→128) |
| B1: Local sampling | 32 | — | ~37K (local MLP + offsets) |
| B1: Routing token | — | — | ~37K (routing MLP) |
| B2: Global summary | — | $P_c = 512$ | ~65K (cross-attn) |
| B2: Routing scores | — | $P_c = 512$ | ~33K (routing proj) |
| B3: Deformable read | 2,376 | — | ~98K (conditioning 288→128 + offset/weight heads 128→144/72 + $W_V$/$W_O$) |
| B4: Type embeddings | — | — | ~0.5K (4 × $d$) |
| B4: KV normalization | — | — | ~0.3K ($\text{LN}_{\text{kv}}$) |
| B4: Fusion (2 layers) | — | $35 \times 2$ | ~400K (transformer) |
| B5: Depth head | — | — | ~17K (MLP 128→128→1) |
| B5: $L_{\text{ctr}}$ auxiliary (train) | — | — | ~8K (2-layer MLP: 128→64→1) |
| **Total** | **2,412** | **~1094** | **~816K** |

For comparison: DepthAnythingV2-Small has ~25M decoder parameters. Our sparse decoder is **31× smaller**. The accuracy-first budget of 2,412 lookups per query is still 382× less than dense decoding (921,600 pixels). The ~44K increase over v4 (772K) comes from: full multi-head deformable structure with $W_V$/$W_O$ (+33K), expanded B3 offset/weight heads for M=3 (+9K), expanded B3 conditioning input (+2K), KV normalization (+0.3K). Speed analysis confirms the decoder cost is dominated by F^3 precompute (~87% of total time), so the generous sampling budget (R=32, M=3) is essentially free in wall-clock time.

### 6.2 Speed Estimate (RTX 4090, 1280×720)

**Precompute (once per window):**
$$
T_{\text{precompute}} = T_{\text{F}^3} + T_{\text{pyramid+pool+cal+VProj}} \approx 8.33 + 0.40 + 0.06 = 8.79 \text{ ms}
$$

The 4-level pyramid adds ~0.05 ms over the 3-level version (one extra AvgPool_2×2 + 1×1 Proj over 57,600 features). The 32×16 latent bank pool from $F_t^{(3)}$ is negligible. Following the Deformable DETR implementation pattern, B3's $W_V$ value projection is pre-applied to the feature maps $F_t^{(2:4)}$ once per frame (~0.06 ms), so grid_sample reads already-projected features during per-query decoding.

**Per-query cost analysis:**

The decoder cost has two components: a **fixed overhead** (kernel launches and PyTorch dispatch, independent of $K$) and a **marginal cost** per query (compute + memory bandwidth, scaling with $K$):
$$
T_{\text{decoder}}(K) = \alpha + \beta K
$$

Per-query FLOPs breakdown (~11.4M FLOPs/query with pre-projected $W_V$):

| Stage | Dominant operations | FLOPs/query |
|-------|-------------------|---:|
| B1 (center + local + routing) | MLPs, 36 grid_samples | ~321K |
| B2 (summary + routing) | Cross-attn over 512 tokens, scoring | ~528K |
| B3 (deformable) | 33× conditioning MLP(288→128), offsets, $W_O$, 2,376 grid_samples | ~5,400K |
| B4 (fusion, 2 layers) | KV proj on 35 tokens, FFN(128→512→128) | ~5,150K |
| B5 (depth head) | MLP(128→128→1) | ~33K |
| **Total** | | **~11.4M** |

For reference, the F^3 backbone (2-stage ConvNeXtV2, 6 blocks at full 1280×720, 32 channels) processes ~65 GFLOPs per frame in 8.33 ms. Our decoder at 11.4M FLOPs/query is **~5,700× less compute** than one backbone pass.

**Three cost components:**

1. **Fixed overhead** ($\alpha$): ~30–40 sequential CUDA kernel launches × ~5–10 μs each = 0.20–0.35 ms per batch. Independent of $K$ — same number of kernel launches whether processing 1 or 1024 queries.
2. **Compute** ($\beta_{\text{compute}}$): Matrix multiplications (B3 conditioning, B4 projections/FFN). GPU utilization scales with $K$: ~2% at $K=1$, ~25% at $K=256$, ~40% at $K=1024$.
3. **Memory** ($\beta_{\text{mem}}$): grid_sample reads from feature maps (bilinear, 4 neighbors × 128 channels per sample). Level 4 (3.7 MB) fits in L2 cache (72 MB); level 2 (59 MB) mostly misses. Per-query memory: ~1 KB per sample × 2,412 samples ≈ 2.4 MB. Each read is channel-contiguous (128 channels stored sequentially), so the "random" part is only the 2D spatial index — actual data transfer is coalesced.

**Conservative estimates** (standard PyTorch, no custom CUDA, no torch.compile):
- $\alpha \approx 0.30$ ms (fixed kernel launch + dispatch overhead).
- $\beta \approx 0.004$ ms/query (combined compute + memory bandwidth at typical utilization; increased from v5-R24 due to 2,376 grid_samples and 35 context tokens).

**Total:**
$$
T_{\text{EventSPD}}(K) \approx 9.09 + 0.004K \text{ ms}
$$

| Query count $K$ | Decoder (ms) | EventSPD total (ms) | Throughput (Hz) | Dense baseline (ms) | Speedup |
|-----|---:|---:|---:|---:|---:|
| 1 | 0.30 | 9.1 | 110 | ~23 | 2.5× |
| 64 | 0.56 | 9.3 | 107 | ~23 | 2.5× |
| 256 | 1.32 | 10.1 | 99 | ~23 | 2.3× |
| 1024 | 4.40 | 13.2 | 76 | ~23 | 1.7× |

Dense baseline: F^3 (8.3 ms) + DepthAnythingV2 decoder (~15 ms) = ~23 ms. The crossover point where EventSPD matches dense cost: $(23 - 9.09) / 0.004 \approx K = 3{,}500$ — far beyond any practical query count. Even at $K = 1024$, EventSPD is **1.7× faster** than dense.

**Why the decoder is so cheap:** The precompute (8.8 ms, dominated by F^3 backbone) accounts for ~87% of total time at $K = 256$. The decoder adds only ~1.3 ms. This means **sampling budget decisions should be driven purely by accuracy, not speed** — even pushing to R=64, M=4 (~4,700 lookups) would add only ~1 ms beyond the current budget, keeping the total under 11.5 ms (2.0× faster than dense).

**Key insight:** The $\alpha + \beta K$ model reveals that the decoder's fixed overhead (~0.3 ms kernel launches) dominates for small $K$, while the marginal cost (~0.004 ms/query) is negligible because: (a) the 11.4M FLOPs/query is tiny for an RTX 4090 GPU, (b) all $K$ queries are batched into a single forward pass, (c) $W_V$ is pre-applied to feature maps, eliminating the costliest per-sample operation, and (d) grid_sample memory reads are channel-contiguous (128 channels per burst), with smaller pyramid levels fitting in L2 cache.

**Design philosophy:** The decoder cost is dominated by the precompute phase, not the per-query sampling budget. This validates the accuracy-first approach: R=32, M=3, H=8 are essentially free in wall-clock time. Once accuracy is validated, ablation targets should focus on **parameter efficiency and training speed** (fewer parameters = faster convergence), not inference speed.

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
- Average cross-attention weight per context token type ($l_q$, $h_{\text{near}}$, $h_{\text{route}}$, $\bar{c}_q$) across queries. Shows whether the model uses all information streams.

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
| 6 | No routing (attend to all $C_t$) | Does sparse routing help efficiency without hurting accuracy? | Remove top-R, use all 512 tokens as deformable anchors (512 × 72 = 36,864 grid_sample calls — much slower but tests routing necessity). |
| 7 | Residual design | Does asymmetric residual help or hurt? | Compare: (a) standard both residuals (current), (b) drop FFN residual (asymmetric), (c) no residuals, (d) add center MLP skip connection. |
| 8 | Fusion depth | How many transformer layers? | 1, 2, 3 layers. |
| 9 | Latent bank size | How many coarse tokens? | $P_c \in \{128, 256, 512, 1024\}$. Current default: $P_c = 512$. Also ablate source level: $F_t^{(2)}$ vs $F_t^{(3)}$. |
| 10 | Routing budget | How many routed tokens? | $R \in \{8, 16, 32, 64, 128\}$. Current default: $R = 32$. |
| 11 | Local budget | How many local samples? | $N_{\text{loc}} \in \{8, 16, 32, 48, 64\}$. |
| 12 | Deformable budget | How many samples per anchor? | $(H, L, M) \in \{(4,3,2), (8,2,2), (8,3,2), (8,3,3), (8,3,4)\}$. Current default: $(8,3,3)$. |

### Tier 3 — Extensions and alternatives

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 13 | Perceiver-style LatentPool | Do learned queries outperform spatial grid? | Replace A3 with MHA-based LatentPool. |
| 14 | Hash encoding vs Fourier PE | Does learned position encoding help? | Replace $\text{pe}_q$ with Instant-NGP hash. |
| 15 | Temporal memory | Does GRU state across windows help? | Add $H_t$ to cache and to $T_q$. |
| 16 | Uncertainty head | Does uncertainty improve hard queries? | Enable $\sigma_q$ + $L_{\text{unc}}$. |
| 17 | Center auxiliary loss | Does $L_{\text{ctr}}$ prevent center collapse? | Disable $L_{\text{ctr}}$ and compare. |
| 18 | Attention dropout rate | What rate is best for cross-attention regularization? | $p_{\text{attn}} \in \{0, 0.05, 0.1, 0.2\}$. |

### Tier 4 — Formula audit ablations (v5 changes)

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 19 | Center MLP bottleneck width | Does wider hidden dim help multi-scale fusion? | Hidden $\in \{128, 256\}$. 128 = current default (~86K), 256 = wider variant (~172K). MobileNetV2 principle: wider intermediates preserve info through GELU nonlinearity. |
| 20 | Offset bounding method | Does tanh bounding limit offset diversity? | (a) $\tanh \cdot \kappa$ (current, DAT), (b) softsign $\cdot \kappa$ (slower saturation), (c) coordinate clamp $[-\kappa, \kappa]$ (DAT++ finding: unbounded ≈ bounded). Monitor pre-activation distribution. |
| 21 | Simplified vs full multi-head deformable | Does $W_V$/$W_O$ structure improve over weighted-sum? | (a) Full multi-head with per-head $W_V$, per-head softmax, concat, $W_O$ (current, ~89K B3). (b) Simplified: direct weighted sum of raw features, joint softmax (v4, ~54K B3). |
| 22 | KV normalization | Does $\text{LN}_{\text{kv}}$ on heterogeneous tokens help? | (a) With $\text{LN}_{\text{kv}}$ (current). (b) Without. Expect larger effect when token types have diverse magnitudes. |
| 23 | Auxiliary calibration | Should $r_q^{\text{ctr}}$ be calibrated by $(s_t, b_t)$? | (a) Calibrated: $\rho_q^{\text{ctr}} = s_t \cdot r_q^{\text{ctr}} + b_t$ (current). (b) Uncalibrated: direct Huber on raw $r_q^{\text{ctr}}$ vs $\rho^*$. |

---

## 9. Implementation Structure

### 9.1 New files

```
src/f3/tasks/depth_sparse/
├── models/
│   ├── gcqd.py                # Main model: Algorithm A + B pipeline
│   ├── feature_pyramid.py     # A2: pyramid construction
│   ├── latent_bank.py         # A3: spatial grid pool + calibration heads
│   ├── query_encoder.py       # B1: feature extraction + token construction
│   ├── context_retrieval.py   # B2: cross-attention summary + routing
│   ├── deformable_read.py     # B3: offset prediction + multiscale sampling
│   ├── fusion_decoder.py      # B4: 2-layer cross-attention transformer
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
| 2 | Minimal query decoder (bilinear + MLP, no global) | Concept viability | 1 week |
| 3 | Add latent bank + cross-attention summary | Global context benefit | 1 week |
| 4 | Add deformable sampling (small budget) | Precise non-local evidence | 1-2 weeks |
| 5 | Add routing (top-R) | Efficiency of sparse reads | 1 week |
| 6 | Full EventSPD pipeline + training | Complete system | 2 weeks |
| 7 | Ablations + paper figures | Publication readiness | 3-4 weeks |

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
|------|-----------|------------|
| Accuracy drop vs dense at depth edges | Medium | Increase local sampling budget; add edge-focused query sampling during training. |
| Router collapse (always same tokens) | Low | Monitor routing entropy; add entropy regularizer if needed. |
| Deformable offsets collapse to zero | Low | Use 0.1× learning rate on offset heads (following Deformable DETR). |
| Speed not better than dense for medium $K$ | Very Low | Bottom-up FLOPs analysis shows ~11.4M FLOPs/query (with $W_V$ pre-projection) — ~5,700× less than one F^3 pass (~65 GFLOPs). Precompute dominates total time (~87% at $K=256$). Conservative crossover at $K \approx 3{,}500$. Even at $K = 1024$, EventSPD is 1.7× faster than dense. |
| Training instability from STE routing | Medium | Start with no routing (attend to all $C_t$), add routing after model converges. |
| Label noise from pseudo depth | Medium | Two-stage training: pseudo first, then metric LiDAR fine-tuning. Already validated in F^3's pipeline. |

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
- E2Depth: https://arxiv.org/abs/2010.08350
- RAM-Net: https://arxiv.org/abs/2102.09320
- Feature Pyramid Networks: https://arxiv.org/abs/1612.03144

**Architecture references:**
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
