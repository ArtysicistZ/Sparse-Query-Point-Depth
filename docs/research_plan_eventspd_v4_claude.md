# Research Plan: Streamlined EventSPD — Sparse Query-Point Depth from Events

Author: Claude (redesigned from Codex v4)
Date: 2026-02-12 to 2026-02-14

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
| $F_t^{(1)}, F_t^{(2)}, F_t^{(3)}, F_t^{(4)}$ | Independent strided conv + proj from $F_t$ | Fine / mid / mid-coarse / coarse features | Projection layers: Yes |
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
| $T_q$ | $[l_q + e_{\text{loc}}; \; h_{r_1} + e_{\text{near}}; \; \ldots; \; \bar{c}_q + e_{\text{glob}}]$ | Context tokens with type embeddings | No (assembly) |
| $e_{\text{loc}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}}$ | Learned | Type embeddings for context token roles ($\in \mathbb{R}^d$) | Yes |
| $h_{\text{fuse}}$ | TransformerDec($h_{\text{point}}, T_q$) | Fused query representation | Yes |
| $r_q$ | $\text{MLP}(h_{\text{fuse}})$ | Relative disparity code | Yes |
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

Default: $P_h = 16, P_w = 16, P_c = 256$.

For HD input: $F_t^{(3)}$ is $320 \times 180$, so each cell covers $20 \times 11.25$ level-3 positions ($\approx 80 \times 45$ original pixels). Compression ratio: $57{,}600 / 256 = 225 : 1$.

Each token $c_i$ has:
- **Content**: pooled features from its spatial cell.
- **Location**: the known center coordinate $\mathbf{c}_i$ of its cell (not learned — deterministic from grid geometry).

**Why pool from $F_t^{(3)}$ (mid-coarse) instead of $F_t^{(2)}$ (mid):**
- The latent bank represents a **coarse global description** of the scene. Level-3 features already carry broader context per position, so the adaptive-average-pool preserves more useful information per cell.
- Pooling from $F_t^{(2)}$ (640×360) with $P_c = 256$ gives 900:1 compression — each cell averages 900 features, which destroys discriminative content. Pooling from $F_t^{(3)}$ gives 225:1, a 4× improvement in per-cell fidelity.
- Fine-grained spatial detail is recovered via direct bilinear lookups (B1) and deformable sampling (B3), not from the latent bank.
- Anchor positions in original-resolution coordinates are identical regardless of source level (16×16 grid cells map to the same physical locations).

**Why $P_c = 256$:**
- The original $P_c = 64$ (8×8 grid) compressed 3,600 features per cell — too aggressive. A cell covering 160×90 original pixels may contain building edges, sky, and foliage; their average is an information-poor centroid that impairs routing.
- With 256 tokens, the router has finer-grained candidates: $R = 4$ out of 256 (1.6%) is more spatially precise than $R = 4$ out of 64 (6.25%).
- Cross-attention cost over 256 tokens remains trivial ($1 \times 256$ attention matrix per head).
- Ablate with $P_c \in \{64, 128, 256, 512\}$ to find the accuracy/efficiency sweet spot.

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

**Multi-scale center features:**

$$
f_q^{(\ell)} = \text{Bilinear}(F_t^{(\ell)}, q / s_\ell), \quad \ell = 1, 2, 3, 4
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

A 2-layer MLP with GELU activation and **no skip connection**. Input dimension: $4d + d_{\text{pe}} = 544$. Hidden dimension: $d = 128$. Output: $h_{\text{point}} \in \mathbb{R}^d$.

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

This attends to ALL $P_c = 256$ tokens with soft weights. No information is lost. The summary captures the full global scene context relevant to this query. The $1 \times 256$ attention matrix per head is computationally trivial.

**Spatial routing (for deformable sampling anchors):**
$$
\alpha_q = \text{softmax}\left(\frac{W_r z_q \cdot (W_k^r C_t)^\top}{\sqrt{d}}\right), \quad
R_q = \text{TopR}(\alpha_q, R)
$$

Separate routing projection from the summary attention. This selects the $R = 4$ coarse tokens whose spatial regions are most relevant for precise evidence gathering. With $P_c = 256$, routing selects 4 out of 256 tokens (1.6%), providing spatially precise anchor selection from a fine-grained candidate set.

**Nearest coarse anchor:**
$$
i_q^{\text{loc}} = \arg\min_{i \in \{1, \ldots, P_c\}} \|\mathbf{c}_i - q^{(3)}\|_2^2
$$

where $q^{(3)} = (u/s_3, v/s_3)$ is the query in level-3 coordinates and $\mathbf{c}_i$ is the grid center of token $i$.

If $i_q^{\text{loc}} \in R_q$, replace the lowest-scoring token in $R_q$ with the next-highest-scoring token, ensuring no duplication.

**Spatial anchor set:**
$$
S_q = R_q \cup \{i_q^{\text{loc}}\}, \quad |S_q| = R + 1 = 5 \text{ (always)}
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
\Delta\mathbf{c}_r = \mathbf{c}_r - q \quad \text{(query-to-anchor offset in original pixel coordinates)}
$$
$$
u_r = \text{LN}(W_u [z_q;\; c_r;\; \phi(\Delta\mathbf{c}_r)] + b_u), \quad u_r \in \mathbb{R}^d
$$

where $c_r \in \mathbb{R}^d$ is the anchor's content vector from the latent bank $C_t$, and $\phi(\cdot)$ is the same Fourier positional encoding. Input dimension: $d + d + 16 = 272$. The offset $\Delta\mathbf{c}_r$ is computed in original pixel coordinates (not level-dependent) — the strided sampling in B3 already handles the level-to-pixel coordinate mapping. This conditioning vector tells the offset head three things: "I'm this query ($z_q$), looking at an anchor **whose content is** $c_r$, **located at** this spatial offset."

**Why include anchor content $c_r$:** Without $c_r$, the offset MLP receives $z_q$ (shared across all 5 anchors) plus only a 16-dim spatial offset — insufficient for content-adaptive sampling. DCNv2 (Zhu et al., 2019) predicts offsets from the input feature map at the reference location (via convolution over local features). Similarly, we provide anchor content so the offset head can ask: "given what this region looks like ($c_r$), where should I sample?" This enables structure-aware offsets (e.g. sampling along edges detected in $c_r$) rather than purely geometry-driven offsets.

**Offset and weight prediction (shared across anchors):**

For each head $h \in [1, H]$, level $\ell \in [1, L]$, sample $m \in [1, M]$:
$$
\Delta p_{r,h,\ell,m} = \kappa \cdot \tanh(W_{h,\ell,m}^\Delta \, u_r + b_{h,\ell,m}^\Delta)
$$
$$
\beta_{r,h,\ell,m} = W_{h,\ell,m}^a \, u_r + b_{h,\ell,m}^a
$$

where $\kappa = 8$ bounds offsets to $\pm 8$ pixels in each level's native coordinate system. The effective real-space reach naturally grows with stride: $\pm 8$ at level 1 (fine precision), $\pm 16$ at level 2, $\pm 32$ at level 3, $\pm 64$ at level 4 (5\% of 1280 image width). This matches Deformable DETR's practice where offsets have similar magnitude at each level's native resolution.

In practice, all offsets and weights are predicted by two shared linear layers (following Deformable DETR):
```python
self.sampling_offsets = nn.Linear(d, H * L * M * 2)   # -> 4*4*3*2 = 96
self.attention_weights = nn.Linear(d, H * L * M)       # -> 4*4*3   = 48
```

**Sampling:**
$$
p_{\text{sample}} = \mathbf{c}_r^{(\ell)} + \Delta p_{r,h,\ell,m}
$$
$$
f_{r,h,\ell,m} = \text{GridSample}_{\text{reflect}}(F_t^{(\ell)}, \text{Normalize}(p_{\text{sample}}))
$$

Reflective padding avoids border-collapse artifacts (proven better than clamp in Deformable DETR implementations).

**Per-anchor aggregation:**

For each anchor $r$, normalize weights over its own samples:
$$
a_{r,h,\ell,m} = \frac{\exp(\beta_{r,h,\ell,m})}{\sum_{h',\ell',m'} \exp(\beta_{r,h',\ell',m'})}, \quad
h_r = \sum_{h,\ell,m} a_{r,h,\ell,m} \, f_{r,h,\ell,m}
$$

Each $h_r \in \mathbb{R}^d$ is a self-contained spatial evidence summary from one anchor region.

**Default budget:** $H = 4$ heads, $L = 4$ levels, $M = 3$ samples.
- Per anchor: $4 \times 4 \times 3 = 48$ samples.
- 5 anchors: $5 \times 48 = 240$ deformable lookups total.

**Why $H = 4$ heads:** Every major deformable attention method (Deformable DETR, Mask2Former, DAB-DETR) uses 8 heads. Multiple heads learn complementary sampling patterns — one head for fine boundary evidence, another for broad structural context. $H = 4$ is a conservative compromise (half of the standard) that provides sufficient sampling diversity. This also matches the 4 heads used in the fusion decoder's cross-attention ($d_{\text{head}} = 128 / 4 = 32$).

**Why $L = 4$ levels:** Deformable sampling should span all available pyramid scales. Level-4 samples provide broad spatial evidence from a specific routed region at the coarsest scale, which is complementary to the global summary $\bar{c}_q$ (which is a compressed mix of all regions).

**Why per-anchor aggregation instead of a single global softmax:**
- Each anchor represents a distinct spatial region. Per-anchor normalization preserves the relative contribution of each region.
- The fusion transformer (B4) then learns how to weigh different regions via cross-attention.
- A single global softmax would allow one dominant region to suppress all others.

#### B4. Fusion decoder (2-layer cross-attention transformer)

**Context token set (with type embeddings):**

$$
T_q = [l_q + e_{\text{loc}}; \; h_{r_1} + e_{\text{near}}; \; h_{r_2} + e_{\text{route}}; \; \ldots; \; h_{r_5} + e_{\text{route}}; \; \bar{c}_q + e_{\text{glob}}]
$$

7 context tokens total, each with a learned type embedding:
- $l_q + e_{\text{loc}}$: aggregated local neighborhood (from B1).
- $h_{r_1} + e_{\text{near}}$: deformable evidence from the nearest anchor (medium-range context).
- $h_{r_2} + e_{\text{route}}, \ldots, h_{r_5} + e_{\text{route}}$: deformable evidence from routed anchors (non-local context).
- $\bar{c}_q + e_{\text{glob}}$: compressed global scene summary (from B2).

**Type embeddings:**

Each context token receives a learned type embedding ($e_{\text{loc}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}} \in \mathbb{R}^d$) identifying its role.

where $e_{\text{loc}}, e_{\text{near}}, e_{\text{route}}, e_{\text{glob}} \in \mathbb{R}^d$ are learned embeddings (4 × 128 = 512 parameters). This is standard practice in DETR-family architectures where different token roles (object queries, encoder tokens, positional queries) receive distinct learned embeddings. Without type embeddings, the cross-attention must infer token roles purely from content, requiring extra capacity.

**2-layer transformer decoder (standard Pre-LN):**

Each layer applies cross-attention with residual, then FFN with residual (standard Pre-LN transformer):
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{MHCrossAttn}(Q = \text{LN}(h_{\text{point}}), \; KV = T_q)
$$
$$
h_{\text{point}} \leftarrow h_{\text{point}} + \text{FFN}(\text{LN}(h_{\text{point}}))
$$

where $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$ with expansion ratio 4 ($d \to 4d \to d$, i.e., $128 \to 512 \to 128$).

Cross-attention uses 4 heads with $d_{\text{head}} = 32$. Attention matrix per head is $1 \times 7$ — trivially cheap.

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
- If the query is in a textureless region, the model can upweight $\bar{c}_q$ (global summary) and $h_{r_2..5}$ (non-local evidence).
- The type embeddings help the attention heads distinguish token roles without relying on content differences alone.
- The residual connections preserve $h_{\text{point}}$'s identity through both sub-layers, while cross-attention and FFN add contextual and nonlinear refinements.

This replaces the original plan's complex bounded-carry gates, explicit BCR/NUR monitoring, and balance losses. The transformer learns the right balance from data. If center collapse is observed, the center auxiliary loss $L_{\text{ctr}}$ addresses it directly.

**Why 2 layers:** SAM's decoder uses 2 layers and achieves excellent point-prompt results. Our context set is smaller (7 tokens vs thousands of image tokens in SAM), so 2 layers provide sufficient mixing depth. Ablate with 1 and 3 layers.

#### B5. Depth prediction

**Relative disparity code:**
$$
r_q = W_{r2} \cdot \text{GELU}(W_{r1} \, h_{\text{fuse}} + b_{r1}) + b_{r2}
$$

A 2-layer MLP with GELU activation. Output: scalar $r_q \in \mathbb{R}$.

**Center-only auxiliary code (training only):**
$$
r_q^{\text{ctr}} = W_{\text{ctr},2} \cdot \text{GELU}(W_{\text{ctr},1} \, h_{\text{point}}^{(0)} + b_{\text{ctr},1}) + b_{\text{ctr},2}
$$

A 2-layer MLP ($128 \to 64 \to 1$) applied to $h_{\text{point}}^{(0)}$, the center token BEFORE fusion (saved from B1). This forces the center branch to remain independently informative.

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
| `M3_DeformRead` | B3 | $S_q, z_q, C_t, F_t^{(1:4)} \to h_{r_1}, \ldots, h_{r_5}$ |
| `M4_FuseDecode` | B4+B5 | $h_{\text{point}}, T_q, s_t, b_t \to \hat{d}_q$ |

---

### 4.6 Why This Design Will Work — Confidence Arguments

1. **The precompute-then-query paradigm is proven.** SAM does exactly this for segmentation. Perceiver IO does it for arbitrary structured outputs. Both achieve state-of-the-art results with lightweight decoders on precomputed features.

2. **F^3 features contain sufficient depth information.** F^3 already achieves dense depth comparable to DepthAnythingV2. We use the same features — only the decoder changes.

3. **Local + global context is standard in depth estimation.** MiDaS, DPT, ZoeDepth, and Metric3D all use multi-scale features capturing both local and global context. Our three-stream design (center + local + global) replicates this in a sparse decoder.

4. **Deformable attention is battle-tested.** Used in Deformable DETR, Mask2Former, DAB-DETR, and many more. Learned offsets + importance weights for spatial sampling is robust and well-understood.

5. **The speed advantage is structural.** Dense decoders process $HW = 921{,}600$ pixels. We process $K$ queries with 276 feature lookups each. Even if per-query cost is 100× a single pixel in dense decoding, we win for $K < HW/100 = 9{,}216$. For practical query counts ($K \leq 1024$), the speedup is substantial. The accuracy-first budget (276 lookups/query) still yields a 3,340:1 ratio per query vs dense — the structural advantage is preserved with generous margins.

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
| Route stability loss $L_{\text{route-stab}}$ | 4.5A | STE with R=4 out of 256 tokens (1.6%) is sparse but well within the range where STE works (MoE models route 1-2 out of 128+ experts). If routing collapses, use entropy regularization (simpler). |
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
L_{\text{point}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(\hat{\rho}(q) - \rho^*(q))
$$

Pointwise Huber loss on inverse depth. Robust to outliers. $Q_v$ is the set of queries with valid ground truth.

**Scale-invariant structure:**
$$
L_{\text{silog}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q^2 - \lambda_{\text{var}} \left(\frac{1}{|Q_v|} \sum_{q \in Q_v} \delta_q\right)^2, \quad \delta_q = \log \hat{d}(q) - \log d^*(q)
$$

Standard SiLog loss. Handles scale ambiguity. Default $\lambda_{\text{var}} = 0.85$ (variance coefficient internal to SiLog, controls mean-shift invariance).

**Center auxiliary (prevents center collapse):**
$$
L_{\text{ctr}} = \frac{1}{|Q_v|} \sum_{q \in Q_v} \text{Huber}(r_q^{\text{ctr}} - \rho^*(q))
$$

Forces the pre-fusion center token to predict depth independently. If the center branch is informative on its own, the fused output will be at least as good.

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
| B2: Global summary | — | $P_c = 256$ | ~65K (cross-attn) |
| B2: Routing scores | — | $P_c = 256$ | ~33K (routing proj) |
| B3: Deformable read | 240 | — | ~54K (conditioning 272→128 + offset/weight heads) |
| B4: Type embeddings | — | — | ~0.5K (4 × $d$) |
| B4: Fusion (2 layers) | — | $7 \times 2$ | ~400K (transformer) |
| B5: Depth head | — | — | ~17K (MLP) |
| B5: $L_{\text{ctr}}$ auxiliary (train) | — | — | ~8K (2-layer MLP: 128→64→1) |
| **Total** | **276** | **~528** | **~772K** |

For comparison: DepthAnythingV2-Small has ~25M decoder parameters. Our sparse decoder is **32× smaller**. The accuracy-first budget of 276 lookups per query is still 3,340× less than dense decoding (921,600 pixels).

### 6.2 Speed Estimate (RTX 4090, 1280×720)

**Precompute (once per window):**
$$
T_{\text{precompute}} = T_{\text{F}^3} + T_{\text{pyramid+pool+cal}} \approx 8.33 + 0.40 = 8.73 \text{ ms}
$$

The 4-level pyramid adds ~0.05 ms over the 3-level version (one extra AvgPool_2×2 + 1×1 Proj over 57,600 features). The 16×16 latent bank pool from $F_t^{(3)}$ is negligible.

**Per-query marginal cost:**
$$
T_{\text{query}}(K) = \alpha + \beta K
$$

Conservative estimates with batched PyTorch operations (no custom CUDA kernels):
- $\alpha \approx 0.10$ ms (batch setup overhead).
- $\beta \approx 0.008$ ms/query (higher than the speed-optimized configuration due to 276 lookups and cross-attention over 256 latent tokens).

**Total:**
$$
T_{\text{EventSPD}}(K) \approx 8.83 + 0.008K \text{ ms}
$$

| Query count $K$ | EventSPD latency (ms) | EventSPD throughput (Hz) | Dense baseline (ms) | Speedup |
|-----|---:|---:|---:|---:|
| 1 | 8.8 | 114 | ~23 | 2.6× |
| 64 | 9.3 | 107 | ~23 | 2.5× |
| 256 | 10.9 | 92 | ~23 | 2.1× |
| 1024 | 17.0 | 59 | ~23 | 1.4× |

Dense baseline: F^3 (8.3 ms) + DepthAnythingV2 decoder (~15 ms) = ~23 ms. EventSPD is faster for all $K < \sim 1{,}800$.

**Design philosophy:** These defaults prioritize accuracy over speed. Once accuracy is validated, the sampling budget can be progressively reduced via ablation (H, L, M, $P_c$, $N_{\text{loc}}$) to find the minimal sufficient configuration. The structural advantage (per-query cost $\ll$ dense decoding) is preserved even with generous budgets.

**The key figure for the paper:** Plot $T_{\text{total}}(K)$ vs $K$ for EventSPD and dense-then-sample. The crossover point demonstrates the structural advantage. A secondary plot can show how the crossover shifts as the sampling budget is ablated down.

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
- Average cross-attention weight per context token type ($l_q$, $h_{\text{near}}$, $h_{\text{far}}$, $\bar{c}_q$) across queries. Shows whether the model uses all information streams.

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
| 6 | No routing (attend to all $C_t$) | Does sparse routing help efficiency without hurting accuracy? | Remove top-R, use all 256 tokens as deformable anchors (much slower). |
| 7 | Residual design | Does asymmetric residual help or hurt? | Compare: (a) standard both residuals (current), (b) drop FFN residual (asymmetric), (c) no residuals, (d) add center MLP skip connection. |
| 8 | Fusion depth | How many transformer layers? | 1, 2, 3 layers. |
| 9 | Latent bank size | How many coarse tokens? | $P_c \in \{64, 128, 256, 512\}$. Also ablate source level: $F_t^{(2)}$ vs $F_t^{(3)}$. |
| 10 | Routing budget | How many routed tokens? | $R \in \{2, 4, 6, 8\}$. |
| 11 | Local budget | How many local samples? | $N_{\text{loc}} \in \{8, 16, 32, 48, 64\}$. |
| 12 | Deformable budget | How many samples per anchor? | $(H, L, M) \in \{(2,3,3), (4,3,3), (4,4,3), (4,4,4)\}$. Current default: $(4,4,3)$. |

### Tier 3 — Extensions and alternatives

| # | Ablation | Question | Config |
|---|----------|----------|--------|
| 13 | Perceiver-style LatentPool | Do learned queries outperform spatial grid? | Replace A3 with MHA-based LatentPool. |
| 14 | Hash encoding vs Fourier PE | Does learned position encoding help? | Replace $\text{pe}_q$ with Instant-NGP hash. |
| 15 | Temporal memory | Does GRU state across windows help? | Add $H_t$ to cache and to $T_q$. |
| 16 | Uncertainty head | Does uncertainty improve hard queries? | Enable $\sigma_q$ + $L_{\text{unc}}$. |
| 17 | Center auxiliary loss | Does $L_{\text{ctr}}$ prevent center collapse? | Disable $L_{\text{ctr}}$ and compare. |
| 18 | Attention dropout rate | What rate is best for cross-attention regularization? | $p_{\text{attn}} \in \{0, 0.05, 0.1, 0.2\}$. |

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
| Speed not better than dense for medium $K$ | Low | The structural advantage (276 lookups/query vs $HW$ dense pixels) guarantees speedup for $K < 1800$. After accuracy validation, budget reduction via ablation can push the crossover higher. |
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
- Deformable ConvNets v2: https://arxiv.org/abs/1811.11168
- GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- Differentiable Top-k: https://arxiv.org/abs/2002.06504

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
