# Research Plan: "Inverse F^3" for Sparse Query-Point Event Depth

Author: Codex
Date: 2026-02-12
Research target: Predict depth only at user-provided pixel coordinates from event streams, with strong accuracy and much lower runtime than dense decoders.

## 1) Problem statement and constraints

Input:

- Raw events in a short temporal context window (e.g., 20 ms).
- Query set of pixel locations requested by user:
  $$
  Q=\{(u_i,v_i)\}_{i=1}^{K}.
  $$

Output:

- Depth (or disparity) only at requested query points.

Hard constraint from project discussion:

- Do not run a heavy full dense depth network over all pixels when only sparse points are needed.

Primary objective:

- Avoid dense per-pixel depth decoding from events.
- Allow one-time shared scene preprocessing from events, then answer many sparse queries efficiently.

Secondary objective:

- Keep query-point accuracy close to dense F^3+DepthAnything baseline.

## 2) Why this is publishable

Current event-depth practice is dominated by dense outputs. Even very good methods pay dense decoding cost. Your framing is a clear computationally meaningful objective with practical robotics value:

- many downstream tasks use sparse keypoints/control points,
- query budgets are often tiny,
- latency and power matter.

If you demonstrate favorable scaling and robust accuracy, this is a strong contribution.

## 3) Baseline system to reproduce first

Before new algorithm design, establish reference baselines in this repository.

### 3.1 Baseline A (dense best): F^3 + EventFFDepthAnythingV2

Use existing code:

- model wrapper: `src/f3/tasks/depth/utils/models/depth_anything_v2.py`
- stage-1: `src/f3/tasks/depth/train_monocular_rel.py`
- stage-2: `src/f3/tasks/depth/finetune.py`

Measure:

- full-map latency
- query-extraction latency (compute dense map, then sample Q points)
- query-point accuracy

### 3.2 Baseline B (dense non-F^3): V3/I3 + DepthAnythingV2

Use `EventDepthAnythingV2` path in same file. This gives representation ablation and stronger evidence that your gain is not only from F^3.

## 4) Proposed Algorithm (Runtime-First Specification)

This section is the implementation spec for GCQD-v2. The logic is strictly runtime-ordered:

1. Algorithm A: one-time precompute per event window.
2. Algorithm B: per-query-batch sparse depth inference.
3. Algorithm C: training loop and parameter optimization.

For every step, we state: what is computed, how it is computed, physical meaning, what uses it next, and whether it is trainable.

### 4.1 Objective and runtime form

Goal:

- Input: event window + user query points.
- Output: depth only at queried points.

Hard constraint:

- No dense full-image depth decoding at inference.

Runtime decomposition:

$$
T_{\text{total}}(K)=T_{\text{precompute}}(E_t)+T_{\text{query}}(K\mid \text{cache}_t)
$$

where $K$ is query count and $\text{cache}_t$ is shared state from Algorithm A.

### 4.2 Symbols and modules (source, meaning, usage, trainability)

| Symbol / module | How calculated from | Meaning | Used by | Trainable |
|---|---|---|---|---|
| $E_t$ | Raw events in $[t-\Delta,t)$ | Brightness-change stream | A1 | No |
| $\mathcal{F}_{\mathrm{F^3}}$ | Backbone network | Event-to-feature encoder | A1 | Yes if fine-tuned, otherwise frozen |
| $F_t$ | $F_t=\mathcal{F}_{\mathrm{F^3}}(E_t)$ | Shared scene/motion feature field | A2, B1, B6 | Depends on backbone mode |
| $F_t^{(1:3)}$ | Pyramid from $F_t$ | Fine/mid/coarse multiscale context | A2, B1, B6 | No (deterministic op) |
| $\mathrm{LatentPool}$ | Compression module over $F_t^{(2)}$ | Global scene summarizer | A3, B4, A4 | Yes |
| $C_t\in\mathbb{R}^{P_c\times d}$ | $C_t=\mathrm{LatentPool}(F_t^{(2)})$ | Compact coarse latent bank | B4, A4 | Yes (through LatentPool) |
| $G_t\in\mathbb{R}^{M_a\times d}$ | $G_t=\mathrm{AnchorPool}(C_t)$ | Always-visible global anchor tokens | B4, B7, B8 | Yes |
| $h_s,h_b$ | Heads on $C_t$ | Global calibration heads | A4, B9 | Yes |
| $s_t,b_t$ | $s_t=\mathrm{softplus}(h_s(C_t)),\ b_t=h_b(C_t)$ | Scale/shift for calibration | B9 | Yes (through heads) |
| $q=(u,v)$ | User input coordinate | Pixel to predict depth at | B1-B11 | No |
| $f_{q,0}$ | $\operatorname{Bilinear}(F_t^{(1)},q)$ | Exact feature at queried pixel | B1, B3, B7, B8, B9 | No |
| $l_q^{\text{ctx}}$ | LocalSample output around $q$ from $F_t^{(1)}$ (excluding center for context branch) | Neighborhood context near query | B3, B5, B8 | Sampling: No; local projection MLP: Yes |
| $M_t$ | $M_t=\alpha\,\mathrm{Norm}(\mathrm{Pool}(\lvert E_t\rvert))+\beta\,\mathrm{Norm}(\lVert\nabla F_t^{(1)}\rVert_1)$ | Local importance map for query-centric sampling | B1.1 | No |
| $R_t$ | $R_t=\mathrm{Norm}(\mathrm{Pool}(\mathrm{LastTS}(E_t)))$ | Local temporal recency map for hash interpolation | B2 | No |
| $N_{\text{loc}}$ | Number of sampled local points around query (adaptive) | Local evidence budget per query | B1.1, 4.7 complexity | No |
| $e(q)$ | Hash-grid coordinate encoding | Multiscale positional code | B3 | Yes (hash tables) |
| $z_q$ | $z_q=W_z[f_{q,0};l_q^{\text{ctx}};e(q)]$ | Query token (point+context+position) | B4, B5, B8 | Yes |
| $h_{\text{point},q}$ | Point-locked branch from $f_{q,0}$ and $e(q)$ | Target-pixel identity representation | B3, B7, B8, B9 | Yes |
| $\alpha_q$ | Attention logits/softmax over $C_t$ | Coarse-region relevance | B4 | Yes |
| $R_q$ | Top-$R$ from $\alpha_q$ | Routed coarse regions | B5-B7 | No (selection op) |
| $U_q$ | HashSelect$(q,U,P_c)$ | Coverage tokens (low-cost non-local fallback) | B4-B7 | No |
| $S_q$ | $S_q=R_q\cup U_q$ | Spatial token subset for deformable reads | B5-B7 | No |
| $\bar c_q$ | Attention summary over $S_q\cup G_t$ | Query-conditioned global scene summary | B4-B10 | Yes (through attention projections) |
| $\Delta p_{r,h,\ell,m}$ | Offset head | Learned sampling displacement | B6 | Yes |
| $a_{r,h,\ell,m}$ | Weight head + softmax | Sampling importance | B7 | Yes |
| $h_{\text{global}}$ | Weighted sum of sampled features | Non-local query context | B8 | Yes (upstream) |
| $h_{\text{ctx}}$ | Context branch output from $h_{\text{global}}$ and $l_q^{\text{ctx}}$ | Context correction representation | B8 | Yes |
| $g_q$ | Gate head + sigmoid | Local/global mixing ratio | B8 | Yes |
| $h_{\text{fuse}}$ | Gated fusion output | Final query representation | B9, B10 | No (fusion op) |
| $r_q$ | Relative disparity head output | Uncalibrated depth code | B9 | Yes |
| $\rho_q$ | $\rho_q=s_t r_q+b_t$ | Calibrated inverse-depth/disparity | B9 | No (algebraic op) |
| $\hat d_q$ | $\hat d_q=1/(\mathrm{softplus}(\rho_q)+\varepsilon)$ | Final depth at query | Output | No (conversion op) |
| $\sigma_q$ | Uncertainty head output | Confidence / ambiguity estimate | B10 | Yes |

### 4.3 Algorithm A: precompute once per event window

Inputs:

- Event set $E_t$.

Outputs (cache):

- $\text{cache}_t=\{F_t^{(1)},F_t^{(2)},F_t^{(3)},C_t,G_t,M_t,R_t,s_t,b_t\}$.

#### A1) Event encoding

Compute:
$$
F_t=\mathcal{F}_{\mathrm{F^3}}(E_t)
$$

Meaning:

- Shared event-derived representation for the whole window.

Used next:

- A2 (pyramid), A3 (latent pooling), B1/B6 (query path).

Trainable:

- Backbone parameters if fine-tuning is enabled.

#### A2) Feature pyramid

Compute:
$$
F_t^{(1)}=\mathrm{Proj}_{1\times1}(F_t)
$$
$$
F_t^{(2)}=\mathrm{Proj}_{1\times1}\!\left(\mathrm{Down}_2(F_t^{(1)})\right)
$$
$$
F_t^{(3)}=\mathrm{Proj}_{1\times1}\!\left(\mathrm{Down}_2(F_t^{(2)})\right)
$$
with
$$
\mathrm{Down}_2(\cdot)\in\{\mathrm{AvgPool}_{2\times2,\mathrm{stride}=2},\ \mathrm{DWConv}_{3\times3,\mathrm{stride}=2}\}.
$$

Meaning:

- $F_t^{(1)}$: fine detail.
- $F_t^{(2)}$: balanced context.
- $F_t^{(3)}$: broad context.

Used next:

- A3 uses $F_t^{(2)}$, B1 uses $F_t^{(1)}$, B6 uses all levels.

Trainable:

- No direct parameters for standard pooling/resize pyramid.

#### A3) Compact coarse latent bank (what LatentPool does)

Compute:
$$
C_t=\mathrm{LatentPool}(F_t^{(2)}),\qquad C_t\in\mathbb{R}^{P_c\times d}
$$

Default implementation:
$$
C_t=\mathrm{MHA}\!\left(Q_{\text{lat}},K=\mathrm{flat}(F_t^{(2)}),V=\mathrm{flat}(F_t^{(2)})\right)
$$

Meaning:

- "Compact": only $P_c$ tokens (much smaller than $HW$).
- "Coarse": each token summarizes a large region/global pattern.

Used next:

- A4 global calibration heads, B4 query routing.

Trainable:

- Yes (`LatentPool` weights and latent queries).

#### A4) Global calibration heads

Compute:
$$
s_t=\mathrm{softplus}(h_s(C_t)),\qquad b_t=h_b(C_t)
$$

Meaning:

- Window-level scale/shift to calibrate per-query predictions.

Used next:

- B9 calibration.

Trainable:

- Yes (`h_s`, `h_b`).

#### A5) Global anchor token bank (anti-separation path)

Compute:
$$
G_t=\mathrm{AnchorPool}(C_t),\qquad G_t\in\mathbb{R}^{M_a\times d}
$$
Default implementation:
$$
\Pi=\mathrm{softmax}\!\left(W_A C_t^\top\right),\qquad G_t=\Pi C_t
$$
where $W_A\in\mathbb{R}^{M_a\times d}$ and $\Pi\in\mathbb{R}^{M_a\times P_c}$.

Meaning:

- Small set of global tokens that every query can read, independent of routing.

Used next:

- B4 (summary), B7 (anchor read), B8 (coupling).

Trainable:

- Yes (`AnchorPool` / $W_A$).

#### A6) Cache state

Compute:
$$
M_t=\alpha\,\mathrm{Norm}\!\big(\mathrm{Pool}(|E_t|)\big)+\beta\,\mathrm{Norm}\!\big(\|\nabla F_t^{(1)}\|_1\big),\qquad
R_t=\mathrm{Norm}\!\big(\mathrm{Pool}(\mathrm{LastTS}(E_t))\big)
$$
$$
\text{cache}_t=\{F_t^{(1)},F_t^{(2)},F_t^{(3)},C_t,G_t,M_t,R_t,s_t,b_t\}
$$

Meaning:

- Shared runtime state reused by many query batches.
- $M_t$ is a precomputed local-importance map reused by `LocalSample` for fast adaptive local selection.
- $R_t$ is a precomputed event-recency map reused by hash embedding (B2) to add local temporal phase.
- $\mathrm{Pool}(|E_t|)$ means event-count rasterization of the current window, resized/pool-aligned to $F_t^{(1)}$ resolution.

Used next:

- Algorithm B.

Trainable:

- No.

### 4.4 Algorithm B: query-batch inference runtime

Inputs:

- Cached state $\text{cache}_t$.
- Query batch $Q_t=\{q_j\}_{j=1}^{K}$.

Outputs:

- Sparse predictions $\{(\hat d_j,\sigma_j)\}_{j=1}^{K}$.

#### B1) Local descriptor

For each query $q_j$:
$$
f_{j,0}=\operatorname{Bilinear}\!\left(F_t^{(1)},\frac{u_j}{s_1},\frac{v_j}{s_1}\right),\qquad
l_j^{\text{ctx}}=\mathrm{LocalSample}(F_t^{(1)},M_t,q_j)
$$

Meaning:

- $f_{j,0}$ is exact target-point evidence at the queried pixel.
- $l_j^{\text{ctx}}$ is neighborhood context around that target point.

Used next:

- B3 query token and point branch, B5 conditioning, B8 fusion.

Trainable:

- Sampling op: No.
- Optional local projection head (`MLP_local`): Yes.

#### B1.1) LocalSample helper function (explicit runtime definition)

Use a structured + importance sampling strategy, not nearest-neighbor search over all pixels.

Inputs:

- query pixel $q_j=(u_j,v_j)$ in image coordinates.
- fine feature map $F_t^{(1)}\in\mathbb{R}^{H_1\times W_1\times C}$.
- precomputed local-importance map $M_t\in\mathbb{R}^{H_1\times W_1}$ from Algorithm A.
- stride $s_1$ mapping image coordinates to $F_t^{(1)}$ coordinates (for current F^3 ds1 setup, $s_1=1$).

Define fixed dense core offsets:
$$
\Omega_{\text{core}}=\{(d_x,d_y)\mid d_x,d_y\in\{-4,-3,\dots,4\}\},\qquad |\Omega_{\text{core}}|=81
$$
Define outer candidate set:
$$
\Omega_{\text{outer}}=\operatorname{Unique}\!\left(\left\{\left(\lfloor r\cos\theta_k\rceil,\lfloor r\sin\theta_k\rceil\right)\mid r\in\{8,12\},\ \theta_k=\frac{2\pi k}{28},\ k=0,\dots,27\right\}\right),\qquad |\Omega_{\text{outer}}|\approx 56
$$
Define optional expansion ring:
$$
\Omega_{16}=\operatorname{Unique}\!\left(\left\{\left(\lfloor 16\cos\varphi_k\rceil,\lfloor 16\sin\varphi_k\rceil\right)\mid \varphi_k=\frac{2\pi k}{48},\ k=0,\dots,47\right\}\right),\qquad |\Omega_{16}|=48
$$

Runtime steps:

1. Map query to feature coordinates:
   $$
   \tilde q_j=(\tilde u_j,\tilde v_j)=\left(\frac{u_j}{s_1},\frac{v_j}{s_1}\right)
   $$
2. Score outer candidates using local importance and distance penalty:
   $$
   s_j(\delta)=\operatorname{Bilinear}(M_t,\tilde u_j+\delta_x,\tilde v_j+\delta_y)-\lambda_r\|\delta\|_2,\qquad \delta\in\Omega_{\text{outer}}\cup\Omega_{16}
   $$
   with $\lambda_r=0.02$ as default.
3. Select informative outer points with spatial de-duplication:
   $$
   \Omega_{\text{imp}}=\operatorname{TopK\_NMS}\!\left(\Omega_{\text{outer}},s_j,K_{\text{out}}=40,r_{\text{nms}}=2\right),\qquad |\Omega_{\text{imp}}|=40
   $$
4. Build default local set:
   $$
   \Omega_j=\Omega_{\text{core}}\cup\Omega_{\text{imp}},\qquad N_{\text{loc}}=|\Omega_j|=121
   $$
5. Optional hard-query local expansion:
   $$
   \bar m_j=\frac{1}{81}\sum_{\delta\in\Omega_{\text{core}}}\operatorname{Bilinear}(M_t,\tilde u_j+\delta_x,\tilde v_j+\delta_y)
   $$
   $$
   \text{if }\bar m_j<\tau_{\text{low}},\quad \Omega_{\text{extra}}=\operatorname{TopK\_NMS}\!\left(\Omega_{16},s_j,K_{\text{extra}}=48,r_{\text{nms}}=2\right),\quad \Omega_j\leftarrow \Omega_j\cup\Omega_{\text{extra}},\quad N_{\text{loc}}=169
   $$
   with $\tau_{\text{low}}$ set by training-set percentile (recommended 20th percentile of $\bar m_j$).
6. Sample local features and form descriptor:
   $$
   f_{j,0}=\operatorname{Bilinear}\!\left(F_t^{(1)},\tilde u_j,\tilde v_j\right),\qquad
   f_{j,\delta}=\operatorname{Bilinear}\!\left(F_t^{(1)},\tilde u_j+\delta_x,\tilde v_j+\delta_y\right),\ \delta\in\Omega_j\setminus\{(0,0)\}
   $$
   $$
   g_j^{\text{ctx}}=\operatorname{Concat}_{\delta\in\Omega_j\setminus\{(0,0)\}}\big[f_{j,\delta};\phi(\delta)\big],\qquad
   l_j^{\text{ctx}}=\operatorname{MLP}_{\text{local}}(g_j^{\text{ctx}})
   $$

Meaning:

- This guarantees one exact center feature ($f_{j,0}$) plus dense context evidence (core and optional expansion rings).

Used next:

- B3 (query token and point branch), B5 (conditioning), and B8 (fusion).

Trainable:

- Bilinear sampler, scoring map construction, and set selection: No.
- $\operatorname{MLP}_{\text{local}}$: Yes.

#### B2) Coordinate embedding with smooth hash interpolation

Compute:
$$
\xi_j=\left(\frac{u_j}{W},\frac{v_j}{H},\tau_j\right),\qquad
\tau_j=\operatorname{Bilinear}\!\left(R_t,\frac{u_j}{s_1},\frac{v_j}{s_1}\right)
$$
$$
e_j=\operatorname{Concat}_{s=1}^{S}(e_{s,j}),\qquad
e_{s,j}=\sum_{\nu\in\{0,1\}^3}w_{s,j,\nu}\,T_s\!\left[h_s\!\left(\mathbf{b}_{s,j}+\nu\right)\right]
$$
with
$$
\mathbf{p}_{s,j}=\mathbf{R}_s\odot\xi_j,\qquad
\mathbf{b}_{s,j}=\lfloor\mathbf{p}_{s,j}\rfloor,\qquad
\mathbf{r}_{s,j}=\mathbf{p}_{s,j}-\mathbf{b}_{s,j}
$$
$$
w_{s,j,\nu}=\prod_{d\in\{x,y,t\}}\left(r_{s,j,d}^{\nu_d}(1-r_{s,j,d})^{1-\nu_d}\right),\qquad
\sum_{\nu\in\{0,1\}^3}w_{s,j,\nu}=1
$$
where $\mathbf{R}_s=(R_{s,x},R_{s,y},R_{s,t})$ is level-$s$ grid resolution, $h_s$ is hash indexing, and $T_s$ is the learnable table.

Meaning:

- Continuous multiscale coordinate encoding for query position and local temporal phase.

Intuition (why this step exists):

- Nearby query coordinates must map to nearby embeddings; vertex interpolation makes hash encoding smooth and trainable.
- This is the same smoothing principle used by F^3/Instant-NGP style hash encoders (8-corner trilinear interpolation).
- If temporal recency is disabled, use 2D bilinear interpolation (4 corners) with the same logic.
- This step creates a compact address fingerprint for the query point without a dense positional map.

Used next:

- B3.

Trainable:

- Yes (hash tables $T_s$).

#### B3) Query token and point-locked branch

Compute:
$$
\tilde f_{j,0}=\mathrm{LN}(f_{j,0}),\qquad
\tilde l_j^{\text{ctx}}=\mathrm{LN}(l_j^{\text{ctx}}),\qquad
\tilde e_j=\mathrm{LN}(e_j)
$$
$$
z_j=W_z[\tilde f_{j,0};\tilde l_j^{\text{ctx}};\tilde e_j]+b_z,\qquad
h_{\text{point},j}=W_p[\tilde f_{j,0};\tilde e_j]+b_p
$$
$$
h_{\text{point},j}\leftarrow\mathrm{LN}(h_{\text{point},j}),\qquad
z_j\in\mathbb{R}^{d},\ h_{\text{point},j}\in\mathbb{R}^{d}
$$

Meaning:

- $z_j$ is the routing/query token for global context retrieval.
- $h_{\text{point},j}$ is a target-pixel identity branch that preserves exact-point information.

Intuition (why this step exists):

- A dedicated point branch prevents the exact target signal from being diluted by surrounding/global features.
- The routing token still uses context and position, so global retrieval remains strong.
- This split is the core fix for local-global separation.

Used next:

- B4 routing, B5 offsets/weights, B7/B8 fusion.

Trainable:

- Yes ($W_z$, $W_p$, and norm layers).

#### B4) Coarse routing with connectivity-safe token subset

Compute content routing scores over coarse bank:
$$
\alpha_j=\mathrm{softmax}\!\left(\frac{(W_q z_j)(W_k C_t)^\top}{\sqrt d}\right),\qquad
R_j=\operatorname{TopR}(\alpha_j)
$$
Compute deterministic coverage tokens (cheap non-local fallback):
$$
u_{j,m}=1+\Big((a_m u_j+b_m v_j+c_m)\bmod P_c\Big),\qquad
U_j=\operatorname{Unique}\!\left(\{u_{j,m}\}_{m=1}^{U_0}\right)_{1:U}
$$
where $\{(a_m,b_m,c_m)\}_{m=1}^{U_0}$ are fixed integer triplets (chosen once, coprime with $P_c$) to provide deterministic pseudo-random coverage.
Build spatial subset:
$$
S_j=R_j\cup U_j
$$
Build query-conditioned global summary using routed/coverage tokens plus anchors:
$$
\mathcal{V}_j=\{c_i\mid i\in S_j\}\cup\{g_a\}_{a=1}^{M_a},\qquad
\bar c_j=\mathrm{softmax}\!\left(\frac{(W_q^{v}z_j)(W_k^{v}\mathcal{V}_j)^\top}{\sqrt d}\right)\mathcal{V}_j
$$
Each coarse token $c_i$ stores metadata $(\mathbf{p}_i,\ell_i)$ where $\mathbf{p}_i$ is coarse center coordinate and $\ell_i$ is level tag.

Meaning:

- $\alpha_j$: content relevance over compact global regions.
- $R_j$: top content-routed regions.
- $U_j$: guaranteed coverage regions (prevents over-confident misses).
- $S_j$: sparse spatial token subset for deformable reads.
- $\bar c_j$: query-conditioned global summary token.
- $W_q z_j$ (Query): what context this query point is asking for.
- $W_k c_i$ (Key): how each coarse token advertises what it contains.
- $v\in\mathcal{V}_j$ (Value): the actual global content that is mixed into $\bar c_j$.

Intuition (why this step exists):

- Content routing alone can miss long-range cues if logits are over-confident.
- Coverage tokens are imported from sparse-attention ideas (local+global+random connectivity): they add a low-cost path to far regions.
- Anchor tokens are imported from latent-bottleneck methods: every query can still access scene-level context even if spatial routing is imperfect.
- This combination directly addresses the global-local separation problem before fusion starts.

Used next:

- B5-B10.

Trainable:

- Yes ($W_q,W_k,W_q^v,W_k^v$). Top-$R$, hash coverage, and set union are non-trainable.

#### B5) Offset and weight prediction with local-global conditioning

This step converts routed coarse tokens into concrete sampling coordinates and sampling importance.

Inputs per query $j$:

- $z_j\in\mathbb{R}^{d}$: query token.
- $f_{j,0}\in\mathbb{R}^{d_f}$ and $l_j^{\text{ctx}}\in\mathbb{R}^{d_l}$: exact-point and neighborhood descriptors.
- $h_{\text{point},j}\in\mathbb{R}^{d}$: point-locked representation from B3.
- $\bar c_j\in\mathbb{R}^{d}$: query-conditioned global summary.
- For each routed token $r\in S_j$: center metadata $(\mathbf{p}_r,\ell_r)$ and token type (`routed`/`coverage`).

Step 1: build token-conditioned context vector:
$$
\hat u_{j,r}=[z_j;\ h_{\text{point},j};\ f_{j,0};\ l_j^{\text{ctx}};\ \bar c_j;\ \psi(\mathbf{p}_r,\ell_r);\ \tau_r]
$$
$$
u_{j,r}=\mathrm{LN}(W_u\hat u_{j,r}+b_u),\qquad u_{j,r}\in\mathbb{R}^{d_u}
$$

Step 2: predict raw offsets for each head/level/sample slot:
$$
o_{r,h,\ell,m}=W^{\Delta}_{h,\ell,m}u_{j,r}+b^{\Delta}_{h,\ell,m},\qquad o_{r,h,\ell,m}\in\mathbb{R}^{2}
$$

Step 3: bound offsets to valid local radius:
$$
\Delta p_{r,h,\ell,m}=\rho_{\ell}\tanh(o_{r,h,\ell,m})
$$
$$
\rho_{\ell}=\kappa s_{\ell}\ \text{ (recommended scale-consistent choice)}
$$

Step 4: predict unnormalized importance logits:
$$
\beta_{r,h,\ell,m}=W^{a}_{h,\ell,m}u_{j,r}+b^{a}_{h,\ell,m}
$$

Step 5: normalize importance over all sampling slots of query $j$:
$$
\mathcal{D}_j=\{(r,h,\ell,m)\mid r\in S_j,\ h\in[1,H],\ \ell\in[1,L],\ m\in[1,M]\}
$$
$$
a_{r,h,\ell,m}=\frac{\exp(\beta_{r,h,\ell,m})}{\sum_{(r',h',\ell',m')\in\mathcal{D}_j}\exp(\beta_{r',h',\ell',m'})}
$$
$$
\sum_{(r,h,\ell,m)\in\mathcal{D}_j}a_{r,h,\ell,m}=1,\qquad a_{r,h,\ell,m}\ge 0
$$

Outputs of B5:

- $\Delta p_{r,h,\ell,m}$: where to read in B6.
- $a_{r,h,\ell,m}$: how much each read contributes in B7.

Meaning:

- Learn both sampling geometry and sampling confidence, jointly conditioned on local and global evidence.

Intuition (why this step exists):

- Coarse routing (B4) only says which regions may matter; B5 refines that into exact points.
- Offset head answers: "where exactly should I look?"
- Weight head answers: "how much should I trust each looked-up point?"
- Because $u_{j,r}$ contains both the point-locked branch and context/global branches, sampling stays centered on the target while still using non-local cues.

Used next:

- B6 and B7.

Trainable:

- Yes.

#### B6) Deformable multiscale sampling over selected spatial subset

What is being computed:

- For each query `j`, routed token `r in S_j`, head `h`, pyramid level `\ell`, and sample index `m`, compute one sampling coordinate and read one feature vector.

Coordinate definitions (all in level-`\ell` feature coordinates):
$$p_{q_j}^{(\ell)}=\left(\frac{u_j}{s_{\ell}},\frac{v_j}{s_{\ell}}\right)$$
$$p_{r}^{(\ell)}=\text{coarse center offset of routed token }r\text{ at level }\ell$$
$$\Delta p_{r,h,\ell,m}=\text{learned residual offset from B5}$$

Step-by-step sampling:
$$
p_{\text{base},r,\ell}=p_{q_j}^{(\ell)}+p_{r}^{(\ell)}
$$
$$
p_{\text{raw},r,h,\ell,m}=p_{\text{base},r,\ell}+\Delta p_{r,h,\ell,m}
$$
$$
p_{\text{valid},r,h,\ell,m}=\mathrm{ClampToValid}\!\left(p_{\text{raw},r,h,\ell,m};\ H_{\ell},W_{\ell}\right)
$$
$$
f_{r,h,\ell,m}=\operatorname{Bilinear}\!\left(F_t^{(\ell)},\ p_{\text{valid},r,h,\ell,m}\right)
$$
$$
f_{\text{ctr},j}^{(\ell)}=\operatorname{Bilinear}\!\left(F_t^{(\ell)},\ p_{q_j}^{(\ell)}\right)
$$

Equivalent compact form:
$$
(x_{r,h,\ell,m},y_{r,h,\ell,m})=p_{q_j}^{(\ell)}+p_r^{(\ell)}+\Delta p_{r,h,\ell,m}
$$
$$
(x,y)\leftarrow\mathrm{ClampToValid}(x_{r,h,\ell,m},y_{r,h,\ell,m}),\qquad
f_{r,h,\ell,m}=\operatorname{Bilinear}(F_t^{(\ell)},x,y)
$$

Meaning:

- `p_{q_j}^{(\ell)}`: where the query is at this scale.
- `p_r^{(\ell)}`: which coarse global region to inspect.
- `\Delta p_{r,h,\ell,m}`: fine correction learned by network.
- `f_{r,h,\ell,m}`: actual evidence fetched from level-`\ell` feature map.
- `f_{\text{ctr},j}^{(\ell)}`: exact query-centered feature at each scale (point-lock bypass).

Intuition (why this step exists):

- First jump to a coarse non-local region (`p_r^{(\ell)}`), then make a precise local correction (`\Delta p`), then read feature evidence there.
- `ClampToValid` is only a safety operation to avoid out-of-bound memory access.
- `Bilinear` allows sub-pixel/fractional sampling, so offsets are continuous and trainable.
- Multi-scale reads provide both local detail (`F_t^{(1)}`) and wider context (`F_t^{(2)},F_t^{(3)}`) without dense decoding.
- The extra center read ensures the exact query location is always represented even if deformable offsets miss.

Used next:

- B7.

Trainable:

- Sampling op: No. Locations are trainable through B5.

#### B7) Global aggregation plus point-locked branch assembly

Aggregate deformable samples:
$$
h_{\text{def},j}=\sum_{r\in S_j}\sum_{h,\ell,m}a_{r,h,\ell,m}\,f_{r,h,\ell,m}
$$
Read anchor bank:
$$
\gamma_{j,a}=\mathrm{softmax}_{a=1}^{M_a}\!\left(\frac{(W_q^{a}z_j)^\top(W_k^{a}g_a)}{\sqrt d}\right),\qquad
h_{\text{anchor},j}=\sum_{a=1}^{M_a}\gamma_{j,a}g_a
$$
Build context-global branch:
$$
h_{\text{global},j}=\mathrm{LN}\!\left(W_{\text{glob}}[h_{\text{def},j};h_{\text{anchor},j};\bar c_j]+b_{\text{glob}}\right)
$$
Build point-locked branch:
$$
h_{\text{ctr},j}=\mathrm{LN}\!\left(W_{\text{ctr}}\!\left[h_{\text{point},j};f_{\text{ctr},j}^{(1)};f_{\text{ctr},j}^{(2)};f_{\text{ctr},j}^{(3)}\right]+b_{\text{ctr}}\right)
$$

Meaning:

- $h_{\text{global},j}$ summarizes non-local evidence.
- $h_{\text{ctr},j}$ preserves exact target-pixel identity across scales.

Intuition (why this step exists):

- Global context is needed to disambiguate depth.
- Exact target-point evidence must remain explicit and cannot be replaced by neighborhood averages.
- Two branches are built in parallel so fusion can be residual (context adds to target point, not overwrite it).

Used next:

- B8 fusion.

Trainable:

- Yes through upstream trainable heads and projections.

#### B8) Residual context fusion (default and recommended)

Compute context residual:
$$
h_{\text{ctx},j}=\mathrm{LN}\!\left(W_{\text{ctx}}[h_{\text{global},j};l_j^{\text{ctx}};\bar c_j]+b_{\text{ctx}}\right)
$$
Compute gate:
$$
g_j=\sigma\!\Big(\mathrm{MLP}_g([h_{\text{ctr},j};h_{\text{ctx},j};z_j])\Big),\qquad 0\le g_j\le 1
$$
Fuse:
$$
h_{\text{fuse},j}=h_{\text{ctr},j}+g_j\odot h_{\text{ctx},j}
$$

Meaning:

- $h_{\text{ctr},j}$ is the base representation of the exact target point.
- $h_{\text{ctx},j}$ is context correction.
- $g_j$ controls how much correction is needed.

Intuition (why this step exists):

- This prevents global/local context from overpowering target-point information.
- The model can still use far evidence, but only as a residual update.
- One-shot residual fusion is simpler and avoids redundant iterative loops.

Used next:

- B9 depth head, B10 uncertainty.

Trainable:

- Yes.

#### B9) Depth code, calibration, and final depth

Compute relative code:
$$
r_j=h_r(h_{\text{fuse},j})
$$
Optional point-only auxiliary code (train-time supervision):
$$
r_j^{\text{ctr}}=h_r^{\text{ctr}}(h_{\text{ctr},j})
$$

Typical head form:
$$
r_j=W_r h_{\text{fuse},j}+b_r
$$

Calibrate:
$$
\rho_j=s_t r_j+b_t
$$

Convert to depth:
$$
\hat d_j=\frac{1}{\mathrm{softplus}(\rho_j)+\varepsilon}
$$

Meaning:

- $r_j$: uncalibrated depth code.
- $r_j^{\text{ctr}}$: point-only auxiliary code (used only for training regularization).
- $\rho_j$: calibrated inverse-depth/disparity.
- $\hat d_j$: final depth at query.

Intuition (why this step exists):

- The decoder first predicts a relative code $r_j$ because relative geometry is easier to learn.
- The auxiliary $r_j^{\text{ctr}}$ forces the center branch to remain informative and prevents target-point information collapse.
- $s_t,b_t$ inject window-level global calibration so all queries in the same window are on a consistent scale.
- Softplus keeps denominator positive and avoids invalid depth values.
- Without calibration, outputs can be locally correct but globally mis-scaled.

Used next:

- Output and losses.

Trainable:

- Yes ($h_r$ and upstream).

#### B10) Uncertainty and optional refinement

Compute uncertainty:
$$
\sigma_j=h_{\sigma}(h_{\text{fuse},j})
$$

Recommended positive form:
$$
\sigma_j=\mathrm{softplus}(W_{\sigma}h_{\text{fuse},j}+b_{\sigma})+\sigma_{\min}
$$

Optional hard-query second-hop rerouting:
$$
\text{if }\sigma_j>\tau_{\text{hop}},\quad
z_j^{(2)}=W_u[z_j;h_{\text{fuse},j};\bar c_j],\quad
\text{run B4-B8 with }(R_2,U_2),\quad
\hat d_j\leftarrow \hat d_j+\Delta d_j^{\text{hop}}
$$

Refinement head:
$$
\Delta d_j^{\text{hop}}=
\mathrm{MLP}_{\text{refine}}\!\big([h_{\text{fuse},j}^{(2)};h_{\text{fuse},j};l_j^{\text{ctx}};e_j]\big)
$$
Recommended small second-hop budget:
$$
R_2=2,\qquad U_2=1
$$

Meaning:

- Spend extra global-local interaction compute only on difficult queries.
- This branch is optional and disabled in the GCQD-lite deployment profile.

Intuition (why this step exists):

- Not all queries are equally hard; uncertainty estimates which points are unreliable.
- Second-hop rerouting lets hard queries gather additional far-range evidence without paying that cost for all queries.
- This preserves speed on easy points while improving hard points (boundaries, low-event zones, motion ambiguity).
- Without uncertainty, you either waste compute everywhere or miss hard-case corrections.

Used next:

- Output and uncertainty-aware objectives.

Trainable:

- Yes ($h_{\sigma}$ and optional refine head).

#### B11) Return sparse outputs

Output:
$$
\{(\hat d_j,\sigma_j)\}_{j=1}^{K}
$$

#### B12) GCQD-lite collapsed runtime (recommended efficient implementation)

The detailed B1-B11 decomposition is for clarity and analysis. For efficient implementation, collapse them into a small active path:

1. Precompute shared state once per window:
   $$
   E_t\rightarrow F_t^{(1:3)},\ C_t,\ G_t,\ M_t,\ R_t,\ s_t,\ b_t
   $$
2. Query embedding:
   $$
   q_j\rightarrow (f_{j,0},l_j^{\text{ctx}})\ (\text{fixed }N_{\text{loc}}=121)\rightarrow e_j\rightarrow (z_j,h_{\text{point},j})
   $$
3. Sparse global subset selection:
   $$
   z_j,C_t,G_t\rightarrow S_j,\ \bar c_j
   $$
4. Sparse non-local read:
   $$
   (z_j,f_{j,0},l_j^{\text{ctx}},h_{\text{point},j},\bar c_j,S_j)\rightarrow (h_{\text{global},j},h_{\text{ctr},j})
   $$
5. Single-shot fusion + depth:
   $$
   (h_{\text{global},j},h_{\text{ctr},j},l_j^{\text{ctx}},z_j,\bar c_j,s_t,b_t)\rightarrow \hat d_j
   $$
6. Optional uncertainty:
   $$
   h_{\text{fuse},j}\rightarrow \sigma_j
   $$

Default deployment profile:

- Keep fixed local sampler (`N_loc=121`) without adaptive expansion.
- Use one-shot fusion (single gate) instead of bidirectional two-way coupling.
- Keep second-hop rerouting disabled by default.

This gives a runtime structure close to the compact patterns in F^3, SAM, LIIF, and Perceiver-style query decoders.

### 4.5 Algorithm C: training runtime and what is optimized

This section marks explicitly which steps run ML modules that are optimized by training.

#### C1) Precompute forward

- Run Algorithm A.
- All trainable modules used in A participate in forward graph.

#### C2) Query sampling

- Sample query set with mixed policy (uniform + event-dense + edges + far-field).
- Not trainable, but critical for optimization target distribution.

#### C3) Sparse forward

- Run Algorithm B for sampled queries.
- All trainable modules in B are active in forward graph.

#### C4) Supervision assembly

- Gather labels at queried points from pseudo depth and/or metric LiDAR valid pixels.

#### C5) Loss computation

Compute:
$$
\mathcal{L}=\lambda_1L_{\text{point}}+\lambda_2L_{\text{silog},q}+\lambda_3L_{\text{rank}}+\lambda_4L_{\text{unc}}+\lambda_5L_{\text{distill}}+\lambda_6L_{\text{comp}}+\lambda_7L_{\text{ctr}}
$$

Detailed formulas are in Section 6.

#### C6) Backprop and optimizer step (explicit optimization step)

- Update trainable modules (default profile): LatentPool, AnchorPool, hash tables, query projection, point branch projection, routing projections, summary projections, offset/weight heads, center-branch projector, residual gate head, depth head, point-aux depth head, uncertainty head.
- Update additional modules only in extended ablation profile: second-hop projector and refine head.
- Backbone $\mathcal{F}_{\mathrm{F^3}}$ is updated only when fine-tune mode is enabled.
- Use context-drop regularization during training: with probability $p_{\text{ctxdrop}}$, set $h_{\text{ctx},j}\leftarrow 0$ before B8 so the center branch remains predictive.

#### C7) Query-budget curriculum

- Train with varying $K$ to make runtime robust across query loads.

### 4.6 Why choose a compact coarse latent bank and how to verify it

Why needed:

- local-only query evidence is ambiguous in many cases.
- $C_t$ carries global context at low cost.

What "preserve information" means here:

- preserve task-relevant global information, not every pixel detail.
- local target identity is preserved in $(f_{q,0},h_{\text{point},q})$ and neighborhood context in $l_q^{\text{ctx}}$; global disambiguation is preserved in $C_t$.

Required proof-by-ablation:

1. Vary $P_c\in\{32,64,128,256,512\}$ and report accuracy/runtime.
2. Compare with local-only baseline (remove global branch).
3. Shuffle/randomize $C_t$ and $G_t$ at inference and measure degradation.
4. Report by depth buckets (near/mid/far) to show global context effect.
5. Remove coverage tokens $U_j$ (set $U=0$) and measure far-field error change.
6. Replace B8 with late-only gate and measure degradation (tests local-global coupling necessity).
7. Vary anchor count $M_a\in\{0,4,8,16\}$ and measure accuracy/runtime.

Acceptance signal:

- moderate $P_c$ reaches near-full accuracy with much lower runtime than dense decoding.

### 4.7 CUDA execution mapping and complexity

Kernel schedule (GCQD-lite default):

1. `K0_F3Precompute`: $E_t\rightarrow F_t^{(1:3)},C_t,G_t,M_t,R_t,s_t,b_t$.
2. `K1_QueryEmbed`: $q\rightarrow f_{q,0},l_q^{\text{ctx}},e(q),z_q,h_{\text{point},q}$ (fixed `LocalSample`, $N_{\text{loc}}=121$).
3. `K2_RouteSubset`: $z_q,C_t,G_t\rightarrow R_q,U_q,S_q,\bar c_q$.
4. `K3_DefGather`: offsets/weights + deformable multiscale gather over $S_q$.
5. `K4_FuseDepth`: fused local-global gate + depth calibration.
6. `K5_Uncertainty`: optional uncertainty head.

Extended research profile (for ablations only):

- add adaptive local expansion in `K1`,
- add context-drop ablation in `K4`,
- add second-hop rerouting after `K5`.

Complexity:

- precompute: $\mathcal{O}(HWC+P_cM_a d)$ (backbone dominated, where $M_a$ is anchor count).
- expected per query:
$$
\mathcal{O}\!\Big(\bar N_{\text{loc}}d+(R+U)d+(R+U)HLMd+M_a d+p_{\text{hop}}(R_2+U_2)HLMd\Big)
$$
where
$$
\bar N_{\text{loc}}=121+48\,p_{\text{loc-hard}}
$$
and $p_{\text{loc-hard}}$ is the fraction of queries that trigger local expansion, while $p_{\text{hop}}$ is the fraction that trigger second-hop rerouting.

Default starting hyperparameters:

- $P_c=256$, $R=6$, $U=2$, $M_a=8$, $H=4$, $L=3$, $M=4$.
- Local sampling budget: fixed $N_{\text{loc}}=121$ in GCQD-lite.
- Optional expansion budget for ablations: $N_{\text{loc}}=169$ when enabled.
- First-hop sampled points/query: $(R+U)HLM=384$.
- Second-hop (hard queries only, ablation mode): $(R_2+U_2)HLM=144$ with $(R_2,U_2)=(2,1)$.

### 4.8 RTX 4090 speed estimate (using 120 Hz HD F^3 baseline)

Updated baseline for this estimate:
$$
f_{\mathrm{F^3,core}}=120\ \mathrm{Hz}\Rightarrow T_{\mathrm{F^3,core}}=\frac{1000}{120}\approx 8.33\ \mathrm{ms}
$$
We still keep conservative non-backbone overhead ranges and do not use the most optimistic deployment-only timings.

Model:
$$
T_{\mathrm{GCQD}}(K)=T_{\mathrm{F^3,core}}+T_M+T_Q(K)
$$

Estimate range:
$$
T_M\in[0.35,1.00]\ \mathrm{ms},\quad T_Q(K)=\alpha+\beta K,
$$
$$
\alpha\in[0.09,0.36]\ \mathrm{ms},\quad \beta\in[0.004,0.010]\ \mathrm{ms/query}
$$
These values are calibrated relative to the README F^3 benchmark regime and kept conservative for non-fused sparse query kernels.

Therefore:
$$
T_{\mathrm{GCQD}}(K)\in[8.77+0.004K,\ 9.69+0.010K]\ \mathrm{ms}
$$

Assumption for this estimate:

- local sampling uses default $N_{\text{loc}}=121$ with adaptive expansion to 169 for a fraction $p_{\text{loc-hard}}\approx0.15$ of queries.
- token subset uses $(R,U,M_a)=(6,2,8)$.
- second-hop reroute is triggered for about $p_{\text{hop}}=0.15$ of queries and is absorbed into the $\beta$ range.

| Query count $K$ | Estimated runtime (ms) | Estimated throughput (Hz) |
|---|---:|---:|
| 1 | 8.77 - 9.70 | 114.0 - 103.1 |
| 64 | 9.03 - 10.33 | 110.8 - 96.8 |
| 256 | 9.79 - 12.25 | 102.1 - 81.6 |
| 1024 | 12.87 - 19.93 | 77.7 - 50.2 |

### 4.8B README-fast baseline comparison (pure F^3 at ~448 Hz on RTX 4090)

README benchmark reports pure F^3 feature extraction around:
$$
T_{\mathrm{F^3,README}}=2.23\ \mathrm{ms}\quad(\approx 448\ \mathrm{Hz})
$$
for `1280x720`, `200K events`, mixed precision settings.

Using the same model form and conservative overhead ranges:
$$
T_{\mathrm{GCQD}}(K)=T_{\mathrm{F^3,README}}+T_M+T_Q(K)
$$
$$
T_M\in[0.35,1.00]\ \mathrm{ms},\qquad T_Q(K)\in[0.09+0.004K,\ 0.36+0.010K]\ \mathrm{ms}
$$
$$
T_{\mathrm{GCQD}}(K)\in[2.67+0.004K,\ 3.59+0.010K]\ \mathrm{ms}
$$

Comparison relative to pure F^3:
$$
\text{Overhead factor}=\frac{T_{\mathrm{GCQD}}(K)}{T_{\mathrm{F^3,README}}}
$$

| Query count $K$ | GCQD runtime (ms) | GCQD throughput (Hz) | Relative to pure F^3 |
|---|---:|---:|---:|
| 1 | 2.67 - 3.60 | 374.5 - 277.8 | 1.20x - 1.61x |
| 64 | 2.93 - 4.23 | 341.3 - 236.4 | 1.31x - 1.90x |
| 256 | 3.69 - 6.15 | 271.0 - 162.6 | 1.66x - 2.76x |
| 1024 | 6.77 - 13.83 | 147.7 - 72.3 | 3.04x - 6.20x |

Interpretation:

- GCQD is expected to be slower than pure F^3 representation (this is unavoidable because it adds sparse depth-query decoding work).
- The correct target is not beating pure F^3, but beating dense depth heads while serving only requested queries.
- This table should be treated as a realistic upper-bound planning estimate until kernel-level fitting is completed.

### 4.8A F^3-first simplification (recommended next step)

Your concern is correct: if $T_{\text{precompute}}$ dominates, sparse querying alone gives limited benefit.

Runtime decomposition:
$$
T_{\text{total}}(K)=T_{\mathrm{F^3}}+T_{\text{sparse}}(K)
$$
When $T_{\mathrm{F^3}}\gg T_{\text{sparse}}(K)$, the first priority is reducing $T_{\mathrm{F^3}}$.

#### A) Simplify F^3 before sparse decoder changes

Use a staged F^3-lite design:

1. Global coarse pass (always-on):
   - compute $F_{t,\text{coarse}}$ at reduced spatial scale (recommended ds2),
   - reduce hash levels and channels.
2. Query-local detail pass (on-demand):
   - for each query, read local fine evidence directly from raw-event neighborhood (already in B1) and center-read branch,
   - optionally run local high-res correction only for hard queries.

This keeps global context while moving expensive high-res processing away from full-frame always-on compute.

#### B) F^3-lite knobs (ordered by compute impact)

1. Spatial scale: global F^3 at ds2 instead of ds1.
2. Hash encoder width: reduce levels/features, e.g. $(L,F): (4,2)\rightarrow(3,2)$ or $(2,2)$.
3. Smoothing network: reduce channels/blocks and keep depthwise separable convolutions.
4. Window schedule: lower update frequency for global cache, reuse cache across nearby query batches.
5. Deployment path: PT2/AOTI + prebuilt hash features when memory allows.

#### C) Conservative runtime targets (relative to 8.33 ms baseline)

Define the new model:
$$
T_{\text{GCQD-lite-F3lite}}(K)=T_{\mathrm{F^3lite}}+T_M+T_Q(K)
$$
with the same $T_M$ and $T_Q(K)$ ranges as Section 4.8.

Recommended target tiers:

- Tier-1 (safe): $T_{\mathrm{F^3lite}}\in[7.0,8.5]$ ms.
- Tier-2 (aggressive but still conservative): $T_{\mathrm{F^3lite}}\in[6.0,7.5]$ ms.

Then:
$$
T_{\text{Tier-1}}(K)\in[7.44+0.004K,\ 9.86+0.010K]\ \text{ms}
$$
$$
T_{\text{Tier-2}}(K)\in[6.44+0.004K,\ 8.86+0.010K]\ \text{ms}
$$

Example totals:

| Query count $K$ | Tier-1 runtime (ms) | Tier-2 runtime (ms) |
|---|---:|---:|
| 1 | 7.44 - 9.87 | 6.44 - 8.87 |
| 64 | 7.70 - 10.50 | 6.70 - 9.50 |
| 256 | 8.46 - 12.42 | 7.46 - 11.42 |
| 1024 | 11.54 - 20.10 | 10.54 - 19.10 |

#### D) What to do immediately (implementation order)

1. Freeze current sparse decoder and benchmark precompute breakdown of F^3:
   - hash encode time,
   - scatter/aggregation time,
   - smoothing CNN time.
2. Build `F3-lite-v1`:
   - ds2 global pass,
   - reduced hash levels/channels,
   - reduced smoothing blocks.
3. Keep point-locked query branch unchanged (B1/B3/B7/B8) so exact-point fidelity is protected.
4. Re-benchmark with the same matrix and compare:
   - old baseline used here: $T_{\mathrm{F^3,core}}\approx8.33$ ms (120 Hz),
   - new: $T_{\mathrm{F^3lite}}$.
5. Only after this, retune query-side budgets $(R,U,H,L,M)$.

#### E) Acceptance criterion for "simplify F^3 first"

- Primary: reduce $T_{\text{precompute}}$ by at least 15% from the measured 120 Hz baseline setup.
- Secondary: keep query-point metrics within 3-5% relative of current GCQD-lite.

### 4.9 Non-acceptable variants removed

Removed from design:

- local-only MLP without global branch.
- event-neighborhood-only predictor with no non-local routing.

Reason:

- violates requirement to use non-local/global information for each query.

### 4.10 Cross-field mechanisms integrated to fix global-local separation

This section is not only evidence. It states exactly what was changed in the algorithm because of other fields.

1. Sparse-attention connectivity (NLP):
   - Source idea: local + global + random connectivity in sparse transformers.
   - Integrated change: B4 now uses $S_j=R_j\cup U_j$ (content-routed + coverage tokens) and always-visible anchors $G_t$.
   - Why it fixes separation: query cannot be trapped in only local or only top-score regions.
2. Latent bottleneck + inducing points (multimodal/set modeling):
   - Source idea: compress large input into small latent memory and decode with query tokens.
   - Integrated change: A3/A5 create $C_t$ and $G_t$; B4/B7 read both subset tokens and anchors.
   - Why it fixes separation: global context becomes available for every query with bounded cost.
3. Deformable sparse sampling (detection/segmentation):
   - Source idea: learn a few informative offsets instead of dense attention over all positions.
   - Integrated change: B5/B6 sparse multiscale deformable reads over $S_j$.
   - Why it fixes separation: local-global coupling happens through learned geometric reads, not dense map decoding.
4. Point-locked target branch (point refinement / implicit local models):
   - Source idea: keep a dedicated center-point representation and add context as residual correction.
   - Integrated change: B1/B3/B7/B8 explicitly keep $(f_{j,0},h_{\text{point},j},h_{\text{ctr},j})$ and fuse context as $h_{\text{fuse},j}=h_{\text{ctr},j}+g_j\odot h_{\text{ctx},j}$.
   - Why it fixes separation: target-point information cannot be overwritten by surrounding/global summaries.
5. Query-conditioned context retrieval (Neural Processes / retrieval models):
   - Source idea: target query attends only relevant context subset.
   - Integrated change: B4 computes query-conditioned global summary $\bar c_j$ over $\mathcal{V}_j$.
   - Why it fixes separation: global summary is specific to each query, not one static scene vector.
6. Smooth hash interpolation (F^3 / Instant-NGP):
   - Source idea: non-discrete coordinates must be interpolated on neighboring hash-grid vertices.
   - Integrated change: B2 uses interpolated hash lookup (3D trilinear with 8 vertices, or 2D bilinear fallback).
   - Why it fixes separation: coordinate encoding remains continuous and stable at sub-pixel query shifts.
7. Coordinate implicit decoding (INR / LIIF):
   - Source idea: output only requested coordinates using coordinate-conditioned decoding.
   - Integrated change: B1+B2+B3 build per-query token from point feature + local context + hash coordinate embedding.
   - Why it fixes separation: prediction stays sparse while preserving geometric location cues.
8. Adaptive compute allocation (token pruning / dynamic inference):
   - Source idea: spend extra compute only on hard instances.
   - Integrated change: B10 uncertainty-triggered second-hop rerouting with small $(R_2,U_2)$.
   - Why it fixes separation: hard queries can request more global evidence without slowing easy queries.
9. Context-drop regularization (stochastic context masking):
   - Source idea: randomly remove auxiliary context branches in training to force robust core signal usage.
   - Integrated change: in C6, with probability $p_{\text{ctxdrop}}$, set $h_{\text{ctx},j}\leftarrow 0$ before fusion.
   - Why it fixes separation: model cannot rely only on context and must keep $h_{\text{ctr},j}$ predictive.

Additional regularizers used in $L_{\text{comp}}$ (C5) to enforce coupling:
$$
L_{\text{cov}}=\mathrm{KL}\!\left(\frac{1}{K}\sum_{j=1}^{K}\alpha_j\ \middle\|\ \pi_t\right),\qquad
\pi_t=\mathrm{softmax}(W_\pi C_t)
$$
$$
L_{\text{align}}=\frac{1}{K}\sum_{j=1}^{K}\left\|W_c h_{\text{ctr},j}-W_g h_{\text{ctx},j}\right\|_1
$$
$$
L_{\text{budget}}=\frac{1}{K}\sum_{j=1}^{K}\mathbf{1}\big[\sigma_j>\tau\big],\qquad
L_{\text{comp}}=\lambda_{\text{cov}}L_{\text{cov}}+\lambda_{\text{align}}L_{\text{align}}+\lambda_{\text{budget}}L_{\text{budget}}
$$
Interpretation:

- $L_{\text{cov}}$ prevents routing collapse to a tiny repeated subset.
- $L_{\text{align}}$ keeps context correction compatible with the point-locked branch instead of adversarial to it.
- $L_{\text{budget}}$ discourages overuse of expensive second-hop refinement.

### 4.11 Primary references used in this design

- Deformable DETR: https://arxiv.org/abs/2010.04159
- Mask2Former: https://arxiv.org/abs/2112.01527
- Segment Anything: https://arxiv.org/abs/2304.02643
- Perceiver IO: https://arxiv.org/abs/2107.14795
- Set Transformer: https://arxiv.org/abs/1810.00825
- BigBird: https://arxiv.org/abs/2007.14062
- Longformer: https://arxiv.org/abs/2004.05150
- Routing Transformer: https://arxiv.org/abs/2003.05997
- Reformer: https://arxiv.org/abs/2001.04451
- Performer: https://arxiv.org/abs/2009.14794
- Linformer: https://arxiv.org/abs/2006.04768
- Nystromformer: https://arxiv.org/abs/2102.03902
- RETRO: https://arxiv.org/abs/2112.04426
- kNN-LM: https://arxiv.org/abs/1911.00172
- RAG: https://arxiv.org/abs/2005.11401
- GraphSAGE: https://arxiv.org/abs/1706.02216
- PinSage: https://arxiv.org/abs/1806.01973
- TokenLearner: https://arxiv.org/abs/2106.11297
- DynamicViT: https://arxiv.org/abs/2106.02034
- Neural Processes: https://arxiv.org/abs/1807.01622
- Attentive Neural Processes: https://arxiv.org/abs/1901.05761
- LIIF: https://arxiv.org/abs/2012.09161
- Meta-SR: https://arxiv.org/abs/1903.00875
- Instant-NGP: https://arxiv.org/abs/2201.05989
- NeRF: https://arxiv.org/abs/2003.08934

### 4.12 Critical audit of GCQD-v2 and corrected GCQD-v3

This subsection is a strict failure-mode review of the current algorithm. The goal is to remove weak parts and keep only mechanisms that are theoretically and practically defensible.

#### 4.12.1 Key bugs (ordered by impact)

| ID | Current step | Key bug | Why it is a real risk | Corrective action |
|---|---|---|---|---|
| P0-1 | B4 hard `TopR` routing | Train-inference mismatch from soft training vs hard top-k execution | Small score perturbations can abruptly swap selected tokens, causing unstable query outputs and calibration drift | Use differentiable sparse routing in training (straight-through top-k or Gumbel-top-k), keep hard top-k only at inference |
| P0-2 | B1.1 handcrafted local sampler | Isotropic fixed rings can miss motion-aligned evidence and thin structures | Event geometry is directional; fixed circular offsets are not aligned to edge/motion orientation | Replace fixed-only outer sampling with anisotropic local sampling guided by local structure tensor |
| P0-3 | B2 + A6 temporal signal | A single recency scalar `\tau_j` is too weak for temporal ambiguity | Event depth is temporally ambiguous in low-texture/low-rate windows; recurrent event-depth papers rely on temporal memory | Add low-cost temporal latent memory across windows and let queries read from it |
| P0-4 | B9 global-only calibration `(s_t,b_t)` | One global affine calibration cannot model spatially varying bias | Depth scale/bias errors are often region-dependent (near/far, texture/noise zones) | Predict per-query calibration `(s_j,b_j)` with strong regularization to global priors |
| P1-1 | B4 deterministic hash coverage `U_j` | Coordinate-hash coverage is content-agnostic | It guarantees access, but not relevance; can waste budget on unrelated regions | Replace with learned landmark coverage (diversity-selected anchors) |
| P1-2 | B6 hard `ClampToValid` | Border clamping creates biased repeated reads | Out-of-bound offsets collapse to borders, reducing effective sampling diversity | Predict bounded offsets in normalized coordinates and use reflective padding/grid-sample |
| P1-3 | Per-query independent decoding | No explicit cross-query coherence | Nearby queried points can become inconsistent despite same scene | Add query-set consistency regularization during training |

Evidence alignment from literature:

- Deformable DETR: sparse sampling works if reference/routing points are well-conditioned, not brittle hard switches.
- Longformer/BigBird: sparse connectivity works best with reliable local+global structure, not arbitrary fallback alone.
- Perceiver IO: compact latent bottleneck is effective, but query access path must remain stable.
- Event depth recurrent lines (e.g., dense monocular event depth and RAM-Net): temporal memory is important for depth disambiguation.

#### 4.12.2 Corrected runtime algorithm (GCQD-v3)

The corrected design keeps one shared precompute and one sparse query pass, but replaces unstable components.

Step V1 (shared event precompute, unchanged backbone philosophy):
$$
E_t\rightarrow F_t^{(1:3)},\ C_t
$$
Keep F^3-style sparse-event representation as the compute foundation.

Step V2 (temporal latent memory, new):
$$
H_t=\eta H_{t-1}+(1-\eta)\,\mathrm{Pool}(C_t),\qquad \eta\in[0,1)
$$
`H_t` is a compact temporal context state (few tokens) shared by all queries in window `t`.

Step V3 (query encoding with point lock + anisotropic local sampling):
$$
f_{j,0}=\mathrm{Bilinear}(F_t^{(1)},q_j),\qquad e_j=e(q_j),\qquad h_{\text{point},j}=W_p[f_{j,0};e_j]
$$
Build local sampling directions from a structure tensor:
$$
J_j=\sum_{\delta\in\Omega_{\text{core}}}w_\delta\,\nabla F_t^{(1)}(q_j+\delta)\nabla F_t^{(1)}(q_j+\delta)^\top
$$
Let eigenvectors of `J_j` be `(v_{j,1},v_{j,2})`; sample anisotropic offsets:
$$
\delta_{a,b}=a\,r_1 v_{j,1}+b\,r_2 v_{j,2},\quad (a,b)\in\mathcal{A}
$$
This keeps local reads aligned with dominant local geometry/motion direction.

Step V4 (stable sparse routing with learned coverage):
$$
\tilde\alpha_j=\mathrm{softmax}\!\left(\frac{(W_q z_j)(W_k C_t)^\top}{\sqrt d\,\tau_r}\right)
$$
Use differentiable top-k during training and hard top-k at inference:
$$
R_j=\mathrm{TopR}(\tilde\alpha_j)
$$
Coverage is from learned landmarks (not coordinate hash):
$$
U_j=\mathrm{TopU}\!\left(\mathrm{softmax}\!\left(\frac{(W_q^u z_j)(W_k^u L_t)^\top}{\sqrt d}\right)\right),\qquad L_t=\mathrm{LandmarkPool}(C_t)
$$
$$
S_j=R_j\cup U_j
$$

Step V5 (sparse deformable multiscale read with bounded offsets):
$$
\Delta p_{r,h,\ell,m}=\rho_\ell\tanh\!\big(W^\Delta_{h,\ell,m}u_{j,r}+b^\Delta_{h,\ell,m}\big)
$$
$$
f_{r,h,\ell,m}=\mathrm{GridSample}_{\text{reflect}}(F_t^{(\ell)},\ p_{q_j}^{(\ell)}+p_r^{(\ell)}+\Delta p_{r,h,\ell,m})
$$
Reflective sampling avoids border-collapse artifacts from hard clamping.

Step V6 (single-pass coupled fusion with temporal memory read):
$$
\bar c_j=\mathrm{Attn}(z_j,\{c_i\}_{i\in S_j}\cup L_t),\qquad \bar h_j=\mathrm{Attn}(z_j,H_t)
$$
$$
h_{\text{ctx},j}=\mathrm{LN}(W_{\text{ctx}}[h_{\text{def},j};\bar c_j;\bar h_j;l_j^{\text{ctx}}]+b_{\text{ctx}})
$$
$$
h_{\text{fuse},j}=h_{\text{ctr},j}+\sigma(\mathrm{MLP}_g([h_{\text{ctr},j};h_{\text{ctx},j};z_j]))\odot h_{\text{ctx},j}
$$
This keeps exact-point information as base and adds context as residual.

Step V7 (query-wise calibration, replacing global-only calibration):
$$
s_j=\mathrm{softplus}(w_s^\top[h_{\text{fuse},j};\bar c_j]+b_s),\qquad b_j=w_b^\top[h_{\text{fuse},j};\bar c_j]+b_b
$$
$$
\rho_j=s_j\,r_j+b_j,\qquad \hat d_j=\frac{1}{\mathrm{softplus}(\rho_j)+\varepsilon}
$$
Regularize to global prior to avoid overfitting:
$$
L_{\text{cal}}=\frac{1}{K}\sum_{j=1}^{K}\Big((s_j-s_t)^2+\lambda_b(b_j-b_t)^2\Big)
$$

Step V8 (uncertainty-gated optional second hop with smooth training gate):
$$
p_j^{\text{hop}}=\sigma\!\left(\frac{\sigma_j-\tau_{\text{hop}}}{T_{\text{hop}}}\right)
$$
Train with expected compute penalty:
$$
L_{\text{budget}}=\frac{1}{K}\sum_{j=1}^{K}p_j^{\text{hop}}
$$
Use hard threshold only at inference for deterministic runtime policies.

#### 4.12.3 Additional loss terms required for this corrected design

Routing stability:
$$
L_{\text{route-stab}}=\frac{1}{K}\sum_{j=1}^{K}\mathrm{KL}\!\left(\tilde\alpha_j\ \middle\|\ \tilde\alpha'_j\right)
$$
Point-preservation:
$$
L_{\text{ctr}}=\frac{1}{K}\sum_{j=1}^{K}\mathrm{Huber}\!\big(r_j^{\text{ctr}}-\rho_j^\star\big)
$$
Query-graph consistency:
$$
L_{\text{q-cons}}=\frac{1}{|E_Q|}\sum_{(i,j)\in E_Q}w_{ij}\,\mathrm{Huber}\!\Big((\hat\rho_i-\hat\rho_j)-(\rho_i^\star-\rho_j^\star)\Big)
$$
Soft compute budget:
$$
L_{\text{budget}}=\frac{1}{K}\sum_{j=1}^{K}p_j^{\text{hop}}
$$
Optional residual-calibration regularizer (hypothesis mode only):
$$
L_{\text{cal-res}}=\frac{1}{K}\sum_{j=1}^{K}\left(\Delta s_j^2+\lambda_b\Delta b_j^2\right)
$$
Updated objective:
$$
\mathcal{L}_{\text{v3.1}}=\mathcal{L}_{\text{base}}+\lambda_{\text{route}}L_{\text{route-stab}}+\lambda_{\text{ctr}}L_{\text{ctr}}+\lambda_{\text{q}}L_{\text{q-cons}}+\lambda_{\text{budget}}L_{\text{budget}}+\lambda_{\text{cal}}L_{\text{cal-res}}
$$

#### 4.12.4 Complexity impact of corrections

Relative to GCQD-v2:

- Added small cost: temporal memory read (`O(d M_t)` per query), anisotropic offset generation, calibration head.
- Removed instability cost: fewer brittle reroutes and fewer wasted coverage tokens.
- Net effect: minor per-query overhead increase, but higher accuracy stability and fewer catastrophic misses.

Expected runtime form remains:
$$
T_{\text{total}}(K)=T_{\text{precompute}}+K\,T_{\text{query}}+\mathcal{O}(K_{\text{hard}})
$$
where `K_hard` is uncertainty-triggered second-hop count.

Conclusion of critical audit:

- The current GCQD-v2 is directionally correct but has four P0 issues (routing discontinuity, weak temporal memory, brittle local sampling, over-global calibration).
- GCQD-v3 fixes these while preserving the main project constraint: no dense full-image depth decoding at inference.

## 5) Incorporating DINO and Instant-NGP (practical plan)

### 5.1 Instant-NGP transfer (high priority)

Use hash-grid encoding as a core part of the query stage, not as an optional add-on.

Query embedding:

$$
e(q)=\operatorname{Concat}_{s=1}^{S}\!\left(\sum_{\nu\in\{0,1\}^3}w_{s,\nu}\,T_s[h_s(\mathbf{b}_s+\nu)]\right),\qquad
z_q=W_z[f_{q,0};l_q^{\text{ctx}};e(q)].
$$

Recommended configuration:

- levels $S\in[12,16]$
- features per level $F\in\{2,4\}$
- hash size per level $2^{17}$ to $2^{19}$
- FP16 tables with fused lookup kernel

Design rule:

- keep hash encoding and query MLP fully in the query path,
- keep smooth interpolated hash lookup (trilinear-8 if 3D, bilinear-4 if 2D); do not use nearest-vertex lookup,
- keep F^3 backbone and pyramid generation in precompute path,
- do not move dense operations into query path.

### 5.2 DINO/DINOv2 transfer (medium priority)

Use DINO as a training-time prior to improve hard-query accuracy without inference overhead.

Compute a semantic saliency map $s(q)$ offline (or on training GPU only), then:

1. Query sampling distribution:
   $$
   p_{\text{train}}(q)=\alpha p_{\text{uniform}}(q)+\beta p_{\text{event}}(q)+\gamma p_{\text{saliency}}(q)
   $$
   where $p_{\text{saliency}}(q)\propto s(q)$.
2. Boundary-aware weighting:
   $$
   w(q)=1+\eta\,\|\nabla s(q)\|
   $$
   applied to pointwise depth loss.
3. Optional teacher distillation weight:
   $$
   \lambda_{\text{distill}}(q)=\lambda_0\big(1+\eta_d\|\nabla s(q)\|\big).
   $$

Do not run DINO in deployment-time inference.

## 6) Training objectives for sparse query depth

Let $\hat d(q)$ be predicted depth at query $q$, $d^\star(q)$ target depth, and $M(q)\in\{0,1\}$ a validity mask.

Define inverse depth $\rho(q)=1/d(q)$ for stable optimization.

Recommended loss stack:

1. Pointwise robust inverse-depth loss:
   $$
   L_{\text{point}}=\frac{1}{|Q_v|}\sum_{q\in Q_v}w(q)\,\mathrm{Huber}\!\big(\hat \rho(q)-\rho^\star(q)\big)
   $$
   where $Q_v=\{q\in Q:M(q)=1\}$.
2. Query-space SiLog:
   $$
   L_{\text{silog},q}=\frac{1}{|Q_v|}\sum_{q\in Q_v}\delta_q^2-\lambda_{\text{si}}\left(\frac{1}{|Q_v|}\sum_{q\in Q_v}\delta_q\right)^2
   $$
   where $\delta_q=\log \hat d(q)-\log d^\star(q)$.
3. Non-local pairwise ordinal loss:
   $$
   L_{\text{rank}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\log\!\Big(1+\exp\!\big(-y_{ij}(\hat\rho_i-\hat \rho_j)\big)\Big)
   $$
   with $y_{ij}=\mathrm{sign}(\rho_i^\star-\rho_j^\star)$ and far-apart pairs $(i,j)$.
4. Uncertainty NLL:
   $$
   L_{\text{unc}}=\frac{1}{|Q_v|}\sum_{q\in Q_v}\left(\frac{(\hat d(q)-d^\star(q))^2}{2\sigma(q)^2}+\frac{1}{2}\log \sigma(q)^2\right).
   $$
5. Optional teacher distillation:
   $$
   L_{\text{distill}}=\frac{1}{|Q|}\sum_{q\in Q}\lambda_{\text{distill}}(q)\big\|\hat d(q)-d_{\text{teacher}}(q)\big\|
   $$
6. Coupling and routing regularizer:
   $$
   L_{\text{cov}}=\mathrm{KL}\!\left(\frac{1}{|Q|}\sum_{q\in Q}\alpha_q\ \middle\|\ \pi_t\right),\qquad
   \pi_t=\mathrm{softmax}(W_\pi C_t)
   $$
   $$
   L_{\text{align}}=\frac{1}{|Q|}\sum_{q\in Q}\left\|W_c h_{\text{ctr},q}-W_g h_{\text{ctx},q}\right\|_1
   $$
   $$
   L_{\text{budget}}=\frac{1}{|Q|}\sum_{q\in Q}\mathbf{1}\big[\sigma(q)>\tau\big],\qquad
   L_{\text{comp}}=\lambda_{\text{cov}}L_{\text{cov}}+\lambda_{\text{align}}L_{\text{align}}+\lambda_{\text{budget}}L_{\text{budget}}
   $$
7. Point-lock auxiliary loss:
   $$
   L_{\text{ctr}}=\frac{1}{|Q_v|}\sum_{q\in Q_v}\mathrm{Huber}\!\big(r_q^{\text{ctr}}-\rho^\star(q)\big)
   $$

Total:

$$
\mathcal{L}=\lambda_1 L_{\text{point}}+\lambda_2 L_{\text{silog},q}+\lambda_3 L_{\text{rank}}+\lambda_4 L_{\text{unc}}+\lambda_5 L_{\text{distill}}+\lambda_6 L_{\text{comp}}+\lambda_7 L_{\text{ctr}}.
$$

For metric fine-tuning, keep LiDAR validity filtering analogous to current code.

## 7) Query sampling strategy (critical)

During training, sample queries from mixed distributions:

- Uniform random (coverage)
- Event-dense regions (high signal)
- High-gradient/edge regions (boundary quality)
- Far-field and low-event regions (hard negatives)

At inference, user supplies queries directly.

## 8) Evaluation protocol

### 8.1 Accuracy metrics (query-level)

At queried pixels only:

- AbsRel
- RMSE
- RMSE_log
- SiLog
- delta thresholds

Also report by depth range buckets (near/mid/far).

### 8.2 Runtime metrics (the key contribution)

Report both amortized and per-query behavior:

- preprocessing latency $T_{\text{precompute}}$
- query latency $T_{\text{query}}(K)$ for
  $$
  |Q|\in\{1,4,16,64,256,1024\}
  $$
- total latency
  $$
  T_{\text{total}}(K)=T_{\text{precompute}}+T_{\text{query}}(K)
  $$

For each:

- end-to-end latency (ms)
- throughput (queries/s)
- speedup vs dense baseline doing same query extraction

Also report memory usage and event-rate sensitivity.

### 8.3 Robustness matrix

Evaluate across:

- day/night
- different platforms (car/spot/flying if available)
- event subsampling rates

Use same spirit as F^3 robustness experiments.

### 8.4 Required ablations for a strong paper

At minimum, include:

- Effect of query count on latency and accuracy.
- Effect of query sampling policy during training.
- Effect of decoder size (tiny/medium/large).
- Effect of freezing vs fine-tuning F^3 backbone.
- Effect of distillation loss weight.
- Effect of hash levels / feature dimensions.
- Effect of coarse token count $P_c$ and routed region count $R$.
- Effect of coverage-token count $U$ and anchor count $M_a$.
- Effect of local sampler design (`81 fixed`, `121 fixed`, `121+adaptive169`).
- Effect of deformable sampling budget $(H,L,M)$.
- Effect of fusion structure (`naive concat-gate` vs `point-locked residual fusion`).
- Effect of uncertainty threshold $\tau$ and refinement on/off.
- Effect of coarse routing (`disabled` vs `enabled`).

## 9) Semester timeline (16 weeks)

### Weeks 1-2: Foundations + reproduction

- Read F^3 deeply and replicate baseline depth runs.
- Build profiling harness for latency decomposition.

### Weeks 3-4: Sparse benchmark setup

- Add query-level dataloader/eval tools.
- Produce baseline curves: dense runtime vs query count.

### Weeks 5-7: Core GCQD implementation

- Implement shared preprocessing cache (`F_t^{(1:3)}`, `C_t`, `G_t`, `M_t`, `R_t`, scale heads) and query decoder.
- Implement coarse routing + coverage tokens + deformable sampling + point-locked residual fusion.
- Train relative + metric query losses.
- First report on accuracy and amortized runtime ($T_{\text{precompute}}$, $T_{\text{query}}$, $T_{\text{total}}$).

### Weeks 8-10: Theory-driven ablations

- Local-only vs global-context query decoder (must show global context benefit).
- Coarse-routing and deformable budget ablation ($P_c$, $R$, $H$, $L$, $M$).
- Adaptive refinement ablation (with/without uncertainty-triggered second pass).
- Global scale head ablation.

### Weeks 11-12: Distillation and DINO options

- Add teacher-student distillation.
- Add optional DINO-guided boundary weighting.

### Weeks 13-14: Robustness + ablations

- Query sampling ablation.
- Loss ablation.
- Hash-level/channel ablation.

### Weeks 15-16: Finalize paper package

- Main tables/figures.
- Failure case analysis.
- Draft submission-ready method + experiments sections.

## 10) Concrete implementation tasks in this repo

### 10.1 New modules to add

Suggested new files:

- `src/f3/tasks/depth_sparse/` (new task folder)
- `src/f3/tasks/depth_sparse/models/query_depth.py`
- `src/f3/tasks/depth_sparse/models/coarse_router.py`
- `src/f3/tasks/depth_sparse/models/anchor_pool.py`
- `src/f3/tasks/depth_sparse/models/deformable_query.py`
- `src/f3/tasks/depth_sparse/models/coupling_fusion.py`
- `src/f3/tasks/depth_sparse/models/hash_encoder.py`
- `src/f3/tasks/depth_sparse/models/uncertainty_refine.py`
- `src/f3/tasks/depth_sparse/utils/dataloader.py`
- `src/f3/tasks/depth_sparse/utils/query_sampler.py`
- `src/f3/tasks/depth_sparse/utils/runtime_profiler.py`
- `src/f3/tasks/depth_sparse/train.py`
- `src/f3/tasks/depth_sparse/validate.py`
- `src/f3/tasks/depth_sparse/benchmark.py`

### 10.2 Reuse existing components

- F^3 backbone init/load from current `init_event_model` and checkpoints.
- Existing depth losses (SiLog/SiLogGrad) as templates.
- Existing benchmarking style from `test_speed.py`.

### 10.3 Fast path engineering

- Keep single-batch deployment mode in mind (similar to `hubconf.py` single-batch path).
- Consider optional precomputed hashed features for ultra-low latency fixed-window deployments.

## 11) Risks and mitigation

Risk 1: Accuracy drop in low-event or textureless regions.

- Mitigation: teacher distillation + uncertainty estimation + mixed query sampling.

Risk 2: Sparse model overfits query distribution.

- Mitigation: randomized query curriculum and robust validation splits.

Risk 3: Runtime not significantly better than dense for medium Q.

- Mitigation: optimize routing/coverage and deformable sampling kernels; tune $P_c$, $R$, $U$, $M_a$, $H$, $L$, $M$.

Risk 4: Label sparsity from LiDAR hurts training.

- Mitigation: two-stage pseudo->metric strategy (already validated in F^3 depth pipeline).

## 12) Expected deliverables

By semester end, target:

1. A working sparse query-point depth model from events.
2. Runtime scaling plots vs query count showing clear win over dense methods.
3. Accuracy tables on M3ED/DSEC/MVSEC query-level metrics.
4. Ablations proving which components matter.
5. Draft paper/report with method, theory intuition, and deployment relevance.

## 13) Immediate next 2-week action list

1. Reproduce dense F^3+DA-V2 baseline and verify metrics/runtime on one dataset split.
2. Implement query benchmark wrapper that samples points from dense outputs and LiDAR valid pixels.
3. Implement GCQD-v3 prototype with coarse routing + coverage tokens + anchor bank + point-locked residual fusion.
4. Produce first plots for $T_{\text{precompute}}$, $T_{\text{query}}$, $T_{\text{total}}$ and first query-accuracy comparison.

---

This plan directly follows your constraint (no full dense NN at inference for sparse requests) while leveraging what already works in F^3.
