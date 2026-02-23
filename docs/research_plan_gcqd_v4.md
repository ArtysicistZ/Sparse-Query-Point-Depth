# Research Plan: "Inverse F^3" for Sparse Query-Point Event Depth

Author: Codex
Date: 2026-02-13
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

This section is the implementation spec for GCQD-v3.1. The logic is strictly runtime-ordered:

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
| $G_t\in\mathbb{R}^{M_a\times d}$ | $G_t=\mathrm{AnchorPool}(C_t)$ | Always-visible global anchor tokens | B4 (default summary path), B8 (via summary token) | Yes |
| $h_s,h_b$ | Heads on $C_t$ | Global calibration heads | A4, B9 | Yes |
| $s_t,b_t$ | $s_t=\mathrm{softplus}(h_s(\mathrm{Pool}(C_t))),\ b_t=h_b(\mathrm{Pool}(C_t))$ | Scale/shift for calibration | B9 | Yes (through heads) |
| $q=(u,v)$ | User input coordinate | Pixel to predict depth at | B1-B11 | No |
| $f_{q,0}$ | $\operatorname{Bilinear}(F_t^{(1)},q)$ | Exact feature at queried pixel | B1, B3, B5, B7, B8, B9 | No |
| $l_q^{\text{ctx}}$ | LocalSample near descriptor around $q$ from $F_t^{(1)}$ (excluding center) | Near-surrounding context near query | B3, B5, B7, B8 | Sampling: No; local projection MLP: Yes |
| $l_q^{\text{ring}}$ | <span style="color:#BDBDBD">Ring-pooled descriptor from LocalSample (optional)</span> | <span style="color:#BDBDBD">Structured local context by distance bands</span> | <span style="color:#BDBDBD">B3, B5, B7, B8</span> | <span style="color:#BDBDBD">Yes (through local heads)</span> |
| $m_q$ | <span style="color:#BDBDBD">$m_q=\sigma(w_m^\top[f_{q,0};l_q^{\text{ring}}]+b_m)$ (optional)</span> | <span style="color:#BDBDBD">Local reliability score for center-surround evidence</span> | <span style="color:#BDBDBD">B4, B5, B8</span> | <span style="color:#BDBDBD">Yes</span> |
| $M_t$ | $M_t=\alpha\,\mathrm{Norm}(\mathrm{Pool}(\lvert E_t\rvert))+\beta\,\mathrm{Norm}(\lVert\nabla F_t^{(1)}\rVert_1)$ | Local importance map for query-centric sampling | B1.1 | No |
| $R_t$ | $R_t=\mathrm{Norm}(\mathrm{Pool}(\mathrm{LastTS}(E_t)))$ | Local temporal recency auxiliary map for hash interpolation | B2 | No |
| $H_t$ | $H_t=\mathrm{ProjMem}(\mathrm{GRU}(H_{t-1},\mathrm{Pool}(C_t)))$ | Compact temporal memory state across windows | B4, B8 | Yes |
| $N_{\text{loc}}$ | Number of sampled local points around query (adaptive) | Local evidence budget per query | B1.1, 4.7 complexity | No |
| $e(q)$ | Hash-grid coordinate encoding | Multiscale positional code | B3 | Yes (hash tables) |
| $z_q$ | $z_q=W_z[f_{q,0};l_q^{\text{ctx}};e(q)]$ (<span style="color:#BDBDBD">optional add $l_q^{\text{ring}}$</span>) | Query token (point+local+position) | B4, B5, B8 | Yes |
| $h_{\text{point},q}$ | Point-locked branch from $f_{q,0}$ and $e(q)$ | Target-pixel identity representation | B3, B7, B8, B9 | Yes |
| $h_{\text{ctr},q}^{0}$ | Early nonlinear center token from $(h_{\text{point},q},l_q^{\text{ctx}})$ (<span style="color:#BDBDBD">optional add $l_q^{\text{ring}}$</span>) | Center token before routing/sampling | B4, B5, B7 | Yes |
| $\alpha_q$ | Attention logits/softmax over $C_t$ | Coarse-region relevance | B4 | Yes |
| $R_q$ | Top-$R$ from $\alpha_q$ | Routed coarse regions | B5-B7 | No (selection op) |
| $U_q$ | LandmarkTopU$(z_q,L_t)$ | Coverage tokens selected from learned landmark bank | B4-B7 | Yes (through landmark pool/projections) |
| $i_q^{\text{loc}}$ | Nearest coarse index to query location | Guaranteed query-neighbor coarse anchor | B4-B7 | No |
| $S_q$ | $S_q=R_q\cup U_q\cup\{i_q^{\text{loc}}\}$ | Spatial token subset for deformable reads (content + coverage + query-neighbor anchor) | B5-B7 | No |
| $\bar c_q$ | Attention summary over $S_q\cup G_t$ | Query-conditioned global scene summary | B4-B10 | Yes (through attention projections) |
| $\Delta p_{q,r,h,\ell,m}$ | Offset head | Learned sampling displacement | B6 | Yes |
| $a_{q,r,h,\ell,m}$ | Weight head + softmax | Sampling importance | B7 | Yes |
| $h_{\text{near}}$ | Aggregated near-surround branch from sampled local neighborhoods | Near-field context around queried pixel | B7, B8 | Yes |
| $h_{\text{global}}$ | Global branch from deformable all-sample aggregation and summary token | Non-local query context | B7, B8 | Yes (upstream) |
| $T_q$ | $[\mathrm{Reshape}(W_{\text{near}}^{\text{tok}}h_{\text{near},q});\ \mathrm{Reshape}(W_{\text{glb}}^{\text{tok}}h_{\text{global},q});\ W_{\text{sum}}\bar c_q;\ W_{\text{mem}}H_t]$ | Compact query-specific token set (near+global+summary+temporal) | B8 | Yes |
| $h_{\text{ctr}}^{+}$ | Center token after query-to-context attention update | Coupled center representation | B8, B9 | Yes |
| $h_{\text{ctx}}$ | Context branch output after center-context coupling (<span style="color:#BDBDBD">optional reciprocal write-back</span>) | Context correction representation | B8 | Yes |
| $g_q$ | Gate head + sigmoid | Local/global mixing ratio | B8 | Yes |
| $\lambda_{\text{ctr},q},\lambda_{\text{ctx},q}$ | Bounded carry gate from $(h_{\text{ctr},q}^{+},h_{\text{ctx},q},z_q)$ with $\lambda_{\text{ctr},q}\in[\lambda_{\min},\lambda_{\max}]$ | Explicit center-context balance weights | B8, 4.5A | Yes |
| $h_{\text{fuse}}$ | Gated fusion output | Final query representation | B9, B10 | No (fusion op) |
| $r_q$ | Relative disparity head output | Uncalibrated depth code | B9 | Yes |
| $\rho_q$ | $\rho_q=s_t r_q+b_t$ (default), <span style="color:#BDBDBD">optional residual form in B9</span> | Calibrated inverse-depth/disparity | B9 | No (algebraic op) |
| $\hat d_q$ | $\hat d_q=1/(\mathrm{softplus}(\rho_q)+\varepsilon)$ | Final depth at query | Output | No (conversion op) |
| $\sigma_q$ | Uncertainty head output | Confidence / ambiguity estimate | B10 | Yes |

### 4.3 Algorithm A: precompute once per event window

Inputs:

- Event set $E_t$.

Outputs (cache):

- $\text{cache}_t=\{F_t^{(1)},F_t^{(2)},F_t^{(3)},C_t,G_t,M_t,R_t,H_t,s_t,b_t\}$.

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

- $\mathrm{Proj}_{1\times1}$ is trainable when implemented as learned projection layers.
- $\mathrm{Down}_2$ is non-trainable for pooling variants and trainable for convolutional downsampling variants.

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
s_t=\mathrm{softplus}(h_s(\mathrm{Pool}(C_t))),\qquad
b_t=h_b(\mathrm{Pool}(C_t))
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

- B4 (summary token construction), B8 (through summary-conditioned context coupling).

Trainable:

- Yes (`AnchorPool` / $W_A$).

#### A6) Cache state

Compute:
$$
M_t=\alpha\,\mathrm{Norm}\!\big(\mathrm{Pool}(|E_t|)\big)+\beta\,\mathrm{Norm}\!\big(\|\nabla F_t^{(1)}\|_1\big),\qquad
R_t=\mathrm{Norm}\!\big(\mathrm{Pool}(\mathrm{LastTS}(E_t))\big)
$$
$$
H_t=\mathrm{ProjMem}\!\left(\mathrm{GRU}\!\left(H_{t-1},\mathrm{Pool}(C_t)\right)\right)
$$
$$
\text{cache}_t=\{F_t^{(1)},F_t^{(2)},F_t^{(3)},C_t,G_t,M_t,R_t,H_t,s_t,b_t\}
$$

Meaning:

- Shared runtime state reused by many query batches.
- $M_t$ is a precomputed local-importance map reused by `LocalSample` for fast adaptive local selection.
- $R_t$ is a precomputed event-recency map reused by hash embedding (B2) to add local temporal phase.
- $H_t$ is a compact temporal memory state reused by all queries in this window.
- $\mathrm{Pool}(|E_t|)$ means event-count rasterization of the current window, resized/pool-aligned to $F_t^{(1)}$ resolution.

Used next:

- Algorithm B.

Trainable:

- Partially:
- $M_t$ and $R_t$ branches are non-trainable (deterministic preprocessing).
- $H_t$ branch is trainable through `GRU` and `ProjMem`.

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
<span style="color:#BDBDBD">Optional richer local descriptors:</span>
$$
\color{#BDBDBD}{\left(l_j^{\text{ring}},m_j\right)=\mathrm{LocalRingAndReliability}(F_t^{(1)},M_t,q_j)}
$$

Meaning:

- $f_{j,0}$ is exact target-point evidence at the queried pixel.
- $l_j^{\text{ctx}}$ is neighborhood context around that target point.
- <span style="color:#BDBDBD">$l_j^{\text{ring}}$ is distance-structured local context (near to far in local window).</span>
- <span style="color:#BDBDBD">$m_j$ is local reliability used to coordinate local/global budget downstream.</span>

Used next:

- B3 query token and center branch, B5 conditioning, B8 fusion.
- <span style="color:#BDBDBD">If enabled, $m_j$ also conditions B4/B5.</span>

Trainable:

- Sampling op: No.
- `MLP_local`: Yes.
- <span style="color:#BDBDBD">Optional local heads (`MLP_ring`, reliability head): Yes.</span>

#### B1.1) LocalSample helper function (explicit runtime definition)

Use a coordinated local strategy with a lightweight default: fixed dense core + learned offsets for $l_j^{\text{ctx}}$. Optional ring pooling is kept only for ablations.

Inputs:

- query pixel $q_j=(u_j,v_j)$.
- fine feature map $F_t^{(1)}\in\mathbb{R}^{H_1\times W_1\times C}$.
- local-importance map $M_t$ (used only for <span style="color:#BDBDBD">optional hard-query expansion</span>).
- local seed token $z_j^{\text{loc}}=W_{\text{loc-seed}}[f_{j,0};\phi(q_j)]$.

Use query coordinates in level-1 feature space:
$$
\tilde u_j=\frac{u_j}{s_1},\qquad \tilde v_j=\frac{v_j}{s_1}
$$

Default fixed core:
$$
\Omega_{\text{core}}=\{(d_x,d_y)\mid d_x,d_y\in\{-4,-3,\dots,4\},\ |\Omega_{\text{core}}|=81\}
$$
Learned adaptive offsets:
$$
\Delta u_{j,m}=r_{\text{loc}}\tanh(W^{\text{loc}}_m z_j^{\text{loc}}+b^{\text{loc}}_m),\qquad m=1,\dots,M_{\text{loc}}
$$
Default budget:
$$
M_{\text{loc}}=40,\qquad N_{\text{loc}}=81+40=121
$$
Build local set:
$$
\Omega_j=\Omega_{\text{core}}\cup\{\Delta u_{j,m}\}_{m=1}^{M_{\text{loc}}}
$$
<span style="color:#BDBDBD">Optional ring partition on local offsets:</span>
$$
\color{#BDBDBD}{\Omega_{j,b}=\{\delta\in\Omega_j\setminus\{(0,0)\}\mid \rho_{b-1}<\|\delta\|_{\infty}\le \rho_b\},\qquad
0=\rho_0<\rho_1<\cdots<\rho_{B_r}}
$$
<span style="color:#BDBDBD">Optional hard-query expansion (ablation mode):</span>
$$
\color{#BDBDBD}{\bar m_j=\frac{1}{81}\sum_{\delta\in\Omega_{\text{core}}}\operatorname{Bilinear}(M_t,\tilde u_j+\delta_x,\tilde v_j+\delta_y)}
$$
$$
\color{#BDBDBD}{\bar m_j<\tau_{\text{low}}\Rightarrow M_{\text{loc}}\leftarrow M_{\text{loc}}+48,\quad N_{\text{loc}}=169}
$$
Sample and aggregate:
$$
f_{j,\delta}=\operatorname{Bilinear}(F_t^{(1)},\tilde u_j+\delta_x,\tilde v_j+\delta_y),\quad \delta\in\Omega_j
$$
$$
g_j^{\text{ctx}}=\operatorname{Concat}_{\delta\in\Omega_j\setminus\{(0,0)\}}\big[f_{j,\delta};\phi(\delta)\big],\qquad
l_j^{\text{ctx}}=\operatorname{MLP}_{\text{local}}(g_j^{\text{ctx}})
$$
<span style="color:#BDBDBD">Optional ring/reliability branch:</span>
$$
\color{#BDBDBD}{r_{j,b}=\frac{1}{|\Omega_{j,b}|}\sum_{\delta\in\Omega_{j,b}}f_{j,\delta},\qquad
l_j^{\text{ring}}=\operatorname{MLP}_{\text{ring}}\!\big([r_{j,1};\ldots;r_{j,B_r}]\big)}
$$
$$
\color{#BDBDBD}{m_j=\sigma\!\left(w_m^\top\!\left[\mathrm{LN}(f_{j,0});\mathrm{LN}(l_j^{\text{ring}})\right]+b_m\right)}
$$

Meaning:

- This guarantees one exact center feature plus dense local evidence.
- <span style="color:#BDBDBD">Ring pooling gives structured near-vs-far local context, not just one flattened local vector.</span>
- <span style="color:#BDBDBD">$m_j$ quantifies whether this query can rely more on local evidence or should request stronger global support.</span>

Used next:

- B3 (query token and center branch), B5 (sampling plan), and B8 (fusion).
- <span style="color:#BDBDBD">If enabled, ring/reliability descriptors also feed B4 and B5.</span>

Trainable:

- Learned local offset head $\{W_m^{\text{loc}},b_m^{\text{loc}}\}$: Yes.
- Bilinear sampling op and set union: No.
- $\operatorname{MLP}_{\text{local}}$: Yes.
- <span style="color:#BDBDBD">Optional $\operatorname{MLP}_{\text{ring}}$ and reliability head $(w_m,b_m)$: Yes.</span>

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
<span style="color:#BDBDBD">Optional local-structure normalization:</span>
$$
\color{#BDBDBD}{\tilde l_j^{\text{ring}}=\mathrm{LN}(l_j^{\text{ring}})}
$$
$$
z_j=W_z[\tilde f_{j,0};\tilde l_j^{\text{ctx}};\tilde e_j]+b_z,\qquad
h_{j}^{\text{pe}}=[\tilde f_{j,0};\tilde e_j]
$$
<span style="color:#BDBDBD">Optional richer token:</span>
$$
\color{#BDBDBD}{z_j=W_z[\tilde f_{j,0};\tilde l_j^{\text{ctx}};\tilde l_j^{\text{ring}};\tilde e_j]+b_z}
$$
$$
h_{\text{point},j}=\mathrm{LN}\!\left(W_{\text{skip}}h_{j}^{\text{pe}}+W_{p2}\,\mathrm{GELU}(W_{p1}h_{j}^{\text{pe}}+b_{p1})+b_{p2}\right),\qquad
z_j\in\mathbb{R}^{d},\ h_{\text{point},j}\in\mathbb{R}^{d}
$$
$$
u_{\text{ctr},j}=\mathrm{GELU}\!\left(W_{c1}[h_{\text{point},j};\tilde l_j^{\text{ctx}}]+b_{c1}\right)
$$
$$
h_{\text{ctr},j}^{0}=\mathrm{LN}\!\left(h_{\text{point},j}+W_{c3}u_{\text{ctr},j}+b_{c3}\right)
$$
<span style="color:#BDBDBD">Optional multiplicative center enhancer:</span>
$$
\color{#BDBDBD}{u_{\text{ctr},j}^{\text{mul}}=\mathrm{GELU}\!\left(W_{c1}[h_{\text{point},j};\tilde l_j^{\text{ctx}}]+b_{c1}\right)\odot \mathrm{GELU}\!\left(W_{c2}\tilde l_j^{\text{ring}}+b_{c2}\right),\quad
h_{\text{ctr},j}^{0}=\mathrm{LN}\!\left(h_{\text{point},j}+W_{c3}u_{\text{ctr},j}^{\text{mul}}+b_{c3}\right)}
$$

Meaning:

- $z_j$ is the routing/query token for global context retrieval.
- $h_{\text{point},j}$ is a target-pixel identity branch that preserves exact-point information.
- $h_{\text{ctr},j}^{0}$ is an early nonlinear center token conditioned on surrounding local evidence.

Intuition (why this step exists):

- A dedicated point branch prevents the exact target signal from being diluted by surrounding/global features.
- The routing token now sees center + near context, so global retrieval is coordinated with local structure.
- The point branch is now explicitly nonlinear (residual MLP), so it can encode higher-order local cues instead of only a linear projection.
- <span style="color:#BDBDBD">Optional multiplicative interaction further strengthens non-linear coupling between exact center identity and surrounding structure.</span>
- Building $h_{\text{ctr},j}^{0}$ before routing prevents late-stage separation between center and surroundings.

Used next:

- B4 routing, B5 offsets/weights, B7 center aggregation, B8 fusion.

Trainable:

- Yes ($W_z$, $W_{p1}$, $W_{p2}$, $W_{\text{skip}}$, $W_{c1}$, $W_{c3}$, and norm layers).
- <span style="color:#BDBDBD">Optional extra center-nonlinearity weight: $W_{c2}$.</span>

#### B4) Coarse routing with connectivity-safe token subset

Build center-aware routing query:
$$
q_j^{\text{route}}=\mathrm{LN}\!\left(W_r[z_j;h_{\text{ctr},j}^{0}]+b_r\right)
$$
<span style="color:#BDBDBD">Optional reliability-conditioned routing query:</span>
$$
\color{#BDBDBD}{q_j^{\text{route}}=\mathrm{LN}\!\left(W_r[z_j;h_{\text{ctr},j}^{0};m_j]+b_r\right)}
$$
Compute content routing scores over coarse bank:
$$
\alpha_j=\mathrm{softmax}\!\left(\frac{(W_q q_j^{\text{route}})(W_k C_t)^\top}{\sqrt d\,\tau_r}\right)
$$
Training-time differentiable sparse selection:
$$
R_j^{\text{train}}=\operatorname{TopR}_{\text{ST}}(\alpha_j)
$$
Inference-time hard selection:
$$
R_j^{\text{infer}}=\operatorname{TopR}(\alpha_j)
$$
Use
$$
R_j=
\begin{cases}
R_j^{\text{train}}, & \text{training}\\
R_j^{\text{infer}}, & \text{inference}
\end{cases}
$$
Build learned landmark coverage bank:
$$
L_t=\mathrm{LandmarkPool}(C_t),\qquad
U_j=\operatorname{TopU}\!\left(\mathrm{softmax}\!\left(\frac{(W_q^{u}q_j^{\text{route}})(W_k^{u}L_t)^\top}{\sqrt d}\right)\right)
$$
Map query to nearest coarse token center:
$$
q_j^{(2)}=\left(\frac{u_j}{s_2},\frac{v_j}{s_2}\right),\qquad
i_j^{\text{loc}}=\arg\min_{i\in\{1,\ldots,P_c\}}\left\|\mathbf{c}_i-q_j^{(2)}\right\|_2^2
$$
Build spatial subset:
$$
S_j=R_j\cup U_j\cup\{i_j^{\text{loc}}\}
$$
Build query-conditioned global summary using routed/coverage tokens plus anchors:
$$
\mathcal{V}_j=\{c_i\mid i\in S_j\}\cup\{g_a\}_{a=1}^{M_a},\qquad
\bar c_j=\mathrm{softmax}\!\left(\frac{(W_q^{v}q_j^{\text{route}})(W_k^{v}\mathcal{V}_j)^\top}{\sqrt d}\right)\mathcal{V}_j
$$
Each coarse token $c_i$ stores metadata $(\mathbf{c}_i,\ell_i)$ where $\mathbf{c}_i$ is coarse center coordinate and $\ell_i$ is level tag.

Meaning:

- $\alpha_j$: content relevance over compact global regions.
- $R_j$: top content-routed regions.
- $U_j$: coverage regions selected from learned landmarks (content-aware fallback).
- $i_j^{\text{loc}}$: guaranteed query-neighbor coarse anchor.
- $S_j$: sparse spatial token subset for deformable reads with mandatory local connectivity.
- $\bar c_j$: query-conditioned global summary token.
- $W_q q_j^{\text{route}}$ (Query): what context this query point is asking for, conditioned on center quality.
- <span style="color:#BDBDBD">If enabled, $m_j$ modulates routing aggressiveness.</span>
- $W_k c_i$ (Key): how each coarse token advertises what it contains.
- $v\in\mathcal{V}_j$ (Value): the actual global content that is mixed into $\bar c_j$.

Intuition (why this step exists):

- Hard top-k alone is brittle; straight-through sparse routing stabilizes train-infer behavior.
- Content routing alone can miss long-range cues if logits are over-confident.
- Landmark coverage gives non-local fallback without relying on fixed coordinate hashing.
- Mandatory inclusion of $i_j^{\text{loc}}$ prevents disconnection between center and routed global context.
- Anchor tokens are imported from latent-bottleneck methods: every query can still access scene-level context even if spatial routing is imperfect.
- This combination coordinates center, near-surround, and global branches before deformable sampling starts.
- In implementation, $i_j^{\text{loc}}$ should come from a precomputed coarse-cell lookup map (built once per window) to avoid per-query exhaustive search cost.

Used next:

- B5-B10.

Trainable:

- Yes ($W_r,W_q,W_k,W_q^u,W_k^u,W_q^v,W_k^v$ and landmark pool). Set union and nearest-index selection are non-trainable.

#### B5) Offset and weight prediction with local-global conditioning

This step converts routed coarse tokens into concrete sampling coordinates and sampling importance.

Inputs per query $j$:

- $z_j\in\mathbb{R}^{d}$: query token.
- $f_{j,0}\in\mathbb{R}^{d_f}$ and $l_j^{\text{ctx}}\in\mathbb{R}^{d_l}$: exact-point and neighborhood descriptors.
- <span style="color:#BDBDBD">$l_j^{\text{ring}}$ and $m_j$: structured local context and local reliability from B1.1 (optional).</span>
- $h_{\text{point},j},h_{\text{ctr},j}^{0}\in\mathbb{R}^{d}$: point-locked representation and early coordinated center token.
- $\bar c_j\in\mathbb{R}^{d}$: query-conditioned global summary.
- For each routed token $r\in S_j$: center metadata $(\mathbf{c}_r,\ell_r)$ and token type (`routed`/`coverage`/`local-anchor`).

Step 1: build token-conditioned context vector:
$$
\mathbf{p}_{q_j}^{(\ell_r)}=\left(\frac{u_j}{s_{\ell_r}},\frac{v_j}{s_{\ell_r}}\right),\qquad
\mathbf{c}_{r}^{(\ell_r)}=\operatorname{ScaleToLevel}\!\left(\mathbf{c}_r,\ell_r\right),\qquad
\Delta \mathbf{c}_{j,r}^{\text{meta}}=\mathbf{c}_r^{(\ell_r)}-\mathbf{p}_{q_j}^{(\ell_r)}
$$
where $\tau_r$ is the token-type embedding (`routed`, `coverage`, or `local-anchor`).
$$
\hat u_{j,r}=[z_j;\ h_{\text{point},j};\ h_{\text{ctr},j}^{0};\ f_{j,0};\ l_j^{\text{ctx}};\ \bar c_j;\ \psi(\Delta \mathbf{c}_{j,r}^{\text{meta}},\ell_r,\tau_r)]
$$
<span style="color:#BDBDBD">Optional richer conditioning vector:</span>
$$
\color{#BDBDBD}{\hat u_{j,r}=[z_j;\ h_{\text{point},j};\ h_{\text{ctr},j}^{0};\ f_{j,0};\ l_j^{\text{ctx}};\ l_j^{\text{ring}};\ m_j;\ \bar c_j;\ \psi(\Delta \mathbf{c}_{j,r}^{\text{meta}},\ell_r,\tau_r)]}
$$
$$
u_{j,r}=\mathrm{LN}(W_u\hat u_{j,r}+b_u),\qquad u_{j,r}\in\mathbb{R}^{d_u}
$$

Step 2: predict raw offsets for each head/level/sample slot:
$$
o_{j,r,h,\ell,m}=W^{\Delta}_{h,\ell,m}u_{j,r}+b^{\Delta}_{h,\ell,m},\qquad o_{j,r,h,\ell,m}\in\mathbb{R}^{2}
$$

Step 3: bound offsets to valid local radius:
$$
\Delta p_{j,r,h,\ell,m}=\rho_{\ell}\tanh(o_{j,r,h,\ell,m})
$$
$$
\rho_{\ell}=\kappa s_{\ell}\ \text{ (recommended scale-consistent choice)}
$$

Step 4: predict unnormalized importance logits:
$$
\beta_{j,r,h,\ell,m}=W^{a}_{h,\ell,m}u_{j,r}+b^{a}_{h,\ell,m}
$$
Inject center-connectivity bias at logit level:
$$
\tilde\beta_{j,r,h,\ell,m}=\beta_{j,r,h,\ell,m}+\mathbf{1}[r=i_j^{\text{loc}}]\cdot b_{\text{loc}}
$$
<span style="color:#BDBDBD">Optional reliability modulation:</span>
$$
\color{#BDBDBD}{\tilde\beta_{j,r,h,\ell,m}\leftarrow \tilde\beta_{j,r,h,\ell,m}+\kappa_m\,m_j}
$$
where $b_{\text{loc}}$ (<span style="color:#BDBDBD">and optional $\kappa_m$</span>) are learnable scalar parameters.

Step 5: normalize importance over all sampling slots of query $j$:
$$
\mathcal{D}_j=\{(r,h,\ell,m)\mid r\in S_j,\ h\in[1,H],\ \ell\in[1,L],\ m\in[1,M]\}
$$
$$
a_{j,r,h,\ell,m}=\frac{\exp(\tilde\beta_{j,r,h,\ell,m})}{\sum_{(r',h',\ell',m')\in\mathcal{D}_j}\exp(\tilde\beta_{j,r',h',\ell',m'})}
$$
$$
\sum_{(r,h,\ell,m)\in\mathcal{D}_j}a_{j,r,h,\ell,m}=1,\qquad a_{j,r,h,\ell,m}\ge 0
$$

Outputs of B5:

- $\Delta p_{j,r,h,\ell,m}$: where to read in B6.
- $a_{j,r,h,\ell,m}$: how much each read contributes in B7.

Meaning:

- Learn both sampling geometry and sampling confidence, jointly conditioned on local and global evidence.
- Center-connectivity bias keeps one explicit path from the target neighborhood into the sampled evidence.

Intuition (why this step exists):

- Coarse routing (B4) only says which regions may matter; B5 refines that into exact points.
- Offset head answers: "where exactly should I look?"
- Weight head answers: "how much should I trust each looked-up point?"
- Because $u_{j,r}$ contains center, near-surrounding, and global branches together, sampling policy is coordinated instead of separated.
- The additive local-anchor bias is a soft prior, not a hard override, so learning remains data-driven.

Used next:

- B6 and B7.

Trainable:

- Yes.

#### B6) Deformable multiscale sampling over selected spatial subset

What is being computed:

- For each query `j`, routed token `r in S_j`, head `h`, pyramid level `\ell`, and sample index `m`, compute one sampling coordinate and read one feature vector.

Coordinate definitions (all in level-`\ell` feature coordinates):
$$p_{q_j}^{(\ell)}=\left(\frac{u_j}{s_{\ell}},\frac{v_j}{s_{\ell}}\right)$$
$$c_{r}^{(\ell)}=\operatorname{ScaleToLevel}\!\left(\mathbf{c}_r,\ell\right),\qquad p_{r}^{(\ell)}=c_{r}^{(\ell)}-p_{q_j}^{(\ell)}$$
$$\Delta p_{j,r,h,\ell,m}=\text{learned residual offset from B5}$$

Step-by-step sampling:
$$
p_{\text{base},r,\ell}=p_{q_j}^{(\ell)}+p_{r}^{(\ell)}
$$
equivalently $p_{\text{base},r,\ell}=c_r^{(\ell)}$.
$$
p_{\text{raw},j,r,h,\ell,m}=p_{\text{base},r,\ell}+\Delta p_{j,r,h,\ell,m}
$$
$$
p_{\text{norm},j,r,h,\ell,m}=\mathrm{NormalizeToGrid}(p_{\text{raw},j,r,h,\ell,m};H_\ell,W_\ell)
$$
$$
f_{j,r,h,\ell,m}=\operatorname{GridSample}_{\text{reflect}}\!\left(F_t^{(\ell)},\ p_{\text{norm},j,r,h,\ell,m}\right)
$$
$$
f_{\text{ctr},j}^{(\ell)}=\operatorname{Bilinear}\!\left(F_t^{(\ell)},\ p_{q_j}^{(\ell)}\right)
$$
Partition sampled slots into near and far groups:
$$
\mathcal{D}_j^{\text{near}}=\left\{(r,h,\ell,m)\in\mathcal{D}_j\ \middle|\ r=i_j^{\text{loc}}\ \text{or}\ \left\|p_r^{(\ell)}\right\|_{\infty}\le r_{\text{near}}\right\}
$$
with recommended $r_{\text{near}}=2$ (in level-$\ell$ feature coordinates).
$$
h_{\text{near},j}^{\text{raw}}=\sum_{(r,h,\ell,m)\in\mathcal{D}_j^{\text{near}}}a_{j,r,h,\ell,m}f_{j,r,h,\ell,m},\qquad
h_{\text{def},j}=\sum_{(r,h,\ell,m)\in\mathcal{D}_j}a_{j,r,h,\ell,m}f_{j,r,h,\ell,m}
$$

Equivalent compact form:
$$
(x_{j,r,h,\ell,m},y_{j,r,h,\ell,m})=p_{q_j}^{(\ell)}+p_r^{(\ell)}+\Delta p_{j,r,h,\ell,m}
$$
$$
\tilde p\leftarrow\mathrm{NormalizeToGrid}(x_{j,r,h,\ell,m},y_{j,r,h,\ell,m}),\qquad
f_{j,r,h,\ell,m}=\operatorname{GridSample}_{\text{reflect}}(F_t^{(\ell)},\tilde p)
$$

Meaning:

- `p_{q_j}^{(\ell)}`: where the query is at this scale.
- `c_r^{(\ell)}`: routed coarse-token center projected to level `\ell`.
- `p_r^{(\ell)}`: query-relative coarse offset used to jump toward routed region.
- `\Delta p_{j,r,h,\ell,m}`: fine correction learned by network.
- `f_{j,r,h,\ell,m}`: actual evidence fetched from level-`\ell` feature map.
- `f_{\text{ctr},j}^{(\ell)}`: exact query-centered feature at each scale (point-lock bypass).
- $h_{\text{near},j}^{\text{raw}}$: sampled near-surround evidence.
- $h_{\text{def},j}$: global deformable aggregation over all sampled points.

Intuition (why this step exists):

- First jump to a routed coarse center (`c_r^{(\ell)}` via `p_r^{(\ell)}`), then make a precise local correction (`\Delta p`), then read feature evidence there.
- Reflective grid sampling avoids border-collapse artifacts that happen with hard clamping.
- `Bilinear` allows sub-pixel/fractional sampling, so offsets are continuous and trainable.
- Multi-scale reads provide both local detail (`F_t^{(1)}`) and wider context (`F_t^{(2)},F_t^{(3)}`) without dense decoding.
- The extra center read ensures the exact query location is always represented even if deformable offsets miss.
- Explicit near extraction keeps local geometry traceable while $h_{\text{def}}$ preserves full global evidence.

Used next:

- B7.

Trainable:

- Sampling op: No. Locations are trainable through B5.

#### B7) Global aggregation plus point-locked branch assembly

Build coordinated near/global branches:
$$
h_{\text{near},j}=\mathrm{LN}\!\left(W_{\text{near}}\!\left[h_{\text{near},j}^{\text{raw}};l_j^{\text{ctx}}\right]+b_{\text{near}}\right)
$$
<span style="color:#BDBDBD">Optional richer near-branch input:</span>
$$
\color{#BDBDBD}{h_{\text{near},j}=\mathrm{LN}\!\left(W_{\text{near}}\!\left[h_{\text{near},j}^{\text{raw}};l_j^{\text{ctx}};l_j^{\text{ring}}\right]+b_{\text{near}}\right)}
$$
$$
h_{\text{global},j}=\mathrm{LN}\!\left(W_{\text{glob}}\!\left[h_{\text{def},j};\bar c_j\right]+b_{\text{glob}}\right)
$$
Build center branch with explicit near-surround infusion:
$$
h_{\text{ctr},j}=\mathrm{LN}\!\left(W_{\text{ctr}}\!\left[h_{\text{point},j};h_{\text{ctr},j}^{0};f_{\text{ctr},j}^{(1)};f_{\text{ctr},j}^{(2)};f_{\text{ctr},j}^{(3)};h_{\text{near},j}\right]+b_{\text{ctr}}\right)
$$

Meaning:

- $h_{\text{near},j}$ summarizes center-surrounding local evidence.
- $h_{\text{global},j}$ is global disambiguation feature built from deformable global aggregation + summary token.
- $h_{\text{ctr},j}$ preserves exact target-pixel identity while already absorbing near context.

Intuition (why this step exists):

- Global context is needed to disambiguate depth.
- Exact target-point evidence must remain explicit and cannot be replaced by neighborhood averages.
- Near evidence and global evidence are separated into two core branches with clear roles.
- Center branch ingests near branch before B8, so center-surround coupling is not delayed until final fusion.
- Extra anchor attention is still unnecessary by default because anchors already contribute through $\bar c_j$.

Used next:

- B8 fusion.

Trainable:

- Yes through upstream trainable heads and projections.

#### B8) Balanced center-context coupling (default and recommended)

Build compact context token set:
$$
T_j^{\text{near}}=\mathrm{Reshape}\!\left(W_{\text{near}}^{\text{tok}}h_{\text{near},j}\right)\in\mathbb{R}^{N_{\text{near}}^{\text{tok}}\times d},\qquad
T_j^{\text{glb}}=\mathrm{Reshape}\!\left(W_{\text{glb}}^{\text{tok}}h_{\text{global},j}\right)\in\mathbb{R}^{N_{\text{glb}}^{\text{tok}}\times d}
$$
$$
T_j^{\text{sum}}=\mathrm{Reshape}\!\left(W_{\text{sum}}\bar c_j\right)\in\mathbb{R}^{1\times d},\qquad
T_j^{\text{mem}}=W_{\text{mem}}H_t\in\mathbb{R}^{N_H\times d}
$$
$$
T_j=[T_j^{\text{near}};T_j^{\text{glb}};T_j^{\text{sum}};T_j^{\text{mem}}],\qquad
T_j=\{t_{j,n}\}_{n=1}^{N_T},\qquad N_T=N_{\text{near}}^{\text{tok}}+N_{\text{glb}}^{\text{tok}}+1+N_H
$$
Query-conditioned subset selection:
$$
s_{j,n}=\frac{(W_s z_j)^\top(W_t t_{j,n})}{\sqrt d},\qquad
I_j=\operatorname{TopK}(s_j,k_t),\qquad
\tilde T_j=\{t_{j,n}\mid n\in I_j\}
$$
Center-to-context update (cross-attention from center token):
$$
h_{\text{att},j}=\mathrm{MHA}\!\left(Q=W_q^{c}h_{\text{ctr},j},K=W_k^{c}\tilde T_j,V=W_v^{c}\tilde T_j\right),\qquad
h_{\text{ctr},j}^{+}=\mathrm{LN}\!\left(h_{\text{ctr},j}+h_{\text{att},j}\right)
$$
Context aggregation and nonlinear context mixer:
$$
\omega_{j,n}=\mathrm{softmax}_{n\in I_j}\!\left(\frac{(W_q^{a}h_{\text{ctr},j}^{+})^\top(W_k^{a}t_{j,n})}{\sqrt d}\right),\qquad
h_{\text{ctx},j}^{0}=\sum_{n\in I_j}\omega_{j,n}t_{j,n}
$$
$$
h_{\text{ctx},j}=\mathrm{LN}\!\left(\mathrm{MLP}_{\text{ctx}}\!\left([h_{\text{att},j};h_{\text{ctx},j}^{0};h_{\text{near},j};h_{\text{global},j}]\right)\right)
$$
<span style="color:#BDBDBD">Optional reciprocal write-back (higher compute):</span>
$$
\color{#BDBDBD}{\eta_{j,n}=\sigma\!\left(w_{\eta}^{\top}[t_{j,n};h_{\text{ctr},j}^{+};z_j]+b_{\eta}\right),\quad
t_{j,n}^{+}=\mathrm{LN}\!\left(t_{j,n}+\eta_{j,n}W_{\text{up}}h_{\text{ctr},j}^{+}\right),\quad
\omega_{j,n}^{+}=\mathrm{softmax}_{n\in I_j}\!\left(\frac{(W_q^{a}h_{\text{ctr},j}^{+})^\top(W_k^{a}t_{j,n}^{+})}{\sqrt d}\right),\quad
h_{\text{ctx},j}^{0}=\sum_{n\in I_j}\omega_{j,n}^{+}t_{j,n}^{+}}
$$
Point-aware context gate and bounded carry fusion:
$$
g_j=\sigma\!\Big(\mathrm{MLP}_g([h_{\text{ctr},j}^{+};h_{\text{ctx},j};z_j])\Big),\qquad 0\le g_j\le 1
$$
$$
\lambda_{\text{ctr},j}=\lambda_{\min}+(\lambda_{\max}-\lambda_{\min})\,\sigma\!\left(w_{\lambda}^{\top}[h_{\text{ctr},j}^{+};h_{\text{ctx},j};z_j]+b_{\lambda}\right),\qquad
\lambda_{\text{ctx},j}=1-\lambda_{\text{ctr},j}
$$
$$
h_{\text{fuse},j}=\lambda_{\text{ctr},j}h_{\text{ctr},j}^{+}+\lambda_{\text{ctx},j}\!\left(g_j\odot h_{\text{ctx},j}\right),\qquad
\lambda_{\text{ctr},j}\in[\lambda_{\min},\lambda_{\max}]
$$
Recommended default:
$$
\lambda_{\min}=0.45,\qquad \lambda_{\max}=0.75,\qquad
N_{\text{near}}^{\text{tok}}=8,\qquad N_{\text{glb}}^{\text{tok}}=4,\qquad
k_t=8
$$

Meaning:

- $h_{\text{ctr},j}^{+}$ is the center token after reading selected context.
- $h_{\text{ctx},j}$ is a nonlinear context representation combining near and global signals.
- $(\lambda_{\text{ctr},j},\lambda_{\text{ctx},j})$ explicitly preserve center while allowing controlled context correction.

Intuition (why this step exists):

- Removes variable-heavy multi-branch fusion while keeping full center-surround-global information in one compact context path.
- Default path uses center-to-context coupling; optional reciprocal write-back enables explicit two-way coupling.
- Nonlinear context mixer prevents the model from being effectively linear around the queried center.
- Default path removes reciprocal write-back to reduce B8 cost while keeping center-to-context coupling.
- Bounded carry prevents center collapse and keeps fusion behavior stable.

Used next:

- B9 depth head, B10 uncertainty.

Trainable:

- Yes.

#### B9) Depth code, calibration, and final depth

Compute relative code:
$$
r_j=h_r(h_{\text{fuse},j})
$$
<span style="color:#BDBDBD">Optional point-only auxiliary code (train-time supervision):</span>
$$
\color{#BDBDBD}{r_j^{\text{ctr}}=h_r^{\text{ctr}}(h_{\text{ctr},j}^{+})}
$$

If $h_r$ is linear, this is equivalently:
$$
r_j=W_r h_{\text{fuse},j}+b_r
$$

Calibrate:
$$
\rho_j=s_t r_j+b_t
$$
<span style="color:#BDBDBD">Optional residual calibration (ablation mode only):</span>
$$
\color{#BDBDBD}{\Delta s_j=\kappa_s\tanh\!\left(w_s^\top[h_{\text{fuse},j};\bar c_j]+b_s\right),\qquad
\Delta b_j=\kappa_b\tanh\!\left(w_b^\top[h_{\text{fuse},j};\bar c_j]+b_b\right)}
$$
$$
\color{#BDBDBD}{\rho_j=(s_t+\Delta s_j)r_j+(b_t+\Delta b_j)}
$$

Convert to depth:
$$
\hat d_j=\frac{1}{\mathrm{softplus}(\rho_j)+\varepsilon}
$$

Meaning:

- $r_j$: uncalibrated depth code.
- $r_j^{\text{ctr}}$: point-only auxiliary code (used only for training regularization).
- $\rho_j$: calibrated inverse-depth/disparity (global default; <span style="color:#BDBDBD">optional residual refinement</span>).
- $\hat d_j$: final depth at query.

Intuition (why this step exists):

- The decoder first predicts a relative code $r_j$ because relative geometry is easier to learn.
- The auxiliary $r_j^{\text{ctr}}$ forces the center branch to remain informative and prevents target-point information collapse.
- $s_t,b_t$ inject window-level global calibration so all queries in the same window are on a consistent scale.
- <span style="color:#BDBDBD">Optional residual $(\Delta s_j,\Delta b_j)$ is treated as hypothesis mode and must pass ablations before default use.</span>
- Softplus keeps denominator positive and avoids invalid depth values.
- Without calibration, outputs can be locally correct but globally mis-scaled.

Used next:

- Output and losses.

Trainable:

- Yes ($h_r$ and upstream).

#### B10) Uncertainty and <span style="color:#BDBDBD">optional refinement</span>

Compute uncertainty:
$$
\sigma_j=h_{\sigma}(h_{\text{fuse},j})
$$

Recommended positive form:
$$
\sigma_j=\mathrm{softplus}(W_{\sigma}h_{\text{fuse},j}+b_{\sigma})+\sigma_{\min}
$$

<span style="color:#BDBDBD">Optional hard-query second-hop rerouting:</span>
$$
\color{#BDBDBD}{p_j^{\text{hop}}=\sigma\!\left(\frac{\sigma_j-\tau_{\text{hop}}}{T_{\text{hop}}}\right)\quad\text{(training gate)}}
$$
$$
\color{#BDBDBD}{\text{if }\sigma_j>\tau_{\text{hop}},\quad
z_j^{(2)}=W_{\text{hop}}[z_j;h_{\text{fuse},j};\bar c_j],\quad
\text{run B4-B8 with }(R_2,U_2)\ \text{to get }h_{\text{fuse},j}^{(2)},\quad
\hat d_j\leftarrow \hat d_j+\Delta d_j^{\text{hop}}\quad\text{(inference gate)}}
$$

<span style="color:#BDBDBD">Refinement head:</span>
$$
\color{#BDBDBD}{\Delta d_j^{\text{hop}}=
\mathrm{MLP}_{\text{refine}}\!\big([h_{\text{fuse},j}^{(2)};h_{\text{fuse},j};l_j^{\text{ctx}};e_j]\big)}
$$
<span style="color:#BDBDBD">Recommended small second-hop budget:</span>
$$
\color{#BDBDBD}{R_2=2,\qquad U_2=1}
$$

Meaning:

- Spend extra global-local interaction compute only on difficult queries.
- <span style="color:#BDBDBD">This branch is optional and disabled in the GCQD-lite deployment profile.</span>

Intuition (why this step exists):

- Not all queries are equally hard; uncertainty estimates which points are unreliable.
- Using a soft gate in training and hard gate in inference reduces train-infer mismatch.
- Second-hop rerouting lets hard queries gather additional far-range evidence without paying that cost for all queries.
- This preserves speed on easy points while improving hard points (boundaries, low-event zones, motion ambiguity).
- Without uncertainty, you either waste compute everywhere or miss hard-case corrections.

Used next:

- Output and uncertainty-aware objectives.

Trainable:

- <span style="color:#BDBDBD">Yes ($h_{\sigma}$ and optional refine head).</span>

#### B11) Return sparse outputs

Output:
$$
\{(\hat d_j,\sigma_j)\}_{j=1}^{K}
$$
If the uncertainty head is disabled in a minimal deployment build, return $\{(\hat d_j,\varnothing)\}_{j=1}^{K}$.

#### B12) GCQD-lite collapsed runtime (recommended efficient implementation)

The detailed B1-B11 decomposition is for clarity and analysis. For efficient implementation, run a merged 5-stage path:

1. M1 `Precompute` (A1-A6), once per event window:
   $$
   E_t\rightarrow F_t^{(1:3)},\ C_t,\ G_t,\ M_t,\ R_t,\ H_t,\ s_t,\ b_t
   $$
2. M2 `QueryEncode` (merge B1+B2+B3):
   $$
   q_j\rightarrow (f_{j,0},l_j^{\text{ctx}})\ (\text{hybrid local }N_{\text{loc}}\approx121)\rightarrow e_j\rightarrow (z_j,h_{\text{point},j},h_{\text{ctr},j}^{0})
   $$
   <span style="color:#BDBDBD">Optional richer local encoding: additionally compute $(l_j^{\text{ring}},m_j)$.</span>
3. M3 `SparsePlan` (merge B4+B5):
   $$
   (z_j,h_{\text{ctr},j}^{0},f_{j,0},l_j^{\text{ctx}},h_{\text{point},j},C_t,G_t)\rightarrow (S_j,\bar c_j,\Delta p_j,a_j)
   $$
   <span style="color:#BDBDBD">Optional richer planning: append $(l_j^{\text{ring}},m_j)$ to routing/sampling conditioning.</span>
4. M4 `SparseReadAggregate` (merge B6+B7):
   $$
   (F_t^{(1:3)},S_j,\Delta p_j,a_j,\bar c_j,h_{\text{point},j},h_{\text{ctr},j}^{0},l_j^{\text{ctx}})\rightarrow (h_{\text{near},j},h_{\text{global},j},h_{\text{ctr},j})
   $$
5. M5 `FuseDecode` (merge B8+B9, <span style="color:#BDBDBD">optional B10 uncertainty</span>):
   $$
   (h_{\text{near},j},h_{\text{global},j},h_{\text{ctr},j},z_j,\bar c_j,H_t,s_t,b_t)\rightarrow (\hat d_j,\sigma_j)
   $$
   <span style="color:#BDBDBD">Optional richer fusion: append $m_j$ into context mixer.</span>

Default deployment profile:

- Keep hybrid local sampler at default budget (`N_loc\approx121`) without hard-query expansion.
- Use a single-pass coupled B8 block with fixed `k_t` and no reciprocal write-back (core low-latency mode).
- Keep anchor information injected via $\bar c_j$ from B4 (no extra standalone anchor-attention branch).
- Keep <span style="color:#BDBDBD">second-hop rerouting</span> disabled by default.

Complexity trade-off summary inside B workflow:

- Core mode (recommended): no ring/reliability conditioning in B3-B5, no reciprocal write-back in B8, fixed `k_t`.
- Optional mode: enable ring/reliability conditioning and/or reciprocal write-back only if ablations show clear accuracy gain.
- This keeps nonlinearity (center encoder + context MLP) and point-near-global balancing (center carry + near/global context), while reducing per-query compute.

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

Compute (default core objective):
$$
\mathcal{L}_{\text{core}}=\lambda_{\text{data}}L_{\text{data}}+\lambda_{\text{geom}}L_{\text{geom}}+\lambda_{\text{stb-core}}L_{\text{stb-core}}
$$
<span style="color:#BDBDBD">Optional extensions (ablation profile):</span>
$$
\color{#BDBDBD}{\mathcal{L}_{\text{ext}}=\lambda_{\text{cov}}L_{\text{cov}}+\lambda_{\text{cmp}}L_{\text{cmp}}+\lambda_{\text{distill}}L_{\text{distill}},\qquad
\mathcal{L}=\mathcal{L}_{\text{core}}+\mathcal{L}_{\text{ext}}}
$$

Detailed formulas are in Section 4.5A.

#### C6) Backprop and optimizer step (explicit optimization step)

- Update trainable modules (default profile): LatentPool, AnchorPool, hash tables, local heads (`MLP_local`, `MLP_ring`, reliability head), query/center projections, routing projections, summary projections, offset/weight heads, near/global aggregators, tokenization projections, reciprocal coupling heads, bounded carry head, depth head, point-aux depth head, uncertainty head.
- <span style="color:#BDBDBD">Update additional modules only in extended ablation profile: second-hop projector and refine head.</span>
- Backbone $\mathcal{F}_{\mathrm{F^3}}$ is updated only when fine-tune mode is enabled.
- Use context-drop regularization during training: with probability $p_{\text{ctxdrop}}$, set $h_{\text{ctx},j}\leftarrow 0$ before B8 so center and near-surround branches remain predictive.

#### C7) Query-budget curriculum

- Train with varying $K$ to make runtime robust across query loads.

### 4.5A Training objectives for sparse query depth (moved from former Section 6)

Let $\hat d(q)$ be predicted depth at query $q$, $d^\star(q)$ target depth, and $M(q)\in\{0,1\}$ a validity mask.

Define inverse depth $\rho(q)=1/d(q)$ for stable optimization.

Is it common to have many losses in works like this?

- Yes, but successful papers usually keep a *small number of top-level blocks* and only add extra terms for specific failure modes.
- So the goal is not "one loss only"; the goal is "few clear blocks + staged activation".

Recommended grouped objective (clearer and equivalent):

1. Data-fit block:
   $$
   L_{\text{point}}=\frac{1}{|Q_v|}\sum_{q\in Q_v}w(q)\,\mathrm{Huber}\!\big(\hat \rho(q)-\rho^\star(q)\big)
   $$
   $$
   L_{\text{silog},q}=\frac{1}{|Q_v|}\sum_{q\in Q_v}\delta_q^2-\lambda_{\text{si}}\left(\frac{1}{|Q_v|}\sum_{q\in Q_v}\delta_q\right)^2,\qquad
   \delta_q=\log \hat d(q)-\log d^\star(q)
   $$
   $$
   L_{\text{unc}}=\frac{1}{|Q_v|}\sum_{q\in Q_v}\left(\frac{(\hat d(q)-d^\star(q))^2}{2\sigma(q)^2}+\frac{1}{2}\log \sigma(q)^2\right)
   $$
   $$
   L_{\text{data}}=L_{\text{point}}+\lambda_{\text{si}}^{(d)}L_{\text{silog},q}+\lambda_{\text{unc}}^{(d)}L_{\text{unc}}
   $$

2. Geometry-consistency block:
   $$
   L_{\text{rank}}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\log\!\Big(1+\exp\!\big(-y_{ij}(\hat\rho_i-\hat \rho_j)\big)\Big)
   $$
   with $y_{ij}=\mathrm{sign}(\rho_i^\star-\rho_j^\star)$.
   $$
   L_{\text{q-cons}}=\frac{1}{|E_Q|}\sum_{(i,j)\in E_Q}w_{ij}\,\mathrm{Huber}\!\Big((\hat\rho_i-\hat\rho_j)-(\rho_i^\star-\rho_j^\star)\Big)
   $$
   $$
   L_{\text{geom}}=L_{\text{rank}}+\lambda_{\text{q}}^{(g)}L_{\text{q-cons}}
   $$

3. Stability-and-coupling block:
   $$
   L_{\text{route-stab}}=\frac{1}{|Q|}\sum_{q\in Q}\mathrm{KL}\!\left(\alpha_q\ \middle\|\ \alpha'_q\right)
   $$
   $$
   L_{\text{cov}}=\mathrm{KL}\!\left(\frac{1}{|Q|}\sum_{q\in Q}\alpha_q\ \middle\|\ \pi_t\right),\qquad
   \pi_t=\mathrm{softmax}(W_\pi C_t)
   $$
   $$
   L_{\text{align}}=\frac{1}{|Q|}\sum_{q\in Q}\left(\left\|W_c h_{\text{ctr},q}^{+}-W_n h_{\text{near},q}\right\|_1+\left\|W_n h_{\text{near},q}-W_g h_{\text{ctx},q}\right\|_1\right)
   $$
   $$
   r_{\text{ctx},q}=\frac{\left\|\lambda_{\text{ctx},q}\!\left(g_q\odot h_{\text{ctx},q}\right)\right\|_2}{\left\|\lambda_{\text{ctr},q}h_{\text{ctr},q}^{+}\right\|_2+\left\|\lambda_{\text{ctx},q}\!\left(g_q\odot h_{\text{ctx},q}\right)\right\|_2+\varepsilon}
   $$
   $$
   r_{\text{near},q}=\frac{\left\|h_{\text{near},q}\right\|_2}{\left\|h_{\text{near},q}\right\|_2+\left\|h_{\text{ctx},q}\right\|_2+\varepsilon}
   $$
   $$
   L_{\text{bal}}=\frac{1}{|Q|}\sum_{q\in Q}\Big[\max(0,r_{\text{ctx,min}}-r_{\text{ctx},q})+\max(0,r_{\text{ctx},q}-r_{\text{ctx,max}})+\max(0,r_{\text{near,min}}-r_{\text{near},q})+\max(0,r_{\text{near},q}-r_{\text{near,max}})\Big]
   $$
   $$
   L_{\text{ctr}}=\frac{1}{|Q_v|}\sum_{q\in Q_v}\mathrm{Huber}\!\big(r_q^{\text{ctr}}-\rho^\star(q)\big)
   $$
   $$
   L_{\text{stb}}=L_{\text{route-stab}}+\lambda_{\text{cov}}^{(s)}L_{\text{cov}}+\lambda_{\text{align}}^{(s)}L_{\text{align}}+\lambda_{\text{bal}}^{(s)}L_{\text{bal}}+\lambda_{\text{ctr}}^{(s)}L_{\text{ctr}}
   $$
   Recommended balance bands:
   $$
   r_{\text{ctx,min}}=0.25,\quad r_{\text{ctx,max}}=0.55,\quad r_{\text{near,min}}=0.20,\quad r_{\text{near,max}}=0.50
   $$

4. Compute-control block:
   $$
   p_q^{\text{hop}}=\sigma\!\left(\frac{\sigma(q)-\tau}{T_{\text{hop}}}\right),\qquad
   L_{\text{budget}}=\frac{1}{|Q|}\sum_{q\in Q}p_q^{\text{hop}}
   $$
   $$
   L_{\text{cmp}}=L_{\text{budget}}
   $$

5. <span style="color:#BDBDBD">Optional teacher prior:</span>
   $$
   \color{#BDBDBD}{L_{\text{distill}}=\frac{1}{|Q|}\sum_{q\in Q}\lambda_{\text{distill}}(q)\big\|\hat d(q)-d_{\text{teacher}}(q)\big\|}
   $$

<span style="color:#BDBDBD">Full objective (research/ablation profile):</span>
$$
\color{#BDBDBD}{\mathcal{L}=\lambda_{\text{data}}L_{\text{data}}+\lambda_{\text{geom}}L_{\text{geom}}+\lambda_{\text{stb}}L_{\text{stb}}+\lambda_{\text{cmp}}L_{\text{cmp}}+\lambda_{\text{distill}}L_{\text{distill}}}
$$

For the main paper model, use a cleaner default:
$$
\mathcal{L}_{\text{core}}=\lambda_{\text{data}}L_{\text{data}}+\lambda_{\text{geom}}L_{\text{geom}}+\lambda_{\text{stb-core}}L_{\text{stb-core}},
\qquad
L_{\text{stb-core}}=L_{\text{route-stab}}+\lambda_{\text{align}}^{(c)}L_{\text{align}}+\lambda_{\text{bal}}^{(c)}L_{\text{bal}}+\lambda_{\text{ctr}}^{(c)}L_{\text{ctr}}
$$
<span style="color:#BDBDBD">and keep the following only for ablations/extended experiments:</span>
$$
\color{#BDBDBD}{\mathcal{L}_{\text{ext}}=\lambda_{\text{cov}}L_{\text{cov}}+\lambda_{\text{cmp}}L_{\text{cmp}}+\lambda_{\text{distill}}L_{\text{distill}}}
$$
$$
\color{#BDBDBD}{\mathcal{L}=\mathcal{L}_{\text{core}}+\mathcal{L}_{\text{ext}}}
$$

Recommended training schedule (this is what keeps optimization stable and readable):

1. Stage-1 (warmup): optimize $L_{\text{data}}$ only.
2. Stage-2 (structure): enable $L_{\text{geom}}$ and $L_{\text{stb-core}}$ with small weights.
3. <span style="color:#BDBDBD">Stage-3 (extensions): optionally enable $L_{\text{cov}},L_{\text{cmp}},L_{\text{distill}}$.</span>

For metric fine-tuning, keep LiDAR validity filtering analogous to current code.

<span style="color:#BDBDBD">Optional automatic weighting (if manual $\lambda$ tuning is unstable):</span>
$$
\color{#BDBDBD}{\mathcal{L}_{\text{blk}}=\sum_{i\in\{\text{data,geom,stb-core,cmp,distill}\}}\exp(-s_i)L_i+s_i}
$$
<span style="color:#BDBDBD">where $s_i$ are learnable log-variance scalars (homoscedastic uncertainty weighting).</span>

Evidence from papers and codebases (why this simplification is sound):

- F\^3 depth training code uses compact objectives, not a long flat list: `SSIMAELoss = L1(scale-shift normalized disparity) + \alpha \cdot gradient` and metric fine-tune with `SiLog` / `SiLogGrad` (`fast-feature-fields-main/src/f3/tasks/depth/utils/losses.py`).
- E2Depth (3DV 2020) and RAM-Net (RA-L 2021) use the same compact template:
  $$
  L_{\text{tot}}=\sum_k \left(L_{k,\text{si}}+\lambda L_{k,\text{grad}}\right)
  $$
  i.e., a data term + edge/gradient regularizer across time.
- Monodepth2 keeps supervision conceptually small: photometric reprojection + edge-aware smoothness, with robustification via minimum reprojection and auto-masking (not many unrelated penalties).
- DETR/Deformable DETR also use a fixed, interpretable set objective (Hungarian matching + classification + box regression terms), rather than many unstructured penalties.
- DINO uses a single cross-entropy distillation objective, and handles collapse mainly through teacher centering/sharpening and EMA teacher updates.

Conclusion: the current grouped design is valid, but for clarity and reproducibility the paper's *main* training recipe should use $\mathcal{L}_{\text{core}}$ and treat <span style="color:#BDBDBD">$\mathcal{L}_{\text{ext}}$ as optional</span>.


### 4.5B Query sampling strategy (critical)

During training, sample queries from mixed distributions:

- Uniform random (coverage)
- Event-dense regions (high signal)
- High-gradient/edge regions (boundary quality)
- Far-field and low-event regions (hard negatives)

At inference, user supplies queries directly.


### 4.5C Evaluation protocol

#### Accuracy metrics (query-level)

At queried pixels only:

- AbsRel
- RMSE
- RMSE_log
- SiLog
- delta thresholds

Also report by depth range buckets (near/mid/far).

#### Runtime metrics (the key contribution)

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

Center-context diagnostics (required for this project):
$$
\mathrm{BCR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{\left\|\lambda_{\text{ctx},q}\left(g_q\odot h_{\text{ctx},q}\right)\right\|_2}{\left\|\lambda_{\text{ctr},q}h_{\text{ctr},q}^{+}\right\|_2+\left\|\lambda_{\text{ctx},q}\left(g_q\odot h_{\text{ctx},q}\right)\right\|_2+\varepsilon}
$$
$$
\mathrm{NUR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{\left\|h_{\text{near},q}\right\|_2}{\left\|h_{\text{near},q}\right\|_2+\left\|h_{\text{ctx},q}\right\|_2+\varepsilon}
$$
Interpretation:

- recommended target bands: $\mathrm{BCR}\in[0.25,0.55],\ \mathrm{NUR}\in[0.20,0.50]$.
- low $\mathrm{BCR}$: context/global under-used (too center-locked).
- high $\mathrm{BCR}$: center may be diluted.
- low $\mathrm{NUR}$: near-surrounding support under-used.
$$
\mathrm{CDR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{\left\|\lambda_{\text{ctr},q}h_{\text{ctr},q}^{+}\right\|_2}{\left\|\lambda_{\text{ctx},q}\left(g_q\odot h_{\text{ctx},q}\right)\right\|_2+\varepsilon}
$$
Use CDR only as a secondary legacy report for continuity with previous versions.

#### Robustness matrix

Evaluate across:

- day/night
- different platforms (car/spot/flying if available)
- event subsampling rates

Use same spirit as F^3 robustness experiments.

#### Required ablations for a strong paper

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
- Effect of fusion structure (`naive concat-gate` vs `point-locked nonlinear center-context coupling`).
- Effect of bounded carry interval $[\lambda_{\min},\lambda_{\max}]$ and $L_{\text{bal}}$ on BCR/NUR/CDR and depth error.
- Effect of uncertainty threshold $\tau$ and refinement on/off.
- Effect of coarse routing (`disabled` vs `enabled`).


### 4.6 CUDA execution mapping and complexity

Kernel schedule (GCQD-lite default):

1. `K0_F3Precompute`: $E_t\rightarrow F_t^{(1:3)},C_t,G_t,M_t,R_t,H_t,s_t,b_t$.
2. `K1_QueryEncode`: merge B1-B3, $q\rightarrow f_{q,0},l_q^{\text{ctx}},e(q),z_q,h_{\text{point},q},h_{\text{ctr},q}^{0}$ (hybrid `LocalSample`, default $N_{\text{loc}}\approx121$).
3. `K2_SparsePlan`: merge B4-B5, $(z_q,h_{\text{ctr},q}^{0},f_{q,0},l_q^{\text{ctx}},h_{\text{point},q},C_t,G_t)\rightarrow (S_q,\bar c_q,\Delta p,a)$.
4. `K3_SparseReadAgg`: merge B6-B7, deformable multiscale gather + coordinated near/global aggregation to $(h_{\text{near},q},h_{\text{global},q},h_{\text{ctr},q})$.
5. `K4_FuseDecode`: merge B8-B9 and <span style="color:#BDBDBD">optional uncertainty head from B10</span>, output $(\hat d_q,\sigma_q)$.

<span style="color:#BDBDBD">Extended research profile (for ablations only):</span>

- <span style="color:#BDBDBD">add $(l_q^{\text{ring}},m_q)$ computation and adaptive local expansion in `K1`,</span>
- <span style="color:#BDBDBD">add optional extra anchor-attention branch in `K3`,</span>
- <span style="color:#BDBDBD">add context-drop ablation in `K4`,</span>
- <span style="color:#BDBDBD">add second-hop rerouting after `K4`.</span>

Complexity:

- precompute: $\mathcal{O}(HWC+P_cM_a d)$ (backbone dominated, where $M_a$ is anchor count).
- expected per query:
$$
\mathcal{O}\!\Big(\bar N_{\text{loc}}d+(R+U+1+M_a)d+(R+U+1)HLMd+N_T d+d^2+p_{\text{hop}}(R_2+U_2)HLMd\Big)
$$
where
$$
\bar N_{\text{loc}}=121+48\,p_{\text{loc-hard}}
$$
and $p_{\text{loc-hard}}$ is the fraction of queries that trigger local expansion, while $p_{\text{hop}}$ is the fraction that trigger second-hop rerouting.
The $N_T d$ and $d^2$ terms are compact tokenization/fusion/head costs in `K4` (small compared with F\^3 precompute and sparse gather for typical $d\in[96,160]$).

Default starting hyperparameters:

- $P_c=256$, $R=6$, $U=2$, $M_a=8$, $H=4$, $L=3$, $M=4$.
- Local sampling budget: hybrid default $N_{\text{loc}}\approx121$ in GCQD-lite.
- <span style="color:#BDBDBD">Optional expansion budget for ablations: $N_{\text{loc}}=169$ when enabled.</span>
- First-hop sampled points/query: $(R+U+1)HLM=432$ (extra `+1` is mandatory local coarse anchor).
- <span style="color:#BDBDBD">Second-hop (hard queries only, ablation mode): $(R_2+U_2)HLM=144$ with $(R_2,U_2)=(2,1)$.</span>

### 4.7 RTX 4090 speed estimate (using 120 Hz HD F^3 baseline)

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
- token subset uses routed/coverage/anchor counts $(R,U,M_a)=(6,2,8)$ plus one mandatory local coarse anchor ($+1$ in B4/B5/B6 complexity).
- <span style="color:#BDBDBD">second-hop reroute is triggered for about $p_{\text{hop}}=0.15$ of queries and is absorbed into the $\beta$ range.</span>

| Query count $K$ | Estimated runtime (ms) | Estimated throughput (Hz) |
|---|---:|---:|
| 1 | 8.77 - 9.70 | 114.0 - 103.1 |
| 64 | 9.03 - 10.33 | 110.8 - 96.8 |
| 256 | 9.79 - 12.25 | 102.1 - 81.6 |
| 1024 | 12.87 - 19.93 | 77.7 - 50.2 |

### 4.8 Non-acceptable variants removed

Removed from default v3.1 path:

- local-only query decoder without global context;
- deterministic coordinate-hash coverage fallback as primary non-local mechanism;
- hard-clamp border sampling;
- late-only fusion where center/surround/global meet only at final gate.

Reason:

- these variants either violate project constraints or are less stable than corrected v3.1 components.

### 4.9 Primary references used in this design

- F^3: https://arxiv.org/abs/2509.25146
- DETR: https://arxiv.org/abs/2005.12872
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Deformable DETR official code (`ms_deform_attn.py`): https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
- Deformable ConvNets v2: https://arxiv.org/abs/1811.11168
- Perceiver IO: https://arxiv.org/abs/2107.14795
- Highway Networks: https://arxiv.org/abs/1505.00387
- FiLM: https://arxiv.org/abs/1709.07871
- GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- Segment Anything: https://arxiv.org/abs/2304.02643
- Segment Anything two-way transformer code: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py
- PointRend: https://arxiv.org/abs/1912.08193
- Feature Pyramid Networks (FPN): https://arxiv.org/abs/1612.03144
- HRNet: https://arxiv.org/abs/1908.07919
- U-Net: https://arxiv.org/abs/1505.04597
- Vision Transformers for Dense Prediction (DPT): https://arxiv.org/abs/2103.13413
- Differentiable Top-k (OT): https://arxiv.org/abs/2002.06504
- Learning Monocular Dense Depth from Events: https://arxiv.org/abs/2010.08350
- RAM-Net: https://arxiv.org/abs/2102.09320
- LIIF: https://arxiv.org/abs/2012.09161
- Instant-NGP: https://arxiv.org/abs/2201.05989
- DINO: https://arxiv.org/abs/2104.14294
- DINOv2: https://arxiv.org/abs/2304.07193

### 4.10 Why choose a compact coarse latent bank and how to verify it

Why needed:

- local-only query evidence is ambiguous in many cases.
- $C_t$ carries global context at low cost.

What "preserve information" means here:

- preserve task-relevant global information, not every pixel detail.
- local target identity is preserved in $(f_{q,0},h_{\text{point},q},h_{\text{ctr},q}^{0})$ and neighborhood context in $(l_q^{\text{ctx}},l_q^{\text{ring}})$; global disambiguation is preserved in $C_t$.

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


### 4.11 Cross-field mechanisms integrated to fix global-local separation

This section summarizes exactly why the corrected v3.1 design is theoretically sound and practically stable.

Audit finding that motivated the correction:

- In the previous design, $h_{\text{ctr}}$ and $h_{\text{global}}/h_{\text{ctx}}$ were mostly independent until late fusion, so center-to-context feedback was weak and context-to-center interaction happened too late.
- The corrected design coordinates B1-B8 as a full pipeline: center, near-surrounding, and global branches are coupled progressively before final decoding.

1. Stable sparse routing:
   - Change: B4 uses train-time differentiable top-k (`TopR_ST`) and infer-time hard top-k.
   - Why correct: removes routing discontinuity between training and deployment (consistent with sparse-routing practice in Deformable DETR-style decoders).
2. Local-global coverage without randomness dependence:
   - Change: B4 uses content-routed tokens plus learned landmark coverage and anchors.
   - Why correct: preserves non-local reach while remaining content-aware.
3. Local geometry fidelity:
   - Change: B1.1 uses fixed local core + learned local offsets + ring pooling + local reliability score.
   - Why correct: maintains guaranteed local support, encodes near-vs-far neighborhood structure, and provides a confidence signal for later global coupling.
4. Boundary-safe sparse reads:
   - Change: B6 uses reflective grid sampling.
   - Why correct: avoids border-collapse artifacts from hard clamp.
5. Point identity preservation:
   - Change: B3 builds nonlinear center token $h_{\text{ctr}}^{0}$ from point + surrounding descriptors (with multiplicative interaction); B7 injects near branch into center branch; B8 applies two-way coupling plus bounded carry.
   - Why correct: target-point information is preserved across multiple stages, while surrounding and global context are integrated progressively instead of one-shot late fusion.
6. Temporal disambiguation:
   - Change: A6/B8 add compact temporal memory $H_t$ shared across queries.
   - Why correct: depth ambiguity from sparse events is reduced by short-horizon temporal state.
7. Compute allocation consistency:
   - Change: B8 uses fixed sparse top-k ($k_t$) and B10 uses soft gate in training/hard threshold in inference.
   - Why correct: keeps deterministic runtime while reducing train-infer mismatch for optional extra compute.
8. Whole-pipeline coupling:
   - Change: B4 enforces query-neighbor coarse anchor $i_j^{\text{loc}}$, B5 adds center-connectivity prior in sampling logits, B6 explicitly extracts near evidence while keeping full global aggregation, and B7 feeds both into center-context coupling.
   - Why correct: avoids structural separation by design; center, near-surrounding, and global branches remain connected from routing through fusion.


### 4.12 Critical audit closure and correctness rationale

This section closes the audit and records what is now logically correct in v3.1.

#### 4.12.1 Applied corrections (already integrated above)

1. Routing stability: train-time differentiable sparse top-k + infer-time hard top-k in B4.
2. Local evidence: hybrid local sampling (fixed core + learned offsets) plus ring pooling and reliability scoring in B1.1.
3. Temporal context: compact memory state H_t shared across queries in A6/B8.
4. Connectivity robustness: query-neighbor coarse anchor is always included in B4 token subset.
5. Sampling robustness: B5 adds center-connectivity logit bias and B6 keeps explicit near evidence plus full global aggregation.
6. Boundary robustness: reflective grid sampling replaces hard clamping in B6.
7. Fusion correction: B8 upgraded to two-way coupling + bounded center-context carry.
8. Gate consistency: soft uncertainty gate in training and hard threshold in inference in B10.

#### 4.12.2 <span style="color:#BDBDBD">Remaining hypothesis-mode items (not default)</span>

1. <span style="color:#BDBDBD">Residual per-query calibration (Delta s_j, Delta b_j) in B9.</span>
2. <span style="color:#BDBDBD">Extra regularization strength for query-graph consistency under extreme low-event windows.</span>

<span style="color:#BDBDBD">These are kept as ablations only until validated.</span>

#### 4.12.3 Why the corrected design is logically sound

1. Constraint compliance: inference remains sparse-query only, with no dense depth decoding.
2. Stability: train/infer behavior is aligned for routing and uncertainty gating.
3. Information path correctness: exact-point identity is preserved by center branch, near-surrounding context is fused progressively, and global/temporal context is integrated through coordinated sparse routing and constrained fusion.
4. Geometric robustness: reflective sampling avoids border-collapse artifacts and preserves offset continuity.
5. Complexity behavior: runtime remains one shared precompute plus query-scaled decoding.

### 4.13 Primary References:

- F^3: https://arxiv.org/abs/2509.25146
- DETR: https://arxiv.org/abs/2005.12872
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Deformable DETR code (`ms_deform_attn.py`): https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
- Deformable ConvNets v2: https://arxiv.org/abs/1811.11168
- Segment Anything: https://arxiv.org/abs/2304.02643
- Segment Anything two-way transformer code: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py
- PointRend: https://arxiv.org/abs/1912.08193
- Feature Pyramid Networks (FPN): https://arxiv.org/abs/1612.03144
- HRNet: https://arxiv.org/abs/1908.07919
- U-Net: https://arxiv.org/abs/1505.04597
- Vision Transformers for Dense Prediction (DPT): https://arxiv.org/abs/2103.13413
- Perceiver IO: https://arxiv.org/abs/2107.14795
- Highway Networks: https://arxiv.org/abs/1505.00387
- FiLM: https://arxiv.org/abs/1709.07871
- GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202
- Differentiable Top-k (OT): https://arxiv.org/abs/2002.06504
- Learning Monocular Dense Depth from Events: https://arxiv.org/abs/2010.08350
- RAM-Net: https://arxiv.org/abs/2102.09320
- LIIF: https://arxiv.org/abs/2012.09161
- Instant-NGP: https://arxiv.org/abs/2201.05989
- DINO: https://arxiv.org/abs/2104.14294
- DINOv2: https://arxiv.org/abs/2304.07193

## 5) Semester timeline (16 weeks)

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

- <span style="color:#BDBDBD">Add teacher-student distillation.</span>
- <span style="color:#BDBDBD">Add optional DINO-guided boundary weighting.</span>

### Weeks 13-14: Robustness + ablations

- Query sampling ablation.
- Loss ablation.
- Hash-level/channel ablation.

### Weeks 15-16: Finalize paper package

- Main tables/figures.
- Failure case analysis.
- Draft submission-ready method + experiments sections.

## 6) Concrete implementation tasks in this repo

### 7.1 New modules to add

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

### 7.2 Reuse existing components

- F^3 backbone init/load from current `init_event_model` and checkpoints.
- Existing depth losses (SiLog/SiLogGrad) as templates.
- Existing benchmarking style from `test_speed.py`.

### 7.3 Fast path engineering

- Keep single-batch deployment mode in mind (similar to `hubconf.py` single-batch path).
- <span style="color:#BDBDBD">Consider optional precomputed hashed features for ultra-low latency fixed-window deployments.</span>

## 7) Risks and mitigation

Risk 1: Accuracy drop in low-event or textureless regions.

- Mitigation: teacher distillation + uncertainty estimation + mixed query sampling.

Risk 2: Sparse model overfits query distribution.

- Mitigation: randomized query curriculum and robust validation splits.

Risk 3: Runtime not significantly better than dense for medium Q.

- Mitigation: optimize routing/coverage and deformable sampling kernels; tune $P_c$, $R$, $U$, $M_a$, $H$, $L$, $M$.

Risk 4: Label sparsity from LiDAR hurts training.

- Mitigation: two-stage pseudo->metric strategy (already validated in F^3 depth pipeline).

## 8) Expected deliverables

By semester end, target:

1. A working sparse query-point depth model from events.
2. Runtime scaling plots vs query count showing clear win over dense methods.
3. Accuracy tables on M3ED/DSEC/MVSEC query-level metrics.
4. Ablations proving which components matter.
5. Draft paper/report with method, theory intuition, and deployment relevance.

## 9) Immediate next 2-week action list

1. Reproduce dense F^3+DA-V2 baseline and verify metrics/runtime on one dataset split.
2. Implement query benchmark wrapper that samples points from dense outputs and LiDAR valid pixels.
3. Implement GCQD-v3 prototype with coarse routing + coverage tokens + anchor bank + point-locked residual fusion.
4. Produce first plots for $T_{\text{precompute}}$, $T_{\text{query}}$, $T_{\text{total}}$ and first query-accuracy comparison.

---

This plan directly follows your constraint (no full dense NN at inference for sparse requests) while leveraging what already works in F^3.
