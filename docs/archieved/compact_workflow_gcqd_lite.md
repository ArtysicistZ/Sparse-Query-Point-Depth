# Compact Workflow: GCQD-lite (Inverse F^3 Sparse Query Depth)

## 1) Goal

Predict depth only at user-requested query pixels (not dense `H x W` depth), while keeping global scene context.

Runtime target:
$$
T_{\text{total}}(K)=T_{\text{precompute}}+T_{\text{query}}(K),\qquad K\ll HW
$$

## 2) Why this is compact

F^3 itself is efficient because the shared representation is computed once, then reused.  
GCQD-lite follows the same principle:

1. Compute shared event features once per window.
2. For each query, run only a small sparse decoder.

No dense depth map is produced at inference.

## 3) GCQD-lite Runtime (6 Steps)

### Step 1: Shared precompute once per event window

Input:
- Event window `E_t`.

Compute:
$$
E_t\rightarrow F_t^{(1:3)},\ C_t,\ G_t,\ M_t,\ R_t,\ s_t,\ b_t
$$

Meaning:
- `F_t^(1:3)`: fine/mid/coarse feature pyramid.
- `C_t`: compact coarse latent bank.
- `G_t`: always-visible global anchor tokens.
- `M_t`: local importance map.
- `R_t`: event-recency map for smooth hash interpolation.
- `s_t,b_t`: window-level depth calibration.

### Step 2: Query embedding

For each query `q_j`:
$$
f_{j,0}=\mathrm{Bilinear}(F_t^{(1)},q_j),\qquad
l_j^{\text{ctx}}=\mathrm{LocalSample}(F_t^{(1)},M_t,q_j),\qquad
e_j=e(q_j),\qquad
z_j=W_z[f_{j,0};l_j^{\text{ctx}};e_j],\qquad
h_{\text{point},j}=W_p[f_{j,0};e_j]
$$

Default local sampler:
- Fixed `N_loc = 121` points (core local grid + informative outer points).

### Step 3: Sparse global subset selection

$$
z_j,C_t,G_t\rightarrow R_j,U_j,S_j,\bar c_j
$$
with:
$$
S_j=R_j\cup U_j
$$

Meaning:
- `R_j`: top content-routed global tokens.
- `U_j`: coverage tokens (prevents missing far context).
- `\bar c_j`: query-conditioned global summary.

### Step 4: Sparse non-local read

Use deformable multiscale sampling only over `S_j`:
$$
h_{\text{global},j}=\mathrm{DefRead}(F_t^{(1:3)},S_j,z_j,l_j^{\text{ctx}},\bar c_j),\qquad
h_{\text{ctr},j}=\mathrm{CenterRead}(F_t^{(1:3)},q_j,h_{\text{point},j})
$$

### Step 5: Single-shot fusion + depth output

$$
h_{\text{ctx},j}=\mathrm{CtxFuse}(h_{\text{global},j},l_j^{\text{ctx}},\bar c_j),\qquad
h_{\text{fuse},j}=h_{\text{ctr},j}+\sigma(\mathrm{MLP}_g([h_{\text{ctr},j};h_{\text{ctx},j};z_j]))\odot h_{\text{ctx},j}
$$
$$
\rho_j=s_t\,h_r(h_{\text{fuse},j})+b_t,\qquad
\hat d_j=\frac{1}{\mathrm{softplus}(\rho_j)+\varepsilon}
$$

### Step 6: Optional uncertainty

$$
\sigma_j=h_\sigma(h_{\text{fuse},j})
$$

Output:
$$
\{(\hat d_j,\sigma_j)\}_{j=1}^{K}
$$

## 4) Default profile vs research profile

### Default deployment profile (recommended)

- Fixed local sampling: `N_loc = 121`.
- Single-shot fusion (no iterative coupling).
- No second-hop reroute.

### Extended research profile (ablations only)

- Adaptive local expansion (`121 -> 169` on hard local queries).
- Context-drop and point-lock ablations.
- Uncertainty-triggered second-hop rerouting.

## 5) Minimal CUDA kernel plan

1. `K0_F3Precompute`: `E_t -> F_t^(1:3), C_t, G_t, M_t, R_t, s_t, b_t`
2. `K1_QueryEmbed`: `q -> f_q0, l_q_ctx, e(q), z_q, h_point,q`
3. `K2_RouteSubset`: `z_q, C_t, G_t -> S_q, \bar c_q`
4. `K3_DefGather`: sparse deformable gather over `S_q`
5. `K4_FuseDepth`: fusion + calibrated depth head
6. `K5_Uncertainty` (optional)

## 6) Complexity (compact form)

Per-window precompute:
$$
\mathcal{O}(HWC+P_cM_a d)
$$

Per-query (default profile):
$$
\mathcal{O}\!\left(N_{\text{loc}}d+(R+U)d+(R+U)HLMd+M_a d\right),\qquad N_{\text{loc}}=121
$$

Total:
$$
T_{\text{total}}(K)=T_{\text{precompute}}+K\cdot T_{\text{query}}
$$

## 7) Step-wise Runtime Estimate (RTX 4090, Conservative)

Assumptions:

- Hardware: 1x RTX 4090, FP16 inference, batch size 1.
- Event window: 20 ms.
- F^3 core baseline used here: `120 Hz` at HD, i.e. `8.33 ms`.
- We keep conservative overhead ranges and do not use the most optimistic deployment-only timings (e.g., heavily precomputed hash paths).
- Configuration: `P_c=256`, `M_a=8`, `R=6`, `U=2`, `H=4`, `L=3`, `M=4`, fixed `N_loc=121`.

Per-step runtime model:
$$
T_i(K)=\alpha_i+\beta_i K
$$

Estimated step budgets:

| Step | Description | Runtime model (ms) |
|---|---|---|
| 1 | Shared precompute (`K0_F3Precompute`) | fixed `8.68 - 9.33` |
| 2 | Query embedding (`K1_QueryEmbed`) | `0.03 - 0.07 + (0.0010 - 0.0020)K` |
| 3 | Subset routing (`K2_RouteSubset`) | `0.02 - 0.05 + (0.0006 - 0.0012)K` |
| 4 | Deformable sparse read (`K3_DefGather`) | `0.03 - 0.08 + (0.0018 - 0.0042)K` |
| 5 | Fuse + depth (`K4_FuseDepth`) | `0.01 - 0.03 + (0.0006 - 0.0016)K` |
| 6 | Uncertainty (`K5_Uncertainty`, optional) | `0.005 - 0.02 + (0.0003 - 0.0010)K` |

Default deployment (without optional uncertainty):
$$
T_{\text{GCQD-lite}}(K)\in\left[8.77+0.0040K,\ 9.69+0.0100K\right]\ \text{ms}
$$

If uncertainty head is enabled:
$$
T_{\text{GCQD-lite+unc}}(K)\in\left[8.78+0.0043K,\ 9.71+0.0110K\right]\ \text{ms}
$$

Predicted totals (default deployment):

| Query count `K` | Total runtime (ms) | Throughput (Hz) |
|---|---:|---:|
| 1 | `8.77 - 9.70` | `114.0 - 103.1` |
| 64 | `9.03 - 10.33` | `110.8 - 96.8` |
| 256 | `9.79 - 12.25` | `102.1 - 81.6` |
| 1024 | `12.87 - 19.93` | `77.7 - 50.2` |

Effectiveness check:

- For sparse usage (`K <= 256`), runtime stays near precompute cost and remains in practical real-time range.
- For heavier sparse loads (`K = 1024`), it is still typically real-time-ish but no longer "near-constant".
- Compared with dense depth decoders that usually run in the `30 - 60 ms` range at HD, GCQD-lite remains favorable across low-to-mid query budgets.
- If a target system needs strict `<20 ms`, default-profile settings remain feasible up to roughly `K <= 1000` in this conservative estimate.

### 7.2 Relative to pure F^3 (~448 Hz README benchmark)

Use pure F^3 reference:
$$
T_{\mathrm{F^3,README}}=2.23\ \text{ms}
$$

With the same conservative query overhead model:
$$
T_{\mathrm{GCQD}}(K)\in[2.67+0.004K,\ 3.59+0.010K]\ \text{ms}
$$

Relative factor:
$$
\text{Overhead factor}=\frac{T_{\mathrm{GCQD}}(K)}{2.23}
$$

| Query count `K` | GCQD runtime (ms) | Throughput (Hz) | Relative to pure F^3 |
|---|---:|---:|---:|
| 1 | `2.67 - 3.60` | `374.5 - 277.8` | `1.20x - 1.61x` |
| 64 | `2.93 - 4.23` | `341.3 - 236.4` | `1.31x - 1.90x` |
| 256 | `3.69 - 6.15` | `271.0 - 162.6` | `1.66x - 2.76x` |
| 1024 | `6.77 - 13.83` | `147.7 - 72.3` | `3.04x - 6.20x` |

Interpretation:

- GCQD is not expected to be faster than pure F^3 representation.
- The useful comparison target is dense depth decoding cost, not pure feature extraction.

### 7.1 If we simplify F^3 first (recommended)

When precompute dominates, optimize:
$$
T_{\text{total}}(K)=T_{\mathrm{F^3}}+T_{\text{query}}(K)\ \ \Rightarrow\ \ T_{\mathrm{F^3}}\ \text{first}
$$

Use conservative `F^3-lite` precompute targets:

- Tier-1: `T_F3lite = 7.0 - 8.5 ms`
- Tier-2: `T_F3lite = 6.0 - 7.5 ms`

Keep query-side costs unchanged from Section 7:
$$
T_M\in[0.35,1.00]\ \text{ms},\qquad T_Q(K)\in[0.09+0.004K,\ 0.36+0.010K]\ \text{ms}
$$

Then:
$$
T_{\text{Tier-1}}(K)\in[7.44+0.004K,\ 9.86+0.010K]\ \text{ms}
$$
$$
T_{\text{Tier-2}}(K)\in[6.44+0.004K,\ 8.86+0.010K]\ \text{ms}
$$

Predicted totals:

| Query count `K` | Tier-1 runtime (ms) | Tier-2 runtime (ms) |
|---|---:|---:|
| 1 | `7.44 - 9.87` | `6.44 - 8.87` |
| 64 | `7.70 - 10.50` | `6.70 - 9.50` |
| 256 | `8.46 - 12.42` | `7.46 - 11.42` |
| 1024 | `11.54 - 20.10` | `10.54 - 19.10` |

Engineering implication:

- If this project needs a clear latency jump, simplifying F^3 precompute is the highest-leverage move.
- Keep point-locked query branch unchanged while simplifying F^3 to protect exact-point accuracy.

## 8) Training workflow (compact)

1. Precompute shared features for each event window.
2. Sample sparse query set from valid depth pixels.
3. Run GCQD-lite forward only on sampled queries.
4. Compute sparse depth losses (`L_point`, `L_silog,q`, rank/uncertainty as needed).
5. Backprop through sparse decoder (and optionally fine-tune F^3 backbone).

## 9) Practical starting hyperparameters

- `P_c=256`, `M_a=8`
- `R=6`, `U=2`
- `H=4`, `L=3`, `M=4`
- `N_loc=121` (fixed default)

## 10) What is explicitly removed for compactness

- Dense per-pixel depth decoding at inference.
- Always-on multi-hop refinement.
- Heavy iterative attention loops per query.

This keeps the method close to F^3's efficiency philosophy while still using global context for each sparse query.

## 11) Evidence-backed efficiency choices (cross-field)

These papers motivate removing redundant iterative loops while keeping local-global interaction:

1. Deformable DETR (`https://arxiv.org/abs/2010.04159`)
- Use a small learned sample set per query instead of dense attention over all keys.
- Direct implication: keep one sparse deformable read pass as default.

2. Perceiver IO (`https://arxiv.org/abs/2107.14795`)
- Query outputs are produced by cross-attending to a compact latent bottleneck.
- Direct implication: one query-to-latent pass is usually enough before output head.

3. Longformer (`https://arxiv.org/abs/2004.05150`) and BigBird (`https://arxiv.org/abs/2007.14062`)
- Efficient sparse attention is built from local + selected global connectivity in one layer/pattern.
- Direct implication: prefer one sparse connectivity pattern over repeated full loops.

4. LIIF (`https://arxiv.org/abs/2012.09161`)
- Coordinate-based querying uses local features + coordinate code in one pass without iterative refinement.
- Direct implication: sparse coordinate queries do not require mandatory multi-hop loops.

Therefore the default GCQD-lite policy is:
- keep one-pass sparse local-global fusion,
- enable extra iterative passes only for hard-query ablations.

## 12) Critical audit updates (GCQD-lite -> GCQD-lite-v3)

The compact workflow keeps the same 6-step runtime path, but replaces four weak points:

1. Routing stability:
- Replace train-time soft / inference-time hard mismatch with differentiable sparse routing during training and hard top-k only at deployment.

2. Local sampling geometry:
- Replace purely isotropic outer ring sampling with anisotropic sampling aligned to local structure/motion direction.

3. Temporal context:
- Add compact temporal latent memory:
$$
H_t=\eta H_{t-1}+(1-\eta)\mathrm{Pool}(C_t)
$$
- Each query reads `H_t` once in Step 5 fusion.

4. Calibration granularity:
- Replace global-only `(s_t,b_t)` with query-wise calibration regularized to global priors:
$$
s_j=\mathrm{softplus}(w_s^\top[h_{\text{fuse},j};\bar c_j]+b_s),\qquad
b_j=w_b^\top[h_{\text{fuse},j};\bar c_j]+b_b
$$

Net effect:
- Slightly higher per-query arithmetic than v2.
- Better stability and lower catastrophic miss risk.
- Still no dense `H x W` depth decoding at inference.
