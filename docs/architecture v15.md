# Research Plan: SPD v15 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-26
Version: v15 (v14 → remove Q2Q + L4 self-attn, native L3/L4 dims, local DPT patch refinement per query)

> **v15 motivation** (post-Exp 9/10): v14 plateau analysis reveals three structural issues:
> 1. **Redundant self-attention.** L4 self-attention (886K) is redundant — ConvNeXt V2 is designed to excel without self-attention (proven on ImageNet without attention), and for depth prediction the queries themselves handle global perception via MSDA and GlobalCrossAttn. Q2Q self-attention (750K across 5 layers) ignores spatial structure — the canvas already mediates inter-query communication spatially, which is the correct inductive bias for depth.
> 2. **Massive L3/L4 information bottleneck.** Projecting L3 (384ch) → 192 loses 50% of channels; L4 (768ch) → 192 loses 75%. These channels encode FCMAE-pretrained scene understanding. A shared global projection destroys the channel diversity before any query can selectively read from it. Per-head projections from native dimensions allow different attention heads to extract different aspects.
> 3. **No sub-stride-8 local precision.** Canvas L2 (stride 8) and MSDA cannot encode depth structure smaller than 8px. Queries are at exact pixel positions — they need fine local context. DPT solves this by progressive upsampling at every fusion stage. v15 applies the same principle **locally** per query: a 7×7 L1 patch (covering 28×28 pixels at stride 4) is upsampled DPT-style to stride 1, giving each query genuine sub-8px structural awareness.
>
> **Philosophy change:** v14 followed "add attention wherever ambiguous." v15 follows "trust the backbone, remove redundant attention, add targeted fine-grained local processing."

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Depth $\hat{d}_i$ at each queried pixel.

**Constraint:** No dense $H \times W$ depth decoding. Shared encoder pass once, then sparse per-query decoding:

$$T_{\text{total}}(K) = T_{\text{encode}}(I) + T_{\text{decode}}(K \mid \text{features})$$

**Baseline:** DAv2-S (ViT-S encoder + DPT dense decoder, ~49G MACs).
**Ours (v15):** ConvNeXt V2-T encoder + sparse MSDA decoder + local DPT patch refinement per query.

---

## 2. Architecture Overview

### 2.1 Key Changes from v14 → v15

| Component | v14 | v15 | Rationale |
|---|---|---|---|
| L4 self-attention | 2× FullSelfAttn₁₉₂ (886K) | **Removed** | ConvNeXt V2 proven without SA; queries handle global perception |
| Q2Q self-attention | In all 5 decoder layers (~150K each) | **Removed** | Canvas mediates inter-query context spatially — right inductive bias for depth |
| L3 dimension | 384 → 192 in neck | **384 (native)** | 2× more information for MSDA sampling and seed |
| L4 dimension | 768 → 192 in neck + SA | **768 (native)** | 4× more information; per-head projections inside attention |
| MSDA V maps | Identity (all 192) | **Per-head proj for L3/L4** | Each head extracts different aspects from 384/768-dim features |
| Decoder sublayers | cross-attn → canvas → Q2Q → FFN (4) | **cross-attn → canvas → FFN (3)** | Q2Q removed; canvas already handles inter-query spatial communication |
| Depth head | MLP(192→384→1) × exp(s) | **MLP([h; local_feat]→384→1) × exp(s)** | Local fine-grained L1 patch feature fused with global h |
| Local refinement | None | **LDRM: 7×7 L1 patch → ×4 upsample → stride-1 feature per query** | Sub-8px structural awareness; DPT-style local→global→local |
| Dense training | Canvas L2 aux head (55K) | **LDRM 28×28 patch dense SILog (training only)** | 128K+ supervised positions per image vs 14K; per-query local patch |

### 2.2 Architecture Diagram

```
RGB Image [H × W × 3]  (416 × 544 at training)
  │
  ▼
ConvNeXt V2-T (FCMAE pre-trained, fine-tuned with 0.1× LR)
  Stage 1: H/4  × W/4  × 96     (3× ConvNeXt V2 block, DW-Conv k7 + GRN)
  Stage 2: H/8  × W/8  × 192    (3× ConvNeXt V2 block)
  Stage 3: H/16 × W/16 × 384    (9× ConvNeXt V2 block)
  Stage 4: H/32 × W/32 × 768    (3× ConvNeXt V2 block)
  │
  ▼
Projection Neck (v15: partial projection)
  L1: Conv(96→192, k1) + LN     → [H/4  × W/4  × 192]   (for canvas, seed, MSDA identity V)
  L2: Conv(192→192, k1) + LN    → [H/8  × W/8  × 192]   (for canvas, seed, MSDA identity V)
  L3: NO global projection       → [H/16 × W/16 × 384]   (native; per-head proj inside MSDA)
  L4: NO global projection       → [H/32 × W/32 × 768]   (native; per-head proj inside MSDA/B3b)
  │
  ▼  (No L4 self-attention in v15)
  │
Pre-compute (once per image)
  proj_L3:    Linear(384→192) applied to L3 → [H/16 × W/16 × 192]  (for canvas + seed)
  proj_L4:    Linear(768→192) applied to L4 → [H/32 × W/32 × 192]  (for canvas + seed)
  B3b L4 KV:  per-head Linear(768→d_head) for K, V on all N_L4 tokens
  │
  ▼
Triple Spatial Canvas (v14: L2+L3+L4, Section 6.5):
  F_L2⁰ = L2         [H/8  × W/8  × 192]   stride 8
  F_L3⁰ = proj_L3    [H/16 × W/16 × 192]   stride 16
  F_L4⁰ = proj_L4    [H/32 × W/32 × 192]   stride 32
  │
Per-query decoder (K queries in parallel):
  B1: Multi-scale seed from all 4 levels (L1@192, L2@192, proj_L3@192, proj_L4@192) + Fourier PE
  │
  3× MSDA DecoderLayer (3 sublayers, no Q2Q):
  │  ├── MSDA Cross-Attn: L1/L2 (identity V@192), L3 (per-head V proj 384→d_head), L4 (per-head V proj 768→d_head)
  │  ├── Triple Canvas Write-Smooth-Read (joint-gate cascade L4→L3→L2, then DWConv smooth)
  │  └── FFN
  │
  2× B3b GlobalCrossAttn (3 sublayers, no Q2Q):
  │  ├── Full Cross-Attn (Q=h, KV from L4@768 via per-head projections, all N_L4 tokens)
  │  ├── Triple Canvas Write-Smooth-Read
  │  └── FFN
  │
  LDRM: Local DPT Patch Refinement per query [NEW in v15]
  │  ├── Extract 7×7 patch from L1@192 (stride 4, covers 28×28 px per query)
  │  ├── Bilinear ×2 → [7×7→14×14, d: 192→96] + Conv(192→96, k3) + GELU + residual
  │  ├── Bilinear ×2 → [14×14→28×28, d: 96→32] + Conv(96→32, k3) + GELU
  │  └── Sample at exact query sub-pixel offset → local_feat [K, 32]
  │
  Depth Head (v15):
    h_combined = concat(h_final [K,192], local_feat [K,32])  → [K, 224]
    depth = exp(MLP(224→384→1) × exp(s))
```

---

## 3. Encoder: ConvNeXt V2-T (unchanged from v14)

| Parameter | Value |
|-----------|-------|
| Architecture | ConvNeXt V2-Tiny |
| Stages | 4 (depths: [3, 3, 9, 3]) |
| Channels | [96, 192, 384, 768] |
| Strides | [4, 8, 16, 32] |
| Input size | 416×544 (BTS augmentation: random crop from 480×640) |
| Block type | ConvNeXt V2 (DW Conv k7 + GRN + PW Conv) |
| Params | ~28.6M |

**Stage outputs (416×544 input):**

```
RGB 416×544×3
  → Stem: Conv(3→96, k4, s4)         104×136×96
  → Stage 1: 3× ConvNeXt V2 block  → 104×136×96    L1, stride 4
  → Down: LN + Conv(96→192, k2, s2)   52×68×192
  → Stage 2: 3× ConvNeXt V2 block  →  52×68×192    L2, stride 8
  → Down: LN + Conv(192→384, k2, s2)  26×34×384
  → Stage 3: 9× ConvNeXt V2 block  →  26×34×384    L3, stride 16
  → Down: LN + Conv(384→768, k2, s2)  13×17×768
  → Stage 4: 3× ConvNeXt V2 block  →  13×17×768    L4, stride 32
```

**Why ConvNeXt V2 does not need added self-attention:**

ConvNeXt V2 was specifically designed and proven to achieve state-of-the-art performance without any self-attention mechanism. The GRN (Global Response Normalization) layer provides channel-wise competition across the spatial dimension — a lightweight global normalization that partially substitutes for attention's channel re-weighting. FCMAE pre-training forces L4 to encode full scene context (masked patches require global reasoning). Stage 3's 9× DW-Conv k7 provides theoretical RF of ~55×55 at stride 16, covering the full image at 416×544.

**For depth prediction specifically:** The queries are the agents of global perception. MSDA and B3b give each query selective access to any part of the feature pyramid. Adding self-attention to L4 before queries touch it computes global context twice — once in ConvNeXt's hierarchical processing, once in the redundant SA. This wastes 886K parameters on a computation the query mechanism already performs.

---

## 4. Projection Neck (v15: Partial Projection)

v15 breaks with v14's uniform d=192 neck. The key insight: global projection to 192 is a SHARED bottleneck that discards 50–75% of L3/L4 channel information before any query can selectively read from it. Per-head projections inside attention allow different heads to extract different semantic aspects from the full native-dimension features.

### 4.1 Neck Architecture

```
Stage 1 [H/4  × W/4  ×  96] → Conv( 96→192, k1) + LN → L1 [H/4  × W/4  × 192]   stride 4
Stage 2 [H/8  × W/8  × 192] → Conv(192→192, k1) + LN → L2 [H/8  × W/8  × 192]   stride 8
Stage 3 [H/16 × W/16 × 384] → NO global projection     → L3 [H/16 × W/16 × 384]   stride 16 (native)
Stage 4 [H/32 × W/32 × 768] → NO global projection     → L4 [H/32 × W/32 × 768]   stride 32 (native)
```

**Why L1 and L2 are still projected to 192:** L1 (96ch) must be upsampled anyway (96 < d=192). L2 is already at 192. Canvas L2 and the seed use 192-dim features. No bottleneck issue at these levels.

**Why L3 (384) and L4 (768) are NOT globally projected:**

| Projection type | What happens | Problem |
|---|---|---|
| Global 768→192 (v14) | Single Linear applied to all 221 L4 tokens uniformly | All heads share one 192-dim subspace; the other 576 dimensions are discarded before any query sees them |
| Per-head 768→32 (v15) | Each of 6 heads applies its own Linear(768→32) during attention sampling | Head 1 might capture depth-from-shading cues, Head 2 captures surface normal cues, etc. — impossible if all heads start from same 192-dim projected space |

The 768-dim ConvNeXt V2 L4 features encode rich, diverse scene semantics from FCMAE pretraining. A global 4× compression before any task-specific selection is architecturally premature.

**Neck params (v15):** Conv(96→192) ~19K + Conv(192→192) ~37K = **~56K** (vs ~278K in v14; save 222K from removing L3/L4 projections).

### 4.2 Projection for Canvas and Seed (pre-compute, once per image)

Canvas and seed require d=192-dim features from L3 and L4. These are produced by separate pre-compute projections:

```
proj_L3: Linear(384→192) applied to L3 → [H/16 × W/16 × 192]  (for canvas F_L3⁰ + seed f_q^(3))
proj_L4: Linear(768→192) applied to L4 → [H/32 × W/32 × 192]  (for canvas F_L4⁰ + seed f_q^(4))
```

These are NOT the MSDA value sources — they only serve canvas initialization and seed construction. MSDA reads directly from the native-dimension L3@384 and L4@768 feature maps via per-head projections.

**Pre-compute projection params:** Linear(384→192) ~74K + Linear(768→192) ~148K = **~222K**.

### ~~4.3 L4 Self-Attention~~ (REMOVED in v15)

> **Removed.** See Section 2.1 rationale. The 886K parameters are reallocated to more impactful use: LDRM (Section 6.5) and richer L3/L4 per-head projections.
>
> **Counterargument addressed:** "Stage 4 has only 3 local conv blocks — insufficient for global relationships." Response: (1) ConvNeXt V2's hierarchical architecture propagates global context through downsampling stages. (2) B3b global cross-attention provides exact global L4 access for every query. The queries receive global L4 coherence through B3b — L4 itself does not need to pre-compute this coherence redundantly.

---

## 5. Pre-compute (once per image)

**Input:** Encoder pyramid {L1@192, L2@192, L3@384, L4@768}.
**Output:** Value maps for MSDA, canvas initialization, seed projections, B3b KV.

### 5.1 MSDA Value Sources

| Level | Native dim | MSDA treatment | Params |
|-------|:---:|---|---:|
| L1 | 192 | Identity V map — each head slices its d_head=32 channels | 0 |
| L2 | 192 | Identity V map — same as v14 | 0 |
| L3 | 384 | Per-head projection: 6 × Linear(384→32) per MSDA layer | 74K/layer |
| L4 | 768 | Per-head projection: 6 × Linear(768→32) per MSDA layer | 147K/layer |

Per-head projection in MSDA for L3 and L4: when deformable attention samples 4 reference positions from L3 or L4, each head applies its own independent Linear to the sampled 384 or 768-dim features before aggregation. This is the "lazy projection" approach — only applied to the sampled positions (4 per head per query), not pre-applied to the entire feature map.

**Why per-head lazy projection is preferable to global pre-projection:**
- Global pre-projection: Linear(768→192) maps ALL 221 L4 tokens, sharing one projection for all heads. Information loss is at all positions before any query selection.
- Lazy per-head: projection happens AFTER spatial selection (at 4 sampled positions). Each head independently decides which 32 dimensions of the 768-dim feature to use. Head 1 might extract surface-reflectance cues for specular depth disambiguation, Head 6 might extract structural edge cues. Impossible with shared global projection.

### 5.2 Canvas and Seed Projections

```
proj_L3(L3) → proj_L3_feat [H/16 × W/16 × 192]   (shared: canvas F_L3⁰ + seed f_q^(3))
proj_L4(L4) → proj_L4_feat [H/32 × W/32 × 192]   (shared: canvas F_L4⁰ + seed f_q^(4))
```

These are pre-computed once and stored.

### 5.3 B3b KV Projections (full attention over all N_L4 tokens)

B3b does full attention over all N_L4 = 221 tokens (at 416×544). Requires K,V projected from L4@768 to attention dimension. Per-head projections preserve diversity:

```
Per B3b layer ℓ:
  K_L4^(ℓ,h) = Linear_K^(ℓ,h)(L4)   [N_L4 × 32]  per head h=1..6
  V_L4^(ℓ,h) = Linear_V^(ℓ,h)(L4)   [N_L4 × 32]  per head h=1..6
```

Params per B3b layer: 6 × (Linear(768→32) for K + Linear(768→32) for V) = 6 × 2 × 24,576 ≈ **295K/layer**, 2 layers = **590K** (vs 148K in v14 — cost of keeping L4 at 768).

**Pre-compute total (v15):**
222K (proj_L3 + proj_L4) + 590K (B3b KV) = **~812K** (vs 148K in v14).

---

## 6. Per-Query Decoder (v15)

### 6.0 Symbol Table

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $d$ | Core query dimension | 192 |
| $d_h$ | Per-head dimension ($d / H$) | 32 |
| $H$ | Attention heads | 6 |
| $L$ | Feature levels for MSDA | 4 |
| $N_{\text{pts}}$ | Sampling points per head per level | 4 |
| $q = (u,v)$ | Query pixel coordinate | — |
| $h^{(n)}$ | Query representation after layer $n$ | 192 |
| $\text{local}_q$ | L1 patch feature at query sub-pixel position | 32 |
| $\hat{d}_q$ | Depth prediction | scalar |

### 6.1 B1: Multi-Scale Query Seed (updated for v15 dims)

Sample from all 4 pyramid levels at query center using their pre-computed d=192 representations:

$$f_q^{(\ell)} = \text{Bilinear}(L_\ell^{192},\; p_q^{(\ell)}) \quad \in \mathbb{R}^{192}$$

where $L_1^{192} = L_1$, $L_2^{192} = L_2$, $L_3^{192} = \text{proj\_L3\_feat}$, $L_4^{192} = \text{proj\_L4\_feat}$.

$$h^{(0)} = \text{LN}\!\left(W_{\text{seed}} \,[f_q^{(1)}; f_q^{(2)}; f_q^{(3)}; f_q^{(4)}; \text{pe}_q]\right) \in \mathbb{R}^d$$

$W_{\text{seed}} \in \mathbb{R}^{d \times (4d+32)} = \mathbb{R}^{192 \times 800}$ — unchanged from v14 (**~154K**). The pre-compute projections (proj_L3, proj_L4) produce d=192 features so the seed projection matrix is the same shape.

**No pos_q for Q2Q** (Q2Q removed). The Fourier PE is still used for the seed only.

### 6.2 MSDA Decoder (3 layers, 3 sublayers per layer)

Each MSDA decoder layer $n = 1, 2, 3$ — **Q2Q removed:**

```
h ──→ LN ──→ MSDA Cross-Attn (L1/L2 identity V; L3/L4 per-head V projection) ──→ +residual
  ──→ LN ──→ Triple Canvas Write-Smooth-Read (Section 6.4) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

#### 6.2.1 MSDA with Per-Head L3/L4 V Projections

**Step 1–2:** Reference points and offset prediction unchanged from v14.

**Step 3 — Attention weights:** unchanged from v14 (directly predicted from h, not QK dot-product).

**Step 4 — Sample and aggregate (v15 change):**

For L1, L2 (d=192): unchanged from v14 — slice d_head=32 channel window.

For L3 (native 384):
$$\text{sample}_{i,\ell=3} = \text{Bilinear}(L_3,\; p_q^{(3)} + \Delta_{i,3,m}) \in \mathbb{R}^{384}$$
$$v_{i,\ell=3,m} = W_{V,i}^{L3} \cdot \text{sample}_{i,\ell=3,m} \in \mathbb{R}^{d_h=32}, \quad W_{V,i}^{L3} \in \mathbb{R}^{32 \times 384}$$

For L4 (native 768): same pattern, $W_{V,i}^{L4} \in \mathbb{R}^{32 \times 768}$.

**MSDA per-layer params (v15):**

| Component | v14 | v15 | Delta |
|-----------|----:|----:|------:|
| $W_\text{off}$, $W_\text{attn}$ | 55K | 55K | 0 |
| $W_O$ (output) | 37K | 37K | 0 |
| V proj L3: 6 × Linear(384→32) | 0 | 74K | +74K |
| V proj L4: 6 × Linear(768→32) | 0 | 147K | +147K |
| Canvas (write/read/gate/LN) | 148K | 148K | 0 |
| Q2Q (4 × d²) | 148K | **0** | −148K |
| FFN (d→4d→d) | 296K | 296K | 0 |
| LNs | ~1K | ~1K | 0 |
| **Per layer total** | **~685K** | **~758K** | **+73K** |

3 MSDA layers total: **~2,274K** (vs ~2,055K in v14).

#### 6.2.2 Why Q2Q is Removed

> **Removed.** Q2Q inter-query self-attention ignores the spatial structure of depth. Two queries communicating directly regardless of their 2D image distance is the wrong inductive bias: a query at the ceiling should not directly influence a query at the floor. The correct mechanism is SPATIAL — nearby queries on the same surface influence each other through shared canvas cells and DWConv propagation.
>
> The canvas already provides exactly this: (1) nearby queries write to nearby cells; (2) DWConv smooth propagates information to adjacent cells; (3) the joint-gate cascade (L4→L3→L2) ensures coarse scale context flows into fine scale. This spatial propagation respects depth scene structure — same-surface queries share canvas cells, cross-boundary queries are separated by distance that DWConv does not bridge in 1–2 layers.
>
> Q2Q's K×K global attention specifically violates the locality principle for depth. A depth network should reason locally (what's this surface?) and globally (where am I in the room?) — canvas handles local, MSDA+B3b handles global. Direct query-to-query dense attention adds a third mechanism that is neither local nor structured globally.

### 6.3 B3b: Global L4 Cross-Attention (2 layers, 3 sublayers per layer)

Full attention over all N_L4=221 tokens (at 416×544) using per-head L4@768 projections:

```
h ──→ LN ──→ Full Cross-Attn (Q=h@192, KV from L4@768 per-head projections) ──→ +residual
  ──→ LN ──→ Triple Canvas Write-Smooth-Read (Section 6.4) ──→ +residual
  ──→ LN ──→ FFN (192 → 768 → 192, GELU) ──→ +residual ──→ h_out
```

Uses pre-computed per-head K,V (Section 5.3). Each head attends to L4 through a different 32-dim projection of the 768-dim features — preserving the full information diversity.

**B3b params per layer:** Q-proj ~37K + KV projections (counted in pre-compute) + canvas ~148K + FFN ~296K + LNs ~1K = **~482K/layer**, 2 layers = **~964K** (vs ~1,334K in v14; savings from removing Q2Q and also the per-query Q-proj is unchanged but B3b KV cost is in pre-compute).

### 6.4 Triple Spatial Canvas (inherited and extended from v14)

Three shared dense feature maps at strides 8, 16, 32 (initialized from proj_L3_feat and proj_L4_feat for L3/L4). Architecture unchanged from v14: per-layer write → joint-gate cascade (L4→L3→L2) → DWConv smooth → read.

**Joint-gate cascade (v14 Section 10.5, now baseline in v15):**

```python
# After write, before smooth:
l4_up = F.interpolate(canvas['L4'], size=canvas['L3'].shape[2:], mode='bilinear')
gate_43 = torch.sigmoid(gate_L4toL3_fine(canvas['L3']) + gate_L4toL3_coarse(l4_up))
canvas['L3'] = canvas['L3'] + gate_43 * l4_up

l3_up = F.interpolate(canvas['L3'], size=canvas['L2'].shape[2:], mode='bilinear')
gate_32 = torch.sigmoid(gate_L3toL2_fine(canvas['L2']) + gate_L3toL2_coarse(l3_up))
canvas['L2'] = canvas['L2'] + gate_32 * l3_up
```

Canvas read: W_read(3d→d) — all three levels concatenated. Gate from query h.

**Canvas total params (v15):** ~809K (same as v14, plus joint-gate DWConv4×DWConv_k3 ≈ 7K) = **~816K**.

### 6.5 LDRM: Local DPT Patch Refinement Module [NEW in v15]

This is the central new contribution of v15. After the main 5-layer decoder produces $h^{(\text{final})}$, each query at pixel $(u, v)$ needs sub-stride-8 local structure — which canvas L2 (stride 8) cannot provide. The LDRM gives each query access to L1 (stride 4) features in a 28×28 pixel neighborhood, upsampled DPT-style to stride 1.

This closes the local→global→local loop:
- **Local (seed):** Each query starts from multi-scale center features (B1)
- **Global (decoder):** 5 layers of MSDA + B3b give global scene context
- **Local (LDRM):** Fine-grained structural context at exact pixel level closes the loop

#### 6.5.1 The Problem LDRM Solves

Canvas L2 at stride 8: each cell covers an 8×8 pixel region. A depth edge between two surfaces 3px apart is invisible — both sides of the edge map to the same canvas cell. The query's final representation $h$ is dominated by coarse spatial context (MSDA samples from stride-8 at minimum, canvas writes at stride-8).

For queries exactly at depth boundaries (railing edges, table edges, door frames), the 8px granularity is the primary accuracy bottleneck. DPT solves this for dense prediction by progressive upsampling + Conv3×3 at each stage. LDRM applies the same principle locally per query — at 1/K the cost (we process only the neighborhood of K query points, not the full image).

#### 6.5.2 DPT Parallel

| DPT (dense) | LDRM (per-query local) |
|---|---|
| Reassemble to stride 4, apply ResidualConvUnit | L1 already at stride 4 — extract 7×7 local patch |
| Bilinear ×2 → stride 2, Conv(256→128, k3) + residual | Bilinear ×2 → [B×K, 192, 14, 14], Conv(192→96, k3) + GELU + residual |
| Bilinear ×2 → stride 1, Conv(128→64, k3) + residual | Bilinear ×2 → [B×K, 96, 28, 28], Conv(96→32, k3) + GELU |
| Predict at every pixel from stride-1 features | Sample at exact query sub-pixel offset from stride-1 local features |
| Combine with ViT global tokens | Combine local_feat [K,32] with global h [K,192] |

DPT is global (all pixels). LDRM is local (28×28 per query). LDRM cost: $O(K)$ vs DPT cost $O(H \times W)$ — orders of magnitude cheaper, targeted exactly at the K positions that matter.

#### 6.5.3 Step-by-Step Architecture

**Input:** $h^{(\text{final})} \in \mathbb{R}^{B \times K \times d}$ (after B3b), L1 feature map $[B, 192, H/4, W/4]$, query coords $[B, K, 2]$.

**Step 1: Extract 7×7 patch from L1 (stride 4)**

For query at pixel $(u, v)$, create a 7×7 sampling grid at stride-4 spacing (7 L1 cells, each 4px apart, covering a 28×28 pixel region):

$$\text{grid}_{q}[i, j] = \left(\frac{2(u + (i-3) \cdot 4)}{W-1} - 1,\;\; \frac{2(v + (j-3) \cdot 4)}{H-1} - 1\right), \quad i,j \in \{0,\ldots,6\}$$

Create full grid: $[B, K \times 7, 7, 2]$, call `F.grid_sample(L1, grid)` → $[B, 192, K \times 7, 7]$, reshape → $[B \times K, 192, 7, 7]$.

This is a single `F.grid_sample` call for all K queries simultaneously. Efficient: $K \times 49 = 12{,}544$ bilinear samples at 416×544.

**Step 2: DPT-style progressive upsampling**

```
patches [B×K, 192, 7, 7]
  → bilinear ×2 → [B×K, 192, 14, 14]
  → Conv2d(192, 96, k=3, p=1) + GELU + residual(Conv2d(192,96,k=1))
  → feat_14 [B×K, 96, 14, 14]

feat_14 [B×K, 96, 14, 14]
  → bilinear ×2 → [B×K, 96, 28, 28]
  → Conv2d(96, 32, k=3, p=1) + GELU
  → feat_28 [B×K, 32, 28, 28]   ← stride-1 local features in 28×28 pixel window
```

The Conv2d after each bilinear upsample learns to synthesize plausible sub-stride-4 structure from the interpolated features. This is the DPT-style "refine the upsampled estimate with local spatial convolution." The conv cannot create information below stride-4, but it CAN learn: "when the two neighboring L1 cells have depth A and B, the boundary between them is most likely at this sub-cell position" — a learned depth boundary model at fine scale.

**Step 3 (inference): Sample at exact query sub-pixel position**

Compute the query's pixel offset within its 28×28 local patch (the patch covers pixels $u \pm 12$, $v \pm 12$ approximately):

$$\text{offset\_norm} = \left(\frac{2 \cdot ((u \bmod 4) + 12)}{27} - 1,\;\; \frac{2 \cdot ((v \bmod 4) + 12)}{27} - 1\right)$$

`F.grid_sample(feat_28, offset_norm)` → $[B \times K, 32, 1, 1]$ → squeeze → $\text{local\_feat} \in \mathbb{R}^{B \times K \times 32}$.

This gives the query a stride-1 feature at its exact pixel position — genuine sub-8px (and sub-4px) structural awareness. The bilinear interpolation within the 28×28 map provides sub-cell precision.

**LDRM params (inference):**

| Component | Params |
|-----------|-------:|
| Conv(192→96, k3) | 192×96×9 + 96 ≈ **166K** |
| Conv(96→96, k1) residual branch | 96×96 ≈ **9K** |
| Conv(96→32, k3) | 96×32×9 + 32 ≈ **28K** |
| **LDRM total** | **~203K** |

LDRM memory: [B×K, 192, 14, 14] at peak. B=6, K=256: 6×256×192×14×14 × 2 bytes (bfloat16) ≈ 144MB peak. If VRAM is tight: reduce to K=128 → 72MB, or reduce intermediate channels 96→64.

#### 6.5.4 Dense Training from LDRM (training only)

In training, `feat_28 [B×K, 32, 28, 28]` covers a 28×28 pixel neighborhood per query at stride 1. This enables dense depth supervision without any extra architecture:

```
# Broadcast global context h across spatial dims (training only)
h_spatial = h_final.reshape(B*K, d, 1, 1).expand(B*K, d, 28, 28)  # [B*K, 192, 28, 28]

# Concat global + local: [B*K, 224, 28, 28]
combined = torch.cat([h_spatial, feat_28], dim=1)

# Dense depth prediction: training-only Conv1×1
dense_depth = torch.exp(dense_head(combined))  # [B*K, 1, 28, 28]
```

`dense_head = Conv2d(224, 1, kernel_size=1)` — 225 params (training-only, discarded at inference).

**GT construction:** For each query at $(u, v)$, extract 28×28 GT depth patch centered at $(u, v)$ from the full-resolution GT depth map.

**Dense supervision coverage:**
- K=256 queries × 28×28 = 2,007,040 total patch positions (with overlap)
- Unique coverage (assuming uniform query distribution): approximately 128K independent pixels ≈ **57% of 226,304 image pixels**
- Even with heavy patch overlap: far exceeds canvas L2 auxiliary head coverage (6.25% at stride 4)

**Why h-broadcast is the right fusion:**
$h^{(\text{final})}$ encodes global scene context (depth scale, room structure, object category). `feat_28` encodes local fine structure (which side of boundary, surface tilt, edge direction). The dense prediction requires both — global scale from $h$, local spatial variation from `feat_28`. Broadcasting $h$ across 28×28 positions and fusing with Conv1×1 implements this naturally: the conv learns "given global depth context $h$ and local feature $f$, predict depth at this exact position."

This is structurally identical to how DPT combines ViT global tokens with fine-scale reassembled features via concatenation and projection.

### 6.6 Depth Head (v15: Global + Local fusion)

**Input:** $h^{(\text{final})} \in \mathbb{R}^{B \times K \times 192}$ and $\text{local\_feat} \in \mathbb{R}^{B \times K \times 32}$.

```
h_combined = concat([h_final, local_feat], dim=-1)  ∈ R^{224}
raw = W_r2 · GELU(W_r1 · h_combined + b_r1) + b_r2   (224 → 384 → 1)
depth_hat = exp(raw · exp(s))
```

Learnable scale $s$ initialized to 0 (same as v14). Bias $b_{r2}$ initialized to $\ln(2.5)$.

**Depth head params:** Linear(224→384) ~86K + Linear(384→1) ~384 + scale ~1 = **~87K** (vs ~74K in v14; +13K from wider input).

---

## 7. Training

### 7.1 Dataset (unchanged from v14)

NYU Depth V2, 416×544 (BTS augmentation: random crop from 480×640, random rotation ±2.5°, horizontal flip, BTS color augmentation).

### 7.2 Loss Functions (v15)

$$\mathcal{L} = L_{\text{silog}}(\hat{d}_q, d_q^*) + \lambda_{\text{canvas}} \cdot L_{\text{dense\_silog}}(\hat{D}_{\text{canvas}}, D^*_{\downarrow 4}) + \lambda_{\text{local}} \cdot L_{\text{dense\_silog}}(\hat{D}_{\text{local}}, D^*_{\text{patch}})$$

**Primary — Sparse SILog on K=256 query predictions:**
$$L_{\text{silog}} = \sqrt{\frac{1}{K}\sum_q \delta_q^2 - 0.50 \left(\frac{1}{K}\sum_q \delta_q\right)^2}, \quad \delta_q = \log\hat{d}_q - \log d_q^*$$

**Canvas auxiliary (training only, inherited from v14):**

Canvas L2 dense head (v14 design, kept in v15): upsampled canvas L2 → `canvas_head_L2: Conv(192→64,k3)+GELU+Conv(64→1,k1)` applied after ×2 upsample → stride-4 predictions → `L_dense_silog` vs GT at stride 4. $\lambda_{\text{canvas}} = 0.5$.

**LDRM local patch dense training (training only, new in v15):**

$\hat{D}_{\text{local}} \in \mathbb{R}^{B \times K \times 1 \times 28 \times 28}$: dense depth map for each query's local 28×28 patch.

$D^*_{\text{patch}}$: for each query at $(u, v)$, extract 28×28 crop from the full-resolution GT depth map.

$\lambda_{\text{local}} = 0.3$ (scale relative to primary; the dense terms provide many more gradient terms — K×784 vs K). **Important:** both $h$-broadcast and local_feat contribute gradient, but the primary depth head (on a single local position) is the inference-critical path.

**Training-only params:** `dense_head` Conv2d(224→1) = 225 params + `canvas_head_L2` ~111K = **~111K training-only params** (not present at inference).

### 7.3 Training Setup

| Setting | v14 | v15 |
|---------|-----|-----|
| Optimizer | AdamW | AdamW |
| LR (decoder + neck) | 1×10⁻⁴ | 1×10⁻⁴ |
| LR (encoder) | 1×10⁻⁵ | 1×10⁻⁵ |
| Batch size | 6 | 4–6 (watch VRAM with LDRM) |
| K queries per image | 256 | 256 (ablate 128 later) |
| Epochs | 25 | 25 |
| Gradient clip | 1.0 | 1.0 |
| Attention dropout | 0.1 | 0.0 (no attention-based self-attn in v15) |
| Resolution | 416×544 | 416×544 |

**VRAM concern:** LDRM adds peak tensor [B×K, 192, 14, 14] ≈ 144MB at B=6, K=256. If OOM, reduce to K=128 during training (evaluation can use K=256 by running in inference mode without LDRM dense head).

---

## 8. Evaluation (unchanged from v14)

Metrics: AbsRel, RMSE, $\delta < 1.25^n$ at K query points on NYU Eigen test split. All evaluations at depth range [1e-3, 10]m.

---

## 9. Parameter Count Summary

| Component | v14 | v15 | Delta |
|-----------|----:|----:|------:|
| Encoder (ConvNeXt V2-T) | 28,600K | 28,600K | 0 |
| Neck projections | 278K | 56K | −222K |
| L4 self-attention | 886K | **0** | −886K |
| Pre-compute (proj_L3/L4 + B3b KV) | 148K | 812K | +664K |
| Seed constructor | 160K | 160K | 0 |
| MSDA (3 layers, incl. canvas) | 2,055K | 2,274K | +219K |
| B3b (2 layers, excl. pre-compute KV) | 1,334K | 964K | −370K |
| Triple canvas (shared with MSDA/B3b) | 816K | 816K | 0 |
| LDRM | 0 | **203K** | +203K |
| Depth head | 74K | 87K | +13K |
| **Total inference** | **~34,351K** | **~33,972K** | **−379K** |
| Training-only (canvas head + dense head) | ~111K | ~111K | ≈0 |

Net: v15 is slightly smaller than v14 (~380K fewer params) while having richer L3/L4 representations and per-query local refinement.

---

## 10. Open Design Questions

### 10.1 K=128 vs K=256

With LDRM providing rich local context per query, each query prediction is more accurate. Fewer queries may suffice at training time while maintaining evaluation quality. Experiment: train v15 with K=128 (halves LDRM VRAM), evaluate with K=256 (model is K-agnostic).

### 10.2 LDRM Patch Size: 7×7 vs 5×5 vs 9×9

7×7 covers 28×28 pixels (±12px from query center). At stride 4, 7 cells × 4px = 28px. Larger patches (9×9 = 36px coverage) increase context but also increase [B×K, d, 9×2, 9×2] intermediate tensors and overlap between nearby queries. Start with 7×7 as the principled midpoint.

### 10.3 L3 Native Dimension: 384 vs 256

384 preserves all ConvNeXt V2 Stage 3 information. 256 provides 33% compression (less lossy than 192, still saves MSDA V proj params). Start with 384; ablate 256 if VRAM is constrained.

### 10.4 Canvas L2 Dense Head vs LDRM Dense Training

Both supervise depth at fine spatial resolution. Canvas L2 head: coarser (stride 4 via ×2 up), covers more area uniformly. LDRM: stride 1, but only covers query neighborhoods. They are COMPLEMENTARY:
- Canvas head supervises the canvas features to encode depth everywhere (including non-query positions)
- LDRM supervises local fine-grained predictions at query positions with sub-pixel precision

Keep both in v15 training.

### 10.5 Ablation Priority

| Ablation | Expected insight |
|----------|-----------------|
| v15 vs v14 baseline | Net effect of all v15 changes together |
| v15 − LDRM | Is local fine-grained refinement actually needed? |
| v15 − L3/L4 high dim | Does keeping native dims actually help? |
| v15 + Q2Q (add back) | Confirms Q2Q removal was correct |
| v15 K=128 vs K=256 | Diminishing returns from more queries given LDRM |

Build these as checkpoints from the same v15 training run if possible — saves training time.

---

## 11. Theoretical Justification

### 11.1 Local→Global→Local Design

The v15 architecture implements a principled three-phase information flow:

1. **Local (B1 Seed):** Each query initializes from multi-scale center features — coarse-to-fine local context at the query position.
2. **Global (MSDA + B3b Decoder):** 5 decoder layers give each query deformable access to multi-scale features anywhere in the image, plus full attention to all L4 tokens. The query learns global scene structure, depth scale, room layout.
3. **Local (LDRM):** After global context is established, the query examines its fine-grained local neighborhood. The LDRM provides depth boundary information, surface tilt, and sub-pixel positional accuracy that the global decoder cannot provide.

This local→global→local pattern appears in many successful architectures:
- **U-Net:** Encoder (local→global downsampling) + Decoder (global→local upsampling)
- **DPT:** ViT encoder (global) + multi-resolution reassemble (local) + progressive upsampling (local→stride 1)
- **PointNet++:** Hierarchical ball queries (local→global) + interpolation (global→local)

v15 is the first sparse depth architecture to explicitly close this loop per-query.

### 11.2 Spatial Inductive Bias

| Mechanism | Spatial structure | Appropriate for |
|---|---|---|
| MSDA | Deformable — learned, any position | Multi-scale feature gathering |
| B3b | Global — all L4 positions | Scene-level semantics |
| Triple Canvas | Dense local grid — proximity-based | Surface continuity, boundary propagation |
| ~~Q2Q~~ (removed) | Dense global — all queries, no position bias | Not appropriate for spatially-structured tasks |
| LDRM | Dense local — 28×28 per query | Fine boundary structure, sub-pixel positioning |

The principle: depth is a spatially-structured prediction. Mechanisms with spatial structure (canvas, LDRM) outperform mechanisms without it (Q2Q) for this task.
