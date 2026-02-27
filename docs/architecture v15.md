# Research Plan: SPD v15 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-26
Version: v15 (DPT-style FPN global fusion + per-query local L1 patch; removes MSDA / B3b / canvas / Q2Q / neck)

> **v15 core insight:** v14's failure is a local-information problem. Every mechanism in v14's decoder — MSDA, canvas write-smooth-read, B3b — operates at stride ≥ 8. Queries have no sub-stride-8 precision, yet they predict depth at exact pixel positions. The canvas additionally has a structural flaw: queries must scatter-write to a dense map they haven't learned to fill, before reading from it — a backwards causal structure. DPT's insight is different: build the dense feature map bottom-up from encoder features (correct causal direction), then queries sample from it.
>
> **v15 applies this in two complementary parts:**
> 1. **Global (DPT-style FPN):** Progressive FPN fusion of L4→L3→L2 produces a stride-8 dense feature map at D=64 channels, using full DPT **FeatureFusionBlock** (ResidualConvUnit) blocks — identical to DAv2-S. D=64 matches DAv2-Small (ViT-S scale ≈ ConvNeXt V2-T scale). Fusion MACs scale as D²: D=256 costs 16× more than D=64 — unjustified at our backbone scale.
> 2. **Local (valid-conv inference / dense-padded training, shared weights):** Same conv layers, two execution modes. **Inference:** per query, sample 9×9 from stride_8_map and 9×9 from L1; apply a cascade of valid Conv3×3 with bilinear-upsample stages; output is a single 1×1 feature vector (local_h[32]) at the query position — minimum samples, no wasted 32×32 computation. **Training:** identical conv layers run with padded convolutions on the full stride_8_map and L1 feature maps (no per-query extraction), producing a dense H×W feature map → dense depth prediction → dense SILog loss (100% pixel coverage). At inference, local_h is sampled from the center of the valid-conv output; at training, local_h is read from the dense feature map at K query positions.
> 3. **Dense training, no queries:** Training runs purely dense — no K-query sampling, no MLP depth head. Identical to DPT training: padded convs on full feature maps → H×W depth prediction → SILog over all valid pixels. The Conv1×1(32→1) dense head is the **same** layer used at inference (applied to the 1×1 center of each valid-conv output). No separate query-based training path.
>
> **Result:** 6 modules removed (neck, L4-SA, seed constructor, canvas, MSDA, B3b). 2 modules added (DPT FPN, local patch path). ~4.1M fewer parameters. Architecturally simpler and better-motivated.

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Metric depth $\hat{d}_i$ at each queried pixel.

**Constraint:** No dense $H \times W$ decoding. Encoder runs once; decoder is $O(K)$ per query.

**Baseline:** DAv2-S (ViT-S + DPT dense decoder, ~49G MACs). v15 targets competitive AbsRel with far fewer decoder MACs at inference.

---

## 2. Key Changes v14 → v15

| Component | v14 | v15 | Rationale |
|---|---|---|---|
| Projection neck | Conv1×1+LN per level → 192 (~278K) | **Removed** | DPT fusion handles projections internally |
| L4 self-attention | 2× FullSelfAttn₁₉₂ (~886K) | **Removed** | ConvNeXt V2 proven without SA; no global reasoning needed before queries sample |
| Multi-scale seed (B1) | 4-level center concat → h (~160K) | **Removed** | Replaced by direct grid_sample from stride-8 map |
| Triple Spatial Canvas | Write-smooth-read, 3 scales (~816K) | **Removed** | DPT builds dense map bottom-up (correct causal structure) |
| MSDA 3L cross-attn | Deformable sampling from L1–L4 (~2,055K) | **Removed** | DPT fusion aggregates multi-scale globally; local patch handles fine detail |
| B3b 2L GlobalCrossAttn | Full attn to all 221 L4 tokens (~1,334K) | **Removed** | L4 is already in DPT fusion |
| Q2Q self-attn | All 5 decoder layers (~750K) | **Removed** | Wrong inductive bias for depth; no K² cost to worry about in v15 |
| **DPT-style FPN fusion** | — | **NEW: L4→L3→L2 → stride-8 map @ D=64 (~317K), full FeatureFusionBlock (RCU)** | D=64 = DAv2-S scale; full RCU = proper DPT blocks; D² scaling makes D=256 prohibitive |
| **Local patch path** | — | **NEW: 9×9 s8 → 2× valid Conv3×3 → upsample → merge with 9×9 L1 → 4× valid Conv3×3 + 2 upsample stages → 1×1 local_h[32] (~246K)** | Valid-conv inference: minimum 9×9 samples, no 32×32 map. Padded-conv training: full H×W dense. Same weights. |
| Depth head | MLP(192→384→1) (~74K) | **Conv1×1(32→1) (~0K)** — same as dense training head; no MLP, no global_h | Same weights used in training (dense) and inference (per-query center) |
| Dense training | Canvas L2 aux head (stride 4, 6.25%) | **Full-image padded convs (same local path weights) → dense H×W prediction → dense SILog** | 100% pixel coverage; no grid patches; Conv1×1(32→1) training-only dense head (~0K) |
| **Total inference params** | **~34,351K** | **~29,001K** | **−5,350K** |

---

## 3. Architecture Overview

```
RGB Image [H × W × 3]   (416 × 544 at training)
  │
  ▼
ConvNeXt V2-T  (FCMAE pre-trained, fine-tuned 0.1× LR)
  L1: [H/4  × W/4  ×  96]   stride 4   ← used ONLY in local patch path
  L2: [H/8  × W/8  × 192]   stride 8
  L3: [H/16 × W/16 × 384]   stride 16
  L4: [H/32 × W/32 × 768]   stride 32
  │
  ▼ ─────────────────────────────────────────────────────────────
  GLOBAL PATH (dense, once per image)
  │
  ▼
DPT-Style FPN Fusion  (2× FeatureFusionBlock/RCU, D=64 — identical to DAv2-S)
  f4 = LN(Conv1×1(768→64)(L4))   [B, 64, H/32, W/32]
  f3 = LN(Conv1×1(384→64)(L3))   [B, 64, H/16, W/16]
  f2 = LN(Conv1×1(192→64)(L2))   [B, 64, H/8,  W/8 ]
  │
  refinenet4: RCU2(f4) → ×2 → Conv1×1                    [B, 64, H/16, W/16]
  refinenet3: RCU1(f3) + add + RCU2 → ×2 → Conv1×1 = stride_8_map [B, 64, H/8, W/8]
  │
  ▼ ─────────────────────────────────────────────────────────────
  LOCAL PATH  (training: dense O(H×W);  inference: per-query O(K))
  │
  TRAINING — padded Conv3×3 on full feature maps (identical to DPT):
  │  stride_8_map [B,64,H/8,W/8] — conv_s8_1,s8_2 (padded) — bilinear ×2
  │  L1 [B,96,H/4,W/4] — proj_l1 — add — conv_m1,m2 (padded) — bilinear ×2
  │  — conv_u1,u2 (padded) — bilinear ×2 — conv_f (padded) → [B,32,H,W]
  │  — dense_head: Conv1×1(32→1) → [B,1,H,W] → dense SILog (all valid pixels)
  │
  INFERENCE — valid Conv3×3 on 9×9 patches per query:
  │  s8 branch: 9×9 from stride_8_map      → [B*K, 64, 9, 9]
  │    valid Conv3×3 (conv_s8_1)           → [B*K, 64, 7, 7]
  │    valid Conv3×3 (conv_s8_2)           → [B*K, 64, 5, 5]
  │    bilinear upsample → 9×9             → [B*K, 64, 9, 9]
  │  L1 branch: 9×9 from L1               → [B*K, 96, 9, 9]
  │    Conv1×1(96→64) (proj_l1)            → [B*K, 64, 9, 9]
  │  Merge (add)                           → [B*K, 64, 9, 9]
  │    valid Conv3×3 (conv_m1)             → [B*K, 64, 7, 7]
  │    valid Conv3×3 (conv_m2)             → [B*K, 64, 5, 5]
  │    bilinear upsample → 7×7             → [B*K, 64, 7, 7]
  │    valid Conv3×3 (conv_u1)             → [B*K, 64, 5, 5]
  │    valid Conv3×3 (conv_u2)             → [B*K, 64, 3, 3]
  │    bilinear upsample → 5×5             → [B*K, 64, 5, 5]
  │    valid Conv3×3 (conv_f, 64→32)       → [B*K, 32, 3, 3]
  │    center [1, 1]                       → [B*K, 32]
  │  depth = exp(dense_head(center))        → [B, K]   (same Conv1×1 weights)
```

---

## 4. Encoder: ConvNeXt V2-T (unchanged from v14)

| Parameter | Value |
|---|---|
| Architecture | ConvNeXt V2-Tiny |
| Stages | 4 (depths [3,3,9,3], channels [96,192,384,768]) |
| Strides | [4, 8, 16, 32] |
| Input | 416×544 (BTS augmentation: random crop from 480×640) |
| Pretraining | FCMAE — masked autoencoding, strong dense-task transfer |
| Params | ~28.6M |

**Stage outputs at 416×544:**
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

**No projection neck in v15.** DPT fusion projects each scale internally. L1 is passed directly (native 96ch) to the local patch path — no neck projection needed.

**On backbone choice:** UniDepth V1 (CVPR 2024) demonstrates ConvNeXt-L [96,192,384,768] + DPT-style decoder for metric depth — confirming this combination is architecturally sound. The top-performing monocular depth models (Depth Anything V2, challenge winners 2024-2025) use ViT/DINOv2 backbones. ConvNeXt V2 FCMAE pretraining is well-suited for dense tasks; however, a ViT-S backbone ablation (same DPT FPN + local patch head, different backbone) would cleanly measure the backbone contribution. For v15, ConvNeXt V2-T is kept to isolate the decoder architecture change.

---

## 5. Global DPT-Style FPN Fusion

### 5.1 Design Rationale

DPT (Ranftl et al., ICCV 2021) established the standard for dense feature fusion from hierarchical backbones: project each scale to a uniform dimension D via 1×1 conv, then progressively fuse coarse-to-fine via **FeatureFusionBlock** — each block applies ResidualConvUnit (RCU = 2× Conv3×3 with residual skip), bilinear ×2 upsample, and Conv1×1. This is what DPT calls "Reassembly" (projection) + "Fusion" (RefineNet-based). **v15 uses the full FeatureFusionBlock — identical to DAv2.**

**Why D=64:** Depth Anything V2-Small (ViT-S backbone, ConvNeXt V2-T equivalent scale) uses D=64. DPT projects to D matching backbone scale: ViT-S → 64, ViT-B → 128, ViT-L → 256. Fusion MACs scale as D²: going D=64 → 256 multiplies RCU cost by 16×. At 8GB VRAM, D=64 is the correct choice for our backbone scale (see Section 11.4 for D comparison).

**Why stops at stride 8 (not stride 4 → 1):**

At 416×544, the position count per level:

| Level | Resolution | Positions | Relative cost |
|---|---|---:|---:|
| L4 (stride 32) | 13×17 | 221 | 1× |
| L3 (stride 16) | 26×34 | 884 | 4× |
| L2 (stride 8) | 52×68 | 3,536 | 16× |
| L1 (stride 4) — **global** | 104×136 | 14,144 | **64×** |
| Per-query local L1, K=221 | 221×8×8 | 14,144 | ~64× |

Adding L1 globally costs 64× the compute of L4, and covers 14,144 positions — of which only K=221 are query positions. Per-query local patches (K=221, 8×8 each) total exactly 14,144 L1 samples — matching full L1 coverage precisely, but only at query-relevant positions. Stopping the global path at stride 8 and using local-only L1 per query gives **100% of the fine-resolution coverage** with zero wasted computation.

### 5.2 Reassembly (Projection to D=64)

```python
# Input: L2 [B,192,52,68], L3 [B,384,26,34], L4 [B,768,13,17]
f4 = F.layer_norm(self.proj_L4(L4), [64])  # Conv1×1(768→64) + LN → [B,64,13,17]
f3 = F.layer_norm(self.proj_L3(L3), [64])  # Conv1×1(384→64) + LN → [B,64,26,34]
f2 = F.layer_norm(self.proj_L2(L2), [64])  # Conv1×1(192→64) + LN → [B,64,52,68]
```

**Why D=64 (not D=128 or D=192):**
- DAv2-S (ViT-S, same parameter scale as ConvNeXt V2-T) uses D=64
- RCU MACs scale as D²×N: D=128 costs 4× more, D=256 costs 16× more — for the same architecture
- L1 (stride-4) global fusion is skipped in v15; fine detail comes from per-query local patches instead
- For depth regression (1D output per query), D=64 provides ample global context capacity

### 5.3 Fusion (Full DPT FeatureFusionBlock with RCU)

Each `FeatureFusionBlock` contains two `ResidualConvUnit` (RCU) blocks. An RCU is:
```
RCU(x) = x + conv2(GELU(conv1(GELU(x))))   # conv1, conv2 = Conv3×3(D→D)
```

**refinenet4** (processes f4 alone — topmost level, no skip):
```python
# RCU2 on f4 (no secondary input at topmost level)
r4 = f4 + self.rn4_conv2(F.gelu(self.rn4_conv1(F.gelu(f4))))  # RCU2: [B,64,13,17]
r4 = F.interpolate(r4, size=f3.shape[2:], mode='bilinear', align_corners=True)
r4 = self.rn4_out(r4)            # Conv1×1(64→64) → [B, 64, 26, 34]
```

**refinenet3** (fuses r4 primary + f3 secondary → stride_8_map):
```python
# RCU1 on secondary input f3, then add primary
skip3 = f3 + self.rn3_conv2(F.gelu(self.rn3_conv1(F.gelu(f3))))  # RCU1(f3): [B,64,26,34]
r3 = r4 + skip3                                                    # add
# RCU2 on fused result
r3 = r3 + self.rn3_conv4(F.gelu(self.rn3_conv3(F.gelu(r3))))     # RCU2: [B,64,26,34]
r3 = F.interpolate(r3, size=f2.shape[2:], mode='bilinear', align_corners=True)
stride_8_map = self.rn3_out(r3)  # Conv1×1(64→64) → [B, 64, 52, 68]
```

`stride_8_map [B, 64, H/8, W/8]` — multi-scale context from L2+L3+L4 at D=64. The 6× Conv3×3 through refinenet4+refinenet3 give substantial RF (~224px), equivalent to DPT's `path_3`. Queries sample from it globally (Section 6.1) and locally (Section 6.2). At D=64, no `proj_s8` projection needed.

### 5.4 DPT Fusion Parameters

| Component | Params |
|---|---:|
| Conv1×1(768→64) + LN | 768×64 + 64×3 ≈ **49K** |
| Conv1×1(384→64) + LN | 384×64 + 64×3 ≈ **25K** |
| Conv1×1(192→64) + LN | 192×64 + 64×3 ≈ **13K** |
| refinenet4: RCU2 (2× Conv3×3(64→64)) + Conv1×1 | 2×(64²×9+64) + 64²+64 ≈ **78K** |
| refinenet3: RCU1+RCU2 (4× Conv3×3(64→64)) + Conv1×1 | 4×(64²×9+64) + 64²+64 ≈ **152K** |
| **DPT total (inference)** | **~317K** |

---

## 6. Per-Query Prediction

**No global_h.** The s8 branch of the local path already samples a 9×9 neighborhood from `stride_8_map`, injecting global depth context. A separate single-pixel global_h lookup is unnecessary and untrained (training is purely dense — no per-query path). All learnable parameters in the decoder are convolution or projection layers; no MLP, no attention.

**Inference is a direct application of the dense-trained Conv weights at query positions using valid convolutions.** The Conv1×1(32→1) depth head is the same layer that produces the H×W dense output during training.

### 6.1 Local Patch Sampling (Inference)

**RF justification:** One pixel in `stride_8_map` (at stride 8) has a receptive field of 9×9 in L2@stride-8 space, accumulated from 6× Conv3×3 through refinenet4+refinenet3. To match this coverage in L1@stride-4 space, we need 18×18 L1 cells. At inference, however, we only need the feature at the single query center pixel — and with valid convolutions, a 9×9 input suffices to produce a 1×1 output after 4× valid Conv3×3 (9→7→5→3→1). The local path samples only what is needed.

**Two grid_sample calls per query group (inference):**

```python
# Inputs:
# L1: [B, 96, H4, W4]           (H4=104, W4=136 at 416×544)
# stride_8_map: [B, 64, H8, W8]  (H8=52, W8=68)

B, K, _ = coords.shape
H4, W4 = L1.shape[2], L1.shape[3]
H8, W8 = stride_8_map.shape[2], stride_8_map.shape[3]

N = 9  # patch size for both s8 and L1

# Half-offsets: [-4,...,4] for 9-point grid (centered at query)
half = N // 2   # = 4
dx = torch.arange(-half, half + 1, device=L1.device, dtype=torch.float32)  # 9 values
dy = torch.arange(-half, half + 1, device=L1.device, dtype=torch.float32)
gy, gx = torch.meshgrid(dy, dx, indexing='ij')  # [9, 9]

# --- s8 patch (stride-8 space) ---
cx_s8 = coords[..., 0] / 8.0   # [B, K]
cy_s8 = coords[..., 1] / 8.0
sx_s8 = cx_s8[..., None, None] + gx   # [B, K, 9, 9]
sy_s8 = cy_s8[..., None, None] + gy
grid_s8 = torch.stack([
    2.0 * sx_s8 / (W8 - 1) - 1.0,
    2.0 * sy_s8 / (H8 - 1) - 1.0
], dim=-1).reshape(B, K * N, N, 2)
raw_s8 = F.grid_sample(stride_8_map, grid_s8, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
# raw_s8: [B, 64, K*9, 9] → [B*K, 64, 9, 9]
s8_patches = raw_s8.reshape(B, 64, K, N, N).permute(0,2,1,3,4).reshape(B*K, 64, N, N)

# --- L1 patch (stride-4 space) ---
cx_l1 = coords[..., 0] / 4.0
cy_l1 = coords[..., 1] / 4.0
sx_l1 = cx_l1[..., None, None] + gx   # [B, K, 9, 9]
sy_l1 = cy_l1[..., None, None] + gy
grid_l1 = torch.stack([
    2.0 * sx_l1 / (W4 - 1) - 1.0,
    2.0 * sy_l1 / (H4 - 1) - 1.0
], dim=-1).reshape(B, K * N, N, 2)
raw_l1 = F.grid_sample(L1, grid_l1, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
# raw_l1: [B, 96, K*9, 9] → [B*K, 96, 9, 9]
L1_patches = raw_l1.reshape(B, 96, K, N, N).permute(0,2,1,3,4).reshape(B*K, 96, N, N)
```

**Memory (B=2, K=128):** `s8_patches` [256, 64, 9, 9] ≈ 5.3MB; `L1_patches` [256, 96, 9, 9] ≈ 8.0MB. Minimal.

### 6.3 Local Fusion — Valid Conv Cascade (Inference) / Padded Conv Dense (Training)

The local path uses **identical conv weights** in two execution modes:

| Mode | Padding | Input | Output |
|---|---|---|---|
| Inference | **valid** (no pad) | 9×9 patches per query | 3×3 per query → take center |
| Training | **same** (pad=1) | Full feature maps [B, C, H/s, W/s] | Dense [B, 32, H, W] |

**Inference — step by step:**

```python
# s8 branch: refine stride_8_map context
x_s8 = F.gelu(self.conv_s8_1(s8_patches))  # valid Conv3×3(64→64): [B*K, 64, 9,9] → [B*K, 64, 7,7]
x_s8 = F.gelu(self.conv_s8_2(x_s8))        # valid Conv3×3(64→64): [B*K, 64, 7,7] → [B*K, 64, 5,5]
x_s8 = F.interpolate(x_s8, size=(9, 9), mode='bilinear', align_corners=True)
# x_s8: [B*K, 64, 9, 9]  ← upsampled back to L1 patch size

# L1 branch: project and merge
x_l1 = self.proj_l1(L1_patches)            # Conv1×1(96→64): [B*K, 96, 9,9] → [B*K, 64, 9,9]
x = F.gelu(x_s8 + x_l1)                    # add: [B*K, 64, 9, 9]

# Post-merge valid conv block 1
x = F.gelu(self.conv_m1(x))    # valid Conv3×3(64→64): 9→7
x = F.gelu(self.conv_m2(x))    # valid Conv3×3(64→64): 7→5
x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=True)

# Post-merge valid conv block 2
x = F.gelu(self.conv_u1(x))    # valid Conv3×3(64→64): 7→5
x = F.gelu(self.conv_u2(x))    # valid Conv3×3(64→64): 5→3
x = F.interpolate(x, size=(5, 5), mode='bilinear', align_corners=True)

# Final: reduce channels, valid conv → 3×3, take center
x = F.gelu(self.conv_f(x))     # valid Conv3×3(64→32): 5→3
local_h = x[:, :, 1, 1]        # center of 3×3 → [B*K, 32]
local_h = local_h.reshape(B, K, 32)
```

**Spatial trace (inference, valid convolutions):**

| Step | Size | Notes |
|---|---|---|
| s8 sample | 9×9 | from stride_8_map |
| conv_s8_1 | 7×7 | valid |
| conv_s8_2 | 5×5 | valid |
| upsample | 9×9 | bilinear to match L1 |
| L1 sample + proj | 9×9 | merge (add) |
| conv_m1 | 7×7 | valid |
| conv_m2 | 5×5 | valid |
| upsample | 7×7 | bilinear |
| conv_u1 | 5×5 | valid |
| conv_u2 | 3×3 | valid |
| upsample | 5×5 | bilinear |
| conv_f (64→32) | 3×3 | valid |
| **center [1,1]** | **1×1** | **= local_h [32-dim]** |

**Training — same weights, padded convolutions, full feature maps:**

```python
# --- s8 branch (on full stride_8_map [B, 64, H/8, W/8]) ---
x_s8 = F.gelu(self.conv_s8_1(stride_8_map))   # padded Conv3×3 → [B, 64, H/8, W/8]
x_s8 = F.gelu(self.conv_s8_2(x_s8))           # padded Conv3×3 → [B, 64, H/8, W/8]
x_s8 = F.interpolate(x_s8, scale_factor=2, mode='bilinear', align_corners=True)
# x_s8: [B, 64, H/4, W/4]

# --- L1 branch (on full L1 [B, 96, H/4, W/4]) ---
x_l1 = self.proj_l1(L1)                       # Conv1×1(96→64) → [B, 64, H/4, W/4]
x = F.gelu(x_s8 + x_l1)                       # merge: [B, 64, H/4, W/4]

# --- post-merge dense ---
x = F.gelu(self.conv_m1(x))                   # padded Conv3×3 → [B, 64, H/4, W/4]
x = F.gelu(self.conv_m2(x))                   # padded Conv3×3 → [B, 64, H/4, W/4]
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
# x: [B, 64, H/2, W/2]

x = F.gelu(self.conv_u1(x))                   # padded Conv3×3 → [B, 64, H/2, W/2]
x = F.gelu(self.conv_u2(x))                   # padded Conv3×3 → [B, 64, H/2, W/2]
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
# x: [B, 64, H, W]

dense_feat = F.gelu(self.conv_f(x))           # padded Conv3×3(64→32) → [B, 32, H, W]

# --- dense training head (training only, ~33 params) ---
dense_depth = torch.exp(self.dense_head(dense_feat))  # Conv1×1(32→1) → [B, 1, H, W]

# --- extract local_h at K positions for depth head training ---
# coords: [B, K, 2] pixel positions (x, y)
local_h = dense_feat[:, :, coords_y, coords_x]        # [B, 32, K] → permute → [B, K, 32]
```

The **upsampling chain in training** (×2 × 3 = ×8 total) takes stride_8_map → stride_1, matching the encoder's stride exactly. The same conv weights that compute per-query valid-conv inference produce full-resolution dense features at training time — no separate conv layers, no weight mismatch.

### 6.4 Training vs Inference: Same Weights, Different Execution

| Property | Inference | Training |
|---|---|---|
| Padding mode | **valid** (no pad) | **same** (pad=1) |
| Input to local path | 9×9 patches per query | Full feature maps |
| Output | 1×1 per query → local_h | H×W dense → dense SILog |
| local_h for depth head | From center of 3×3 (valid conv output) | Indexed from dense_feat at K positions |
| Compute | O(K) | O(H×W) |
| VRAM at B=2, K=128 | ~14MB patches | ~dense_feat [2,32,416,544] ≈ 55MB |

**Why valid convolutions give the same result as dense padded at the center pixel:** A Conv3×3 with same-padding at a pixel far from the border computes the same output as a Conv3×3 without padding on a patch centered at that pixel. Valid conv on a P×P patch gives a (P−2)×(P−2) output — exactly the center pixel of the padded result. Repeating for 4 conv layers: 9→7→5→3→1 (with upsample restoring spatial size between valid-conv blocks). The center [1,1] of the final 3×3 is the exact per-pixel value from the dense padded path.

### 6.5 Depth Head (Inference)

The depth head is **the same Conv1×1(32→1) layer** used in dense training — no separate MLP, no global_h concat. At inference, it applies to the 1×1 local feature at the query center:

```python
# local_feat: [B*K, 32] — center [1,1] of conv_f output
depth = torch.exp(self.dense_head(local_feat.unsqueeze(-1).unsqueeze(-1)))
# dense_head = Conv1×1(32→1), same weights as training
# depth: [B, K]
```

Bias of Conv1×1 initialized to ln(2.5) (metric depth prior). No `log_scale` parameter.

**Why Conv1×1 not MLP:** Training is purely dense — no K-query forward pass at training. The only trained depth head is Conv1×1(32→1). At inference, applying the same Conv1×1 to the 1×1 center feature is equivalent to a linear(32→1). This is consistent: conv weights learned from dense prediction generalize to per-query prediction because the valid-conv center pixel matches the padded-conv dense output at that position.

**Depth head params:** Conv1×1(32→1) = 32 + 1 = **33 params ≈ ~0K** (vs ~74K in v14).

---

## 7. Training Strategy

### 7.1 Training: Purely Dense, No K Queries

Training uses **no per-query sampling at all**. The full forward pass at training time:

1. Encoder → L1, L2, L3, L4
2. DPT FPN → stride_8_map [B, 64, H/8, W/8]
3. Local path (padded convs on full maps) → dense_feat [B, 32, H, W]
4. dense_head: Conv1×1(32→1)(dense_feat) → dense_depth [B, 1, H, W]
5. Loss: dense SILog against GT depth map (all valid pixels)

All learnable parameters — DPT projections, RCU Conv3×3, proj_l1, conv_s8_1/2, conv_m1/2, conv_u1/2, conv_f, dense_head — are trained via this single dense pass. No MLP, no Q-sampling, no K² cost.

**At inference:** K=128 random valid-pixel queries. Each query uses the valid-conv path (9×9 patches → center 1×1). The dense_head Conv1×1(32→1) applies to the 32-dim center feature → depth.

### 7.2 Loss Function

Single dense SILog loss:
```python
# Training forward pass (no queries)
x_s8 = self.conv_s8_block(stride_8_map)      # padded convs + bilinear ×2 → [B, 64, H/4, W/4]
x_l1 = self.proj_l1(L1)                       # [B, 64, H/4, W/4]
x = self.merge_block(x_s8 + x_l1)             # padded convs + bilinear ×2 → [B, 64, H/2, W/2]
x = self.upsample_block(x)                    # padded convs + bilinear ×2 → [B, 64, H, W]
dense_feat = self.conv_f(x)                   # padded Conv3×3(64→32) → [B, 32, H, W]
dense_depth = torch.exp(self.dense_head(dense_feat))  # Conv1×1(32→1) → [B, 1, H, W]

loss = l_dense_silog(dense_depth, gt_depth_map, lambda_var=0.5)
# Masks d=0 (invalid GT). All valid pixels contribute.
```

$$\mathcal{L} = \mathcal{L}_{\text{dense SILog}}$$

No secondary loss terms. No K-dependent terms.

### 7.3 Dense Coverage Comparison

| Training supervision | Pixel coverage |
|---|---|
| v14: Canvas L2 aux head (×2 upsample → stride 4) | **6.25%** |
| v15: Dense padded-conv local path → H×W prediction | **100% (all valid pixels)** |
| DPT (DAv2-S): Dense decoder to stride 1 | **100%** |

v15 training is architecturally equivalent to DPT training: same dense padded convs, same H×W supervision. The only difference is the s8 branch (which injects stride_8_map context before L1 fusion) and stopping the global FPN at stride 8 instead of stride 1.

### 7.4 Training Setup

| Setting | v14 | v15 |
|---|---|---|
| K (training) | 256 random valid pixels | **None — purely dense** |
| K (inference / evaluation) | 128–256 | **128** |
| Local path training | Per-query patches | **Dense padded conv (H×W)** |
| Batch size | 4–6 | **4** |
| LR encoder | 1×10⁻⁵ | 1×10⁻⁵ |
| LR decoder (DPT fusion + local path) | 1×10⁻⁴ | **1×10⁻⁴** |
| Epochs | 25 | 25 |
| Resolution | 416×544 | 416×544 |
| Dataset | NYU Depth V2, BTS augmentation | Same |
| λ_var (SILog) | 0.50 | **0.85** (standard dense-model value) |
| Loss | Sparse SILog + dense aux | **Single dense SILog** |

---

## 8. Parameter Count

| Component | v14 | v15 | Delta |
|---|---:|---:|---:|
| Encoder ConvNeXt V2-T | 28,600K | 28,600K | 0 |
| Projection neck (4 levels → 192) | 278K | **0** | −278K |
| L4 self-attention (2 layers) | 886K | **0** | −886K |
| Pre-compute (proj_L3/L4 + B3b KV) | 148K | **0** | −148K |
| Seed constructor (B1) | 160K | **0** | −160K |
| MSDA 3 layers (incl. canvas) | 2,055K | **0** | −2,055K |
| B3b 2 layers (incl. canvas) | 1,334K | **0** | −1,334K |
| Triple spatial canvas | 816K | **0** | −816K |
| **DPT-style FPN fusion** | — | **317K** | +317K |
| **Local patch path** | — | **246K** | +246K |
| Depth head | 74K | **~0K** (Conv1×1(32→1) = 33 params) | −74K |
| **Total (inference)** | **~34,351K** | **~29,163K** | **−5,188K** |
| Training-only heads | ~111K | **~0K** (dense_head = same as inference) | −111K |

Local patch path breakdown:
- Conv3×3(64→64) `conv_s8_1`: 64×64×9 + 64 ≈ **37K**
- Conv3×3(64→64) `conv_s8_2`: **37K**
- Conv1×1(96→64) `proj_l1`: 96×64 + 64 ≈ **6K**
- Conv3×3(64→64) `conv_m1`: **37K**
- Conv3×3(64→64) `conv_m2`: **37K**
- Conv3×3(64→64) `conv_u1`: **37K**
- Conv3×3(64→64) `conv_u2`: **37K**
- Conv3×3(64→32) `conv_f`: 64×32×9 + 32 ≈ **18K**
- **Local path total: ~246K**

DPT fusion breakdown:
- Projections (3× Conv1×1 + LN): 49K + 25K + 13K = **87K**
- refinenet4 (RCU2 + Conv1×1): 2×(64²×9+64) + 64²+64 = **78K**
- refinenet3 (RCU1 + RCU2 + Conv1×1): 4×(64²×9+64) + 64²+64 = **152K**
- **DPT total: ~317K**

---

## 9. Open Design Questions

### 9.1 s8 Branch Channel Width: D=64 Throughout vs Bottleneck

Current: all conv layers in the local path use D=64 (matching stride_8_map). If s8 context requires more capacity before merging with L1, consider expanding: conv_s8_1/2 at D=128, then project back to D=64 before merge. Tradeoff: more params/MACs in s8 branch but richer representation.

Start with D=64 flat. Ablate if s8 signal is not propagating effectively (diagnostic: measure performance when s8 branch is zeroed out).

### 9.2 Local Path Final Channels: 32 vs 64

Current: conv_f(64→32) → center[32] → Conv1×1(32→1). Increasing to 64 would give: conv_f(64→64) → center[64] → Conv1×1(64→1). Tradeoff: 18K more params, no MACs change at inference (valid conv, same spatial sizes). Likely negligible at this stage.

Start with 32. If depth predictions are noisy at boundaries, try 64.

### 9.3 λ_var for Dense SILog: 0.5 vs 0.85

Standard SILog for dense depth models (BTS, DPT) typically uses λ=0.85. v14 used λ=0.5 to reduce variance sensitivity during sparse training. With dense training, λ=0.85 is appropriate — more scale-invariant, less sensitive to depth scale drift. If training is unstable, reduce to 0.5.

Start with λ=0.85.

### 9.4 Inference K: Fixed vs Adaptive

At inference, K=128 random valid-pixel queries. Since training is purely dense, the model generalizes to ANY query positions (no distribution shift from grid training). Options:
- **Random valid-pixel** (default): maximizes coverage in valid regions
- **Uniform grid at inference:** avoids clustering, predictable error distribution

Since training is dense (no K-sampling bias), start with random valid-pixel queries.

### 9.5 Backbone: ConvNeXt V2-T vs ViT-S

For peak performance: ViT-S + DPT is the state-of-the-art choice (Depth Anything V2 style). ConvNeXt V2-T + DPT-style is validated by UniDepth V1. To cleanly evaluate the decoder architecture (v15 contribution): run the same DPT FPN + local patch head with ViT-S backbone and compare. If ViT-S gives major gains, the current approach is backbone-limited; if not, the decoder is the key contribution.

---

## 10. Theoretical Justification

### 10.1 Causal Correctness: DPT vs Canvas

| | Triple Canvas (v14) | DPT Fusion (v15) |
|---|---|---|
| **Direction** | Top-down: queries scatter-write → canvas learns | Bottom-up: encoder features → dense map |
| **Dependency** | Dense map quality depends on query predictions | Dense map built purely from encoder, independent of queries |
| **Supervision** | Canvas starts random; queries must learn write AND read simultaneously | Dense map directly supervised (stride-8 aux head); queries learn to read from a supervised map |
| **Cold start** | Circular: bad queries → bad canvas → bad queries | Stable: encoder gives good stride-8 features from epoch 1; queries can read from a meaningful map immediately |

The canvas forces queries to bootstrap the dense map from scratch. DPT builds the map independently and queries sample from it — a clean, one-directional dependency.

### 10.2 The MSDA Problem

MSDA's deformable sampling performs $K \times L \times H_{\text{attn}} \times N_{\text{pts}}$ bilinear lookups (at 416×544: 256×4×6×4 = 24,576 lookups per forward pass), each from a different predicted offset. These offsets are predicted per query per layer per level — a complex optimization landscape. The DPT-style global fusion pre-aggregates all multi-scale information into one stride-8 map, replacing 24,576 scattered lookups with K=128 straightforward bilinear samples from a supervised dense map.

### 10.3 Dense Training Signal Effectiveness

| | SILog loss terms (B=2, 416×544, ~80% valid pixels) |
|---|---|
| v14: Sparse point (K=256) | **512 terms** |
| v14: Canvas L2 aux (stride 4, 52×68 = 3,536 positions) | **7,072 terms** |
| **v15: Full-image dense (226,304 × 80% valid)** | **~181,000 terms** |

v15's dense training loss contributes ~25× more gradient terms than v14's total. The conv layers (conv_m1/2, conv_u1/2, conv_f) receive dense gradients at every spatial position — not scattered at K query centers — enabling proper learning of depth boundary sharpness.

### 10.4 Local→Global→Local Design

v15 restores the principled information flow:

1. **ConvNeXt (Local→Global):** Hierarchical encoder aggregates local texture into global scene features across 4 scales.
2. **DPT Fusion (Global):** Top-down FPN fuses all scales into a stride-8 representation with global context at every spatial position.
3. **Per-Query Local L1 (Global→Local):** After global context is encoded in `stride_8_map`, each query re-examines its fine-grained local neighborhood at stride 4→1 for sub-pixel precision.

This mirrors successful patterns: U-Net (encoder↓ + decoder↑), DPT (ViT global + reassemble local), PointNet++ (ball queries local → hierarchical global → interpolation local). v15 implements the same principle for sparse query depth in an efficient O(K) local path.

---

## 11. MACs Analysis

**Setup:** 416×544 input. ConvNeXt V2-T encoder ~10G MACs (scales roughly linearly with image area from 4.5G at 224×224). All decoder MACs computed as $H \times W \times C_{out} \times C_{in} \times k^2$ per Conv2d layer.

### 11.1 Decoder MACs Breakdown

**Global DPT path (once per image, K-independent, D=64, full RCU):**

| Component | Resolution | MACs |
|---|---|---:|
| Conv1×1(768→64) proj_L4 | 13×17=221 | **10.9M** |
| Conv1×1(384→64) proj_L3 | 26×34=884 | **21.7M** |
| Conv1×1(192→64) proj_L2 | 52×68=3,536 | **43.4M** |
| refinenet4: RCU2 — 2× Conv3×3(64→64) | 13×17=221 | **16.3M** |
| refinenet4: Conv1×1(64→64) out | 26×34=884 | **3.6M** |
| refinenet3: RCU1 — 2× Conv3×3(64→64) | 26×34=884 | **65.1M** |
| refinenet3: RCU2 — 2× Conv3×3(64→64) | 26×34=884 | **65.1M** |
| refinenet3: Conv1×1(64→64) out | 52×68=3,536 | **14.5M** |
| **Global path total (inference)** | | **~241M** |

**Per-query local path (valid conv, inference only):**

MACs per query = sum over all valid-conv layers (no padding, spatial size shrinks):

| Component | Input→Output spatial | MACs per query |
|---|---|---:|
| conv_s8_1: Conv3×3(64→64) valid | 9×9 → 7×7 = 49 pos | **1.81M** |
| conv_s8_2: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| bilinear upsample 5→9 | — | ~0 |
| proj_l1: Conv1×1(96→64) | 9×9 = 81 pos | **0.50M** |
| conv_m1: Conv3×3(64→64) valid | 9×9 → 7×7 = 49 pos | **1.81M** |
| conv_m2: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| bilinear upsample 5→7 | — | ~0 |
| conv_u1: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| conv_u2: Conv3×3(64→64) valid | 5×5 → 3×3 = 9 pos | **0.33M** |
| bilinear upsample 3→5 | — | ~0 |
| conv_f: Conv3×3(64→32) valid | 5×5 → 3×3 = 9 pos | **0.17M** |
| dense_head: Conv1×1(32→1) | 1×1 | ~0 |
| **Per-query total** | | **~7.4M** |

Valid convolutions never exceed 9×9 = 81 positions per layer — vs the old design's 32×32 = 1,024 positions for `conv_local_2` (9.44M alone). The largest input is 9×9 (81 pos), giving 56% lower per-query MACs.

**Training dense path MACs (one full image forward pass):**

| Component | Resolution | MACs |
|---|---|---:|
| Global DPT (refinenet4+3) | — | **241M** |
| conv_s8_1+s8_2 (padded) | 52×68 = 3,536 | 2×3,536×64²×9 = **262M** |
| proj_l1 (padded) | 104×136 = 14,144 | 14,144×64×96 = **87M** |
| conv_m1+m2 (padded) | 104×136 = 14,144 | 2×14,144×64²×9 = **1,048M** |
| conv_u1+u2 (padded) | 208×272 = 56,576 | 2×56,576×64²×9 = **4,196M** |
| conv_f (padded, 64→32) | 416×544 = 226,304 | 226,304×32×64×9 = **4,196M** |
| dense_head (Conv1×1, 32→1) | 226,304 | 226,304×32 = **7M** |
| **Training total** | | **~10.0G** |

Training cost is equivalent to a full DPT-style dense decoder pass — expected and acceptable.

### 11.2 Total v15 Decoder MACs vs K (Inference)

Per-query cost: **~7.4M** (valid-conv path).

| K | Global path | Per-query | **v15 decoder** |
|---:|---:|---:|---:|
| 32 | 241M | 237M | **478M** |
| 64 | 241M | 474M | **715M** |
| 128 | 241M | 947M | **1,188M** |

### 11.3 Model-Level MACs Comparison

| Model | Encoder | Decoder | **Total** | NYU AbsRel |
|---|---:|---:|---:|---:|
| DAv2-S (ViT-S + DPT D=64, dense, 518×518) | ~38G | ~11.6G | **~49G** | ~0.048 |
| Full DPT D=64, ConvNeXt, 416×544 (to stride 2) | ~10G | ~3.2G | **~13.2G** | — |
| v14 (ConvNeXt + MSDA + canvas, K=256) | ~10G | ~8G | **~18G** | 0.240+ |
| **v15 (K=64, inference)** | ~10G | ~0.7G | **~10.7G** | TBD |
| **v15 (K=128, inference)** | ~10G | ~1.2G | **~11.2G** | TBD |
| **v15 (training, dense)** | ~10G | ~10.0G | **~20G** | — |

**Full DPT dense decoder (D=64) breakdown** — what we avoid by stopping at stride 8 and using per-query patches:

| Component | Resolution | MACs |
|---|---|---:|
| Projections (all 4 levels) | L4/L3/L2/L1 | **163M** |
| refinenet4: RCU2 | 13×17=221 | **16.3M** |
| refinenet4: Conv1×1 out | 26×34=884 | **3.6M** |
| refinenet3: RCU1+RCU2 | 26×34=884 | **130M** |
| refinenet3: Conv1×1 out | 52×68=3,536 | **14.5M** |
| refinenet2: RCU1+RCU2 | 52×68=3,536 | **520M** |
| refinenet2: Conv1×1 out | 104×136=14,144 | **57.9M** |
| refinenet1: RCU1+RCU2 | 104×136=14,144 | **2,082M** |
| refinenet1: Conv1×1 out | 208×272=56,576 | **231M** |
| **Full DPT (4 refinenets, to stride 2)** | | **~3,219M** |

`refinenet1` at stride-4 (14,144 positions) alone costs **2,082M** — more than our entire v15 decoder at K=64 (1,329M). Stopping the global path at stride 8 and handling fine detail per-query saves this cost for K ≤ 175.

### 11.4 D-Dimension Comparison (Global Path Only)

All rows assume full RCU FeatureFusionBlock. `proj_s8` = Conv1×1(D→64) needed at D>64 for the per-query local path.

| D | Projections | refinenet4 | refinenet3 | proj_s8 | **Global total** | v15 K=64 | v15 K=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **64** | 76M | 20M | 145M | 0M | **241M** | **1,329M** | **2,417M** |
| 128 | 152M | 80M | 578M | 29M | **839M** | **1,927M** | **3,015M** |
| 192 | 228M | 179M | 1,300M | 43M | **1,750M** | **2,838M** | **3,926M** |
| 256 | 304M | 318M | 2,317M | 58M | **2,997M** | **4,085M** | **5,173M** |

At D=64, the global path (241M) is only 18% of the total decoder at K=64 (1,329M) — dominated by per-query local path (82%). At D=256, the global path (2,997M) dominates. D=64 keeps the global path inexpensive relative to the per-query work.

### 11.5 v15 (D=64) vs Full DPT Dense (D=64) — Decoder MACs

DPT dense at 416×544 (to stride 2, 4 refinenets + projections) = **3,219M** (K-independent).

**Break-even at K ≈ 402** (241 + 7.4K = 3,219 → K = 2,978 / 7.4 = **402**). At 7.4M/query, v15 inference is faster than full DPT dense for all K ≤ 402 — covering every practical inference scenario.

| K | v15 decoder (7.4M/query) | DPT dense decoder (D=64) | **Decoder speedup** |
|---:|---:|---:|---:|
| 16 | 241+118 = **359M** | 3,219M | **9.0×** |
| 32 | 241+237 = **478M** | 3,219M | **6.7×** |
| 64 | 241+474 = **715M** | 3,219M | **4.5×** |
| 100 | 241+740 = **981M** | 3,219M | **3.3×** |
| 128 | 241+947 = **1,188M** | 3,219M | **2.7×** |
| 256 | 241+1,894 = **2,135M** | 3,219M | **1.5×** |
| **402** | 241+2,975 = **3,216M** | 3,219M | **≈1×** |

**Structural comparison — training path (both D=64, padded conv):**

| Processing step | DPT dense (D=64) | v15 (D=64) training | Same? |
|---|---|---|---|
| Conv1×1 projections L4/L3/L2 | ✓ | ✓ | **identical** |
| refinenet4 (L4 → stride 16) | FeatureFusionBlock RCU2 | **identical** | **identical** |
| refinenet3 (stride 16+L3 → stride 8) | FeatureFusionBlock RCU1+RCU2 | **identical** | **identical** |
| stride 8 → stride 4 | refinenet2: RCU1+RCU2 (**578M**) | s8 padded conv ×2 + merge with L1 (**262M+87M**) | **different** — v15 uses s8 branch + L1 merge instead of refinenet2 |
| stride 4 → stride 2 | refinenet1 RCU1 (**1,041M**) | conv_m1+m2 padded (**1,048M**) → bilinear | **same cost**, different structure |
| stride 2 → stride 1 | refinenet1 RCU2+out (**1,041M**) | conv_u1+u2+f padded (**4,196M+4,196M**) | **more expensive** in v15 (going fully to stride 1) |
| Final prediction | Dense Conv3×3 | Dense Conv1×1(32→1) | **simpler** in v15 |

**Key structural difference:** DPT's refinenet2 fuses stride_8_map+L2 globally. v15's s8 branch processes stride_8_map alone, then merges with L1 directly — skipping L2 completely, using L1 for fine detail. The s8 branch at stride 8 then upsamples ×2 to stride 4 to merge with L1, equivalent in spirit to refinenet2 but with explicit L1 injection.
