# Research Plan: SPD v15 — Sparse Query-Point Depth from RGB

Author: Claude
Date: 2026-02-26
Version: v15 (DPT-style FPN global fusion + per-query progressive refinenet L2→L1; removes MSDA / B3b / canvas / Q2Q / neck)

> **v15 core insight:** v14's failure is a local-information problem. Every mechanism in v14's decoder — MSDA, canvas write-smooth-read, B3b — operates at stride ≥ 8. Queries have no sub-stride-8 precision, yet they predict depth at exact pixel positions. The canvas additionally has a structural flaw: queries must scatter-write to a dense map they haven't learned to fill, before reading from it — a backwards causal structure. DPT's insight is different: build the dense feature map bottom-up from encoder features (correct causal direction), then queries sample from it.
>
> **v15 applies this in two complementary parts:**
> 1. **Global (DPT-style FPN):** Progressive FPN fusion of L4→L3 produces a stride-8 dense feature map at D=64 channels, using full DPT **FeatureFusionBlock** (ResidualConvUnit) blocks — identical to DAv2-S. D=64 matches DAv2-Small (ViT-S scale ≈ ConvNeXt V2-T scale). Fusion MACs scale as D²: D=256 costs 16× more than D=64 — unjustified at our backbone scale. L2 is projected to D=64 (f2) but fused in the local path, not the global path.
> 2. **Local (per-query progressive refinenet — valid-conv inference / dense-padded training, shared weights):** The local path IS the continuation of DPT's refinenet hierarchy: **refinenet2** (stride_8_map + f2 → stride 4) → **refinenet1** (main + L1 → stride 2) → **head** (output_conv1 64→32 + ×2 + output_conv2 Conv3×3(32→32)+ReLU+Conv1×1(32→1) → depth). Same conv layers, two execution modes. **Inference:** per query, sample 9×9 from stride_8_map and f2, 7×7 from L1; each step applies RCU + merge + ×2 upsample. Channel reduction at stride 2 (DPT pattern). Two-step output_conv2 matches DPT exactly. **Training:** identical conv layers run with padded convolutions on full feature maps → dense H×W depth prediction → dense SILog loss (100% pixel coverage). All 4 encoder levels used: L4+L3 (global refinenet4+3), L2 (local refinenet2), L1 (local refinenet1).
> 3. **Dense training, no queries:** Training runs purely dense — no K-query sampling, no MLP depth head. Identical to DPT training: padded convs on full feature maps → H×W depth prediction → SILog over all valid pixels. The output_conv2 that produces the final 1×1 output per query at inference is the same layer that produces the dense H×W prediction at training. No separate query-based training path.
>
> **Result:** 6 modules removed (neck, L4-SA, seed constructor, canvas, MSDA, B3b). 2 modules added (DPT FPN, progressive refinenet local path). ~5.1M fewer parameters. Architecturally simpler — the full decoder is a DPT refinenet hierarchy (with Conv1×1 output projections and two-step output_conv2, matching DPT exactly) split at the stride-8 boundary between global (dense) and local (per-query) execution. Training cost matches full DPT dense (~6.4G).

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
| **DPT-style FPN fusion** | — | **NEW: L4→L3 → stride-8 map @ D=64 (~317K), full FeatureFusionBlock (RCU). f2 (L2 projected to D=64) feeds into local refinenet2.** | D=64 = DAv2-S scale; full RCU = proper DPT blocks; D² scaling makes D=256 prohibitive |
| **Progressive refinenet local path** | — | **NEW: refinenet2 (s8 RCU 9→5 + L2 RCU 9→5 → merge → ×2 → 7×7) → refinenet1 (main RCU 7→3 + L1 RCU 7→3 → merge → ×2 → 5×5) → head (output_conv1 64→32, 5→3, ×2 → output_conv2 Conv3×3(32→32)+ReLU+Conv1×1(32→1), 3→1) (~337K)** | Progressive refinenet hierarchy matching DPT: refinenet2 fuses L2, refinenet1 fuses L1. All 4 encoder levels used. Channel reduction at s2, two-step output_conv2 (DPT pattern). |
| Depth head | MLP(192→384→1) (~74K) | **Integrated: output_conv2 Conv3×3(32→32)+Conv1×1(32→1) is the last layer of the local path (~9.3K)** | No separate head — depth emerges from the refinenet chain directly, matching DPT's two-step prediction |
| Dense training | Canvas L2 aux head (stride 4, 6.25%) | **Full-image padded convs (same local path weights) → dense H×W prediction → dense SILog** | 100% pixel coverage; training cost matches full DPT dense (~6.4G) |
| **Total inference params** | **~34,351K** | **~29,254K** | **−5,097K** |

---

## 3. Architecture Overview

```
RGB Image [H × W × 3]   (416 × 544 at training)
  │
  ▼
ConvNeXt V2-T  (FCMAE pre-trained, fine-tuned 0.1× LR)
  L1: [H/4  × W/4  ×  96]   stride 4   ← local refinenet1 (L1 branch)
  L2: [H/8  × W/8  × 192]   stride 8   ← projected to f2 (D=64), local refinenet2 (L2 branch)
  L3: [H/16 × W/16 × 384]   stride 16
  L4: [H/32 × W/32 × 768]   stride 32
  │
  ▼ ─────────────────────────────────────────────────────────────
  GLOBAL PATH — refinenet4 + refinenet3 (dense, once per image)
  │
  ▼
DPT-Style FPN Fusion  (2× FeatureFusionBlock/RCU, D=64 — identical to DAv2-S)
  f4 = LN(Conv1×1(768→64)(L4))   [B, 64, H/32, W/32]
  f3 = LN(Conv1×1(384→64)(L3))   [B, 64, H/16, W/16]
  f2 = LN(Conv1×1(192→64)(L2))   [B, 64, H/8,  W/8 ]   ← fed to local refinenet2
  │
  refinenet4: RCU2(f4) → ×2 → Conv1×1                    [B, 64, H/16, W/16]
  refinenet3: RCU1(f3) + add + RCU2 → ×2 → Conv1×1 = stride_8_map [B, 64, H/8, W/8]
  │
  ▼ ─────────────────────────────────────────────────────────────
  LOCAL PATH — refinenet2 + refinenet1 + head
  (training: dense O(H×W);  inference: per-query O(K))
  │
  TRAINING — padded Conv3×3 on full feature maps:
  │  ┌ refinenet2 (stride 8 → stride 4):
  │  │  stride_8_map [B,64,H/8,W/8] — conv_s8_1, conv_s8_2 (s8 RCU, padded)
  │  │  f2 [B,64,H/8,W/8]           — rcu_l2_1, rcu_l2_2 (L2 RCU, padded)
  │  │  add → bilinear ×2 → rn2_out Conv1×1(64→64) → [B,64,H/4,W/4]
  │  ├ refinenet1 (stride 4 → stride 2):
  │  │  main [B,64,H/4,W/4]         — rn1_conv1, rn1_conv2 (main RCU, padded)
  │  │  L1 [B,96,H/4,W/4] — proj_l1 — rcu_l1_1, rcu_l1_2 (L1 RCU, padded)
  │  │  add → bilinear ×2 → rn1_out Conv1×1(64→64) → [B,64,H/2,W/2]
  │  └ head (stride 2 → depth):
  │     output_conv1 (64→32, padded, no activation) → bilinear ×2 → [B,32,H,W]
  │     output_conv2: Conv3×3(32→32, padded) → ReLU → Conv1×1(32→1) → [B,1,H,W]
  │     → dense SILog (all valid pixels)
  │
  INFERENCE — valid Conv3×3 on patches per query:
  │  ┌ refinenet2 (stride 8):
  │  │  s8 branch: 9×9 from stride_8_map  → [B*K, 64, 9, 9]
  │  │    valid Conv3×3 (conv_s8_1)        → [B*K, 64, 7, 7]
  │  │    valid Conv3×3 (conv_s8_2)        → [B*K, 64, 5, 5]
  │  │  L2 branch: 9×9 from f2            → [B*K, 64, 9, 9]
  │  │    valid Conv3×3 (rcu_l2_1)         → [B*K, 64, 7, 7]
  │  │    valid Conv3×3 (rcu_l2_2)         → [B*K, 64, 5, 5]
  │  │  Merge (add)                        → [B*K, 64, 5, 5]    stride 8
  │  │  bilinear ×2 → 10×10, crop 7×7     → [B*K, 64, 7, 7]    stride 4
  │  │  Conv1×1(64→64) (rn2_out)          → [B*K, 64, 7, 7]    stride 4
  │  ├ refinenet1 (stride 4):
  │  │  Main RCU:
  │  │    valid Conv3×3 (rn1_conv1)        → [B*K, 64, 5, 5]
  │  │    valid Conv3×3 (rn1_conv2)        → [B*K, 64, 3, 3]
  │  │  L1 branch: 7×7 from L1            → [B*K, 96, 7, 7]
  │  │    Conv1×1(96→64) (proj_l1)         → [B*K, 64, 7, 7]
  │  │    valid Conv3×3 (rcu_l1_1)         → [B*K, 64, 5, 5]
  │  │    valid Conv3×3 (rcu_l1_2)         → [B*K, 64, 3, 3]
  │  │  Merge (add)                        → [B*K, 64, 3, 3]    stride 4
  │  │  bilinear ×2 → 6×6, crop 5×5       → [B*K, 64, 5, 5]    stride 2
  │  │  Conv1×1(64→64) (rn1_out)          → [B*K, 64, 5, 5]    stride 2
  │  └ head (stride 2 → depth):
  │     valid Conv3×3 (output_conv1, 64→32) → [B*K, 32, 3, 3]   stride 2  (no activation)
  │     bilinear ×2 → 6×6, crop 3×3        → [B*K, 32, 3, 3]   stride 1
  │     valid Conv3×3 (output_conv2a, 32→32)→ [B*K, 32, 1, 1]   stride 1  + ReLU
  │     Conv1×1 (output_conv2b, 32→1)      → [B*K, 1,  1, 1]   depth!  (no activation)
  │  depth = exp(output)                     → [B, K]
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

**No projection neck in v15.** DPT fusion projects L2/L3/L4 to D=64 internally. L1 is passed directly (native 96ch) to local refinenet1 — projected to D=64 by `proj_l1` there. L2's projection `f2` (computed in the DPT FPN) feeds directly into local refinenet2 — no additional projection needed.

**On backbone choice:** UniDepth V1 (CVPR 2024) demonstrates ConvNeXt-L [96,192,384,768] + DPT-style decoder for metric depth — confirming this combination is architecturally sound. The top-performing monocular depth models (Depth Anything V2, challenge winners 2024-2025) use ViT/DINOv2 backbones. ConvNeXt V2 FCMAE pretraining is well-suited for dense tasks; however, a ViT-S backbone ablation (same DPT FPN + local patch head, different backbone) would cleanly measure the backbone contribution. For v15, ConvNeXt V2-T is kept to isolate the decoder architecture change.

---

## 5. Global DPT-Style FPN Fusion

### 5.1 Design Rationale

DPT (Ranftl et al., ICCV 2021) established the standard for dense feature fusion from hierarchical backbones: project each scale to a uniform dimension D via 1×1 conv, then progressively fuse coarse-to-fine via **FeatureFusionBlock** — each block applies ResidualConvUnit (RCU = 2× Conv3×3 with residual skip), bilinear ×2 upsample, and Conv1×1. This is what DPT calls "Reassembly" (projection) + "Fusion" (RefineNet-based). **v15 uses the full FeatureFusionBlock — identical to DAv2.**

**Why D=64:** Depth Anything V2-Small (ViT-S backbone, ConvNeXt V2-T equivalent scale) uses D=64. DPT projects to D matching backbone scale: ViT-S → 64, ViT-B → 128, ViT-L → 256. Fusion MACs scale as D²: going D=64 → 256 multiplies RCU cost by 16×. At 8GB VRAM, D=64 is the correct choice for our backbone scale (see Section 11.4 for D comparison).

**Why the global path stops at stride 8:**

The full DPT hierarchy is refinenet4 → refinenet3 → refinenet2 → refinenet1 → head. v15 splits this at the stride-8 boundary:
- **Global (dense):** refinenet4 (L4) + refinenet3 (L3) → stride_8_map. Runs once per image.
- **Local (per-query):** refinenet2 (stride_8_map + f2) + refinenet1 (main + L1) + head. Runs per query at inference, dense at training.

At 416×544, the position count per level:

| Level | Resolution | Positions | Relative cost |
|---|---|---:|---:|
| L4 (stride 32) | 13×17 | 221 | 1× |
| L3 (stride 16) | 26×34 | 884 | 4× |
| L2 (stride 8) | 52×68 | 3,536 | 16× |
| L1 (stride 4) | 104×136 | 14,144 | **64×** |

refinenet2 and refinenet1 each process 4× more positions than the previous level. Running them globally costs 16×+64× = 80× base, benefiting only K query positions. Running them per-query (valid-conv patches) makes cost O(K) instead of O(H×W). The stride-8 boundary is the natural split: refinenet4+3 are cheap enough to run globally; refinenet2+1 are where per-query execution pays off.

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

`stride_8_map [B, 64, H/8, W/8]` — multi-scale context from L3+L4 at D=64. The 6× Conv3×3 through refinenet4+refinenet3 give substantial RF (~224px), equivalent to DPT's `path_3`. `f2 [B, 64, H/8, W/8]` — L2 projected to D=64; feeds into local refinenet2 as a skip connection (matching DPT's refinenet2 structure). Both are sampled per-query by the local path (Section 6). At D=64, no `proj_s8` projection needed.

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

## 6. Per-Query Prediction (Progressive Refinenet Local Path)

**No global_h.** The local path samples 9×9 neighborhoods from `stride_8_map` and `f2`, injecting global depth context. A separate single-pixel global_h lookup is unnecessary and untrained (training is purely dense — no per-query path). All learnable parameters in the decoder are convolution or projection layers; no MLP, no attention.

**The local path IS the continuation of DPT's refinenet hierarchy.** The full DPT decoder is: refinenet4 → refinenet3 → refinenet2 → refinenet1 → head. v15's global path runs refinenet4+3 (dense, once per image). The local path runs refinenet2+1+head — structurally matching DPT but executed per-query at inference (valid conv) and dense at training (padded conv). The output_conv2 Conv3×3(32→1) that produces the 1×1 depth output per query at inference is the same layer that produces the H×W dense output during training. No separate depth head.

### 6.1 Local Patch Sampling (Inference)

**Progressive structure:** Each refinenet step fuses two branches (main flow + encoder skip) via RCU, matching DPT's FeatureFusionBlock pattern:
- **refinenet2** (stride 8): s8 RCU on `stride_8_map` (main) + L2 RCU on `f2` (skip) → merge → ×2 → stride 4
- **refinenet1** (stride 4): main RCU on refinenet2 output + L1 RCU on `L1` (skip) → merge → ×2 → stride 2
- **head** (stride 2 → 1): output_conv1 (64→32, no activation) → ×2 → output_conv2 (Conv3×3(32→32)+ReLU+Conv1×1(32→1)) → depth

**Three grid_sample calls per query group (inference):**

```python
# Inputs:
# L1: [B, 96, H4, W4]           (H4=104, W4=136 at 416×544)
# f2: [B, 64, H8, W8]            (H8=52, W8=68, from DPT FPN projection)
# stride_8_map: [B, 64, H8, W8]  (H8=52, W8=68)

B, K, _ = coords.shape
H4, W4 = L1.shape[2], L1.shape[3]
H8, W8 = stride_8_map.shape[2], stride_8_map.shape[3]

N_s8 = 9   # s8 and L2 patch size (both at stride 8)
N_l1 = 7   # L1 patch size (stride 4)

# --- s8 patch (stride-8 space, 9×9) ---
half_s8 = N_s8 // 2   # = 4
dx_s8 = torch.arange(-half_s8, half_s8 + 1, device=L1.device, dtype=torch.float32)
dy_s8 = torch.arange(-half_s8, half_s8 + 1, device=L1.device, dtype=torch.float32)
gy_s8, gx_s8 = torch.meshgrid(dy_s8, dx_s8, indexing='ij')  # [9, 9]

cx_s8 = coords[..., 0] / 8.0   # [B, K]
cy_s8 = coords[..., 1] / 8.0
sx_s8 = cx_s8[..., None, None] + gx_s8   # [B, K, 9, 9]
sy_s8 = cy_s8[..., None, None] + gy_s8
grid_s8 = torch.stack([
    2.0 * sx_s8 / (W8 - 1) - 1.0,
    2.0 * sy_s8 / (H8 - 1) - 1.0
], dim=-1).reshape(B, K * N_s8, N_s8, 2)

# Same grid coordinates for both stride-8 maps
raw_s8 = F.grid_sample(stride_8_map, grid_s8, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
s8_patches = raw_s8.reshape(B, 64, K, N_s8, N_s8).permute(0,2,1,3,4).reshape(B*K, 64, N_s8, N_s8)
# s8_patches: [B*K, 64, 9, 9]

raw_l2 = F.grid_sample(f2, grid_s8, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
l2_patches = raw_l2.reshape(B, 64, K, N_s8, N_s8).permute(0,2,1,3,4).reshape(B*K, 64, N_s8, N_s8)
# l2_patches: [B*K, 64, 9, 9]

# --- L1 patch (stride-4 space, 7×7) ---
half_l1 = N_l1 // 2   # = 3
dx_l1 = torch.arange(-half_l1, half_l1 + 1, device=L1.device, dtype=torch.float32)
dy_l1 = torch.arange(-half_l1, half_l1 + 1, device=L1.device, dtype=torch.float32)
gy_l1, gx_l1 = torch.meshgrid(dy_l1, dx_l1, indexing='ij')  # [7, 7]

cx_l1 = coords[..., 0] / 4.0
cy_l1 = coords[..., 1] / 4.0
sx_l1 = cx_l1[..., None, None] + gx_l1   # [B, K, 7, 7]
sy_l1 = cy_l1[..., None, None] + gy_l1
grid_l1 = torch.stack([
    2.0 * sx_l1 / (W4 - 1) - 1.0,
    2.0 * sy_l1 / (H4 - 1) - 1.0
], dim=-1).reshape(B, K * N_l1, N_l1, 2)
raw_l1 = F.grid_sample(L1, grid_l1, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
L1_patches = raw_l1.reshape(B, 96, K, N_l1, N_l1).permute(0,2,1,3,4).reshape(B*K, 96, N_l1, N_l1)
# L1_patches: [B*K, 96, 7, 7]
```

**Memory (B=2, K=128):** `s8_patches` [256, 64, 9, 9] ≈ 5.3MB; `l2_patches` [256, 64, 9, 9] ≈ 5.3MB; `L1_patches` [256, 96, 7, 7] ≈ 4.8MB. Total ≈ 15.4MB. (Previously 17.2MB with 11×11 L1 patches.)

### 6.2 Local Fusion — Progressive Refinenet (Inference) / Padded Conv Dense (Training)

The local path uses **identical conv weights** in two execution modes:

| Mode | Padding | Input | Output |
|---|---|---|---|
| Inference | **valid** (no pad) | s8: 9×9, f2: 9×9, L1: 7×7 per query | 1×1 per query → direct depth |
| Training | **same** (pad=1) | Full feature maps [B, C, H/s, W/s] | Dense [B, 1, H, W] depth |

**Inference — step by step (progressive refinenet):**

```python
# ── STEP 1: refinenet2 (stride 8 → stride 4) ──────────────────────────
# s8 branch: RCU on stride_8_map context (2× Conv3×3)
x_s8 = F.gelu(self.conv_s8_1(s8_patches))  # valid Conv3×3(64→64): 9→7
x_s8 = F.gelu(self.conv_s8_2(x_s8))        # valid Conv3×3(64→64): 7→5

# L2 branch: RCU on f2 skip (2× Conv3×3) — f2 already D=64, no proj needed
x_l2 = F.gelu(self.rcu_l2_1(l2_patches))   # valid Conv3×3(64→64): 9→7
x_l2 = F.gelu(self.rcu_l2_2(x_l2))         # valid Conv3×3(64→64): 7→5

# Merge refinenet2
x = x_s8 + x_l2                             # add: [B*K, 64, 5, 5]  stride 8
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
# x: [B*K, 64, 10, 10]
x = recenter_crop(x, query_offset, crop=7)  # [B*K, 64, 7, 7]  stride 4
x = self.rn2_out(x)                         # Conv1×1(64→64): [B*K, 64, 7, 7]

# ── STEP 2: refinenet1 (stride 4 → stride 2) ──────────────────────────
# Main branch: RCU on refinenet2 output (2× Conv3×3)
x_main = F.gelu(self.rn1_conv1(x))          # valid Conv3×3(64→64): 7→5
x_main = F.gelu(self.rn1_conv2(x_main))     # valid Conv3×3(64→64): 5→3

# L1 branch: project + RCU (Conv1×1 + 2× Conv3×3)
x_l1 = self.proj_l1(L1_patches)             # Conv1×1(96→64): [B*K, 96, 7,7] → [B*K, 64, 7,7]
x_l1 = F.gelu(self.rcu_l1_1(x_l1))         # valid Conv3×3(64→64): 7→5
x_l1 = F.gelu(self.rcu_l1_2(x_l1))         # valid Conv3×3(64→64): 5→3

# Merge refinenet1
x = x_main + x_l1                           # add: [B*K, 64, 3, 3]  stride 4
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
# x: [B*K, 64, 6, 6]
x = recenter_crop(x, query_offset, crop=5)  # [B*K, 64, 5, 5]  stride 2
x = self.rn1_out(x)                         # Conv1×1(64→64): [B*K, 64, 5, 5]

# ── STEP 3: head (stride 2 → depth) ───────────────────────────────────
x = self.output_conv1(x)                    # valid Conv3×3(64→32): 5→3  ← channel reduction, NO activation
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
# x: [B*K, 32, 6, 6]
x = recenter_crop(x, query_offset, crop=3)  # [B*K, 32, 3, 3]  stride 1

x = self.output_conv2(x)                    # Conv3×3(32→32) valid 3→1 + ReLU + Conv1×1(32→1)
# output_conv2 = nn.Sequential(Conv2d(32,32,3), ReLU, Conv2d(32,1,1))
depth = torch.exp(x.reshape(B, K))           # [B, K]  (no final activation — log-depth)
```

**Upsample ×2 + recenter-crop:** All three upsamples in the valid-conv path use exact `scale_factor=2` — never fractional scaling. After each exact 2× upsample, the query pixel lands on one of the center 2×2 pixels of the upsampled grid. `recenter_crop` identifies which pixel the query maps to, sets it as the new center, and crops. Margins at each step:
- 5→10→crop 7: margin = 3 (sufficient)
- 3→6→crop 5: margin = 1 (sufficient)
- 3→6→crop 3: margin = 3 (sufficient)

```python
def recenter_crop(x, query_offset, crop):
    """Crop 'crop×crop' from upsampled feature, centered on query pixel.

    After exact 2× upsample from (N×N) → (2N×2N), the query pixel that was
    at center [N//2, N//2] now maps to one of the center 2×2 pixels in the
    upsampled grid. query_offset (0 or 1 in each dim) selects which one.
    """
    H, W = x.shape[2], x.shape[3]
    # Query pixel position in the upsampled grid
    cy = H // 2 - 1 + query_offset[..., 1]  # [B*K] or scalar
    cx = W // 2 - 1 + query_offset[..., 0]
    # Crop centered on query pixel
    half = crop // 2
    return x[:, :, cy - half : cy - half + crop, cx - half : cx - half + crop]
```

**Spatial trace (inference, valid convolutions):**

| Step | Size | Space | Notes |
|---|---|---|---|
| **refinenet2** | | **stride 8** | |
| s8 sample | 9×9 | s8 | from stride_8_map |
| conv_s8_1 | 7×7 | s8 | valid (s8 RCU) |
| conv_s8_2 | 5×5 | s8 | valid (s8 RCU) |
| L2 sample | 9×9 | s8 | from f2 |
| rcu_l2_1 | 7×7 | s8 | valid (L2 RCU) |
| rcu_l2_2 | 5×5 | s8 | valid (L2 RCU) |
| **Merge (add)** | **5×5** | **s8** | **s8 + L2 branches** |
| upsample ×2, crop | 7×7 | s4 | 5→10→crop 7 |
| rn2_out (64→64) | **7×7** | **s4** | Conv1×1 output projection |
| **refinenet1** | | **stride 4** | |
| rn1_conv1 (64→64) | 5×5 | s4 | valid (main RCU) |
| rn1_conv2 (64→64) | 3×3 | s4 | valid (main RCU) |
| L1 sample | 7×7 | s4 | from L1 |
| proj_l1 (96→64) | 7×7 | s4 | Conv1×1 |
| rcu_l1_1 | 5×5 | s4 | valid (L1 RCU) |
| rcu_l1_2 | 3×3 | s4 | valid (L1 RCU) |
| **Merge (add)** | **3×3** | **s4** | **main + L1 branches** |
| upsample ×2, crop | 5×5 | s2 | 3→6→crop 5 |
| rn1_out (64→64) | **5×5** | **s2** | Conv1×1 output projection |
| **head** | | **stride 2→1** | |
| output_conv1 (64→32) | 3×3 | s2 | valid, **channel reduction**, no activation |
| upsample ×2, crop | 3×3 | s1 | 3→6→crop 3 |
| output_conv2 Conv3×3(32→32) | 1×1 | s1 | valid + **ReLU** |
| **output_conv2 Conv1×1(32→1)** | **1×1** | **s1** | **→ log-depth (no activation)** |

**Training — same weights, padded convolutions, full feature maps:**

```python
# ── refinenet2: stride 8 → stride 4 ───────────────────────────────────
# s8 branch (on full stride_8_map [B, 64, H/8, W/8])
x_s8 = F.gelu(self.conv_s8_1(stride_8_map))   # padded Conv3×3(64→64) → [B, 64, H/8, W/8]
x_s8 = F.gelu(self.conv_s8_2(x_s8))           # padded Conv3×3(64→64) → [B, 64, H/8, W/8]

# L2 branch (on full f2 [B, 64, H/8, W/8])
x_l2 = F.gelu(self.rcu_l2_1(f2))             # padded Conv3×3(64→64) → [B, 64, H/8, W/8]
x_l2 = F.gelu(self.rcu_l2_2(x_l2))           # padded Conv3×3(64→64) → [B, 64, H/8, W/8]

# Merge refinenet2 + upsample to stride 4
x = x_s8 + x_l2                               # add: [B, 64, H/8, W/8]
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
x = self.rn2_out(x)                           # Conv1×1(64→64) → [B, 64, H/4, W/4]

# ── refinenet1: stride 4 → stride 2 ───────────────────────────────────
# Main branch: RCU on refinenet2 output
x_main = F.gelu(self.rn1_conv1(x))            # padded Conv3×3(64→64) → [B, 64, H/4, W/4]
x_main = F.gelu(self.rn1_conv2(x_main))       # padded Conv3×3(64→64) → [B, 64, H/4, W/4]

# L1 branch: project + RCU (on full L1 [B, 96, H/4, W/4])
x_l1 = self.proj_l1(L1)                       # Conv1×1(96→64) → [B, 64, H/4, W/4]
x_l1 = F.gelu(self.rcu_l1_1(x_l1))           # padded Conv3×3(64→64) → [B, 64, H/4, W/4]
x_l1 = F.gelu(self.rcu_l1_2(x_l1))           # padded Conv3×3(64→64) → [B, 64, H/4, W/4]

# Merge refinenet1 + upsample to stride 2
x = x_main + x_l1                             # add: [B, 64, H/4, W/4]
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
x = self.rn1_out(x)                           # Conv1×1(64→64) → [B, 64, H/2, W/2]

# ── head: stride 2 → depth ────────────────────────────────────────────
x = self.output_conv1(x)                      # padded Conv3×3(64→32) → [B, 32, H/2, W/2]  ← reduce, NO activation
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
# x: [B, 32, H, W]

dense_depth = torch.exp(self.output_conv2(x))  # Conv3×3(32→32)+ReLU+Conv1×1(32→1) → [B, 1, H, W]
# output_conv2 = nn.Sequential(Conv2d(32,32,3,1,1), ReLU, Conv2d(32,1,1))
# No separate dense_head — output_conv2 IS the depth prediction layer
```

The **upsampling chain in training** follows the progressive refinenet structure: stride 8 → ×2 → Conv1×1 → stride 4 → ×2 → Conv1×1 → stride 2 → ×2 → stride 1. Each ×2 upsample is followed by a Conv1×1 output projection (matching DPT's FeatureFusionBlock pattern). Channel reduction happens at output_conv1 (stride 2, 64→32, no activation), following DPT's pattern. The two-step output_conv2 (Conv3×3(32→32)+ReLU+Conv1×1(32→1)) matches DPT exactly. **Training MACs ~6.4G** — identical to full DPT dense decoder at D=64.

### 6.3 Training vs Inference: Same Weights, Different Execution

| Property | Inference | Training |
|---|---|---|
| Padding mode | **valid** (no pad) | **same** (pad=1) |
| Input to local path | s8: 9×9, f2: 9×9, L1: 7×7 per query | Full feature maps |
| Output | 1×1 per query → depth directly (via output_conv2) | H×W dense depth → dense SILog |
| Depth prediction | output_conv2: Conv3×3(32→32) valid 3→1 + ReLU + Conv1×1(32→1) | output_conv2: Conv3×3(32→32, padded) + ReLU + Conv1×1(32→1) → H×W |
| Compute | O(K) | O(H×W) |
| VRAM at B=2, K=128 | ~15MB patches | ~stride-1 features [2,32,416,544] ≈ 55MB |

**Why valid convolutions give the same result as dense padded at the query pixel:** A Conv3×3 with same-padding at a pixel far from the border computes the same output as a Conv3×3 without padding on a patch centered at that pixel. Valid conv on a P×P patch gives a (P−2)×(P−2) output — exactly the center pixels of the padded result. Between valid-conv blocks, exact 2× upsample + recenter-crop restores the spatial size for the next block while keeping the query pixel centered. The progressive refinenet chain naturally converges: 9→5 (refinenet2) → 7→3 (refinenet1) → 5→3 (output_conv1) → 3→1 (output_conv2 Conv3×3) → Conv1×1 on 1×1 (no spatial change), producing depth directly with no center extraction needed.

### 6.4 Depth Prediction (Two-Step output_conv2 — Matching DPT)

There is **no separate depth head**. The last module — `output_conv2: Conv3×3(32→32) + ReLU + Conv1×1(32→1)` — directly outputs depth, matching DPT's two-step prediction structure exactly. At inference, Conv3×3 valid-reduces 3→1, then Conv1×1 maps 32→1 on the single pixel. At training, both layers run with padding on the full H×W feature map.

```python
# output_conv2 definition (matches DPT exactly, minus final ReLU — we use log-depth)
self.output_conv2 = nn.Sequential(
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # training: padded; inference: valid 3→1
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),   # 32→1 channel reduction
)

# Inference:
# x: [B*K, 32, 3, 3] — output of recenter_crop after output_conv1 + ×2
depth = torch.exp(self.output_conv2(x).reshape(B, K))  # Conv3×3 valid 3→1 + ReLU + Conv1×1 → [B, K]
```

Bias of output_conv2's Conv1×1 initialized to ln(2.5) (metric depth prior). No activation after Conv1×1 — output is log-depth, exponentiated to metric depth. (DPT uses ReLU after Conv1×1 because it predicts positive depth directly; v15 uses exp() on log-depth instead.) No `log_scale` parameter.

**Why two-step (matching DPT):** DPT's output_conv2 uses Conv3×3(D/2→32) + ReLU + Conv1×1(32→1). At our scale (D=64, D/2=32), the Conv3×3 is 32→32 — it provides a nonlinear feature transformation at full resolution before the final Conv1×1 depth projection. This gives 32 channels of learned features at stride 1 before prediction, matching DPT's proven design. At inference (valid conv, 1 position), the cost is negligible (~0.01M). At training (226K positions at stride 1), this is the dominant cost: ~2,093M — but it's exactly what DPT pays.

**output_conv1 has no activation** — matching DPT. The bilinear ×2 upsample operates on linear features, and the nonlinearity comes from ReLU inside output_conv2.

**Channel reduction rationale (DPT pattern):** DPT keeps D through all refinenets, reducing only in the head (output_conv1: D→D/2, output_conv2: D/2→32→1). v15 follows the same principle: 64 channels through refinenet2 and refinenet1, reducing to 32 at output_conv1 (stride 2, no activation). Full-width processing through both merges preserves information from all branches. output_conv1 directly parallels DPT's output_conv1 (both Conv3×3, D→D/2, at stride 2, no activation).

---

## 7. Training Strategy

### 7.1 Training: Purely Dense, No K Queries

Training uses **no per-query sampling at all**. The full forward pass at training time:

1. Encoder → L1, L2, L3, L4
2. DPT FPN → stride_8_map [B, 64, H/8, W/8], f2 [B, 64, H/8, W/8]
3. Progressive refinenet local path (padded convs on full maps):
   - refinenet2: s8 RCU(stride_8_map) + L2 RCU(f2) → merge → ×2 → stride 4
   - refinenet1: main RCU + L1 RCU(L1) → merge → ×2 → stride 2
   - head: output_conv1(64→32, no activation) → ×2 → output_conv2(Conv3×3(32→32)+ReLU+Conv1×1(32→1)) → dense_depth [B, 1, H, W]
4. Loss: dense SILog against GT depth map (all valid pixels)

All learnable parameters — DPT projections, RCU Conv3×3, proj_l1, conv_s8_1/2, rcu_l2_1/2, rn1_conv1/2, rcu_l1_1/2, output_conv1, output_conv2 — are trained via this single dense pass. No MLP, no Q-sampling, no K² cost. Training architecture is structurally identical to full DPT dense (~6.4G decoder MACs).

**At inference:** K=128 random valid-pixel queries. Each query uses the valid-conv progressive refinenet path (9×9 s8 + 9×9 f2 + 7×7 L1 patches → refinenet2 → refinenet1 → head → depth directly).

### 7.2 Loss Function

Single dense SILog loss:
```python
# Training forward pass (no queries) — see Section 6.2 for full code
# refinenet2: s8 RCU(stride_8_map) + L2 RCU(f2) → add → ×2 → [B, 64, H/4, W/4]
# refinenet1: main RCU + proj_l1(L1) + L1 RCU → add → ×2 → [B, 64, H/2, W/2]
# head: output_conv1(64→32, no activation) → ×2 → output_conv2(32→32+ReLU+32→1) → [B, 1, H, W]
dense_depth = torch.exp(self.output_conv2(x))  # Conv3×3(32→32)+ReLU+Conv1×1(32→1) → [B, 1, H, W]

loss = l_dense_silog(dense_depth, gt_depth_map, lambda_var=0.85)
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

v15 training is architecturally equivalent to DPT training: same dense padded convs, same H×W supervision, same progressive refinenet structure. The decoder is refinenet4 → refinenet3 (global) → refinenet2 → refinenet1 → head (local), matching DPT's hierarchy exactly. The only difference is execution mode at inference: global refinenets run dense, local refinenets run per-query.

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
| **Progressive refinenet local path** | — | **337K** | +337K |
| Depth head | 74K | **Integrated (output_conv2 = last module of local path, ~9.3K)** | −74K |
| **Total (inference)** | **~34,351K** | **~29,254K** | **−5,097K** |
| Training-only heads | ~111K | **~0K** (output_conv2 = same as inference) | −111K |

Progressive refinenet local path breakdown:

*refinenet2 (stride 8):*
- Conv3×3(64→64) `conv_s8_1`: 64×64×9 + 64 ≈ **37K** (s8 RCU)
- Conv3×3(64→64) `conv_s8_2`: **37K** (s8 RCU)
- Conv3×3(64→64) `rcu_l2_1`: **37K** (L2 RCU)
- Conv3×3(64→64) `rcu_l2_2`: **37K** (L2 RCU)
- Conv1×1(64→64) `rn2_out`: 64×64 + 64 ≈ **4K** (output projection)

*refinenet1 (stride 4):*
- Conv3×3(64→64) `rn1_conv1`: **37K** (main RCU)
- Conv3×3(64→64) `rn1_conv2`: **37K** (main RCU)
- Conv1×1(96→64) `proj_l1`: 96×64 + 64 ≈ **6K** (L1 projection)
- Conv3×3(64→64) `rcu_l1_1`: **37K** (L1 RCU)
- Conv3×3(64→64) `rcu_l1_2`: **37K** (L1 RCU)
- Conv1×1(64→64) `rn1_out`: 64×64 + 64 ≈ **4K** (output projection)

*head (stride 2 → depth):*
- Conv3×3(64→32) `output_conv1`: 64×32×9 + 32 ≈ **18K** (**channel reduction**, no activation)
- Conv3×3(32→32) `output_conv2[0]`: 32×32×9 + 32 ≈ **9.2K** (+ ReLU)
- Conv1×1(32→1)  `output_conv2[2]`: 32×1 + 1 ≈ **0.03K** (depth output, no activation)

- **Local path total: ~337K** (matches DPT exactly: Conv1×1 output projections + two-step output_conv2)

DPT fusion breakdown:
- Projections (3× Conv1×1 + LN): 49K + 25K + 13K = **87K**
- refinenet4 (RCU2 + Conv1×1): 2×(64²×9+64) + 64²+64 = **78K**
- refinenet3 (RCU1 + RCU2 + Conv1×1): 4×(64²×9+64) + 64²+64 = **152K**
- **DPT total: ~317K**

---

## 9. Open Design Questions

### 9.1 Channel Width: D=64 Throughout vs Bottleneck

Current: all conv layers in the local path use D=64 (matching stride_8_map and f2). All refinenet RCU blocks (s8, L2, main, L1) are Conv3×3(64→64). Consider D=128 in earlier refinenet steps if representation is insufficient. Tradeoff: 4× more MACs per RCU but richer fusion.

Start with D=64 flat. Ablate if ablation shows bottleneck (diagnostic: measure performance when specific branches are zeroed out).

### 9.2 Channel Reduction Point: Resolved

**Decided: reduce at output_conv1 (stride 2, 64→32, no activation).** This directly parallels DPT's output_conv1 (D→D/2 at stride 2, no activation). 64-dim through both refinenet merges preserves information, then 32-dim in the prediction head. output_conv2 (Conv3×3(32→32)+ReLU+Conv1×1(32→1)) matches DPT's two-step prediction exactly.

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

v15's dense training loss contributes ~25× more gradient terms than v14's total. The conv layers (RCU blocks, output_conv1, output_conv2) receive dense gradients at every spatial position — not scattered at K query centers — enabling proper learning of depth boundary sharpness.

### 10.4 Local→Global→Local Design

v15 restores the principled information flow:

1. **ConvNeXt (Local→Global):** Hierarchical encoder aggregates local texture into global scene features across 4 scales (L1–L4).
2. **DPT Global FPN (Global):** Top-down FPN fuses L4+L3 into a stride-8 representation with global context at every spatial position. L2 projected to D=64 for local fusion.
3. **Per-Query Progressive Refinenet (Global→Local):** After global context is encoded in `stride_8_map`, each query continues the DPT hierarchy locally: refinenet2 fuses L2 (stride 8), refinenet1 fuses L1 (stride 4), then the head produces depth at stride 1. All 4 encoder levels are used progressively.

This mirrors successful patterns: U-Net (encoder↓ + decoder↑), DPT (ViT global + reassemble local), PointNet++ (ball queries local → hierarchical global → interpolation local). v15 implements the same principle for sparse query depth — the full DPT refinenet hierarchy is preserved, split at stride 8 between global (dense) and local (per-query) execution.

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

**Per-query local path (valid conv, progressive refinenet, inference only):**

MACs per query = sum over all valid-conv layers (no padding, spatial size shrinks):

| Component | Input→Output spatial | MACs per query |
|---|---|---:|
| **refinenet2 (stride 8)** | | |
| conv_s8_1: Conv3×3(64→64) valid | 9×9 → 7×7 = 49 pos | **1.81M** |
| conv_s8_2: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| rcu_l2_1: Conv3×3(64→64) valid | 9×9 → 7×7 = 49 pos | **1.81M** |
| rcu_l2_2: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| merge (add) + ×2 + crop 7×7 | 5×5 → 7×7 | ~0 |
| rn2_out: Conv1×1(64→64) | 7×7 = 49 pos | **0.20M** |
| *Subtotal refinenet2* | | *5.66M* |
| **refinenet1 (stride 4)** | | |
| rn1_conv1: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| rn1_conv2: Conv3×3(64→64) valid | 5×5 → 3×3 = 9 pos | **0.33M** |
| proj_l1: Conv1×1(96→64) | 7×7 = 49 pos | **0.30M** |
| rcu_l1_1: Conv3×3(64→64) valid | 7×7 → 5×5 = 25 pos | **0.92M** |
| rcu_l1_2: Conv3×3(64→64) valid | 5×5 → 3×3 = 9 pos | **0.33M** |
| merge (add) + ×2 + crop 5×5 | 3×3 → 5×5 | ~0 |
| rn1_out: Conv1×1(64→64) | 5×5 = 25 pos | **0.10M** |
| *Subtotal refinenet1* | | *2.90M* |
| **head (stride 2 → depth)** | | |
| output_conv1: Conv3×3(64→32) valid | 5×5 → 3×3 = 9 pos | **0.17M** |
| ×2 + crop 3×3 | 3×3 → 3×3 | ~0 |
| output_conv2 Conv3×3(32→32) valid | 3×3 → 1×1 = 1 pos | **0.01M** |
| output_conv2 Conv1×1(32→1) | 1×1 = 1 pos | **~0M** |
| *Subtotal head* | | *0.18M* |
| **Per-query total** | | **~8.7M** |

Largest inputs are 9×9 = 81 positions (s8 and L2 patches). L1 patches are only 7×7 = 49 positions (down from 11×11 in previous design). The progressive refinenet structure concentrates compute at stride 8 (5.66M, 65% of per-query cost) where patch sizes are largest. Conv1×1 output projections (rn2_out, rn1_out) match DPT's FeatureFusionBlock pattern at negligible cost (+0.30M). Channel reduction at output_conv1 (64→32) makes the head negligible (0.18M). Two-step output_conv2 (Conv3×3+Conv1×1) matches DPT exactly — costs only 0.01M per query at 1 position. **13% cheaper per query** than previous design (8.7M vs 10.0M).

**Training dense path MACs (one full image forward pass):**

| Component | Resolution | MACs |
|---|---|---:|
| Global DPT (refinenet4+3) | — | **241M** |
| **refinenet2 (stride 8):** | | |
| conv_s8_1+s8_2 (padded) | 52×68 = 3,536 | 2×3,536×64²×9 = **262M** |
| rcu_l2_1+rcu_l2_2 (padded) | 52×68 = 3,536 | 2×3,536×64²×9 = **262M** |
| rn2_out Conv1×1(64→64) | 104×136 = 14,144 | 14,144×64×64 = **58M** |
| **refinenet1 (stride 4):** | | |
| rn1_conv1+rn1_conv2 (padded) | 104×136 = 14,144 | 2×14,144×64²×9 = **1,048M** |
| proj_l1 (padded) | 104×136 = 14,144 | 14,144×64×96 = **87M** |
| rcu_l1_1+rcu_l1_2 (padded) | 104×136 = 14,144 | 2×14,144×64²×9 = **1,048M** |
| rn1_out Conv1×1(64→64) | 208×272 = 56,576 | 56,576×64×64 = **232M** |
| **head:** | | |
| output_conv1 (padded, 64→32) | 208×272 = 56,576 | 56,576×32×64×9 = **1,043M** |
| output_conv2 Conv3×3(32→32, padded) | 416×544 = 226,304 | 226,304×32×32×9 = **2,086M** |
| output_conv2 Conv1×1(32→1) | 416×544 = 226,304 | 226,304×1×32 = **7M** |
| **Training total** | | **~6.4G** |

Training cost now **matches full DPT dense** at D=64 (~6.4G). The two-step output_conv2 (Conv3×3(32→32) + Conv1×1(32→1)) at stride 1 costs 2,093M — identical to DPT's head. This is the dominant cost at stride 1 (226K positions). The architecture is structurally identical to a full DPT dense decoder during training.

### 11.2 Total v15 Decoder MACs vs K (Inference)

Per-query cost: **~8.7M** (progressive refinenet valid-conv path with Conv1×1 output projections, channel reduction at s2).

| K | Global path | Per-query | **v15 decoder** |
|---:|---:|---:|---:|
| 32 | 241M | 278M | **519M** |
| 64 | 241M | 557M | **798M** |
| 128 | 241M | 1,114M | **1,355M** |

### 11.3 Model-Level MACs Comparison

| Model | Encoder | Decoder | **Total** | NYU AbsRel |
|---|---:|---:|---:|---:|
| DAv2-S (ViT-S + DPT D=64, dense, 518×518) | ~38G | ~11.6G | **~49G** | ~0.048 |
| Full DPT D=64, ConvNeXt, 416×544 (dense, with head) | ~10G | ~6.4G | **~16.4G** | — |
| v14 (ConvNeXt + MSDA + canvas, K=256) | ~10G | ~8G | **~18G** | 0.240+ |
| **v15 (K=64, inference)** | ~10G | ~0.8G | **~10.8G** | TBD |
| **v15 (K=128, inference)** | ~10G | ~1.4G | **~11.4G** | TBD |
| **v15 (training, dense)** | ~10G | ~6.4G | **~16.4G** | — |

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
| *Subtotal (4 refinenets, to stride 2)* | | *~3,219M* |
| **DPT head — output_conv1:** Conv3×3(64→32) | 208×272=56,576 (stride 2) | **1,043M** |
| bilinear ×2 to stride 1 | — | ~0 |
| **DPT head — output_conv2:** Conv3×3(32→32) | 416×544=226,304 (stride 1) | **2,086M** |
| **DPT head — output_conv2:** Conv1×1(32→1) | 416×544=226,304 (stride 1) | **7M** |
| *Subtotal (DPT head, stride 2 → prediction)* | | *~3,136M* |
| **Full DPT dense decoder (refinenets + head)** | | **~6,355M** |

`refinenet1` at stride-4 (14,144 positions) alone costs **2,082M** — more than our entire v15 decoder at K=64 (798M). The DPT head (output_conv1 + output_conv2) adds another **3,136M** at stride 2→1 — nearly as much as the 4 refinenets combined. v15 training pays all of this (identical to DPT). The inference advantage: running refinenet2+1+head per-query (valid conv, 8.7M/query) instead of densely, saving all stride-4/2/1 dense costs for K ≤ 703.

### 11.4 D-Dimension Comparison (Global Path Only)

All rows assume full RCU FeatureFusionBlock. `proj_s8` = Conv1×1(D→64) needed at D>64 for the per-query local path.

| D | Projections | refinenet4 | refinenet3 | proj_s8 | **Global total** | v15 K=64 | v15 K=128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **64** | 76M | 20M | 145M | 0M | **241M** | **798M** | **1,355M** |
| 128 | 152M | 80M | 578M | 29M | **839M** | **1,396M** | **1,953M** |
| 192 | 228M | 179M | 1,300M | 43M | **1,750M** | **2,307M** | **2,864M** |
| 256 | 304M | 318M | 2,317M | 58M | **2,997M** | **3,554M** | **4,111M** |

At D=64, the global path (241M) is 30% of the total decoder at K=64 (798M) — still dominated by per-query local path. At D=256, the global path (2,997M) dominates. D=64 keeps the global path inexpensive relative to the per-query work.

### 11.5 v15 (D=64) vs Full DPT Dense (D=64) — Decoder MACs

Full DPT dense decoder at D=64, 416×544 (4 refinenets + projections + head) = **6,355M** (K-independent).

**Break-even at K ≈ 703** (241 + 8.7K = 6,355 → K = 6,114 / 8.7 = **703**). At 8.7M/query, v15 inference is faster than full DPT dense for all K ≤ 703 — covering every practical inference scenario by a wide margin.

| K | v15 decoder (8.7M/query) | DPT dense decoder (D=64) | **Decoder speedup** |
|---:|---:|---:|---:|
| 16 | 241+139 = **380M** | 6,355M | **16.7×** |
| 32 | 241+278 = **519M** | 6,355M | **12.2×** |
| 64 | 241+557 = **798M** | 6,355M | **8.0×** |
| 100 | 241+870 = **1,111M** | 6,355M | **5.7×** |
| 128 | 241+1,114 = **1,355M** | 6,355M | **4.7×** |
| 256 | 241+2,227 = **2,468M** | 6,355M | **2.6×** |
| 512 | 241+4,454 = **4,695M** | 6,355M | **1.4×** |
| **703** | 241+6,116 = **6,357M** | 6,355M | **≈1×** |

**Structural comparison — training path (both D=64, padded conv):**

| Processing step | DPT dense (D=64) | v15 (D=64) training | Same? |
|---|---|---|---|
| Conv1×1 projections L4/L3/L2 | ✓ | ✓ | **identical** |
| refinenet4 (L4 → stride 16) | FeatureFusionBlock RCU2 | **identical** | **identical** |
| refinenet3 (stride 16+L3 → stride 8) | FeatureFusionBlock RCU1+RCU2 | **identical** | **identical** |
| refinenet2 (stride 8+L2 → stride 4) | RCU1(f2) + add + RCU2 (**578M**) + Conv1×1 (**58M**) | s8 RCU (**262M**) + L2 RCU (**262M**) → add → ×2 → Conv1×1 (**58M**) | **structurally analogous** — both apply RCU + merge + ×2 + Conv1×1. Cost: 582M vs 636M. |
| refinenet1 (stride 4+L1 → stride 2) | RCU1(L1) + add + RCU2 (**2,082M**) + Conv1×1 (**231M**) | main RCU (**1,048M**) + proj_l1+L1 RCU (**87M+1,048M**) → add → ×2 → Conv1×1 (**232M**) | **structurally analogous** — both fuse L1 at stride 4 with RCU + Conv1×1 output. Cost: 2,415M vs 2,313M. |
| stride 2: channel reduction | output_conv1: Conv3×3(64→32, **1,043M**), no activation → ×2 | output_conv1: Conv3×3(64→32, **1,043M**), no activation → ×2 | **identical** — same layer, same function, no activation |
| stride 1: prediction | output_conv2: Conv3×3(32→32) + ReLU + Conv1×1(32→1) (**2,093M**) | output_conv2: Conv3×3(32→32) + ReLU + Conv1×1(32→1) (**2,093M**) | **identical** — same two-step structure, same cost |

**Key structural insight:** v15's training path IS a full DPT dense decoder — structurally identical at every stage. Each refinenet has the full DPT FeatureFusionBlock structure: RCU on both branches → merge → ×2 → Conv1×1 output projection. The head matches DPT exactly: output_conv1 Conv3×3(64→32, no activation) → ×2 → output_conv2 Conv3×3(32→32)+ReLU+Conv1×1(32→1). The only structural difference is RCU placement: v15 applies RCU to both branches separately before merge (vs DPT's RCU-on-skip then RCU-on-fused). Training cost is **~6.4G** — matching full DPT dense at D=64. The inference advantage comes entirely from executing refinenet2+1+head per-query (valid conv, 8.7M/query) instead of densely.
