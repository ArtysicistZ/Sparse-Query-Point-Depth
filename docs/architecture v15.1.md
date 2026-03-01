# SPD v15.1 Architecture — DINOv2 ViT-S + DPT Decoder

Author: Claude
Date: 2026-03-01
Version: v15.1

---

## 1. Problem Statement

**Input:** RGB image $I \in \mathbb{R}^{H \times W \times 3}$ and query pixel coordinates $Q = \{(u_i, v_i)\}_{i=1}^{K}$.

**Output:** Metric depth $\hat{d}_i$ at each queried pixel.

**Core idea:** The encoder runs once per image. The decoder is split into a **global path** (dense, $O(HW)$, runs once) and a **local path** (per-query, $O(K)$ at inference). At training time, the local path runs densely with padded convolutions on full feature maps — identical weights, two execution modes.

**Baseline:** DAv2-S (DINOv2 ViT-S + full DPT decoder).

---

## 2. Architecture Overview

```
RGB Image [H × W × 3]
  │
  ▼
DINOv2 ViT-S/14  (pretrained, fine-tuned)
  Features tapped at layers {2, 5, 8, 11}
  All: [B, 384, H/14, W/14]
  │
  ▼
Projection Neck  (Conv1×1 + LN + spatial resize per level)
  L1: [B, D, 4H/14, 4W/14]     ≈ stride 3.5   ← local refinenet1
  L2: [B, D, 2H/14, 2W/14]     ≈ stride 7     ← local refinenet2
  L3: [B, D,  H/14,  W/14]     = stride 14     ← global refinenet3
  L4: [B, D,  H/28,  W/28]     ≈ stride 28     ← global refinenet4
  │
  ▼ ────────────────────────────────────────────
  GLOBAL PATH  (dense, once per image)
  │
  refinenet4:  RCU(L4) → ×2 bilinear → Conv1×1
  refinenet3:  RCU(L3) + add → RCU → ×2 bilinear → Conv1×1
               = stride_8_map  [B, D, 2H/14, 2W/14]
  │
  ▼ ────────────────────────────────────────────
  LOCAL PATH
  (training: dense padded convs on full maps)
  (inference: per-query valid convs on small patches)
  │
  refinenet2:  RCU(stride_8_map) + RCU(L2) → add → ×2 → Conv1×1
  refinenet1:  RCU(rn2_out) + RCU(L1) → add → ×2 → Conv1×1
  head:        Conv3×3(D→D/2) → ×2 → Conv3×3(D/2→D/2) → ReLU → Conv1×1(D/2→1)
               → softplus → depth
```

Where $D = 64$.

---

## 3. Encoder: DINOv2 ViT-S/14

| Parameter | Value |
|---|---|
| Architecture | Vision Transformer Small (ViT-S/14) |
| Patch size | $14 \times 14$ |
| Embedding dim | 384 |
| Transformer blocks | 12, each with 6 attention heads ($384 / 6 = 64$ dim/head) |
| Pretraining | DINOv2 self-supervised (DINO + iBOT) on LVD-142M — zero depth labels |
| Parameters | ~22M |

### 3.1 Feature Extraction

DINOv2 is a plain (non-hierarchical) ViT. Given input $I \in \mathbb{R}^{B \times 3 \times H \times W}$, the image is split into $14 \times 14$ patches, yielding a token grid of size:

$$h = \lfloor H / 14 \rfloor, \quad w = \lfloor W / 14 \rfloor$$

All transformer blocks operate at the same spatial resolution $h \times w$ with uniform channel dimension 384. We tap intermediate features at blocks $\{2, 5, 8, 11\}$ (0-indexed), creating a pseudo-hierarchy:

$$F_l \in \mathbb{R}^{B \times 384 \times h \times w}, \quad l \in \{L1, L2, L3, L4\}$$

Unlike ConvNeXt whose stages have naturally decreasing spatial resolution, ViT features are all spatially identical. The projection neck (Section 4) creates the multi-scale pyramid via learned spatial resizing.

### 3.2 Concrete Dimensions at $350 \times 476$

$$h = 350 / 14 = 25, \quad w = 476 / 14 = 34$$

All four feature maps: $[B, 384, 25, 34]$.

---

## 4. Projection Neck

The neck projects all four levels from encoder dimension 384 to decoder dimension $D = 64$, and resizes each to a different spatial scale to create the multi-resolution pyramid that the DPT decoder expects.

### 4.1 Per-Level Projection

For each level $l \in \{L1, L2, L3, L4\}$:

$$\hat{F}_l = \text{LN}\!\left(\text{Conv}_{1 \times 1}^{384 \to D}(F_l)\right)$$

where $\text{LN}$ is LayerNorm applied channel-wise.

### 4.2 Spatial Resize

Each projected feature is resized to approximate the standard DPT multi-scale strides:

| Level | Resize Operation | Output Size ($350 \times 476$ input) | Effective Stride |
|-------|-----------------|--------------------------------------|-----------------|
| L1 | $\text{ConvTranspose}_{4 \times 4}^{D \to D}(\text{stride}=4)$ | $[B, 64, 100, 136]$ | ~3.5 |
| L2 | $\text{ConvTranspose}_{2 \times 2}^{D \to D}(\text{stride}=2)$ | $[B, 64, 50, 68]$ | ~7 |
| L3 | Identity | $[B, 64, 25, 34]$ | 14 |
| L4 | $\text{Conv}_{3 \times 3}^{D \to D}(\text{stride}=2, \text{pad}=1)$ | $[B, 64, 13, 17]$ | ~28 |

These approximate the standard DPT strides $\{4, 8, 16, 32\}$ from a hierarchical backbone.

### 4.3 Parameters

| Component | Count |
|---|---|
| 4× Conv1×1(384→64) | 98,560 |
| 4× LayerNorm(64) | 512 |
| ConvTranspose(64→64, k=4, s=4) | 65,600 |
| ConvTranspose(64→64, k=2, s=2) | 16,448 |
| Conv2d(64→64, k=3, s=2, p=1) | 36,928 |
| **Total Neck** | **218,048** |

---

## 5. Global Path: DPT FPN Fusion

The global path fuses deep features (L4, L3) into a stride-8 dense feature map, following the standard DPT/RefineNet top-down fusion pattern. It runs once per image.

### 5.1 Residual Convolutional Unit (RCU)

Every refinenet block uses RCUs as its core building block. An RCU is a two-layer residual conv block:

$$\text{RCU}(x) = x + \text{Conv}_{3 \times 3}\!\left(\text{GELU}\!\left(\text{Conv}_{3 \times 3}\!\left(\text{GELU}(x)\right)\right)\right)$$

Both convolutions are $D \to D$ with $3 \times 3$ kernels (padded at training, valid at inference). Parameters per RCU: $2 \times (D^2 \cdot 9 + D) = 73{,}856$.

### 5.2 Refinenet4 (L4 → L3 resolution)

$$r_4 = \text{Conv}_{1 \times 1}\!\left(\text{Upsample}_{\times 2}\!\left(\text{RCU}(L4)\right)\right)$$

Upsamples L4 to match L3 spatial resolution via bilinear interpolation, then a $1 \times 1$ projection.

### 5.3 Refinenet3 (L3 + r4 → stride-8 map)

$$r_3 = \text{Conv}_{1 \times 1}\!\left(\text{Upsample}_{\times 2}\!\left(\text{RCU}\!\left(\text{RCU}(L3) + r_4\right)\right)\right)$$

The L3 features are refined by an RCU, summed with $r_4$, refined again by a merge RCU, upsampled to L2 resolution, and projected. The output is the **stride-8 map**:

$$\text{stride\_8\_map} = r_3 \in \mathbb{R}^{B \times D \times 2h \times 2w}$$

At $350 \times 476$: $[B, 64, 50, 68]$.

### 5.4 Parameters

| Component | Count |
|---|---|
| rcu_L4 (RCU) | 73,856 |
| rcu_L4_out (Conv1×1) | 4,160 |
| rcu_L3 (RCU) | 73,856 |
| rcu_L3_merge (RCU) | 73,856 |
| rcu_L3_out (Conv1×1) | 4,160 |
| **Total Global** | **229,888** |

---

## 6. Local Path: Progressive RefineNet

The local path continues the DPT refinenet hierarchy below stride 8. It fuses L2 at stride 8→4, then L1 at stride 4→2, then predicts depth at stride 1. **The same weights** serve two execution modes:

- **Training:** Padded $3 \times 3$ convolutions on full feature maps → dense $H \times W$ depth prediction.
- **Inference:** Valid $3 \times 3$ convolutions on small patches extracted per query via `grid_sample`.

### 6.1 Refinenet2 (stride 8 → stride 4)

$$\text{rn2} = \text{Conv}_{1 \times 1}\!\left(\text{Upsample}_{\times 2}\!\left(\text{RCU}(\text{stride\_8\_map}) + \text{RCU}(L2)\right)\right)$$

The stride-8 map and L2 features are each refined by an RCU, summed, upsampled $\times 2$, and projected.

### 6.2 Refinenet1 (stride 4 → stride 2)

$$\text{rn1} = \text{Conv}_{1 \times 1}\!\left(\text{Upsample}_{\times 2}\!\left(\text{RCU}(\text{rn2}) + \text{RCU}(L1)\right)\right)$$

Same pattern: refine, fuse L1, upsample, project.

### 6.3 Depth Head (stride 2 → depth)

Channel reduction and two-step prediction, matching the DPT head pattern:

$$o_1 = \text{Upsample}_{\times 2}\!\left(\text{Conv}_{3 \times 3}^{D \to D/2}(\text{rn1})\right)$$

$$\hat{d} = \text{softplus}\!\left(\text{Conv}_{1 \times 1}^{D/2 \to 1}\!\left(\text{ReLU}\!\left(\text{Conv}_{3 \times 3}^{D/2 \to D/2}(o_1)\right)\right)\right)$$

The softplus activation ensures $\hat{d} > 0$ with gradient everywhere:

$$\text{softplus}(x) = \ln(1 + e^x)$$

At initialization ($x \approx 0$), predictions start at $\ln 2 \approx 0.69$ m.

**Training output:** Dense depth map interpolated to image size:

$$\hat{D} = \text{Resize}_{H \times W}\!\left(\hat{d}\right) \in \mathbb{R}^{B \times 1 \times H \times W}$$

**Inference output:** Scalar depth per query:

$$\hat{d}_i \in \mathbb{R}, \quad i = 1, \dots, K$$

### 6.4 Parameters

| Component | Count |
|---|---|
| s8_rcu (RCU) | 73,856 |
| l2_rcu (RCU) | 73,856 |
| rn2_out (Conv1×1, 64→64) | 4,160 |
| s4_rcu (RCU) | 73,856 |
| l1_rcu (RCU) | 73,856 |
| rn1_out (Conv1×1, 64→64) | 4,160 |
| output_conv1 (Conv3×3, 64→32) | 18,464 |
| output_conv2a (Conv3×3, 32→32) | 9,248 |
| output_conv2b (Conv1×1, 32→1) | 33 |
| **Total Local** | **331,489** |

---

## 7. Inference: Per-Query Valid Convolution

At inference, the local path processes each query independently by extracting small patches from precomputed feature maps and applying valid (unpadded) convolutions. The receptive field shrinks at each layer, ultimately producing a single depth value per query.

### 7.1 Patch Extraction

For query pixel $(u, v)$, we compute feature-space coordinates at each stride and extract patches via bilinear `grid_sample`:

| Source Map | Patch Size | Grid Sample From |
|------------|-----------|-----------------|
| stride_8_map | $9 \times 9$ | Global path output |
| L2 | $9 \times 9$ | Neck output |
| L1 | $7 \times 7$ | Neck output |

### 7.2 Valid Conv Spatial Trace

Each valid $3 \times 3$ convolution reduces spatial extent by 2 (1 pixel per side). The bilinear $\times 2$ upsample doubles the size. A **recenter crop** realigns the upsampled patch to the query's sub-pixel position before the next stage.

| Step | Operation | Spatial Size |
|------|-----------|-------------|
| **Refinenet2** | | |
| | Sample stride_8_map | $9 \times 9$ |
| | s8_rcu: 2× valid Conv3×3 | $5 \times 5$ |
| | Sample L2 | $9 \times 9$ |
| | l2_rcu: 2× valid Conv3×3 | $5 \times 5$ |
| | Add | $5 \times 5$ |
| | Upsample ×2 | $10 \times 10$ |
| | Recenter crop | $7 \times 7$ |
| | Conv1×1 (rn2_out) | $7 \times 7$ |
| **Refinenet1** | | |
| | s4_rcu: 2× valid Conv3×3 | $3 \times 3$ |
| | Sample L1 | $7 \times 7$ |
| | l1_rcu: 2× valid Conv3×3 | $3 \times 3$ |
| | Add | $3 \times 3$ |
| | Upsample ×2 | $6 \times 6$ |
| | Recenter crop | $5 \times 5$ |
| | Conv1×1 (rn1_out) | $5 \times 5$ |
| **Head** | | |
| | Valid Conv3×3 (output_conv1, 64→32) | $3 \times 3$ |
| | Upsample ×2 | $6 \times 6$ |
| | Recenter crop | $3 \times 3$ |
| | Valid Conv3×3 (output_conv2a, 32→32) + ReLU | $1 \times 1$ |
| | Conv1×1 (output_conv2b, 32→1) | $1 \times 1$ |
| | Softplus | **scalar depth** |

### 7.3 Recenter Crop

After each bilinear $\times 2$ upsample, the patch center may not align exactly with the query's pixel. The recenter crop selects the correct sub-patch based on the query's sub-stride offset:

$$\text{offset}_x = \begin{cases} 1 & \text{if } (u \bmod s) \geq s/2 \\ 0 & \text{otherwise} \end{cases}$$

where $s$ is the current stride. This shifts the crop window by 1 pixel when the query lies in the second half of a stride cell, ensuring the final $1 \times 1$ output is centered on the query pixel.

---

## 8. Training

### 8.1 Dense Execution

At training time, the local path operates on full feature maps with padded convolutions (standard `nn.Conv2d` with `padding=1`). The output is a dense depth map $\hat{D} \in \mathbb{R}^{B \times 1 \times H \times W}$, resized from the decoder's native resolution to image size via bilinear interpolation.

No query sampling occurs during training — every valid pixel provides supervision.

### 8.2 Loss Function: Scale-Invariant Logarithmic Error (SILog)

Given predicted depth $\hat{D}$ and ground truth $D^*$, define the log-depth residual at each valid pixel $i$:

$$\delta_i = \ln \hat{d}_i - \ln d_i^*$$

where validity is defined by $d_i^* > 0$. The per-image SILog loss is:

$$\mathcal{L}_{\text{SILog}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} \delta_i^2 - \lambda_{\text{var}} \left(\frac{1}{N}\sum_{i=1}^{N} \delta_i\right)^2 + \epsilon}$$

The first term penalizes all errors equally in log-space. The second term ($\lambda_{\text{var}}$-weighted) subtracts the squared mean, suppressing the global scale component and emphasizing structural (relative) accuracy. $\epsilon = 10^{-8}$ for numerical stability.

**$\lambda_{\text{var}} = 0.85$** follows the standard established by BTS, AdaBins, and PixelFormer for from-scratch metric depth training. Lower values (e.g., $\lambda_{\text{var}} = 0.50$ used by DAv2) are appropriate for fine-tuning where scale is already calibrated.

### 8.3 Optimizer and Schedule

| Parameter | Value |
|---|---|
| Optimizer | AdamW, weight decay $0.01$ |
| Encoder LR | $5 \times 10^{-6}$ |
| Decoder LR | $1 \times 10^{-4}$ |
| Warmup | Linear, 500 steps, start factor $0.01$ |
| Schedule | Cosine annealing after warmup, $\eta_{\min} = 10^{-6}$ |
| AMP | bfloat16 |
| Batch size | 2 |
| Gradient clipping | Max norm $1.0$ |
| Epochs | 10 (237,920 total steps) |

The encoder LR is set to $1/20\times$ the decoder LR. DINOv2's self-supervised features are already highly depth-relevant; higher encoder LR causes feature drift that destabilizes the decoder's scale mapping.

### 8.4 Dataset

| Parameter | Value |
|---|---|
| Dataset | NYU Depth V2 |
| Train split | 47,584 images |
| Val split | 654 images |
| Resolution | $350 \times 476$ |
| Depth format | Float32 meters, range $[0, 10]$ m |
| Augmentation | BTS-style (rotation, gamma, brightness, color jitter, horizontal flip) |

---

## 9. Parameter Summary

| Component | Parameters |
|---|---|
| **Encoder** (DINOv2 ViT-S/14) | ~22,000,000 |
| **Projection Neck** | 218,048 |
| **Global DPT** (refinenet4 + refinenet3) | 229,888 |
| **Local DPT** (refinenet2 + refinenet1 + head) | 331,489 |
| **Total** | **~22,779,425** |

Decoder total (neck + global + local): **779,425** (~0.78M).

---

## 10. Computational Cost: Dense vs Sparse

All MACs computed at input resolution $350 \times 476$, with $D = 64$, using the formula:

$$\text{MACs}_{\text{Conv}} = C_{\text{out}} \times C_{\text{in}} \times k^2 \times H_{\text{out}} \times W_{\text{out}}$$

### 10.1 Encoder

DINOv2 ViT-S/14 tokenizes into $h \times w = 25 \times 34 = 850$ patches. Per transformer block:

| Operation | Formula | MACs |
|---|---|---|
| QKV projection | $N \times d \times 3d$ | 376.0M |
| Attention scores $QK^\top$ | $n_{\text{heads}} \times N \times d_h \times N$ | 277.4M |
| Attention weighted sum $AV$ | $n_{\text{heads}} \times N \times d_h \times N$ | 277.4M |
| Output projection | $N \times d \times d$ | 125.3M |
| FFN up ($d \to 4d$) | $N \times d \times 4d$ | 501.4M |
| FFN down ($4d \to d$) | $N \times 4d \times d$ | 501.4M |
| **Per block** | | **2,058.9M** |

$$\text{Encoder total} = \underbrace{192\text{M}}_{\text{patch embed}} + 12 \times 2{,}059\text{M} = \mathbf{24.9\text{G}}$$

### 10.2 Neck

| Operation | Map Size | MACs |
|---|---|---|
| 4× Conv1×1 ($384 \to 64$) | $25 \times 34$ | 83.6M |
| ConvTranspose ($64 \to 64$, $k\!=\!4$, $s\!=\!4$) for L1 | $25 \times 34 \to 100 \times 136$ | 55.7M |
| ConvTranspose ($64 \to 64$, $k\!=\!2$, $s\!=\!2$) for L2 | $25 \times 34 \to 50 \times 68$ | 13.9M |
| Conv2d ($64 \to 64$, $k\!=\!3$, $s\!=\!2$) for L4 | $25 \times 34 \to 13 \times 17$ | 8.1M |
| **Neck total** | | **161.3M** |

### 10.3 Global DPT

| Operation | Map Size | MACs |
|---|---|---|
| rcu_L4 (2× Conv3×3) | $13 \times 17$ | 16.3M |
| rcu_L4_out (Conv1×1) | $25 \times 34$ | 3.5M |
| rcu_L3 (2× Conv3×3) | $25 \times 34$ | 62.7M |
| rcu_L3_merge (2× Conv3×3) | $25 \times 34$ | 62.7M |
| rcu_L3_out (Conv1×1) | $50 \times 68$ | 13.9M |
| Upsample L4→L3 ($D\!=\!64$) | $25 \times 34$ | 0.2M |
| Upsample L3→L2 ($D\!=\!64$) | $50 \times 68$ | 0.9M |
| **Global total** | | **160.2M** |

### 10.4 Shared Cost (Encoder + Neck + Global)

$$C_{\text{shared}} = 24.9\text{G} + 0.16\text{G} + 0.16\text{G} = \mathbf{25.22\text{G}}$$

This cost is identical for dense and sparse inference — both execute the encoder, neck, and global path once per image.

### 10.5 Local Path: Dense vs Sparse

The local path is where dense and sparse diverge. Dense processes full feature maps with padded convolutions; sparse processes small per-query patches with valid convolutions.

Feature map sizes at each stride (dense path):

| Stage | Map Size | Pixels |
|---|---|---|
| Stride 8 (stride\_8\_map, L2) | $50 \times 68$ | 3,400 |
| Stride 4 (after rn2 ×2) | $100 \times 136$ | 13,600 |
| Stride 2 (after rn1 ×2) | $200 \times 272$ | 54,400 |
| Stride 1 (after head ×2) | $400 \times 544$ | 217,600 |

Detailed MACs comparison — dense (full map, padded) vs sparse (per query, valid):

| Layer | Dense Map | Dense MACs | Sparse Patch | Sparse MACs |
|---|---|---|---|---|
| **Refinenet2** | | | | |
| s8\_rcu (2× Conv3×3, $64 \to 64$) | $50 \times 68$ | 250.7M | $9 \to 7 \to 5$ | 2.73M |
| l2\_rcu (2× Conv3×3, $64 \to 64$) | $50 \times 68$ | 250.7M | $9 \to 7 \to 5$ | 2.73M |
| rn2\_out (Conv1×1, $64 \to 64$) | $100 \times 136$ | 55.7M | $7 \times 7$ | 0.20M |
| **Refinenet1** | | | | |
| s4\_rcu (2× Conv3×3, $64 \to 64$) | $100 \times 136$ | 1,002.7M | $7 \to 5 \to 3$ | 1.25M |
| l1\_rcu (2× Conv3×3, $64 \to 64$) | $100 \times 136$ | 1,002.7M | $7 \to 5 \to 3$ | 1.25M |
| rn1\_out (Conv1×1, $64 \to 64$) | $200 \times 272$ | 222.8M | $5 \times 5$ | 0.10M |
| **Head** | | | | |
| output\_conv1 (Conv3×3, $64 \to 32$) | $200 \times 272$ | 1,002.7M | $5 \to 3$ | 0.17M |
| output\_conv2a (Conv3×3, $32 \to 32$) | $400 \times 544$ | 2,005.4M | $3 \to 1$ | 0.01M |
| output\_conv2b (Conv1×1, $32 \to 1$) | $400 \times 544$ | 7.0M | $1 \times 1$ | ~0M |
| **Local total** | | **5,800M** | | **8.44M** |

Bilinear upsampling costs $\sim\!4$ MACs per output pixel per channel (2D interpolation weights):

| Upsample | Dense Output | Dense MACs | Sparse Output | Sparse MACs |
|---|---|---|---|---|
| rn2 ×2 ($D\!=\!64$) | $100 \times 136$ | 3.5M | $10 \times 10$ | 25.6K |
| rn1 ×2 ($D\!=\!64$) | $200 \times 272$ | 13.9M | $6 \times 6$ | 9.2K |
| head ×2 ($C\!=\!32$) | $400 \times 544$ | 27.9M | $6 \times 6$ | 4.6K |
| Resize to $H \times W$ ($C\!=\!1$) | $350 \times 476$ | 0.7M | — | — |
| **Upsample total** | | **46.0M** | | **39.4K** |

Including upsamples: dense local = **5,846M**, sparse per query = **8.48M**. The upsamples contribute $<\!1\%$ in both cases — convolutions dominate overwhelmingly.

The dense head (output\_conv2a at stride 1) alone costs 2.0G — more than all other local layers combined. At sparse inference this collapses to 0.01M because the valid conv operates on a $3 \times 3 \to 1 \times 1$ patch.

### 10.6 Total MACs Comparison

$$C_{\text{dense}} = C_{\text{shared}} + C_{\text{local,dense}} = 25.22\text{G} + 5.85\text{G} = \mathbf{31.07\text{G}}$$

$$C_{\text{sparse}}(K) = C_{\text{shared}} + K \times C_{\text{local,query}} = 25.22\text{G} + K \times 8.48\text{M}$$

| $K$ | Decoder MACs | Total MACs | vs Dense Total | vs Dense Decoder |
|---|---|---|---|---|
| 64 | 0.54G | 25.76G | 82.9% | 9.3% |
| 128 | 1.09G | 26.31G | 84.7% | 18.6% |
| 256 | 2.17G | 27.39G | 88.2% | 37.1% |
| 512 | 4.34G | 29.56G | 95.1% | 74.2% |
| **689** | **5.85G** | **31.07G** | **100%** | **100%** |
| 1024 | 8.68G | 33.90G | 109.1% | 148.4% |

**Break-even point:**

$$K^* = \frac{C_{\text{local,dense}}}{C_{\text{local,query}}} = \frac{5{,}846\text{M}}{8.48\text{M}} \approx 689 \text{ queries}$$

### 10.7 Where the Encoder Bottleneck Comes From

At $350 \times 476$, the encoder accounts for $24.9 / 31.0 = 80\%$ of total dense MACs. This limits total savings: even $K = 1$ still costs $25.2 / 31.0 = 81\%$ of dense. The sparse architecture's advantage grows at higher resolution because:

- **Encoder:** $O(N^2)$ where $N = HW/p^2$ (attention is quadratic in patch count)
- **Dense decoder:** $O(HW)$ at each stride level, dominated by stride-1 and stride-2 layers
- **Sparse decoder:** $O(K)$, independent of image resolution

At $480 \times 640$ (DAv2 standard), $N = 1{,}530$ patches and the decoder maps grow $\sim\!3.2\times$, pushing the break-even to $K^* \approx 2{,}200$. At $1024 \times 1024$ and beyond, the dense decoder dominates total cost, and sparse inference with moderate $K$ yields substantial savings.

---

## 11. File Structure

| File | Component |
|---|---|
| `src/models/spd.py` | Top-level model, routes train/infer |
| `src/models/encoder_vits/vit_s.py` | DINOv2 ViT-S encoder |
| `src/models/encoder_vits/pyramid_neck.py` | Projection neck |
| `src/models/decoder/global_dpt.py` | Global path (refinenet4 + refinenet3) |
| `src/models/decoder/local_dpt.py` | Local path (refinenet2 + refinenet1 + head) |
| `src/models/decoder/rcu.py` | Residual Convolutional Unit |
| `src/config.py` | Hyperparameters |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Dense + sparse evaluation |
| `src/utils/losses.py` | SILog loss |
| `src/data/nyu_dataset.py` | NYU Depth V2 dataloader |
