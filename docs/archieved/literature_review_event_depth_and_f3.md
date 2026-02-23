# Literature Review: Event-Camera Depth Estimation and F^3 (Beginner-Friendly)

Author: Codex (for Prof. Chaudhari research planning)
Date: 2026-02-12
Scope: Foundations, key papers, method taxonomy, runtime bottlenecks, and implications for a sparse query-point depth predictor inspired by F^3.

## 1) What problem are we solving?

You want a model that takes raw event-camera data and returns depth only at user-selected pixels, not a full dense depth map. The target is:

- Keep accuracy close to dense methods.
- Reduce runtime substantially.
- Avoid running a heavy full-image depth decoder over all pixels.

That is a very good research direction because most current event-depth pipelines still pay a dense decoding cost even if you only need a few points.

## 2) Prerequisites (from zero to ready)

### 2.1 Event cameras in one page

A standard frame camera outputs full images at fixed FPS. An event camera outputs asynchronous events:

- Event tuple: `(x, y, t, p)`
- `(x, y)`: pixel
- `t`: timestamp (microsecond-level)
- `p`: polarity (`+` brightness increase, `-` decrease)

Important consequences:

- Very high temporal resolution
- High dynamic range
- Sparse data (most pixels do not fire at a given time)
- Data is not naturally an image tensor

### 2.2 Why depth from events is hard

Depth from a single sensor is usually ambiguous. For monocular event depth, scale and texture ambiguities still exist. You often need:

- Motion assumptions
- Multi-view constraints
- Priors learned from data
- Extra supervision (LiDAR, stereo, pseudo depth from RGB)

### 2.3 Core representations you will see

Event methods typically convert event streams into one of:

- Event frames (collapse time, keep polarity)
- Voxel grids (space + time bins)
- Time surfaces
- Learned representations (e.g., F^3)

Dense representations are easy for CNNs but expensive and often waste computation on empty voxels.

### 2.4 Dense vs sparse prediction

- Dense prediction: output depth for every pixel in `H x W`.
- Sparse/query prediction: output only for `Q` requested pixels.

When `Q << H*W`, sparse prediction should be much faster if architecture is designed correctly.

### 2.5 Geometry essentials you need before reading depth papers

You should be comfortable with four geometric ideas:

- **Pinhole camera projection**: 3D point projects to image pixel by focal length and depth.
- **Depth vs disparity**: for stereo, `disparity = f * b / depth` (larger disparity means closer point).
- **Optical flow**: apparent motion of pixels between two times; depends on camera/object motion + scene depth.
- **Scale ambiguity**: monocular depth can be correct up to unknown global scale.

Why this matters:

- Event-depth papers mix depth/disparity/flow language frequently.
- Stage-1 and stage-2 training losses are often written in disparity space.

### 2.6 Why event data often gets converted to grids

Most deep models expect regular tensors. Events are sparse asynchronous tuples, so papers build:

- frames (`H x W x channels`)
- voxel grids (`H x W x T`)
- learned fields (F^3-style channels over image plane)

The representation choice is often the real bottleneck for speed and robustness.

### 2.7 Common losses you will repeatedly see

- **BCE / focal loss**: for event presence prediction (future event voxel/classification style).
- **L1/L2 depth losses**: direct regression on depth/disparity.
- **SiLog loss**: scale-invariant depth error.
- **Gradient regularizers**: sharpen object boundaries by matching depth gradients.
- **Photometric/warping consistency losses**: self-supervised geometry constraints.

### 2.8 Why LiDAR supervision is tricky for event depth

LiDAR projected to camera image is sparse, especially at high resolution. This causes:

- weak supervision in many pixels,
- blurry boundaries if trained directly dense,
- need for pseudo depth pretraining or regularization.

This exact issue motivates the two-stage training strategy used in F^3 depth.

### 2.9 Runtime model you should use mentally

For dense depth decoders, cost usually scales with image tokens:

- CNN-like: roughly with `H*W*channels*layers`
- ViT-like: token-heavy, often even more expensive in practice

For sparse query decoders, target cost should scale with query count:

- `shared_cost(events)` + `query_cost(Q)`

Your project is about making `query_cost(Q)` dominant and lightweight.

## 3) Big picture of the literature

I group relevant work into six buckets.

### 3.1 Classical/self-supervised geometric event methods

Early influential direction: learn flow/depth/egomotion via photometric or warping constraints from events.

- EV-FlowNet style unsupervised event learning established that events can support geometry tasks without dense labels.
- Good geometric intuition, but often brittle and not always fast enough for modern high-res/high-rate use.

Representative:
- Zhu et al., CVPR 2019, "Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion".

### 3.2 Dense monocular event depth with recurrent models

A major step was learning dense depth directly from event streams using recurrent architectures.

- "Learning Monocular Dense Depth from Events" (3DV 2020) is key.
- Idea: recurrently aggregate temporal event information and predict dense depth.
- Improvement over feed-forward methods, but still dense output.

### 3.3 Multimodal transfer (events + images/frames)

Several methods improve event depth by borrowing supervision or features from image branch networks.

- RAM Net (RA-L 2021): asynchronous recurrent multimodal fusion.
- DTL (ICCV 2021): event-to-image branch transfers rich visual priors to event end-task branch.

These methods often improve accuracy but still do dense decoding.

### 3.4 Transformer-era event depth

Recent works move from CNN/RNN-only to transformer-based encoders and temporal recurrence modules.

- EReFormer (TCSVT 2024) emphasizes global spatial modeling + temporal recurrence.

Common pattern:
- Better accuracy and global context.
- Higher computational overhead than lightweight event-specific pipelines.

### 3.5 2025 wave: stronger transfer, self-supervision, and generalized event depth

Recent papers broaden training and transfer settings:

- On-device self-supervised event-only depth (CVPR 2025): focuses latency and deployment.
- Depth AnyEvent (ICCVW 2025): cross-domain transfer for event metric depth.
- Depth Any Event Stream (2025): explores what event streams can/cannot reveal about depth under broad conditions.

Trend:
- Better robustness/generalization.
- Still mostly dense output at inference.

### 3.6 F^3: representation-first speed and robustness

F^3 (2025) reframes the problem:

- Learn a predictive representation of past events sufficient for future events.
- Use a sparse event-native architecture (multi-resolution hash + permutation-invariant pooling + small CNN smoothing).
- Then plug this representation into downstream tasks (flow, segmentation, depth).

This is very relevant to your idea because F^3 already gives a strong speed/accuracy representation backbone.

## 4) What F^3 contributes (and why it matters for your idea)

From the paper, F^3 makes three important moves:

1. Theory: predictive sufficient statistic framing.
2. Architecture: hash-encoded sparse event processing, not dense voxel brute force.
3. Systems: practical real-time throughput on HD/VGA + fast downstream tasks.

In short: F^3 spends compute where events exist.

That is exactly the principle you want for sparse query depth.

### 4.1 The logic of the F^3 theory, in plain language

F^3 paper builds this chain:

1. Assume events are generated from latent scene statistics + noise.
2. Show that if you can denoise past events in a good sparse basis and learn their dynamics, you can predict future events well.
3. Use future-event prediction as the training objective for a representation.
4. Argue this representation keeps structure and motion information useful for downstream tasks.

Practical translation:

- Future-event prediction is not only an end task; it is a self-supervised pretext that shapes a useful geometry-aware representation.

### 4.2 Why hash encoding is central (not incidental)

In F^3, hash encoding is not just a \"faster embedding\":

- It implements multiscale event-coordinate feature lookup.
- It avoids dense encoding of empty voxels.
- It updates only touched entries during backprop.

This is the same philosophy as Instant-NGP, adapted to event spatiotemporal coordinates.

### 4.3 Why F^3 still leaves room for your work

F^3 representation is sparse-aware, but many downstream heads (including depth) remain dense-output. So:

- F^3 solves the front-end efficiency problem well.
- Your project targets the back-end efficiency problem: sparse query output instead of dense map output.

## 5) Depth-specific methods: what they optimize, and where they waste time

### 5.1 Typical modern event-depth pipeline

Most pipelines do:

1. Build event representation over full image.
2. Run full dense backbone/decoder.
3. Output full `H x W` depth.

Even if you only need 10 points, they still compute all pixels.

### 5.2 Why runtime stays high

The expensive part is usually not only event encoding; it is dense decoder computation (transformer/CNN over all spatial tokens).

So even with efficient encoding, dense prediction can dominate runtime.

### 5.3 Why your sparse-query formulation is meaningful

If user asks for `Q` points only, optimal complexity should scale closer to `O(Q)` (plus shared event encoding cost), not `O(HW)`.

Current literature largely does not optimize for this objective directly.

## 6) DINO and Instant-NGP: why Prof. Chaudhari likely mentioned them

### 6.1 DINO / DINOv2

DINO family gives strong self-supervised visual features with strong semantic structure.

Potential use here:

- Distillation teacher for boundary-aware priors.
- Query sampling policy (choose informative points).
- Regularizer to preserve semantic edges/object boundaries in sparse depth outputs.

But DINO itself is not a sparse event-depth runtime solution; it is a representation/teacher tool.

### 6.2 Instant-NGP

Instant-NGP introduced multi-resolution hash encoding for fast neural fields. Its key ideas map directly to your setting:

- Hash features indexed by coordinate
- Multiscale interpolation
- Small MLP decoders
- Great speed-quality tradeoff

F^3 already borrows this spirit for event representation. Your "inverse F^3" can push it further: query-wise depth decoding instead of dense map decoding.

## 7) What appears underexplored (your opportunity)

Strong gap in the field:

- Many event-depth works optimize accuracy.
- Some optimize dense latency.
- Few optimize query-only depth at arbitrary sparse pixel sets.

This is a publishable gap if you show:

- Runtime scaling vs number of requested points.
- Comparable or controlled accuracy drop vs dense baselines.
- Robustness across event rate, platform, lighting.

## 8) Suggested paper reading order (beginner -> advanced)

1. Event camera basics and survey
2. Unsupervised event geometry (2019)
3. Learning Monocular Dense Depth from Events (2020)
4. RAM Net + DTL (2021)
5. EReFormer (2024)
6. Depth Anything V2 (2024)
7. On-device event self-supervised depth (2025)
8. Depth AnyEvent and Depth Any Event Stream (2025)
9. F^3 paper (deep read)
10. Instant-NGP and DINO/DINOv2 (for method transfer ideas)

## 9) Practical takeaways for your project

- F^3 is the right starting representation for speed-sensitive event tasks.
- Depth decoders in current pipelines are mostly dense and thus misaligned with sparse-query needs.
- An "inverse F^3" research program is both technically justified and novel enough.
- DINO can help training signal/regularization; Instant-NGP can inspire coordinate-query architecture and caching.

## 10) Key references and links

### Core event-camera/F^3

- F^3 paper: https://arxiv.org/abs/2509.25146
- F^3 project page (from paper): https://www.seas.upenn.edu/~richeek/f3/

### Event depth classics and major baselines

- Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion (CVPR 2019):
  https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.html
- Learning Monocular Dense Depth from Events (3DV 2020):
  https://arxiv.org/abs/2010.08350
- RAM Net (RA-L 2021):
  https://arxiv.org/abs/2102.09320
- DTL (ICCV 2021):
  https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Dual_Transfer_Learning_for_Event-Based_End-Task_Prediction_via_Pluggable_Event_ICCV_2021_paper.html
- EReFormer (2024 TCSVT DOI landing):
  https://ieeexplore.ieee.org/document/10474340

### Newer event-depth directions (2025)

- On-device self-supervised learning of low-latency monocular depth from only events (CVPR 2025):
  https://openaccess.thecvf.com/content/CVPR2025/html/Hagenaars_On-device_self-supervised_learning_of_low-latency_monocular_depth_from_only_events_CVPR_2025_paper.html
- Depth AnyEvent (ICCVW 2025):
  https://openaccess.thecvf.com/content/ICCV2025W/NeurArch/html/Zhang_Depth_AnyEvent_Learning_Event-based_Metric_Depth_via_Cross-Domain_Transfer_ICCVW_2025_paper.html
- Depth Any Event Stream (arXiv 2025):
  https://arxiv.org/abs/2507.00306

### Foundation/representation papers relevant to your method design

- Depth Anything V2:
  https://arxiv.org/abs/2406.09414
- DINO:
  https://arxiv.org/abs/2104.14294
- DINOv2:
  https://arxiv.org/abs/2304.07193
- Instant Neural Graphics Primitives (Instant-NGP):
  https://arxiv.org/abs/2201.05989

### Survey

- Recent Event Camera Innovations: A Survey (2024):
  https://arxiv.org/abs/2408.13627

## 11) Notes on uncertainty

- Some recent 2025 event-depth works are workshop/preprint-stage and may evolve quickly.
- For papers not fully open in IEEE, methodology here is based on available abstracts/official summaries where needed.

