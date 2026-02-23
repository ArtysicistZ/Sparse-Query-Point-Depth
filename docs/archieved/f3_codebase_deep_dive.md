# F^3 Codebase Deep Dive (fast-feature-fields-main)

Author: Codex
Date: 2026-02-12
Goal: Explain exactly how this codebase trains F^3, runs downstream tasks, and achieves high speed.

## 1) Repository map

Main training/inference entry points:

- `fast-feature-fields-main/main.py`: Train F^3 (future-event prediction pretraining).
- `fast-feature-fields-main/everything.py`: Joint inference script (events + F^3 + depth + flow + segmentation).
- `fast-feature-fields-main/test_speed.py`: Runtime benchmark for F^3 and baselines.
- `fast-feature-fields-main/hubconf.py`: Minimal torch.hub version + deployment-friendly options.
- `fast-feature-fields-main/_aoti_pt2/*`: PT2 AOTI export and inference path.

Core model implementation:

- `fast-feature-fields-main/src/f3/event_FF.py`
- `fast-feature-fields-main/src/f3/utils/utils_MRHE.py`
- `fast-feature-fields-main/src/f3/utils/utils_op.py`
- `fast-feature-fields-main/src/f3/utils/dataloader.py`
- `fast-feature-fields-main/src/f3/utils/utils_train.py`
- `fast-feature-fields-main/src/f3/utils/utils_val.py`

Depth task implementation:

- `fast-feature-fields-main/src/f3/tasks/depth/train_monocular_rel.py`
- `fast-feature-fields-main/src/f3/tasks/depth/finetune.py`
- `fast-feature-fields-main/src/f3/tasks/depth/utils/models/depth_anything_v2.py`
- `fast-feature-fields-main/src/f3/tasks/depth/utils/dataloader.py`
- `fast-feature-fields-main/src/f3/tasks/depth/utils/losses.py`
- `fast-feature-fields-main/src/f3/tasks/depth/utils/trainer.py`
- `fast-feature-fields-main/src/f3/tasks/depth/utils/validator.py`

## 2) How F^3 pretraining works in code

### 2.1 Training objective and loop

In `main.py`:

- Model init: `init_event_model(..., return_logits=True, return_loss=True)` (`main.py:45`)
- Optional compile: `torch.compile(..., fullgraph=False)` (`main.py:48`)
- Optimizer: `AdamW` (`main.py:56`)
- Scheduler: linear LR decay (`main.py:57`)

Training/validation calls:

- Train: `train_fixed_time(...)` (`main.py:95`)
- Validate: `validate_fixed_time(...)` (`main.py:105`)

Losses/metrics are computed against future event grids from context windows.

### 2.2 Data pipeline for F^3 pretraining

`src/f3/utils/dataloader.py` does:

- Load event stream from HDF5 by dataset-specific paths (`dataloader.py:83-108`).
- Sample context window (`get_ctx_fixedtime`) and future window (`get_pred_fixedtime`) (`dataloader.py:151-188`).
- Normalize event coordinates to model frame (`dataloader.py:174`).
- Use custom collate to support variable event count per batch (`dataloader.py:349`).

This is crucial: event counts are dynamic, so model path must support variable-length event lists.

### 2.3 Core architecture path (EventPatchFF)

Main class: `EventPatchFF` (`event_FF.py:223`).

#### Step A: Multi-resolution hash encode each event

- `self.multi_hash_encoder = MultiResolutionHashEncoder(...)` (`event_FF.py:312`)
- Encoder internals in `utils_MRHE.py`:
  - Hash over integer grid corners (`utils_MRHE.py:53-57`, `98-106`, `136-140`)
  - Trilinear interpolation weights (`utils_MRHE.py:93-95`)
  - Concatenate features over levels (`utils_MRHE.py:209-210`)

#### Step B: Permutation-invariant aggregation into spatial field

In variable mode:

- Event coords -> pixel indices (`event_FF.py:504-505`)
- Encoded events shape `(N, L*F)` (`event_FF.py:507`)
- Scatter-add into dense feature map using `index_put_(..., accumulate=True)` (`event_FF.py:511`)

This realizes the set aggregation idea: order of events does not matter.

#### Step C: Spatial smoothing with small ConvNeXt-style CNN

- Downsample stages + blocks (`event_FF.py:323-339`, forward `542-545`)
- Optional upsampling/skip path (`event_FF.py:341-381`, forward `548-559`)
- Prediction head outputs patch logits then `unpatchify` to full future event volume (`event_FF.py:568-571`)

### 2.4 Pretraining loss

`voxel_FocalLoss` in `utils_op.py:206-227`:

- BCE-with-logits base + focal modulation `(1 - p_t)^gamma` (`utils_op.py:216-218`)
- Class reweighting using positive ratio (`utils_op.py:219-220`)
- Valid-mask aware reduction (`utils_op.py:214`, `223-227`)

This is designed for severe event/non-event imbalance and noise.

### 2.5 Paper-to-code mapping (important for deep understanding)

The F^3 paper describes a decomposition of event representation learning into:

- event-coordinate encoding (`phi`)
- permutation-invariant set pooling over events
- spatial smoothing network (`rho`)
- future-event predictor (`psi`)

Concrete code mapping:

- `phi` (multires hash event encoding): `utils_MRHE.py` (`index3d`, `forward_nopol`)
- set pooling: scatter-add accumulation into `feature_field` (`event_FF.py:511`)
- `rho` (spatial CNN): `downsample_layers` + `stages` in `EventPatchFF` (`event_FF.py:323-339`, `542-545`)
- `psi` (future event predictor): `pred` + `unpatchify` (`event_FF.py:397-401`, `568-571`)
- robust event objective: focal loss (`utils_op.py:206-227`)

This is not a loose analogy: the implementation directly realizes the architecture described in the paper.

### 2.6 End-to-end tensor flow (variable mode)

Typical pretraining batch path:

1. Input events (concatenated): `(N_total, 3/4)`
2. Hash encode: `(N_total, L*F)`
3. Scatter-add to spatial field: `(B, C, W, H)` after permute
4. CNN backbone/downsampling: `(B, C', W/ps, H/ps)` (task-config dependent)
5. Prediction head: `(B, patch_size^2 * T, W/ps, H/ps)`
6. Unpatchify logits: `(B, W, H, T)`
7. Compare to future event grid and valid mask.

## 3) Why F^3 is fast (as implemented, not just paper-level)

### 3.1 Compute only where events exist (front half of pipeline)

Instead of dense voxel convolutions over all `W x H x T`, the code:

- Encodes only `N_events` points (hash lookups/interpolation).
- Aggregates via sparse scatter add.

Core lines:

- `event_FF.py:507`, `511`
- `utils_MRHE.py:196-210`

### 3.2 Multi-resolution hash encoding gives compact expressive features

- Small trainable feature vectors per hash table entry (`utils_MRHE.py:41-43`)
- O(1)-style table lookup and interpolation per event
- Lower parameter and memory traffic than large MLP encoders

### 3.3 Small downstream heads

In F^3 optical flow, paper reports small flow head (~28k params). In code, downstream models are lightweight wrappers compared with large full-dense event transformers.

### 3.4 Aggressive compiler path

Widespread use of `torch.compile` in:

- Core F^3 training/inference (`main.py:48`, `test_speed.py:97-105`)
- Downstream parts (`everything.py:126,136,137,147,157`)
- hub helper (`hubconf.py:777-792`)

### 3.5 Deployment mode: single-batch graph + precomputed hash features

In `hubconf.py`, special inference path:

- `single_batch_mode` path for full-graph-friendly export (`hubconf.py:687-723`)
- Optional `_build_hashed_feats` to precompute full `(x,y,t)->feature` table (`hubconf.py:629-644`)
- Fast inference can use table indexing instead of online hash encoding (`hubconf.py:699-704`)

Tradeoff:

- Faster latency
- Much larger memory footprint

### 3.6 AOTI PT2 export pipeline

`_aoti_pt2/export_f3_to_pt2.py`:

- Loads model in single batch mode (`export_f3_to_pt2.py:58-66`)
- Optionally prebuilds hashed features (`export_f3_to_pt2.py:81-83`)
- Exports with `torch.export` + AOTI package (`export_f3_to_pt2.py:97-108`)

This is production-focused optimization.

## 4) Depth pipeline in this codebase

## 4.1 Model wrappers

`EventFFDepthAnythingV2` (`depth_anything_v2.py:65`):

- Loads pretrained F^3 backbone (`depth_anything_v2.py:73-76`)
- Adapts DepthAnythingV2 input channels from RGB(3) to F^3 channels by tiling first conv weights (`depth_anything_v2.py:36-53`)
- Usually freezes F^3 unless retrain flag enabled (`depth_anything_v2.py:78-80`)

Forward path:

- Get F^3 features: `self.eventff(...)[1]` (`depth_anything_v2.py:109`)
- Crop and resize
- Predict depth/disparity using DA-V2
- Upsample back to original crop size (`depth_anything_v2.py:112-115`)

## 4.2 Two-stage strategy reflected in scripts

Stage 1 (relative/pseudo disparity): `train_monocular_rel.py`

- Uses pseudo disparity datasets (generated from image models) and `ScaleAndShiftInvariantLoss` (`train_monocular_rel.py:98`, `losses.py:109-122`)

Stage 2 (metric fine-tuning): `finetune.py`

- Loads stage-1 checkpoint as pretrained model (`finetune.py:60-64`)
- Fine-tunes with metric targets using `SiLog` or `SiLogGrad` (`finetune.py:103-109`)
- `SiLogGrad` adds gradient regularization against pseudo model outputs for sharper boundaries (`losses.py:16-44`)

This exactly matches F^3 paper logic: pseudo dense first, metric sparse/noisy second.

## 4.3 Depth data loader behavior

`TimeAlignedDepthAndEvents` (`depth/utils/dataloader.py:17`) aligns event context with disparity/depth timestamps and supports modes:

- `m3ed_pseudo`
- `m3ed_gt`
- `dsec_gt`
- `mvsec_gt`
- `tartanair-v2_gt`

Ground-truth disparity conversion and invalid masking are handled in `__getitem__` (`dataloader.py:120-155`).

## 5) How speed is measured in repo

`test_speed.py`:

- Generates synthetic event tensors (`test_speed.py:53-58`)
- Warmup + synchronized timed loop (`test_speed.py:72-86`)
- Benchmarks:
  - F^3 feature extraction (`test_speed.py:96-117`)
  - F^3 + flow/depth/segmentation (`test_speed.py:119-151`)
  - Baseline voxelgrid/frame pipelines (`test_speed.py:155-224`)

The script compares runtime directly across representations under same synthetic load.

## 6) Important implementation details and caveats

### 6.1 Variable-mode compile caveat

There are comments that `torch.repeat_interleave(eventCounts)` in variable mode can be compile-problematic (`event_FF.py:510`, `763`; similar in hub file). The code works, but not all paths are equally compiler-friendly.

### 6.2 Dense tensors still appear after scatter

Even though event encoding is sparse, the aggregated feature field becomes dense before CNN smoothing. This is efficient enough for F^3, but it matters for your sparse-query research: dense downstream decoding still costs `O(HW)`.

### 6.3 Current depth head is dense by design

`EventFFDepthAnythingV2` always predicts dense map over crop. This is the exact bottleneck your proposed "inverse F^3" should remove.

## 7) Summary: what this code already gives you for new research

Strong reusable components:

- Event-native sparse representation backbone (F^3)
- High-quality pretrained checkpoints and configs
- Depth training code with pseudo + metric stages
- Speed benchmarking and deployment pathways

Main missing piece for your new direction:

- A query-conditioned sparse depth head that outputs only selected points, without dense decoder passes.

That is the best insertion point for your project.

