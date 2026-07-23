# MiohRestorer V5 integrated specification

Mioh means **Motion-Informed Optical Healing**. V5 is the shared data,
evaluation and curriculum generation. It now has two deliberately separate
model families: greenfield ANE-oriented V5-Q/S and quality-first hybrid V5-HQ.
V5-Q/S do not inherit V4 weights. V5-HQ initializes its recurrent backbone
from the established BasicVSR++ v1.2 weights and adds a new ROI refiner.

This document is the implementation contract. Items marked as measurements
must be resolved by the Stage 0 prototype before full training starts.

## Implementation status (2026-07-22)

The Stage 0 implementation now exists independently of V4/V4.1:

- folded-space V5-Q and V5-S model definitions;
- exact fixed-convolution phase shifts for all nine +/-1px candidates;
- five-stage alignment, reliability, occlusion and entropy signals;
- RGB + mask + mask-reliability input;
- zero-initialized base/texture/confidence heads and exact ROI gating;
- monolithic, split encoder/decoder and explicit-state Core ML contracts;
- five-size ROI selection, even-origin crops, asymmetric hysteresis, cut-safe
  window padding and isolated mask-miss repair;
- six independent loss curricula and a model-only wall-clock profiler;
- fixed-shape Core ML export and residual-motion measurement tools.

The quality-first V5-HQ path is also implemented and verified independently:

- nine-frame BasicVSR++ recurrent propagation with SPyNet optical flow;
- true bilinear sub-pixel `grid_sample` warping;
- second-order modulated DCNv2 propagation;
- learned flow-residual deformable attention and visibility-weighted fusion;
- five output-specific ROI restorations with exact source preservation outside
  the mask;
- six stages that progressively unfreeze the refiner, recurrent backbone and
  finally SPyNet;
- a Core AI exporter that registers the existing custom Metal kernels for
  `grid_sample` and DCNv2.

On the M5 Pro, an actual 256px MPS training step completed in 5.33 seconds with
8.32 GiB driver memory in Stage 1. A full backward pass through the dynamic
warp completed with 11.70 GiB driver memory. PyTorch MPS lacks
`grid_sampler_2d_backward`, so **training only** uses its documented CPU
fallback for that derivative. The exported inference program does not use that
fallback: the conversion smoke test succeeded with 9,627 operations, including
111 custom Metal grid-sample calls and 32 custom Metal DCNv2 calls.

Run a fresh V5-HQ curriculum with:

```zsh
TRAIN_MANIFEST=/path/to/train-native-256.jsonl \
VALIDATION_MANIFEST=/path/to/validation-native-256.jsonl \
WORK_ROOT=/path/to/mioh-restorer-v5-hq-run \
zsh scripts/training/run-mioh-restorer-v5-hq-local.sh
```

Export a trained EMA checkpoint with:

```zsh
.venv-coreai/bin/python scripts/apple/export_mioh_restorer_v5_hq_coreai.py \
  --checkpoint /path/to/mioh-v5-hq-stage6-latest.pth \
  --output /path/to/mioh-restorer-v5-hq.aimodel
```

The first M5 Pro measurements use the untrained, uncompromised V5-S baseline:
9-frame context, full 49-candidate coarse search and the original 32/48/80/128
encoder widths.

| 128px execution | Median | Numeric agreement |
|---|---:|---:|
| monolithic V5-S | 91.59 ms | 79.12 dB |
| split encoder | 0.39 ms | measured separately |
| split decoder | 96.21 ms | 79.12 dB |
| explicit-state V5-S | 109.39 ms | RGB 79.12 dB |
| monolithic V5-Q, five outputs | 459.41 ms | 79.11 dB |

Passing all nine frames through the converted encoder and then the converted
decoder also gives 79.12 dB RGB agreement (confidence 99.05 dB). Feature
caching is therefore quality-neutral at the model boundary; it does not justify
reducing frame context, channels or alignment candidates.

The V5-S monolithic program contains 5,599 operations; 2,074 measured
operations prefer the Neural Engine and 34 prefer CPU. The split and stateful
forms preserve RGB output numerically but are slower on this machine, so model
division is not assumed to be an optimization. The explicit feature state also
shows only 53-65 dB fp16 agreement in learned feature tensors and therefore
needs a long-run drift test before it can be considered quality-neutral.

An 18-candidate factorized coarse search reached 34.69 ms at 128px with a
five-frame context, but it is not numerically equivalent to the 49-candidate
baseline and is retained only behind an explicit smoke-test flag. Feature-width
or neighbour reductions are not the default and are not used for quality
training without a trained parity test.

Run the baseline Stage 0 conversion with:

```zsh
python scripts/apple/smoke-test-mioh-restorer-v5.py \
  --variant s \
  --sizes 128,192,256,384,512 \
  --execution all \
  --output-dir /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5/stage0
```

The full five-size sweep remains pending. Full V5 training must not start from
these random weights merely because conversion works.

Validate a completed report with a fail-closed quality gate:

```zsh
python scripts/apple/validate-mioh-restorer-v5-stage0.py \
  /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5/stage0/stage0-report.json
```

## Objectives and invariants

- Clean ground truth is the primary teacher in every stage.
- Restore only the detected ROI; pixels outside the compositing mask are copied
  exactly from the source.
- Preserve source pixels at their native scale. Never enlarge a small crop or
  shrink a large crop merely to fit one model input.
- Put no learned convolution on a thin full-resolution feature plane.
- V5-Q/S use only ANE-oriented fixed operators. V5-HQ intentionally keeps
  deformable convolution, `grid_sample`, optical-flow warp and recurrent
  propagation, implemented by Core AI plus custom Metal kernels.
- A quality-reference model and a shipping model are separate products. A
  quality experiment is not promoted merely because it converts successfully.

## Native-resolution ROI buckets

### Next full-training dataset rebuild

The next from-scratch training run must replace the current boundary-tile-heavy
manifest with a native, mask-centred multi-bucket dataset. The present run is
allowed to finish unchanged so it remains a comparable baseline.

- Reuse clean native source videos or `crop_unscaled_img`; do not resize or
  regenerate every video merely to obtain a fixed square file.
- Store source path, nine-frame range, per-frame tracked mask, stabilized even
  crop origin, motion bucket and crop size in the manifest. Decode and crop at
  training time to avoid another lossy RGB encode.
- Train shared weights with native 192, 256 and 384 crops, initially sampled at
  approximately 10%, 30% and 60%. Never upscale a smaller ROI. Tile only ROIs
  larger than 384, using overlapping native 384 tiles.
- Require a non-empty mask in the five output frames. Prefer mask occupancy of
  20-70%; retain 70-95% occupancy for large-ROI coverage, limit 5-20% boundary
  tiles to 10-15% of samples, and reject empty or token edge fragments.
- Keep the tracked mask near the crop centre while retaining unmasked context.
  Quantize the stabilized crop trajectory to even integer pixels and never
  cross a scene cut.
- Stratify by crop size and motion (static/low/medium/high), cap each source
  video at roughly 30-50 representative windows, and split train/validation/
  test strictly by source video.
- Generate mosaic shape, block size, compression, blur, noise, mask jitter and
  detection misses online. Validation and test degradations must be completely
  deterministic.
- Before restarting Stage 1, run a real 384px MPS forward/backward memory probe
  on the fully trainable V5-HQ graph. If it fits 48 GiB, make 384 the primary
  quality bucket; otherwise alternate 256/384 batches without resizing.

This is the required input pipeline for the next run, not a request to mutate
the currently running Stage 2 experiment.

V5 uses five fixed, square input shapes compiled from shared weights:

| Bucket | Intended use |
|---:|---|
| 128 | very small ROI plus local context |
| 192 | small ROI |
| 256 | medium-small ROI |
| 384 | medium-large ROI |
| 512 | large ROI and quality-critical inference |

An ROI larger than 512 is processed as overlapping 512 tiles. The runner
chooses the smallest bucket that contains the tracked ROI union plus 25-40%
context. Missing area at a frame boundary is padded; the image is not resized.
An ROI too small to contain recoverable evidence is still placed in a native
128 crop and the confidence system is expected to prefer conservative output.

The crop trajectory is low-pass filtered in time and its top-left coordinate
is quantized to an even integer pixel. No resampling is used. Even origins keep
PixelUnshuffle phase assignments stable across frames. Padding at image edges
must preserve the same source-grid phase.

Bucket selection is stable per tracked segment. Expansion occurs at the next
window boundary as soon as the nine-frame union no longer fits. Contraction is
allowed only after the ROI plus safety margin fits the smaller bucket for at
least 18 consecutive frames. A bucket transition rebuilds the feature cache
and blends the old and new output briefly through the existing ROI feather.

The shipping graph retains all five inference shapes, but Apple Silicon
quality training uses native 128, 192 and 256 crops. This matches the current
BasicVSR++ 256-pixel training/restoration unit and fits V5-Q backward passes in
48 GB. A direct 512 V5-Q training probe reached approximately 63 GB and is
therefore prohibited on this machine. Larger regions are represented by
overlapping native 256 tiles; they are never resized. ROI losses are normalized
by valid ROI pixels and validation is reported by tile size so a regression in
one bucket cannot be hidden by the global average.

The V5 loader is deliberately separate from the legacy fixed-size loader. A
native manifest stores the original clean and mask videos, exact nine-frame
range, even crop origin for every frame, mask reliability and bucket. Oversize
regions become overlapping tracked 256 training tiles; they are not discarded. Build a
manifest for each already source-separated split with, for example:

```zsh
python scripts/training/build-mioh-restorer-v5-native-manifest.py \
  --metadata-root /path/to/train/crop_unscaled_meta \
  --output /path/to/v5/train-native.jsonl \
  --stride 4 \
  --context-fraction 0.30 \
  --maximum-bucket 256 \
  --tile-overlap 64
```

`MiohRestorerV5NativeDataset` performs only exact native crops and edge
padding. `V5BucketBatchSampler` keeps every batch shape-homogeneous. The
manifest report must say `resized_frames: 0` and `maximum_bucket: 256` before
the local six-stage run starts.

## Tracking, cuts and temporal boundaries

The crop center follows a temporally smoothed tracked ROI. Alignment therefore
spends its range on residual object motion rather than global crop motion.
Before freezing the search ranges, residual motion must be remeasured after
this centering for every size and motion bucket. The old p99 of approximately
40 pixels is evidence, not the final V5 design constant.

The runner detects hard cuts and gradual transitions before constructing a
nine-frame window. No inference or feature cache may cross a confirmed cut.
At a segment boundary, unavailable temporal frames are replicated from the
nearest valid frame. Temporal reflection is not used because it creates a
synthetic reversal of motion. Training and inference use the identical edge
padding rule.

Training includes hard cuts, short dissolves and unrelated injected frames.
The frame-reliability target for a frame from the other side of a cut is zero.
During an ambiguous dissolve, the runner either splits the window or suppresses
cross-transition fusion according to the measured frame reliability.

## Detection-mask robustness

A raw temporal union of masks is forbidden because it over-covers a moving
subject. The runner propagates masks along the stabilized crop motion, merges
them in the aligned coordinate system, and interpolates isolated detection
misses. It records whether each mask came from direct detection, propagation or
interpolation.

Each input frame contains RGB, a mosaic mask and a mask-reliability map. The
initial V5 input contract is therefore five channels per frame, not four. A
missing detection is never represented as an authoritative all-zero mask. The
training degradation generator includes partial masks, dropped masks, shifted
masks and false-positive margins so the reliability and fusion heads learn to
reject them.

The canonical model mask uses the same native-pixel feather implementation in
training and inference. The initial canonical feather is 8 pixels. The user
facing extra feather control remains a downstream compositing adjustment and
does not silently change the model input distribution.

## Canonical colour and precision contract

Decoding, dataset creation, evaluation and application inference call one
shared colour-conversion implementation. V5 initially supports SDR BT.601,
BT.709 and BT.2020 sources in limited or full range at 8 or 10 bits.

- YUV is expanded with the matrix and range declared by the stream metadata.
- Missing or contradictory metadata is diagnosed and resolved once before
  processing; it is never allowed to vary from frame to frame.
- The model works in canonical non-linear BT.709 RGB represented as float.
- Ten-bit source values are converted directly to float without an intermediate
  eight-bit quantization.
- Output is converted back to the requested matrix, range and bit depth only
  after restoration and compositing.

HDR PQ/HLG is outside the first V5 training distribution. It must be detected
and explicitly bypassed or routed to a separately validated HDR model; silent
tone mapping into the SDR model is not permitted.

## V5-Q quality-reference model

V5-Q is the quality-first reference, initially targeting 12-18 million
parameters. It consumes nine frames and emits the middle five frames. Every
output has itself as the alignment reference; aligning all outputs to frame
four is forbidden because it reproduces the V4 centre-drag/ghosting failure.

The number of neighbours per output is a Stage 0 measurement, not an assumption.
Local 5, local 7 and all 9 frame contexts are converted and benchmarked. Output
contexts and the shared decoder are folded into the batch axis where that
reduces graph duplication without changing output-specific alignment.

V5-Q also has an untrained-conversion gate. Its Core ML program should remain
at or below 20,000 operations, roughly twice the V4 reference graph. A larger
graph requires redesign before training, even when numerical conversion works.

## V5-S shipping model

V5-S is a 3-6 million parameter, single-centre-output streaming student with
the same architectural family as V5-Q. It receives V5-Q central-output,
feature and alignment supervision, while paired clean GT remains primary.

The preferred execution design separates the per-frame encoder from the
fusion decoder and keeps nine encoded frames in a ring buffer. This should
reduce steady-state work to one encode and one decode per new frame. It is not
declared the shipping implementation until Stage 0 compares three complete
paths:

1. separate encoder and decoder Core ML models with an external feature ring;
2. one Core ML model with explicit feature-state inputs and outputs;
3. one monolithic nine-frame model.

The benchmark includes model-call overhead, ANE-to-memory transfers, cache
updates and synchronization. If the split-model transfer erases the saved
compute, the stateful or monolithic form wins.

### Core AI temporal-I/O contract

Every Core AI model that repeats work across frames must expose the repeated
frames, contexts, flows or features as one contiguous temporal tensor such as
`[K,C,H,W]`. It must not expand that axis into `K` separately named inputs.
The variable-length BasicVSR++ investigation measured a Core AI numerical
regression for the separate-input graph while the equivalent contiguous
tensor graph retained parity. This is therefore a numerical-correctness
requirement, not merely a performance preference. All Stage 0 execution
variants, including future stateful variants, must pass the contiguous-input
canary before training or packaging.

## Folded-space architecture

The input enters PixelUnshuffle immediately. All learned correlation, fusion
and reconstruction work happens in folded half-, quarter-, eighth- and
sixteenth-resolution spaces. The final output returns to native resolution
with one PixelShuffle.

The alignment pyramid is initialized around this coverage and then corrected
from the remeasured residual-motion distribution:

1. 1/16 coarse search: up to approximately +/-48 source pixels;
2. 1/8 residual search: approximately +/-8 pixels;
3. 1/4 residual search: approximately +/-4 pixels;
4. 1/2 residual search: approximately +/-2 pixels;
5. folded phase-channel permutation: approximately +/-1 source pixel.

Every shift bank is a frozen one-hot convolution from the first implementation;
explicit pad/slice enumeration is forbidden. Correlation uses L2-normalized
features, learned offset bias and a bounded temperature. Fusion observes
correlation confidence, entropy, occlusion probability, mask reliability and
per-frame reliability.

The decoder first restores low-frequency structure and then high-frequency
detail. Folded base, texture and confidence heads return through PixelShuffle.
The final training expression is:

```text
source + mask * (base + confidence * texture)
```

The final residual layers are zero-initialized, so an untrained model is an
identity mapping. Confidence supervision comes from the pre-gate candidate,
not the gated output. Training logs include confidence mean, standard
deviation, calibration error and correlation with negative candidate error.
Clamping to [0,1] occurs only in the export/inference wrapper.

## Training curriculum

Each stage is an independent job with its own optimizer, EMA, completion
manifest and JSON evaluation gate. A completed parent initializes the next
stage; `RESUME` is reserved for resuming the same interrupted stage.

1. **Known-motion alignment:** integer and subpixel translation, all folded
   phase residues, occlusion, scale and small rotation. Exact hierarchical
   shift targets supervise alignment and reliability.
2. **Natural-motion alignment:** the same nine-frame information available at
   inference. Clean non-mosaic context supplies photometric correspondence,
   feature consistency and reliability/occlusion targets. No pretrained flow
   network or information from outside the window is used.
3. **Faithful structure:** clean GT reconstruction, colour, shape and
   low-frequency structure.
4. **Detail recovery:** Haar/wavelet bands, gradients, multiscale frequency
   losses and weak perceptual supervision. Texture is never trained only
   through the confidence gate.
5. **Temporal consistency:** detached local correspondences calculated from
   paired clean GT provide aligned temporal and acceleration losses with
   visibility masking. Unaligned frame-difference loss is forbidden because it
   penalizes legitimate motion and causes blur.
6. **Fidelity polish:** fresh low-rate optimizer against clean GT. GAN is an
   optional separately evaluated experiment, never an automatic stage.

The production V5 curriculum is independent. It does not load V3/V4 weights,
BasicVSR++, SPyNet or an external optical-flow model. Stage 1 generates exact
quarter-pixel translations analytically; Stage 2 uses V5's own alignment over
clean context; Stages 5/6 derive detached local correspondences directly from
paired clean GT. Clean GT remains the primary target in every stage.

## Apple Silicon training

The local runner supports all six stages on MPS. Each completed stage writes
its raw weights, EMA weights, optimizer/scaler state, metrics JSONL and a
`stage-complete.json` gate. The next stage loads the previous EMA weights and
always creates a fresh optimizer. An interrupted stage resumes automatically
from its own `latest.pth`; it never crosses a stage boundary with the old
optimizer.

Build separate native manifests for the source-separated training and
validation splits, then run:

```zsh
cd /Users/okatti/Documents/lada

TRAIN_MANIFEST=/path/to/train-native.jsonl \
VALIDATION_MANIFEST=/path/to/validation-native.jsonl \
WORK_ROOT=/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5/runs/v5-q-native \
zsh scripts/training/run-mioh-restorer-v5-local.sh
```

For a ten-step end-to-end pilot of every stage, add `STEPS=10`. For a partial
run, set `START_STAGE` and `END_STAGE`. Stage 5/6 require V5-Q because temporal
loss needs its five consecutive outputs. MPS mixed precision remains off;
batch size 1 plus gradient accumulation is the safe default for the M5 Pro
48 GB machine. The macOS runner defaults to `WORKERS=0` so PyTorch does not
depend on `torch_shm_manager`; opt into worker processes only after verifying
that shared storage is supported by the selected temporary volume.

Before full training, every model size runs 100 measured steps. The report
must include seconds per step, peak memory, data-loading time and projected
wall-clock duration for every stage. If the projected MPS run is impractical,
the heavy stages move to a cloud NVIDIA worker without changing the data split,
loss definitions or evaluation protocol.

## BasicVSR++ parity gate

The comparison set is frozen before training and contains the representative
validation split, size-by-motion buckets, codec/bit-depth buckets and named
real scenes including the established MIDA-176 comparisons. BasicVSR++ output
is generated once with fixed settings and retained as the reference.

V5-Q reaches numerical parity only when all of the following hold:

- aggregate ROI PSNR is no more than 0.2 dB below BasicVSR++;
- no populated size-by-motion bucket is more than 0.5 dB below it;
- aggregate ROI high-frequency RMSE and flow-aligned temporal error are no
  more than 5% worse;
- no populated bucket is more than 10% worse on either error;
- severe ghosting, anatomy corruption and boundary failures do not exceed the
  BasicVSR++ failure count.

Real footage is evaluated with randomized, label-hidden A/B pairs. The minimum
protocol is 60 scenes spanning every required bucket. A score counts a win as
one and a tie as one half. Parity requires non-inferiority within five
percentage points; claiming superiority requires the confidence interval to
clear 50%. More raters are preferred, but the presentation order and labels
remain blinded even for a single rater.

If metrics disagree, visible severe artifacts or temporal regression block
promotion. A PSNR loss within the allowed 0.2 dB may pass when high-frequency,
temporal and blinded preference results improve. Higher PSNR alone cannot
override worse temporal stability or blinded preference.

## Confidence calibration and runtime routing

Confidence is not trusted as a router merely because its training loss falls.
It is calibrated on held-out synthetic and real distributions, with reliability
curves, expected calibration error and failure-detection precision reported by
size and motion bucket. Cut confidence, mask reliability and alignment entropy
also participate in the out-of-distribution decision.

A single low-confidence window never changes models. Sustained low confidence
uses asymmetric hysteresis and changes route only at a safe window or scene
boundary. The old and new outputs are blended across a short transition. The
fallback is BasicVSR++ when installed and affordable; otherwise it is the
original ROI or a conservative base-only V5 result. This permits gradual V5
deployment without model-switch flicker.

## Timing formats and hardware floor

V5-S must execute correctly on an M1 Mac with 16 GB unified memory. Every one
of the five shapes is converted and tested on that floor. Real-time 512
performance is not assumed; when memory or latency fails its device gate, the
runner uses overlapping 384 tiles rather than crashing or silently changing
scale.

Interlaced material is deinterlaced before detection, tracking and restoration;
telecined material is inverse-telecined when cadence is confidently detected.
The operation and field order are recorded in metadata. Variable-frame-rate
input is normalized to an explicit working cadence before temporal windows are
formed, with the time mapping retained for audio synchronization and output
timestamps. Large timestamp discontinuities split a temporal segment. Training
data uses the same cadence and boundary semantics as inference.

## Known subpixel limitation

The explicit search terminates at integer source-pixel phase shifts. True
fractional residual motion can therefore leave slight high-frequency blur.
Stage 1 includes 0.25, 0.5 and 0.75 pixel synthetic displacements so folded
phase fusion can learn interpolation, but this remains a documented limitation
and is evaluated separately rather than hidden in aggregate metrics.

## Stage 0 implementation order and gates

No full V5 training begins until the following sequence completes:

1. finish and evaluate the V4.1 3,000-step experiment as a detail-path
   diagnostic;
2. remeasure crop-centred residual motion for all five input sizes;
3. implement untrained V5-Q and V5-S skeletons without modifying V4/V4.1;
4. convert all five shapes and all three V5-S execution forms;
5. verify PyTorch/Core ML agreement, compute placement and program operation
   counts;
6. benchmark end-to-end encoder, cache transfer, decoder and compositing;
7. run the 100-step wall-clock and memory pilot;
8. freeze the winning graph, search ranges and stage-duration plan;
9. only then start Stage 1 training.

Initial latency goals are <=15 ms per new frame and tile for V5-S, with 30 ms
as the hard shipping ceiling, and <=60 ms for V5-Q. These are measured goals,
not reasons to remove quality-critical evidence before the quality hypothesis
has been tested. Any speed reduction must be a separately measured V5-S design
choice rather than an unrecorded compromise in V5-Q.
