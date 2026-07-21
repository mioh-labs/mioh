# MiohRestorer V4-Q

V4-Q is the quality-first, Apple-native mosaic restoration model. It is a new
model rather than a conversion of BasicVSR++. Its deployment graph contains no
deformable convolution, `grid_sample`, optical-flow warp, or recurrent state.

## Fixed contract

- Input: 9 frames of RGB plus an exact mosaic mask, `1×9×4×384×384`.
- Output: the middle 5 restored RGB frames plus confidence.
- Video stride: 4 frames. Adjacent windows share one output frame.
- Window boundary: the shared frame is linearly blended.
- ROI safety: pixels outside the mosaic mask are copied exactly from the input.
- Alignment: three fixed nine-way cosine-correlation banks at 1/8 and 1/4
  resolution, covering ±40 source pixels.
- Parameter count: 8,933,669 with the default quality configuration.

Each of the five output frames is its own alignment reference. The five local
contexts share one encoder and one decoder, and are folded into the batch axis
for Apple inference. This avoids the quality defect of aligning every output to
the temporal center while retaining most of the five-output efficiency.

The low-frequency base residual and high-frequency texture residual use
separate heads. Confidence gates only texture. The pre-gate candidate supplies
the confidence target, preventing confidence from hiding texture errors by
collapsing to zero. Training output is not clamped; only the export wrapper
clamps to `[0, 1]`.

## Verified Apple behavior

The untrained graph was tested on M5 Pro before training:

- Core ML and Core AI conversion both succeed.
- At `384×384`, all 9,617 operations in the five-output graph support the
  Neural Engine, and Core ML selects the Neural Engine for all of them.
- Core ML five-output latency: 90.9 ms per window, approximately 44 new video
  frames/s at stride 4.
- Compiled Core AI steady-state latency: 46.4 ms median after warmup,
  approximately 86 new video frames/s at stride 4.
- Apple FP16 vs PyTorch FP32: 79.44 dB, maximum absolute error 0.000244.

These are architecture and conversion measurements, not restoration-quality
results. Quality must be established after training using mosaic-ROI metrics.

## V4.1-Q high-resolution detail experiment

V4.1-Q keeps the trained V4 low-resolution trunk and adds a parallel detail
path.  It retains all 32 full-resolution encoder channels and all 48
half-resolution channels, aligns every one of the four neighbouring frames,
and refines motion through half-resolution dilation-two, half-resolution
dilation-one and full-resolution dilation-one correlation banks.  The final
detail residual starts at exactly zero, so whitelist-loading the completed V4
Stage 3 EMA preserves the old output bit for bit before the first update.

Static shift candidates are generated with frozen one-hot dilated depthwise
convolutions.  This is a graph rewrite only: on MPS the convolutional and
pad/slice implementations matched exactly in both fp32 and fp16.  No frame,
candidate, channel or refinement level is removed by this rewrite.

The quality-expanded reference has 9,210,534 parameters.  An apples-to-apples
M5 Pro Core ML test at `1x36x384x384`, five outputs and ten steady-state runs
gave:

| Model | Median | p10 | p90 | FP16 agreement |
|---|---:|---:|---:|---:|
| V4-Q | 92.25 ms | 91.19 ms | 93.52 ms | 79.44 dB |
| V4.1-Q | 1042.95 ms | 1035.30 ms | 1048.59 ms | 76.50 dB |

V4.1-Q is therefore 11.31 times slower than V4-Q (`+1030.6%`) and fails both
the former `+50%` deployment gate and a relaxed `+100%` gate.  This result is
recorded rather than hidden.  Under the explicit quality-first decision,
latency does not block the 3,000-step experiment: V4.1-Q is a quality-reference
model.  It must not replace the shipping model directly.  If its quality gate
passes, a separate V4-S student must be distilled and measured before app
integration.

The separately recorded PyTorch MPS series reaches the same conclusion:

| Model | MPS median | p10 | p90 |
|---|---:|---:|---:|
| V4-Q | 192.04 ms | 189.75 ms | 195.19 ms |
| V4.1-Q | 1379.20 ms | 1374.14 ms | 1387.59 ms |

That is 7.18 times slower (`+618.2%`).  The older 221.9/962 ms observations
belonged to an earlier MPS implementation and must not be mixed with the Core
ML table above.

Start the frozen detail-path bootstrap from the completed Stage 3 EMA:

```zsh
cd /Users/okatti/Documents/lada
zsh scripts/training/run-mioh-restorer-v41-detail-bootstrap.sh
```

The bootstrap saves and validates every 500 steps and ends at 3,000.  Its gate
uses a fixed 32-sample paired protocol and reports motion from the p75
Farneback magnitude inside the clean-target ROI.  At 384px the buckets are
`static <1`, `low 1-3`, `medium 3-5`, and `high >=5` pixels/frame.  The first
32 validation samples measured 0.32-7.70 pixels/frame, so these boundaries
keep the diagnostic buckets populated.

The continuation rules are fixed before training:

- strong signal: combined static+low high-frequency RMSE improves at least
  5%, while ROI PSNR changes by no less than -0.3 dB;
- small but positive high-frequency improvement in every populated motion
  bucket is also a continuation signal for the later full-unfreeze phase;
- static/low failure is diagnosed as a detail-path or loss/freeze problem
  first, not as proof that fixed-shift alignment is impossible;
- static/low improvement with high-speed failure points to fine-alignment
  features or range;
- the structure is rejected only after the applicable diagnosis is cleared
  and the corresponding full-unfreeze attempt still produces no useful gain.

Run the paired, motion-bucketed evaluation against the Stage 3 parent with:

```zsh
/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/training/evaluate-mioh-restorer-v4-checkpoint.py \
  --checkpoint /path/to/v41-step-0003000.pth \
  --baseline-checkpoint \
    /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v4/runs/stage-3-detail-recovery/mioh-restorer-v4-step-0020000.pth \
  --baseline-weights ema \
  --metadata-root \
    /Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative/validation/crop_unscaled_meta \
  --output-dir /path/to/v41-step-3000-evaluation \
  --device mps \
  --batches 32 \
  --weights ema
```

The JSON report includes each bucket, a combined static+low result, direct
improvement against the V4 parent and a machine-readable bootstrap-gate
status.

### Reserved V4-S deployment design

Do not retrofit this into V4.1-Q while its quality experiment is running.
V4.1-Q remains the fixed teacher/reference.  If it demonstrates useful detail
recovery, V4-S starts as a separate student with these measured design rules:

- never run learned convolutions on a thin `384x384` feature plane;
- PixelUnshuffle the full-resolution evidence at the entrance and preserve
  all four pixel phases as channels;
- perform correlation, softmax, shift mixing and detail fusion entirely in
  the folded half/quarter-resolution space;
- express a full-resolution one-pixel displacement as fixed phase-channel
  permutation plus half-grid displacement, returning to full resolution with
  one final PixelShuffle;
- concentrate ANE work in spatially smaller, channel-deeper tensors and test a
  GPU-only detail assignment as a separate deployment variant;
- convert the untrained V4-S prototype before distillation and measure its
  incremental latency against V4-Q under identical Core ML settings.

PixelUnshuffle does not reduce arithmetic by itself.  A same-width 3x3
convolution after a 2x fold can increase MACs by four times; its purpose is to
rearrange work into a shape where Apple accelerators can reach higher
utilization while retaining subpixel phase information.  The prototype is
accepted only on measured wall-clock latency, not FLOP estimates.

## Staged quality training

Run the preflight without starting training:

```zsh
cd /Users/okatti/Documents/lada
CHECK_ONLY=1 zsh scripts/training/run-mioh-restorer-v4-pilot.sh
```

Start the first independent stage from random weights:

```zsh
cd /Users/okatti/Documents/lada
zsh scripts/training/run-mioh-restorer-v4-pilot.sh
```

The representative split contains 1,500 training, 188 validation, and 187
test clips. Training consists of five separate jobs, not one job with internal
phase switches:

1. Stage 1, `foundation`, 10,000 local steps: alignment, shape and
   low-frequency colour. Texture and confidence stay frozen. The main loss is
   paired clean GT. Two auxiliary alignment targets are used together:
   frozen SPyNet flow from natural nine-frame windows and exact displacement
   from a synthetic sequence made by shifting one duplicated frame. No DCNv2
   offset projection is used.
2. Stage 2, `faithful_reconstruction`, 15,000 additional steps: start from the
   selected Stage 1 weights, release texture/confidence, and learn faithful
   clean reconstruction. A trainer-owned `1x1` adapter applies a low-weight
   feature loss against BasicVSR++ quarter-resolution reconstruction features.
   The teacher is run on exactly the same nine input frames, so it cannot pass
   information from outside V4's observable window.
3. Stage 3, `detail_recovery`, 20,000 additional steps: remove the teacher and
   emphasize multi-scale
   structure, edges, high-frequency detail and weak perceptual similarity.
4. Stage 4, `temporal_consistency`, 15,000 additional steps: emphasize motion,
   temporal deltas and acceleration to reduce unstable detail and shimmer.
5. Stage 5, `fidelity_polish`, 10,000 additional steps: start a fresh low-rate
   optimizer and finish against the clean ground truth. Stages 3-5 do not load
   BasicVSR++ at all.

The hierarchy is deliberately not a BasicVSR++ cloning exercise. Clean GT is
the primary teacher in every stage. BasicVSR++ supplies only internal
alignment/feature hints in Stages 1-2, after which the model is free to exceed
the teacher. RGB output distillation is never used on the paired synthetic
training set, because it would preserve the teacher's blur and make its output
the quality ceiling. If unpaired real mosaic footage is added in the future,
pseudo-RGB supervision must be treated as a separate data path rather than
mixed into this paired-GT objective.

Each stage has its own directory, log, local step counter, optimizer, EMA and
completion manifest. After a stage completes, evaluate both raw and EMA,
choose the better checkpoint weights, and explicitly pass that completed
checkpoint to the next stage. The next optimizer is newly initialized and its
EMA starts from the selected parent weights. `RESUME` is reserved for
continuing an interrupted job within the same stage.

The Stage 2 `1x1` adapter belongs to the trainer and is stored only so an
interrupted Stage 2 can resume exactly. It is not part of the V4 model state,
is not inherited by Stage 3, and is not exported to Core ML/Core AI. The
normal inference path does not expose alignment or intermediate features;
those tensors are retained only by a training-only diagnostic forward method.

Each stage begins with a 500-step learning-rate warmup. Stage 2-5 also ramp
their objectives from the preceding stage during those 500 local steps.
Training samples use a 20/50/30 mixture of
clean, mildly degraded and fully degraded inputs plus random time reversal.
This preserves real high-frequency evidence instead of training every sample
under an unrealistically destructive codec/blur stack.

Confidence is frozen in the foundation stage and then trained against the
pre-gate candidate error with a 0.05 scale and only a very weak pull toward
one. Logs include both confidence mean and standard deviation so a constant
gate is visible instead of being mistaken for successful calibration.

GAN is intentionally not enabled automatically. It can make output look more
detailed while inventing anatomy and lowering restoration fidelity. The final
stage therefore returns to direct ground-truth, structural, perceptual and
temporal losses.

After evaluating Stage 1, start Stage 2 from its completed checkpoint (EMA is
the default; set `INITIALIZE_WEIGHTS=raw` when raw wins evaluation):

```zsh
STAGE=2 \
INITIALIZE_FROM=/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v4/runs/stage-1-foundation/mioh-restorer-v4-step-0010000.pth \
zsh scripts/training/run-mioh-restorer-v4-pilot.sh
```

The same pattern applies to Stages 3, 4 and 5. A checkpoint is accepted as a
parent only when it is marked complete and belongs to the immediately
preceding stage.

Resume an interrupted Stage 2 without resetting optimizer, EMA or RNG:

```zsh
STAGE=2 \
RESUME=/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v4/runs/stage-2-faithful-reconstruction/mioh-restorer-v4-latest.pth \
zsh scripts/training/run-mioh-restorer-v4-pilot.sh
```

The earlier adjacent-window overlap objective was removed. Both windows use
the exact same local five-frame context for their shared output, so that loss
was a constant Charbonnier epsilon with zero gradient while nearly doubling
the affected training step. Temporal losses now use the intersection of
adjacent ROIs and include acceleration matching.

## Evaluation

```zsh
/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/training/evaluate-mioh-restorer-v4-checkpoint.py \
  --checkpoint /path/to/mioh-restorer-v4-step-0000500.pth \
  --metadata-root \
    /Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative/validation/crop_unscaled_meta \
  --output-dir /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v4/evaluation/step-500 \
  --device mps \
  --batches 16 \
  --weights both
```

The report compares mosaic input, raw weights, and EMA weights using mosaic-ROI
PSNR, whole-image PSNR, temporal-delta error, high-frequency ROI RMSE,
confidence mean, and confidence/error correlation. The primary summary reports
ROI, temporal, and high-frequency error reduction as percentages, where a
positive number means improvement over the mosaic input. It also writes a
visual comparison image.

## Apple export

Run this with the unified Core AI Python environment:

```zsh
.venv-coreai/bin/python scripts/apple/export_mioh_restorer_v4.py \
  --checkpoint /path/to/mioh-restorer-v4-step-0000500.pth \
  --output-dir /path/to/export/v4 \
  --allow-overwrite
```

The exporter selects EMA by default and writes the portable PyTorch checkpoint,
Core ML package, Core AI asset, and a machine-readable export report.
