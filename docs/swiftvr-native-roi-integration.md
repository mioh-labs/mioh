# SwiftVR native ROI integration

Status: one-step native Swift/Core ML export (2026-09-25) passed a 3-second
MIDV-670 export, a mid-inference stop and a repeat run with identical output.
The converted model pack remains external. Do not present this as a broadly
validated ROI enhancer. The earlier visual evaluation
(`/Volumes/Project_HD/swiftvr-eval/README-evaluation.md`) failed its
face-restoration gate.

## Runtime (one step)

- Conversion may use Python offline. Export inference uses Swift and Apple
  model runtimes only. mioh never invokes PyTorch.
- `NativeFrameProcessor.process` hands each restored BasicVSR++ scene (256px
  planar FP16, after restore effects) to `SwiftVRSceneEnhancer`
  (`SwiftVRInlineEnhancer.swift`). The enhancer runs SwiftVR
  (`SwiftVRNativeClip.swift`, compiled into `mioh-native-coreai-preview`) in
  the export process, like every other ROI enhancer, and composites every
  512px or 1024px result before encoding.
  - Compositing uses the direct-replacement path shared with PiperSR:
    `base + (SwiftVR - base) * strength * mask`.
  - There is no sidecar, no second pass and no re-encode. The final movie is
    the normal export.
- Scene frames go from memory straight into Core ML. Each result frame is
  handed to the compositor as soon as it is decoded, and a frame is
  composited once its successor exists (the ±1 stabilization needs it), so
  only frames not yet composited are held instead of a whole scene (up to
  1.1 GB at 4x). On the 12 s MIDV clip (6 SwiftVR scenes, 363 frames) the
  streamed export was pixel-identical to the whole-scene one; 2x took 46–49 s
  instead of 64–67 s (the DiT groups themselves ran about 25% faster; the
  cause is not established), 4x took 145–155 s either way, and the 4x peak
  footprint fell from 14.8–15.0 GB to 14.3 GB. Until
  2026-09-26 a separate `mioh-native-swiftvr-clip --serve` worker exchanged
  them as temporary files. The in-process result is bit-identical to that
  worker's (30 frames at 2x, 100 frames at 4x), and the file exchange had
  cost well under 1% of a scene.
- Local export only, Expert ROI off. Export lanes follow the native
  parallel setting and take turns running SwiftVR; the other lanes keep
  detecting, restoring, compositing and encoding meanwhile. Crossfade stays available;
  overlap frames are simply enhanced in both batches.
- Frame smoothing ("枠のなめらかさ", 0–30 frames, default 15; 0 is off and
  pixel-identical to before). SwiftVR itself is stable: a static input
  changes its output by 0.24 levels a frame and it damps noise, but a ±3%
  scale or ±1 px position jitter of its input changes the detail it adds
  about 5x / 3x more. The detector crop grows and shrinks by up to ~8% a
  frame, so SwiftVR redrew outlines thicker and thinner from frame to frame
  (MIZD-534 54–55 s: +55–59% flicker over BasicVSR++). SwiftVR now gets the
  BasicVSR++ result through a view whose grid placement is averaged over the
  surrounding frames; outside the crop the view shows the source frame
  (blended over 4 grid px) instead of the grid's mirrored padding, whose
  seams move with the crop. Its output is mapped back to each frame's grid
  before stabilization, colour matching and compositing; BasicVSR++ is
  unchanged. On MIZD-534 50–62 s at 2x (filter 1.0, range 1) flicker went
  from +5.2% to +1.0% (54 s +54.5% → −9.3%, 55 s +58.8% → −11.6%), texture
  1–3 px from 109.7% to 107.9%; on the MIDV-670 12 s clip +1.5% → −0.3%.
  Smoothing without the source-frame surround only reached +47% → +36% at
  55 s. One second (60 s) got worse (+2.8% → +6.1%).
- Stabilization range ("なじませ範囲", 1–8 frames on each side, default 1 =
  the previous ±1). Each neighbour's SwiftVR change is added to the centre's
  BasicVSR++ result, weighted by how similar the two BasicVSR++ frames are
  at that pixel, so moving areas are left out and still areas are averaged
  longer. SwiftVR flickers most in near-still shots. On two segments of
  MIZD-534 at 2x with the noise filter at 1.0 (10–18 s and 45–57 s), the
  flicker added over BasicVSR++ went from +4.7% / +3.9% at range 1 to +3.9% /
  +2.3% at 4 and +3.2% / +1.7% at 8; in the worst second (55 s) from +47% to
  +25% and +21%. The 1–3 px texture band fell from 119.1% / 108.7% of
  BasicVSR++ to 113.5% / 106.0% at 8. Range 8 added about 1.5–2 s of
  composition to each segment.
- Temporal noise filter ("揺らぎ低減", 0–1, default 1.0; 0 is off and
  pixel-identical to no filter). SwiftVR is deterministic (no noise is
  sampled: one DiT pass at t = 1000); its flicker is small input changes
  amplified into re-synthesized texture. Apple's VTTemporalNoiseFilter
  (macOS 26, one previous and two next frames, compressed 4:2:0 input only)
  runs on the composited frames within one rectangle around every crop of
  the scene, and the result is blended back with SwiftVR's own blend
  weights. Filtering SwiftVR's 512/1024 px frames before composition worked
  much less well (4x: +19.5% → +13.9% on the old measure), because
  composition resamples them into a crop whose size and position change
  every frame. On the 12 s MIDV clip, measured only where SwiftVR changed the
  frame (15×15 mean difference > 3 levels; a plain per-pixel threshold also
  counts the scattered differences between two separate encodes), the
  flicker SwiftVR adds over BasicVSR++ went from +9.0% to +6.5% (4x) and from
  +4.7% to +2.3% (2x). It removes mainly single-pixel grain; the 1–3 px
  texture band stays above BasicVSR++ (4x 115.5% → 113.3%, 2x 111.2% →
  109.2%). The same filter applied after encoding reached +4.2% / −0.2%,
  partly by also removing the encoder's own flicker, which an export cannot
  do before encoding. Cost: about 2–4 s of composition on that clip.
- Stopping the export cancels SwiftVR between chunks and DiT groups (0.06 s
  after the stop in a 4x test). Frames of the current scene that SwiftVR
  did not produce keep the BasicVSR++ result, and the export ends exactly
  like a stop without SwiftVR (exit 0, no error event).
- SwiftVR progress lines go to stderr. This process's stdout carries mioh's
  JSON event stream.
- Core ML keeps a per-executable cache in `~/Library/Caches/<executable>`.
  Moving SwiftVR into `mioh-native-coreai-preview` rebuilt it once (a 4x test
  scene took 51 s instead of 21 s) and it reached 28 GB; the old worker's
  `mioh-native-swiftvr-clip` cache (93 GB) is no longer used.
- `SwiftVRNativeClip.swift` still builds on its own as a command-line tool
  for folders of PNG or `.f16` frames (`--serve` included); the app no longer
  ships it.
- Keep checkpoint assets external to the app. The local checkpoint is about
  19 GiB; the converted FP16 T6/T7 packs occupy about 18.4 GiB and have not
  been approved for redistribution.

- Scale: the ROI enhancer scale setting chooses 2x (512px output) or 4x
  (1024px) for every scene, whatever the ROI size. 2x needs its own pack in
  the same model root:
  - `reae-stateful-{encoder-28f,encoder-24f,decoder-7latent,decoder-6latent}-512-fp32.mlpackage`
  - `native-2x-t7-fp16/components` and `native-2x-t6-fp16/components`
  - `native-2x-fp16-grouped` (multifunction t7/t6 groups)
  The scale is chosen per scene.
- Composition fades SwiftVR's change in over 4% of the crop's short side
  (smoothstep). The previous seam taper was at most 4 px, which left a visible
  edge wherever SwiftVR shifted colour or texture.
- Temporal stabilization: SwiftVR re-synthesizes texture per frame, which
  added frame-to-frame jitter inside the ROI (+13% on a 65 s export). The
  added jitter was uniform, not at scene or chunk seams. Each output pixel is
  averaged with the neighbouring frames where the 256px BasicVSR++ base moved
  less than about 8 levels, carrying the base difference along.
- Colour match: SwiftVR shifts the ROI brighter and bluer than its input
  (+3.1/+4.1/+4.8 levels R/G/B at 4x on a 12 s excerpt). The upstream
  pipeline has no colour correction. Each stabilized frame is reduced to the
  256px grid and its difference from the restoration is box-blurred (9 px).
  That coarse shift is upsampled and subtracted. The bias fell to
  +0.2/+0.4/+0.3, the mean change from 4.80 to 1.73 levels, and added jitter
  from +5.2% to +4.3%. Detail stayed at 104% of BasicVSR++.
- Small scenes skip SwiftVR: when a scene's largest crop side is at most the
  256px restoration grid, BasicVSR++ restored it at native resolution and
  SwiftVR has no lost resolution to rebuild. It added +1.4% detail there, the
  least of any size. Those scenes keep the BasicVSR++ result, and the log says
  `SwiftVR: skipped a N-frame scene (largest crop Xpx <= 256px)`. On MIDV-995
  they were about 48% of SwiftVR jobs.
- Expert ROI is hidden from the settings and always off; its code remains.

12 s excerpt of a real export (180-frame clips), second-difference jitter
inside the SwiftVR region, relative to BasicVSR++ alone:

| | per scene | added jitter | change vs old 4x | detail vs BasicVSR++ |
| --- | --- | --- | --- | --- |
| old 4x | — | +1.03 | 100 | 108% |
| new 4x | 8–56 s | +0.34 | 68 | 104% |
| new 2x | 1–9 s | +0.06 | 66 | 98% |

The first 2x scene also compiled the 2x pack (86 s once).

The 2x rows above were measured with a corrupted 2x path and are void. Core ML
ran the 2x FP16 patch embedding on the ANE (`.all`) and computed it wrong:
mean error 0.31, max 6.5, against 0.0004 on CPU/GPU. The component exporter
had recorded that error without failing. The worker now loads every model
with `.cpuAndGPU`, and the exporter validates on CPU+GPU and fails above a
mean error of 0.01. Against the upstream PyTorch outputs on the evaluation
clip (the BasicVSR++-restored `basicvsrpp` frames):

| | before | after |
| --- | --- | --- |
| 2x | 4.96/255, 29.5 dB | 0.35/255, 47.8 dB |
| 4x | 0.35/255, 52.1 dB | 0.35/255, 52.1 dB |

One-step measurements (3 s MIDV-670 excerpt, 90 frames, M5 Pro):
- Time: 2.8 s without SwiftVR; 48.6–63.5 s with SwiftVR, including one-time
  model loading.
- The SwiftVR change is confined to the ROI: mean 3.6 levels inside, 0.27
  outside. The two-stage re-encode had changed the outside by 1.35.
- Stop: 2.1 s after the stop command, exit 0, no worker left. A repeat run was
  bit-identical.

The two-stage implementation (sidecar recorder, postprocess executable) was
removed. It remains in branch history (commits 5ef75da..e83c08e).

## Evidence gathered on M5 Pro / macOS 27.2

The experimental scripts are in `scripts/apple/probe_swiftvr_*.py`; generated
packages live outside the repository in `/Volumes/Project_HD/swiftvr-eval/`.
All differences below compare Apple runtime output with the corresponding
PyTorch graph on deterministic numeric input, not final video quality.

| Component / test | Result |
| --- | --- |
| ReAE first encoder chunk, FP32 | Core ML conversion passed; max absolute difference 0.000015 |
| ReAE stateful encoder, two 4-frame chunks, FP32 | Official PyTorch chunk results matched at 256, 512 and 1024px; at 1024px Apple output max difference 0.000019 / 0.000013 |
| ReAE stateful decoder, two 1-latent chunks, FP32 | Official first/continued chunks matched at 256 and 1024px after replacing unsupported nearest-3D upsample with equivalent repeat; at 1024px Apple output max difference 0.000010 / 0.000008 |
| ReAE stateful encoder, two 24-frame chunks, 1024px FP32 | Official chunk behavior matched; Apple max difference 0.000030 / 0.000026 |
| ReAE stateful encoder, two 28-frame chunks, 1024px FP32 | Official chunk behavior matched; Apple max difference 0.000030 / 0.000027 |
| ReAE stateful decoder, two 6-latent chunks, 1024px FP32 | Official chunk behavior matched; Apple max difference 0.000013 / 0.000019 |
| ReAE stateful decoder, two 7-latent chunks, 1024px FP32 | Official chunk behavior matched; Apple max difference 0.000023 / 0.000021 |
| Trained DiT block 0, unshifted | Conversion and inference passed on small input |
| Trained DiT block 1, shifted | Conversion and inference passed including a 17×17 window-boundary case |
| Trained DiT block 1, 512px-output shape (1,536 tokens) | FP32 mean difference 0.000034; runtime not measured |
| Trained DiT block 1, 1024px-output shape (6,144 tokens) | FP32 mean difference 0.000033; warm prediction 0.304 s |
| Same 1024px-output shape, FP16 | Mean difference from FP32 PyTorch 0.00365; warm prediction 0.096 s; visual impact unknown |
| Fixed condition, patch embedding and output head | Condition tensors exported; FP16 patch/head mean differences 0.00010 / 0.00056 at the 1024px-output shape |
| Swift executable loading trained block 1 | `tests/apple/SwiftVRDiTCoreMLCanary.swift` matches the Python Core ML parity result at the 6,144-token shape (mean 0.00365) |
| Thirty trained DiT block packages, 6-latent/1024px shape, FP16 | All 30 converted and independently predicted; approximately 9.2 GiB external asset directory. Block 29 reached a max difference of 2.50 on unconditioned random input, so the conditioned chain comparison below matters more. |
| Swift 30-block chain, actual checkpoint patch weights + fixed inference conditioning | Ran every converted block in order at the 6-latent/1024px shape; final mean error 0.001685 versus upstream PyTorch, 0.274% of reference mean absolute signal, max error 0.1889. Approximately 57.6 seconds including package compilation/loading on M5 Pro. This is tensor-level parity, **not** a restored-video comparison. |
| Swift 30-block chain, 7-latent/1024px shape | Final mean error 0.001995 versus upstream PyTorch, 0.310% of reference mean absolute signal, max error 0.3056. |
| Real MIDV-670 BasicVSR++ ROI, first 25 frames at 1024px | The Swift/Core ML canary produced 25 PNG frames in 73.3 seconds. Against the upstream Python/MPS output from the identical `/Volumes/Project_HD/swiftvr-eval/midv670-real800/basicvsrpp256` input, RGB MAE is 0.515/255 and mean frame PSNR is 50.85 dB (minimum 48.82 dB). This proves one-chunk video parity, not scene-boundary handling or mioh integration. |
| Real MIDV-670 BasicVSR++ ROI, 49 frames at 1024px | The Swift/Core ML canary produced 49 PNG frames in 152.4 seconds across two ReAE chunks, carrying encoder/decoder states and using temporal RoPE offset 6 in the second chunk. The change in restored residual across the frame-25 seam was 1.025 times the local median; no obvious numerical jump was measured. The first 25 frames matched the separate native 25-frame run at mean PSNR 60.79 dB. This is not yet a validation of arbitrary scene lengths or native app integration. |
| Two-stage native mioh export on a 1.7017s MIDV-670 clip | BasicVSR++ completed a 51-frame, 1920×1080 H.264/AAC restored movie before SwiftVR started. Five ROI scenes (48+11+7+3+3 frames) completed and a 51-frame H.264/AAC final movie was muxed with the same 1.7017s video and 1.700s audio duration. Total elapsed was 511s; almost all time was SwiftVR, not the extra video pass. The test output is `/Volumes/Project_HD/swiftvr-eval/midv670-real800/native-app-swiftvr-test/output.mp4`. |
| Retry of only the second pass | With saved SwiftVR scene output, re-composition and final mux completed in under one second to `output-high-bitrate.mp4`; no BasicVSR++ or SwiftVR inference was rerun. |
| Bounded second-pass cache | Re-composition with default cache cleanup reduced the 1.7s test sidecar from 482MB to 50MB and removed all 72 high-resolution `.f16` files, while retaining the ROI inputs and mask/geometry manifest. |
| 51-frame image/sync comparison | Restored and final MP4s both have 51 frames, video duration 1.7017s, and AAC duration 1.700s, with both streams starting at zero. At the higher second-pass bitrate the mean absolute RGB difference from the restored movie was 3.064/255 inside the ROI mask and 1.347/255 outside across the clip. The outside difference is caused by the required full-frame re-encode, not direct SwiftVR compositing. |
| Final dedicated build, temporal overlap | A 12-frame MIDV-670 excerpt with 8-frame batches and two overlapping context frames completed in 178.2s. The sidecar marked the second scene's first two frames as context-only (8+4 output-eligible frames), and the final H.264/AAC movie contains exactly 12 frames, with 0.4004s video/0.4000s audio both starting at zero. No high-resolution `.f16` cache remained; the ROI sidecar occupied 11MB. The signed dedicated app passed 14 Core AI and 14 Core ML asset checks. |

The DiT's in-place residual and RoPE operations are rewritten out of place for
conversion. The probe compares the rewritten block to upstream before exporting.
Fixed input shapes are traced; the model cannot accept arbitrary frame counts
or output sizes without additional variants or a validated dynamic export.

## Runtime structure and GPU utilization (2026-09-25)

- Compiled Core ML models persist in
  `~/Library/Caches/com.okatti.lada.coreai/mioh/swiftvr-compiled`, or in the
  directory named by `MIOH_SWIFTVR_COMPILED_CACHE`. The cache key includes each
  package's path plus the size and modification time of its model and
  weights, so a replaced pack is recompiled. Entries unused for 30 days are
  pruned. The cache occupies about 9.5 GB per model pack. Previously a
  per-run temporary cache was deleted after every export, so every export
  recompiled about 34 packages while the GPU sat idle.
- One process handles every scene of an export, one scene at a time (at
  first a long-lived worker, now the export process itself). This replaced a
  worker process per scene.
- Models load for each use; none stay resident. A loaded DiT block holds far
  more than its 312 MB of weights. Keeping the stack resident reached 42–63 GB
  and swapped, while on-demand loading ran a warm 49-frame scene in 16.7 s.
  Every prediction, chunk and scene runs inside an autorelease pool. Without
  the pools a long-lived process grew until the system swapped.
- Five-scene MIDV-670 sidecar (48+11+7+3+3 frames, M5 Pro, GPU sampled from
  ioreg): the previous code took 132.7 s at 39% mean GPU. The new code took
  87.8 s at 57% on its first export (cache build) and 55.8 s at 74% once
  cached. All 144 high-resolution frames were bit-identical to the previous
  code.
- Loading the next DiT block overlaps with running the current one. The
  256→1024 input upscale runs on all CPU cores (0.03 s per chunk; a Metal
  version measured slower). A warm 49-frame scene went from 21.4 s to 15.8 s,
  with bit-identical output.
- Optional grouped stack: `scripts/apple/export_swiftvr_dit_group_coreml.py`
  exports layers as multi-layer Core ML programs into
  `<model-root>/native-4x-t7-fp16-grouped/dit-group-AA-BB-t7-4x-float16.mlpackage`.
  Five 6-layer groups take about 100 s each and 5.5 GB peak to convert, and
  occupy 9 GB. When present, SwiftVR loads them once and keeps them
  resident, using about 8.5 GB instead of the 42–63 GB of 30 resident blocks.
  - Export clips reach 180 frames, whose middle chunks need the 6-latent
    stack. `--latent-frames 7 6` writes each group as one multifunction
    package with `t7` and `t6` functions into
    `<model-root>/native-4x-fp16-grouped/dit-group-AA-BB-4x-float16.mlpackage`.
    SwiftVR prefers that directory and falls back to the t7-only one.
  - The functions share their weights on disk (1.8 GB per group, the same as
    t7 alone) and in memory: adding t6 to a loaded group raised the footprint
    by 0.48 GB, against about 1.5 GB for a separate package. The t6 function
    runs six layers in 0.45 s.
  - A 180-frame scene took 45–49 s instead of 58–65 s (6-latent chunks loaded
    block by block), with all 180 output frames bit-identical. Peak footprint
    was 29.4 GB (28.0 GB before), almost all of it the 1024px ReAE graphs.
  - ReAE slices: with `reae-stateful-encoder-4f-<size>-fp32` and
  `reae-stateful-decoder-1latent-<size>-fp32` in the pack, each chunk is
  encoded four frames and decoded one latent at a time, carrying the states.
  A whole-chunk 1024px decoder call needed 15-20 GB of transient memory. In an
  app export on a 48 GB Mac, that evicted the idle DiT groups to swap, and a
  group then took 10-15 s instead of 0.5 s.
  180-frame scene:

  | scale | peak | swap written | time | output vs whole-chunk |
  | --- | --- | --- | --- | --- |
  | 4x | 29.3 → 10.5 GB | 1463 → 106 MB | 42.1 → 36.0 s | identical |
  | 2x | 10.6 → 3.1 GB | 192 → 0 MB | 9.0 → 8.7 s | mean 0.01, max 2.2 levels |

- Keeping the ReAE encoders and decoders resident is counterproductive: the
    same scene reached a 49.8 GB footprint and took 123–135 s.
  - One group's conversion once failed with "Caught an unknown exception"
    while validating t6, and succeeded on retry.
  - Speed and output: 30 layers run in 2.8 s per chunk. A warm 49-frame scene
    takes 9.5–10.9 s, versus 12.8 s per block, with output bit-identical to
    the per-block stack.
  - Loading the five groups costs about 20–30 s once per export.
  - The same group exported to Core AI was 1.8x slower (0.51 s vs 0.29 s for
    three layers) at equal accuracy.
- One uncatchable Core ML "MPSGraph unexpected rank" abort was seen during
  development and did not reproduce. The worker used to be restarted once
  for it; in the export process such an abort would end the export.
- Still open: a scene shorter than 25 frames is padded to a full
  28-frame/7-latent chunk. A 4-frame scene costs the same 10.8 s as a
  25-frame one. Removing this needs smaller exported graph variants.

## Remaining release gates

1. Validate the generalized clip runner at short, odd and three-chunk scene
   lengths. The 25- and 49-frame MIDV-670 paths have passed, but the tail
   padding and third-chunk handling need separate upstream comparison.
2. Measure peak memory, cold/warm runtime and Core ML cache growth for the
   complete 30-block pipeline. Select FP16 only after video-level comparison.
3. Validate the export-only UI and stop/retry handling in the app, not just
   the subprocess CLI. The model-less build used the installed detection and
   restoration model assets by absolute path.
4. Validate several real clips, including long scenes and scene boundaries.
   The short test only establishes a functional two-stage path, not a quality
   improvement across all content. The additional full-frame encode causes a
   small outside-ROI difference even though SwiftVR changes are masked.

The real MIDV-670 test input is documented by
`/Volumes/Project_HD/swiftvr-eval/prepare_midv670_roi.py`: frames 150–198
of `MIDV-670-2h18m00s-source-10s.mp4`, crop x=700, y=280, w=800,
h=800, followed by the shipped BasicVSR++ stage2.19 restoration to 256px.
Do not use the unrelated `/Volumes/Project_HD/swiftvr-eval/basicvsrpp/`
directory for this verification.

The earlier proof-of-concept clip at
`/Volumes/Project_HD/swiftvr-eval/midv670-real800/comparison.mp4` used the
Python SwiftVR implementation and a manually selected ROI. It does not satisfy
any of the native app integration gates.
