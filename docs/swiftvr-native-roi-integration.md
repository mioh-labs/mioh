# SwiftVR native ROI integration — conversion gate

Status: two-stage native Swift/Core ML export passed one end-to-end MIDV-670
clip in an isolated model-less app. The converted model pack remains external,
and the installed `/Applications/mioh.app` is unchanged. Do not present this
as a broadly validated or installed ROI enhancer.

## Intended runtime

- Conversion may use Python offline; playback/export inference must use Swift
  and Apple model runtimes only. Do not embed the test virtual environment or
  invoke PyTorch from mioh.
- Complete the ordinary BasicVSR++ export first, saving the independently
  playable `<name>.restored.mp4`. While restoring, persist each 256px FP16
  scene, its detector/composite mask, ROI geometry and source PTS in
  `<name>.swiftvr-roi/`. Write `manifest.json` only after the restored movie
  has been finalized. Then launch the separate SwiftVR postprocess executable.
- Keep temporal overlap for the first pass. Crossfade is disabled for this
  export mode. The sidecar marks overlap-prefix frames that were used only as
  temporal context, so the second pass does not composite them a second time.
  Expert ROI is not yet supported.
- Run SwiftVR in one export lane initially. Its 30-layer DiT is not a per-frame
  model, and starting an independent copy per lane would multiply the model's
  memory demand. Do not enable it for realtime playback until latency is
  measured end to end.
- Composite the 1024px SwiftVR output over the completed first-pass movie
  using the saved 256px input as a difference base and the saved ROI mask.
  The base movie is not modified. The final movie is encoded separately and
  retains audio from the first-pass movie.
- The second pass now infers scenes only when their first output frame is
  reached, and deletes each high-resolution scene cache after its last frame
  is composited. The 256px ROI inputs, masks, manifest and completed first-pass
  movie remain for retry. Set `MIOH_SWIFTVR_KEEP_HIGH_CACHE=1` only when a
  fast re-composite is needed for testing; this can consume substantial disk.
- Keep checkpoint assets external to the app. The local checkpoint is about
  19 GiB; the converted FP16 T6/T7 packs occupy about 18.4 GiB and have not
  been approved for redistribution.

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
- One `mioh-native-swiftvr-clip --serve` worker handles every scene of an
  export. It still receives scenes one at a time, over stdin, when their
  first output frame is reached. This replaces a worker process per scene.
- Models load for each use; none stay resident. A loaded DiT block holds far
  more than its 312 MB of weights. Keeping the stack resident reached 42–63 GB
  and swapped, while on-demand loading ran a warm 49-frame scene in 16.7 s.
  Every prediction, chunk and scene runs inside an autorelease pool. Without
  the pools the long-lived worker grew until the system swapped.
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
  occupy 9 GB. When present, the worker loads them once and keeps them
  resident, using about 8.5 GB instead of the 42–63 GB of 30 resident blocks.
  Scenes are at most 48 frames, so only the 7-latent stack is used.
  - Speed and output: 30 layers run in 2.8 s per chunk. A warm 49-frame scene
    takes 9.5–10.9 s, versus 12.8 s per block, with output bit-identical to
    the per-block stack.
  - Loading the five groups costs about 20–30 s once per export.
  - The same group exported to Core AI was 1.8x slower (0.51 s vs 0.29 s for
    three layers) at equal accuracy.
- If the worker dies mid-scene (one uncatchable Core ML
  "MPSGraph unexpected rank" abort was seen and did not reproduce), the
  postprocess restarts it once and retries that scene.
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
