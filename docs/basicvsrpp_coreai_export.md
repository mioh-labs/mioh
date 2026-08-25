# BasicVSR++ Core AI Export Probe

`scripts/apple/export_basicvsrpp_coreai.py` diagnoses conversion of the full
BasicVSR++ v1.2 EMA generator to fixed FP16 Core AI models. The supported input
and output frame counts are T18, T36, and T90 at 256 by 256.

## Environment

Core AI Torch 0.4.2 requires PyTorch 2.11 or older, while Lada's normal Apple
extra pins PyTorch 2.12. Keep the conversion dependencies isolated:

```bash
uv venv .venv-coreai --python 3.12
uv pip install --python .venv-coreai/bin/python \
  -e . 'coreai-torch==0.4.2' 'torch==2.11.0' 'torchvision==0.26.0'
```

The probe also requires Xcode 27 and its Metal toolchain:

```bash
xcodebuild -version
xcrun --find metal
xcrun --find coreai-build
```

After changing macOS, Xcode, Core AI Torch, or Core AI Core, verify native
mutable-state lowering and both source/compiled Swift execution:

```bash
.venv-coreai/bin/python scripts/apple/canary_coreai_native_state.py
```

The report is written to
`/tmp/mioh-coreai-native-state-canary/report.json`. This canary does not alter
the shipping BasicVSR++ assets by itself. Production adoption is decided by
the end-to-end MIDV-670 benchmark below.

### Native recurrent-state A/B

`scripts/apple/benchmark_basicvsrpp_native_state.py` compares the shipping
forward_2 continuation asset, a freshly exported explicit-I/O control, and a
native-state candidate at the production `[6, 256, 64, 64]` context shape. It
uses asynchronous Metal buffers for all three paths:

```bash
.venv-coreai/bin/python scripts/apple/benchmark_basicvsrpp_native_state.py
```

On the M5 Pro, six isolated-process A-B-B-A pairs with Core AI Torch 0.4.2
measured 8.139 ms for explicit boundary I/O and 8.191 ms for native state, a
0.994x change. That single-block microbenchmark did not predict the complete
pipeline result. On the actual 300-frame MIDV-670 mosaic clip, median
restoration time fell from 5.223 s to 4.512 s (13.6%) and median wall time fell
from 7.330 s to 6.646 s (9.3%). After visual acceptance, production adopted
native state for all four continuation assets on 2026-08-25. The start, flow,
spatial, and reconstruction assets remain unchanged.

The durable end-to-end report is
`output/evaluations/basicvsrpp-native-state-midv670-20260825/report.json`.

The freshly converted explicit control differed from the previous shipping
compiled asset by as much as 0.008789. The adopted decision therefore used a
matched rebuild and an end-to-end video comparison rather than attributing the
entire difference to state alone.

## Export

The complete CPU reference inference is useful for output validation but is
very slow. Use `--skip-reference-inference` for the normal export path:

```bash
.venv-coreai/bin/python scripts/apple/export_basicvsrpp_coreai.py \
  --model model_weights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --output model_weights/basicvsrpp-v1.2-t18-fp16.aimodel \
  --frames 18 \
  --imgsz 256 \
  --skip-reference-inference \
  --allow-overwrite \
  --verbose
```

For the fixed-T36 asset, `--frames 36` automatically selects
`model_weights/basicvsrpp-v1.2-t36-fp16.aimodel`:

```bash
.venv-coreai/bin/python scripts/apple/export_basicvsrpp_coreai.py \
  --model model_weights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --frames 36 \
  --skip-reference-inference \
  --allow-overwrite \
  --verbose
```

Use `--frames 90` to export the fixed-T90 asset. T90 provides a longer temporal
window and should be paired with `--max-clip-length 178` so two windows cover
each clip without a one-frame third window.

The checkpoint enables EMA inference. The exporter must therefore select
`generator_ema`, matching `BasicVSRPlusPlusGan.forward_tensor`; exporting
`generator` produces a valid but different restoration model.

The exporter replaces BasicVSR++ flow warping and modulated deformable
alignment with inline Metal 4 kernels. It does not modify the normal Lada
runtime. The command writes a 76 MB `.aimodel` and an adjacent report containing
the checkpoint SHA-256, package versions, stage timings, custom kernels, and
exported operator counts.

## Metal kernel execution smoke test

The unit tests verify the PyTorch references and Core AI conversion graph. Run
the compiled-kernel smoke test after an Xcode/Core AI beta update to execute the
actual Metal 4 grid-sample and TensorOps DCNv2 kernels on this Mac:

```bash
.venv-coreai/bin/python \
  scripts/apple/smoke_test_basicvsrpp_coreai_kernels.py \
  --operation all \
  --runner build/macos-standalone/mioh.app/Contents/Resources/bin/lada-coreai-runner
```

The command fails when either kernel exceeds `0.002` maximum absolute error
against the FP16 PyTorch reference. On the M5 Pro with Xcode 27 build
27A5228h, grid sample at `[1,64,64,64]` measured 116.30 dB and 0.81 ms median.
DCNv2 with the production BasicVSR++ shape `[1,128,64,64] -> [1,64,64,64]`
measured 84.46 dB and 2.16 ms median. Both had `0.00048828125` maximum absolute
error.

### Experimental fused flow warp (not adopted)

`--fuse-flow-warp` is a research-only export option. It combines the
meshgrid, flow addition, coordinate normalization, and bilinear sampling into
one Metal kernel. The fixed-T18 graph became 29.2% smaller and an isolated
random-input fixed model ran 13.7% faster, so the optimization looked
promising in a synthetic probe.

The recurrent and real-video gates rejected it. With the iter9000 weights and
the production chunk6 Swift runner, the current model measured 109.11 fps at
T18 and 114.13 fps at T90. The fused model measured 99.06 and 103.84 fps. On
five real validation clips, median full-frame parity against MPS fell from
73.30 to 44.75 dB and median mosaic-ROI parity fell from 66.61 to 38.61 dB.
The standalone fused kernel itself remained numerically valid at 105.02 dB
against its high-precision reference; the failure is the small rounding change
being amplified by recurrent propagation, not a broken Metal sampler.

The normal fixed and variable exporters therefore retain the validated
grid-construction plus grid-sample path. The standalone app build does not pass
`--fuse-flow-warp`. Keep the option only for future Core AI/compiler canaries;
it must pass the real-video recurrent gate before it can become a default.

## Ahead-of-time Compilation

```bash
xcrun coreai-build compile \
  model_weights/basicvsrpp-v1.2-t18-fp16.aimodel \
  --output /tmp/basicvsrpp-v1.2-t18-coreai-compiled \
  --platform macOS \
  --min-deployment-version 27.0 \
  --preferred-compute gpu
```

Xcode 27 successfully compiles the asset for all 20 default Mac hardware
targets.

## Measured Result

The exported graph contains 208 custom grid-sample calls and 68 custom
modulated-deform-convolution calls. Neither
`aten.grid_sampler_2d.default` nor `torchvision.deform_conv2d.default` remains.

The deformable-convolution kernel samples eight output pixels into an 18 KiB
FP16 threadgroup tile, then multiplies that `[8, K]` tile by the `[K, 64]`
weight matrix with Metal 4 TensorOps and eight SIMD groups. Offset, mask and
bilinear-coordinate calculations are shared across the eight input channels
in each production deform group; the previous kernel repeated those operations
for every channel. Bias addition and the final NCHW reshape stay in the Core AI
graph.

On the M5 Pro, a 50-run same-environment comparison at the production
`[1,128,64,64] -> [1,64,64,64]` shape reduced median DCNv2 latency from
2.63 ms to 2.16 ms (17.8%). An ABBA comparison of the complete iter9000
chunk6 model improved T18 throughput from 99.45 to 108.07 fps (8.7%) and T90
from 105.02 to 109.85 fps (4.6%). Outputs were bit-identical at T18 and T90,
including all five real-video validation clips. A 12-row TensorOps tile is not
valid because the Metal primitive requires its M dimension to be a multiple of
8 or 16; a 16-row tile would exceed the 32 KiB threadgroup-memory budget with
the 1152-element reduction, so the validated 8-row tile remains intentional.

On this Mac, random T18 runs measured about 15.2 seconds for first-load
specialization and 1.36 to 5.73 seconds for inference. The output was FP16 with
shape `[1, 18, 3, 256, 256]` and contained no NaN or Inf values. The standalone
Metal kernels were also compared against PyTorch: grid sample had zero maximum
error, while deformable convolution had 0.00195 maximum and 0.00044 mean
absolute error.

## Lada Runtime

Run Lada from the isolated Core AI environment and select the registered
restoration model:

```bash
.venv-coreai/bin/python -m lada.cli.main \
  --input input.mp4 \
  --output output.mp4 \
  --device mps \
  --fp16 \
  --encoding-preset hevc-apple-gpu-balanced \
  --mosaic-restoration-model basicvsrpp-v1.2-coreai \
  --mosaic-detection-model v4-fast
```

Use `basicvsrpp-v1.2-coreai-t36` to select the fixed-T36 asset. The Core AI
backend pads short chunks to the selected fixed size and uses a two-frame
crossfade between longer chunks. `--restore-max-frames` does not change the
selected Core AI contract.

For clips longer than the fixed contract, a trailing partial chunk is evaluated
using the final complete T18/T36 window and only the required tail outputs are
kept. Clips shorter than the contract still repeat the final frame for padding;
their quality impact has not yet been benchmarked, and there is no PyTorch
fallback in the Core AI runtime.

Long clips submit fixed-shape windows through an ordered stream with up to two
Core AI calls in flight. This keeps the GPU fed while preserving the existing
window order and crossfade. On the 60-second MIDA-726 benchmark, fixed-T18
runtime fell from 97.49 to 95.47 seconds. The streamed and original outputs
were pixel-identical after decoding (`PSNR=inf`, 1,800 frames).

`process_video_parallel.py` also chooses padding-free Clip lengths when
`--max-clip-length` is omitted: 98 frames for T18, 104 frames for T36, and 178
frames for T90. These
lengths keep detection, restoration, and composition moving in shorter bursts
and align exactly with each model's two-frame-overlap stride. Explicit values
are always preserved. On the same 60-second benchmark, T18/98 completed in
88.06 seconds and T36/104 in 86.44 seconds; the previous T36/180 run took 98.82
seconds.

The T36 Batch 2 experiment was not adopted. It reduced the warm 60-second
T36/104 run from 86.44 to 83.94 seconds (2.9%), but its first run took 128.33
seconds because specialization added about 40 seconds. The active runtime
therefore remains Batch 1 with two ordered calls in flight.

The original direct end-to-end comparison on the same 60-second MIDA-726 sample
was 46.88 seconds for PyTorch/MPS FP16 and 86.44 seconds for the scalar Core AI
T36/104 path. Threadgroup sample sharing reduced the warm Core AI run to 52.64
seconds. The adopted TensorOps kernel completed its first run in 66.19 seconds,
including about 46 seconds of one-time kernel specialization, and its warm run
in 22.20 seconds. That initial warm TensorOps path was 57.8% faster than the
threadgroup SIMD path and 52.6% faster than PyTorch/MPS on this benchmark.

Increasing the TensorOps execution scope from four to eight SIMD groups reduced
the warm T36 call from roughly 0.325 to 0.318 seconds without changing any FP16
output values. Expanding grid-sample threadgroups from 8 by 8 to 16 by 16 also
preserved bit-identical FP16 output. Together these changes reduced the
one-minute end-to-end run from 22.20 to 21.62 seconds, 58.9% faster than
threadgroup SIMD and 53.9% faster than PyTorch/MPS. M12 was rejected because
TensorOps requires M to be a multiple of 8 or
16; direct M16 exceeded the 32 KiB threadgroup limit. M16 with K split into
576-element chunks, relaxed precision, and shared deform-coordinate tables all
compiled and ran, but none improved the warm T36 runtime.

Against the previous threadgroup SIMD output, the decoded TensorOps video
measured 49.34 dB PSNR and 0.9938 SSIM. A direct fixed-random-input comparison
of the two T36 model outputs measured maximum absolute error 0.00390625, mean
absolute error 0.00010339, and 77.86 dB PSNR.

An earlier quality audit accidentally compared a Core AI export of `generator`
against PyTorch/MPS inference from `generator_ema`. On an actual pipeline T36
window, the mismatched export measured 45.15 dB against EMA but 67.44 dB
against the non-EMA generator. Selecting EMA in the exporter corrected T36 to
69.33 dB against MPS, with 0.028 mean 8-bit-level error.

On the 178-frame SNOS-252 crop, corrected Core AI T90 measured 69.03 dB against
MPS T90. Against unbounded MPS it measured 47.29 dB, effectively matching the
47.33 dB of MPS T90 itself. The remaining difference is therefore the fixed
T90 window, not Core AI numerical precision. The first T90 two-window call took
272.38 seconds including specialization and the warm call took 1.78 seconds.
