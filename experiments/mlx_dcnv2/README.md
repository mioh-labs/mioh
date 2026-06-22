# MLX DCNv2 Experiment

This directory contains an experimental MLX implementation of the LADA
BasicVSR++ inference subset. It intentionally avoids training, generic MMagic
compatibility, and BasicVSR++ features that are not needed by LADA.

## Current Scope

- Inference-only forward pass.
- Float32 MLX arrays.
- `mx.fast.metal_kernel` deformable im2col plus MLX `matmul` for the BasicVSR++
  `groups=1` alignment shape.
- MLX implementation of the BasicVSR++ second-order deformable alignment
  offset-conv stack plus DCNv2 forward.
- MLX `flow_warp`, feature extraction, reconstruction, and the four propagation
  branches used by LADA: `backward_1`, `forward_1`, `backward_2`, `forward_2`.
- MLX SPyNet core for inputs already sized to multiples of 32.
- MLX SPyNet forward wrapper for arbitrary ROI sizes via resize-to-multiple-of-32.
- A minimal short-sequence LADA inference path that composes feature extraction,
  SPyNet flow, four-branch propagation, and reconstruction.
- Masked ROI crop/restore/composite helpers that connect detection masks to
  the MLX LADA sequence path.
- Windowed video restore fixture that keeps only a short temporal window plus
  overlap in memory and prints restored frame counts.
- Correctness is checked against `torchvision.ops.deform_conv2d`.
- This is not wired into LADA runtime yet.

## Why Not Directly Patch PyTorch MPS?

MLX 0.31.2 does not expose DLPack import/export in `mlx.core`, so PyTorch MPS
tensors cannot currently be handed to MLX without a copy-oriented bridge. The
practical route is to use this as a building block for an Apple/MLX BasicVSR++
path rather than replacing one PyTorch op in-place.

## BasicVSR++ ROI Matrix

Command:

```sh
python -m experiments.mlx_dcnv2.bench_forward \
  --matrix \
  --iters 5 \
  --json-output /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/basicvsrpp_roi_matrix.json \
  --csv-output /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/basicvsrpp_roi_matrix.csv
```

Latest local result on this Mac:

| ROI | MLX Metal | torchvision CPU | Speedup | Max Abs Error |
| --- | ---: | ---: | ---: | ---: |
| 16x16 | 0.556 ms | 1.910 ms | 3.43x | 7.05719e-05 |
| 32x32 | 1.398 ms | 8.495 ms | 6.07x | 9.15527e-05 |
| 64x64 | 3.080 ms | 42.536 ms | 13.81x | 2.67029e-05 |
| 96x96 | 6.133 ms | 61.408 ms | 10.01x | 2.28882e-05 |
| 128x128 | 9.944 ms | 175.250 ms | 17.62x | 2.67029e-05 |

Shape is `1x128xHxW -> 64xHxW`, `kernel=3x3`, `deform_groups=16`, matching
BasicVSR++ deform alignment.

## Weight Export

Command:

```sh
python -m experiments.mlx_dcnv2.export_deform_alignment \
  model_weights/lada_mosaic_restoration_model_generic_v1.2.pth \
  --output-dir /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/generic_v1_2_deform_alignment \
  --prefix generator_ema
```

Exports the LADA inference subset into NPZ files and a manifest:

- `generator_ema.feat_extract.npz`
- `generator_ema.spynet.npz`
- `generator_ema.deform_align.{backward_1,forward_1,backward_2,forward_2}.npz`
- `generator_ema.backbone.{backward_1,forward_1,backward_2,forward_2}.npz`
- `generator_ema.reconstruction.npz`
- `deform_alignment_manifest.json`

Each deform alignment module has DCNv2 weight shape `[64, 128, 3, 3]` and
`deform_groups=16`.

## Second-Order Alignment

The exported `generator_ema` weights can be loaded directly into the MLX
second-order deformable alignment experiment:

```sh
python -m experiments.mlx_dcnv2.bench_alignment \
  --height 64 \
  --width 64 \
  --npz /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/generic_v1_2_deform_alignment/generator_ema.deform_align.backward_1.npz \
  --iters 3
```

Latest local result with the real `generic_v1.2` `backward_1` alignment weights:

| ROI | MLX Alignment | PyTorch CPU Alignment | Speedup | Max Abs Error |
| --- | ---: | ---: | ---: | ---: |
| 32x32 | 3.387 ms | 13.664 ms | 4.03x | 1.72853e-05 |
| 64x64 | 8.582 ms | 66.769 ms | 7.78x | 2.86102e-05 |

This benchmark includes the BasicVSR++ offset prediction conv stack, mask
generation, flow offset addition, and the DCNv2 forward pass.

## LADA Inference Components

The MLX LADA inference pieces now match PyTorch fixtures for:

- feature extraction
- `flow_warp`
- SPyNet basic module and core pyramid flow computation
- second-order deformable alignment
- branch backbones
- one propagation step
- four-branch propagation order
- reconstruction and x4 pixel shuffle output

The real-weight benchmark artifacts are saved here:

- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/feature_extract_real_weight_bench.txt`
- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/backbone_real_weight_bench.txt`
- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/propagation_step_real_weight_bench.txt`
- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/reconstruction_real_weight_bench.txt`
- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/sequence_real_weight_smoke.txt`
- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/masked_roi_restore_real_weight_smoke.txt`

Latest local real-weight results:

| Component | MLX Time | Max Abs Error |
| --- | ---: | ---: |
| Feature extract `256x256 -> 64x64` | 8.005 ms | 1.23978e-05 |
| Backbone `backward_1` | 13.531 ms | 5.36442e-06 |
| Propagation step `backward_1` | 19.055 ms | checked by fixture |
| Reconstruction `64x64 -> 256x256` | 27.841 ms | 2.90871e-05 |
| Sequence smoke `2x128x128` | 289.539 ms | shape/runtime smoke |
| Masked ROI restore `2x256x256, ROI 128x128` | 260.109 ms | shape/runtime smoke |

## Video Fixture Restore

Short clips can be restored from a video plus per-frame grayscale masks:

```sh
python -m experiments.mlx_dcnv2.run_restore_fixture \
  --manifest /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/generic_v1_2_deform_alignment/deform_alignment_manifest.json \
  --video-input input.mp4 \
  --mask-glob 'masks/*.png' \
  --video-output restored.mp4
```

For longer clips, use the streaming window path to avoid loading the whole
video into memory:

```sh
python -m experiments.mlx_dcnv2.run_restore_fixture \
  --manifest /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/generic_v1_2_deform_alignment/deform_alignment_manifest.json \
  --video-input input.mp4 \
  --mask-glob 'masks/*.png' \
  --video-output restored.mp4 \
  --video-window-size 15 \
  --video-window-overlap 4 \
  --max-restore-roi-area 262144 \
  --print-window-timing \
  --copy-audio
```

The streaming path writes each frame once, carries the overlap frames only as
temporal context, and prints `restored frames: N` after each window. `--copy-audio`
muxes the optional source audio track back into the restored MP4 with FFmpeg.

To keep detection on LADA's native YOLO path and only use the MLX experiment for
restoration, pass the original `.pt` detection model instead of `--mask-glob`:

```sh
python -m experiments.mlx_dcnv2.run_restore_fixture \
  --manifest /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/generic_v1_2_deform_alignment/deform_alignment_manifest.json \
  --video-input input.mp4 \
  --native-detection-model model_weights/lada_mosaic_detection_model_v4_fast.pt \
  --native-detection-device mps \
  --generated-mask-dir generated_masks \
  --video-output restored.mp4 \
  --video-window-size 20 \
  --video-window-overlap 4 \
  --max-restore-roi-area 262144 \
  --print-window-timing \
  --copy-audio
```

In this mode, `Yolo11SegmentationModel` generates one grayscale mask PNG per
frame, then the MLX BasicVSR++ subset restores those masked ROIs.
The default ROI policy on this branch intentionally does not expand masks
(`--expansion-ratio 0.0`) and aligns ROIs to 32 pixels.
`--max-restore-roi-area` splits a temporal window when the union mask ROI grows
too large, which avoids the huge-ROI slowdown and Metal OOM behavior seen with
longer windows.

Latest 20-second native-detection + MLX-restore fixture:

```sh
/usr/bin/time -p /Users/okatti/.pyenv/versions/lada/bin/python -m experiments.mlx_dcnv2.run_restore_fixture \
  --manifest /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/generic_v1_2_deform_alignment/deform_alignment_manifest.json \
  --video-input /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/native_detection_20sec/input_mida176_42min_20sec_30fps.mp4 \
  --mask-glob '/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/native_detection_20sec/generated_masks/mask_*.png' \
  --video-output /Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/native_detection_20sec/restored_native_detection_mlx_20sec_opt_w20.mp4 \
  --video-window-size 20 \
  --video-window-overlap 4 \
  --max-restore-roi-area 262144 \
  --print-window-timing \
  --copy-audio
```

Result on this Mac: 600 frames in 424.34 seconds, about 1.41 fps end-to-end,
with audio muxed back into the MP4. The previous 20-second run with larger
expanded ROIs took 860.92 seconds, so this no-expansion + ROI-split policy was
about 2.0x faster and avoided Metal OOM.

Output:

- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/native_detection_20sec/restored_native_detection_mlx_20sec_opt_w20.mp4`
- `/Users/okatti/Desktop/lada_coreml_test/mlx_dcnv2/native_detection_20sec/run_restore_mlx_20sec_opt_w20.log`

## Remaining Work

1. The path is functional for LADA-native detection plus MLX BasicVSR++ restore,
   but still far slower than the production PyTorch/MPS path.
2. The main remaining target is sequence-level performance for large ROIs:
   propagation/backbone work dominates once the mask union grows.
3. This branch intentionally excludes Apple sidecar/CoreML detection plumbing;
   use LADA native `.pt` detection here.
