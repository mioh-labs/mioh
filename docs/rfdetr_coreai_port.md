# RF-DETR Seg Core AI port

This backend supports fixed-shape RF-DETR 1.8.3 segmentation models:

- Seg Small at 384×384 for architecture tests;
- Jasna `rfdetr-v6` (Seg Medium) at 576×576 for mosaic detection.
- Jasna `rfdetr-v6-large` (Seg Large) at 768×768 for higher-quality
  4K mosaic detection.

## Design

- RF-DETR's rank-six multi-scale deformable-attention tensor is folded to
  rank five. Both supported variants have one feature level, so this is
  numerically exact.
- The bilinear sampling and weighted accumulation are implemented by one
  dedicated Metal kernel.
- The kernel consumes the projection's native contiguous
  `[batch, spatial, head, channel]` layout. It does not pass a transposed view
  across the Core AI custom-kernel boundary.
- FP32 is the supported export. FP16 remains available only for experiments:
  the Transformer accumulated unacceptable numerical error in FP16 even
  though the custom Metal kernel itself remained accurate.

## Environment

Keep the Core AI tools isolated from the normal application environment:

```sh
uv pip install --python .venv-coreai/bin/python 'rfdetr==1.8.3'
```

The tested environment uses Python 3.12, PyTorch 2.11, torchvision 0.26,
Core AI Torch 0.4.1, and macOS 27 beta.

## Export

The RF-DETR loader downloads the official Seg Small weights when the default
weight file is absent:

```sh
.venv-coreai/bin/python scripts/apple/export_rfdetr_seg_coreai.py
```

This creates:

```text
model_weights/rf-detr-seg-small-384-fp32.aimodel
model_weights/rf-detr-seg-small-384-fp32.report.json
```

Use `--fp16` only to investigate half-precision behavior. It is not the
quality-approved path.

Export Jasna v6 after obtaining its checkpoint from the Jasna v0.9.0
distribution:

```sh
.venv-coreai/bin/python scripts/apple/export_rfdetr_seg_coreai.py \
  --weights /absolute/path/to/rfdetr-v6.pt \
  --output model_weights/rfdetr-v6-576-fp32.aimodel \
  --variant medium \
  --resolution 576 \
  --allow-overwrite
```

## Validation

Validate with a real image rather than random noise:

```sh
.venv-coreai/bin/python scripts/apple/validate_rfdetr_seg_coreai.py \
  model_weights/rf-detr-seg-small-384-fp32.aimodel \
  --image /absolute/path/to/image.jpg \
  --runs 20
```

The dedicated Metal kernel can be checked independently:

```sh
.venv-coreai/bin/python scripts/apple/smoke_test_rfdetr_coreai_kernel.py \
  --runner /absolute/path/to/lada-coreai-runner
```

## M5 Pro result

On the development M5 Pro, using one real 384×384 image:

| Path | Median latency |
|---|---:|
| PyTorch MPS FP32 | 43.78 ms |
| Core AI FP32 | 14.07 ms |

Core AI was 3.11× faster for the model-only forward pass. Its maximum
absolute differences from the PyTorch CPU FP32 reference were:

| Output | Maximum absolute error |
|---|---:|
| Boxes | 0.00000168 |
| Logits | 0.00012541 |
| Masks | 0.00285339 |

The custom deformable-attention Metal kernel alone had maximum absolute error
0.001953 in FP16 testing.

Jasna v6 Medium 576 was also validated with its production checkpoint:

| Path | Median latency |
|---|---:|
| PyTorch MPS FP32 | 86.09 ms |
| Core AI FP32 | 35.56 ms |

Core AI was 2.42× faster. Maximum absolute differences from PyTorch CPU FP32
were 0.0000154 for boxes, 0.0000148 for logits, and 0.000635 for masks.
A MIDA-024 frame produced one high-confidence detection through the complete
mioh preprocessing and postprocessing path.

## Fixed quality benchmark

On 100 held-out synthetic-mosaic frames from 50 validation clips, Jasna v6
reached macro mask IoU 0.964 versus 0.944 for v4-accurate. It won 71 of the
100 per-frame IoU comparisons and produced no false-positive detections on
the paired clean frames. Its complete single-frame path was slower:
39.8 ms versus 11.4 ms for v4-accurate.

Jasna v6 Large reached macro mask IoU 0.970 and won 70 of 100 comparisons
against regular v6. Its complete path took 85.2 ms, making it an
accuracy-first option for 4K offline processing rather than real-time use.

The 4K-oriented Large checkpoint can be exported with:

```bash
.venv-coreai/bin/python scripts/apple/export_rfdetr_seg_coreai.py \
  --weights /absolute/path/to/rfdetr-v6-large.pt \
  --output model_weights/rfdetr-v6-large-768-fp32.aimodel \
  --variant large \
  --resolution 768
```

The dedicated mioh build exposes it as `jasna-v6-large-coreai` and applies
Jasna's recommended confidence threshold of `0.40`.

## Memory-efficient preprocessing

The production RF-DETR path keeps source frames as unbatched `uint8` tensors
until each frame is consumed. It resizes first and converts only the four
source samples needed for each output pixel to `float32`. The inference queue
also carries only the frame count after inference instead of retaining a
second full-resolution batch.

On a 3840×2160 frame resized to the 768×768 Large input, a separate-process
microbenchmark measured:

| Preprocessing path | Maximum RSS | Peak memory footprint |
|---|---:|---:|
| Previous full-frame float path | 345.1 MB | 283.5 MB |
| Memory-efficient path | 292.6 MB | 213.8 MB |

This removes about 52.5 MB of maximum RSS and 69.7 MB of peak footprint from
preprocessing. Avoiding the stacked source batch additionally removes about
23.7 MiB per queued 4K RGB frame.

Quality parity was checked on the fixed 100-frame validation set with the same
Core AI assets:

| Model | Detection count mismatches | Minimum/mean mask IoU | Maximum box difference |
|---|---:|---:|---:|
| Jasna v6 576 | 0 | 1.000 / 1.000 | 2.24e-8 |
| Jasna v6 Large 768 | 0 | 1.000 / 1.000 | 8.94e-8 |

The reproducible comparison is:

```bash
.venv-coreai/bin/python scripts/apple/compare_rfdetr_preprocessing.py \
  model_weights/rfdetr-v6-576-fp32.aimodel \
  --benchmark-json /absolute/path/to/v4-accurate-vs-jasna-v6-benchmark.json \
  --dataset-root /absolute/path/to/validation/crop_unscaled_img \
  --resolution 576
```

The final model-space-to-source mask resize intentionally remains `float32`.
Changing that boundary operation to `float16` or nearest-neighbor reduces a
temporary allocation but can move mask edges, so it is not part of the
quality-approved optimization.

## Swift-native deployment

The dedicated mioh build can run both RF-DETR assets directly from the Swift
export pipeline. The Swift adapter preserves the production Python contract:

- direct square bilinear resize with `align_corners=False` coordinates;
- FP32 RGB ImageNet normalization;
- normalized `cx, cy, width, height` boxes;
- one selection per query using the maximum of the three logits;
- direct low-resolution logit-mask projection to the source frame; and
- confidence thresholds `0.35` for v6 and `0.40` for v6 Large.

RF-DETR remains dedicated-build and offline-export only. The Universal build
does not bundle either asset, and realtime preview continues to use the
measured `v4-fast` path so that RF-DETR cannot reduce playback throughput.
