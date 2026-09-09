# Jasna RF-DETR v6 Core ML port

`jasna-v6-coreml` and `jasna-v6-large-coreml` are the 576 px and 768 px Jasna
RF-DETR v6 detectors exported as fixed-shape FP32 Core ML ML Programs. They
are separate choices from the existing Core AI assets, which remain
available.

## Export

The exporter is `scripts/apple/export_rfdetr_seg_coreml.py`:

```console
python scripts/apple/export_rfdetr_seg_coreml.py \
  --weights model_weights/rfdetr-v6.pt \
  --output model_weights/rfdetr-v6-576-fp32.mlpackage

python scripts/apple/export_rfdetr_seg_coreml.py \
  --weights /path/to/rfdetr-v6-large.pt \
  --variant large --resolution 768 \
  --output model_weights/rfdetr-v6-large-768-fp32.mlpackage
```

RF-DETR normally forms a rank-six deformable-attention sampling tensor, while
Core ML supports tensors through rank five. The exporter uses the fact that
this model has one feature level: it removes the unit level dimension and
evaluates the same bilinear samples with `grid_sample`, which coremltools
lowers to native ML Program `resample` operations. The rewrite is temporary
and does not alter the PyTorch model on disk.

The 576 px runtime contract is:

- input: ImageNet-normalized RGB FP32 NCHW `[1, 3, 576, 576]`;
- boxes: FP32 `[1, 200, 4]`;
- logits: FP32 `[1, 200, 3]`;
- masks: FP32 `[1, 200, 144, 144]`.

The Large contract changes the input to `[1, 3, 768, 768]` and masks to
`[1, 200, 192, 192]`; the box, logit, query-count and postprocessing contracts
are unchanged.

## Validation

The fixed-graph rewrite was bit-exact against the source PyTorch model. The
generated ML Program differed from PyTorch by at most:

| output | maximum absolute error | mean absolute error |
| --- | ---: | ---: |
| boxes | 0.00001121 | 0.00000033 |
| logits | 0.00002193 | 0.00000240 |
| masks | 0.00053024 | 0.00003132 |

On a real 1080p validation frame with a positive mosaic detection, Core ML
and the existing Core AI asset selected the same RF-DETR query with the same
reported score. The selected mask's maximum absolute difference was
`0.0000515`.

On an M5 Pro, batch-one steady inference through Core ML `CPU_AND_GPU` had a
median of about 35 ms. `CPU_AND_NE` was about 174 ms, so mioh explicitly uses
the GPU path for this model. This is approximately the same raw inference
latency as the existing 576 px Core AI asset.

The 768 px Large model measured about 76.6 ms on Core ML `CPU_AND_GPU`, versus
about 79.9 ms for its existing Core AI asset. Its Core ML conversion had a
maximum absolute error of `0.0000081` for logits and `0.0002441` for masks;
boxes were identical.

## Packaging

Dedicated builds compile both RF-DETR `.mlpackage` assets to `.mlmodelc` and
expose both Core ML variants in the detection-model selector. Universal builds
do not compile, bundle, or expose these large optional detector assets. The
native Swift RF-DETR preprocessing and postprocessing implementation remains
shared by the build configurations.
