# Core AI numerical regression with separately named temporal inputs

> Submission status: **hold**. The reduced July 23 canary described below is
> numerically clean for both contracts on the current runtime. It is useful as
> a beta canary, but it is not by itself a valid reproducer for Apple. Do not
> submit this report until the missing full-pipeline trigger is reduced into
> the script and `apple_feedback_reproduction_ready` becomes true.

## Summary

The original full variable-length BasicVSR++ experiment produced materially
incorrect output with a many-input unrolled contract. The production graph
retained parity after repeated values were packed into contiguous
`[K,C,H,W]` temporal tensors. A fixed T18 graph rebuilt with the current
toolchain remained bit-compatible, excluding a general inability to unroll
propagation. The current reduced graph does not reproduce the full failure, so
the exact interaction responsible has not yet been isolated tightly enough for
an Apple submission.

## Environment

- Hardware: MacBook Pro (Mac17,8), Apple M5 Pro, 48 GB
- OS: macOS 27.0, build 26A5388g
- `coreai-build`: 3600.75.3
- PyTorch: 2.12.0
- Model: LADA BasicVSR++ v1.2 propagation branch, FP16
- Custom operations: flow warp and deformable convolution Metal kernels

## Original end-to-end observations

The original full separate-input K6 experiment fell to approximately 23 dB
agreement.
The contiguous-tensor probes instead showed bounded FP16 accumulation:

| Graph | Result |
|---|---:|
| packed K2 | mean absolute error about 0.00074 |
| packed K6 | mean absolute error about 0.00141 |
| rebuilt fixed T18 control | bit-compatible |

The production workaround was then validated end to end:

| Clip | Variable contiguous K6 speed |
|---|---:|
| T18 | 111.0 fps |
| T90 | 115.9 fps |

T90 agreement remained in the high-60/low-70 dB range depending on the
comparison path, which is consistent with FP16 boundary accumulation rather
than the catastrophic separate-input failure.

## Reduced-canary result on July 23

The isolated `forward_2` propagation canary uses four separately named feature
components per frame plus separately named flows (29 inputs at K6). Both the
source asset and the M5 Pro `h17s` specialization matched the packed graph:

| Contract | K2 | K6 |
|---|---:|---:|
| packed, h17s | 74.66 dB | 69.68 dB |
| separate, h17s | 74.66 dB | 69.68 dB |

Therefore input count alone is not a sufficient reproducer. The missing
trigger is in the complete four-sweep pipeline, its boundary layout, or their
interaction. The reduced canary remains valuable because a future beta must
not make this clean control regress, but this report must not claim that it
currently reproduces the original failure.

## Reproduction

From the repository root:

```zsh
.venv-coreai/bin/python \
  scripts/apple/canary_basicvsrpp_coreai_temporal_io.py \
  --output-dir /tmp/mioh-coreai-temporal-io-canary
```

The script exports K2 and K6 versions of both contracts from identical weights
and inputs, compiles the selected specialization, runs both source and compiled
assets, and writes `report.json`. The packed contract is the correctness gate.
The report explicitly marks whether the reduced graph has reproduced a
separate-input regression and whether an Apple submission is ready.

## Expected behavior for the eventual full reproducer

Separating a fixed-length tensor into multiple named inputs must not change the
numeric result beyond normal FP16 conversion error. Packed and separate graphs
should have comparable agreement with their PyTorch reference.

## Production workaround

The end-to-end variable BasicVSR++ implementation packs the temporal axis into
one contiguous tensor and has passed the complete T18/T90 quality and speed
validation. Mioh therefore uses contiguous temporal I/O for every
frame-repeated Core AI graph even though the reduced control above is clean.

## Related state-I/O issue

The same toolchain also lowered an attempted Core AI mutable state buffer to
regular function I/O rather than exposing native state. This is a separate
issue, but the canary accepts `--state-asset` so each beta can record whether
state fields have reappeared:

```zsh
.venv-coreai/bin/python \
  scripts/apple/canary_basicvsrpp_coreai_temporal_io.py \
  --state-asset /path/to/stateful.aimodel
```

Until native state lowering is verified, production uses the already validated
contiguous boundary tensors.
