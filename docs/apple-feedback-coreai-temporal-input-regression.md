# Core AI numerical regression with separately named temporal inputs

> Submission status: **do not submit on the current toolchain**. The original
> regression was real, but a reconstructed full-pipeline T18 canary is now
> numerically clean on macOS 27 Developer Beta 7, Xcode 27 Beta 6, and Core AI
> Torch 0.4.2. There is no current reproducer for Apple.

## Summary

The original full variable-length BasicVSR++ experiment produced materially
incorrect output with a many-input unrolled contract. The production graph
retained parity after repeated values were packed into contiguous
`[K,C,H,W]` temporal tensors. A fixed T18 graph rebuilt with the current
toolchain remained bit-compatible, excluding a general inability to unroll
propagation. The exact historical trigger was never reduced. The current full
T18 reconstruction also does not reproduce the failure, so there is no active
Apple-submission case on the tested Beta 7 toolchain.

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

Therefore input count alone was not a sufficient reproducer. At that time the
remaining suspect was the complete four-sweep pipeline, its boundary layout,
or their interaction. The later full-pipeline result below is also clean. Both
canaries remain useful future-beta controls, but neither currently reproduces
the original failure.

## Full-pipeline revalidation on August 25

The old many-input contract was reconstructed around the current production
weights. The full canary executes the spatial encoder, bidirectional flow, all
four BasicVSR++ propagation sweeps, two continuation chunks per sweep, and the
reconstruction head. Its T18 input consists of decoded frames from the actual
MIDV-670 ten-second source clip. The only A/B difference is propagation I/O:

- packed: `contexts` and `flows` contiguous temporal tensors;
- separate: every frame/component pair and every flow is a separately named
  input. The `forward_2_start6` specialization has 29 inputs rather than two.

Both arms were freshly exported and specialized with the current toolchain.
They produced bit-identical FP16 output:

| Comparison | Result |
|---|---:|
| packed vs separate maximum error | `0.0` |
| packed vs separate mean error | `0.0` |
| packed vs separate PSNR | infinite |
| packed vs PyTorch reference | `81.3772 dB` |
| separate vs PyTorch reference | `81.3772 dB` |

This is stronger than the earlier reduced single-branch result and includes
the complete four-sweep topology and recurrent boundary layout. The original
approximately 23 dB regression is not present on the current runtime.

Reproduction:

```zsh
.venv-coreai/bin/python \
  scripts/apple/canary_basicvsrpp_full_temporal_io.py
```

The report is
`/tmp/mioh-basicvsrpp-full-temporal-io/report.json`. Production may retain the
packed contract because it has fewer inputs and is already deployed, but it is
no longer justified as a correctness workaround on this tested toolchain.

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
validation. Mioh continues using contiguous temporal I/O because it has fewer
inputs and is already deployed, not because Beta 7 still requires it for
correctness.

## Related state-I/O issue

Core AI Torch 0.4.2 now lowers an optimized mutable buffer to native Core AI
state. On macOS 27 Developer Beta 7 with Xcode 27 Beta 6, the repository's
state canary passed source execution, h17s ahead-of-time compilation, and
compiled Swift execution. Two calls produced 2 then 3 while the persistent
state ended at 2:

```zsh
.venv-coreai/bin/python scripts/apple/canary_coreai_native_state.py
```

The report is `/tmp/mioh-coreai-native-state-canary/report.json`. A subsequent
full BasicVSR++ A/B on the actual 300-frame MIDV-670 mosaic clip measured a
13.6% restoration-time reduction and passed visual acceptance. Production
therefore adopted native state for the three recurrent boundary values in all
four continuation assets on 2026-08-25. Context and flow sequences remain
packed contiguous tensors, so this does not remove the separate-temporal-input
workaround documented above.
