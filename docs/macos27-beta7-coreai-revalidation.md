# macOS 27 Developer Beta 7 / Xcode 27 Beta 6 revalidation

Validated on 2026-08-25 with:

- macOS 27.0 build `26A5421a`;
- Xcode 27.0 build `27A5252f`;
- `coreai-build 3600.83.1`;
- Python 3.12, PyTorch 2.11.0 and Core AI Torch 0.4.2.

## Results

- All 14 Core AI assets packaged by the dedicated mioh build completed their
  runtime smoke inference. The dedicated verifier also accepted all 13 Core ML
  assets.
- The RF-DETR deformable-attention Metal kernel matched the established FP16
  result: maximum absolute error `0.001953125`, mean absolute error
  `0.000105721`, and median latency `1.192 ms`.
- The MiniMax H3 Qwen prompt profile remained `4152` tokens with a `16` token
  free-prompt budget. A production BF16 four-layer DiT asset accepted a dynamic
  `[64, 5376]` input and returned the same shape.
- FlashVSR and AdcSR each completed a six-frame end-to-end decode, inference,
  encode, and decode smoke test. Both returned a valid 256x256 H.264 video with
  six frames.
- Native Core AI state is now functional when the exported program is optimized.
  A single isolated production-shape propagation block was `0.6%` slower, but
  an end-to-end A/B on the actual 300-frame MIDV-670 mosaic clip reversed that
  result: median restoration time fell from `5.223 s` to `4.512 s` (`13.6%`),
  and median wall time fell from `7.330 s` to `6.646 s` (`9.3%`). The two modes
  were individually deterministic, but were not bit-equivalent to each other
  (`52.02 dB` decoded-video PSNR; active-ROI MAE `0.584/255`). After visual
  acceptance, the four continuation assets in the production variable model
  were replaced with the native-state variants. Start, flow, spatial, and
  reconstruction assets remain unchanged.
- Packed temporal inputs, variable first dimensions, `grid_sample`, `flow_warp`,
  DCNv2, and the current application model set all remain compatible.
- A newly reconstructed full T18 BasicVSR++ A/B used 18 decoded MIDV-670
  frames and included all four propagation sweeps plus two continuation
  boundaries per sweep. The 29-input `forward_2` separate contract and the
  two-input packed contract were bit-identical; both matched PyTorch at
  `81.3772 dB`. The old approximately 23 dB temporal-input regression is not
  reproducible on this toolchain.
- Two separate Core AI compiler defects remain reproducible. A fixed BF16
  source asset runs correctly but `coreai-build` aborts after substituting an
  FP16 input for the BF16 contract. Separately, same-structure assets with
  different weights compile to byte-identical h17s programs and execute the
  first model's weights. Fixed-BF16 avoidance and structural graph identity
  salts must therefore remain.
- Both mioh and mioh upscaler built and passed strict ad-hoc signature
  verification. Mioh Remote also built for generic iOS with the four variable
  continuation assets using native Core AI state in both source and M5 h17g
  specialization form.

## Workarounds that remain necessary

The Xcode Beta 6 `CoreAIDelegates` Swift interface exposes
`SpecializationOptions.allowedComputeUnitKinds` as a getter, but still omits a
public initializer that accepts the allowed set. The existing constrained
CPU/GPU initializer shim therefore remains necessary to exclude ANE rather than
merely prefer GPU.

`AIModelCache.Policy` exposes `default`, `persistent`, and purge conditions, but
does not expose a no-cache or ephemeral policy. The application should continue
using the default non-persistent policy and explicit cache cleanup rather than
claiming that specialization caching can be disabled.

The macOS media paths now use the macOS 27 AVFoundation interfaces throughout:
`AVAssetReaderOutput.Provider`, throwing `AVAssetReader.start()`,
`AVAssetWriter` pixel/sample receivers, typed `CVPixelBufferAttributes`,
`pixelBufferAndDisplayTime(forItemTime:)`, and the scoped `AVPlayerItem`
notification names. The passthrough export path uses async
`AVAssetExportSession.export(to:as:)`. The dedicated build reports no legacy
AVFoundation deprecation warnings. On the actual 10.01-second MIDV-670 sample,
the migrated build decoded, detected, and encoded all 300 frames. Its decoded
video is pixel-identical to the pre-migration native-state build (infinite
PSNR), so this is a compatibility and lifetime-safety migration rather than a
quality change.

The first MIDV-670 benchmark attempts exposed a test-harness bug: closing the
preview process's standard input is a documented stop signal, so the harness
was cancelling `AVAssetReader` while `next()` was in flight. Keeping the control
pipe open fixed the crash; it was not a native-state or production mioh failure.

## Rebuilt artifacts

- `/Users/okatti/Documents/lada/build/macos-standalone/mioh.app`
- `/Users/okatti/Documents/lada/build/macos-standalone/mioh-0.14.3-unsigned.dmg`
- `/Users/okatti/Documents/lada/build/mioh-upscaler/mioh upscaler.app`
- `/Users/okatti/Documents/lada/build/mioh-upscaler/mioh-upscaler-0.14.3-unsigned.dmg`

The dedicated mioh build and installed `/Applications/mioh.app` were replaced
after native-state adoption and the AVFoundation 27 migration. mioh upscaler
was rebuilt separately with its current AVFoundation 27 media paths. Mioh
Remote's verified unsigned development product is
`/private/tmp/mioh-remote-native-state-derived/Build/Products/Debug-iphoneos/MiohRemote.app`.
