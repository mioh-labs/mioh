# MIOH Native-HF 512 prototype

## Purpose

Native-HF 512 is a native-pixel high-frequency refiner for mosaic ROIs.  It is
not a 4K-only model and it does not resize an entire video to 512 pixels.  A
fixed 512 x 512 crop is taken around an ROI without resampling, so the same
prototype can be evaluated on FHD, 4K, and 8K sources.

The existing BasicVSR++ restoration remains the frozen global solution.  The
new model learns only detail that is present in the native pixels and therefore
does not independently redraw low-frequency colour or shape in each tile.

## Fixed prototype contract

- Global guide: frozen BasicVSR++ v1.2, 9 frames at 256 x 256.
- Native refiner: centre 5 frames, one centre-frame output.
- Core ML input: one continuous tensor `[1, 40, 512, 512]`.
- Per-frame channels: mosaic RGB (3), mask (1), mask reliability (1), and
  upsampled global reconstruction RGB (3).
- Core ML outputs: restored RGB `[1, 3, 512, 512]` and confidence
  `[1, 1, 512, 512]`.
- Learned operations begin after `PixelUnshuffle(2)`.  `PixelShuffle(2)` is the
  only return to native resolution.
- The output is an identity at initialization: outside the ROI it is exactly
  the source, and inside the ROI it is exactly the frozen global restoration.
- The learned residual is high-pass limited and gated by mask, reliability,
  and confidence.
- A zero-initialized packed-pixel detail skip connects the aligned native
  samples directly to the residual output.  This preserves exact identity at
  initialization while avoiding the cold-start failure observed when every
  high-frequency sample had to cross the random deep fusion trunk.

Implementation:

- `lada/models/mioh_restorer/model_native_hf.py`
- `lada/models/mioh_restorer/native_hf_dataset.py`
- `lada/models/mioh_restorer/losses_native_hf.py`
- `scripts/training/train-mioh-native-hf.py`
- `scripts/apple/smoke-test-mioh-native-hf.py`

## 1. Build fixed 512 manifests

The fixed bucket flag changes crop size only.  It never scales or rewrites a
clean frame.

```zsh
cd /Users/okatti/Documents/lada

/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/training/build-mioh-restorer-v5-native-manifest.py \
  --metadata-root /Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative/train/crop_unscaled_meta \
  --output /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/manifests/train-native-hf-512.jsonl \
  --stride 16 \
  --context-fraction 0.30 \
  --maximum-bucket 512 \
  --fixed-bucket 512 \
  --tile-overlap 96

/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/training/build-mioh-restorer-v5-native-manifest.py \
  --metadata-root /Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative/validation/crop_unscaled_meta \
  --output /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/manifests/validation-native-hf-512.jsonl \
  --stride 16 \
  --context-fraction 0.30 \
  --maximum-bucket 512 \
  --fixed-bucket 512 \
  --tile-overlap 96
```

The current representative set is suitable for proving the architecture, but
most of it is 1080p.  A final 4K/8K quality ceiling must also be measured after
adding genuinely sharp native 4K/8K clean sources.  Upscaled FHD is not a
substitute because it contains no new high-frequency training signal.

## 2. Run the architecture smoke test before training

PyTorch trace and latency only:

```zsh
cd /Users/okatti/Documents/lada

/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/apple/smoke-test-mioh-native-hf.py \
  --skip-coreml \
  --runs 3 \
  --output-dir output/native-hf-512-smoke
```

Full FP16 Core ML conversion, numerical comparison, latency, and compute-plan
report:

```zsh
/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/apple/smoke-test-mioh-native-hf.py \
  --runs 20 \
  --allow-overwrite \
  --output-dir output/native-hf-512-smoke
```

The report is written to
`output/native-hf-512-smoke/native-hf-512-smoke-report.json`; the package is
`mioh-native-hf-512-fp16.mlpackage`.  This smoke model has deterministic tiny
head weights solely to keep the entire HF graph visible to the converter.

### Current untrained result (2026-08-02, M5 Pro)

- Parameters: 1,255,722.
- Core ML conversion and Core AI source conversion/h17s compilation: pass.
- Core ML FP16 agreement: RGB 77.74 dB; confidence 80.28 dB.
- Core AI agreement against PyTorch FP16: RGB 81.65 dB; confidence identical.
- Warm Core AI latency: median 138.1 ms per 512 ROI.
- Core ML compute plan: 3,257 graph operations; all 1,236 reported non-constant
  operations prefer `MLGPUComputeDevice`; zero prefer the ANE.

Thus conversion and numerical fidelity pass, but the current architecture does
**not** pass the ANE-placement gate.  Fixed/Folded shift-bank convolutions are
GPU-preferred, while hard argmax/scatter is CPU-preferred in isolation; the
complete graph remains on the GPU to avoid device transfers.  This is a
quality-ceiling prototype, not a shipping-speed result.  Do not interpret a
first cold prediction as steady-state latency because it includes model
preparation.

## 3. Train in independent stages

The first two 500-step direct HF pilots are diagnostic failures and must not be
resumed.  The second source-balanced run changed ROI PSNR by only
`-0.000019 dB` and made HF error `0.0184%` worse (EMA: `0.0591%` worse).
Independent 64-clip evaluation also found essentially zero residual/target
correlation.  The cause was structural: soft alignment remained near its
initial high-entropy distribution, while the native samples reached the
zero-initialized output only through a deep random fusion path.

The learned Stage 0 was removed.  Exact-motion experiments showed that the
fixed native descriptor already reached the required mean/p95 range, whereas
optimizing the random feature projection made held-out alignment worse.  Known
quarter-pixel motion remains a mandatory regression test, but it is no longer
a checkpoint-producing training stage.

The production curriculum therefore contains two independent runs:
`hf-bootstrap -> joint`.  `hf-bootstrap` always starts from the versioned,
reproducible `frozen-analytic-alignment-v2` initialization.  The model seed is
fixed at `20260802`; `--seed` controls only data order and degradation.  The
encoder and alignment stay frozen while the fusion, residual and direct
packed-detail paths learn from GT.  `--resume` continues the same run, while
`joint --initialize-from` loads only the completed hf-bootstrap EMA and starts
a fresh optimizer.

The block-size and degradation curriculum is also fixed.  The representative manifests are
overwhelmingly large-block (the current train manifest is about 95% >=17px),
and a direct 50-step run on that distribution made HF error 3.17% worse.  A
controlled same-content test showed that an 8px mosaic learned 6.4 times more
HF improvement than a 23px mosaic at 300 updates.  Stage 1 therefore samples
block sizes uniformly from 6 through 12px, where native evidence remains
recoverable, and uses clean mosaic observations only.  Compression, blur, and
noise are deliberately deferred: a 500-step mixed-degradation bootstrap
improved HF error by only 0.025%, so robustness was pulling the EMA toward the
safe zero-residual solution before the inverse mapping existed.  Stage 2
returns to the manifest-derived deployment distribution, restores the
clean/mild/full degradation mix, and trains confidence to retain the frozen
BasicVSR++ result where native evidence is insufficient.  Validation JSON
reports aggregate metrics plus
separate `block_small_le8`, `block_medium_9_16`, and `block_large_ge17`
buckets.  Stage-1 metrics are a learnability gate, not the final deployment
quality claim.

The original HF gate was retired after a controlled baseline exposed a false
success mode.  Subtracting `0.8-1.0 * HF(BasicVSR++ base)` without any learned
model improved the old HF error by about 3%, because the GAN/perceptual guide
contains excess edge energy.  A 300-step block-8 overfit likewise obtained 77%
of its apparent gain from guide smoothing on one representative sample.  The
v7 bootstrap therefore makes projected GT-detail innovation the promotion
criterion and explicitly penalizes the two-scale filter span of both the
frozen guide and centre mosaic observation.  Reconstruction, HF, gradient,
and wavelet losses remain only as auxiliary fidelity terms.  In a controlled
block-8 overfit this recovered +1.17 dB and 6.23% legacy-HF improvement while
retaining 65.93% Innovation EV and reducing nuisance-span energy to 26.37%.
The innovation-only ablation reached similar Innovation EV but only +0.72 dB;
the old objective reached +2.25 dB partly by raising nuisance-span energy to
32.60%.  Residual-target and remosaic losses stay disabled in Stage 1.  The
packed-pixel direct skip can read aligned source RGB only; guide-aware cleanup
remains available through the deeper fusion path in Stage 2.

### Stage 1: high-frequency bootstrap

```zsh
TMPDIR=/Volumes/Project_HD/lada_finetune_aozora_hikari/tmp \
PYTHONPATH=/Users/okatti/Documents/lada \
caffeinate -dimsu \
/Users/okatti/.pyenv/versions/lada/bin/python -u \
  scripts/training/train-mioh-native-hf.py \
  --train-manifest /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/manifests/train-native-hf-512.jsonl \
  --validation-manifest /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/manifests/validation-native-hf-512.jsonl \
  --basicvsrpp-checkpoint /Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_finetune_v12_full/checkpoints/run-gan-perceptual-torch213-fresh/iter_9000.pth \
  --work-dir /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/run-hf-bootstrap-v7-hybrid \
  --stage hf-bootstrap \
  --steps 500 \
  --warmup-steps 50 \
  --save-every 100 \
  --validate-every 100 \
  --validation-batches 12 \
  --ema-decay 0.99 \
  --workers 0 \
  --device mps
```

On the M5 Pro, the earlier direct-HF pilot measured about 2.05 seconds per
update, so 500 updates took about 17 minutes.  The v5 path must be timed again
from its first 10 updates because the corrected alignment graph is larger.

Convert the EMA weights from a pilot checkpoint:

```zsh
/Users/okatti/.pyenv/versions/lada/bin/python \
  scripts/apple/smoke-test-mioh-native-hf.py \
  --checkpoint /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/run-hf-bootstrap-v7-hybrid/mioh-native-hf-512-step-000500.pth \
  --runs 20 \
  --allow-overwrite \
  --output-dir output/native-hf-512-step-500
```

### Stage 2: deployment-distribution joint calibration

Run Stage 2 only after the Stage-1 EMA passes its HF gate.  Alignment and its
encoder features remain frozen because updating either degraded held-out
motion.  Fusion, residual, confidence, and frame-reliability gating are
calibrated on the real manifest block-size distribution with a fresh optimizer.

```zsh
TMPDIR=/Volumes/Project_HD/lada_finetune_aozora_hikari/tmp \
PYTHONPATH=/Users/okatti/Documents/lada \
caffeinate -dimsu \
/Users/okatti/.pyenv/versions/lada/bin/python -u \
  scripts/training/train-mioh-native-hf.py \
  --train-manifest /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/manifests/train-native-hf-512.jsonl \
  --validation-manifest /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/manifests/validation-native-hf-512.jsonl \
  --basicvsrpp-checkpoint /Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_finetune_v12_full/checkpoints/run-gan-perceptual-torch213-fresh/iter_9000.pth \
  --work-dir /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/run-joint-v7 \
  --stage joint \
  --initialize-from /Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_native_hf/run-hf-bootstrap-v7-hybrid/mioh-native-hf-512-step-000500.pth \
  --steps 500 \
  --warmup-steps 50 \
  --save-every 100 \
  --validate-every 100 \
  --validation-batches 24 \
  --ema-decay 0.99 \
  --workers 0 \
  --device mps
```

## Prototype promotion gate

Do not integrate this into the application merely because training loss falls.
Promote it only when the same fixed validation protocol shows all of the
following:

- At the 500-step continuation gate, small-block Innovation EV is above 2%,
  innovation correlation is above 0.10, nuisance-span energy is below 25%,
  and non-support correction energy is below 5%.
- At Stage-1 completion, Innovation EV is at least 10%, innovation correlation
  is at least 0.25, and at least 64 source-balanced validation samples have a
  95% confidence-interval lower bound above zero.
- Correct five-frame context beats a centre-repeat control by at least three
  Innovation-EV points.  Smoothing and unsharp controls must remain at zero or
  below under the same metric.
- The legacy HF metric remains diagnostic only; it cannot promote a model by
  itself because simple guide smoothing can improve it.
- ROI PSNR is no worse than 0.2 dB below the global reconstruction.
- Temporal error does not regress in any motion bucket.
- Pixels outside the mask remain bit-exact before export clamp.
- The FP16 package remains numerically close to PyTorch, and compute-plan plus
  latency are acceptable on the target Mac.

If the 500-step pilot produces no high-frequency signal in low-motion scenes,
stop and diagnose the dataset/phase path before extending training.  A longer
run cannot manufacture detail that is absent from the native clean sources.
