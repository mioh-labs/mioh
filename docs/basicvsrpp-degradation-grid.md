# BasicVSR++ degradation grid benchmark

`scripts/training/evaluate-basicvsrpp-degradation-grid.py` checks the installed
Dedicated model's source checkpoint against controlled synthetic degradations.
It **does not train** or replace the app model.

The default grid uses a single clean 256px source crop and scores the same
central frame under every condition:

- Mosaic blocks: 8, 16, 32 pixels **at the 256px model input**.
- Distinct observed frames: 1, 9, 26, 48. The Python model needs optical-flow
  input with at least two frames, so the 1-frame condition repeats that frame
  once. It does not add independent information.
- No *additional* compression and H.264 re-encoding at CRF 18 and 28. The source
  video may itself already be compressed; that is reported as the source, not
  treated as a lossless ground truth. This first benchmark re-encodes the
  256px model input; it does not exactly reproduce the original full-frame
  encoder followed by ROI cropping and scaling.
- The grid phase is held fixed in image coordinates for the full clip.
- A central 128×128 ROI is mosaicked by default. Outside it, source pixels
  remain intact, as in the masked training observations.

Example on a **verified clean** video (not one that already contains a mosaic):

```zsh
cd /Users/okatti/Documents/lada
LADA_DEFORM_CONV_BACKEND=mps_deform_conv \
  .venv_torch213/bin/python \
  scripts/training/evaluate-basicvsrpp-degradation-grid.py \
  --clean-video /absolute/path/to/clean-source.mp4 \
  --start-frame 1800 \
  --crop 100,100,512,512 \
  --roi 64,64,128,128 \
  --output-dir /absolute/path/to/new-results-directory \
  --trust-checkpoint
```

The output directory must not already exist. The script writes `metrics.json`
and `metrics.csv`, including checkpoint SHA-256, baseline observation scores,
restored scores, and differences. The key measures are `roi_psnr_db` and
`hf_corr_times_amp`; the latter is correlation multiplied by RMS amplitude
ratio of a Gaussian high-pass (same definition as the existing multi-frame
oracle experiment). Larger is better for both, but an increase in high-frequency
amplitude alone is not evidence of correct detail.

For a quick check, use `--blocks 8 --frames 1,9 --crfs none`. To investigate a
production ROI, compute the **effective input-space** block dimensions after
the app's ROI crop and 256px scaling. Do not compare native 4K block widths
directly with this grid. Rectangular cells, changing grid size, detector masks,
interlace, and temporal flicker are not covered by this first benchmark.
It measures one center frame, not full-video quality. It must not be used to
claim recovery from real mosaic footage without a verified clean reference.
