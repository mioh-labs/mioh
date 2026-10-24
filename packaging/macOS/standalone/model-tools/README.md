# mioh model tools

The maintainer build and release procedure is documented in
[`docs/mioh-universal-build-release-runbook.md`](../../../../docs/mioh-universal-build-release-runbook.md).
It includes the required bundled-runtime conversion checks and the clean-Mac
acceptance test.

The universal DMG does not bundle model weights. Install the app first, then run
these scripts from the mounted DMG or from this folder.

Recommended first run:

```zsh
./download-mioh-models.zsh --app /Applications/mioh-universal.app
./convert-mioh-models.zsh --app /Applications/mioh-universal.app
```

The downloader writes source weights into:

```text
/Applications/mioh-universal.app/Contents/Resources/models
```

The converter then creates the Core ML and Core AI model assets used by the app.
Core AI conversion requires macOS 27 and the packaged Python environment with
`coreai-torch`. Core ML conversion requires the packaged `coremltools` and
`ultralytics` dependencies.

Useful options:

```zsh
./download-mioh-models.zsh --app /Applications/mioh-universal.app --minimal
./convert-mioh-models.zsh --app /Applications/mioh-universal.app --coreml-only
./convert-mioh-models.zsh --app /Applications/mioh-universal.app --coreai-only
```

`--minimal` downloads the standard restoration model, v2/v3.1/v4 detection
models, and Real-ESRGAN x2/x4. Omit it to also download optional ROI enhancer
models. The VR detection checkpoint is not distributed or downloaded by mioh;
the converter only accepts an existing local copy if the user provides one.

## ROI Refine pack (macOS 27+)

ROI Refine (Sulphur 2 / LTX-2.3 with the mioh ROI IC-LoRA) uses an external pack. It is
not stored in the app, so it survives app updates. The Hugging Face repository
`okatti/mioh-roi-refine` is gated: accept the LTX-2 Community License there first, then
run `hf auth login` or set `HF_TOKEN`.

```zsh
./install-roi-refine.zsh --app /Applications/mioh-universal.app                 # download the converted pack (~18 GiB)
./install-roi-refine.zsh --app /Applications/mioh-universal.app --mode build    # convert locally from the official checkpoint
./install-roi-refine.zsh --app /Applications/mioh-universal.app --verify-only
```

The pack is written to `~/Library/Application Support/mioh/ROIRefine/pack-sulphur-iclora1750-v5`
unless `--pack` is given. Build mode downloads the ~43 GiB Sulphur 2 distilled
checkpoint and a pinned LTX-2 source checkout. It converts the 12 Core AI DiT groups
with the packaged `coreai-torch`, and exports the VAE and the window constants. Plan for
about 70 GiB free. Select the printed folder in 復元 → ROIエンハンサー → ROI Refine.
