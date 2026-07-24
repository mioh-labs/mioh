# mioh model tools

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

`--minimal` downloads only the standard restoration model, v4 detection models,
and Real-ESRGAN x2/x4. Omit it to also download optional ROI enhancer models.
