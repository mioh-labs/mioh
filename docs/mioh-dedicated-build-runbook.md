# mioh dedicated build notes

This file records the local source-of-truth inputs required to reproduce the
dedicated `mioh.app`. It is intentionally separate from the public Universal
release runbook.

## Active restoration checkpoint

The dedicated application currently ships
`basicvsrpp-v1.2-coreai-variable` generated from:

```text
model_weights/hf2500-plus-fc2-500-ema.pth
```

Expected SHA-256:

```text
8706f83c4ea0dc1abd83544b21eaeba3bf7df4ea2ca1b327cd1d5fc8146a396f
```

This is the `HF2500 + fc2_best 500-step EMA` checkpoint. It is not the old
iter9000 model and not the generic v1.2 checkpoint.

## Rebuilding after deleting `build/`

The `build/` directory contains generated artifacts and may be deleted. The
installed `/Applications/mioh.app`, source tree, virtual environment, model
weights, and the checkpoint above remain under the `lada` source directory.

The dedicated build now selects the local active checkpoint automatically:

```zsh
cd /path/to/lada
packaging/macOS/standalone/build_app.sh
```

Portable/Universal builds continue to select
`model_weights/lada_mosaic_restoration_model_generic_v1.2.pth`. An explicit
`VARIABLE_COREAI_CHECKPOINT` still overrides either default when needed.

After building, verify the packaged provenance file:

```zsh
cat build/macos-standalone/mioh.app/Contents/Resources/models/basicvsrpp-v1.2-variable-coreai.provenance.json
```

The recorded filename and SHA-256 must match the values above before the DMG
is distributed or copied to `/Applications`.
