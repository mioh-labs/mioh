# mioh dedicated build notes

This file records the local source-of-truth inputs required to reproduce the
dedicated `mioh.app`. It is intentionally separate from the public Universal
release runbook.

## Native model set

The Dedicated application is built without Python. It consumes the immutable,
M5 Pro-specialized native model set at:

```text
model_weights/mioh-dedicated-h17s/
```

That directory contains the compiled `.aimodelc` assets, variable-model
collections, checkpoint provenance, and cluster identity manifest required to
rebuild the app after deleting `build/`. It is intentionally ignored by Git
because it is approximately 2.8 GB, but it lives on the internal system drive
with the other `model_weights` assets.

The Dedicated build exposes the standard variable model as
`basicvsrpp-v1.2-coreai-variable`. Its source checkpoint is:

```text
model_weights/basicvsrpp-v1.2-detail-recovery-30000-ema.pth
```

Expected SHA-256:

```text
a8428a0ba7b056664914cd9f653232cfa7cd3d628f34e56399adb59bd943c065
```

The large-ROI Phase A/B model, native tile compositor, model alias, and UI
switch are deliberately not packaged. All ROI sizes use the established
single-ROI 256 px restoration path and the standard checkpoint above.

## Rebuilding after deleting `build/`

The `build/` directory contains generated artifacts and may be deleted. The
installed `/Applications/mioh.app`, source tree, and native model set remain
under the `lada` source directory. A Python environment and `.pth` checkpoint
conversion are not part of the Dedicated build.

The Dedicated build selects the local native model set automatically:

```zsh
cd /path/to/lada
packaging/macOS/standalone/build_app.sh
```

Universal builds keep their separate portable model-export path. That path does
not affect the normal M5 Pro Dedicated build.

After building, verify the packaged provenance file:

```zsh
cat build/macos-standalone/mioh.app/Contents/Resources/models/basicvsrpp-v1.2-variable-coreai.provenance.json
```

The recorded filename and SHA-256 must match the values above before the DMG
is distributed or copied to `/Applications`.
