# mioh Universal build and release runbook

This is the canonical procedure for building and publishing the model-free
`mioh-universal` distribution. Follow the checks in order. Do not rebuild a
release from memory or copy an older script from `lada_git` over the working
tree.

## Distribution contract

- The public application is the Universal build only.
- The DMG contains no model weights and no RF-DETR implementation.
- The DMG contains the Japanese manual, the bundled Python 3.12 runtime, model
  conversion tools, and direct links to the two `.zsh` entry points.
- Source weights are downloaded after installation into
  `/Applications/mioh-universal.app/Contents/Resources/models`.
- macOS 26 uses Core ML assets. macOS 27 can additionally create and use Core
  AI assets.
- Replacing the application also replaces its `Contents/Resources/models`
  directory. Run the download and conversion scripts again after installing a
  new application build.

## Repositories and source of truth

Development repository:

```text
/Users/okatti/Documents/lada
```

Public repository:

```text
/Users/okatti/Documents/lada_git
```

After reconciling history, `lada` is the source of truth. Before copying a
file to `lada_git`, inspect both histories and both versions. This prevents an
older development copy from overwriting a public-only fix.

```zsh
cd /Users/okatti/Documents/lada
git status -sb
git log --oneline -- packaging/macOS/standalone/model-tools

git -C /Users/okatti/Documents/lada_git status -sb
git -C /Users/okatti/Documents/lada_git log --oneline -- \
  packaging/macOS/standalone/model-tools

diff -u \
  packaging/macOS/standalone/model-tools/download-mioh-models.zsh \
  /Users/okatti/Documents/lada_git/packaging/macOS/standalone/model-tools/download-mioh-models.zsh
diff -u \
  packaging/macOS/standalone/model-tools/convert-mioh-models.zsh \
  /Users/okatti/Documents/lada_git/packaging/macOS/standalone/model-tools/convert-mioh-models.zsh
```

If `lada_git` contains a valid fix that is absent from `lada`, port and test
that fix in `lada` first. Only then synchronize the public copy. Never use
repository-wide `git add -A` or a repository-wide `rsync --delete` because the
development checkout contains private experiments, weights, evaluations, and
RF-DETR files that must not enter the public distribution.

## Critical model-tool invariants

Keep these rules when editing the downloader or converter:

1. The downloader writes to `.part`, verifies a known SHA-256, and only then
   renames the file. It never resumes a partial file from an older URL.
2. A failed model does not hide the remaining downloads. The script reports a
   complete failure list and exits non-zero at the end.
3. The VR detector is a `mioh-labs/mioh` GitHub Release asset, not a
   `ladaapp/lada` Hugging Face asset. Update `MIOH_RELEASE_TAG`, its URL, and
   its SHA-256 together when publishing a new release.
4. Real-ESRGAN conversion must not require `basicsr` in the bundled runtime.
   It uses the vendored RRDBNet implementation.
5. SwinIR conversion must not require `timm` in the bundled runtime. The
   exporter supplies the three helpers used by the pinned official SwinIR
   source.
6. The variable BasicVSR++ source collection must contain exactly 11
   `.aimodel` assets. The application accepts either this portable source
   collection or a matching architecture-specific `.aimodelc` collection.
7. Core ML detectors and enhancers may be portable `.mlpackage` assets; Core
   AI assets may be portable `.aimodel` assets. Compiling them ahead of time is
   an optimization, not a requirement for detection by the application.

The messages below are warnings and do not mean conversion has stopped:

```text
Torch version ... has not been tested with coremltools
Redirects are currently not supported in Windows or MacOs
```

A Python traceback, `ModuleNotFoundError`, or a non-zero shell exit is a real
failure.

## Pre-build checks

Regenerate the manual when its Markdown source changes:

```zsh
cd /Users/okatti/Documents/lada
uv run --with reportlab scripts/docs/build_mioh_manual_pdf.py
test -f output/pdf/mioh-user-manual-ja.pdf
```

Validate the two public entry points and run the test suite:

```zsh
zsh -n packaging/macOS/standalone/model-tools/download-mioh-models.zsh
zsh -n packaging/macOS/standalone/model-tools/convert-mioh-models.zsh
/Users/okatti/.pyenv/versions/lada/bin/python -m pytest tests -q
```

For a model-tool change, test with the Python interpreter that will actually
be bundled, not only the developer virtual environment:

```zsh
PY=build/macos-standalone-universal/mioh-universal.app/Contents/Resources/runtime/bin/python3.12
"$PY" -c 'import importlib.util; print(importlib.util.find_spec("basicsr"))'
"$PY" -c 'import importlib.util; print(importlib.util.find_spec("timm"))'
```

Both may print `None`; the Real-ESRGAN and SwinIR exporters are deliberately
self-contained. A dependency fix is not complete until an actual checkpoint
has produced its final `.mlpackage` with this bundled interpreter.

## Build

From the development repository:

```zsh
cd /Users/okatti/Documents/lada
zsh packaging/macOS/standalone/build_universal_app.sh
```

Expected outputs:

```text
build/macos-standalone-universal/mioh-universal.app
build/macos-standalone-universal/mioh-universal-0.14.3-unsigned.dmg
```

The build is model-free by default. It ad-hoc signs the application, includes
the packaged runtime and model tools, and creates these DMG-root links:

```text
download-mioh-models.zsh -> model-tools/download-mioh-models.zsh
convert-mioh-models.zsh  -> model-tools/convert-mioh-models.zsh
Applications             -> /Applications
```

For a release suffix such as `-002`, keep the application version at `0.14.3`
and copy the verified DMG to the release asset name after the build:

```zsh
cp -p \
  build/macos-standalone-universal/mioh-universal-0.14.3-unsigned.dmg \
  build/macos-standalone-universal/mioh-universal-0.14.3-002-unsigned.dmg
```

## Verify the built application and DMG

The embedded scripts must be byte-for-byte identical to the canonical source:

```zsh
APP=build/macos-standalone-universal/mioh-universal.app
cmp -s \
  scripts/apple/export_realesrgan_coreml.py \
  "$APP/Contents/Resources/model-tools/scripts/apple/export_realesrgan_coreml.py"
cmp -s \
  scripts/apple/export_swinir_coreml.py \
  "$APP/Contents/Resources/model-tools/scripts/apple/export_swinir_coreml.py"
cmp -s \
  packaging/macOS/standalone/model-tools/download-mioh-models.zsh \
  "$APP/Contents/Resources/model-tools/download-mioh-models.zsh"
cmp -s \
  packaging/macOS/standalone/model-tools/convert-mioh-models.zsh \
  "$APP/Contents/Resources/model-tools/convert-mioh-models.zsh"
```

Verify the disk image before publishing:

```zsh
DMG=build/macos-standalone-universal/mioh-universal-0.14.3-unsigned.dmg
hdiutil verify "$DMG"
shasum -a 256 "$DMG"
```

Mount it read-only and verify the top-level links and the scripts inside the
DMG, not only the app in the build folder:

```zsh
diskutil image attach --readOnly --mountOptions nobrowse "$DMG"
ls -l "/Volumes/mioh-universal/download-mioh-models.zsh"
ls -l "/Volumes/mioh-universal/convert-mioh-models.zsh"
zsh -n "/Volumes/mioh-universal/download-mioh-models.zsh"
zsh -n "/Volumes/mioh-universal/convert-mioh-models.zsh"
```

Detach the disk identifier reported by `diskutil image attach` after the
checks.

## Synchronize the public repository

Copy only reviewed source files. Use explicit paths and inspect the result:

```zsh
ROOT=/Users/okatti/Documents/lada
PUBLIC=/Users/okatti/Documents/lada_git

rsync -a \
  "$ROOT/packaging/macOS/standalone/model-tools/download-mioh-models.zsh" \
  "$PUBLIC/packaging/macOS/standalone/model-tools/download-mioh-models.zsh"
rsync -a \
  "$ROOT/packaging/macOS/standalone/model-tools/convert-mioh-models.zsh" \
  "$PUBLIC/packaging/macOS/standalone/model-tools/convert-mioh-models.zsh"
rsync -a \
  "$ROOT/scripts/apple/export_realesrgan_coreml.py" \
  "$PUBLIC/scripts/apple/export_realesrgan_coreml.py"
rsync -a \
  "$ROOT/scripts/apple/export_swinir_coreml.py" \
  "$PUBLIC/scripts/apple/export_swinir_coreml.py"

git -C "$PUBLIC" status -sb
git -C "$PUBLIC" diff --check
```

Add other reviewed Universal source files explicitly when they changed. Tests,
RF-DETR prototypes, private weights, generated evaluation output, and temporary
directories are not copied to the public repository.

Synchronize the built application only within its ignored artifact directory:

```zsh
rsync -a --delete \
  "$ROOT/build/macos-standalone-universal/mioh-universal.app/" \
  "$PUBLIC/build/macos-standalone-universal/mioh-universal.app/"
rsync -a \
  "$ROOT/build/macos-standalone-universal/mioh-universal-0.14.3-unsigned.dmg" \
  "$PUBLIC/build/macos-standalone-universal/mioh-universal-0.14.3-002-unsigned.dmg"
```

Compare SHA-256 values after every artifact copy.

## Commit, push, and publish

Stage only the reviewed source paths:

```zsh
cd /Users/okatti/Documents/lada_git
git fetch origin main
git add -- <explicit-reviewed-paths>
git diff --cached --check
git commit -m "<concise description>"
git push origin main
```

The VR checkpoint referenced by the downloader must exist in the target
release before publishing the DMG. Replace the verified DMG asset and then
download it again for an end-to-end checksum check:

```zsh
gh release upload v0.14.3-002 \
  build/macos-standalone-universal/mioh-universal-0.14.3-002-unsigned.dmg \
  --clobber

VERIFY_DIR=$(mktemp -d /private/tmp/mioh-release-verify.XXXXXX)
gh release download v0.14.3-002 \
  --pattern 'mioh-universal-0.14.3-002-unsigned.dmg' \
  --dir "$VERIFY_DIR"
shasum -a 256 \
  build/macos-standalone-universal/mioh-universal-0.14.3-002-unsigned.dmg \
  "$VERIFY_DIR/mioh-universal-0.14.3-002-unsigned.dmg"
```

The two hashes must match. Merely seeing a successful upload message is not a
complete release verification.

## Clean-machine acceptance test

On another supported Mac:

1. Download the DMG from GitHub Release rather than copying the local app.
2. Replace `/Applications/mioh-universal.app`.
3. Run `download-mioh-models.zsh` from the mounted DMG.
4. Run `convert-mioh-models.zsh` from the mounted DMG.
5. On macOS 26, confirm Core ML models are created and the app does not claim
   Swift native/Core AI restoration is available.
6. On macOS 27, confirm the variable BasicVSR++ source collection contains 11
   assets and is selected successfully.
7. Confirm Real-ESRGAN and SwinIR conversion completes without installing
   `basicsr` or `timm`.
8. Open a short video, test playback, seek twice, and perform a short export.

Do not declare the release complete until the clean-machine test has passed.

## Build record: v0.14.3-002 on 2026-08-02 JST

This release established the procedure above.

- Public repository: `https://github.com/mioh-labs/mioh`
- Release: `https://github.com/mioh-labs/mioh/releases/tag/v0.14.3-002`
- Universal setup commit: `fab795f` (`Fix universal model setup and portable assets`)
- SwinIR dependency fix: `a8d408c` (`Remove SwinIR timm runtime dependency`)
- Test result: 599 passed, 4 skipped, 68 warnings, and 6 subtests passed
- Published DMG size: 375,766,528 bytes
- Published DMG SHA-256:
  `1b2e10e3f8b0eb53a77d43e46f94e2c96b10fc3dec0a9c503790e1bd96c99eb0`
- The locally built DMG, the `lada_git` artifact, and the DMG downloaded back
  from GitHub Release had the same SHA-256.
- The DMG passed `hdiutil verify` and contained both top-level `.zsh` links.

Actual conversion checks used the Python runtime inside the Universal app:

- Real-ESRGAN x2 converted successfully without `basicsr`.
- Official SwinIR medium x4 (11,715,559 parameters) converted successfully
  without `timm`, producing a 26 MB Core ML package.
- The variable BasicVSR++ source asset discovery accepts the portable 11-asset
  `.aimodel` collection; Core ML detection accepts portable `.mlpackage`
  assets.

Failures found and corrected during this build:

1. The VR detector URL incorrectly pointed to Hugging Face and returned 404.
   It now points to the release asset and is checksum-pinned.
2. Real-ESRGAN conversion failed because the Universal runtime did not contain
   `basicsr`. The exporter now uses the vendored RRDBNet implementation.
3. Real-ESRGAN `pixel_unshuffle` required an iOS 16-or-newer Core ML opset. The
   exporter now sets that deployment target explicitly.
4. SwinIR conversion failed because the Universal runtime did not contain
   `timm`. The exporter now supplies the three pinned-source compatibility
   helpers locally.
5. Portable Core AI/Core ML conversion outputs were present but the Swift app
   looked only for precompiled `.aimodelc`/`.mlmodelc` assets. Asset discovery
   now also accepts `.aimodel` and `.mlpackage`.
6. An older `lada` model-tool copy had previously overwritten fixes already in
   `lada_git`. Both repositories and the embedded DMG scripts were reconciled
   and verified byte-for-byte before this release was published.
