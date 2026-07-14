# mioh Pre-M5-Pro Snapshot Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the exact pre-M5-Pro-specialization standalone mioh source snapshot under distinct artifact names without changing the current M5 Pro-dedicated app, DMG, or compiled-model cache.

**Architecture:** A detached worktree at commit `695854b06d2a6a59fbc246a536d4c2336722185f` supplies historical application source. Ignored model assets and the Core AI environment are overlaid from the current checkout, while a separate build directory holds the historical app and its clean 20-architecture T90 cache. Hash manifests protect the current dedicated artifacts before and after the build.

**Tech Stack:** Git worktrees, zsh, Python 3.12, pytest, `coreai-build`, Swift 6, `codesign`, `diskutil`, `hdiutil`, SHA-256 manifests.

## Global Constraints

- Historical source commit is exactly `695854b06d2a6a59fbc246a536d4c2336722185f`.
- Final artifact names are `mioh-pre-m5pro.app` and `mioh-pre-m5pro-0.11.0-unsigned.dmg`.
- Historical build root is `/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro`.
- Current `/Users/okatti/Documents/lada/build/macos-standalone` is never written by the historical build.
- Historical app contains exactly six source `.aimodel` directories and exactly 20 T90 `.aimodelc` directories.
- T18, T36, detection, and both enhancer compiled directories are absent.
- Experimental `basicvsrpp-v1.2-t36-b2-fp16.aimodel` is absent from the app.
- The current dedicated app, DMG, and six-model `h17s` cache remain byte-for-byte unchanged.
- Internal bundle display name and bundle identifier remain historical; only filesystem artifact and DMG names change.
- No repository production source changes are made while executing this artifact-only plan.

---

## File and Artifact Structure

- Create temporarily: `/Users/okatti/Documents/lada-pre-m5pro-snapshot/` — detached historical worktree; removed after verification.
- Create: `/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/` — isolated historical build and cache.
- Produce: `/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro.app` — renamed historical app.
- Produce: `/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro-0.11.0-unsigned.dmg` — renamed verified disk image.
- Create temporarily: `/tmp/mioh-dedicated-app-before.sha256`, `/tmp/mioh-dedicated-cache-before.sha256`, `/tmp/mioh-dedicated-dmg-before.sha256` — preservation baselines; removed after final comparison.

### Task 1: Protect the current dedicated artifacts

**Files:**
- Read: `build/macos-standalone/mioh.app`
- Read: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`
- Read: `build/macos-standalone/compiled-models`
- Create temporarily: `/tmp/mioh-dedicated-*-before.sha256`

**Interfaces:**
- Produces: relative-path SHA-256 manifests consumed by Task 7.

- [ ] **Step 1: Confirm the current checkout and dedicated artifacts**

Run:

```zsh
cd /Users/okatti/Documents/lada
test "$(git branch --show-current)" = main
test -d build/macos-standalone/mioh.app
test -f build/macos-standalone/mioh-0.11.0-unsigned.dmg
test "$(find build/macos-standalone/compiled-models -maxdepth 1 -type d -name '*.h17s.aimodelc' | wc -l | tr -d ' ')" = 6
git status --short --branch
```

Expected: branch is `main`, all three tests exit 0, and the worktree has no uncommitted files.

- [ ] **Step 2: Capture deterministic dedicated-app and cache manifests**

Run:

```zsh
(cd build/macos-standalone/mioh.app && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) > /tmp/mioh-dedicated-app-before.sha256
(cd build/macos-standalone/compiled-models && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) > /tmp/mioh-dedicated-cache-before.sha256
shasum -a 256 build/macos-standalone/mioh-0.11.0-unsigned.dmg > /tmp/mioh-dedicated-dmg-before.sha256
test -s /tmp/mioh-dedicated-app-before.sha256
test -s /tmp/mioh-dedicated-cache-before.sha256
test -s /tmp/mioh-dedicated-dmg-before.sha256
```

Expected: all three manifest files are nonempty.

### Task 2: Materialize the historical source snapshot

**Files:**
- Create temporarily: `/Users/okatti/Documents/lada-pre-m5pro-snapshot`
- Consume: `/Users/okatti/Documents/lada/model_weights`
- Consume: `/Users/okatti/Documents/lada/.venv-coreai`

**Interfaces:**
- Produces: detached historical source root in `LEGACY_SOURCE` for Tasks 3–6.

- [ ] **Step 1: Use `superpowers:using-git-worktrees` to create the detached worktree**

Run after that skill validates the location:

```zsh
cd /Users/okatti/Documents/lada
LEGACY_SOURCE=/Users/okatti/Documents/lada-pre-m5pro-snapshot
test ! -e "$LEGACY_SOURCE"
git worktree add --detach "$LEGACY_SOURCE" 695854b06d2a6a59fbc246a536d4c2336722185f
test "$(git -C "$LEGACY_SOURCE" rev-parse HEAD)" = 695854b06d2a6a59fbc246a536d4c2336722185f
```

Expected: Git reports a detached checkout at `695854b`, and the final test exits 0.

- [ ] **Step 2: Overlay ignored build inputs without changing historical source**

Run:

```zsh
LEGACY_SOURCE=/Users/okatti/Documents/lada-pre-m5pro-snapshot
ditto /Users/okatti/Documents/lada/model_weights "$LEGACY_SOURCE/model_weights"
ln -s /Users/okatti/Documents/lada/.venv-coreai "$LEGACY_SOURCE/.venv-coreai"
for asset in \
  basicvsrpp-v1.2-t18-fp16.aimodel \
  basicvsrpp-v1.2-t36-fp16.aimodel \
  basicvsrpp-v1.2-t90-fp16.aimodel \
  lada_mosaic_detection_model_v4_fast-fp16.aimodel \
  RealESRGAN_x4plus-256-fp16.aimodel \
  realesr-general-x4v3-256-fp16.aimodel; do
  test -d "$LEGACY_SOURCE/model_weights/$asset"
done
test -d "$LEGACY_SOURCE/.venv-coreai/lib/python3.12/site-packages"
test -z "$(git -C "$LEGACY_SOURCE" status --short --untracked-files=no)"
```

Expected: all six source assets and Core AI packages exist, while tracked historical source remains clean.

### Task 3: Verify the historical checkout before packaging

**Files:**
- Test: `/Users/okatti/Documents/lada-pre-m5pro-snapshot/tests`
- Read: historical source tree.

**Interfaces:**
- Consumes: `LEGACY_SOURCE` from Task 2.
- Produces: passing baseline required before the release build.

- [ ] **Step 1: Run the historical Python suite from its own source root**

Run:

```zsh
cd /Users/okatti/Documents/lada-pre-m5pro-snapshot
PYTHONPATH="$PWD" python -m pytest -q
```

Expected: `290 passed, 2 skipped` and exit status 0.

- [ ] **Step 2: Confirm historical Core AI routing and build contracts**

Run:

```zsh
cd /Users/okatti/Documents/lada-pre-m5pro-snapshot
rg -n "basicvsrpp-v1.2-t90-fp16.aimodel" lada/__init__.py packaging/macOS/standalone/build_app.sh
test "$(rg -c -- '--architecture' packaging/macOS/standalone/build_app.sh)" = 0
test "$(rg -c 'CommandLine.arguments.count == 5' packaging/macOS/standalone/CoreAIRunner.swift)" = 1
```

Expected: model resolution points to source `.aimodel`, the build has no architecture restriction, and the historical runner uses its five-argument fixed-frame contract.

### Task 4: Build the isolated pre-specialization application

**Files:**
- Execute: historical `packaging/macOS/standalone/build_app.sh`
- Create: `build/macos-standalone-pre-m5pro`

**Interfaces:**
- Consumes: historical source and ignored inputs from Tasks 2–3.
- Produces: historical `mioh.app`, original DMG, and clean all-architecture T90 cache for Tasks 5–6.

- [ ] **Step 1: Remove only the isolated historical build directory**

Run:

```zsh
LEGACY_BUILD=/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro
rm -rf "$LEGACY_BUILD"
test ! -e "$LEGACY_BUILD"
```

Expected: the isolated directory is absent; the current `build/macos-standalone` remains present.

- [ ] **Step 2: Run the historical standalone build with isolated model caches**

Run:

```zsh
LEGACY_SOURCE=/Users/okatti/Documents/lada-pre-m5pro-snapshot
LEGACY_BUILD=/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro
cd "$LEGACY_SOURCE"
BUILD_DIR="$LEGACY_BUILD" \
COMPILED_MODELS="$LEGACY_BUILD/compiled-models" \
COMPILED_COREML_MODELS="$LEGACY_BUILD/compiled-coreml-models" \
FFMPEG_CACHE=/Users/okatti/Documents/lada/build/macos-standalone/ffmpeg-static \
  packaging/macOS/standalone/build_app.sh
```

Expected: MPS smoke passes, `coreai-build` reports `1 of 20` through `20 of 20`, and the script produces `$LEGACY_BUILD/mioh.app` plus `$LEGACY_BUILD/mioh-0.11.0-unsigned.dmg`.

### Task 5: Prove the historical Core AI asset layout

**Files:**
- Read: `build/macos-standalone-pre-m5pro/mioh.app/Contents/Resources/models`
- Read: `build/macos-standalone-pre-m5pro/compiled-models`

**Interfaces:**
- Consumes: historical build from Task 4.
- Produces: exact six-source and 20-T90 verification evidence.

- [ ] **Step 1: Validate exact source and compiled directory sets**

Run:

```zsh
python - /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh.app/Contents/Resources/models <<'PY'
from pathlib import Path
import subprocess
import sys

models = Path(sys.argv[1])
expected_sources = {
    "basicvsrpp-v1.2-t18-fp16.aimodel",
    "basicvsrpp-v1.2-t36-fp16.aimodel",
    "basicvsrpp-v1.2-t90-fp16.aimodel",
    "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
    "RealESRGAN_x4plus-256-fp16.aimodel",
    "realesr-general-x4v3-256-fp16.aimodel",
}
architectures = {
    item.strip()
    for item in subprocess.check_output(
        [
            "xcrun", "coreai-build", "list-architectures",
            "--platform", "macOS",
            "--minimum-deployment-version", "27.0",
        ],
        text=True,
    ).split(",")
}
expected_compiled = {
    f"basicvsrpp-v1.2-t90-fp16.{architecture}.aimodelc"
    for architecture in architectures
}
actual_sources = {item.name for item in models.iterdir() if item.name.endswith(".aimodel")}
actual_compiled = {item.name for item in models.iterdir() if item.name.endswith(".aimodelc")}
assert len(architectures) == 20, architectures
assert actual_sources == expected_sources, (actual_sources, expected_sources)
assert actual_compiled == expected_compiled, (actual_compiled, expected_compiled)
assert "basicvsrpp-v1.2-t36-b2-fp16.aimodel" not in actual_sources
print("historical Core AI layout verified: 6 source + 20 T90 compiled")
PY
```

Expected: `historical Core AI layout verified: 6 source + 20 T90 compiled`.

- [ ] **Step 2: Inspect all 20 compiled outputs and their filename architecture**

Run:

```zsh
python - /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/compiled-models <<'PY'
import json
from pathlib import Path
import subprocess
import sys

cache = Path(sys.argv[1])
models = sorted(cache.glob("*.aimodelc"))
assert len(models) == 20, len(models)
for model in models:
    architecture = model.name.removesuffix(".aimodelc").rsplit(".", 1)[1]
    details = json.loads(
        subprocess.check_output(
            ["xcrun", "coreai-build", "inspect", str(model), "--json"],
            text=True,
        )
    )
    assert details["supportedArchitectures"] == [architecture], (model, details)
print("all 20 compiled T90 specializations inspected")
PY
```

Expected: `all 20 compiled T90 specializations inspected`.

- [ ] **Step 3: Verify all six historical names resolve to source assets**

Run:

```zsh
LEGACY_APP=/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh.app
PYTHONHOME="$LEGACY_APP/Contents/Resources/runtime" \
PYTHONPATH="$LEGACY_APP/Contents/Resources/runtime/lib/python3.12/site-packages" \
LADA_MODEL_WEIGHTS_DIR="$LEGACY_APP/Contents/Resources/models" \
  "$LEGACY_APP/Contents/Resources/runtime/bin/python3.12" - <<'PY'
from lada import ModelFiles

models = [
    ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2-coreai"),
    ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2-coreai-t36"),
    ModelFiles.get_restoration_model_by_name("basicvsrpp-v1.2-coreai-t90"),
    ModelFiles.get_detection_model_by_name("v4-fast-coreai"),
    ModelFiles.get_enhancer_model_by_name("realesrgan-x4-coreai"),
    ModelFiles.get_enhancer_model_by_name("realesr-general-x4v3-coreai"),
]
assert all(model is not None and model.path.endswith(".aimodel") for model in models)
print("all six historical Core AI names resolve to source assets")
PY
```

Expected: `all six historical Core AI names resolve to source assets`.

### Task 6: Promote the verified snapshot under distinct names

**Files:**
- Rename: `mioh.app` to `mioh-pre-m5pro.app`
- Create: `mioh-pre-m5pro-0.11.0-unsigned.dmg`
- Recreate: isolated `dmg-root` only.

**Interfaces:**
- Consumes: validated historical app from Task 5.
- Produces: final user-facing artifacts.

- [ ] **Step 1: Rename the app and rebuild a DMG containing the renamed bundle**

Run:

```zsh
LEGACY_BUILD=/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro
rm -f "$LEGACY_BUILD/mioh-0.11.0-unsigned.dmg" "$LEGACY_BUILD/mioh-pre-m5pro-0.11.0-unsigned.dmg"
mv "$LEGACY_BUILD/mioh.app" "$LEGACY_BUILD/mioh-pre-m5pro.app"
rm -rf "$LEGACY_BUILD/dmg-root"
mkdir -p "$LEGACY_BUILD/dmg-root"
ditto "$LEGACY_BUILD/mioh-pre-m5pro.app" "$LEGACY_BUILD/dmg-root/mioh-pre-m5pro.app"
ln -s /Applications "$LEGACY_BUILD/dmg-root/Applications"
diskutil image create from \
  --volumeName "mioh pre-M5Pro" \
  --format UDZO \
  "$LEGACY_BUILD/dmg-root" \
  "$LEGACY_BUILD/mioh-pre-m5pro-0.11.0-unsigned.dmg"
```

Expected: the renamed app and DMG exist, and no unrenamed `mioh.app` remains in the isolated build root.

- [ ] **Step 2: Verify signature, DMG checksum, and historical identity**

Run:

```zsh
LEGACY_BUILD=/Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro
codesign --verify --deep --strict --verbose=2 "$LEGACY_BUILD/mioh-pre-m5pro.app"
hdiutil verify "$LEGACY_BUILD/mioh-pre-m5pro-0.11.0-unsigned.dmg"
test "$(plutil -extract CFBundleDisplayName raw "$LEGACY_BUILD/mioh-pre-m5pro.app/Contents/Info.plist")" = mioh
test "$(plutil -extract CFBundleIdentifier raw "$LEGACY_BUILD/mioh-pre-m5pro.app/Contents/Info.plist")" = com.okatti.lada.coreai
```

Expected: signature is valid, DMG checksum is valid, display name is `mioh`, and bundle identifier is `com.okatti.lada.coreai`.

### Task 7: Prove the dedicated build was untouched

**Files:**
- Compare: current dedicated app/cache/DMG against Task 1 manifests.
- Read: current Git state.

**Interfaces:**
- Consumes: preservation manifests from Task 1.
- Produces: final non-regression evidence.

- [ ] **Step 1: Regenerate and compare current dedicated manifests**

Run:

```zsh
cd /Users/okatti/Documents/lada
(cd build/macos-standalone/mioh.app && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) | diff -u /tmp/mioh-dedicated-app-before.sha256 -
(cd build/macos-standalone/compiled-models && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) | diff -u /tmp/mioh-dedicated-cache-before.sha256 -
shasum -a 256 -c /tmp/mioh-dedicated-dmg-before.sha256
```

Expected: both `diff` commands produce no output and the DMG check reports `OK`.

- [ ] **Step 2: Run final artifact and repository checks**

Run:

```zsh
test -d /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro.app
test -f /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro-0.11.0-unsigned.dmg
git -C /Users/okatti/Documents/lada diff --check
git -C /Users/okatti/Documents/lada status --short --branch
du -sh /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro.app
du -sh /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro-0.11.0-unsigned.dmg
```

Expected: both artifacts exist, `git diff --check` exits 0, and `main` has no uncommitted changes.

### Task 8: Remove only temporary execution state

**Files:**
- Remove: detached worktree `/Users/okatti/Documents/lada-pre-m5pro-snapshot`
- Remove: three `/tmp/mioh-dedicated-*-before.sha256` manifests.
- Preserve: all contents of `build/macos-standalone-pre-m5pro`.

**Interfaces:**
- Consumes: fully verified artifacts and preservation evidence.
- Produces: clean repository/worktree state with historical artifacts retained.

- [ ] **Step 1: Remove the detached worktree from the main repository**

Run:

```zsh
cd /Users/okatti/Documents/lada
git worktree remove /Users/okatti/Documents/lada-pre-m5pro-snapshot
git worktree prune
test ! -e /Users/okatti/Documents/lada-pre-m5pro-snapshot
```

Expected: detached source tree is absent and Git reports no stale worktree.

- [ ] **Step 2: Remove temporary manifests and report final paths**

Run:

```zsh
rm -f /tmp/mioh-dedicated-app-before.sha256 \
  /tmp/mioh-dedicated-cache-before.sha256 \
  /tmp/mioh-dedicated-dmg-before.sha256
git status --short --branch
print "App: /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro.app"
print "DMG: /Users/okatti/Documents/lada/build/macos-standalone-pre-m5pro/mioh-pre-m5pro-0.11.0-unsigned.dmg"
```

Expected: `main` is clean and both final artifact paths are printed.

## Final Review Checklist

- [ ] Re-read `docs/superpowers/specs/2026-07-14-mioh-pre-m5pro-snapshot-build-design.md` and map every success criterion to Tasks 1–8.
- [ ] Record the historical pytest pass/skip count.
- [ ] Record all 20 architecture names and confirm each compiled model's inspected architecture.
- [ ] Confirm six source model names resolve to `.aimodel` inside the historical app.
- [ ] Confirm MPS smoke, code signature, and DMG checksum pass.
- [ ] Confirm dedicated app/cache/DMG manifests are unchanged.
- [ ] Confirm current checkout remains clean on `main`.
