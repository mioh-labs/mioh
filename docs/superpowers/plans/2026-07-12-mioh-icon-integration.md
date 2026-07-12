# mioh Icon Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standalone macOS app icon with the user-approved `m + (i inside h) + o` design.

**Architecture:** Store the approved 1254x1254 PNG as a project-owned source asset. Point the existing iconset generation pipeline at that asset so all standard macOS icon sizes and `AppIcon.icns` continue to be produced by the current build script.

**Tech Stack:** PNG, zsh, `sips`, `iconutil`, Swift standalone macOS packaging, Python `unittest`.

## Global Constraints

- Work directly on `main` as requested.
- Preserve `lada/gui/icons/lada-logo-gray.png`.
- The `m`, `i`, `o`, and enclosing `h` use one common stroke thickness.
- The `i` stem reaches the `m/o` x-height, its dot sits above it, and the `h` encloses the complete `i`.
- Do not change the bundle identifier or the lowercase `mioh` product naming.

---

### Task 1: Require the mioh icon source

**Files:**
- Modify: `tests/test_standalone_app_options.py`
- Create: `lada/gui/icons/mioh-icon.png`
- Create: `lada/gui/icons/mioh-icon.png.license`
- Modify: `packaging/macOS/standalone/build_app.sh`

**Interfaces:**
- Consumes: `BUILD_SCRIPT` and repository root paths already defined by the standalone packaging test.
- Produces: a build-time source path ending in `lada/gui/icons/mioh-icon.png`.

- [x] **Step 1: Write the failing test**

Add a test that asserts the new source image exists, is a PNG, and the build script sets `SOURCE_ICON` to it:

```python
def test_standalone_app_uses_mioh_icon(self):
    icon = ROOT / "lada" / "gui" / "icons" / "mioh-icon.png"
    script = BUILD_SCRIPT.read_text()

    self.assertTrue(icon.is_file())
    self.assertEqual(icon.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
    self.assertIn('SOURCE_ICON="$ROOT/lada/gui/icons/mioh-icon.png"', script)
```

- [x] **Step 2: Run the test to verify RED**

Run: `python -m unittest tests.test_standalone_app_options.StandaloneAppOptionTests.test_standalone_app_uses_mioh_icon -v`

Expected: FAIL because `mioh-icon.png` does not yet exist and the build script still names `lada-logo-gray.png`.

- [x] **Step 3: Add the approved asset and switch the source path**

Copy the approved PNG to `lada/gui/icons/mioh-icon.png`, add its AGPL-3.0 sidecar license, and change only this build-script line:

```zsh
SOURCE_ICON="$ROOT/lada/gui/icons/mioh-icon.png"
```

- [x] **Step 4: Run the focused test to verify GREEN**

Run: `python -m unittest tests.test_standalone_app_options.StandaloneAppOptionTests.test_standalone_app_uses_mioh_icon -v`

Expected: PASS.

### Task 2: Build and verify the packaged icon

**Files:**
- Verify: `build/macos-standalone/mioh.app/Contents/Resources/AppIcon.icns`
- Verify: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`

**Interfaces:**
- Consumes: `lada/gui/icons/mioh-icon.png` through `SOURCE_ICON`.
- Produces: a signed `mioh.app` and its unsigned distribution DMG.

- [x] **Step 1: Run the complete standalone test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [x] **Step 2: Build the app and DMG**

Run: `packaging/macOS/standalone/build_app.sh`

Expected: exit 0 with `mioh.app` and `mioh-0.11.0-unsigned.dmg` present.

- [x] **Step 3: Verify icon resources and signature**

Run: `test -s build/macos-standalone/mioh.app/Contents/Resources/AppIcon.icns && codesign --verify --deep --strict build/macos-standalone/mioh.app`

Expected: exit 0.

- [x] **Step 4: Inspect the rendered application icon**

Extract or open the built `AppIcon.icns` at a representative size and verify the `m + (i inside h) + o` mark remains legible and centered.

- [x] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-12-mioh-icon-concept-design.md \
  docs/superpowers/plans/2026-07-12-mioh-icon-integration.md \
  tests/test_standalone_app_options.py \
  lada/gui/icons/mioh-icon.png \
  lada/gui/icons/mioh-icon.png.license \
  packaging/macOS/standalone/build_app.sh
git commit -m "Use approved mioh app icon"
```
