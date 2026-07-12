# mioh App Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the standalone macOS product and all distribution artifacts from `Lada` to lowercase `mioh` while preserving the Lada engine and existing Bundle ID.

**Architecture:** Product naming is defined by the Swift header, `Info.plist`, and `build_app.sh`. Tests inspect those sources and the built bundle so visible branding changes together while internal processing names remain stable.

**Tech Stack:** Swift 6, SwiftUI, macOS property lists, zsh build script, Python `unittest`.

## Global Constraints

- Keep `CFBundleIdentifier` exactly `com.okatti.lada.coreai`.
- Keep Python modules, CLI names, environment variables, model filenames, helper executables, and icon source unchanged.
- Use lowercase `mioh` for all user-visible product and artifact names.
- Remove obsolete generated `Lada.app` and `Lada-0.11.0-unsigned.dmg` during the renamed build.

---

### Task 1: Rename the standalone macOS product

**Files:**
- Move: `packaging/macOS/standalone/LadaApp.swift` to `packaging/macOS/standalone/MiohApp.swift`
- Modify: `packaging/macOS/standalone/MiohApp.swift`
- Modify: `packaging/macOS/standalone/Info.plist`
- Modify: `packaging/macOS/standalone/build_app.sh`
- Modify: `tests/test_standalone_app_options.py`

**Interfaces:**
- Produces: `build/macos-standalone/mioh.app/Contents/MacOS/mioh`
- Produces: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`
- Preserves: `CFBundleIdentifier=com.okatti.lada.coreai`

- [ ] **Step 1: Write failing product-name tests**

Update the test source path to `MiohApp.swift`. Add a test that asserts the Swift header contains `Text("mioh")`, the temporary-directory label contains `mioh一時フォルダ`, the plist names and executable equal `mioh`, the Bundle ID remains unchanged, and the build script produces `mioh.app`, executable `mioh`, DMG `mioh-0.11.0-unsigned.dmg`, and volume `mioh`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_standalone_app_options.StandaloneAppOptionTests.test_product_is_named_mioh -v`

Expected: FAIL because `MiohApp.swift` and the renamed product settings do not exist.

- [ ] **Step 3: Apply the minimal rename**

Move the Swift file, change the header and visible temporary-directory label, rename the `@main` type to `MiohStandaloneApp`, update the three plist product-name values, and update all build paths and distribution names. Add cleanup of the old generated app and DMG without changing `lada-coreai-runner`, `lada` package paths, model assets, or environment variable names.

- [ ] **Step 4: Run tests and build verification**

Run: `python -m unittest tests.test_standalone_app_options -v`

Expected: PASS.

Run: `packaging/macOS/standalone/build_app.sh`

Expected: exit 0 with `mioh.app` and `mioh-0.11.0-unsigned.dmg` paths.

Run: `codesign --verify --deep --strict build/macos-standalone/mioh.app`

Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add packaging/macOS/standalone/MiohApp.swift packaging/macOS/standalone/Info.plist packaging/macOS/standalone/build_app.sh tests/test_standalone_app_options.py
git commit -m "Rename standalone app to mioh"
```

### Task 2: Verify the live app identity

**Files:**
- Verify: `build/macos-standalone/mioh.app`

**Interfaces:**
- Consumes the Task 1 bundle.
- Produces a launched app whose accessibility-visible title is `mioh`.

- [ ] **Step 1: Verify bundle metadata and artifacts**

Run `plutil` against the built Info.plist and verify display name, bundle name, and executable are `mioh`, while the Bundle ID is `com.okatti.lada.coreai`. Verify the old generated app and DMG are absent.

- [ ] **Step 2: Launch and inspect**

Quit the idle old app, launch `build/macos-standalone/mioh.app`, and confirm the window/header identity reports `mioh`.

- [ ] **Step 3: Run the complete focused regression suite**

Run: `python -m unittest tests.test_process_video_parallel_progress tests.test_process_video_parallel_executor_selection tests.test_process_video_parallel_shutdown tests.test_standalone_app_options -v`

Expected: all tests pass with no failures or errors.
