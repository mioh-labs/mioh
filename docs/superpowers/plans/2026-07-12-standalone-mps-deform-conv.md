# Standalone mps-deform-conv Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bundle a working `mps-deform-conv 0.2.2` native extension in `mioh.app` and make GUI-launched PyTorch BasicVSR++ restoration use it directly.

**Architecture:** Vendor the pinned PyPI source under standalone packaging, patch only its C++ language level to C++20, and compile it against the exact Torch installation copied into the application runtime. The Swift launcher selects the backend only for GUI child processes; normal CLI behavior remains unchanged.

**Tech Stack:** Python 3.12, PyTorch MPS, Objective-C++, Metal, Swift, zsh, `uv`, Python `unittest`.

## Global Constraints

- Work directly on `main` as requested.
- Vendor exactly version 0.2.2 from PyPI sdist SHA-256 `560659ba50f62f708c710a468174faccf88444fd7f5879c9390a354e054cd1d6`.
- Retain upstream MIT attribution and do not alter the Metal kernel or deformable-convolution algorithm.
- Change both upstream `-std=c++17` compiler paths to `-std=c++20`.
- Build with `--no-deps` and `--no-build-isolation` against the Torch libraries copied into `mioh.app`.
- Keep all GUI model choices. Do not change CLI defaults, CLI environment variables, MPS memory policy, or parallel-worker policy.

---

### Task 1: Vendor the pinned C++20-compatible source

**Files:**
- Create: `packaging/macOS/standalone/vendor/mps-deform-conv-0.2.2/`
- Create: `packaging/macOS/standalone/vendor/mps-deform-conv-0.2.2/LICENSE`
- Create: `packaging/macOS/standalone/vendor/mps-deform-conv-0.2.2/README.lada.md`
- Create: `tests/test_standalone_mps_deform_conv.py`

**Interfaces:**
- Consumes: immutable PyPI 0.2.2 source distribution.
- Produces: a local PEP 517 package installable with CPython 3.12 and bundled Torch headers.

- [ ] **Step 1: Write the failing vendor-integrity test**

Create `tests/test_standalone_mps_deform_conv.py`:

```python
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "macOS" / "standalone"
VENDOR = PACKAGE / "vendor" / "mps-deform-conv-0.2.2"
BUILD_SCRIPT = PACKAGE / "build_app.sh"
APP_SOURCE = PACKAGE / "MiohApp.swift"
VERIFY_SCRIPT = PACKAGE / "verify_mps_deform_conv.py"


class StandaloneMPSDeformConvTests(unittest.TestCase):
    def test_vendored_source_is_pinned_and_uses_cxx20(self):
        setup = (VENDOR / "setup.py").read_text()
        runtime = (VENDOR / "mps_deform_conv" / "__init__.py").read_text()
        metadata = (VENDOR / "pyproject.toml").read_text()
        license_text = (VENDOR / "LICENSE").read_text()

        self.assertIn('version="0.2.2"', setup)
        self.assertIn('version = "0.2.2"', metadata)
        self.assertNotIn("-std=c++17", setup + runtime)
        self.assertEqual((setup + runtime).count("-std=c++20"), 2)
        self.assertIn("MIT License", license_text)
        self.assertIn("imperatormk", license_text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m unittest tests.test_standalone_mps_deform_conv.StandaloneMPSDeformConvTests.test_vendored_source_is_pinned_and_uses_cxx20 -v`

Expected: ERROR because the vendored `setup.py` does not exist.

- [ ] **Step 3: Download and verify the exact source**

```bash
SDIST=/tmp/mps_deform_conv-0.2.2.tar.gz
curl -fL -o "$SDIST" \
  https://files.pythonhosted.org/packages/2b/6b/a7a0d4d90a9bde62d321a32ff383d2c288df85ccf58715062c4aa89bc2d4/mps_deform_conv-0.2.2.tar.gz
echo "560659ba50f62f708c710a468174faccf88444fd7f5879c9390a354e054cd1d6  $SDIST" | shasum -a 256 -c -
```

Expected: `/tmp/mps_deform_conv-0.2.2.tar.gz: OK`.

- [ ] **Step 4: Import the upstream package**

Extract the archive under `/tmp`. Copy only `README.md`, `pyproject.toml`, `setup.cfg`, `setup.py`, and the complete `mps_deform_conv/` directory into the vendor directory. Do not copy generated `build/`, `*.egg-info`, `PKG-INFO`, or upstream tests.

- [ ] **Step 5: Apply the compatibility patch and attribution**

Use `apply_patch` for these exact replacements:

```diff
-extra_compile_args=["-std=c++17", "-O3"],
+extra_compile_args=["-std=c++20", "-O3"],
```

```diff
-extra_cflags=["-std=c++17"],
+extra_cflags=["-std=c++20"],
```

Create `LICENSE` with the standard MIT text and `Copyright (c) imperatormk`. Create `README.lada.md` containing the PyPI source URL, version, SHA-256, upstream repository URL, and the two C++20-only changes.

- [ ] **Step 6: Verify GREEN**

Run the focused test from Step 2. Expected: PASS.

---

### Task 2: Install and select the bundled backend

**Files:**
- Modify: `tests/test_standalone_mps_deform_conv.py`
- Modify: `packaging/macOS/standalone/build_app.sh`
- Modify: `packaging/macOS/standalone/MiohApp.swift`

**Interfaces:**
- Consumes: vendored package from Task 1.
- Produces: CPython 3.12 extension in `mioh.app` and a GUI-only backend environment variable.

- [ ] **Step 1: Write failing build and launcher tests**

Add to `StandaloneMPSDeformConvTests`:

```python
    def test_build_installs_vendor_against_bundled_torch(self):
        script = BUILD_SCRIPT.read_text()
        self.assertIn('VENDORED_MPS_DEFORM_CONV="$PACKAGE_DIR/vendor/mps-deform-conv-0.2.2"', script)
        self.assertIn('--no-deps', script)
        self.assertIn('--no-build-isolation', script)
        self.assertIn('"$VENDORED_MPS_DEFORM_CONV"', script)

    def test_gui_selects_bundled_backend_without_removing_models(self):
        source = APP_SOURCE.read_text()
        self.assertIn('result["LADA_DEFORM_CONV_BACKEND"] = "mps_deform_conv"', source)
        self.assertIn('"basicvsrpp-v1.2"', source)
        self.assertIn('"カスタム"', source)
```

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_standalone_mps_deform_conv -v`

Expected: two new failures because the build install and environment setting are absent.

- [ ] **Step 3: Install the vendor package in `build_app.sh`**

Define:

```zsh
VENDORED_MPS_DEFORM_CONV="$PACKAGE_DIR/vendor/mps-deform-conv-0.2.2"
MPS_DEFORM_BUILD_SOURCE="$BUILD_DIR/mps-deform-conv-source"
```

After installing Lada itself, add:

```zsh
rm -rf "$MPS_DEFORM_BUILD_SOURCE"
ditto "$VENDORED_MPS_DEFORM_CONV" "$MPS_DEFORM_BUILD_SOURCE"
uv pip install \
  --python "$RESOURCES/runtime/bin/python3.12" \
  --break-system-packages \
  --no-deps \
  --no-build-isolation \
  --reinstall \
  "$MPS_DEFORM_BUILD_SOURCE"
```

- [ ] **Step 4: Select it in the GUI environment**

In `RestorationRunner.environment(resources:python:)`, add:

```swift
result["LADA_DEFORM_CONV_BACKEND"] = "mps_deform_conv"
```

Do not add this variable to Python CLI code or shell configuration.

- [ ] **Step 5: Verify GREEN**

Run the command from Step 2. Expected: all tests pass.

---

### Task 3: Fail the build unless real MPS execution works

**Files:**
- Create: `packaging/macOS/standalone/verify_mps_deform_conv.py`
- Modify: `packaging/macOS/standalone/build_app.sh`
- Modify: `tests/test_standalone_mps_deform_conv.py`

**Interfaces:**
- Consumes: installed extension and bundled Torch runtime.
- Produces: exit code 0 only after a finite MPS result with shape `(1, 4, 8, 8)`.

- [ ] **Step 1: Add the failing verifier contract**

Extend `test_build_installs_vendor_against_bundled_torch`:

```python
        self.assertIn('verify_mps_deform_conv.py', script)
        verifier = VERIFY_SCRIPT.read_text()
        self.assertIn("from mps_deform_conv import deform_conv2d", verifier)
        self.assertIn("torch.backends.mps.is_available()", verifier)
        self.assertIn("torch.mps.synchronize()", verifier)
        self.assertIn("torch.isfinite(output).all()", verifier)
```

Run the test module. Expected: ERROR because the verifier does not exist.

- [ ] **Step 2: Create `verify_mps_deform_conv.py`**

```python
import torch
from mps_deform_conv import deform_conv2d


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    device = torch.device("mps")
    torch.manual_seed(7)
    input_tensor = torch.randn(1, 4, 8, 8, device=device)
    offset = torch.randn(1, 18, 8, 8, device=device) * 0.1
    weight = torch.randn(4, 4, 3, 3, device=device)
    bias = torch.randn(4, device=device)
    mask = torch.sigmoid(torch.randn(1, 9, 8, 8, device=device))
    output = deform_conv2d(
        input_tensor, offset, weight, bias,
        stride=1, padding=1, dilation=1, mask=mask,
    )
    torch.mps.synchronize()
    if tuple(output.shape) != (1, 4, 8, 8):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("mps-deform-conv returned non-finite values")
    print("mps-deform-conv smoke test passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Invoke the verifier during the build**

After vendor installation, add:

```zsh
PYTHONHOME="$RESOURCES/runtime" \
PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
  "$RESOURCES/runtime/bin/python3.12" \
  "$PACKAGE_DIR/verify_mps_deform_conv.py"
```

`set -euo pipefail` makes import, link, and MPS execution failures abort the build.

- [ ] **Step 4: Verify GREEN**

Run: `python -m unittest tests.test_standalone_mps_deform_conv tests.test_standalone_app_options -v`

Expected: all focused tests pass.

---

### Task 4: Build, inspect, and commit

**Files:**
- Verify: `build/macos-standalone/mioh.app/Contents/Resources/runtime/lib/python3.12/site-packages/mps_deform_conv/`
- Verify: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: signed app and DMG containing a working CPython 3.12 arm64 extension.

- [ ] **Step 1: Build**

Run: `packaging/macOS/standalone/build_app.sh`

Expected: output includes `mps-deform-conv smoke test passed`, then app and DMG paths.

- [ ] **Step 2: Verify module and linkage**

```bash
APP=build/macos-standalone/mioh.app
PYTHONHOME="$APP/Contents/Resources/runtime" \
PYTHONPATH="$APP/Contents/Resources/runtime/lib/python3.12/site-packages" \
  "$APP/Contents/Resources/runtime/bin/python3.12" -c \
  'import mps_deform_conv; print(mps_deform_conv.__version__)'
file "$APP"/Contents/Resources/runtime/lib/python3.12/site-packages/mps_deform_conv/_C*.so
otool -L "$APP"/Contents/Resources/runtime/lib/python3.12/site-packages/mps_deform_conv/_C*.so
```

Expected: version `0.2.2`, `Mach-O 64-bit bundle arm64`, and Torch libraries referenced through `@rpath`.

- [ ] **Step 3: Verify tests and signature**

```bash
python -m unittest tests.test_standalone_mps_deform_conv tests.test_standalone_app_options -v
codesign --verify --deep --strict build/macos-standalone/mioh.app
git diff --check
```

Expected: focused tests pass and both verification commands exit 0.

- [ ] **Step 4: Commit on `main`**

```bash
git add packaging/macOS/standalone/vendor/mps-deform-conv-0.2.2 \
  packaging/macOS/standalone/verify_mps_deform_conv.py \
  packaging/macOS/standalone/build_app.sh \
  packaging/macOS/standalone/MiohApp.swift \
  tests/test_standalone_mps_deform_conv.py \
  docs/superpowers/plans/2026-07-12-standalone-mps-deform-conv.md
git commit -m "Bundle MPS deform conv in mioh"
```
