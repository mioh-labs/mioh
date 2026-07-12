# Standalone mps-deform-conv Integration Design

## Goal

Make the standalone `mioh` GUI match the working CLI environment by bundling `mps-deform-conv 0.2.2` and selecting it for PyTorch BasicVSR++ restoration on Apple MPS. Keep the CLI behavior and the GUI model choices unchanged.

## Problem and root cause

The project already declares `mps-deform-conv` in the `apple` optional dependency group, and the local CLI environment has version 0.2.2 installed. The standalone build does not install that dependency group: it copies `.venv-coreai` into the application runtime and then installs Lada itself with `--no-deps`. Because `.venv-coreai` does not contain `mps_deform_conv`, the packaged GUI reaches the fallback import and reports `ModuleNotFoundError`.

Installing the upstream 0.2.2 source unchanged is not sufficient. Its native extension uses `-std=c++17`, while the Torch headers copied into the standalone Python 3.12 runtime use C++20 APIs including `std::type_identity` and `unordered_map.contains`. A temporary C++20 build against the standalone environment produced an arm64 CPython 3.12 extension, imported successfully, and completed a finite MPS deformable-convolution operation.

## Packaged source

- Vendor the exact upstream `mps-deform-conv 0.2.2` source required to build and run the package under `packaging/macOS/standalone/vendor/mps-deform-conv-0.2.2/`.
- Include the upstream MIT license and retain upstream copyright and attribution.
- Change the extension and just-in-time fallback compiler flags from `-std=c++17` to `-std=c++20`. Do not alter the Metal kernel or deformable-convolution algorithm.
- Keep the vendored package isolated to standalone macOS packaging. Do not replace the dependency declared in `pyproject.toml`, and do not change normal CLI installation.

## Standalone build integration

After copying `.venv-coreai` and installing Lada into the application runtime, `build_app.sh` copies the immutable vendored source into the ignored build directory and installs that staging copy with the bundled Python interpreter. This prevents setuptools build artifacts from modifying the checked-in upstream source:

```zsh
uv pip install \
  --python "$RESOURCES/runtime/bin/python3.12" \
  --break-system-packages \
  --no-deps \
  --no-build-isolation \
  --reinstall \
  "$BUILD_DIR/mps-deform-conv-source"
```

`--no-deps` prevents the package from replacing the bundled Torch build. `--no-build-isolation` ensures the extension compiles against the exact Torch headers and libraries shipped in `mioh.app`.

The build fails immediately if the bundled interpreter cannot import `mps_deform_conv`. A failed native build or import must never produce an apparently successful application bundle.

## GUI runtime behavior

The Swift launcher adds this variable only to the standalone GUI child-process environment:

```text
LADA_DEFORM_CONV_BACKEND=mps_deform_conv
```

This makes PyTorch BasicVSR++ use the bundled MPS implementation directly instead of first attempting TorchVision deformable convolution. The setting has no effect on Core AI restoration models and does not alter the shell CLI environment.

The GUI retains all current restoration choices, including `basicvsrpp-v1.2` and custom models. No model is removed or silently redirected to Core AI.

## Error handling

- If the native extension cannot be built or imported, the application build fails.
- Runtime errors from the extension remain visible as real processing errors; the design does not hide failures or report failed restoration as successful.
- The existing MPS memory fraction, adaptive restoration chunks, and T90 single-worker guard remain unchanged. This task fixes the GUI/CLI dependency mismatch rather than redesigning memory scheduling.

## Verification

Automated and build verification must cover all of the following:

1. The vendored package declares version 0.2.2, uses C++20 in both native-build paths, and contains its MIT license.
2. `build_app.sh` installs the vendored package with `--no-deps` and `--no-build-isolation`.
3. `MiohApp.swift` sets `LADA_DEFORM_CONV_BACKEND=mps_deform_conv` for GUI-launched jobs.
4. The rebuilt application runtime imports `mps_deform_conv` with Python 3.12.
5. The bundled `_C` extension is an arm64 Mach-O binary linked against the bundled Torch libraries through `@rpath`.
6. A small MPS deformable-convolution smoke test returns the expected shape and finite values.
7. The standalone app tests pass, the app builds, and strict code-signature verification succeeds.

## Non-goals

- Do not change CLI defaults, CLI dependencies, or CLI environment variables.
- Do not remove PyTorch BasicVSR++ from the GUI.
- Do not modify the upstream Metal kernel or compensate for model-output differences in this task.
- Do not add global GUI execution locking or change parallel-worker policy in this task.
