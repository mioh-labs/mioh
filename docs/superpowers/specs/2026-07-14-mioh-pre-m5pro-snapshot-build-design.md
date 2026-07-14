# mioh Pre-M5-Pro Snapshot Build Design

## Goal

Preserve the standalone mioh application exactly as it existed immediately
before M5 Pro-only Core AI specialization work began, while retaining the
current M5 Pro-dedicated application and its build cache unchanged.

The historical application will be rebuilt from commit
`695854b06d2a6a59fbc246a536d4c2336722185f` and saved under distinct artifact
names:

- `mioh-pre-m5pro.app`
- `mioh-pre-m5pro-0.11.0-unsigned.dmg`

## Historical Behavior Being Preserved

The selected commit predates the generic compiled Core AI transport and the
M5 Pro-only packaging changes. Its standalone build has the following exact
Core AI behavior:

- the three BasicVSR++ choices, v4-fast detection, and two ROI enhancers resolve
  to source `.aimodel` directories;
- six production source `.aimodel` directories are copied into the application;
- BasicVSR++ T90 is additionally compiled for all 20 architectures reported by
  `coreai-build` for macOS 27;
- those 20 T90 `.aimodelc` directories are packaged, although model-name
  resolution still selects the source T90 `.aimodel`;
- the historical fixed-shape BasicVSR++ Swift runner is retained;
- the experimental T36 batch-two source model remains excluded from the app.

This is intentionally a historical snapshot, not a newly designed universal
runtime. It preserves the old behavior, including the fact that the bundled
T90 specializations were not selected by the application.

## Isolation

The historical commit will be checked out into a temporary detached worktree.
No source file from the current `main` checkout will be replaced or reverted.

The build uses these isolated paths:

- source worktree: a temporary worktree outside the current checkout;
- build directory: `build/macos-standalone-pre-m5pro`;
- compiled Core AI cache:
  `build/macos-standalone-pre-m5pro/compiled-models`;
- Core ML cache:
  `build/macos-standalone-pre-m5pro/compiled-coreml-models`.

The existing paths below are read-only inputs or remain untouched:

- `build/macos-standalone/mioh.app`;
- `build/macos-standalone/mioh-0.11.0-unsigned.dmg`;
- `build/macos-standalone/compiled-models` containing the six M5 Pro `h17s`
  models.

## Build Inputs

Large model assets and `.venv-coreai` are ignored by Git and therefore are not
materialized automatically in a detached worktree. The temporary source tree
will receive an overlay of the current repository's `model_weights` directory
and a link to the current `.venv-coreai` environment. These inputs are the same
model sources and Core AI Python environment used by the current standalone
build; application source code still comes entirely from commit `695854b`.

The existing FFmpeg cache may be reused because it contains immutable external
binaries and does not affect application source behavior. All model compilation
caches remain separate.

## Artifact Naming

The historical build script initially produces `mioh.app` and
`mioh-0.11.0-unsigned.dmg` inside the isolated build directory. After successful
validation, the app directory is renamed to `mioh-pre-m5pro.app`, and a new DMG
is created containing that renamed app.

The bundle's internal display name and bundle identifier remain historical.
Only the filesystem artifact and DMG names change. This avoids altering the
snapshot's application code or `Info.plist`. Consequently, it is an archival
parallel build, not a separately identified macOS product for simultaneous
installation alongside the dedicated app.

## Verification

The build is accepted only if all of the following checks pass:

1. the detached source HEAD is exactly commit `695854b`;
2. the historical Python test suite passes before packaging;
3. the standalone build completes successfully;
4. the embedded `mps-deform-conv` smoke test passes;
5. the app contains exactly the six expected source `.aimodel` directories;
6. the app contains exactly 20 compiled T90 `.aimodelc` directories;
7. the 20 compiled directories cover exactly the architecture list returned by
   `coreai-build list-architectures` for macOS 27;
8. no compiled T18, T36, detection, or enhancer model appears;
9. the current M5 Pro-dedicated app, DMG, and six-model cache remain byte-for-byte
   unchanged, checked with hashes captured before the historical build;
10. `codesign --verify --deep --strict` succeeds for the renamed app;
11. `hdiutil verify` succeeds for the renamed DMG.

The temporary worktree is removed only after all artifact and preservation
checks pass. The isolated build directory and its historical cache remain so
the artifacts can be inspected or copied later.

## Failure Handling

Any missing ignored asset, failed test, Core AI compilation error, unexpected
architecture set, signature failure, DMG failure, or change to the dedicated
artifacts stops the process. A failed build is not promoted to the final
`mioh-pre-m5pro` names.

The current `main` checkout is never reset. If the historical build fails, its
temporary build output may be removed and retried without affecting the M5 Pro
version.

## Success Criteria

- The pre-specialization application is available as
  `build/macos-standalone-pre-m5pro/mioh-pre-m5pro.app`.
- Its verified disk image is available as
  `build/macos-standalone-pre-m5pro/mioh-pre-m5pro-0.11.0-unsigned.dmg`.
- Its app contents match the historical six-source-plus-20-T90 layout.
- The current M5 Pro-dedicated artifacts and cache are unchanged.
- The current checkout remains on `main` with no unrelated working-tree
  modifications.
