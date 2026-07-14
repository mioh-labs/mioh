# mioh Portable Core AI Build Design

## Goal

Add a current, offline-capable standalone mioh build that runs on supported
Apple silicon Macs without shipping architecture-specific Core AI
specializations for every machine. The application packages the six production
source `.aimodel` assets and lets Core AI specialize and cache only the model
that the user actually selects.

The existing M5 Pro `h17s`-dedicated build remains available and unchanged.
Both products keep the same application display name and bundle identifier, as
requested. They are alternative builds of mioh rather than applications that
can be installed side by side.

## Chosen Approach

The current standalone build pipeline will gain an explicit distribution mode:

- `dedicated` remains the default and preserves the existing six-model `h17s`
  build;
- `portable` packages the six source `.aimodel` assets and no `.aimodelc`
  directories.

A small portable-build entry point will select `portable` mode and a separate
build directory. The shared script continues to own Python runtime assembly,
non-Core-AI model copying, Swift compilation, signing, MPS verification, and
DMG creation. This avoids maintaining a second copy of the standalone build
logic.

Reusing the archived pre-M5-Pro snapshot is rejected. That snapshot represents
old application code and packages 20 unused T90 specializations in addition to
the source models. Duplicating the complete build script is also rejected
because the two distributions would drift as mioh changes.

## Product Identity and Artifacts

Both distributions retain the existing internal identity:

- display name: `mioh`;
- bundle name: `mioh`;
- bundle identifier: `com.okatti.lada.coreai`;
- executable name: `mioh`.

The portable build is isolated from the dedicated build only at the filesystem
artifact level:

- build directory: `build/macos-standalone-universal`;
- application artifact: `mioh-universal.app`;
- disk image: `mioh-universal-<version>-unsigned.dmg`.

The different artifact filename makes the build output identifiable before
installation. Because the internal bundle identity is shared, installing one
build replaces the other in the normal Applications workflow.

## Packaged Core AI Models

Portable mode packages exactly these six source assets:

1. `basicvsrpp-v1.2-t18-fp16.aimodel`
2. `basicvsrpp-v1.2-t36-fp16.aimodel`
3. `basicvsrpp-v1.2-t90-fp16.aimodel`
4. `lada_mosaic_detection_model_v4_fast-fp16.aimodel`
5. `RealESRGAN_x4plus-256-fp16.aimodel`
6. `realesr-general-x4v3-256-fp16.aimodel`

The experimental `basicvsrpp-v1.2-t36-b2-fp16.aimodel` remains excluded. It
uses a batch-two contract that is not supported by the production restoration
pipeline.

No `.aimodelc` directory may be present anywhere in the portable application.
Portable mode does not invoke `coreai-build`, does not require Xcode or the
Metal Toolchain on the destination Mac, and does not package the 20 historical
architecture variants.

## Runtime Selection and Specialization

The portable Swift application must not export `LADA_COREAI_ARCHITECTURE`.
Without that override, `lada._coreai_model_path` resolves the six well-known
model choices to their source `.aimodel` assets. The existing Python Core AI
adapters then load the selected source through `coreai.runtime.AIModel.load`.

Core AI performs device-specific specialization when that model is first
loaded and manages the resulting cache outside the signed application bundle.
The application does not pre-specialize all six models at launch. For example,
choosing T36 specializes T36 when it is first used; detection and ROI enhancer
models are specialized later only if their corresponding features are enabled.

Subsequent loads reuse Core AI's managed cache when it is valid. mioh does not
manually delete or rewrite the cache and never deletes assets from its signed
bundle.

Dedicated mode continues to export `LADA_COREAI_ARCHITECTURE=h17s`, resolves to
the six packaged `.h17s.aimodelc` assets, and uses the persistent Swift compiled
model runner. Portable mode continues to use the existing in-process Python
Core AI path for source assets.

## Build Configuration

`packaging/macOS/standalone/build_app.sh` will validate a distribution setting
such as `COREAI_DISTRIBUTION=dedicated|portable`. Its default is `dedicated` so
the established M5 Pro build command and output remain compatible.

The build will derive Core AI behavior from the selected mode:

- dedicated: compile, inspect, copy, and verify six selected-architecture
  `.aimodelc` assets exactly as today;
- portable: copy and verify the six source `.aimodel` assets, skip Core AI
  ahead-of-time compilation, and reject compiled assets in the app.

The Swift executable will be compiled with a portable-only build definition.
That definition controls whether the application environment exports the
architecture override. Runtime inference must not depend on the artifact
filename or infer distribution mode by probing the host chip.

The portable entry point will set its isolated build directory and artifact
names before calling the shared build. It must not remove, overwrite, or reuse
the dedicated build's `compiled-models` cache.

## User-Visible First-Use Behavior

Before loading a source Core AI model for the first time in a process, mioh
will emit one concise status message identifying the model and explaining that
the Mac-specific optimization occurs on first use. A successful load emits a
matching completion message. Existing progress output should continue on the
same logical line where practical so the first-use notice does not recreate the
previous high-volume logging problem.

Failure must identify the model that could not be loaded and retain the
underlying Core AI error. The application must not silently switch from the
selected Core AI model to a different restoration backend.

The message does not promise that the cache is absent; Core AI may satisfy the
load from an existing cache. It describes the first-use phase without trying to
inspect or manage private cache state.

## Signing and Mutability

The finished application is treated as read-only after signing. Neither first
launch nor first model use removes unneeded source models or writes a compiled
model back into `Contents/Resources`.

All generated caches, Python bytecode, temporary shared-memory files, and Core
AI specialization data must remain outside the bundle. Final verification
removes incidental embedded Python bytecode before signing and then verifies
that running the packaged application does not invalidate the signature.

## Verification

Tests are added before implementation and cover:

- invalid distribution values failing with a clear error;
- dedicated mode remaining the default;
- portable Swift environment omitting `LADA_COREAI_ARCHITECTURE`;
- dedicated Swift environment continuing to export `h17s`;
- portable model resolution selecting source `.aimodel` paths;
- all six source runtimes retaining their existing adapters;
- portable packaging containing exactly the six expected `.aimodel`
  directories and zero `.aimodelc` directories;
- the experimental T36 batch-two asset remaining absent;
- portable build commands never invoking `coreai-build`;
- the first-use status and model-specific error messages;
- dedicated packaging still containing exactly six `h17s` `.aimodelc`
  directories and no source Core AI assets.

Release verification for the new artifact includes:

1. the complete Python test suite;
2. Swift compilation in portable mode;
3. an embedded-runtime MPS deformable-convolution smoke test;
4. at least one real source `.aimodel` load and inference in the packaged
   runtime, proving automatic specialization works on the build Mac;
5. exact portable model-layout inspection;
6. strict recursive code-signature verification after runtime smoke tests;
7. DMG verification and checksum;
8. before-and-after hashes proving the existing dedicated application, DMG,
   and six-model compiled cache were not changed by the portable build.

## Success Criteria

- `mioh-universal.app` is built from current `main` and runs without a model
  download server or development toolchain.
- It contains exactly the six production source `.aimodel` assets and no
  architecture-specific `.aimodelc` assets.
- Only models actually selected by the user enter Core AI's first-use
  specialization path.
- The application remains validly signed after a real packaged-runtime model
  load.
- The M5 Pro `h17s`-dedicated build path and existing artifacts remain intact.
- Both builds identify themselves internally as the same `mioh` application,
  so installation intentionally replaces the other build.
