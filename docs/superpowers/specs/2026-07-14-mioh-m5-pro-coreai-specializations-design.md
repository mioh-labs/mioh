# mioh M5 Pro Core AI Specializations Design

## Goal

Make the standalone mioh application use precompiled Core AI models on this
MacBook Pro with Apple M5 Pro instead of merely bundling unused compiled
variants. The application is intentionally machine-specific: it contains only
the `h17s` specialization supported by M5 Pro and removes the other nineteen
architecture variants from both the build cache and the finished application.

## Scope

The following six production Core AI models are compiled for `h17s` and routed
through the compiled runtime:

1. BasicVSR++ v1.2 T18 restoration
2. BasicVSR++ v1.2 T36 restoration
3. BasicVSR++ v1.2 T90 restoration
4. YOLO v4-fast mosaic detection
5. RealESRGAN x4 ROI enhancement
6. RealESRGAN compact x4v3 ROI enhancement

The experimental `basicvsrpp-v1.2-t36-b2-fp16.aimodel` is excluded. Its fixed
batch dimension is two, while the production restoration pipeline and current
runner contract use batch one. Supporting it would require a separate batching
feature and is outside this change.

Source `.aimodel` directories remain in the repository because they are the
inputs needed to rebuild the specializations. They are not copied into the
finished standalone application when a required compiled specialization exists.

## Target Architecture

The standalone build uses `h17s`, which `coreai-build inspect` identifies as
supporting Apple M5 Pro. The build variable is named `COREAI_ARCHITECTURE` and
defaults to `h17s`. An explicit environment override remains available for
development, but the distributed artifact produced by the normal build is M5
Pro-specific.

The build must inspect every compiled output and fail unless:

- `supportedArchitectures` includes the requested architecture;
- `supportedChips` includes `M5 Pro` for the default `h17s` build; and
- the compiled input and output tensor contracts match the source asset.

This prevents a mislabeled or stale cache from being packaged silently.

## Build and Packaging

`packaging/macOS/standalone/build_app.sh` will replace the current all-
architectures Core AI compilation with a table of the six source models. For
each entry it will:

1. derive the expected `<basename>.h17s.aimodelc` output name;
2. compile only `--architecture h17s` when the output is absent or older than
   its source `.aimodel`;
3. inspect and validate the result;
4. copy only that specialization into `Contents/Resources/models`.

Before compiling, the build removes cached `.aimodelc` directories whose
architecture suffix differs from the selected architecture. Before packaging,
it also removes stale Core AI compiled directories from the application model
directory. This makes the cache and final bundle converge on the six required
specializations rather than preserving artifacts from an older universal build.

All seven Core AI source `.aimodel` directories, including the excluded T36
batch-two experiment, are omitted from the standalone application's
`MODEL_ASSETS` copy list. Non-Core-AI weights, Core ML models, and the
repository's source model directories are unchanged.

## Model Resolution

`lada/__init__.py` will resolve each well-known Core AI model through one helper.
Given a source filename, the helper first looks for the selected compiled
specialization name in `LADA_MODEL_WEIGHTS_DIR`, using the architecture exported
as `LADA_COREAI_ARCHITECTURE`. If it exists, it returns the `.aimodelc` path. If
it is missing, it falls back to the source `.aimodel` path so the ordinary CLI
development environment retains its current behavior.

The standalone Swift application exports `LADA_COREAI_ARCHITECTURE=h17s` only
on macOS 27 or newer, alongside the existing Core AI runner path. The three
restoration names, the detection name, and both ROI enhancer names therefore
continue to be selected exactly as before while resolving to compiled assets in
the packaged application.

## Generic Swift Runner

Python's Core AI runtime loads source `.aimodel` assets but does not load the
compiled `.aimodelc` directories directly. The existing Swift runner handles
only BasicVSR++ because it hard-codes one `frames` input and one `restored`
output. It will be generalized for all six fixed-shape models.

### Descriptor

Python writes a JSON descriptor for a runner session containing:

- function name (`main`);
- slot count;
- each input tensor's name, FP16 shape, shared-memory offset, and byte count;
- each output tensor's name, FP16 shape, shared-memory offset, and byte count.

Only contiguous FP16 tensors are supported. Every production Core AI model in
scope already has a fixed FP16 tensor contract. The runner validates dimensions,
offsets, total mapping size, duplicate names, and integer overflow before mapping
the file or loading the model.

### Protocol

The persistent subprocess keeps the existing one-byte slot protocol:

- Python writes one slot byte to request inference;
- Swift reads all input tensors for that slot from shared memory;
- Swift runs the Core AI function;
- Swift validates and writes every output tensor back to that slot;
- Swift returns the completed slot byte;
- byte `255` requests graceful shutdown and byte `254` reports failure.

Requests may remain concurrent where the Python adapter permits it. The current
BasicVSR++ safety limits remain: compiled restoration and T90 use one in-flight
request. Detection and ROI enhancement also start with one slot; concurrency is
not expanded by this change.

## Python Runtime Adapter

A reusable compiled-runtime adapter will own the subprocess, descriptor,
shared-memory file, slots, request lock, response validation, and cleanup. The
existing BasicVSR++ compiled backend will use this adapter instead of embedding
its own fixed layout. Core AI detection and ROI enhancement will select it when
their model path ends in `.aimodelc`; source `.aimodel` paths continue using the
existing in-process Python Core AI runtime.

The model-specific adapters remain responsible for semantic preprocessing and
postprocessing:

- BasicVSR++ supplies `frames` and reads `restored`;
- detection supplies `image` and reads `candidates` plus `prototypes`;
- ROI enhancement supplies `image` and reads `enhanced`.

This keeps the transport generic without moving image or model semantics into
Swift.

## Errors and Cleanup

The build fails immediately when compilation, inspection, architecture
validation, or expected output discovery fails. It must not fall back to copying
all architectures.

At runtime, malformed descriptors, tensor mismatches, subprocess termination,
invalid slot responses, or missing outputs raise explicit model-specific errors.
Temporary shared-memory and descriptor files are removed on normal close,
initialization failure, and interpreter shutdown. A stale or late runner response
cannot be accepted for another slot.

If a compiled asset is absent outside the standalone application, well-known
model resolution falls back to its source `.aimodel`. Inside the completed app,
all six compiled assets are required by packaging verification, so a missing
asset is a build error rather than a silent runtime fallback.

## Testing

Tests are added before production changes and cover:

- all six well-known model names resolving to `.h17s.aimodelc` when present;
- source `.aimodel` fallback when a compiled specialization is absent;
- the standalone environment exporting `h17s` only where Core AI is supported;
- the build command using `--architecture "$COREAI_ARCHITECTURE"`;
- deletion of nonselected cached specializations;
- omission of the six source `.aimodel` assets from the finished app;
- descriptor validation, byte offsets, multiple outputs, response handling,
  cleanup, and subprocess failure in the reusable Python adapter;
- restoration, detection, and ROI adapters choosing compiled transport for
  `.aimodelc` and retaining source transport for `.aimodel`;
- Swift compilation for the macOS 27 Core AI helper;
- a real M5 Pro smoke test loading and running one inference for each of the six
  compiled assets;
- final app inspection proving exactly six `*.h17s.aimodelc` directories, zero
  other Core AI architecture variants, and zero packaged source Core AI assets.

The complete Python test suite, MPS deformable-convolution smoke test, app code
signature verification, embedded-source comparison, and DMG checksum validation
remain final release gates.

## Success Criteria

- Selecting any of the six production Core AI choices in mioh uses an `h17s`
  `.aimodelc` through the persistent Swift runner.
- No source `.aimodel` or non-`h17s` Core AI specialization is present in the
  finished application.
- The Core AI compiled-model cache contains only the six selected `h17s`
  specializations.
- All six models complete a real inference on this Apple M5 Pro.
- Existing non-Core-AI and CLI source-model behavior remains unchanged.
