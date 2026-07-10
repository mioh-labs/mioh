# BasicVSR++ Core AI T18 Export Probe Design

## Goal

Create a focused export probe for Lada's BasicVSR++ v1.2 restoration model. The probe will attempt to export the full inference generator as a fixed-shape FP16 Core AI model with an 18-frame input and will report the exact stage and operation that prevents conversion.

This phase is successful when it produces either a loadable `.aimodel` or a precise, machine-readable conversion failure report. It does not add a Core AI runtime backend to Lada.

## Current Facts

- The restoration crop supplied to BasicVSR++ is always `256x256`.
- The generator accepts and returns tensors shaped `(batch, time, channels, height, width)`.
- The export contract is FP16 input and output with shape `[1, 18, 3, 256, 256]`.
- The three channels retain the existing Lada restoration pipeline's channel order; the exporter does not perform a color conversion.
- Temporal propagation uses Python loops and lists whose lengths depend on the number of frames. A fixed 18-frame contract lets `torch.export` specialize and unroll those paths.
- The likely conversion blockers are `grid_sample` in `flow_warp` and modulated deformable convolution in `deform_align`, but the probe must discover the actual blockers instead of assuming them.
- macOS 27, Xcode 27, the Metal Toolchain, `coreai-build`, and `aimodelc` are available on the development Mac.
- `coreai` and `coreai-torch` are not part of Lada's current Python environment.

## Scope

The implementation will add:

- `scripts/apple/export_basicvsrpp_coreai.py`
- focused unit tests for argument handling, the fixed tensor contract, stage reporting, and dependency failures
- a machine-readable JSON report adjacent to the requested output path
- concise usage documentation for the isolated Core AI environment and export command

The probe will use `model.generator`, excluding MMEngine training, preprocessing, discriminator, and GUI/runtime wrappers from the exported graph.

## Approach

The probe will attempt the complete generator first. This gives the most useful evidence because it tests the real interaction among SPyNet, temporal propagation, deformable alignment, reconstruction, and upsampling.

If full export fails, the script records the failure and exits. It will not silently replace operations, switch precision, export only a submodule, or fall back to Core ML. Isolated submodule probes and custom Metal kernels belong to the next phase, guided by this report.

## Command Interface

The script will follow the existing `scripts/apple` exporter conventions and accept:

- `--model`: BasicVSR++ checkpoint, defaulting to the v1.2 model
- `--output`: `.aimodel` destination
- `--frames`: fixed temporal length, default `18`
- `--imgsz`: fixed spatial size, default `256`
- `--seed`: deterministic example input seed
- `--report`: optional JSON report path; otherwise derived from `--output`
- `--allow-overwrite`: permit replacing generated artifacts

The initial implementation accepts configurable values for diagnostics, but the supported production probe contract remains T18, 256x256, batch 1, RGB/BGR channel count 3, and FP16.

## Data Flow

1. Validate arguments, output paths, checkpoint availability, and optional dependencies.
2. Load the v1.2 model on CPU and select its inference generator.
3. Cast the generator and deterministic example input to FP16.
4. Run one PyTorch reference inference and record input/output shapes, dtypes, and summary statistics.
5. Export the fixed graph with `torch.export.export`.
6. Apply the Core AI decomposition table.
7. Convert with `coreai_torch.TorchConverter`.
8. Optimize and save the `.aimodel` asset.
9. Record stage durations and the final outcome in JSON.

The stages are named `preflight`, `load_model`, `reference_inference`, `torch_export`, `decompose`, `coreai_convert`, `optimize`, and `save_asset`.

## Failure Reporting

Every failure report will include:

- success status and failed stage
- exception type and message
- relevant operator names extracted from the exception when available
- Python, PyTorch, coreai, and coreai-torch versions
- model path, checkpoint identity, input contract, and selected dtype
- elapsed time for each completed stage

The report must not include full tensor values, environment variables, credentials, or a Python traceback containing unrelated local data. A concise traceback may be printed to the terminal in verbose mode.

The script returns a nonzero exit code on any failed stage. A conversion failure is an expected probe result, not a silent success.

## Dependency Isolation

`coreai-torch` will not be added to Lada's normal `pyproject.toml` extras in this phase. The beta package may constrain PyTorch or Python versions, so the export command will run in a dedicated environment containing Lada plus Core AI tooling.

The script module will not import Core AI packages at import time. Executing the probe will check and import them during `preflight`, then fail immediately with an actionable installation error when they are unavailable. Unit tests that import and exercise the script helpers must run without Core AI installed.

## Testing

Unit tests will verify:

- default and explicit command arguments
- the fixed `[1, 18, 3, 256, 256]` FP16 input contract
- wrapper output validation using a small fake generator
- deterministic report-path derivation
- stage-specific JSON reports for missing checkpoint and missing Core AI dependencies
- overwrite protection

The real checkpoint probe is an integration command, not a unit test. Its output report will be retained outside Git under `model_weights` or a temporary artifact directory.

## Success Criteria

This phase is complete when all of the following are true:

- the exporter and focused tests are committed on `codex/coreai-basicvsrpp-t18`
- normal unit tests do not require Core AI packages
- the real v1.2 checkpoint reaches `torch_export` or records the exact reason it cannot
- if `torch_export` succeeds, the probe reaches Core AI conversion and records every unsupported operation reported by the converter
- if conversion succeeds, the saved `.aimodel` has the expected fixed FP16 signature
- no Lada CLI, restoration runtime, chunking, or model-selection behavior changes

## Out of Scope

- 36-frame export
- dynamic temporal shapes
- custom Metal or TensorOps kernels
- Core AI runtime integration
- automatic fallback between Core AI and MPS
- changes to overlap or chunk scheduling
- quantization below FP16
- performance or visual-quality claims before a runnable converted model exists
