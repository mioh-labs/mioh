# mioh M5 Pro Core AI Specializations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all six production Core AI choices in the standalone mioh app use M5 Pro-specific `h17s` compiled models and remove every other Core AI specialization and source asset from the finished app.

**Architecture:** A reusable Python shared-memory adapter sends fixed FP16 tensor dictionaries to a generalized persistent Swift Core AI runner. Model resolution prefers `<source-base>.h17s.aimodelc` when the standalone environment exports `LADA_COREAI_ARCHITECTURE=h17s`; the build compiles and packages exactly six such assets.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Swift 6, CoreAI, mmap, subprocess pipes, unittest/pytest, zsh standalone packaging.

## Global Constraints

- The normal standalone build is dedicated to Apple M5 Pro architecture `h17s`.
- Core AI remains available only on macOS 27 and newer; the main GUI target remains macOS 26.
- Compile T18, T36, T90, v4-fast detection, RealESRGAN x4, and RealESRGAN compact x4v3 only.
- Do not package the experimental T36 batch-two model.
- Keep source `.aimodel` directories in repository `model_weights`; omit all seven from the finished app.
- Keep source `.aimodel` fallback behavior for non-standalone CLI development.
- The compiled transport accepts contiguous FP16 tensors with fixed shapes only.
- Do not increase model concurrency; use one shared-memory slot per compiled runtime.
- Preserve non-Core-AI, Core ML, MPS deformable-convolution, encoding, and parallel-worker behavior.

---

## File Structure

- Create `lada/coreai/__init__.py`: package boundary for compiled Core AI transport.
- Create `lada/coreai/compiled_runtime.py`: tensor descriptors, shared-memory layout, runner lifecycle, one-slot request protocol.
- Create `tests/test_coreai_compiled_runtime.py`: real layout validation and fake-process transport tests.
- Modify `packaging/macOS/standalone/CoreAIRunner.swift`: descriptor-driven multi-input/multi-output inference.
- Modify `lada/restorationpipeline/basicvsrpp_coreai_restorer.py`: delegate compiled BasicVSR++ calls to the shared adapter.
- Modify `lada/models/yolo/yolo11_coreai_segmentation_model.py`: accept `.aimodelc` and select compiled transport.
- Modify `lada/restorationpipeline/coreai_roi_enhancer.py`: select compiled transport for `.aimodelc`.
- Modify `lada/__init__.py`: resolve all six well-known Core AI names to the selected specialization when present.
- Modify `packaging/macOS/standalone/MiohApp.swift`: export `LADA_COREAI_ARCHITECTURE=h17s` on supported systems.
- Modify `packaging/macOS/standalone/build_app.sh`: compile, validate, clean, and package only the six `h17s` assets.
- Create `packaging/macOS/standalone/verify_coreai_models.py`: run one real inference through each packaged compiled model.
- Modify focused existing tests for model routing and standalone build contracts.

---

### Task 1: Reusable compiled Core AI transport

**Files:**
- Create: `lada/coreai/__init__.py`
- Create: `lada/coreai/compiled_runtime.py`
- Create: `tests/test_coreai_compiled_runtime.py`

**Interfaces:**
- Produces: `TensorSpec(name: str, shape: tuple[int, ...])`
- Produces: `CompiledCoreAIRuntime(model_path: Path, inputs: tuple[TensorSpec, ...], outputs: tuple[TensorSpec, ...], runner_path: str | None = None)`
- Produces: `CompiledCoreAIRuntime.infer(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]`
- Produces: `CompiledCoreAIRuntime.close() -> None`

- [ ] **Step 1: Write failing descriptor and transport tests**

Create tests that require deterministic contiguous offsets, exact FP16 validation, two output tensors, slot response validation, and cleanup:

```python
import io
import numpy as np

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec


class FakeProcess:
    def __init__(self, completed_slots: bytes):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(completed_slots)
        self.stderr = io.BytesIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15


def test_descriptor_assigns_nonoverlapping_offsets(tmp_path):
    model = tmp_path / "model.h17s.aimodelc"
    model.mkdir()
    runtime = CompiledCoreAIRuntime(
        model,
        inputs=(TensorSpec("image", (1, 3, 2, 2)),),
        outputs=(
            TensorSpec("candidates", (1, 2, 3)),
            TensorSpec("prototypes", (1, 1, 2, 2)),
        ),
    )
    descriptor = runtime.descriptor
    assert [item["offset"] for item in descriptor["inputs"]] == [0]
    assert descriptor["outputs"][0]["offset"] == 24
    assert descriptor["outputs"][1]["offset"] == 36
    assert descriptor["slotStride"] == 44


def test_infer_writes_input_and_reads_two_outputs(tmp_path):
    model = tmp_path / "detect.h17s.aimodelc"
    model.mkdir()
    process = FakeProcess(completed_slots=b"\x00")
    runtime = CompiledCoreAIRuntime(
        model,
        inputs=(TensorSpec("image", (1, 3, 2, 2)),),
        outputs=(TensorSpec("a", (1, 2)), TensorSpec("b", (1, 1))),
        runner_path="/fake/lada-coreai-runner",
        process_factory=lambda *args, **kwargs: process,
    )
    runtime._ensure_started()
    runtime._write_tensor("a", np.array([[7, 8]], dtype=np.float16))
    runtime._write_tensor("b", np.array([[9]], dtype=np.float16))
    result = runtime.infer({"image": np.zeros((1, 3, 2, 2), np.float16)})
    assert result["a"].tolist() == [[7.0, 8.0]]
    assert result["b"].tolist() == [[9.0]]
    runtime.close()
    assert process.stdin.getvalue() == b"\x00\xff"
    assert not runtime.shared_path.exists()
    assert not runtime.descriptor_path.exists()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest -q tests/test_coreai_compiled_runtime.py`

Expected: collection fails because `lada.coreai.compiled_runtime` does not exist.

- [ ] **Step 3: Implement tensor layout and subprocess transport**

Implement the public types and use a JSON descriptor shaped exactly as follows:

```python
@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]

    @property
    def byte_count(self) -> int:
        return math.prod(self.shape) * np.dtype(np.float16).itemsize


class CompiledCoreAIRuntime:
    def __init__(self, model_path, inputs, outputs, runner_path=None,
                 process_factory=subprocess.Popen):
        self.model_path = Path(model_path)
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.runner_path = runner_path
        self._process_factory = process_factory
        self._lock = threading.Lock()

    def infer(self, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._lock:
            self._ensure_started()
            self._write_inputs(values)
            self._request_slot_zero()
            return self._read_outputs()

    def close(self) -> None:
        process, mapping = self._process, self._mapping
        self._process = self._mapping = None
        try:
            if process is not None and process.stdin is not None:
                process.stdin.write(b"\xff")
                process.stdin.flush()
                process.wait(timeout=5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            if process is not None:
                process.terminate()
        finally:
            if mapping is not None:
                mapping.close()
            if self._shared_file is not None:
                self._shared_file.close()
                self._shared_file = None
            self.shared_path.unlink(missing_ok=True)
            self.descriptor_path.unlink(missing_ok=True)
```

Reject empty/duplicate names, nonpositive dimensions, non-FP16 values, missing
or extra inputs, wrong shapes, response `254`, any response other than slot
zero, premature EOF, integer overflow, and runner startup failure. JSON must use
`function`, `slotCount`, `slotStride`, `inputs`, and `outputs`; every tensor item
must contain `name`, `shape`, `offset`, and `byteCount`.

- [ ] **Step 4: Run focused transport tests and verify GREEN**

Run: `python -m pytest -q tests/test_coreai_compiled_runtime.py`

Expected: all tests pass; temporary descriptor and mapping paths are absent after close and startup failure.

- [ ] **Step 5: Commit the transport**

```bash
git add lada/coreai/__init__.py lada/coreai/compiled_runtime.py tests/test_coreai_compiled_runtime.py
git commit -m "feat: add generic compiled Core AI transport"
```

---

### Task 2: Descriptor-driven Swift Core AI runner

**Files:**
- Modify: `packaging/macOS/standalone/CoreAIRunner.swift`
- Modify: `tests/test_standalone_app_options.py`

**Interfaces:**
- Consumes: descriptor schema from Task 1.
- Produces CLI: `lada-coreai-runner <model.aimodelc> <descriptor.json> <shared-file>`
- Preserves protocol: request/completion slot bytes, `255` stop, `254` failure.

- [ ] **Step 1: Add failing Swift source-contract tests**

Require Codable descriptor structs, validation, dynamic input/output loops, and
the new three-argument runner contract:

```python
def test_coreai_runner_is_descriptor_driven(self):
    source = COREAI_RUNNER_SOURCE.read_text()
    for contract in [
        "struct TensorDescriptor: Decodable",
        "struct RunnerDescriptor: Decodable",
        "descriptor.slotCount",
        "descriptor.slotStride",
        "for input in descriptor.inputs",
        "for output in descriptor.outputs",
        'model.loadFunction(named: descriptor.function)',
        "CommandLine.arguments.count == 4",
    ]:
        self.assertIn(contract, source)
    self.assertNotIn('missingOutput("restored")', source)
```

- [ ] **Step 2: Run the contract test and verify RED**

Run: `python -m pytest -q tests/test_standalone_app_options.py::StandaloneAppOptionTests::test_coreai_runner_is_descriptor_driven`

Expected: FAIL because descriptor structs are absent.

- [ ] **Step 3: Generalize the Swift runner**

Replace fixed frame-count logic with these decoded types and checked byte ranges:

```swift
struct TensorDescriptor: Decodable {
  let name: String
  let shape: [Int]
  let offset: Int
  let byteCount: Int
}

struct RunnerDescriptor: Decodable {
  let function: String
  let slotCount: Int
  let slotStride: Int
  let inputs: [TensorDescriptor]
  let outputs: [TensorDescriptor]
}
```

For each request, calculate `slotBase = slot * descriptor.slotStride`, copy every
input region into an FP16 `NDArray`, run `function.run(inputs:)`, validate each
named output's scalar type, shape, contiguity, and byte count, then copy it into
its described output region. Validate all descriptor ranges before `mmap`.

- [ ] **Step 4: Compile with warnings as errors and run the contract tests**

Run:

```bash
xcrun swiftc -warnings-as-errors -O -parse-as-library \
  -target arm64-apple-macosx27.0 -framework CoreAI \
  packaging/macOS/standalone/CoreAIRunner.swift \
  -o /tmp/lada-coreai-runner-plan-test
python -m pytest -q tests/test_standalone_app_options.py
rm -f /tmp/lada-coreai-runner-plan-test
```

Expected: Swift exit 0 and standalone tests pass.

- [ ] **Step 5: Commit the runner**

```bash
git add packaging/macOS/standalone/CoreAIRunner.swift tests/test_standalone_app_options.py
git commit -m "feat: generalize compiled Core AI runner"
```

---

### Task 3: BasicVSR++ compiled adapter migration

**Files:**
- Modify: `lada/restorationpipeline/basicvsrpp_coreai_restorer.py`
- Modify: `tests/test_basicvsrpp_coreai_restorer.py`

**Interfaces:**
- Consumes: `CompiledCoreAIRuntime` and `TensorSpec` from Task 1.
- Preserves: `CoreAIModelRuntime.infer_many(inputs) -> list[torch.Tensor]`.

- [ ] **Step 1: Replace the old runner assertion with a failing generic-adapter test**

```python
def test_compiled_basicvsr_uses_generic_tensor_contract(self):
    runtime = CoreAIModelRuntime(Path("model-t36.h17s.aimodelc"), frame_count=36)
    with mock.patch(
        "lada.restorationpipeline.basicvsrpp_coreai_restorer.CompiledCoreAIRuntime"
    ) as compiled:
        runtime._ensure_loaded()
    compiled.assert_called_once_with(
        Path("model-t36.h17s.aimodelc"),
        inputs=(TensorSpec("frames", (1, 36, 3, 256, 256)),),
        outputs=(TensorSpec("restored", (1, 36, 3, 256, 256)),),
    )
```

- [ ] **Step 2: Run focused BasicVSR++ tests and verify RED**

Run: `python -m pytest -q tests/test_basicvsrpp_coreai_restorer.py`

Expected: FAIL because `CoreAIModelRuntime` still owns its fixed subprocess bridge.

- [ ] **Step 3: Delegate compiled calls to the shared adapter**

For `.aimodelc`, construct fixed `frames` and `restored` specs from
`frame_count`, call `compiled.infer({"frames": input_array})`, and convert
`result["restored"]` back to a copied CPU tensor. Keep the current source
`.aimodel` asyncio backend, memory-pressure concurrency, frame validation,
padding, crossfade, and public class names unchanged. Remove the duplicated mmap,
Popen, slot, and `_read_exact` implementation from this module.

- [ ] **Step 4: Run BasicVSR++ and parallel Core AI tests**

Run:

```bash
python -m pytest -q \
  tests/test_basicvsrpp_coreai_restorer.py \
  tests/test_process_video_parallel_coreai.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the migration**

```bash
git add lada/restorationpipeline/basicvsrpp_coreai_restorer.py tests/test_basicvsrpp_coreai_restorer.py
git commit -m "refactor: share compiled Core AI restoration transport"
```

---

### Task 4: Compiled Core AI detection

**Files:**
- Modify: `lada/models/yolo/yolo11_coreai_segmentation_model.py`
- Modify: `tests/test_coreai_segmentation_model.py`

**Interfaces:**
- Consumes: `CompiledCoreAIRuntime` from Task 1.
- Produces the existing callable result `(candidates, prototypes)`.

- [ ] **Step 1: Write failing compiled-detection tests**

```python
def test_compiled_detection_uses_swift_tensor_contract(tmp_path):
    model_path = tmp_path / "detect.h17s.aimodelc"
    model_path.mkdir()
    with mock.patch(
        "lada.models.yolo.yolo11_coreai_segmentation_model.CompiledCoreAIRuntime"
    ) as compiled:
        runtime = CoreAISegmentationRuntime(model_path)
        runtime._ensure_loaded()
    compiled.assert_called_once_with(
        model_path,
        inputs=(TensorSpec("image", (1, 3, 640, 640)),),
        outputs=(
            TensorSpec("candidates", (1, 38, 8400)),
            TensorSpec("prototypes", (1, 32, 160, 160)),
        ),
    )


def test_detection_model_accepts_compiled_path(tmp_path):
    path = tmp_path / "detect.h17s.aimodelc"
    path.mkdir()
    model = Yolo11CoreAISegmentationModel(path, runtime=RecordingRuntime())
    assert model.runtime is not None
```

- [ ] **Step 2: Run detection tests and verify RED**

Run: `python -m pytest -q tests/test_coreai_segmentation_model.py`

Expected: FAIL because `.aimodelc` is rejected and compiled transport is absent.

- [ ] **Step 3: Add compiled detection selection**

Accept both `.aimodel` and `.aimodelc`. Keep the existing Python runtime for
`.aimodel`; for `.aimodelc`, instantiate the shared adapter with the exact three
tensor specs above. In `__call__`, return copied NumPy arrays from
`compiled.infer({"image": image})`. Close the compiled runtime when the model is
released.

- [ ] **Step 4: Run detection and loader tests**

Run:

```bash
python -m pytest -q \
  tests/test_coreai_segmentation_model.py \
  tests/test_detection_backend_selection.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit detection support**

```bash
git add lada/models/yolo/yolo11_coreai_segmentation_model.py tests/test_coreai_segmentation_model.py
git commit -m "feat: run compiled Core AI detection"
```

---

### Task 5: Compiled Core AI ROI enhancement

**Files:**
- Modify: `lada/restorationpipeline/coreai_roi_enhancer.py`
- Modify: `tests/test_srvgg_coreai.py`

**Interfaces:**
- Consumes: `CompiledCoreAIRuntime` from Task 1.
- Preserves: `CoreAIROIEnhancer.enhance(...)` output behavior.

- [ ] **Step 1: Write a failing compiled-enhancer transport test**

```python
def test_compiled_enhancer_uses_swift_tensor_contract(tmp_path):
    model_path = tmp_path / "enhancer.h17s.aimodelc"
    model_path.mkdir()
    with mock.patch(
        "lada.restorationpipeline.coreai_roi_enhancer.CompiledCoreAIRuntime"
    ) as compiled:
        runtime = CoreAIEnhancerRuntime(model_path, imgsz=256, scale=4)
        runtime._ensure_loaded()
    compiled.assert_called_once_with(
        model_path,
        inputs=(TensorSpec("image", (1, 3, 256, 256)),),
        outputs=(TensorSpec("enhanced", (1, 3, 1024, 1024)),),
    )
```

- [ ] **Step 2: Run ROI tests and verify RED**

Run: `python -m pytest -q tests/test_srvgg_coreai.py`

Expected: FAIL because compiled transport is absent.

- [ ] **Step 3: Add compiled ROI selection**

Keep `.aimodel` on the existing asyncio Core AI path. For `.aimodelc`, create
the shared adapter with the exact `image` and `enhanced` specs above and return
`compiled.infer({"image": image})["enhanced"]`. Delegate `close()` to whichever
backend was created. Preserve resizing, RGB/BGR conversion, FP16 normalization,
and output validation.

- [ ] **Step 4: Run ROI and frame-restorer tests**

Run:

```bash
python -m pytest -q \
  tests/test_srvgg_coreai.py \
  tests/test_coreml_roi_enhancer.py \
  tests/test_frame_restorer_roi_enhancer.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit ROI support**

```bash
git add lada/restorationpipeline/coreai_roi_enhancer.py tests/test_srvgg_coreai.py
git commit -m "feat: run compiled Core AI ROI enhancement"
```

---

### Task 6: Resolve and package exactly six `h17s` assets

**Files:**
- Modify: `lada/__init__.py`
- Modify: `packaging/macOS/standalone/MiohApp.swift`
- Modify: `packaging/macOS/standalone/build_app.sh`
- Create: `tests/test_coreai_model_resolution.py`
- Modify: `tests/test_basicvsrpp_coreai_restorer.py`
- Modify: `tests/test_coreai_segmentation_model.py`
- Modify: `tests/test_srvgg_coreai.py`
- Modify: `tests/test_standalone_app_options.py`

**Interfaces:**
- Produces: `_coreai_model_path(filename: str) -> str`
- Produces environment: `LADA_COREAI_ARCHITECTURE=h17s`
- Produces exactly six `<basename>.h17s.aimodelc` app resources.

- [ ] **Step 1: Write failing resolution and packaging tests**

Test compiled preference and source fallback by temporarily replacing
`MODEL_WEIGHTS_DIR` and `LADA_COREAI_ARCHITECTURE`, then rebuilding the well-known
model tables. Add build-script assertions:

```python
def test_coreai_model_path_prefers_selected_specialization(tmp_path, monkeypatch):
    source = tmp_path / "basicvsrpp-v1.2-t18-fp16.aimodel"
    compiled = tmp_path / "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc"
    source.mkdir()
    compiled.mkdir()
    monkeypatch.setattr(lada, "MODEL_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("LADA_COREAI_ARCHITECTURE", "h17s")
    assert lada._coreai_model_path(source.name) == str(compiled)


def test_coreai_model_path_falls_back_to_source(tmp_path, monkeypatch):
    source = tmp_path / "basicvsrpp-v1.2-t18-fp16.aimodel"
    source.mkdir()
    monkeypatch.setattr(lada, "MODEL_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setenv("LADA_COREAI_ARCHITECTURE", "h17s")
    assert lada._coreai_model_path(source.name) == str(source)


def test_build_targets_only_m5_pro_coreai_specialization(self):
    script = BUILD_SCRIPT.read_text()
    self.assertIn('COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"', script)
    self.assertIn('--architecture "$COREAI_ARCHITECTURE"', script)
    self.assertIn('! -name "*.$COREAI_ARCHITECTURE.aimodelc"', script)
    self.assertNotIn('for model in "$COMPILED_MODELS"/*.aimodelc', script)
    for source in EXPECTED_SIX_COREAI_SOURCES:
        self.assertIn(source, script)


def test_app_exports_m5_pro_coreai_architecture(self):
    source = APP_SOURCE.read_text()
    self.assertIn('result["LADA_COREAI_ARCHITECTURE"] = "h17s"', source)
    self.assertIn('result.removeValue(forKey: "LADA_COREAI_ARCHITECTURE")', source)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_coreai_model_resolution.py \
  tests/test_standalone_app_options.py \
  tests/test_basicvsrpp_coreai_restorer.py \
  tests/test_coreai_segmentation_model.py \
  tests/test_srvgg_coreai.py
```

Expected: FAIL because resolution still points to `.aimodel` and the build compiles all architectures for T90 only.

- [ ] **Step 3: Implement compiled model resolution**

Add and use one helper for all six model entries:

```python
def _coreai_model_path(filename: str) -> str:
    source = os.path.join(MODEL_WEIGHTS_DIR, filename)
    architecture = os.environ.get("LADA_COREAI_ARCHITECTURE")
    if architecture:
        stem = os.path.splitext(filename)[0]
        compiled = os.path.join(
            MODEL_WEIGHTS_DIR, f"{stem}.{architecture}.aimodelc"
        )
        if os.path.isdir(compiled):
            return compiled
    return source
```

Use it for the three restoration entries, `v4-fast-coreai`, and both Core AI
enhancer entries. Export/remove the architecture variable in the same Swift
branch as the existing runner environment.

- [ ] **Step 4: Implement dedicated packaging**

Define these six source names in `COREAI_MODEL_ASSETS`, remove all seven source
`.aimodel` names from `MODEL_ASSETS`, delete cached nonmatching `.aimodelc`
directories, and compile each source with:

```zsh
xcrun coreai-build compile "$source_model" \
  --output "$COMPILED_MODELS" \
  --platform macOS \
  --min-deployment-version 27.0 \
  --preferred-compute gpu \
  --architecture "$COREAI_ARCHITECTURE"
```

Inspect each output as JSON and fail unless `supportedArchitectures` contains
`h17s`; for the default architecture also require `supportedChips` to contain
`M5 Pro`. Copy only the six expected output directories.

- [ ] **Step 5: Run focused tests and shell syntax verification**

Run:

```bash
python -m pytest -q \
  tests/test_coreai_model_resolution.py \
  tests/test_standalone_app_options.py \
  tests/test_basicvsrpp_coreai_restorer.py \
  tests/test_coreai_segmentation_model.py \
  tests/test_srvgg_coreai.py
zsh -n packaging/macOS/standalone/build_app.sh
```

Expected: all tests pass and shell syntax exits 0.

- [ ] **Step 6: Commit resolution and packaging**

```bash
git add lada/__init__.py packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/build_app.sh \
  tests/test_coreai_model_resolution.py \
  tests/test_basicvsrpp_coreai_restorer.py \
  tests/test_coreai_segmentation_model.py tests/test_srvgg_coreai.py \
  tests/test_standalone_app_options.py
git commit -m "feat: package M5 Pro Core AI specializations"
```

---

### Task 7: Real six-model smoke test and release verification

**Files:**
- Create: `packaging/macOS/standalone/verify_coreai_models.py`
- Create: `tests/test_verify_coreai_models.py`
- Modify: `packaging/macOS/standalone/build_app.sh`

**Interfaces:**
- Produces CLI: `verify_coreai_models.py --resources <Resources directory>`
- Requires six compiled assets and the embedded Swift runner.

- [ ] **Step 1: Write failing verifier tests**

Require an exact asset manifest and deterministic zero/gradient inputs for all
six contracts:

```python
EXPECTED_MODELS = {
    "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-t36-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-t90-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v4_fast-fp16.h17s.aimodelc",
    "RealESRGAN_x4plus-256-fp16.h17s.aimodelc",
    "realesr-general-x4v3-256-fp16.h17s.aimodelc",
}


def test_verifier_requires_exact_specialization_set(tmp_path):
    extras = {"unexpected.h17g.aimodelc"}
    with pytest.raises(RuntimeError, match="unexpected Core AI assets"):
        verify_asset_set(tmp_path, EXPECTED_MODELS, extras)
```

- [ ] **Step 2: Run verifier tests and verify RED**

Run: `python -m pytest -q tests/test_verify_coreai_models.py`

Expected: collection fails because the verifier does not exist.

- [ ] **Step 3: Implement the real verifier**

The verifier imports the embedded Lada runtime, resolves every well-known model
name, asserts the `.h17s.aimodelc` suffix, and runs one inference:

- restoration: one FP16 `[1,T,3,256,256]` gradient for T=18,36,90;
- detection: one FP16 `[1,3,640,640]` zero image, validating both output shapes;
- enhancers: one FP16 `[1,3,256,256]` gradient, validating `[1,3,1024,1024]`.

Every output must have the expected shape, FP16 dtype, and finite values. Close
all runtimes in `finally` blocks. The script exits nonzero on missing, extra,
source, or wrong-architecture Core AI assets.

- [ ] **Step 4: Add verifier to the standalone build after installation**

Invoke the embedded Python with `PYTHONHOME`, `PYTHONPATH`, model directory,
runner path, and `LADA_COREAI_ARCHITECTURE=h17s`. Run it after signing is not
required; run before DMG creation so failure prevents packaging.

- [ ] **Step 5: Run unit tests, then perform a clean full build**

Run:

```bash
python -m pytest -q
rm -rf build/macos-standalone/compiled-models
packaging/macOS/standalone/build_app.sh
```

Expected: the full suite passes, all six `h17s` compilations complete, and the six-model smoke script exits 0.

- [ ] **Step 6: Verify final artifact contents and signatures**

Run:

```bash
find build/macos-standalone/compiled-models -maxdepth 1 \
  -type d -name '*.aimodelc' -print | sort
find build/macos-standalone/dmg-root/mioh.app/Contents/Resources/models \
  -maxdepth 1 -type d \( -name '*.aimodelc' -o -name '*.aimodel' \) -print | sort
codesign --verify --deep --strict --verbose=2 \
  build/macos-standalone/dmg-root/mioh.app
hdiutil verify build/macos-standalone/mioh-0.11.0-unsigned.dmg
git diff --check
git status --short --branch
```

Expected: exactly six `*.h17s.aimodelc` paths in cache and app, zero `.aimodel`
source paths in the app, valid signature, valid DMG checksum, no tracked changes
outside the planned commit.

- [ ] **Step 7: Commit the verifier**

```bash
git add packaging/macOS/standalone/verify_coreai_models.py \
  packaging/macOS/standalone/build_app.sh tests/test_verify_coreai_models.py
git commit -m "test: verify packaged M5 Pro Core AI models"
```

---

## Final Review Checklist

- [ ] Re-read `docs/superpowers/specs/2026-07-14-mioh-m5-pro-coreai-specializations-design.md` and map every success criterion to Tasks 1–7.
- [ ] Run `python -m pytest -q` and record exact pass/skip counts.
- [ ] Compile both Swift executables with `-warnings-as-errors`.
- [ ] Confirm each of the six runtime selections resolves to `.h17s.aimodelc`.
- [ ] Confirm no other Core AI architecture or source asset is packaged.
- [ ] Run one real inference for all six models on this M5 Pro.
- [ ] Confirm MPS deformable-convolution smoke, code signature, and DMG verification pass.
- [ ] Run `git diff --check` and confirm the worktree has no unintended changes.
