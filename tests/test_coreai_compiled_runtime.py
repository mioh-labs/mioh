import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec


class FakeProcess:
    def __init__(self, completed_slots: bytes, stderr: bytes = b""):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(completed_slots)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15


class CompiledCoreAIRuntimeTests(unittest.TestCase):
    def test_descriptor_assigns_nonoverlapping_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.h17s.aimodelc"
            model.mkdir()
            runtime = CompiledCoreAIRuntime(
                model,
                inputs=(TensorSpec("image", (1, 3, 2, 2)),),
                outputs=(
                    TensorSpec("candidates", (1, 2, 3)),
                    TensorSpec("prototypes", (1, 1, 2, 2)),
                ),
                runner_path="/fake/lada-coreai-runner",
            )

            descriptor = runtime.descriptor

        self.assertEqual([item["offset"] for item in descriptor["inputs"]], [0])
        self.assertEqual(descriptor["outputs"][0]["offset"], 24)
        self.assertEqual(descriptor["outputs"][1]["offset"], 36)
        self.assertEqual(descriptor["slotStride"], 44)
        self.assertEqual(descriptor["slotCount"], 1)
        self.assertEqual(descriptor["function"], "main")

    def test_infer_writes_input_and_reads_multiple_outputs(self):
        process = FakeProcess(b"\x00")
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "detect.h17s.aimodelc"
            model.mkdir()
            runtime = CompiledCoreAIRuntime(
                model,
                inputs=(TensorSpec("image", (1, 3, 2, 2)),),
                outputs=(
                    TensorSpec("a", (1, 2)),
                    TensorSpec("b", (1, 1)),
                ),
                runner_path="/fake/lada-coreai-runner",
                process_factory=lambda *args, **kwargs: process,
            )
            runtime._ensure_started()
            runtime._write_tensor("a", np.array([[7, 8]], dtype=np.float16))
            runtime._write_tensor("b", np.array([[9]], dtype=np.float16))
            shared_path = runtime.shared_path
            descriptor_path = runtime.descriptor_path

            result = runtime.infer(
                {"image": np.arange(12, dtype=np.float16).reshape(1, 3, 2, 2)}
            )
            descriptor_json = json.loads(descriptor_path.read_text())
            runtime.close()

        self.assertEqual(result["a"].tolist(), [[7.0, 8.0]])
        self.assertEqual(result["b"].tolist(), [[9.0]])
        self.assertEqual(process.stdin.getvalue(), b"\x00\xff")
        self.assertEqual(descriptor_json["outputs"][1]["name"], "b")
        self.assertFalse(shared_path.exists())
        self.assertFalse(descriptor_path.exists())

    def test_rejects_wrong_input_contract_before_request(self):
        process = FakeProcess(b"")
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.h17s.aimodelc"
            model.mkdir()
            runtime = CompiledCoreAIRuntime(
                model,
                inputs=(TensorSpec("image", (1, 3, 2, 2)),),
                outputs=(TensorSpec("output", (1, 1)),),
                runner_path="/fake/lada-coreai-runner",
                process_factory=lambda *args, **kwargs: process,
            )
            with self.assertRaisesRegex(ValueError, "FP16"):
                runtime.infer({"image": np.zeros((1, 3, 2, 2), np.float32)})
            runtime.close()

        self.assertEqual(process.stdin.getvalue(), b"\xff")

    def test_runner_failure_includes_stderr_and_cleans_up(self):
        process = FakeProcess(b"\xfe", b"compiled model failed\n")
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.h17s.aimodelc"
            model.mkdir()
            runtime = CompiledCoreAIRuntime(
                model,
                inputs=(TensorSpec("image", (1, 1)),),
                outputs=(TensorSpec("output", (1, 1)),),
                runner_path="/fake/lada-coreai-runner",
                process_factory=lambda *args, **kwargs: process,
            )
            runtime._ensure_started()
            shared_path = runtime.shared_path
            descriptor_path = runtime.descriptor_path
            with self.assertRaisesRegex(RuntimeError, "compiled model failed"):
                runtime.infer({"image": np.zeros((1, 1), np.float16)})
            runtime.close()

        self.assertFalse(shared_path.exists())
        self.assertFalse(descriptor_path.exists())

    def test_rejects_duplicate_tensor_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.h17s.aimodelc"
            model.mkdir()
            with self.assertRaisesRegex(ValueError, "duplicate tensor name"):
                CompiledCoreAIRuntime(
                    model,
                    inputs=(TensorSpec("value", (1,)),),
                    outputs=(TensorSpec("value", (1,)),),
                )


if __name__ == "__main__":
    unittest.main()
