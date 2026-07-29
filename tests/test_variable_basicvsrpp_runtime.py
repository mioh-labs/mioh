import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lada.coreai.variable_basicvsrpp_runtime import VariableBasicVSRPPRuntime


class FakeProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(b"\x00")
        self.stderr = io.BytesIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15


class VariableBasicVSRPPRuntimeTests(unittest.TestCase):
    def test_infer_uses_shared_mapping_and_returns_owning_output(self):
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temp_dir:
            models = Path(temp_dir) / "models.aimodelc"
            models.mkdir()
            runtime = VariableBasicVSRPPRuntime(
                models,
                runner_path="/fake/lada-basicvsrpp-variable-runner",
                process_factory=lambda *args, **kwargs: process,
            )
            frames = np.zeros((1, 2, 3, 256, 256), dtype=np.float16)
            expected = np.full_like(frames, 0.25)
            runtime._ensure_started(2)
            assert runtime._mapping is not None
            runtime._mapping.seek(runtime._sequence_bytes)
            runtime._mapping.write(memoryview(expected).cast("B"))

            restored = runtime.infer(frames)
            restored[0, 0, 0, 0, 0] = np.float16(1.0)
            runtime._mapping.seek(runtime._sequence_bytes)
            mapped_first_value = np.frombuffer(
                runtime._mapping.read(2),
                dtype=np.float16,
            )[0]
            runtime.close()

        self.assertEqual(float(mapped_first_value), 0.25)
        self.assertEqual(float(restored[0, 1, 0, 0, 0]), 0.25)
        self.assertEqual(process.stdin.getvalue(), b"\x02\x00\xff\xff")


if __name__ == "__main__":
    unittest.main()
