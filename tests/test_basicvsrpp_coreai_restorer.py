# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from lada import ModelFiles
from lada.cli import main as cli_main
import lada.restorationpipeline as restorationpipeline
from lada.restorationpipeline.basicvsrpp_coreai_restorer import (
    CoreAIBasicvsrppMosaicRestorer,
    CoreAIModelRuntime,
)


class RecordingRuntime:
    def __init__(self, step: float = 0.0):
        self.calls = []
        self.step = step

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(inputs.shape))
        value = (len(self.calls) - 1) * self.step / 255.0
        return torch.full_like(inputs, value)


class StreamingRecordingRuntime(RecordingRuntime):
    def __init__(self, step: float = 0.0):
        super().__init__(step)
        self.stream_calls = 0

    def infer_many(self, inputs):
        self.stream_calls += 1
        return [self(item) for item in inputs]


class InputRecordingRuntime:
    def __init__(self):
        self.inputs = []

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs.append(inputs.clone())
        return inputs


class FakeNDArray:
    def __init__(self, data):
        self.data = data


class FakeOutput:
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data


class ConcurrentFakeFunction:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def __call__(self, inputs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"restored": FakeOutput(inputs["frames"].data)}


class FakeRunner:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeBridgeProcess:
    def __init__(self, completed_slots: bytes):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(completed_slots)
        self.stderr = io.BytesIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class CoreAIBasicVSRPPRestorerTests(unittest.TestCase):
    def test_compiled_model_uses_persistent_swift_runner(self):
        inputs = [
            torch.full((1, 18, 3, 2, 2), value, dtype=torch.float16)
            for value in (1, 2)
        ]
        process = FakeBridgeProcess(b"\x00\x00")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model-t18.aimodelc"
            model_path.mkdir()
            with (
                mock.patch.dict(
                    os.environ,
                    {"LADA_COREAI_SWIFT_RUNNER": "/fake/lada-coreai-runner"},
                ),
                mock.patch(
                    "subprocess.Popen",
                    return_value=process,
                ) as popen,
            ):
                runtime = CoreAIModelRuntime(model_path, frame_count=18)
                self.assertEqual(getattr(runtime, "_backend", None), "swift")
                self.assertEqual(
                    getattr(runtime, "_transport", None), "shared-memory"
                )
                self.assertEqual(runtime.max_inflight, 1)
                outputs = runtime.infer_many(inputs)
                shared_memory_path = runtime._shared_memory_path
                runtime.close()

        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(
            command[:3],
            [
                "/fake/lada-coreai-runner",
                str(model_path.resolve()),
                "18",
            ],
        )
        self.assertEqual(command[3], str(shared_memory_path))
        self.assertEqual(command[4], "1")
        self.assertEqual(
            [output.tolist() for output in outputs],
            [input_tensor.tolist() for input_tensor in inputs],
        )
        request = process.stdin.getvalue()
        self.assertEqual(request, b"\x00\x00\xff")
        self.assertFalse(shared_memory_path.exists())

    def test_runtime_close_releases_asyncio_runner_and_loaded_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = CoreAIModelRuntime(Path(temp_dir))
            runner = FakeRunner()
            runtime._runner = runner
            runtime._model = object()
            runtime._function = object()

            runtime.close()

        self.assertTrue(runner.closed)
        self.assertIsNone(runtime._runner)
        self.assertIsNone(runtime._model)
        self.assertIsNone(runtime._function)

    def test_runtime_keeps_two_ordered_coreai_calls_in_flight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = CoreAIModelRuntime(Path(temp_dir))
            function = ConcurrentFakeFunction()
            runtime._runner = asyncio.Runner()
            runtime._function = function
            coreai_runtime = types.ModuleType("coreai.runtime")
            coreai_runtime.NDArray = FakeNDArray

            inputs = [
                torch.full((1, 18, 3, 2, 2), value, dtype=torch.float16)
                for value in range(3)
            ]
            with (
                mock.patch.dict(sys.modules, {"coreai.runtime": coreai_runtime}),
                mock.patch(
                    "psutil.virtual_memory",
                    return_value=types.SimpleNamespace(total=100, available=80),
                ),
            ):
                outputs = runtime.infer_many(inputs)
            runtime._runner.close()

        self.assertEqual(function.max_active, 2)
        self.assertEqual([int(output[0, 0, 0, 0, 0]) for output in outputs], [0, 1, 2])

    def test_t90_runtime_limits_inflight_calls_to_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = CoreAIModelRuntime(Path(temp_dir), frame_count=90)

        self.assertEqual(runtime.max_inflight, 1)

    def test_system_memory_pressure_limits_inflight_calls_to_one(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = CoreAIModelRuntime(Path(temp_dir), frame_count=18)
            function = ConcurrentFakeFunction()
            runtime._runner = asyncio.Runner()
            runtime._function = function
            coreai_runtime = types.ModuleType("coreai.runtime")
            coreai_runtime.NDArray = FakeNDArray
            inputs = [
                torch.zeros((1, 18, 3, 2, 2), dtype=torch.float16)
                for _ in range(2)
            ]
            with (
                mock.patch.dict(sys.modules, {"coreai.runtime": coreai_runtime}),
                mock.patch(
                    "psutil.virtual_memory",
                    return_value=types.SimpleNamespace(total=100, available=20),
                ),
            ):
                runtime.infer_many(inputs)
            runtime._runner.close()

        self.assertEqual(function.max_active, 1)

    def test_short_clip_is_padded_to_t18_and_trimmed(self):
        runtime = RecordingRuntime()
        restorer = CoreAIBasicvsrppMosaicRestorer(
            Path("unused.aimodel"),
            runtime=runtime,
        )
        video = [torch.zeros((4, 4, 3), dtype=torch.uint8) for _ in range(3)]

        restored = restorer.restore(video)

        self.assertEqual(runtime.calls, [(1, 18, 3, 4, 4)])
        self.assertEqual(len(restored), 3)
        self.assertTrue(all(frame.shape == (4, 4, 3) for frame in restored))

    def test_long_clip_uses_t18_chunks_with_two_frame_crossfade(self):
        runtime = RecordingRuntime(step=90.0)
        restorer = CoreAIBasicvsrppMosaicRestorer(
            Path("unused.aimodel"),
            runtime=runtime,
        )
        video = [torch.zeros((2, 2, 3), dtype=torch.uint8) for _ in range(20)]

        restored = restorer.restore(video, max_frames=-1)

        self.assertEqual(
            runtime.calls,
            [
                (1, 18, 3, 2, 2),
                (1, 18, 3, 2, 2),
            ],
        )
        values = [int(frame[0, 0, 0]) for frame in restored]
        self.assertEqual(values[:16], [0] * 16)
        self.assertEqual(values[16:], [30, 60, 90, 90])

    def test_long_clip_uses_full_window_from_end_instead_of_padding_tail(self):
        runtime = InputRecordingRuntime()
        restorer = CoreAIBasicvsrppMosaicRestorer(
            Path("unused.aimodel"),
            runtime=runtime,
        )
        video = [
            torch.full((2, 2, 3), value, dtype=torch.uint8)
            for value in range(20)
        ]

        restored = restorer.restore(video)

        second_window = (
            runtime.inputs[1][0, :, 0, 0, 0].mul(255).round().to(torch.uint8)
        )
        self.assertEqual(second_window.tolist(), list(range(2, 20)))
        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in restored],
            list(range(20)),
        )

    def test_long_clip_submits_all_chunks_through_streaming_runtime(self):
        runtime = StreamingRecordingRuntime(step=90.0)
        restorer = CoreAIBasicvsrppMosaicRestorer(
            Path("unused.aimodel"),
            runtime=runtime,
        )
        video = [torch.zeros((2, 2, 3), dtype=torch.uint8) for _ in range(36)]

        restored = restorer.restore(video)

        self.assertEqual(runtime.stream_calls, 1)
        self.assertEqual(len(runtime.calls), 3)
        self.assertEqual(len(restored), 36)

    def test_t36_model_pads_and_processes_fixed_36_frame_chunks(self):
        runtime = RecordingRuntime(step=90.0)
        restorer = CoreAIBasicvsrppMosaicRestorer(
            Path("unused-t36.aimodel"),
            frame_count=36,
            runtime=runtime,
        )
        video = [torch.zeros((2, 2, 3), dtype=torch.uint8) for _ in range(40)]

        restored = restorer.restore(video)

        self.assertEqual(
            runtime.calls,
            [
                (1, 36, 3, 2, 2),
                (1, 36, 3, 2, 2),
            ],
        )
        self.assertEqual(len(restored), 40)

    def test_t90_model_pads_and_processes_fixed_90_frame_chunks(self):
        runtime = RecordingRuntime(step=90.0)
        restorer = CoreAIBasicvsrppMosaicRestorer(
            Path("unused-t90.aimodel"),
            frame_count=90,
            runtime=runtime,
        )
        video = [torch.zeros((2, 2, 3), dtype=torch.uint8) for _ in range(92)]

        restored = restorer.restore(video)

        self.assertEqual(
            runtime.calls,
            [
                (1, 90, 3, 2, 2),
                (1, 90, 3, 2, 2),
            ],
        )
        self.assertEqual(len(restored), 92)

    def test_coreai_model_is_registered_as_well_known_restoration_model(self):
        models = {
            model.name: model.path
            for model in ModelFiles._WELL_KNOWN_RESTORATION_MODELS
        }

        self.assertIn("basicvsrpp-v1.2-coreai", models)
        self.assertTrue(models["basicvsrpp-v1.2-coreai"].endswith(".aimodel"))
        self.assertIn("basicvsrpp-v1.2-coreai-t36", models)
        self.assertTrue(models["basicvsrpp-v1.2-coreai-t36"].endswith("t36-fp16.aimodel"))
        self.assertIn("basicvsrpp-v1.2-coreai-t90", models)
        self.assertTrue(models["basicvsrpp-v1.2-coreai-t90"].endswith("t90-fp16.aimodel"))

    def test_restoration_loader_selects_t36_contract_from_model_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "basicvsrpp-t36.aimodel"
            model_path.mkdir()

            restorer, _pad_mode = restorationpipeline.load_restoration_model(
                torch.device("mps"),
                "basicvsrpp-v1.2-coreai-t36",
                str(model_path),
                None,
                fp16=True,
            )

        self.assertEqual(restorer.frame_count, 36)

    def test_restoration_loader_detects_t36_from_custom_aimodel_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "custom-basicvsrpp-t36-fp16.aimodel"
            model_path.mkdir()

            restorer, _pad_mode = restorationpipeline.load_restoration_model(
                torch.device("mps"),
                "basicvsrpp-coreai",
                str(model_path),
                None,
                fp16=True,
            )

        self.assertEqual(restorer.frame_count, 36)

    def test_restoration_loader_selects_t90_contract_from_model_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "basicvsrpp-t90.aimodel"
            model_path.mkdir()

            restorer, _pad_mode = restorationpipeline.load_restoration_model(
                torch.device("mps"),
                "basicvsrpp-v1.2-coreai-t90",
                str(model_path),
                None,
                fp16=True,
            )

        self.assertEqual(restorer.frame_count, 90)

    def test_restoration_loader_detects_t90_from_custom_aimodel_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "custom-basicvsrpp-t90-fp16.aimodel"
            model_path.mkdir()

            restorer, _pad_mode = restorationpipeline.load_restoration_model(
                torch.device("mps"),
                "basicvsrpp-coreai",
                str(model_path),
                None,
                fp16=True,
            )

        self.assertEqual(restorer.frame_count, 90)

    def test_restoration_loader_selects_coreai_for_aimodel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "basicvsrpp.aimodel"
            model_path.mkdir()

            restorer, pad_mode = restorationpipeline.load_restoration_model(
                torch.device("mps"),
                "basicvsrpp-v1.2-coreai",
                str(model_path),
                None,
                fp16=True,
            )

        self.assertIsInstance(restorer, CoreAIBasicvsrppMosaicRestorer)
        self.assertEqual(restorer.device, torch.device("cpu"))
        self.assertEqual(pad_mode, "zero")

    def test_restoration_loader_selects_coreai_for_compiled_aimodel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "basicvsrpp-t90.aimodelc"
            model_path.mkdir()

            restorer, pad_mode = restorationpipeline.load_restoration_model(
                torch.device("mps"),
                "basicvsrpp-v1.2-coreai-t90",
                str(model_path),
                None,
                fp16=True,
            )

        self.assertIsInstance(restorer, CoreAIBasicvsrppMosaicRestorer)
        self.assertEqual(restorer.frame_count, 90)
        self.assertEqual(pad_mode, "zero")

    def test_cli_accepts_aimodel_directory_as_custom_restoration_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "custom.aimodel"
            model_path.mkdir()

            self.assertTrue(cli_main.is_restoration_model_path(str(model_path)))

    def test_cli_accepts_aimodelc_directory_as_custom_restoration_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "custom.aimodelc"
            model_path.mkdir()

            self.assertTrue(cli_main.is_restoration_model_path(str(model_path)))


if __name__ == "__main__":
    unittest.main()
