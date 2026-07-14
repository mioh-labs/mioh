import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from lada import ModelFiles
from lada.coreai.compiled_runtime import TensorSpec
from lada.models.yolo.yolo11_coreai_segmentation_model import (
    CoreAISegmentationRuntime,
    Yolo11CoreAISegmentationModel,
)


class RecordingRuntime:
    def __init__(self):
        self.calls = []

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((image.shape, image.dtype))
        return (
            np.zeros((1, 38, 8400), dtype=np.float16),
            np.zeros((1, 32, 160, 160), dtype=np.float16),
        )


class Yolo11CoreAISegmentationModelTests(unittest.TestCase):
    def test_compiled_detection_uses_swift_tensor_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "detect.h17s.aimodelc"
            model_path.mkdir()
            with mock.patch(
                "lada.models.yolo.yolo11_coreai_segmentation_model."
                "CompiledCoreAIRuntime"
            ) as compiled:
                runtime = CoreAISegmentationRuntime(model_path)
                runtime._ensure_loaded()
                runtime.close()

        compiled.assert_called_once_with(
            model_path,
            inputs=(TensorSpec("image", (1, 3, 640, 640)),),
            outputs=(
                TensorSpec("candidates", (1, 38, 8400)),
                TensorSpec("prototypes", (1, 32, 160, 160)),
            ),
        )

    def test_detection_model_accepts_compiled_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "detect.h17s.aimodelc"
            model_path.mkdir()
            model = Yolo11CoreAISegmentationModel(
                model_path,
                torch.device("cpu"),
                runtime=RecordingRuntime(),
            )

        self.assertIsNotNone(model.runtime)

    def test_v4_fast_coreai_is_registered(self):
        models = {
            model.name: model.path
            for model in ModelFiles._WELL_KNOWN_DETECTION_MODELS
        }

        self.assertIn("v4-fast-coreai", models)
        self.assertTrue(models["v4-fast-coreai"].endswith(".aimodel"))

    def test_rejects_non_aimodel_path(self):
        with self.assertRaises(ValueError):
            Yolo11CoreAISegmentationModel("detect.pt", torch.device("cpu"))

    def test_runs_fixed_fp16_inputs_on_cpu_tensors(self):
        model = Yolo11CoreAISegmentationModel(
            Path("detect.aimodel"),
            torch.device("mps"),
            runtime=RecordingRuntime(),
        )

        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(model.dtype, torch.float16)
        self.assertFalse(model.letterbox.auto)

    def test_inference_merges_per_frame_coreai_outputs(self):
        runtime = RecordingRuntime()
        model = Yolo11CoreAISegmentationModel(
            Path("detect.aimodel"),
            torch.device("mps"),
            runtime=runtime,
        )

        preds = model.inference(torch.zeros((3, 3, 640, 640), dtype=torch.float16))

        self.assertEqual(
            runtime.calls,
            [((1, 3, 640, 640), np.dtype(np.float16))] * 3,
        )
        candidates, prototypes = preds[0]
        self.assertEqual(tuple(candidates.shape), (3, 38, 8400))
        self.assertEqual(tuple(prototypes.shape), (3, 32, 160, 160))


if __name__ == "__main__":
    unittest.main()
