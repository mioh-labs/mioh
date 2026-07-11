# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from lada import ModelFiles
from lada.restorationpipeline.coreai_roi_enhancer import CoreAIROIEnhancer
from scripts.apple import export_realesrgan_coreai as rrdb_exporter
from scripts.apple import export_srvgg_coreai as exporter


class RecordingRuntime:
    def __init__(self):
        self.inputs = []

    def __call__(self, image: np.ndarray) -> np.ndarray:
        self.inputs.append((image.shape, image.dtype, image.copy()))
        return np.zeros((1, 3, 1024, 1024), dtype=np.float16)


class SRVGGCoreAIExportTests(unittest.TestCase):
    def test_defaults_target_fixed_fp16_x4_asset(self):
        args = exporter.parse_args([])

        self.assertEqual(args.model, Path("model_weights/realesr-general-x4v3.pth"))
        self.assertEqual(
            args.output,
            Path("model_weights/realesr-general-x4v3-256-fp16.aimodel"),
        )
        self.assertEqual(args.imgsz, 256)
        self.assertEqual(args.scale, 4)
        self.assertEqual(args.num_conv, 32)

    def test_export_wrapper_keeps_fp16_nchw_contract(self):
        net, _ = exporter.build_srvgg(scale=4, num_conv=1)
        wrapper = exporter.CoreAIImageWrapper(net.half()).eval()
        image = torch.zeros((1, 3, 8, 8), dtype=torch.float16)

        output = wrapper(image)

        self.assertEqual(output.shape, (1, 3, 32, 32))
        self.assertEqual(output.dtype, torch.float16)

    def test_coreai_prelu_matches_torch_prelu(self):
        prelu = torch.nn.PReLU(num_parameters=3)
        with torch.no_grad():
            prelu.weight.copy_(torch.tensor([0.1, 0.2, 0.3]))
        replacement = exporter.CoreAIPReLU(prelu.weight.detach().clone())
        image = torch.tensor(
            [[[[-2.0, 1.0]], [[-3.0, 2.0]], [[-4.0, 3.0]]]],
            dtype=torch.float32,
        )

        torch.testing.assert_close(replacement(image), prelu(image))


class CoreAIROIEnhancerTests(unittest.TestCase):
    def test_registered_name_resolves_to_coreai_asset(self):
        with mock.patch("lada.os.path.exists", return_value=True):
            model = ModelFiles.get_enhancer_model_by_name(
                "realesr-general-x4v3-coreai"
            )

        self.assertIsNotNone(model)
        self.assertTrue(model.path.endswith(".aimodel"))

    def test_enhance_converts_bgr_to_fixed_fp16_nchw(self):
        runtime = RecordingRuntime()
        enhancer = CoreAIROIEnhancer(
            Path("enhancer.aimodel"),
            imgsz=256,
            scale=4,
            runtime=runtime,
        )
        roi = np.zeros((200, 180, 3), dtype=np.uint8)
        roi[:, :, 0] = 255

        output, metadata = enhancer.enhance(roi, outscale=4)

        self.assertIsNone(metadata)
        self.assertEqual(runtime.inputs[0][0], (1, 3, 256, 256))
        self.assertEqual(runtime.inputs[0][1], np.dtype(np.float16))
        self.assertEqual(runtime.inputs[0][2][0, 2, 0, 0], 1.0)
        self.assertEqual(output.shape, (800, 720, 3))
        self.assertEqual(output.dtype, np.uint8)

    def test_factory_selects_coreai_backend_for_aimodel(self):
        from lada.restorationpipeline.frame_restorer import create_realesrgan_enhancer

        with mock.patch(
            "lada.restorationpipeline.coreai_roi_enhancer.CoreAIROIEnhancer"
        ) as coreai_cls:
            create_realesrgan_enhancer("enhancer.aimodel", scale=4)

        coreai_cls.assert_called_once_with("enhancer.aimodel", scale=4)


class RealESRGANCoreAIExportTests(unittest.TestCase):
    def test_defaults_target_fixed_fp16_x4_asset(self):
        args = rrdb_exporter.parse_args([])

        self.assertEqual(args.model, Path("model_weights/RealESRGAN_x4plus.pth"))
        self.assertEqual(
            args.output,
            Path("model_weights/RealESRGAN_x4plus-256-fp16.aimodel"),
        )
        self.assertEqual(args.imgsz, 256)
        self.assertEqual(args.scale, 4)

    def test_x4plus_coreai_name_is_registered(self):
        with mock.patch("lada.os.path.exists", return_value=True):
            model = ModelFiles.get_enhancer_model_by_name("realesrgan-x4-coreai")

        self.assertIsNotNone(model)
        self.assertTrue(
            model.path.endswith("RealESRGAN_x4plus-256-fp16.aimodel")
        )


if __name__ == "__main__":
    unittest.main()
