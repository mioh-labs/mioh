# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest
from unittest import mock

import numpy as np
import torch

from lada.restorationpipeline.spandrel_roi_enhancer import SpandrelROIEnhancer


class FakeDescriptor:
    scale = 2
    supports_half = True

    def to(self, device=None, dtype=None):
        self.device = torch.device(device)
        self.dtype = dtype
        return self

    def eval(self):
        return self

    def __call__(self, image):
        return torch.nn.functional.interpolate(image, scale_factor=self.scale, mode="nearest")


class SpandrelROIEnhancerTests(unittest.TestCase):
    def make_enhancer(self, **kwargs):
        descriptor = FakeDescriptor()
        with mock.patch("spandrel.ModelLoader.load_from_file", return_value=descriptor), \
             mock.patch("spandrel.ImageModelDescriptor", FakeDescriptor):
            enhancer = SpandrelROIEnhancer("model.safetensors", device="cpu", **kwargs)
        return enhancer, descriptor

    def test_bgr_round_trip_and_requested_output_scale(self):
        enhancer, _ = self.make_enhancer()
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        output, _ = enhancer.enhance(image, outscale=2)
        self.assertEqual(output.shape, (6, 8, 3))
        np.testing.assert_array_equal(output[0, 0], [10, 20, 30])

    def test_cpu_stays_float32_when_fp16_requested(self):
        _, descriptor = self.make_enhancer(fp16=True)
        self.assertEqual(descriptor.dtype, torch.float32)

    def test_tile_path_preserves_shape(self):
        enhancer, _ = self.make_enhancer(tile=2)
        output, _ = enhancer.enhance(np.zeros((5, 7, 3), dtype=np.uint8), outscale=2)
        self.assertEqual(output.shape, (10, 14, 3))

    def test_factory_routes_coreml_package_without_importing_spandrel(self):
        from lada.restorationpipeline.frame_restorer import create_spandrel_enhancer
        with mock.patch("lada.restorationpipeline.coreml_roi_enhancer.CoreMLROIEnhancer") as cls:
            create_spandrel_enhancer("model.mlpackage", scale=4)
        cls.assert_called_once_with("model.mlpackage")

    def test_factory_routes_coreai_asset_with_scale(self):
        from lada.restorationpipeline.frame_restorer import create_spandrel_enhancer
        with mock.patch("lada.restorationpipeline.coreai_roi_enhancer.CoreAIROIEnhancer") as cls:
            create_spandrel_enhancer("model.aimodel", scale=4)
        cls.assert_called_once_with("model.aimodel", scale=4)

    def test_factory_routes_compiled_coreai_asset_with_scale(self):
        from lada.restorationpipeline.frame_restorer import create_spandrel_enhancer
        with mock.patch("lada.restorationpipeline.coreai_roi_enhancer.CoreAIROIEnhancer") as cls:
            create_spandrel_enhancer("model.aimodelc", scale=4)
        cls.assert_called_once_with("model.aimodelc", scale=4)


if __name__ == "__main__":
    unittest.main()
