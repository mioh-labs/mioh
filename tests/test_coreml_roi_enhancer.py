# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from lada.restorationpipeline.coreml_roi_enhancer import CoreMLROIEnhancer
from scripts.apple import export_realesrgan_coreml as export_mod


def make_fake_mlmodel(metadata=None):
    fake = mock.MagicMock()
    fake.user_defined_metadata = metadata if metadata is not None else {
        "lada.enhancer": "realesrgan",
        "lada.scale": "4",
        "lada.imgsz": "256",
    }
    spec = SimpleNamespace(description=SimpleNamespace(
        input=[SimpleNamespace(name="image")],
        output=[SimpleNamespace(name="enhanced")],
    ))
    fake.get_spec.return_value = spec
    fake.predict.return_value = {"enhanced": Image.new("RGB", (1024, 1024), (10, 20, 30))}
    return fake


class CoreMLROIEnhancerTests(unittest.TestCase):
    def make_enhancer(self, metadata=None):
        with mock.patch("coremltools.models.MLModel", return_value=make_fake_mlmodel(metadata)):
            return CoreMLROIEnhancer("enhancer.mlpackage")

    def test_rejects_foreign_mlpackage(self):
        with self.assertRaises(ValueError):
            self.make_enhancer(metadata={})

    def test_composition_does_not_need_torch_device(self):
        enhancer = self.make_enhancer()
        self.assertFalse(enhancer.uses_torch_device)

    def test_enhance_returns_scaled_bgr(self):
        enhancer = self.make_enhancer()
        roi = np.zeros((200, 180, 3), dtype=np.uint8)
        out, _ = enhancer.enhance(roi, outscale=4)
        self.assertEqual(out.shape, (800, 720, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_enhance_native_size_skips_resize(self):
        enhancer = self.make_enhancer()
        roi = np.zeros((256, 256, 3), dtype=np.uint8)
        out, _ = enhancer.enhance(roi, outscale=4)
        self.assertEqual(out.shape, (1024, 1024, 3))


class ExportRealESRGANArgsTests(unittest.TestCase):
    def test_defaults(self):
        args = export_mod.parse_args([])
        self.assertEqual(str(args.model), "model_weights/RealESRGAN_x4plus.pth")
        self.assertEqual(str(args.output_dir), "model_weights")
        self.assertEqual(args.imgsz, 256)
        self.assertEqual(args.scale, 4)


class ExportMewZoomArgsTests(unittest.TestCase):
    def test_defaults(self):
        from scripts.apple import export_mewzoom_coreml as mewzoom_mod
        args = mewzoom_mod.parse_args([])
        self.assertEqual(args.repo, "andrewdalpino/MewZoom-V1-4X-Unet")
        self.assertEqual(str(args.output_dir), "model_weights")
        self.assertEqual(args.imgsz, 256)


class MetadataGuardTests(unittest.TestCase):
    def test_accepts_any_lada_enhancer_metadata(self):
        with mock.patch("coremltools.models.MLModel", return_value=make_fake_mlmodel({
            "lada.enhancer": "mewzoom", "lada.scale": "4", "lada.imgsz": "256",
        })):
            enhancer = CoreMLROIEnhancer("mewzoom.mlpackage")
        self.assertEqual(enhancer.enhancer_name, "mewzoom")
        self.assertEqual(enhancer.scale, 4)


class EnhancerFactoryTests(unittest.TestCase):
    def test_mlpackage_path_uses_coreml_backend(self):
        from lada.restorationpipeline.frame_restorer import create_realesrgan_enhancer
        with mock.patch("lada.restorationpipeline.coreml_roi_enhancer.CoreMLROIEnhancer") as coreml_cls:
            create_realesrgan_enhancer("enhancer.mlpackage", scale=4)
        coreml_cls.assert_called_once_with("enhancer.mlpackage")


if __name__ == "__main__":
    unittest.main()
