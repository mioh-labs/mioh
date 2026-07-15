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


class ExportSwinIRArgsTests(unittest.TestCase):
    def test_defaults(self):
        from scripts.apple import export_swinir_coreml as swinir_mod
        args = swinir_mod.parse_args(["--swinir-repo-dir", "vendor/SwinIR"])
        self.assertEqual(str(args.model), "model_weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
        self.assertEqual(str(args.swinir_repo_dir), "vendor/SwinIR")
        self.assertEqual(str(args.output_dir), "model_weights")
        self.assertEqual(args.output_name, "swinir-real-x4")
        self.assertEqual(args.imgsz, 256)
        self.assertEqual(args.scale, 4)
        self.assertEqual(args.arch, "medium")


class ExportSpandrelArgsTests(unittest.TestCase):
    def test_coreml_defaults(self):
        from scripts.apple import export_spandrel_coreml as mod
        args = mod.parse_args([])
        self.assertEqual(str(args.model), "model_weights/4xNomosWebPhoto_RealPLKSR.safetensors")
        self.assertEqual(str(args.output_dir), "model_weights")
        self.assertEqual(args.imgsz, 256)

    def test_coreai_defaults(self):
        from scripts.apple import export_spandrel_coreai as mod
        args = mod.parse_args([])
        self.assertEqual(str(args.model), "model_weights/4xNomosWebPhoto_RealPLKSR.safetensors")
        self.assertEqual(str(args.output), "model_weights/4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel")
        self.assertEqual(args.imgsz, 256)


class MetadataGuardTests(unittest.TestCase):
    def test_accepts_any_lada_enhancer_metadata(self):
        with mock.patch("coremltools.models.MLModel", return_value=make_fake_mlmodel({
            "lada.enhancer": "mewzoom", "lada.scale": "4", "lada.imgsz": "256",
        })):
            enhancer = CoreMLROIEnhancer("mewzoom.mlpackage")
        self.assertEqual(enhancer.enhancer_name, "mewzoom")
        self.assertEqual(enhancer.scale, 4)


class EnhancerNameResolutionTests(unittest.TestCase):
    def test_well_known_names_resolve_to_weights_paths(self):
        import lada
        with mock.patch("lada.os.path.exists", return_value=True):
            mf = lada.ModelFiles.get_enhancer_model_by_name("mewzoom-x4-coreml")
            self.assertIsNotNone(mf)
            self.assertTrue(mf.path.endswith("MewZoom-V1-4X-Unet_256.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("mewzoom-x4-coreml-512")
            self.assertTrue(mf.path.endswith("MewZoom-V1-4X-Unet_512.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("realesrgan-x4-coreml")
            self.assertTrue(mf.path.endswith("RealESRGAN_x4plus_256.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("realesr-general-x4v3-coreml")
            self.assertTrue(mf.path.endswith("realesr-general-x4v3_256.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("swinir-x4-coreml")
            self.assertTrue(mf.path.endswith("swinir-real-x4_256.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("swinir-real-x4-coreml")
            self.assertTrue(mf.path.endswith("swinir-real-x4_256.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("nomos-webphoto-realplksr-x4")
            self.assertTrue(mf.path.endswith("4xNomosWebPhoto_RealPLKSR.safetensors"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("nomos-webphoto-realplksr-x4-coreml")
            self.assertTrue(mf.path.endswith("4xNomosWebPhoto_RealPLKSR_256.mlpackage"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("nomos-webphoto-realplksr-x4-coreai")
            self.assertTrue(mf.path.endswith("4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("nomos-uni-span-x4")
            self.assertTrue(mf.path.endswith("4xNomosUni_span_multijpg.safetensors"))
            mf = lada.ModelFiles.get_enhancer_model_by_name("nomos-uni-compact-x2")
            self.assertTrue(mf.path.endswith("2xNomosUni_compact_multijpg.safetensors"))

    def test_unknown_name_returns_none(self):
        import lada
        with mock.patch("lada.os.path.exists", return_value=True):
            self.assertIsNone(lada.ModelFiles.get_enhancer_model_by_name("nope"))


class EnhancerFactoryTests(unittest.TestCase):
    def test_mlpackage_path_uses_coreml_backend(self):
        from lada.restorationpipeline.frame_restorer import create_realesrgan_enhancer
        with mock.patch("lada.restorationpipeline.coreml_roi_enhancer.CoreMLROIEnhancer") as coreml_cls:
            create_realesrgan_enhancer("enhancer.mlpackage", scale=4)
        coreml_cls.assert_called_once_with("enhancer.mlpackage")

    def test_mewzoom_coreml_prefers_pre_resize_application(self):
        with mock.patch("coremltools.models.MLModel", return_value=make_fake_mlmodel({
            "lada.enhancer": "mewzoom", "lada.scale": "4", "lada.imgsz": "512",
        })):
            enhancer = CoreMLROIEnhancer("mewzoom.mlpackage")
        self.assertTrue(enhancer.prefer_pre_resize)

    def test_metadata_flag_controls_pre_resize_application(self):
        with mock.patch("coremltools.models.MLModel", return_value=make_fake_mlmodel({
            "lada.enhancer": "swinir", "lada.scale": "4", "lada.imgsz": "256",
            "lada.prefer_pre_resize": "1",
        })):
            enhancer = CoreMLROIEnhancer("swinir.mlpackage")
        self.assertTrue(enhancer.prefer_pre_resize)


if __name__ == "__main__":
    unittest.main()
