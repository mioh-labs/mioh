import unittest

import numpy as np
import torch

from lada.restorationpipeline.frame_restorer import (
    FrameRestorer,
    apply_restore_roi_enhancer,
    apply_restore_effect_upscale,
    apply_restore_detail_boost,
    apply_restore_sharpening,
    apply_restore_smoothing,
    apply_restore_texture_mix,
)
from lada.cli.main import setup_argparser
import process_video_parallel as pvp


class RestoreSharpenTests(unittest.TestCase):
    def test_apply_restore_sharpening_zero_strength_returns_input(self):
        image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)

        sharpened = apply_restore_sharpening(image, strength=0.0)

        self.assertIs(sharpened, image)

    def test_apply_restore_sharpening_positive_strength_changes_edges(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[:, 3:, :] = 128

        sharpened = apply_restore_sharpening(image, strength=1.0)

        self.assertEqual(sharpened.dtype, np.uint8)
        self.assertEqual(sharpened.shape, image.shape)
        self.assertFalse(np.array_equal(sharpened, image))

    def test_apply_restore_detail_boost_zero_strength_returns_input(self):
        image = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)

        boosted = apply_restore_detail_boost(image, strength=0.0)

        self.assertIs(boosted, image)

    def test_apply_restore_detail_boost_positive_strength_changes_local_contrast(self):
        image = np.full((9, 9, 3), 64, dtype=np.uint8)
        image[3:6, 3:6, :] = 96

        boosted = apply_restore_detail_boost(image, strength=0.5)

        self.assertEqual(boosted.dtype, np.uint8)
        self.assertEqual(boosted.shape, image.shape)
        self.assertFalse(np.array_equal(boosted, image))

    def test_apply_restore_texture_mix_zero_strength_returns_restored(self):
        original = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
        restored = np.full((5, 5, 3), 64, dtype=np.uint8)

        mixed = apply_restore_texture_mix(restored, original, strength=0.0)

        self.assertIs(mixed, restored)

    def test_apply_restore_texture_mix_positive_strength_adds_mid_frequency_detail(self):
        original = np.full((9, 9, 3), 64, dtype=np.uint8)
        original[4, :, :] = 96
        restored = np.full((9, 9, 3), 64, dtype=np.uint8)

        mixed = apply_restore_texture_mix(restored, original, strength=0.2)

        self.assertEqual(mixed.dtype, np.uint8)
        self.assertEqual(mixed.shape, restored.shape)
        self.assertFalse(np.array_equal(mixed, restored))

    def test_apply_restore_texture_mix_ignores_unmasked_source_detail(self):
        original = np.full((9, 9, 3), 64, dtype=np.uint8)
        original[0, :, :] = 255
        restored = np.full((9, 9, 3), 64, dtype=np.uint8)
        mask = np.zeros((9, 9, 1), dtype=np.uint8)
        mask[3:6, 3:6, :] = 255

        mixed = apply_restore_texture_mix(restored, original, strength=1.0, mask=mask)

        self.assertTrue(np.array_equal(mixed, restored))

    def test_apply_restore_texture_mix_only_changes_masked_pixels(self):
        original = np.full((9, 9, 3), 64, dtype=np.uint8)
        original[4, 3:6, :] = 96
        restored = np.full((9, 9, 3), 64, dtype=np.uint8)
        mask = np.zeros((9, 9, 1), dtype=np.uint8)
        mask[3:6, 3:6, :] = 255

        mixed = apply_restore_texture_mix(restored, original, strength=1.0, mask=mask)

        self.assertTrue(np.array_equal(mixed[mask.squeeze() == 0], restored[mask.squeeze() == 0]))
        self.assertFalse(np.array_equal(mixed[mask.squeeze() > 0], restored[mask.squeeze() > 0]))

    def test_apply_restore_smoothing_zero_strength_returns_input(self):
        image = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)

        smoothed = apply_restore_smoothing(image, strength=0.0)

        self.assertIs(smoothed, image)

    def test_apply_restore_smoothing_positive_strength_softens_variation(self):
        image = np.zeros((9, 9, 3), dtype=np.uint8)
        image[::2, ::2, :] = 255

        smoothed = apply_restore_smoothing(image, strength=0.5)

        self.assertEqual(smoothed.dtype, np.uint8)
        self.assertEqual(smoothed.shape, image.shape)
        self.assertLess(smoothed.astype(np.float32).var(), image.astype(np.float32).var())

    def test_apply_restore_roi_enhancer_zero_strength_returns_restored(self):
        restored = np.full((5, 5, 3), 64, dtype=np.uint8)

        enhanced = apply_restore_roi_enhancer(restored, enhancer=object(), strength=0.0)

        self.assertIs(enhanced, restored)

    def test_apply_restore_roi_enhancer_only_changes_masked_pixels(self):
        class FakeEnhancer:
            def enhance(self, image_bgr, outscale=2):
                return np.full_like(image_bgr, 192), None

        restored = np.full((5, 5, 3), 64, dtype=np.uint8)
        mask = np.zeros((5, 5, 1), dtype=np.uint8)
        mask[1:4, 1:4, :] = 255

        enhanced = apply_restore_roi_enhancer(restored, enhancer=FakeEnhancer(), strength=1.0, mask=mask)

        self.assertTrue(np.array_equal(enhanced[mask.squeeze() == 0], restored[mask.squeeze() == 0]))
        self.assertFalse(np.array_equal(enhanced[mask.squeeze() > 0], restored[mask.squeeze() > 0]))

    def test_restore_frame_applies_pre_resize_enhancer_before_original_roi_resize(self):
        class FakeEnhancer:
            prefer_pre_resize = True
            uses_torch_device = False

            def __init__(self):
                self.seen_shapes = []

            def enhance(self, image_bgr, outscale=4):
                self.seen_shapes.append(image_bgr.shape)
                return np.full_like(image_bgr, 128), None

        class FakeClip:
            frame_start = 0

            def __init__(self):
                self.frame_end = 0
                self.frames = [torch.full((6, 6, 3), 64, dtype=torch.uint8)]
                self.masks = [torch.ones((6, 6), dtype=torch.uint8)]
                self.boxes = [(0, 0, 7, 7)]
                self.crop_shapes = [(8, 8, 3)]
                self.pad_after_resizes = [(1, 1, 1, 1)]

            def pop(self):
                return (
                    self.frames.pop(0),
                    self.masks.pop(0),
                    self.boxes.pop(0),
                    self.crop_shapes.pop(0),
                    self.pad_after_resizes.pop(0),
                )

        enhancer = FakeEnhancer()
        restorer = FrameRestorer.__new__(FrameRestorer)
        restorer.restore_roi_enhancer = enhancer
        restorer.restore_roi_enhancer_strength = 1.0
        restorer.restore_roi_enhancer_scale = 4
        restorer.restore_texture_mix = 0.0
        restorer.restore_detail_boost = 0.0
        restorer.restore_sharpen_strength = 0.0
        restorer.restore_smooth_strength = 0.0
        restorer.restore_effect_upscale = 1
        restorer.restore_blend_feather = 0.0
        restorer.mosaic_restoration_model = type("FakeModel", (), {"dtype": torch.float32})()

        frame = torch.zeros((8, 8, 3), dtype=torch.uint8)
        restorer._restore_frame(frame, 0, [FakeClip()])

        self.assertEqual(enhancer.seen_shapes, [(4, 4, 3)])

    def test_apply_restore_effect_upscale_one_returns_existing_effects(self):
        original = np.full((9, 9, 3), 64, dtype=np.uint8)
        original[4, 3:6, :] = 96
        restored = np.full((9, 9, 3), 64, dtype=np.uint8)
        mask = np.zeros((9, 9, 1), dtype=np.uint8)
        mask[3:6, 3:6, :] = 255

        enhanced = apply_restore_effect_upscale(
            restored,
            original,
            mask,
            scale=1,
            texture_mix=1.0,
            detail_boost=0.0,
            sharpen_strength=0.0,
        )

        expected = apply_restore_texture_mix(restored, original, strength=1.0, mask=mask)
        self.assertTrue(np.array_equal(enhanced, expected))

    def test_apply_restore_effect_upscale_only_changes_masked_pixels(self):
        original = np.zeros((11, 11, 3), dtype=np.uint8)
        original[:, :5, :] = 255
        restored = np.full((11, 11, 3), 64, dtype=np.uint8)
        mask = np.zeros((11, 11, 1), dtype=np.uint8)
        mask[4:7, 4:7, :] = 255

        enhanced = apply_restore_effect_upscale(
            restored,
            original,
            mask,
            scale=2,
            texture_mix=1.0,
            detail_boost=0.5,
            sharpen_strength=0.8,
            smooth_strength=0.4,
        )

        self.assertTrue(np.array_equal(enhanced[mask.squeeze() == 0], restored[mask.squeeze() == 0]))
        self.assertFalse(np.array_equal(enhanced[mask.squeeze() > 0], restored[mask.squeeze() > 0]))

    def test_lada_cli_accepts_restore_sharpen_strength(self):
        parser = setup_argparser()

        args = parser.parse_args([
            "--input", "in.mp4",
            "--restore-sharpen-strength", "0.35",
            "--restore-detail-boost", "0.15",
            "--restore-blend-feather", "1.0",
            "--restore-texture-mix", "0.08",
            "--restore-smooth-strength", "0.25",
            "--restore-roi-enhancer", "realesrgan",
            "--restore-roi-enhancer-model-path", "RealESRGAN_x2plus.pth",
            "--restore-roi-enhancer-scale", "2",
            "--restore-roi-enhancer-strength", "0.25",
            "--restore-roi-enhancer-tile", "64",
            "--restore-effect-upscale", "2",
            "--mosaic-detection-empty-lookahead", "10",
        ])

        self.assertEqual(args.restore_sharpen_strength, 0.35)
        self.assertEqual(args.restore_detail_boost, 0.15)
        self.assertEqual(args.restore_blend_feather, 1.0)
        self.assertEqual(args.restore_texture_mix, 0.08)
        self.assertEqual(args.restore_smooth_strength, 0.25)
        self.assertEqual(args.restore_roi_enhancer, "realesrgan")
        self.assertEqual(args.restore_roi_enhancer_model_path, "RealESRGAN_x2plus.pth")
        self.assertEqual(args.restore_roi_enhancer_scale, 2)
        self.assertEqual(args.restore_roi_enhancer_strength, 0.25)
        self.assertEqual(args.restore_roi_enhancer_tile, 64)
        self.assertEqual(args.restore_effect_upscale, 2)
        self.assertEqual(args.mosaic_detection_empty_lookahead, 10)

    def test_lada_cli_accepts_swinir_roi_enhancer(self):
        parser = setup_argparser()

        args = parser.parse_args([
            "--input", "in.mp4",
            "--restore-roi-enhancer", "swinir",
            "--restore-roi-enhancer-model-path", "swinir-x4-coreml",
            "--restore-roi-enhancer-scale", "4",
            "--restore-roi-enhancer-strength", "0.25",
        ])

        self.assertEqual(args.restore_roi_enhancer, "swinir")
        self.assertEqual(args.restore_roi_enhancer_model_path, "swinir-x4-coreml")
        self.assertEqual(args.restore_roi_enhancer_scale, 4)
        self.assertEqual(args.restore_roi_enhancer_strength, 0.25)

    def test_parallel_command_passes_restore_sharpen_strength(self):
        config = pvp.WorkerRuntimeConfig(
            device="mps",
            fp16=False,
            mps_memory_fraction=None,
            log_mps_memory=False,
            encoding_preset=None,
            encoder=None,
            encoder_options=None,
            optimal_encoder_options=None,
            mp4_fast_start=False,
            mosaic_restoration_model="basicvsrpp-v1.2",
            max_clip_length=180,
            mosaic_detection_model="v4-fast",
            detect_face_mosaics=False,
            lada_temp_dir=None,
            overwrite=False,
            mosaic_detection_empty_lookahead=10,
            restore_sharpen_strength=0.35,
            restore_detail_boost=0.15,
            restore_blend_feather=1.0,
            restore_texture_mix=0.08,
            restore_smooth_strength=0.25,
            restore_roi_enhancer="realesrgan",
            restore_roi_enhancer_model_path="RealESRGAN_x2plus.pth",
            restore_roi_enhancer_scale=2,
            restore_roi_enhancer_strength=0.25,
            restore_roi_enhancer_tile=64,
            restore_effect_upscale=2,
        )

        cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))

        self.assertIn("--restore-sharpen-strength", cmd)
        self.assertIn("0.35", cmd)
        self.assertIn("--restore-detail-boost", cmd)
        self.assertIn("0.15", cmd)
        self.assertIn("--restore-blend-feather", cmd)
        self.assertIn("1.0", cmd)
        self.assertIn("--restore-texture-mix", cmd)
        self.assertIn("0.08", cmd)
        self.assertIn("--restore-smooth-strength", cmd)
        self.assertIn("0.25", cmd)
        self.assertIn("--restore-roi-enhancer", cmd)
        self.assertIn("realesrgan", cmd)
        self.assertIn("--restore-roi-enhancer-model-path", cmd)
        self.assertIn("RealESRGAN_x2plus.pth", cmd)
        self.assertIn("--restore-roi-enhancer-scale", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--restore-roi-enhancer-strength", cmd)
        self.assertIn("0.25", cmd)
        self.assertIn("--restore-roi-enhancer-tile", cmd)
        self.assertIn("64", cmd)
        self.assertIn("--restore-effect-upscale", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--mosaic-detection-empty-lookahead", cmd)
        self.assertIn("10", cmd)


if __name__ == "__main__":
    unittest.main()
