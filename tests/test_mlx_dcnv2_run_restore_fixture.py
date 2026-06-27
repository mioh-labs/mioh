import argparse
import tempfile
import unittest
from pathlib import Path

from experiments.mlx_dcnv2.run_restore_fixture import build_parser, resolve_mask_paths


class MLXRunRestoreFixtureTests(unittest.TestCase):
    def test_parser_accepts_native_detection_model_instead_of_mask_glob(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "--manifest",
                "bundle.json",
                "--native-detection-model",
                "v4-fast.pt",
                "--video-input",
                "input.mp4",
                "--video-output",
                "out.mp4",
            ]
        )

        self.assertEqual(args.native_detection_model, "v4-fast.pt")
        self.assertIsNone(args.mask_glob)
        self.assertEqual(args.expansion_ratio, 0.0)
        self.assertEqual(args.align_multiple, 32)
        self.assertIsNone(args.max_restore_roi_area)
        self.assertFalse(args.print_window_timing)

    def test_resolve_mask_paths_uses_native_detection_callback(self):
        generated = [Path("mask_0000.png")]
        calls = []

        def fake_detect(*args, **kwargs):
            calls.append((args, kwargs))
            return generated

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                mask_glob=None,
                native_detection_model="v4-fast.pt",
                video_input="input.mp4",
                generated_mask_dir=str(Path(tmp) / "masks"),
                native_detection_device="mps",
                native_detection_fp16=True,
                native_detection_conf=0.15,
                native_detection_batch_size=2,
                detect_face_mosaics=False,
            )

            mask_paths = resolve_mask_paths(args, detect_video_to_mask_dir=fake_detect)

        self.assertEqual(mask_paths, generated)
        self.assertEqual(calls[0][0][:2], ("input.mp4", Path(tmp) / "masks"))
        self.assertEqual(calls[0][1]["model_path"], "v4-fast.pt")
        self.assertEqual(calls[0][1]["device"], "mps")
        self.assertTrue(calls[0][1]["fp16"])


if __name__ == "__main__":
    unittest.main()
