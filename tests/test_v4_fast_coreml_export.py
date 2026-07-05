# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import unittest

from scripts.apple import export_v4_fast_coreml as export_mod
from scripts.apple import validate_v4_fast_coreml as validate_mod


class V4FastCoreMLExportTests(unittest.TestCase):
    def test_default_export_paths(self):
        args = export_mod.parse_args([])
        self.assertEqual(args.model, Path("model_weights/lada_mosaic_detection_model_v4_fast.pt"))
        self.assertEqual(args.output_dir, Path("build/apple/coreml"))
        self.assertEqual(args.imgsz, 640)

    def test_export_options_are_raw_segmentation_outputs(self):
        opts = export_mod.build_export_options(imgsz=640)
        self.assertEqual(opts["format"], "coreml")
        self.assertEqual(opts["imgsz"], 640)
        self.assertFalse(opts["half"])
        self.assertFalse(opts["nms"])
        self.assertTrue(opts["simplify"])


class V4FastCoreMLValidationTests(unittest.TestCase):
    def test_box_difference_uses_max_abs_delta(self):
        self.assertEqual(
            validate_mod.max_box_abs_diff([1, 2, 3, 4], [1, 5, 2, 4]),
            3,
        )

    def test_relative_area_difference_handles_zero(self):
        self.assertEqual(validate_mod.relative_area_diff(0, 0), 0.0)
        self.assertEqual(validate_mod.relative_area_diff(10, 5), 0.5)


if __name__ == "__main__":
    unittest.main()
