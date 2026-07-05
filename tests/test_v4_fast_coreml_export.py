# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path
import unittest

from scripts.apple import export_v4_fast_coreml as export_mod


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


if __name__ == "__main__":
    unittest.main()
