# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest
from pathlib import Path

from scripts.apple import export_basicvsrpp_variable_chunk6 as exporter


class BasicVSRPPVariableChunk6ExportTests(unittest.TestCase):
    def test_default_keeps_legacy_grid_sample_export(self):
        args = exporter.parse_args(["--output-dir", "/tmp/basicvsrpp-variable"])

        self.assertFalse(args.fuse_flow_warp)
        self.assertEqual(args.output_dir, Path("/tmp/basicvsrpp-variable"))

    def test_fused_flow_warp_can_be_enabled_explicitly(self):
        args = exporter.parse_args(
            [
                "--output-dir",
                "/tmp/basicvsrpp-variable",
                "--fuse-flow-warp",
            ]
        )

        self.assertTrue(args.fuse_flow_warp)


if __name__ == "__main__":
    unittest.main()
