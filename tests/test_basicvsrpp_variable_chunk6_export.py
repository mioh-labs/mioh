# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import unittest
from pathlib import Path

from scripts.apple import export_basicvsrpp_variable_chunk6 as exporter

ROOT = Path(__file__).resolve().parents[1]


class BasicVSRPPVariableChunk6ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_script = (
            ROOT / "packaging" / "macOS" / "standalone" / "build_app.sh"
        ).read_text()

    def test_default_keeps_legacy_grid_sample_export(self):
        args = exporter.parse_args(["--output-dir", "/tmp/basicvsrpp-variable"])

        self.assertFalse(args.fuse_flow_warp)
        self.assertFalse(args.native_state_continuations)
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

    def test_native_state_continuations_can_be_enabled_explicitly(self):
        args = exporter.parse_args(
            [
                "--output-dir",
                "/tmp/basicvsrpp-variable",
                "--native-state-continuations",
            ]
        )

        self.assertTrue(args.native_state_continuations)

    def test_standalone_build_rejects_stale_explicit_io_continuations(self):
        for contract in [
            "variable_continuations_use_native_state()",
            "xcrun coreai-build inspect",
            "state_n1 state_n2 flow_previous",
            "--native-state-continuations",
        ]:
            self.assertIn(contract, self.build_script)


if __name__ == "__main__":
    unittest.main()
