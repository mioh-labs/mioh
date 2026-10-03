"""Verify measured cadence and source-time preservation with the real Swift code."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "packaging/macOS/standalone/NativePreviewPipeline.swift"
ENCODER = ROOT / "packaging/macOS/standalone/PreviewVideoToolboxEncoder.swift"
HARNESS = ROOT / "tests/swift/SourceFrameRateHarness.swift"


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "macOS Swift required")
class SourceFrameRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="mioh-source-rate-")
        cls.addClassCleanup(cls.workspace.cleanup)
        folder = Path(cls.workspace.name)
        source = PIPELINE.read_text()
        definitions = "private enum SourceFrameRate" + source.split(
            "private enum SourceFrameRate", 1
        )[1].split("private struct DecodedFrame", 1)[0]
        swift = folder / "rate.swift"
        swift.write_text("import AVFoundation\nimport Foundation\n" + definitions + HARNESS.read_text())
        cls.executable = folder / "rate"
        compiled = subprocess.run([
            "xcrun", "swiftc", "-O", "-parse-as-library", "-D", "MIOH_NATIVE_PREVIEW_PIPELINE",
            "-target", "arm64-apple-macosx27.0", "-framework", "AVFoundation",
            "-framework", "Accelerate", "-framework", "CoreVideo", "-framework", "VideoToolbox",
            str(ENCODER), str(swift), "-o", str(cls.executable),
        ], capture_output=True, text=True, timeout=120)
        if compiled.returncode:
            raise AssertionError(compiled.stderr)
        run = subprocess.run([str(cls.executable)], capture_output=True, text=True, timeout=30)
        if run.returncode:
            raise AssertionError(run.stdout + run.stderr)
        cls.cases = json.loads(run.stdout)

    def test_normal_ntsc_cadence_survives_long_gaps(self):
        case = self.cases["gaps"]
        self.assertEqual((case["numerator"], case["denominator"]), (30000, 1001))
        self.assertTrue(case["matches_ntsc"])

    def test_edit_boundary_does_not_raise_cadence(self):
        self.assertEqual(self.cases["short_first_sample"], self.cases["gaps"])

    def test_equivalent_durations_share_a_vote(self):
        self.assertEqual(self.cases["equivalent_timebases"], self.cases["gaps"])

    def test_true_lower_cadence_still_requires_upconversion(self):
        case = self.cases["true_29936"]
        self.assertFalse(case["matches_ntsc"])
        self.assertEqual((case["numerator"], case["denominator"]), (3742, 125))

    def test_uncertain_cadence_retains_average_rate(self):
        for name, expected in [
            ("variable", (20, 1)), ("short_clip", (3742, 125)),
            ("distant_cadence", (24, 1)), ("invalid", (25, 1)),
        ]:
            with self.subTest(name=name):
                case = self.cases[name]
                self.assertEqual((case["numerator"], case["denominator"]), expected)

    def test_writer_preserves_gap_and_segment_boundary(self):
        case = self.cases["writer"]
        self.assertEqual([len(times) for times in case["times"]], [3, 3])
        expected_boundary = (192282 - 90) / 90000
        self.assertAlmostEqual(case["durations"][0], expected_boundary, delta=1 / 30000)
        self.assertAlmostEqual(case["ends"][0], case["starts"][1], places=8)
        for times in case["times"]:
            for actual, expected in zip(times, [0, 1001 / 30000, 2002 / 30000]):
                self.assertAlmostEqual(actual, expected, delta=1 / 30000)
        self.assertAlmostEqual(case["durations"][1], 3003 / 30000, delta=1 / 30000)


if __name__ == "__main__":
    unittest.main()
