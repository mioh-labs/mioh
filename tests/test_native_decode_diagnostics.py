"""Exercise the production diagnostic formatter without models or source media."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "packaging/macOS/standalone/NativePreviewPipeline.swift"
HARNESS = ROOT / "tests/swift/NativeDecodeDiagnosticsHarness.swift"


@unittest.skipUnless(sys.platform == "darwin", "Swift Foundation harness requires macOS")
class NativeDecodeDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        swiftc = shutil.which("swiftc")
        if not swiftc:
            raise unittest.SkipTest("Swift compiler is required")
        cls.source = PIPELINE.read_text(encoding="utf-8")
        # Compile the actual production definitions, in the same Swift file as
        # the harness so that the implementation may remain file-private.
        definitions = cls.source.split("private enum NativePreviewError", 1)[1]
        definitions = "private enum NativePreviewError" + definitions.split(
            "private enum MiohCoreAIModelLoader", 1
        )[0]
        with tempfile.TemporaryDirectory(prefix="mioh-decode-diagnostics-") as temp:
            source = Path(temp) / "diagnostics.swift"
            source.write_text(
                "import Foundation\n" + definitions + HARNESS.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            executable = Path(temp) / "diagnostics"
            build = subprocess.run(
                [swiftc, "-parse-as-library", "-target", "arm64-apple-macosx27.0",
                 str(source), "-o", str(executable)],
                text=True, capture_output=True, timeout=120,
            )
            if build.returncode:
                raise AssertionError(f"diagnostic harness compile failed:\n{build.stderr}")
            output = subprocess.check_output([str(executable)], text=True, timeout=20)
        cls.output = output
        cls.cases = json.loads(output)

    def test_preserves_nested_and_reader_error_codes(self):
        case = self.cases["nested"]
        self.assertEqual([e["code"] for e in case["errors"]], [-11821, -12909, -12911, 256])
        self.assertEqual(case["errors"][1]["source"], "thrown.underlying")
        self.assertEqual(case["errors"][1]["domain"], "NSOSStatusErrorDomain")
        self.assertEqual(case["reader_status"], 3)
        self.assertEqual(case["sidecar_status"], 2)

    def test_records_last_decoded_time_and_count(self):
        case = self.cases["nested"]
        self.assertEqual(case["stage"], "decoded.next")
        self.assertEqual(case["decoded_frames"], 123)
        self.assertEqual(case["last_decoded_pts_ns"], 4_100_000_000)
        self.assertEqual(case["last_decoded_seconds"], 4.1)
        self.assertRegex(case["macos"], r"^\d+\.\d+\.\d+$")

    def test_start_failure_does_not_invent_a_timestamp(self):
        case = self.cases["start"]
        self.assertEqual(case["decoded_frames"], 0)
        self.assertIsNone(case["last_decoded_pts_ns"])
        self.assertIsNone(case["last_decoded_seconds"])

    def test_does_not_disclose_source_metadata(self):
        for secret in ("private-title", "/Volumes/", "example.mp4", "private.invalid", "token=secret"):
            self.assertNotIn(secret, self.output)
        self.assertEqual(self.cases["unsafe_domain"]["errors"][0]["domain"], "[redacted-domain]")

    def test_supports_multiple_underlying_errors(self):
        self.assertEqual([e["code"] for e in self.cases["multiple"]["errors"]],
                         [-11800, -12909, -12911])

    def test_bounds_deep_and_wide_error_graphs(self):
        self.assertTrue(self.cases["deep"]["error_chain_truncated"])
        self.assertEqual(len(self.cases["deep"]["errors"]), 8)
        self.assertTrue(self.cases["wide"]["error_chain_truncated"])
        self.assertEqual(len(self.cases["wide"]["errors"]), 16)

    def test_deduplicates_repeated_errors(self):
        self.assertEqual(len(self.cases["duplicate"]["errors"]), 2)

    def test_keeps_fixed_decoder_invariant_messages(self):
        self.assertEqual(self.cases["native"]["decoder_reason"],
                         "decoded H.264 sample has no image buffer")

    def test_decode_path_keeps_original_errors_until_formatting(self):
        decoder = self.source.split("private final class ContinuousVideoDecoder", 1)[1].split(
            "private enum DetectionMaskProjection", 1
        )[0]
        self.assertNotIn("localizedDescription", decoder)
        self.assertIn("decodedFrames += 1", decoder)
        self.assertIn("lastDecodedPTS = ptsNanoseconds", decoder)
        self.assertIn("sidecarError: compressedReader?.error", decoder)
        self.assertIn('failureStage = "decoded.next"', decoder)
        self.assertNotIn("Swiftネイティブプレビューを開始できませんでした", self.source)


if __name__ == "__main__":
    unittest.main()
