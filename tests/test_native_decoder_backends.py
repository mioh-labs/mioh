"""Decode synthetic media through the production readers, with no ML models."""
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "packaging/macOS/standalone/NativePreviewPipeline.swift"
BACKENDS = ("avfoundationAsync", "avfoundationLegacy", "ffmpegSoftware")


@unittest.skipUnless(sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 27,
                     "Requires macOS 27 SDK/runtime")
class NativeDecoderBackendsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = os.environ.get("MIOH_TEST_FFMPEG") or shutil.which("ffmpeg")
        cached = ROOT / "build/macos-standalone/ffmpeg-static/ffmpeg"
        if not cls.ffmpeg and cached.is_file():
            cls.ffmpeg = str(cached)
        if not cls.ffmpeg or not shutil.which("swiftc"):
            raise unittest.SkipTest("Requires Swift and FFmpeg")
        cls.temp = tempfile.TemporaryDirectory(prefix="mioh-reader-tests-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        source = PIPELINE.read_text()
        errors = "private enum NativePreviewError" + source.split(
            "private enum NativePreviewError", 1)[1].split("private enum MiohCoreAIModelLoader", 1)[0]
        decoder = "private struct VideoDescription" + source.split(
            "private struct VideoDescription", 1)[1].split("private enum DetectionMaskProjection", 1)[0]
        harness = ROOT / "tests/swift/NativeDecoderBackendsHarness.swift"
        extracted = cls.directory / "decoder.swift"
        extracted.write_text("import Foundation\nimport AVFoundation\nimport CoreVideo\nimport Darwin\n"
                             + errors + decoder + harness.read_text())
        cls.binary = cls.directory / "decoder"
        result = subprocess.run([
            "swiftc", "-parse-as-library", "-target", f"{platform.machine()}-apple-macosx27.0",
            str(extracted),
            str(ROOT / "packages/MiohRemoteKit/Sources/MiohRemoteKit/MiohHTTPRangeAsset.swift"),
            "-o", str(cls.binary)], text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.cfr = cls.directory / "synthetic-cfr.mp4"
        cls.vfr = cls.directory / "synthetic-vfr.mp4"
        base = [cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc2=size=160x96:rate=30000/1001:duration=2"]
        subprocess.run(base + ["-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p", str(cls.cfr)],
                       check=True, capture_output=True, timeout=30)
        subprocess.run(base + ["-vf", "select=not(eq(mod(n\\,5)\\,2))", "-fps_mode", "vfr",
                              "-c:v", "libx264", "-bf", "2", str(cls.vfr)],
                       check=True, capture_output=True, timeout=30)
        cls.padded = cls.directory / "synthetic-padded.mp4"
        subprocess.run(base + ["-vf", "scale=158:94,setsar=1", "-c:v", "libx264", str(cls.padded)],
                       check=True, capture_output=True, timeout=30)
        cls.rotated = cls.directory / "synthetic-rotated.mp4"
        subprocess.run([cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-display_rotation", "90",
                        "-i", str(cls.cfr), "-c", "copy", str(cls.rotated)],
                       check=True, capture_output=True, timeout=30)
        cls.offset = cls.directory / "synthetic-offset.mp4"
        subprocess.run([cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(cls.cfr),
                        "-c", "copy", "-output_ts_offset", "0.2", str(cls.offset)],
                       check=True, capture_output=True, timeout=30)

    def decode(self, backend, path=None, start=0, end=None, mode="normal", executable=None):
        run = subprocess.run([str(self.binary), backend, str(path or self.cfr), str(start),
                              str(end) if end is not None else "none", mode,
                              executable or self.ffmpeg], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        events = [json.loads(line) for line in run.stdout.splitlines()]
        result = next(event for event in events if event["kind"] == "result")
        return result, events

    def test_cfr_frames_and_timestamps_match(self):
        results = [self.decode(backend)[0] for backend in BACKENDS]
        for result in results:
            self.assertNotIn("error", result)
            self.assertEqual(len(result["pts"]), 60)
            self.assertTrue(all(size == [160, 96] for size in result["sizes"]))
        self.assertEqual(results[0]["pts"], results[1]["pts"])
        for a, b in zip(results[0]["pts"], results[2]["pts"]):
            self.assertLessEqual(abs(a - b), 1)

    def test_vfr_preserves_actual_pts(self):
        results = [self.decode(backend, self.vfr)[0] for backend in BACKENDS]
        for result in results:
            self.assertNotIn("error", result)
            self.assertEqual(len(result["pts"]), 48)
        self.assertEqual(results[0]["pts"], results[1]["pts"])
        for a, b in zip(results[0]["pts"], results[2]["pts"]):
            self.assertLessEqual(abs(a - b), 1)
        deltas = {b - a for a, b in zip(results[2]["pts"], results[2]["pts"][1:])}
        self.assertGreater(max(deltas), 1.9 * min(deltas))

    def test_nonzero_seek_keeps_absolute_timeline(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                result, _ = self.decode(backend, start=500_500_000, end=1_501_500_000)
                self.assertNotIn("error", result)
                self.assertEqual(len(result["pts"]), 30)
                self.assertEqual(result["pts"][0], 500_500_000)
                self.assertLess(result["pts"][-1], 1_501_500_000)

    def test_stop_releases_full_ring_and_subprocess(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                result, events = self.decode(backend, mode="cancel")
                self.assertTrue(result.get("cancelled"), result)
                pid = events[-1]["gauges"].get("ffmpeg_pid")
                if pid:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(pid), 0)

    def test_timer_reports_while_decode_is_blocked_on_full_ring(self):
        result, events = self.decode("ffmpegSoftware", mode="idle")
        self.assertTrue(result.get("cancelled"), result)
        interval = next(event for event in events if event.get("event") == "interval")
        self.assertEqual(interval["gauges"]["ring_buffered_frames"], 3)
        self.assertGreater(interval["active_stage_seconds"]["ring_full_wait"], 4)
        self.assertAlmostEqual(interval["interval_fps"]["decoded"],
                               interval["frame_counts"]["decoded"] / interval["interval_seconds"])
        self.assertIn("process_cpu_user_seconds", interval["memory"])

    def test_ffmpeg_failure_is_not_silent_eof(self):
        result, events = self.decode("ffmpegSoftware", executable="/usr/bin/false")
        self.assertIn("error", result)
        self.assertIn("FFmpeg software decoder exited", result["error"])
        self.assertTrue(any(event.get("event") == "decode_failure" for event in events))

    def test_missing_executable_keeps_paths_private(self):
        result, _ = self.decode("ffmpegSoftware", executable=str(self.directory / "private-name"))
        self.assertIn("error", result)
        self.assertIn("ffmpeg.start", result["error"])
        self.assertNotIn(str(self.directory), result["error"])
        self.assertNotIn("private-name", result["error"])

    def test_ffmpeg_handles_row_padding_and_rotation(self):
        for path, size in ((self.padded, [158, 94]), (self.rotated, [96, 160])):
            with self.subTest(size=size):
                result, _ = self.decode("ffmpegSoftware", path=path)
                self.assertNotIn("error", result)
                self.assertEqual(len(result["pts"]), 60)
                self.assertTrue(all(item == size for item in result["sizes"]), result["sizes"][:2])

    def test_ffmpeg_vfr_seek_does_not_restart_timestamps(self):
        full, _ = self.decode("ffmpegSoftware", self.vfr)
        part, _ = self.decode("ffmpegSoftware", self.vfr, start=511_000_000, end=1_411_000_000)
        self.assertNotIn("error", part)
        self.assertEqual(part["pts"], [pts for pts in full["pts"] if 511_000_000 <= pts < 1_411_000_000])

    def test_ffmpeg_preserves_nonzero_track_start(self):
        full, _ = self.decode("ffmpegSoftware", self.offset)
        self.assertNotIn("error", full)
        self.assertEqual(full["pts"][0], 200_000_000)
        part, _ = self.decode("ffmpegSoftware", self.offset, start=511_000_000, end=1_411_000_000)
        self.assertNotIn("error", part)
        self.assertEqual(part["pts"], [pts for pts in full["pts"] if 511_000_000 <= pts < 1_411_000_000])

    def test_diagnostic_metrics_and_privacy(self):
        _, events = self.decode("avfoundationAsync")
        stop = events[-1]
        self.assertEqual(stop["event"], "stop")
        self.assertEqual(stop["decoder_backend"], "avfoundationAsync")
        self.assertEqual(stop["frame_counts"]["decoded"], 60)
        self.assertGreater(stop["stage_seconds_total"]["decode"], 0)
        self.assertGreater(stop["memory"]["process_footprint_mib"], 0)
        self.assertNotIn(str(self.directory), json.dumps(events))
        self.assertNotIn(self.cfr.name, json.dumps(events))


class NativeDecoderConfigurationTests(unittest.TestCase):
    def test_remote_controls_and_validation_match_local_options(self):
        source = (ROOT / "packaging/macOS/standalone/RemoteControlServer.swift").read_text()
        self.assertIn("['decoderBackend','動画の読み込み方式','select','decoderBackends']", source)
        self.assertIn("['detailedDiagnostics','詳細な診断ログ（5秒間隔）','bool']", source)
        self.assertIn('.contains(value.decoderBackend ?? "avfoundationAsync")', source)
        for backend in BACKENDS:
            self.assertIn(f'["id": "{backend}", "label":', source)

    def test_options_reach_local_export_preview_and_preferences(self):
        source = (ROOT / "packaging/macOS/standalone/MiohApp.swift").read_text()
        for key in ("decoderBackend", "detailedDiagnostics"):
            self.assertEqual(source.count(f"{key}: {key},"), 3)
            self.assertIn(f"case {key}", source)
            self.assertIn(f"var {key}:", source)
        self.assertIn('snapshot.detailedDiagnostics ?? false', source)
        self.assertIn('Text("AVFoundation 非同期（既定）").tag("avfoundationAsync")', source)
        self.assertIn('Text("AVFoundation 同期（比較用）").tag("avfoundationLegacy")', source)
        self.assertIn('Text("FFmpeg ソフトウェア（比較用）").tag("ffmpegSoftware")', source)

    def test_old_configuration_and_disabled_diagnostics_keep_defaults(self):
        source = PIPELINE.read_text()
        self.assertIn('guard let value else { return .avfoundationAsync }', source)
        self.assertIn('(config.detailedDiagnostics ?? false)', source)
        self.assertIn('let detailedDiagnostics: Bool?', source)
        self.assertIn('let decoderBackend: String?', source)
        self.assertIn('diagnostics?.start()', source)

    def test_both_ui_readers_accept_diagnostics(self):
        app = (ROOT / "packaging/macOS/standalone/MiohApp.swift").read_text()
        player = (ROOT / "packaging/macOS/standalone/RealtimePlayer.swift").read_text()
        self.assertIn('case "diagnostic":', app)
        self.assertIn('diagnostic["kind"] as? String == "diagnostic"', player)
        self.assertIn('diagnostic["generation"] as? Int == generation', player)


if __name__ == "__main__":
    unittest.main()
