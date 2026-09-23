import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STANDALONE = ROOT / "packaging" / "macOS" / "standalone"
REMUXER = ROOT / "apps" / "MiohRemote" / "MiohRemote" / "IPadMPEGTSRemuxer.swift"
AUDIO = STANDALONE / "MacHLSAudio.swift"
HARNESS = ROOT / "tests" / "swift" / "MacHLSAudioHarness.swift"
PIPELINE = STANDALONE / "MacHLSRealtimePipeline.swift"
PLAYER = STANDALONE / "RealtimePlayer.swift"


class MacHLSAudioSourceTests(unittest.TestCase):
    def test_restored_outputs_take_audio_from_held_media_not_a_second_player(self):
        pipeline = PIPELINE.read_text(encoding="utf-8")
        player = PLAYER.read_text(encoding="utf-8")
        self.assertIn("MacHLSAudio.decode(", pipeline)
        self.assertIn("avFoundationCapture.audio.pcm(", pipeline)
        self.assertIn("MacHLSAudio.writeMovie(", pipeline)
        self.assertFalse((STANDALONE / "MacHLSUnifiedPlayback.swift").exists())
        for removed in (
            "MacHLSUnifiedPlayback",
            "beginSynchronizedHLSStart",
            "seekHLSClockWhenReady",
            "degradeHLSSourcePlayback",
            "hlsRestoredClockFallbackActive",
        ):
            self.assertNotIn(removed, player)

    @unittest.skipUnless(sys.platform == "darwin", "AVFoundation harness is macOS-only")
    def test_consecutive_outputs_tile_the_source_audio_exactly(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        swiftc = shutil.which("swiftc")
        if not (ffmpeg and ffprobe and swiftc):
            self.skipTest("ffmpeg, ffprobe and swiftc are required")

        with tempfile.TemporaryDirectory(prefix="mioh-hls-audio-") as directory:
            work = Path(directory)
            subprocess.run(
                [
                    ffmpeg, "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001",
                    "-f", "lavfi", "-i", "anoisesrc=color=pink:sample_rate=48000:seed=7",
                    "-t", "9", "-c:v", "libx264", "-g", "30", "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", "-ar", "48000",
                    "-f", "hls", "-hls_time", "2", "-hls_playlist_type", "vod",
                    "-hls_segment_filename", str(work / "seg%03d.ts"),
                    str(work / "index.m3u8"),
                ],
                check=True,
            )
            segments = sorted(str(path) for path in work.glob("seg*.ts"))
            self.assertGreaterEqual(len(segments), 4)

            executable = work / "harness"
            subprocess.run(
                [
                    swiftc, "-O", "-parse-as-library",
                    "-module-cache-path", str(work / "module-cache"),
                    str(REMUXER), str(AUDIO), str(HARNESS), "-o", str(executable),
                ],
                check=True,
            )
            result = json.loads(
                subprocess.run(
                    [str(executable), ffmpeg, str(work), *segments],
                    check=True, capture_output=True, text=True,
                ).stdout
            )

            concat = "concat:" + "|".join(segments)
            reference = np.frombuffer(
                subprocess.run(
                    [ffmpeg, "-v", "error", "-i", concat, "-map", "0:a:0",
                     "-ac", "2", "-ar", "48000", "-f", "s16le", "-"],
                    check=True, capture_output=True,
                ).stdout,
                dtype=np.int16,
            ).reshape(-1, 2)
            joined = np.fromfile(work / "joined.pcm", dtype=np.int16).reshape(-1, 2)

            probe = json.loads(
                subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries",
                     "stream=codec_type,start_time", "-of", "json", concat],
                    check=True, capture_output=True, text=True,
                ).stdout
            )
            starts = {s["codec_type"]: float(s["start_time"]) for s in probe["streams"]}
            # Output sample 0 is the first video frame of the first segment.
            lead = round((starts["video"] - starts["audio"]) * 48_000)
            self.assertGreaterEqual(lead, 0)
            compared = min(len(joined), len(reference) - lead)
            self.assertGreater(compared, 48_000 * 7)
            mismatched = np.flatnonzero(
                np.any(joined[:compared] != reference[lead : lead + compared], axis=1)
            )
            self.assertEqual(
                mismatched.size, 0,
                f"first mismatching frame {mismatched[:1]} of {compared}",
            )

            movie = json.loads(
                subprocess.run(
                    [ffprobe, "-v", "error", "-show_entries",
                     "stream=codec_type,codec_name,start_time,duration",
                     "-of", "json", result["movie"]],
                    check=True, capture_output=True, text=True,
                ).stdout
            )
            streams = {s["codec_type"]: s for s in movie["streams"]}
            self.assertEqual(streams["audio"]["codec_name"], "pcm_s16le")
            self.assertAlmostEqual(float(streams["audio"]["start_time"]), 0, places=3)
            self.assertAlmostEqual(float(streams["audio"]["duration"]), 2.002, places=3)
            self.assertIn("video", streams)


if __name__ == "__main__":
    unittest.main()
