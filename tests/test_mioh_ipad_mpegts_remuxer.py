import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "MiohRemote" / "MiohRemote"
REMUXER = APP / "IPadMPEGTSRemuxer.swift"
HARNESS = ROOT / "tests" / "swift" / "IPadMPEGTSRemuxerHarness.swift"


def _swift_compiler_command():
    xcrun = shutil.which("xcrun")
    if xcrun:
        return [xcrun, "swiftc"]
    swiftc = shutil.which("swiftc")
    return [swiftc] if swiftc else None


class IPadMPEGTSRemuxerRuntimeTests(unittest.TestCase):
    def test_h264_mpegts_fixture_remuxes_to_playable_mp4(self):
        if sys.platform != "darwin":
            self.skipTest("the AVFoundation remux probe requires macOS")
        if not REMUXER.exists():
            self.skipTest("IPadMPEGTSRemuxer.swift has not been added yet")

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        compiler_command = _swift_compiler_command()
        if not ffmpeg or not ffprobe:
            self.skipTest("ffmpeg and ffprobe are required for the remux fixture")
        if not compiler_command:
            self.skipTest("a Swift compiler is required for the remux harness")

        with tempfile.TemporaryDirectory(prefix="mioh-mpegts-remux-") as directory:
            working_directory = Path(directory)
            input_path = working_directory / "fixture.ts"
            output_path = working_directory / "remuxed.mp4"
            executable = working_directory / "mpegts-remux-harness"

            generated = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-f",
                    "lavfi",
                    "-i",
                    (
                        "testsrc2=size=854x480:rate=5,"
                        "setsar=1280/1281:max=10000"
                    ),
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:sample_rate=48000",
                    "-t",
                    "1.2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "5",
                    "-bf",
                    "2",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "32k",
                    "-f",
                    "mpegts",
                    str(input_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            self.assertEqual(
                generated.returncode,
                0,
                f"ffmpeg could not generate the MPEG-TS fixture:\n{generated.stdout}",
            )
            self.assertGreater(input_path.stat().st_size, 188)

            built = subprocess.run(
                compiler_command
                + [
                    "-parse-as-library",
                    str(REMUXER),
                    str(HARNESS),
                    "-o",
                    str(executable),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            self.assertEqual(
                built.returncode,
                0,
                f"MPEG-TS remux harness did not compile:\n{built.stdout}",
            )

            completed = subprocess.run(
                [str(executable), str(input_path), str(output_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"MPEG-TS remux harness failed:\n{completed.stdout}",
            )
            self.assertIn("iPad MPEG-TS remux probe passed", completed.stdout)

            probed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_entries",
                    "format=format_name,duration",
                    "-show_entries",
                    (
                        "stream=codec_name,codec_type,has_b_frames,nb_read_frames,"
                        "width,height,sample_aspect_ratio"
                    ),
                    "-of",
                    "json",
                    str(output_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
            self.assertEqual(
                probed.returncode,
                0,
                f"ffprobe could not read the remuxed MP4:\n{probed.stdout}",
            )
            metadata = json.loads(probed.stdout)
            self.assertIn("mp4", metadata["format"]["format_name"].split(","))
            self.assertGreater(float(metadata["format"]["duration"]), 0.8)
            video_streams = [
                stream
                for stream in metadata.get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            self.assertEqual(len(video_streams), 1)
            self.assertEqual(video_streams[0].get("codec_name"), "h264")
            self.assertEqual(video_streams[0].get("width"), 854)
            self.assertEqual(video_streams[0].get("height"), 480)
            self.assertEqual(
                video_streams[0].get("sample_aspect_ratio"),
                "1280:1281",
            )
            self.assertGreater(int(video_streams[0].get("has_b_frames", 0)), 0)
            self.assertGreaterEqual(
                int(video_streams[0].get("nb_read_frames", 0)), 5
            )

    def test_timestamp_reset_hls_segments_are_concatenated_before_decode(self):
        if sys.platform != "darwin":
            self.skipTest("the AVFoundation HLS interval probe requires macOS")
        if not REMUXER.exists():
            self.skipTest("IPadMPEGTSRemuxer.swift has not been added yet")

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        compiler_command = _swift_compiler_command()
        if not ffmpeg or not ffprobe:
            self.skipTest("ffmpeg and ffprobe are required for the HLS fixtures")
        if not compiler_command:
            self.skipTest("a Swift compiler is required for the HLS harness")

        with tempfile.TemporaryDirectory(prefix="mioh-hls-interval-") as directory:
            working_directory = Path(directory)
            first_input = working_directory / "segment-0001.ts"
            second_input = working_directory / "segment-0002.ts"
            output_path = working_directory / "concatenated.mp4"
            temporary_directory = working_directory / "normalized"
            executable = working_directory / "mpegts-remux-harness"

            # These are separate encoder/muxer invocations, exactly like two
            # HLS resources whose PTS/DTS timelines both restart at zero.
            fixture_filters = (
                "testsrc2=size=854x480:rate=5,setsar=1280/1281:max=10000",
                (
                    "testsrc2=size=854x480:rate=5,"
                    "hue=h=90,setsar=1280/1281:max=10000"
                ),
            )
            for input_path, fixture_filter in zip(
                (first_input, second_input), fixture_filters
            ):
                generated = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-f",
                        "lavfi",
                        "-i",
                        fixture_filter,
                        "-t",
                        "1.4",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-pix_fmt",
                        "yuv420p",
                        "-g",
                        "5",
                        "-bf",
                        "2",
                        "-muxpreload",
                        "0",
                        "-muxdelay",
                        "0",
                        "-f",
                        "mpegts",
                        str(input_path),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                self.assertEqual(
                    generated.returncode,
                    0,
                    f"ffmpeg could not generate {input_path.name}:\n{generated.stdout}",
                )
                self.assertGreater(input_path.stat().st_size, 188)

            first_timestamps = []
            for input_path in (first_input, second_input):
                timestamp_probe = subprocess.run(
                    [
                        ffprobe,
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_packets",
                        "-show_entries",
                        "packet=pts_time,dts_time",
                        "-of",
                        "json",
                        str(input_path),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=15,
                )
                self.assertEqual(
                    timestamp_probe.returncode,
                    0,
                    f"ffprobe could not inspect {input_path.name}:\n{timestamp_probe.stdout}",
                )
                packets = json.loads(timestamp_probe.stdout).get("packets", [])
                self.assertTrue(packets, f"{input_path.name} contains no video packets")
                first_timestamps.append(
                    (
                        float(packets[0]["pts_time"]),
                        float(packets[0]["dts_time"]),
                    )
                )
            self.assertAlmostEqual(first_timestamps[0][0], first_timestamps[1][0], places=3)
            self.assertAlmostEqual(first_timestamps[0][1], first_timestamps[1][1], places=3)

            built = subprocess.run(
                compiler_command
                + [
                    "-parse-as-library",
                    str(REMUXER),
                    str(HARNESS),
                    "-o",
                    str(executable),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            self.assertEqual(
                built.returncode,
                0,
                f"HLS interval harness did not compile:\n{built.stdout}",
            )

            completed = subprocess.run(
                [
                    str(executable),
                    "concatenate",
                    str(first_input),
                    str(second_input),
                    str(output_path),
                    str(temporary_directory),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=45,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"HLS interval concatenation probe failed:\n{completed.stdout}",
            )
            self.assertIn(
                "iPad HLS interval concatenation probe passed", completed.stdout
            )

            probed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_entries",
                    "format=format_name,duration",
                    "-show_entries",
                    "stream=codec_name,codec_type,nb_read_frames,width,height",
                    "-of",
                    "json",
                    str(output_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
            self.assertEqual(
                probed.returncode,
                0,
                f"ffprobe could not read the concatenated MP4:\n{probed.stdout}",
            )
            metadata = json.loads(probed.stdout)
            self.assertIn("mp4", metadata["format"]["format_name"].split(","))
            duration = float(metadata["format"]["duration"])
            self.assertGreater(duration, 2.2)
            # B-frame decode lead is retained by the passthrough remuxer, so
            # each 1.4 s fixture may expose roughly 1.8 s of track time.
            self.assertLess(duration, 4.1)
            video_streams = [
                stream
                for stream in metadata.get("streams", [])
                if stream.get("codec_type") == "video"
            ]
            self.assertEqual(len(video_streams), 1)
            self.assertEqual(video_streams[0].get("codec_name"), "h264")
            self.assertEqual(video_streams[0].get("width"), 854)
            self.assertEqual(video_streams[0].get("height"), 480)
            self.assertGreaterEqual(
                int(video_streams[0].get("nb_read_frames", 0)), 10
            )


if __name__ == "__main__":
    unittest.main()
