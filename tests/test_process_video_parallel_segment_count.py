import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import process_video_parallel as pvp


class SegmentCountTests(unittest.TestCase):
    def test_argparser_accepts_segment_count(self):
        parser = pvp.build_arg_parser()

        args = parser.parse_args([
            "--input", "in.mp4",
            "--output", "out.mp4",
            "--segment-count", "8",
        ])

        self.assertEqual(args.segment_count, 8)

    def test_resolve_segment_duration_uses_equal_count_when_given(self):
        self.assertEqual(
            pvp.resolve_segment_duration(duration=1200.0, segment_duration=60, segment_count=8),
            150.0,
        )

    def test_split_video_uses_equal_segment_count_for_ffmpeg_segment_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "segments"

            with mock.patch.object(pvp, "get_video_duration", return_value=1200.0), \
                    mock.patch.object(pvp.subprocess, "run") as run_mock:
                run_mock.return_value.stderr = ""

                pvp.split_video(
                    Path("input.mp4"),
                    output_dir,
                    segment_duration=60,
                    segment_count=8,
                    force_split=True,
                )

        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-segment_time") + 1], "150.0")

    def test_pre_fps_keeps_bitrate_and_uses_quality_prioritized_videotoolbox(self):
        options = (
            "-b:v 12.486M -maxrate 12.486M -bufsize 24.972M "
            "-pix_fmt yuv420p "
            "-power_efficient 1 -realtime 0 -frames_before 0 "
            "-frames_after 0 -prio_speed 0"
        )

        def fake_run(cmd, **_kwargs):
            output = Path(cmd[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            if "-f" in cmd and cmd[cmd.index("-f") + 1] == "segment":
                (output.parent / "segment_000.mp4").write_bytes(b"source")
            else:
                output.write_bytes(b"converted")
            return mock.Mock(stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "segments"
            with (
                mock.patch.object(pvp, "get_video_duration", return_value=1200.0),
                mock.patch.object(pvp.subprocess, "run", side_effect=fake_run) as run_mock,
            ):
                pvp.split_video(
                    Path("input.mp4"),
                    output_dir,
                    pre_fps=30,
                    encoder_options=options,
                    force_split=True,
                )

        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-b:v") + 1], "12.486M")
        self.assertEqual(cmd[cmd.index("-maxrate") + 1], "12.486M")
        self.assertEqual(cmd[cmd.index("-bufsize") + 1], "24.972M")
        self.assertEqual(cmd[cmd.index("-power_efficient") + 1], "0")
        self.assertEqual(cmd[cmd.index("-realtime") + 1], "0")
        self.assertEqual(cmd[cmd.index("-prio_speed") + 1], "0")
        self.assertEqual(cmd[cmd.index("-spatial_aq") + 1], "1")
        self.assertNotIn("-pix_fmt", cmd)
        self.assertNotIn("-frames_before", cmd)
        self.assertNotIn("-frames_after", cmd)
        map_values = [cmd[i + 1] for i, item in enumerate(cmd) if item == "-map"]
        self.assertEqual(map_values, ["0:v:0", "0:a?"])

    def test_pre_fps_splits_first_then_converts_segments_with_two_ffmpeg_workers(self):
        options = "-b:v 12.486M -maxrate 12.486M -bufsize 24.972M"

        def fake_run(cmd, **_kwargs):
            output = Path(cmd[-1])
            if "-f" in cmd and cmd[cmd.index("-f") + 1] == "segment":
                output.parent.mkdir(parents=True, exist_ok=True)
                for index in range(4):
                    (output.parent / f"segment_{index:03d}.mp4").write_bytes(b"source")
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"converted")
            return mock.Mock(stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "segments"
            with (
                mock.patch.object(pvp, "get_video_duration", return_value=1200.0),
                mock.patch.object(pvp.subprocess, "run", side_effect=fake_run) as run_mock,
                mock.patch.object(
                    pvp,
                    "ThreadPoolExecutor",
                    wraps=ThreadPoolExecutor,
                ) as executor_mock,
            ):
                segments = pvp.split_video(
                    Path("input.mp4"),
                    output_dir,
                    segment_count=4,
                    pre_fps=30,
                    encoder_options=options,
                    force_split=True,
                )

        self.assertEqual(len(segments), 4)
        executor_mock.assert_called_once_with(max_workers=2)

        commands = [call.args[0] for call in run_mock.call_args_list]
        split_commands = [cmd for cmd in commands if "-f" in cmd and "segment" in cmd]
        conversion_commands = [cmd for cmd in commands if "-r" in cmd]
        self.assertEqual(len(split_commands), 1)
        self.assertEqual(len(conversion_commands), 4)
        self.assertEqual(split_commands[0][split_commands[0].index("-c:v") + 1], "copy")
        for cmd in conversion_commands:
            self.assertEqual(cmd[cmd.index("-b:v") + 1], "12.486M")
            self.assertEqual(cmd[cmd.index("-hwaccel") + 1], "videotoolbox")
            self.assertEqual(
                cmd[cmd.index("-hwaccel_output_format") + 1],
                "videotoolbox_vld",
            )
            self.assertNotIn("-segment_time", cmd)


if __name__ == "__main__":
    unittest.main()
