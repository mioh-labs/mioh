import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
