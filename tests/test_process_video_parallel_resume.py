import tempfile
import unittest
from pathlib import Path

import process_video_parallel as pvp


class ResumeSegmentProcessingTests(unittest.TestCase):
    def test_detects_pending_work_when_segments_exist_without_processed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            segments_dir = temp_dir / "segments"
            segments_dir.mkdir()
            (segments_dir / "segment_000.mp4").write_bytes(b"x" * 200 * 1024)

            self.assertTrue(pvp.has_pending_segment_work(temp_dir))

    def test_no_pending_work_when_all_segments_have_valid_processed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            segments_dir = temp_dir / "segments"
            processed_dir = temp_dir / "processed"
            segments_dir.mkdir()
            processed_dir.mkdir()
            (segments_dir / "segment_000.mp4").write_bytes(b"x" * 200 * 1024)
            (processed_dir / "processed_000.mp4").write_bytes(b"x" * 200 * 1024)

            self.assertFalse(pvp.has_pending_segment_work(temp_dir))


if __name__ == "__main__":
    unittest.main()
