import tempfile
import unittest
from pathlib import Path

import process_video_parallel as pvp


class OutputPathTests(unittest.TestCase):
    def test_single_file_output_directory_gets_uc_filename(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            input_path = root_path / "sample.mp4"
            output_dir = root_path / "out"
            input_path.write_bytes(b"")
            output_dir.mkdir()

            resolved = pvp.resolve_single_output_path(input_path, output_dir)

            self.assertEqual(resolved, output_dir / "sample-UC.mp4")

    def test_single_file_extensionless_output_is_treated_as_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            input_path = root_path / "sample.mkv"
            output_dir = root_path / "out"
            input_path.write_bytes(b"")

            resolved = pvp.resolve_single_output_path(input_path, output_dir)

            self.assertEqual(resolved, output_dir / "sample-UC.mkv")
            self.assertTrue(output_dir.is_dir())

    def test_single_file_output_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            input_path = root_path / "sample.mp4"
            output_file = root_path / "custom.mp4"
            input_path.write_bytes(b"")

            resolved = pvp.resolve_single_output_path(input_path, output_file)

            self.assertEqual(resolved, output_file)


if __name__ == "__main__":
    unittest.main()
