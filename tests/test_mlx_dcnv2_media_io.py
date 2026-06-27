import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.mlx_dcnv2.media_io import (
    mux_audio_from_source_video,
    read_image_sequence_tchw,
    read_video_tchw,
    write_image_sequence_tchw,
    write_video_tchw,
)


class MLXMediaIOTests(unittest.TestCase):
    def test_image_sequence_roundtrip_preserves_bgr_tchw(self):
        frames = np.zeros((2, 3, 4, 5), dtype=np.float32)
        frames[0, 0] = 1.0
        frames[1, 1] = 0.5

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "frames"
            paths = write_image_sequence_tchw(frames, out_dir, prefix="frame")

            loaded = read_image_sequence_tchw(paths)

        np.testing.assert_allclose(loaded[0, 0], 1.0)
        np.testing.assert_allclose(loaded[0, 1:], 0.0)
        np.testing.assert_allclose(loaded[1, 1], 128 / 255.0, atol=1 / 255.0)

    def test_video_read_write_roundtrip_shape_and_metadata(self):
        frames = np.zeros((3, 3, 16, 16), dtype=np.float32)
        frames[:, 2, 4:12, 4:12] = 1.0

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            write_video_tchw(frames, path, fps=12)

            loaded, metadata = read_video_tchw(path, max_frames=2)

        self.assertEqual(loaded.shape, (2, 3, 16, 16))
        self.assertEqual(metadata["fps"], 12)
        self.assertEqual(metadata["width"], 16)
        self.assertEqual(metadata["height"], 16)

    def test_mux_audio_from_source_video_uses_restored_video_and_source_audio(self):
        calls = []
        replaces = []

        def fake_run(command, check):
            calls.append((command, check))

        def fake_replace(src, dst):
            replaces.append((Path(src), Path(dst)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            restored = root / "restored.mp4"
            source.write_bytes(b"source")
            restored.write_bytes(b"restored")

            output = mux_audio_from_source_video(
                source,
                restored,
                ffmpeg="ffmpeg-test",
                run=fake_run,
                replace=fake_replace,
            )

        self.assertEqual(output, restored)
        command, check = calls[0]
        self.assertTrue(check)
        self.assertEqual(command[:5], ["ffmpeg-test", "-y", "-i", str(restored), "-i"])
        self.assertIn(str(source), command)
        self.assertIn("-map", command)
        self.assertEqual(command[command.index("-map") + 1], "0:v:0")
        self.assertEqual(command[command.index("-map", command.index("-map") + 1) + 1], "1:a?")
        self.assertEqual(replaces[0][1], restored)


if __name__ == "__main__":
    unittest.main()
