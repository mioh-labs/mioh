import tempfile
import unittest
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.media_io import read_image_sequence_tchw, read_video_tchw, write_image_sequence_tchw, write_video_tchw
from experiments.mlx_dcnv2.restore_fixture import (
    restore_image_sequence_with_masks,
    restore_video_with_mask_windows,
    restore_video_with_masks,
)


class MLXRestoreFixtureTests(unittest.TestCase):
    def test_restore_image_sequence_with_masks_writes_composited_frames(self):
        frames = np.zeros((2, 3, 8, 8), dtype=np.float32)
        frames[:, 0] = 0.1
        masks = np.zeros((2, 8, 8), dtype=np.float32)
        masks[:, 2:6, 2:6] = 1.0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_paths = write_image_sequence_tchw(frames, root / "input")
            mask_paths = _write_mask_sequence(masks, root / "masks")
            output_paths = restore_image_sequence_with_masks(
                frame_paths,
                mask_paths,
                root / "out",
                bundle={"fake": True},
                sequence_forward=lambda roi, bundle: roi + 0.5,
                expansion_ratio=0.0,
                align_multiple=4,
            )

            restored = read_image_sequence_tchw(output_paths)

        self.assertEqual(len(output_paths), 2)
        np.testing.assert_allclose(restored[:, :, 2:6, 2:6], frames[:, :, 2:6, 2:6] + 0.5, atol=1 / 255.0)
        np.testing.assert_allclose(restored[:, :, :2, :], frames[:, :, :2, :], atol=1 / 255.0)

    def test_restore_video_with_masks_writes_video(self):
        frames = np.zeros((2, 3, 16, 16), dtype=np.float32)
        frames[:, 1] = 0.2
        masks = np.zeros((2, 16, 16), dtype=np.float32)
        masks[:, 4:12, 4:12] = 1.0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "out.mp4"
            write_video_tchw(frames, input_path, fps=10)
            mask_paths = _write_mask_sequence(masks, root / "masks")

            restore_video_with_masks(
                input_path,
                mask_paths,
                output_path,
                bundle={"fake": True},
                sequence_forward=lambda roi, bundle: roi + 0.25,
                expansion_ratio=0.0,
                align_multiple=4,
            )
            restored, metadata = read_video_tchw(output_path)

        self.assertEqual(restored.shape, (2, 3, 16, 16))
        self.assertEqual(metadata["fps"], 10)

    def test_restore_video_with_mask_windows_writes_each_frame_once(self):
        frames = np.zeros((5, 3, 16, 16), dtype=np.float32)
        for idx in range(5):
            frames[idx, 0] = idx / 10.0
        masks = np.ones((5, 16, 16), dtype=np.float32)
        calls = []
        progress = []

        def fake_sequence(roi, bundle):
            calls.append(tuple(roi.shape))
            return roi + 0.1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "out.mp4"
            write_video_tchw(frames, input_path, fps=10)
            mask_paths = _write_mask_sequence(masks, root / "masks")

            restore_video_with_mask_windows(
                input_path,
                mask_paths,
                output_path,
                bundle={"fake": True},
                sequence_forward=fake_sequence,
                window_size=3,
                overlap=1,
                expansion_ratio=0.0,
                align_multiple=4,
                progress_callback=progress.append,
            )
            restored, metadata = read_video_tchw(output_path)

        self.assertEqual(calls, [(1, 3, 3, 16, 16), (1, 3, 3, 16, 16)])
        self.assertEqual(progress, [2, 5])
        self.assertEqual(restored.shape, (5, 3, 16, 16))
        self.assertEqual(metadata["fps"], 10)

    def test_restore_video_with_mask_windows_splits_large_roi_windows(self):
        frames = np.zeros((5, 3, 16, 16), dtype=np.float32)
        masks = np.zeros((5, 16, 16), dtype=np.float32)
        masks[0:2, 1:5, 1:5] = 1.0
        masks[2:5, 1:5, 10:14] = 1.0
        calls = []

        def fake_sequence(roi, bundle):
            calls.append(tuple(roi.shape))
            return roi + 0.1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            output_path = root / "out.mp4"
            write_video_tchw(frames, input_path, fps=10)
            mask_paths = _write_mask_sequence(masks, root / "masks")

            restore_video_with_mask_windows(
                input_path,
                mask_paths,
                output_path,
                bundle={"fake": True},
                sequence_forward=fake_sequence,
                window_size=5,
                overlap=1,
                expansion_ratio=0.0,
                align_multiple=1,
                max_restore_roi_area=32,
            )

        self.assertEqual(calls, [(1, 2, 3, 4, 4), (1, 3, 3, 4, 4)])

    def test_restore_image_sequence_defaults_to_no_expansion_and_32_alignment(self):
        frames = np.zeros((2, 3, 96, 96), dtype=np.float32)
        masks = np.zeros((2, 96, 96), dtype=np.float32)
        masks[:, 33:38, 35:42] = 1.0
        calls = []

        def fake_sequence(roi, bundle):
            calls.append(tuple(roi.shape))
            return roi + 0.1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_paths = write_image_sequence_tchw(frames, root / "input")
            mask_paths = _write_mask_sequence(masks, root / "masks")

            restore_image_sequence_with_masks(
                frame_paths,
                mask_paths,
                root / "out",
                bundle={"fake": True},
                sequence_forward=fake_sequence,
            )

        self.assertEqual(calls, [(1, 2, 3, 32, 32)])


def _write_mask_sequence(masks: np.ndarray, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True)
    paths = []
    for idx, mask in enumerate(masks):
        path = output_dir / f"mask_{idx:04d}.png"
        cv2.imwrite(str(path), (mask * 255).astype(np.uint8))
        paths.append(path)
    return paths


if __name__ == "__main__":
    unittest.main()
