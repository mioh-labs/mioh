import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lada.utils import Detection
from experiments.mlx_dcnv2.media_io import write_video_tchw
from experiments.mlx_dcnv2.native_detection import detections_to_mask, detect_video_to_mask_dir


class MLXNativeDetectionBridgeTests(unittest.TestCase):
    def test_detections_to_mask_merges_lada_detection_masks(self):
        mask_a = np.zeros((4, 5, 1), dtype=np.uint8)
        mask_a[1:3, 1:3, 0] = 255
        mask_b = np.zeros((4, 5, 1), dtype=np.uint8)
        mask_b[2:4, 3:5, 0] = 255

        mask = detections_to_mask(
            [
                Detection(cls=4, box=(1, 1, 2, 2), mask=mask_a, confidence=0.9),
                Detection(cls=4, box=(2, 3, 3, 4), mask=mask_b, confidence=0.8),
            ],
            frame_shape=(4, 5),
        )

        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(np.count_nonzero(mask)), 8)

    def test_detect_video_to_mask_dir_uses_native_detector_factory(self):
        frames = np.zeros((3, 3, 8, 8), dtype=np.float32)
        frames[:, 0, 2:6, 2:6] = 1.0
        calls = []

        class FakeDetector:
            def detect_batch(self, images):
                calls.append(len(images))
                output = []
                for image in images:
                    height, width = image.shape[:2]
                    mask = np.zeros((height, width, 1), dtype=np.uint8)
                    mask[1:3, 1:3, 0] = 255
                    output.append([Detection(cls=4, box=(1, 1, 2, 2), mask=mask, confidence=0.9)])
                return output

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.mp4"
            write_video_tchw(frames, input_path, fps=10)

            paths = detect_video_to_mask_dir(
                input_path,
                root / "masks",
                detector_factory=lambda: FakeDetector(),
                batch_size=2,
            )
            first_mask = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)

        self.assertEqual(calls, [2, 1])
        self.assertEqual(len(paths), 3)
        self.assertEqual(int(np.count_nonzero(first_mask)), 4)


if __name__ == "__main__":
    unittest.main()
