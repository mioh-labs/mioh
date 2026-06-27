import unittest

import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.roi_restore import (
    bbox_from_masks,
    restore_masked_roi_sequence_with_lada,
    restore_masked_roi_sequence,
    split_ranges_by_max_roi_area,
)


class MLXROIRestoreTests(unittest.TestCase):
    def test_bbox_from_masks_expands_and_aligns_to_multiple(self):
        masks = np.zeros((2, 10, 12), dtype=np.float32)
        masks[:, 3:6, 4:8] = 1

        bbox = bbox_from_masks(masks, expansion_ratio=0.25, align_multiple=4)

        self.assertEqual(bbox, (0, 0, 12, 8))

    def test_bbox_from_masks_defaults_to_no_expansion_and_32_alignment(self):
        masks = np.zeros((2, 80, 96), dtype=np.float32)
        masks[:, 33:38, 35:42] = 1

        bbox = bbox_from_masks(masks)

        self.assertEqual(bbox, (32, 32, 64, 64))

    def test_restore_masked_roi_sequence_composites_only_masked_pixels(self):
        frames = np.zeros((2, 3, 8, 8), dtype=np.float32)
        frames[:, :, :, :] = 0.2
        masks = np.zeros((2, 8, 8), dtype=np.float32)
        masks[:, 2:6, 2:6] = 1.0

        def fake_restore(roi_frames):
            return roi_frames + 0.5

        restored = np.array(
            restore_masked_roi_sequence(
                mx.array(frames),
                mx.array(masks),
                fake_restore,
                expansion_ratio=0.0,
                align_multiple=4,
            )
        )

        np.testing.assert_allclose(restored[:, :, 0:2, :], frames[:, :, 0:2, :])
        np.testing.assert_allclose(restored[:, :, :, 0:2], frames[:, :, :, 0:2])
        np.testing.assert_allclose(restored[:, :, 2:6, 2:6], frames[:, :, 2:6, 2:6] + 0.5)

    def test_restore_masked_roi_sequence_returns_input_when_mask_empty(self):
        frames = np.ones((2, 3, 8, 8), dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.float32)

        restored = restore_masked_roi_sequence(
            mx.array(frames),
            mx.array(masks),
            lambda roi: roi + 1,
        )

        np.testing.assert_allclose(np.array(restored), frames)

    def test_restore_masked_roi_sequence_with_lada_uses_sequence_forward(self):
        frames = np.zeros((2, 3, 8, 8), dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.float32)
        masks[:, 2:6, 2:6] = 1.0
        calls = []

        def fake_sequence(roi_frames, bundle):
            calls.append((roi_frames.shape, bundle))
            return roi_frames + 0.25

        restored = np.array(
            restore_masked_roi_sequence_with_lada(
                mx.array(frames),
                mx.array(masks),
                {"bundle": True},
                sequence_forward=fake_sequence,
                expansion_ratio=0.0,
                align_multiple=4,
            )
        )

        self.assertEqual(calls[0][0], (1, 2, 3, 8, 8))
        np.testing.assert_allclose(restored[:, :, 2:6, 2:6], 0.25)
        np.testing.assert_allclose(restored[:, :, :2, :], 0.0)

    def test_split_ranges_by_max_roi_area_keeps_roi_under_limit(self):
        masks = np.zeros((5, 16, 16), dtype=np.float32)
        masks[0:2, 1:5, 1:5] = 1.0
        masks[2:5, 1:5, 10:14] = 1.0

        ranges = split_ranges_by_max_roi_area(
            masks,
            max_roi_area=32,
            expansion_ratio=0.0,
            align_multiple=1,
        )

        self.assertEqual(ranges, [(0, 2), (2, 5)])

    def test_split_ranges_by_max_roi_area_keeps_empty_window_together(self):
        masks = np.zeros((4, 8, 8), dtype=np.float32)

        ranges = split_ranges_by_max_roi_area(masks, max_roi_area=16)

        self.assertEqual(ranges, [(0, 4)])

    def test_split_ranges_by_max_roi_area_avoids_single_frame_tail(self):
        masks = np.zeros((5, 16, 16), dtype=np.float32)
        masks[0:2, 1:5, 1:5] = 1.0
        masks[2:4, 1:5, 10:14] = 1.0
        masks[4, 10:14, 10:14] = 1.0

        ranges = split_ranges_by_max_roi_area(
            masks,
            max_roi_area=32,
            expansion_ratio=0.0,
            align_multiple=1,
        )

        self.assertEqual(ranges, [(0, 2), (2, 5)])

    def test_split_ranges_by_max_roi_area_avoids_single_frame_middle(self):
        masks = np.zeros((6, 16, 16), dtype=np.float32)
        masks[0:2, 1:5, 1:5] = 1.0
        masks[2, 1:5, 10:14] = 1.0
        masks[3:6, 10:14, 10:14] = 1.0

        ranges = split_ranges_by_max_roi_area(
            masks,
            max_roi_area=32,
            expansion_ratio=0.0,
            align_multiple=1,
        )

        self.assertEqual(ranges, [(0, 3), (3, 6)])


if __name__ == "__main__":
    unittest.main()
