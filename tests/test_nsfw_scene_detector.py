import unittest

import numpy as np

from lada.datasetcreation.nsfw_scene_detector import (
    Scene,
    box_iou,
    build_probe_windows,
    crop_meets_minimum_size,
    detection_meets_confidence,
    determine_min_scene_frames,
    merge_adjacent_windows,
)


class FakeNsfwFrame:
    def __init__(self, frame_number, object_id, box, frame_shape=(32, 32, 3)):
        self.frame_number = frame_number
        self.object_id = object_id
        self._box_value = box
        self.frame = np.zeros(frame_shape, dtype=np.uint8)
        self.interpolated_box = None
        self.interpolated_mask = None
        self._mask_value = np.zeros((*frame_shape[:2], 1), dtype=np.uint8)
        if box is not None:
            top, left, bottom, right = box
            self._mask_value[top:bottom + 1, left:right + 1] = 255

    @property
    def box(self):
        return self.interpolated_box or self._box_value

    @property
    def mask(self):
        return self.interpolated_mask if self.interpolated_mask is not None else self._mask_value


class FakeVideoMetadata:
    def __init__(self, video_fps):
        self.video_fps = video_fps


class FakeConfidenceFrame:
    def __init__(self, confidence, object_detected=True):
        self.confidence = confidence
        self.object_detected = object_detected


class NsfwSceneDetectorTests(unittest.TestCase):
    def test_box_iou(self):
        self.assertEqual(box_iou((0, 0, 9, 9), (0, 0, 9, 9)), 1.0)
        self.assertEqual(box_iou((0, 0, 9, 9), (20, 20, 29, 29)), 0.0)

    def test_tracker_id_change_continues_when_boxes_overlap(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=8)
        scene.add_frame(FakeNsfwFrame(0, 10, (10, 10, 40, 40)))

        next_frame = FakeNsfwFrame(1, 99, (12, 12, 42, 42))

        self.assertTrue(scene.continues_with(next_frame, min_iou=0.2))

    def test_tracker_id_change_splits_when_boxes_do_not_overlap(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=8)
        scene.add_frame(FakeNsfwFrame(0, 10, (10, 10, 40, 40)))

        next_frame = FakeNsfwFrame(1, 99, (100, 100, 140, 140))

        self.assertFalse(scene.continues_with(next_frame, min_iou=0.2))

    def test_same_tracker_id_continues_despite_box_jump(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=8)
        scene.add_frame(FakeNsfwFrame(0, 10, (10, 10, 40, 40)))

        next_frame = FakeNsfwFrame(1, 10, (100, 100, 140, 140))

        self.assertTrue(scene.continues_with(next_frame, min_iou=0.2))

    def test_scene_bridges_up_to_configured_number_of_missed_frames(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=10)
        scene.add_frame(FakeNsfwFrame(0, 10, (2, 2, 11, 11)))

        within_tolerance = FakeNsfwFrame(4, 10, (8, 8, 17, 17))
        beyond_tolerance = FakeNsfwFrame(5, 10, (8, 8, 17, 17))

        self.assertTrue(scene.continues_with(within_tolerance, min_iou=0.2, max_gap_frames=3))
        self.assertFalse(scene.continues_with(beyond_tolerance, min_iou=0.2, max_gap_frames=3))

    def test_interpolated_gap_keeps_frames_contiguous_and_moves_masks(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=10)
        scene.add_frame(FakeNsfwFrame(0, 10, (2, 2, 11, 11)))
        gap_frames = [
            FakeNsfwFrame(1, None, None),
            FakeNsfwFrame(2, None, None),
        ]
        next_frame = FakeNsfwFrame(3, 10, (8, 8, 17, 17))

        scene.add_interpolated_gap(gap_frames, next_frame)
        scene.add_frame(next_frame)

        self.assertEqual(len(scene), 4)
        self.assertEqual(scene.frame_end, 3)
        self.assertEqual(gap_frames[0].box, (4, 4, 13, 13))
        self.assertEqual(gap_frames[1].box, (6, 6, 15, 15))
        self.assertGreater(np.count_nonzero(gap_frames[0].mask), 0)
        self.assertGreater(np.count_nonzero(gap_frames[1].mask), 0)

    def test_minimum_scene_length_defaults_to_training_window(self):
        metadata = FakeVideoMetadata(video_fps=60)

        self.assertEqual(determine_min_scene_frames(metadata, minimum_frames=24, minimum_seconds=0), 24)
        self.assertEqual(determine_min_scene_frames(metadata, minimum_frames=24, minimum_seconds=0.5), 30)

    def test_scene_start_and_continuation_use_different_confidence_thresholds(self):
        high_confidence = FakeConfidenceFrame(0.6)
        continuation_confidence = FakeConfidenceFrame(0.25)

        self.assertTrue(detection_meets_confidence(high_confidence, 0.6))
        self.assertFalse(detection_meets_confidence(continuation_confidence, 0.6))
        self.assertTrue(detection_meets_confidence(continuation_confidence, 0.25))
        self.assertFalse(detection_meets_confidence(FakeConfidenceFrame(None, False), 0.25))

    def test_probe_windows_cover_each_interval_without_overlap(self):
        self.assertEqual(
            build_probe_windows(frame_count=125, stride_frames=60),
            [(0, 59, 59), (60, 119, 119), (120, 124, 124)],
        )

    def test_adjacent_selected_windows_are_scanned_as_one_tracker_session(self):
        self.assertEqual(
            merge_adjacent_windows([(0, 59), (60, 119), (180, 239)]),
            [(0, 119), (180, 239)],
        )

    def test_probe_crop_accepts_either_dimension_at_minimum_size(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        mask = np.zeros((600, 600, 1), dtype=np.uint8)
        mask[100:451, 100:451] = 255

        accepted, dimensions = crop_meets_minimum_size(
            frame,
            mask,
            (100, 100, 450, 450),
            target_size=256,
            minimum_size=384,
        )

        self.assertTrue(accepted)
        self.assertTrue(dimensions[0] >= 384 or dimensions[1] >= 384)

    def test_probe_crop_rejects_when_both_dimensions_are_small(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        mask = np.zeros((600, 600, 1), dtype=np.uint8)
        mask[100:201, 100:201] = 255

        accepted, dimensions = crop_meets_minimum_size(
            frame,
            mask,
            (100, 100, 200, 200),
            target_size=256,
            minimum_size=384,
        )

        self.assertFalse(accepted)
        self.assertLess(dimensions[0], 384)
        self.assertLess(dimensions[1], 384)

    def test_full_scene_rejects_more_frames_instead_of_silently_dropping_them(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=2)
        scene.add_frame(FakeNsfwFrame(0, 10, (10, 10, 40, 40)))
        scene.add_frame(FakeNsfwFrame(1, 10, (10, 10, 40, 40)))

        with self.assertRaisesRegex(ValueError, "complete scene"):
            scene.add_frame(FakeNsfwFrame(2, 10, (10, 10, 40, 40)))


if __name__ == "__main__":
    unittest.main()
