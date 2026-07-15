import unittest

from lada.datasetcreation.nsfw_scene_detector import Scene, box_iou


class FakeNsfwFrame:
    def __init__(self, frame_number, object_id, box):
        self.frame_number = frame_number
        self.object_id = object_id
        self._box_value = box

    @property
    def box(self):
        return self._box_value


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

    def test_full_scene_rejects_more_frames_instead_of_silently_dropping_them(self):
        scene = Scene(video_meta_data=None, id=10, scene_min_length=1, scene_max_length=2)
        scene.add_frame(FakeNsfwFrame(0, 10, (10, 10, 40, 40)))
        scene.add_frame(FakeNsfwFrame(1, 10, (10, 10, 40, 40)))

        with self.assertRaisesRegex(ValueError, "complete scene"):
            scene.add_frame(FakeNsfwFrame(2, 10, (10, 10, 40, 40)))


if __name__ == "__main__":
    unittest.main()
