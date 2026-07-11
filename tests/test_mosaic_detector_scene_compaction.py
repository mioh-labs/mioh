import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from lada.restorationpipeline.mosaic_detector import Clip, MosaicDetector, Scene


class MosaicDetectorSceneCompactionTests(unittest.TestCase):
    @staticmethod
    def _make_frame(offset: int) -> torch.Tensor:
        values = torch.arange(80 * 96 * 3, dtype=torch.int32)
        return ((values + offset) % 256).to(torch.uint8).reshape(80, 96, 3)

    @staticmethod
    def _make_mask(box) -> torch.Tensor:
        mask = torch.zeros((80, 96, 1), dtype=torch.uint8)
        top, left, bottom, right = box
        mask[top : bottom + 1, left : right + 1] = 255
        return mask

    def test_compacted_scene_produces_identical_clip(self):
        metadata = SimpleNamespace(video_file="input.mp4")
        boxes = [(18, 24, 42, 50), (20, 28, 48, 58)]
        frames = [self._make_frame(0), self._make_frame(17)]
        masks = [self._make_mask(box) for box in boxes]

        legacy_scene = Scene("input.mp4", metadata)
        compacted_scene = Scene("input.mp4", metadata)
        self.assertTrue(hasattr(compacted_scene, "compact_last_frame"))
        for frame_num, (frame, mask, box) in enumerate(zip(frames, masks, boxes)):
            legacy_scene.add_frame(frame_num, frame, mask, box)
            compacted_scene.add_frame(frame_num, frame, mask, box)
            compacted_scene.compact_last_frame(64)

        legacy_clip = Clip(legacy_scene, size=64, pad_mode="reflect", id=1)
        compacted_clip = Clip(compacted_scene, size=64, pad_mode="reflect", id=2)

        self.assertEqual(compacted_clip.boxes, legacy_clip.boxes)
        self.assertEqual(compacted_clip.crop_shapes, legacy_clip.crop_shapes)
        self.assertEqual(compacted_clip.pad_after_resizes, legacy_clip.pad_after_resizes)
        for actual, expected in zip(compacted_clip.frames, legacy_clip.frames):
            self.assertTrue(torch.equal(actual, expected))
        for actual, expected in zip(compacted_clip.masks, legacy_clip.masks):
            self.assertTrue(torch.equal(actual, expected))

    def test_compaction_releases_full_frame_storage_after_mask_merge(self):
        metadata = SimpleNamespace(video_file="input.mp4")
        frame = self._make_frame(0)
        first_box = (18, 24, 36, 44)
        second_box = (26, 36, 48, 58)
        scene = Scene("input.mp4", metadata)
        self.assertTrue(hasattr(scene, "compact_last_frame"))
        scene.add_frame(0, frame, self._make_mask(first_box), first_box)
        scene.merge_mask_box(self._make_mask(second_box), second_box)

        scene.compact_last_frame(64)

        cropped_frame = scene.frames[0]
        cropped_mask = scene.masks[0]
        self.assertLess(cropped_frame.numel(), frame.numel())
        self.assertEqual(
            cropped_frame.untyped_storage().nbytes(),
            cropped_frame.numel() * cropped_frame.element_size(),
        )
        self.assertEqual(
            cropped_mask.untyped_storage().nbytes(),
            cropped_mask.numel() * cropped_mask.element_size(),
        )
        frame.zero_()
        self.assertGreater(int(cropped_frame.max()), 0)

    def test_detector_compacts_after_all_same_frame_masks_are_merged(self):
        metadata = SimpleNamespace(video_file="input.mp4")
        detector = MosaicDetector(
            model=object(),
            video_metadata=metadata,
            frame_detection_queue=None,
            mosaic_clip_queue=None,
            error_handler=lambda marker: None,
            clip_size=64,
        )
        frame = self._make_frame(0)
        boxes = [(18, 24, 36, 44), (26, 36, 48, 58)]
        masks = [self._make_mask(box) for box in boxes]
        results = SimpleNamespace(
            boxes=[object(), object()],
            masks=[object(), object()],
            orig_shape=frame.shape,
            orig_img=frame,
        )
        scenes = []

        with (
            mock.patch(
                "lada.restorationpipeline.mosaic_detector.convert_yolo_box",
                side_effect=boxes,
            ),
            mock.patch(
                "lada.restorationpipeline.mosaic_detector.convert_yolo_mask_tensor",
                side_effect=masks,
            ),
        ):
            detector._create_or_append_scenes_based_on_prediction_result(
                results, scenes, frame_num=0
            )

        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].boxes[0], (18, 24, 48, 58))
        self.assertIsNotNone(scenes[0].cropped_boxes[0])
        self.assertLess(scenes[0].frames[0].numel(), frame.numel())


if __name__ == "__main__":
    unittest.main()
