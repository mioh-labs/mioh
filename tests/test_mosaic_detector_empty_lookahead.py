import unittest
from types import SimpleNamespace

from lada.restorationpipeline.mosaic_detector import MosaicDetector


class FakeModel:
    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.inference_calls = []

    def preprocess(self, frames):
        return list(frames)

    def inference_and_postprocess(self, frames_batch, frames):
        self.inference_calls.append(list(frames))
        return [
            SimpleNamespace(boxes=[object()] * self.boxes_by_frame.get(frame, 0))
            for frame in frames
        ]


class MosaicDetectorEmptyLookaheadTests(unittest.TestCase):
    def test_empty_lookahead_skips_full_range_when_first_and_last_are_empty(self):
        model = FakeModel({})
        detector = MosaicDetector(
            model=model,
            video_metadata=SimpleNamespace(video_file="input.mp4"),
            frame_detection_queue=None,
            mosaic_clip_queue=None,
            error_handler=lambda marker: None,
            empty_lookahead_frames=10,
        )

        result = detector._run_empty_lookahead_inference(list(range(10)), frame_num=100)

        self.assertEqual(result, ("skip_empty_range", 100, 10))
        self.assertEqual(model.inference_calls, [[0, 9]])

    def test_empty_lookahead_falls_back_to_full_inference_when_endpoint_has_mosaic(self):
        model = FakeModel({9: 1})
        detector = MosaicDetector(
            model=model,
            video_metadata=SimpleNamespace(video_file="input.mp4"),
            frame_detection_queue=None,
            mosaic_clip_queue=None,
            error_handler=lambda marker: None,
            empty_lookahead_frames=10,
        )

        results, frames, frame_num = detector._run_empty_lookahead_inference(list(range(10)), frame_num=100)

        self.assertEqual(frame_num, 100)
        self.assertEqual(frames, list(range(10)))
        self.assertEqual([len(result.boxes) for result in results], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(model.inference_calls, [[0, 9], list(range(10))])


if __name__ == "__main__":
    unittest.main()
