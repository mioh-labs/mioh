from types import SimpleNamespace
from threading import Thread
from unittest import mock

import torch

from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.utils.threading_utils import PipelineQueue


def test_zero_strength_roi_enhancer_does_not_load_model_weights():
    metadata = SimpleNamespace(
        video_width=1920,
        video_height=1080,
    )
    with (
        mock.patch(
            "lada.restorationpipeline.frame_restorer.video_utils.get_video_meta_data",
            return_value=metadata,
        ),
        mock.patch(
            "lada.restorationpipeline.frame_restorer.create_realesrgan_enhancer",
        ) as create_enhancer,
    ):
        restorer = FrameRestorer(
            device="cpu",
            video_file="unused.mp4",
            max_clip_length=180,
            mosaic_restoration_model_name="basicvsrpp-v1.2",
            mosaic_detection_model=object(),
            mosaic_restoration_model=object(),
            preferred_pad_mode="zero",
            restore_roi_enhancer="realesrgan",
            restore_roi_enhancer_model_path="unused.mlpackage",
            restore_roi_enhancer_strength=0.0,
        )

    create_enhancer.assert_not_called()
    assert restorer.restore_roi_enhancer is None
    assert restorer.frame_detection_queue.maxsize == 0


def test_detector_stop_unblocks_full_bounded_detection_queue():
    detection_queue = PipelineQueue("frame_detection_queue", maxsize=1)
    detection_queue.put((0, 0))
    detector = MosaicDetector(
        model=object(),
        video_metadata=object(),
        frame_detection_queue=detection_queue,
        mosaic_clip_queue=PipelineQueue("mosaic_clip_queue", maxsize=1),
        error_handler=lambda marker: None,
        device=torch.device("cpu"),
        batch_size=1,
    )
    blocked_producer = Thread(
        target=lambda: detection_queue.put((1, 0)),
        daemon=True,
    )
    detector.frame_detector_thread = blocked_producer
    blocked_producer.start()

    detector.stop()

    assert not blocked_producer.is_alive()
