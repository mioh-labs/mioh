"""Bridge LADA native mosaic detection into the MLX restore experiment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel
from lada.utils import Detection, ImageTensor
from lada.utils import ultralytics_utils


class LadaNativeMosaicDetector:
    """Small wrapper around LADA's native YOLO segmentation detector."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        fp16: bool = False,
        conf: float = 0.15,
        detect_face_mosaics: bool = False,
    ):
        classes = [0] if detect_face_mosaics else None
        self.model = Yolo11SegmentationModel(str(model_path), device, classes=classes, conf=conf, fp16=fp16)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        tensors = [torch.from_numpy(image) for image in images]
        preprocessed = self.model.preprocess(tensors)
        results = self.model.inference_and_postprocess(preprocessed, tensors)
        return [yolo_result_to_detections(result) for result in results]


def yolo_result_to_detections(result) -> list[Detection]:
    detections: list[Detection] = []
    if result.boxes is None or result.masks is None:
        return detections
    for yolo_box, yolo_mask in zip(result.boxes, result.masks):
        mask = ultralytics_utils.convert_yolo_mask(yolo_mask, result.orig_img.shape)
        box = ultralytics_utils.convert_yolo_box(yolo_box, result.orig_img.shape)
        t, l, b, r = box
        width, height = r - l + 1, b - t + 1
        if min(width, height) < 20:
            continue
        confidence = ultralytics_utils.convert_yolo_conf(yolo_box)
        detections.append(Detection(cls=4, box=box, mask=mask, confidence=confidence))
    return detections


def detections_to_mask(
    detections: list[Detection],
    *,
    frame_shape: tuple[int, int],
    confidence_threshold: float = 0.0,
) -> np.ndarray:
    height, width = frame_shape
    merged = np.zeros((height, width), dtype=np.uint8)
    for detection in detections:
        if detection.confidence is not None and detection.confidence < confidence_threshold:
            continue
        mask = detection.mask
        if mask.shape[:2] != (height, width):
            raise ValueError(f"detection mask shape {mask.shape[:2]} does not match frame shape {(height, width)}")
        mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
        merged = np.maximum(merged, mask_2d.astype(np.uint8))
    return merged


def detect_video_to_mask_dir(
    input_video: str | Path,
    output_dir: str | Path,
    *,
    detector_factory: Callable[[], object] | None = None,
    model_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    fp16: bool = False,
    conf: float = 0.15,
    detect_face_mosaics: bool = False,
    batch_size: int = 4,
    confidence_threshold: float = 0.0,
) -> list[Path]:
    """Run LADA native detection over a video and write `mask_XXXX.png` files."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if detector_factory is None:
        if model_path is None:
            raise ValueError("model_path is required when detector_factory is not provided")
        detector_factory = lambda: LadaNativeMosaicDetector(
            model_path,
            device=device,
            fp16=fp16,
            conf=conf,
            detect_face_mosaics=detect_face_mosaics,
        )

    detector = detector_factory()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {input_video}")

    paths: list[Path] = []
    batch: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            batch.append(frame)
            if len(batch) >= batch_size:
                paths.extend(_flush_detection_batch(detector, batch, output_dir, frame_index, confidence_threshold))
                frame_index += len(batch)
                batch = []
        if batch:
            paths.extend(_flush_detection_batch(detector, batch, output_dir, frame_index, confidence_threshold))
    finally:
        capture.release()

    if not paths:
        raise ValueError(f"no frames read from video: {input_video}")
    return paths


def _flush_detection_batch(
    detector: object,
    images: list[np.ndarray],
    output_dir: Path,
    start_index: int,
    confidence_threshold: float,
) -> list[Path]:
    batch_detections = detector.detect_batch(images)
    paths: list[Path] = []
    for offset, (image, detections) in enumerate(zip(images, batch_detections)):
        mask = detections_to_mask(
            detections,
            frame_shape=image.shape[:2],
            confidence_threshold=confidence_threshold,
        )
        path = output_dir / f"mask_{start_index + offset:04d}.png"
        if not cv2.imwrite(str(path), mask):
            raise ValueError(f"failed to write mask: {path}")
        paths.append(path)
    return paths
