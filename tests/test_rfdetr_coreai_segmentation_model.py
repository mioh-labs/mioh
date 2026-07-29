# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import numpy as np
import torch

from lada.models.rfdetr import RFDETRCoreAISegmentationModel
from lada.utils.ultralytics_utils import convert_direct_resize_mask_tensor


class FakeRFDETRRuntime:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.inputs.append(image.copy())
        boxes = np.zeros((1, 200, 4), dtype=np.float32)
        logits = np.full((1, 200, 3), -20.0, dtype=np.float32)
        masks = np.full((1, 200, 144, 144), -20.0, dtype=np.float32)
        boxes[0, 7] = (0.5, 0.5, 0.5, 0.5)
        logits[0, 7, 1] = 8.0
        masks[0, 7, 36:108, 36:108] = 8.0
        return boxes, logits, masks


def test_jasna_v6_preprocess_and_postprocess() -> None:
    runtime = FakeRFDETRRuntime()
    model = RFDETRCoreAISegmentationModel(
        Path("rfdetr-v6-576-fp32.aimodel"),
        runtime=runtime,
    )
    image = torch.zeros((120, 200, 3), dtype=torch.uint8)

    results = model.inference_and_postprocess(
        model.preprocess([image]),
        [image],
    )

    assert len(runtime.inputs) == 1
    assert runtime.inputs[0].shape == (1, 3, 576, 576)
    assert runtime.inputs[0].dtype == np.float32
    assert len(results) == 1
    assert len(results[0].boxes) == 1
    assert torch.allclose(
        results[0].boxes.xyxy[0],
        torch.tensor((50.0, 30.0, 150.0, 90.0)),
    )
    assert results[0].boxes.conf[0] > 0.99
    assert results[0].masks.data.shape == (1, 144, 144)
    assert getattr(results[0], "_lada_direct_resize_masks") is True
    expanded = convert_direct_resize_mask_tensor(
        results[0].masks[0],
        results[0].orig_shape,
    )
    assert expanded.shape == (120, 200, 1)
    assert expanded.sum() > 0


def test_jasna_v6_preprocess_does_not_stack_source_frames() -> None:
    model = RFDETRCoreAISegmentationModel(
        Path("rfdetr-v6-576-fp32.aimodel"),
        runtime=FakeRFDETRRuntime(),
    )
    first = torch.zeros((120, 200, 3), dtype=torch.uint8)
    second = torch.ones((120, 200, 3), dtype=torch.uint8)

    source = [first, second]
    preprocessed = model.preprocess(source)

    assert isinstance(preprocessed, list)
    assert preprocessed is source
    assert preprocessed[0].data_ptr() == first.data_ptr()
    assert preprocessed[1].data_ptr() == second.data_ptr()


def test_jasna_v6_resize_before_float_stays_close_to_previous_path() -> None:
    model = RFDETRCoreAISegmentationModel(
        Path("rfdetr-v6-576-fp32.aimodel"),
        resolution=57,
        runtime=FakeRFDETRRuntime(),
    )
    generator = torch.Generator().manual_seed(20260729)
    image = torch.randint(
        0,
        256,
        (120, 200, 3),
        dtype=torch.uint8,
        generator=generator,
    )

    previous = image.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    previous = torch.nn.functional.interpolate(
        previous,
        size=(57, 57),
        mode="bilinear",
        align_corners=False,
    )
    previous = previous.sub(model._mean).div(model._std)
    optimized = model._normalize_one(image)
    difference = (optimized - previous).abs()

    assert difference.max().item() < 1e-4
    assert difference.mean().item() < 1e-6


def test_jasna_v6_keeps_each_query_only_once() -> None:
    class DuplicateClassRuntime(FakeRFDETRRuntime):
        def __call__(self, image):
            boxes, logits, masks = super().__call__(image)
            logits[0, 7] = 8.0
            return boxes, logits, masks

    model = RFDETRCoreAISegmentationModel(
        Path("rfdetr-v6-576-fp32.aimodel"),
        runtime=DuplicateClassRuntime(),
    )
    image = torch.zeros((120, 200, 3), dtype=torch.uint8)

    result = model.inference_and_postprocess(
        model.preprocess([image]),
        [image],
    )[0]

    assert len(result.boxes) == 1
    assert result.masks.data.shape == (1, 144, 144)


def test_jasna_v6_rejects_low_confidence_results() -> None:
    class EmptyRuntime(FakeRFDETRRuntime):
        def __call__(self, image):
            boxes, logits, masks = super().__call__(image)
            logits.fill(-20.0)
            return boxes, logits, masks

    model = RFDETRCoreAISegmentationModel(
        Path("rfdetr-v6-576-fp32.aimodel"),
        runtime=EmptyRuntime(),
    )
    image = torch.zeros((32, 48, 3), dtype=torch.uint8)

    result = model.inference_and_postprocess(
        model.preprocess([image]),
        [image],
    )[0]

    assert len(result.boxes) == 0


def test_jasna_v6_uses_compact_runtime_selection_when_available() -> None:
    class CompactRuntime:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, image):
            raise AssertionError("full RF-DETR mask bank must not be copied")

        def infer_selected(self, image, *, conf, max_det):
            self.calls.append((image.copy(), conf, max_det))
            return (
                np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
                np.asarray([0.95], dtype=np.float32),
                np.ones((1, 144, 144), dtype=np.float32),
            )

    runtime = CompactRuntime()
    model = RFDETRCoreAISegmentationModel(
        Path("rfdetr-v6-576-fp32.aimodel"),
        runtime=runtime,
        conf=0.4,
        max_det=8,
    )
    image = torch.zeros((120, 200, 3), dtype=torch.uint8)

    result = model.inference_and_postprocess(
        model.preprocess([image]),
        [image],
    )[0]

    assert len(runtime.calls) == 1
    assert runtime.calls[0][1:] == (0.4, 8)
    assert len(result.boxes) == 1
    assert result.masks.data.shape == (1, 144, 144)
