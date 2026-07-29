# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core AI runtime and Lada-compatible postprocessing for RF-DETR Seg."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Results

from lada.coreai.source_runtime import load_source_model
from lada.utils import ImageTensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RFDETRCoreAIRuntime:
    """Lazy source-runtime adapter for the FP32 Jasna v6 Core AI asset."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        resolution: int,
        queries: int,
        logit_classes: int,
    ) -> None:
        path = Path(model_path)
        if path.suffix != ".aimodel" or not path.is_dir():
            raise ValueError(f"Expected a source .aimodel directory, got {path}")
        self.model_path = path
        self.resolution = int(resolution)
        self.queries = int(queries)
        self.logit_classes = int(logit_classes)
        self._runner: asyncio.Runner | None = None
        self._model = None
        self._function = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._function is not None:
            return
        self._runner = asyncio.Runner()
        self._model = load_source_model(
            self._runner,
            self.model_path,
            purpose="RF-DETRモザイク検出",
        )
        self._function = self._model.load_function("main")

    def __call__(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected = (1, 3, self.resolution, self.resolution)
        if image.shape != expected:
            raise ValueError(
                f"unexpected RF-DETR Core AI input shape: {image.shape}; "
                f"expected {expected}"
            )
        if image.dtype != np.float32:
            raise ValueError(
                f"unexpected RF-DETR Core AI input dtype: {image.dtype}"
            )

        with self._lock:
            self._ensure_loaded()
            assert self._runner is not None and self._function is not None
            from coreai.runtime import NDArray

            async def infer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                outputs = await self._function({"image": NDArray(image)})
                return (
                    outputs["boxes"].numpy().copy(),
                    outputs["logits"].numpy().copy(),
                    outputs["masks"].numpy().copy(),
                )

            boxes, logits, masks = self._runner.run(infer())

        if boxes.shape != (1, self.queries, 4):
            raise ValueError(f"unexpected RF-DETR boxes shape: {boxes.shape}")
        if logits.shape != (1, self.queries, self.logit_classes):
            raise ValueError(f"unexpected RF-DETR logits shape: {logits.shape}")
        return boxes, logits, masks

    def infer_selected(
        self,
        image: np.ndarray,
        *,
        conf: float,
        max_det: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and copy only masks that survive query selection."""
        expected = (1, 3, self.resolution, self.resolution)
        if image.shape != expected:
            raise ValueError(
                f"unexpected RF-DETR Core AI input shape: {image.shape}; "
                f"expected {expected}"
            )
        if image.dtype != np.float32:
            raise ValueError(
                f"unexpected RF-DETR Core AI input dtype: {image.dtype}"
            )

        with self._lock:
            self._ensure_loaded()
            assert self._runner is not None and self._function is not None
            from coreai.runtime import NDArray

            async def infer() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                outputs = await self._function({"image": NDArray(image)})
                boxes_view = outputs["boxes"].numpy()
                logits_view = outputs["logits"].numpy()
                masks_view = outputs["masks"].numpy()

                query_logits = logits_view[0].max(axis=1)
                selection_count = min(
                    max(0, int(max_det)),
                    int(query_logits.shape[0]),
                )
                if selection_count:
                    query_indexes = np.argsort(query_logits)[::-1][
                        :selection_count
                    ]
                    selected_logits = query_logits[query_indexes]
                    scores = 1.0 / (
                        1.0 + np.exp(-np.clip(selected_logits, -80.0, 80.0))
                    )
                    valid = scores > float(conf)
                    query_indexes = query_indexes[valid]
                    scores = scores[valid]
                else:
                    query_indexes = np.empty((0,), dtype=np.int64)
                    scores = np.empty((0,), dtype=np.float32)

                # Advanced indexing creates compact arrays. This prevents the
                # full 200-query v6 mask bank from escaping the Core AI call.
                return (
                    boxes_view[0, query_indexes].copy(),
                    scores.astype(np.float32, copy=True),
                    masks_view[0, query_indexes].copy(),
                )

            return self._runner.run(infer())

    def close(self) -> None:
        with self._lock:
            runner = self._runner
            self._runner = None
            self._model = None
            self._function = None
            if runner is not None:
                runner.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RFDETRCoreAISegmentationModel:
    """Jasna v6 RF-DETR detector with the interface used by MosaicDetector."""

    def __init__(
        self,
        model_path: str | Path,
        device=None,
        *,
        resolution: int = 576,
        queries: int = 200,
        logit_classes: int = 3,
        conf: float = 0.35,
        max_det: int = 16,
        runtime: Callable[
            [np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
        | None = None,
        **_kwargs,
    ) -> None:
        del device
        self.resolution = int(resolution)
        self.queries = int(queries)
        self.logit_classes = int(logit_classes)
        self.conf = float(conf)
        self.max_det = int(max_det)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.runtime = runtime or RFDETRCoreAIRuntime(
            model_path,
            resolution=self.resolution,
            queries=self.queries,
            logit_classes=self.logit_classes,
        )
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def preprocess(self, imgs: list[ImageTensor]) -> list[ImageTensor]:
        """Keep source frames unbatched until each frame is consumed.

        RF-DETR's Core AI function is single-frame. Stacking full-resolution
        frames here only duplicates the source queue and makes a second 4K
        allocation survive until postprocessing.
        """
        return imgs

    @staticmethod
    def _resize_uint8_bilinear_to_float(
        image: ImageTensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """Bilinear resize HWC uint8 without a full-resolution float copy.

        Only the four source samples required for each output pixel are
        converted to float32. The arithmetic follows align_corners=False and
        matches the previous float32 ``F.interpolate`` path.
        """
        input_height, input_width = image.shape[:2]
        if (input_height, input_width) == (output_height, output_width):
            return image.float()

        y = (
            (torch.arange(output_height, dtype=torch.float32) + 0.5)
            * (input_height / output_height)
            - 0.5
        )
        x = (
            (torch.arange(output_width, dtype=torch.float32) + 0.5)
            * (input_width / output_width)
            - 0.5
        )
        y0 = y.floor().to(torch.int64)
        x0 = x.floor().to(torch.int64)
        y1 = y0 + 1
        x1 = x0 + 1
        weight_y = (y - y0.float())[:, None, None]
        weight_x = (x - x0.float())[None, :, None]
        y0.clamp_(0, input_height - 1)
        y1.clamp_(0, input_height - 1)
        x0.clamp_(0, input_width - 1)
        x1.clamp_(0, input_width - 1)

        top = image[y0[:, None], x0[None, :]].float()
        horizontal_delta = image[y0[:, None], x1[None, :]].float()
        horizontal_delta.sub_(top).mul_(weight_x)
        top.add_(horizontal_delta)
        del horizontal_delta

        bottom = image[y1[:, None], x0[None, :]].float()
        horizontal_delta = image[y1[:, None], x1[None, :]].float()
        horizontal_delta.sub_(bottom).mul_(weight_x)
        bottom.add_(horizontal_delta)
        del horizontal_delta

        bottom.sub_(top).mul_(weight_y)
        return top.add_(bottom)

    def _normalize_one(self, image: ImageTensor) -> torch.Tensor:
        """Resize uint8 first, then create the model-sized float32 tensor."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"expected HWC RGB image, got shape {tuple(image.shape)}"
            )
        source = image.detach().cpu()
        if source.dtype != torch.uint8:
            source = source.clamp(0, 255).to(torch.uint8)
        # Delaying float conversion avoids a ~95 MiB temporary for each 4K
        # RGB frame while preserving the previous interpolation numerics.
        value = self._resize_uint8_bilinear_to_float(
            source,
            self.resolution,
            self.resolution,
        )
        return (
            value.permute(2, 0, 1)
            .unsqueeze(0)
            .div_(255.0)
            .sub_(self._mean)
            .div_(self._std)
            .contiguous()
        )

    def inference_and_postprocess(
        self,
        imgs: list[ImageTensor],
        orig_imgs: list[ImageTensor],
    ) -> list[Results]:
        if len(imgs) != len(orig_imgs):
            raise ValueError(
                f"RF-DETR input/original count mismatch: "
                f"{len(imgs)} != {len(orig_imgs)}"
            )
        results: list[Results] = []
        infer_selected = getattr(self.runtime, "infer_selected", None)
        for source_img, orig_img in zip(imgs, orig_imgs, strict=True):
            normalized = self._normalize_one(source_img)
            image = normalized.numpy()
            if callable(infer_selected):
                selected = infer_selected(
                    image,
                    conf=self.conf,
                    max_det=self.max_det,
                )
                results.append(self._postprocess_selected(selected, orig_img))
            else:
                raw = self.runtime(image)
                results.append(self._postprocess_one(raw, orig_img))
            del image
            del normalized
        return results

    def _postprocess_one(
        self,
        raw: tuple[np.ndarray, np.ndarray, np.ndarray],
        orig_img: ImageTensor,
    ) -> Results:
        pred_boxes = torch.from_numpy(raw[0])
        pred_logits = torch.from_numpy(raw[1])
        pred_masks = torch.from_numpy(raw[2])
        _, query_count, _class_count = pred_logits.shape

        probabilities = pred_logits.sigmoid()
        selection_count = min(self.max_det, query_count)
        # Jasna v6 is a single semantic detector. Select each RF-DETR query
        # only once even when multiple auxiliary logits are high; flattening
        # query × class can otherwise retain the same large mask repeatedly.
        query_probabilities = probabilities.max(dim=2).values
        scores, query_indexes = torch.topk(
            query_probabilities,
            selection_count,
            dim=1,
        )
        valid = scores[0] > self.conf
        selected_queries = query_indexes[0][valid]
        selected_scores = scores[0][valid]
        # Jasna v6 is a single semantic mosaic detector.  RF-DETR retains
        # auxiliary/background logit slots in the deployed checkpoint, but
        # Lada's downstream tracker expects every accepted mask to be class 0.
        if selected_queries.numel() == 0:
            selected_boxes = np.empty((0, 4), dtype=np.float32)
            selected_masks = np.empty(
                (0, pred_masks.shape[-2], pred_masks.shape[-1]),
                dtype=np.float32,
            )
        else:
            selected_boxes = pred_boxes[0, selected_queries].numpy()
            selected_masks = pred_masks[0, selected_queries].numpy()

        return self._postprocess_selected(
            (
                selected_boxes,
                selected_scores.numpy(),
                selected_masks,
            ),
            orig_img,
        )

    def _postprocess_selected(
        self,
        selected: tuple[np.ndarray, np.ndarray, np.ndarray],
        orig_img: ImageTensor,
    ) -> Results:
        pred_boxes = torch.from_numpy(selected[0]).float()
        selected_scores = torch.from_numpy(selected[1]).float()
        masks = torch.from_numpy(selected[2]).float()
        height, width = orig_img.shape[:2]

        if pred_boxes.numel() == 0:
            boxes_data = torch.empty((0, 6), dtype=torch.float32)
        else:
            cx, cy, box_width, box_height = pred_boxes.unbind(-1)
            xyxy = torch.stack(
                (
                    cx - box_width * 0.5,
                    cy - box_height * 0.5,
                    cx + box_width * 0.5,
                    cy + box_height * 0.5,
                ),
                dim=-1,
            )
            xyxy = xyxy * xyxy.new_tensor((width, height, width, height))
            xyxy[:, 0::2].clamp_(0, width)
            xyxy[:, 1::2].clamp_(0, height)
            selected_classes = torch.zeros_like(selected_scores)
            boxes_data = torch.cat(
                (
                    xyxy,
                    selected_scores[:, None],
                    selected_classes[:, None],
                ),
                dim=1,
            )

        result = Results(
            orig_img,
            path="",
            names={0: "mosaic"},
            boxes=boxes_data,
            masks=masks,
        )
        result._lada_direct_resize_masks = True
        return result

    def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()
