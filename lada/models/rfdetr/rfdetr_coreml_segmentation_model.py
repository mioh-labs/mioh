# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core ML runtime for the fixed-shape Jasna RF-DETR v6 detector."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .rfdetr_coreai_segmentation_model import RFDETRCoreAISegmentationModel


class RFDETRCoreMLRuntime:
    """Lazy Core ML adapter with the same raw contract as the Core AI lane."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        resolution: int,
        queries: int,
        logit_classes: int,
    ) -> None:
        path = Path(model_path)
        if path.suffix not in (".mlpackage", ".mlmodelc") or not path.is_dir():
            raise ValueError(f"Expected a Core ML model directory, got {path}")
        self.model_path = path
        self.resolution = int(resolution)
        self.queries = int(queries)
        self.logit_classes = int(logit_classes)
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import coremltools as ct

        unit_name = os.environ.get(
            "LADA_RFDETR_COREML_COMPUTE_UNITS",
            "CPU_AND_GPU",
        ).upper()
        try:
            compute_units = getattr(ct.ComputeUnit, unit_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported RF-DETR Core ML compute units: {unit_name}"
            ) from exc
        if self.model_path.suffix == ".mlmodelc":
            self._model = ct.models.CompiledMLModel(
                str(self.model_path),
                compute_units=compute_units,
            )
        else:
            self._model = ct.models.MLModel(
                str(self.model_path),
                compute_units=compute_units,
            )

    def __call__(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        expected = (1, 3, self.resolution, self.resolution)
        if image.shape != expected or image.dtype != np.float32:
            raise ValueError(
                f"unexpected RF-DETR Core ML input: {image.shape} {image.dtype}; "
                f"expected {expected} float32"
            )
        with self._lock:
            self._ensure_loaded()
            outputs = self._model.predict(
                {"image": np.ascontiguousarray(image)}
            )
            boxes = np.asarray(outputs["boxes"]).copy()
            logits = np.asarray(outputs["logits"]).copy()
            masks = np.asarray(outputs["masks"]).copy()
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
        boxes, logits, masks = self(image)
        query_logits = logits[0].max(axis=1)
        selection_count = min(max(0, int(max_det)), query_logits.shape[0])
        query_indexes = np.argsort(query_logits)[::-1][:selection_count]
        selected_logits = query_logits[query_indexes]
        scores = 1.0 / (
            1.0 + np.exp(-np.clip(selected_logits, -80.0, 80.0))
        )
        valid = scores > float(conf)
        query_indexes = query_indexes[valid]
        return (
            boxes[0, query_indexes].astype(np.float32, copy=True),
            scores[valid].astype(np.float32, copy=True),
            masks[0, query_indexes].astype(np.float32, copy=True),
        )

    def close(self) -> None:
        with self._lock:
            self._model = None


class RFDETRCoreMLSegmentationModel(RFDETRCoreAISegmentationModel):
    """Lada-compatible Jasna detector backed by an ML Program."""

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
        runtime: Callable | None = None,
        **kwargs,
    ) -> None:
        runtime = runtime or RFDETRCoreMLRuntime(
            model_path,
            resolution=resolution,
            queries=queries,
            logit_classes=logit_classes,
        )
        super().__init__(
            model_path,
            device,
            resolution=resolution,
            queries=queries,
            logit_classes=logit_classes,
            conf=conf,
            max_det=max_det,
            runtime=runtime,
            **kwargs,
        )
