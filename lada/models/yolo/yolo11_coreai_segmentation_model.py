# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core AI backend for fixed-shape YOLO11 segmentation models."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz

from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel
from lada.utils import ImageTensor


class CoreAISegmentationRuntime:
    def __init__(self, model_path: Path):
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self._runner: asyncio.Runner | None = None
        self._model = None
        self._function = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._function is not None:
            return
        try:
            from coreai.runtime import AIModel
        except ImportError as exc:
            raise RuntimeError(
                "Core AI mosaic detection requires the isolated coreai-torch environment"
            ) from exc

        self._runner = asyncio.Runner()
        self._model = self._runner.run(AIModel.load(self.model_path))
        self._function = self._model.load_function("main")

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.shape != (1, 3, 640, 640):
            raise ValueError(f"unexpected Core AI detection input shape: {image.shape}")
        if image.dtype != np.float16:
            raise ValueError(f"unexpected Core AI detection input dtype: {image.dtype}")

        with self._lock:
            self._ensure_loaded()
            assert self._runner is not None and self._function is not None
            from coreai.runtime import NDArray

            async def infer() -> tuple[np.ndarray, np.ndarray]:
                outputs = await self._function({"image": NDArray(image)})
                return (
                    outputs["candidates"].numpy().copy(),
                    outputs["prototypes"].numpy().copy(),
                )

            return self._runner.run(infer())


class Yolo11CoreAISegmentationModel(Yolo11SegmentationModel):
    def __init__(
        self,
        model_path: str | Path,
        device=None,
        imgsz=640,
        fp16=True,
        runtime: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        **kwargs,
    ):
        del device, fp16
        model_path = Path(model_path)
        if model_path.suffix != ".aimodel":
            raise ValueError(f"Expected an .aimodel path, got {str(model_path)!r}")

        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = LetterBox(self.imgsz, auto=False, stride=self.stride)
        custom = {
            "conf": 0.25,
            "batch": 1,
            "save": False,
            "mode": "predict",
            "device": "cpu",
            "half": True,
        }
        self.args = get_cfg(DEFAULT_CFG, {**custom, **kwargs})
        self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.model = SimpleNamespace(
            names={0: "mosaic_nsfw", 1: "mosaic_sfw_head"},
            end2end=False,
        )
        self.runtime = runtime or CoreAISegmentationRuntime(model_path)

    def preprocess(self, imgs: list[ImageTensor]) -> torch.Tensor:
        imgs = [img if img.device.type == "cpu" else img.cpu() for img in imgs]
        return self._preprocess_cpu(imgs)

    def inference(self, image_batch: torch.Tensor):
        candidates = []
        prototypes = []
        for index in range(image_batch.shape[0]):
            image = image_batch[index:index + 1].detach().cpu().contiguous().numpy()
            frame_candidates, frame_prototypes = self.runtime(image)
            candidates.append(torch.from_numpy(frame_candidates))
            prototypes.append(torch.from_numpy(frame_prototypes))
        return [(torch.cat(candidates, dim=0), torch.cat(prototypes, dim=0))]
