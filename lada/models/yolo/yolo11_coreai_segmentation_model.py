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

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec
from lada.coreai.source_runtime import load_source_model
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel
from lada.utils import ImageTensor


def detection_candidate_channels(model_path: str | Path) -> int:
    """Return the raw YOLO candidate width for a shipped detector.

    The original v2 checkpoint has one class (4 box + 1 class + 32 masks);
    every later shipped detector has two classes.
    """
    name = Path(model_path).name
    return 37 if name.startswith("lada_mosaic_detection_model_v2-") else 38


class CoreAISegmentationRuntime:
    def __init__(self, model_path: Path, candidate_channels: int | None = None):
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.candidate_channels = (
            detection_candidate_channels(model_path)
            if candidate_channels is None
            else candidate_channels
        )
        self._runner: asyncio.Runner | None = None
        self._model = None
        self._function = None
        self._compiled_runtime: CompiledCoreAIRuntime | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self.model_path.suffix == ".aimodelc":
            if self._compiled_runtime is None:
                self._compiled_runtime = CompiledCoreAIRuntime(
                    self.model_path,
                    inputs=(TensorSpec("image", (1, 3, 640, 640)),),
                    outputs=(
                        TensorSpec(
                            "candidates",
                            (1, self.candidate_channels, 8400),
                        ),
                        TensorSpec("prototypes", (1, 32, 160, 160)),
                    ),
                )
            return
        if self._function is not None:
            return
        self._runner = asyncio.Runner()
        self._model = load_source_model(
            self._runner,
            self.model_path,
            purpose="モザイク検出",
        )
        self._function = self._model.load_function("main")

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.shape != (1, 3, 640, 640):
            raise ValueError(f"unexpected Core AI detection input shape: {image.shape}")
        if image.dtype != np.float16:
            raise ValueError(f"unexpected Core AI detection input dtype: {image.dtype}")

        with self._lock:
            self._ensure_loaded()
            if self._compiled_runtime is not None:
                outputs = self._compiled_runtime.infer({"image": image})
                return outputs["candidates"], outputs["prototypes"]
            assert self._runner is not None and self._function is not None
            from coreai.runtime import NDArray

            async def infer() -> tuple[np.ndarray, np.ndarray]:
                outputs = await self._function({"image": NDArray(image)})
                return (
                    outputs["candidates"].numpy().copy(),
                    outputs["prototypes"].numpy().copy(),
                )

            return self._runner.run(infer())

    def close(self) -> None:
        with self._lock:
            runner = self._runner
            compiled_runtime = self._compiled_runtime
            self._runner = None
            self._model = None
            self._function = None
            self._compiled_runtime = None
            if runner is not None:
                runner.close()
            if compiled_runtime is not None:
                compiled_runtime.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
        if model_path.suffix not in {".aimodel", ".aimodelc"}:
            raise ValueError(
                f"Expected an .aimodel or .aimodelc path, got {str(model_path)!r}"
            )

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
        candidate_channels = detection_candidate_channels(model_path)
        names = (
            {0: "nsfw"}
            if candidate_channels == 37
            else {0: "mosaic_nsfw", 1: "mosaic_sfw_head"}
        )
        self.model = SimpleNamespace(
            names=names,
            end2end=False,
        )
        self.runtime = runtime or CoreAISegmentationRuntime(
            model_path,
            candidate_channels=candidate_channels,
        )

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
