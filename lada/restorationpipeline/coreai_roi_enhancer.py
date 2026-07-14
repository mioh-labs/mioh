# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Fixed-shape Core AI backend for ROI enhancement models."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec
from lada.coreai.source_runtime import load_source_model


class CoreAIEnhancerRuntime:
    def __init__(self, model_path: Path, imgsz: int = 256, scale: int = 4):
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.imgsz = imgsz
        self.scale = scale
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
                    inputs=(
                        TensorSpec("image", (1, 3, self.imgsz, self.imgsz)),
                    ),
                    outputs=(
                        TensorSpec(
                            "enhanced",
                            (1, 3, self.imgsz * self.scale, self.imgsz * self.scale),
                        ),
                    ),
                )
            return
        if self._function is not None:
            return
        self._runner = asyncio.Runner()
        self._model = load_source_model(
            self._runner,
            self.model_path,
            purpose="ROIエンハンサー",
        )
        self._function = self._model.load_function("main")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        expected = (1, 3, self.imgsz, self.imgsz)
        if image.shape != expected or image.dtype != np.float16:
            raise ValueError(
                f"Core AI enhancer requires FP16 input shape {expected}; "
                f"got {image.dtype} {image.shape}"
            )
        with self._lock:
            self._ensure_loaded()
            if self._compiled_runtime is not None:
                output = self._compiled_runtime.infer({"image": image})["enhanced"]
            else:
                assert self._runner is not None and self._function is not None
                from coreai.runtime import NDArray

                async def infer() -> np.ndarray:
                    outputs = await self._function({"image": NDArray(image)})
                    return outputs["enhanced"].numpy().copy()

                output = self._runner.run(infer())
        expected_output = (1, 3, self.imgsz * self.scale, self.imgsz * self.scale)
        if output.shape != expected_output:
            raise ValueError(
                f"unexpected Core AI enhancer output shape {output.shape}; "
                f"expected {expected_output}"
            )
        return output

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


class CoreAIROIEnhancer:
    uses_torch_device = False
    prefer_pre_resize = True
    enhancer_name = "realesrgan"

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 256,
        scale: int = 4,
        runtime: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.scale = scale
        self.runtime = runtime or CoreAIEnhancerRuntime(
            self.model_path,
            imgsz,
            scale,
        )

    def enhance(self, img_bgr: np.ndarray, outscale: int | None = None):
        del outscale
        height, width = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if (height, width) != (self.imgsz, self.imgsz):
            interpolation = (
                cv2.INTER_AREA
                if max(height, width) > self.imgsz
                else cv2.INTER_CUBIC
            )
            rgb = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=interpolation)
        image = (
            rgb.astype(np.float16)
            .transpose(2, 0, 1)[None]
            / np.float16(255.0)
        )
        enhanced = self.runtime(np.ascontiguousarray(image))[0]
        enhanced = np.clip(enhanced.transpose(1, 2, 0), 0.0, 1.0)
        enhanced = np.rint(enhanced * 255.0).astype(np.uint8)
        target = (width * self.scale, height * self.scale)
        if enhanced.shape[:2] != (target[1], target[0]):
            enhanced = cv2.resize(enhanced, target, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), None

    def close(self) -> None:
        close_runtime = getattr(self.runtime, "close", None)
        if callable(close_runtime):
            close_runtime()
