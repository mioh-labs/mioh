#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Verify every Core AI model embedded in the standalone mioh app."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np


EXPECTED_MODEL_ASSETS = {
    "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-t36-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-t90-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v4_fast-fp16.h17s.aimodelc",
    "RealESRGAN_x4plus-256-fp16.h17s.aimodelc",
    "realesr-general-x4v3-256-fp16.h17s.aimodelc",
}

MODEL_CONTRACTS = {
    "basicvsrpp-v1.2-coreai": {
        "kind": "restoration",
        "frames": 18,
        "asset": "basicvsrpp-v1.2-t18-fp16.h17s.aimodelc",
    },
    "basicvsrpp-v1.2-coreai-t36": {
        "kind": "restoration",
        "frames": 36,
        "asset": "basicvsrpp-v1.2-t36-fp16.h17s.aimodelc",
    },
    "basicvsrpp-v1.2-coreai-t90": {
        "kind": "restoration",
        "frames": 90,
        "asset": "basicvsrpp-v1.2-t90-fp16.h17s.aimodelc",
    },
    "v4-fast-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_v4_fast-fp16.h17s.aimodelc",
    },
    "realesrgan-x4-coreai": {
        "kind": "enhancer",
        "asset": "RealESRGAN_x4plus-256-fp16.h17s.aimodelc",
    },
    "realesr-general-x4v3-coreai": {
        "kind": "enhancer",
        "asset": "realesr-general-x4v3-256-fp16.h17s.aimodelc",
    },
}


def gradient_input(shape: tuple[int, ...]) -> np.ndarray:
    values = np.linspace(0.0, 1.0, num=math.prod(shape), dtype=np.float32)
    return np.ascontiguousarray(values.reshape(shape).astype(np.float16))


def verify_asset_set(models_dir: Path) -> None:
    if not models_dir.is_dir():
        raise RuntimeError(f"missing standalone model directory: {models_dir}")
    source_assets = {
        item.name
        for item in models_dir.iterdir()
        if item.is_dir() and item.name.endswith(".aimodel")
    }
    if source_assets:
        raise RuntimeError(f"source Core AI assets are packaged: {sorted(source_assets)}")
    compiled_assets = {
        item.name
        for item in models_dir.iterdir()
        if item.is_dir() and item.name.endswith(".aimodelc")
    }
    unexpected = compiled_assets - EXPECTED_MODEL_ASSETS
    if unexpected:
        raise RuntimeError(f"unexpected Core AI assets: {sorted(unexpected)}")
    missing = EXPECTED_MODEL_ASSETS - compiled_assets
    if missing:
        raise RuntimeError(f"missing Core AI assets: {sorted(missing)}")


def _verify_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise RuntimeError(f"{name} returned shape {value.shape}, expected {shape}")
    if value.dtype != np.float16:
        raise RuntimeError(f"{name} returned {value.dtype}, expected float16")
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} returned non-finite values")


def _resolve_model(name: str, kind: str):
    from lada import ModelFiles

    if kind == "restoration":
        model = ModelFiles.get_restoration_model_by_name(name)
    elif kind == "detection":
        model = ModelFiles.get_detection_model_by_name(name)
    else:
        model = ModelFiles.get_enhancer_model_by_name(name)
    if model is None:
        raise RuntimeError(f"standalone model name did not resolve: {name}")
    return model


def _verify_restoration(name: str, path: Path, frames: int) -> None:
    import torch

    from lada.restorationpipeline.basicvsrpp_coreai_restorer import (
        CoreAIModelRuntime,
    )

    runtime = CoreAIModelRuntime(path, frame_count=frames)
    try:
        input_array = gradient_input((1, frames, 3, 256, 256))
        output = runtime(torch.from_numpy(input_array)).numpy()
        _verify_array(name, output, input_array.shape)
    finally:
        runtime.close()


def _verify_detection(name: str, path: Path) -> None:
    from lada.models.yolo.yolo11_coreai_segmentation_model import (
        CoreAISegmentationRuntime,
    )

    runtime = CoreAISegmentationRuntime(path)
    try:
        candidates, prototypes = runtime(
            np.zeros((1, 3, 640, 640), dtype=np.float16)
        )
        _verify_array(name + "/candidates", candidates, (1, 38, 8400))
        _verify_array(name + "/prototypes", prototypes, (1, 32, 160, 160))
    finally:
        runtime.close()


def _verify_enhancer(name: str, path: Path) -> None:
    from lada.restorationpipeline.coreai_roi_enhancer import CoreAIEnhancerRuntime

    runtime = CoreAIEnhancerRuntime(path, imgsz=256, scale=4)
    try:
        output = runtime(gradient_input((1, 3, 256, 256)))
        _verify_array(name, output, (1, 3, 1024, 1024))
    finally:
        runtime.close()


def verify_models(resources: Path) -> None:
    models_dir = resources / "models"
    verify_asset_set(models_dir)
    for name, contract in MODEL_CONTRACTS.items():
        model = _resolve_model(name, str(contract["kind"]))
        path = Path(model.path)
        expected_path = models_dir / str(contract["asset"])
        if path != expected_path or path.suffix != ".aimodelc":
            raise RuntimeError(
                f"{name} resolved to {path}, expected packaged model {expected_path}"
            )
        kind = contract["kind"]
        if kind == "restoration":
            _verify_restoration(name, path, int(contract["frames"]))
        elif kind == "detection":
            _verify_detection(name, path)
        else:
            _verify_enhancer(name, path)
        print(f"Core AI smoke passed: {name} -> {path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    resources = args.resources.resolve()
    models_dir = resources / "models"
    architecture = os.environ.setdefault("LADA_COREAI_ARCHITECTURE", "h17s")
    if architecture != "h17s":
        raise RuntimeError(f"standalone Core AI architecture must be h17s, got {architecture}")
    os.environ["LADA_MODEL_WEIGHTS_DIR"] = str(models_dir)
    os.environ.setdefault(
        "LADA_COREAI_SWIFT_RUNNER",
        str(resources / "bin" / "lada-coreai-runner"),
    )
    verify_models(resources)


if __name__ == "__main__":
    main()
