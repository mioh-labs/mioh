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
    "lada_mosaic_detection_model_v2-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v3.1_fast-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v3.1_accurate-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v4_fast-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_v4_accurate-fp16.h17s.aimodelc",
    "lada_mosaic_detection_model_vr_v2_accurate-fp16.h17s.aimodelc",
    "RealESRGAN_x2plus-256-fp16.h17s.aimodelc",
    "RealESRGAN_x4plus-256-fp16.h17s.aimodelc",
    "realesr-general-x4v3-256-fp16.h17s.aimodelc",
    "4xNomosWebPhoto_RealPLKSR-256-fp16.h17s.aimodelc",
    "basicvsrpp-v1.2-variable-coreai.h17s.aimodelc",
    "basicvsrpp-v1.2-variable-hq-coreai.h17s.aimodelc",
}

EXPECTED_SOURCE_MODEL_ASSETS = {
    "basicvsrpp-v1.2-t18-fp16.aimodel",
    "basicvsrpp-v1.2-t36-fp16.aimodel",
    "basicvsrpp-v1.2-t90-fp16.aimodel",
    "lada_mosaic_detection_model_v2-fp16.aimodel",
    "lada_mosaic_detection_model_v3.1_fast-fp16.aimodel",
    "lada_mosaic_detection_model_v3.1_accurate-fp16.aimodel",
    "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
    "lada_mosaic_detection_model_v4_accurate-fp16.aimodel",
    "lada_mosaic_detection_model_vr_v2_accurate-fp16.aimodel",
    "RealESRGAN_x2plus-256-fp16.aimodel",
    "RealESRGAN_x4plus-256-fp16.aimodel",
    "realesr-general-x4v3-256-fp16.aimodel",
    "4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel",
    "basicvsrpp-v1.2-variable-coreai.aimodel",
}
EXPECTED_DEDICATED_SOURCE_MODEL_ASSETS = {
    "rfdetr-v6-576-fp32.aimodel",
    "rfdetr-v6-large-768-fp32.aimodel",
}
VARIABLE_MODEL_NAME = "basicvsrpp-v1.2-coreai-variable"
VARIABLE_HQ_MODEL_NAME = "basicvsrpp-v1.2-coreai-variable-hq"
VARIABLE_STEP1_ASSET_NAMES = {
    "spatial",
    "flow",
    "backward_1_init",
    "backward_1_first",
    "backward_1_later",
    "forward_1_init",
    "forward_1_first",
    "forward_1_later",
    "backward_2_init",
    "backward_2_first",
    "backward_2_later",
    "forward_2_init",
    "forward_2_first",
    "forward_2_later",
    "reconstruction",
}
VARIABLE_CHUNK6_ASSET_NAMES = {
    "spatial6",
    "flow6",
    "backward_1_start6",
    "backward_1_continue6",
    "forward_1_start6",
    "forward_1_continue6",
    "backward_2_start6",
    "backward_2_continue6",
    "forward_2_start6",
    "forward_2_continue6",
    "reconstruction6",
}

MODEL_CONTRACTS = {
    "basicvsrpp-v1.2-coreai": {
        "kind": "restoration",
        "frames": 18,
        "asset": "basicvsrpp-v1.2-t18-fp16.aimodel",
    },
    "basicvsrpp-v1.2-coreai-t36": {
        "kind": "restoration",
        "frames": 36,
        "asset": "basicvsrpp-v1.2-t36-fp16.aimodel",
    },
    "basicvsrpp-v1.2-coreai-t90": {
        "kind": "restoration",
        "frames": 90,
        "asset": "basicvsrpp-v1.2-t90-fp16.aimodel",
    },
    "v4-fast-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_v4_fast-fp16.aimodel",
        "candidate_channels": 38,
    },
    "v2-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_v2-fp16.aimodel",
        "candidate_channels": 37,
    },
    "v3.1-fast-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_v3.1_fast-fp16.aimodel",
        "candidate_channels": 38,
    },
    "v3.1-accurate-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_v3.1_accurate-fp16.aimodel",
        "candidate_channels": 38,
    },
    "v4-accurate-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_v4_accurate-fp16.aimodel",
        "candidate_channels": 38,
    },
    "vr-v2-accurate-coreai": {
        "kind": "detection",
        "asset": "lada_mosaic_detection_model_vr_v2_accurate-fp16.aimodel",
        "candidate_channels": 38,
    },
    "realesrgan-x4-coreai": {
        "kind": "enhancer",
        "asset": "RealESRGAN_x4plus-256-fp16.aimodel",
        "scale": 4,
    },
    "realesrgan-x2-coreai": {
        "kind": "enhancer",
        "asset": "RealESRGAN_x2plus-256-fp16.aimodel",
        "scale": 2,
    },
    "realesr-general-x4v3-coreai": {
        "kind": "enhancer",
        "asset": "realesr-general-x4v3-256-fp16.aimodel",
        "scale": 4,
    },
    "nomos-webphoto-realplksr-x4-coreai": {
        "kind": "enhancer",
        "asset": "4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel",
        "scale": 4,
    },
}


def gradient_input(shape: tuple[int, ...]) -> np.ndarray:
    values = np.linspace(0.0, 1.0, num=math.prod(shape), dtype=np.float32)
    return np.ascontiguousarray(values.reshape(shape).astype(np.float16))


def expected_model_assets(distribution: str, architecture: str) -> set[str]:
    if distribution == "portable":
        return set(EXPECTED_SOURCE_MODEL_ASSETS)
    if distribution == "dedicated":
        assets = {
            f"{Path(asset).stem}.{architecture}.aimodelc"
            for asset in EXPECTED_SOURCE_MODEL_ASSETS
        }
        assets.update(EXPECTED_DEDICATED_SOURCE_MODEL_ASSETS)
        assets.add(f"basicvsrpp-v1.2-variable-coreai.{architecture}.aimodelc")
        assets.add(
            f"basicvsrpp-v1.2-variable-hq-coreai.{architecture}.aimodelc"
        )
        return assets
    raise ValueError(f"unsupported Core AI distribution: {distribution}")


def model_asset_name(
    source_asset: str,
    distribution: str,
    architecture: str,
) -> str:
    if distribution == "portable":
        return source_asset
    if distribution == "dedicated":
        return f"{Path(source_asset).stem}.{architecture}.aimodelc"
    raise ValueError(f"unsupported Core AI distribution: {distribution}")


def verify_asset_set(
    models_dir: Path,
    distribution: str = "dedicated",
    architecture: str = "h17s",
) -> None:
    if not models_dir.is_dir():
        raise RuntimeError(f"missing standalone model directory: {models_dir}")
    source_assets = {
        item.name
        for item in models_dir.iterdir()
        if item.is_dir() and item.name.endswith(".aimodel")
    }
    if distribution == "dedicated":
        unexpected_sources = source_assets - EXPECTED_DEDICATED_SOURCE_MODEL_ASSETS
        if unexpected_sources:
            raise RuntimeError(
                f"source Core AI assets are packaged: {sorted(unexpected_sources)}"
            )
    compiled_assets = {
        item.name
        for item in models_dir.iterdir()
        if item.is_dir() and item.name.endswith(".aimodelc")
    }
    expected = expected_model_assets(distribution, architecture)
    actual = source_assets | compiled_assets
    unexpected = actual - expected
    if unexpected:
        raise RuntimeError(f"unexpected Core AI assets: {sorted(unexpected)}")
    missing = expected - actual
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


def _verify_variable_restoration(
    name: str,
    path: Path,
    architecture: str,
    asset_names: set[str],
) -> None:
    import torch

    from lada.restorationpipeline.basicvsrpp_coreai_restorer import (
        VariableCoreAIModelRuntime,
    )

    compiled = path.suffix == ".aimodelc"
    expected = {
        (
            f"basicvsrpp-variable-{asset}.{architecture}.aimodelc"
            if compiled
            else f"basicvsrpp-variable-{asset}.aimodel"
        )
        for asset in asset_names
    }
    actual = {
        item.name
        for item in path.iterdir()
        if item.is_dir()
        and item.name.endswith(".aimodelc" if compiled else ".aimodel")
    }
    if actual != expected:
        raise RuntimeError(
            f"variable Core AI asset mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    runtime = VariableCoreAIModelRuntime(path)
    try:
        input_array = gradient_input((1, 3, 3, 256, 256))
        output = runtime(torch.from_numpy(input_array)).numpy()
        _verify_array(name, output, input_array.shape)
    finally:
        runtime.close()


def _verify_detection(
    name: str,
    path: Path,
    candidate_channels: int = 38,
) -> None:
    from lada.models.yolo.yolo11_coreai_segmentation_model import (
        CoreAISegmentationRuntime,
    )

    runtime = CoreAISegmentationRuntime(
        path,
        candidate_channels=candidate_channels,
    )
    try:
        candidates, prototypes = runtime(
            np.zeros((1, 3, 640, 640), dtype=np.float16)
        )
        _verify_array(
            name + "/candidates",
            candidates,
            (1, candidate_channels, 8400),
        )
        _verify_array(name + "/prototypes", prototypes, (1, 32, 160, 160))
    finally:
        runtime.close()


def _verify_enhancer(name: str, path: Path, scale: int) -> None:
    from lada.restorationpipeline.coreai_roi_enhancer import CoreAIEnhancerRuntime

    runtime = CoreAIEnhancerRuntime(path, imgsz=256, scale=scale)
    try:
        output = runtime(gradient_input((1, 3, 256, 256)))
        _verify_array(name, output, (1, 3, 256 * scale, 256 * scale))
    finally:
        runtime.close()


def verify_models(
    resources: Path,
    *,
    distribution: str = "dedicated",
    architecture: str = "h17s",
    smoke_names: set[str] | None = None,
) -> None:
    models_dir = resources / "models"
    verify_asset_set(models_dir, distribution, architecture)
    all_smokes = set(MODEL_CONTRACTS)
    all_smokes.add(VARIABLE_MODEL_NAME)
    if distribution == "dedicated":
        all_smokes.add(VARIABLE_HQ_MODEL_NAME)
    selected_smokes = all_smokes if smoke_names is None else smoke_names
    unknown_smokes = selected_smokes - all_smokes
    if unknown_smokes:
        raise ValueError(f"unknown Core AI smoke models: {sorted(unknown_smokes)}")
    for name, contract in MODEL_CONTRACTS.items():
        model = _resolve_model(name, str(contract["kind"]))
        path = Path(model.path)
        asset = model_asset_name(str(contract["asset"]), distribution, architecture)
        expected_path = models_dir / asset
        expected_suffix = ".aimodelc" if distribution == "dedicated" else ".aimodel"
        if path != expected_path or path.suffix != expected_suffix:
            raise RuntimeError(
                f"{name} resolved to {path}, expected packaged model {expected_path}"
            )
        if name not in selected_smokes:
            print(f"Core AI asset passed: {name} -> {path.name}", flush=True)
            continue
        kind = contract["kind"]
        if kind == "restoration":
            _verify_restoration(name, path, int(contract["frames"]))
        elif kind == "detection":
            _verify_detection(
                name,
                path,
                int(contract.get("candidate_channels", 38)),
            )
        else:
            _verify_enhancer(name, path, int(contract.get("scale", 4)))
        print(f"Core AI smoke passed: {name} -> {path.name}", flush=True)
    variable_models = [
        (
            VARIABLE_MODEL_NAME,
            (
                f"basicvsrpp-v1.2-variable-coreai.{architecture}.aimodelc"
                if distribution == "dedicated"
                else "basicvsrpp-v1.2-variable-coreai.aimodel"
            ),
            VARIABLE_CHUNK6_ASSET_NAMES,
            None,
        )
    ]
    if distribution == "dedicated":
        variable_models.append(
            (
                VARIABLE_HQ_MODEL_NAME,
                f"basicvsrpp-v1.2-variable-hq-coreai.{architecture}.aimodelc",
                VARIABLE_STEP1_ASSET_NAMES,
                resources / "bin" / "lada-basicvsrpp-variable-hq-runner",
            )
        )
    for name, asset, asset_names, runner in variable_models:
        model = _resolve_model(name, "restoration")
        path = Path(model.path)
        expected_path = models_dir / asset
        if path != expected_path:
            raise RuntimeError(f"{name} resolved to {path}, expected {expected_path}")
        if name in selected_smokes:
            previous_runner = os.environ.get("LADA_VARIABLE_COREAI_SWIFT_RUNNER")
            if runner is not None:
                os.environ["LADA_VARIABLE_COREAI_SWIFT_RUNNER"] = str(runner)
            try:
                _verify_variable_restoration(
                    name,
                    path,
                    architecture,
                    asset_names,
                )
            finally:
                if runner is not None:
                    if previous_runner is None:
                        os.environ.pop("LADA_VARIABLE_COREAI_SWIFT_RUNNER", None)
                    else:
                        os.environ["LADA_VARIABLE_COREAI_SWIFT_RUNNER"] = previous_runner
            print(f"Core AI smoke passed: {name} -> {path.name}", flush=True)
        else:
            print(f"Core AI asset passed: {name} -> {path.name}", flush=True)


def configure_environment(
    resources: Path,
    distribution: str,
    architecture: str,
) -> None:
    os.environ["LADA_MODEL_WEIGHTS_DIR"] = str(resources / "models")
    if distribution == "dedicated":
        if architecture != "h17s":
            raise RuntimeError(
                f"standalone Core AI architecture must be h17s, got {architecture}"
            )
        os.environ["LADA_COREAI_ARCHITECTURE"] = architecture
        os.environ.setdefault(
            "LADA_COREAI_SWIFT_RUNNER",
            str(resources / "bin" / "lada-coreai-runner"),
        )
        os.environ.setdefault(
            "LADA_VARIABLE_COREAI_SWIFT_RUNNER",
            str(resources / "bin" / "lada-basicvsrpp-variable-runner"),
        )
        os.environ.setdefault(
            "LADA_VARIABLE_COREAI_HQ_SWIFT_RUNNER",
            str(resources / "bin" / "lada-basicvsrpp-variable-hq-runner"),
        )
    elif distribution == "portable":
        os.environ.pop("LADA_COREAI_ARCHITECTURE", None)
        os.environ.pop("LADA_COREAI_SWIFT_RUNNER", None)
        os.environ.setdefault(
            "LADA_VARIABLE_COREAI_SWIFT_RUNNER",
            str(resources / "bin" / "lada-basicvsrpp-variable-runner"),
        )
    else:
        raise ValueError(f"unsupported Core AI distribution: {distribution}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--distribution",
        choices=("dedicated", "portable"),
        default="dedicated",
    )
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument(
        "--smoke-model",
        action="append",
        choices=tuple(MODEL_CONTRACTS)
        + (VARIABLE_MODEL_NAME, VARIABLE_HQ_MODEL_NAME),
    )
    args = parser.parse_args()
    resources = args.resources.resolve()
    configure_environment(resources, args.distribution, args.architecture)
    verify_models(
        resources,
        distribution=args.distribution,
        architecture=args.architecture,
        smoke_names=set(args.smoke_model) if args.smoke_model is not None else None,
    )


if __name__ == "__main__":
    main()
