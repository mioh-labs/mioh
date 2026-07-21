#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Evaluate raw/EMA V4 weights on mosaic ROI and save a visual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lada.models.mioh_restorer.losses_v4 import confidence_error_correlation
from lada.models.mioh_restorer.model_v4 import MiohRestorerV4Q
from lada.models.mioh_restorer.training_dataset import MiohRestorationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--weights", choices=("ema", "raw", "both"), default="both")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=1,
        help="save evenly spaced visual comparisons from the validation set",
    )
    parser.add_argument(
        "--mosaic-size-multiplier",
        type=float,
        default=1.0,
        help="multiply synthetic mosaic block dimensions for A/B evaluation",
    )
    return parser.parse_args()


def device_for(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_from_payload(payload: dict, key: str, device: torch.device) -> MiohRestorerV4Q:
    config = payload.get("config", {})
    if int(config.get("version", 0)) != 4:
        raise ValueError("checkpoint is not MiohRestorerV4")
    model = MiohRestorerV4Q(
        alignment_variant=str(config["alignment_variant"]),
        execution_mode=str(config["execution_mode"]),
        quarter_channels=int(config["quarter_channels"]),
        eighth_channels=int(config["eighth_channels"]),
        fusion_eighth_channels=int(config["fusion_eighth_channels"]),
        fusion_quarter_channels=int(config["fusion_quarter_channels"]),
        eighth_blocks=int(config["eighth_blocks"]),
        quarter_blocks=int(config["quarter_blocks"]),
    )
    model.load_state_dict(payload[key], strict=True)
    return model.eval().to(device)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.expand_as(prediction)
    return (
        (prediction.float() - target.float()).square().mul(weights).sum()
        / weights.sum().clamp_min(1.0)
    )


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10 * torch.log10(mse.clamp_min(1e-12))


def high_frequency(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    flattened = values.reshape(-1, 3, shape[-2], shape[-1])
    blurred = torch.nn.functional.avg_pool2d(flattened, 3, stride=1, padding=1)
    return (flattened - blurred).reshape(shape)


def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> dict[str, float]:
    roi_mse = masked_mse(prediction, target, mask)
    whole_mse = (prediction.float() - target.float()).square().mean()
    temporal = masked_mse(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        torch.maximum(mask[:, 1:], mask[:, :-1]),
    )
    hf = masked_mse(high_frequency(prediction), high_frequency(target), mask)
    result = {
        "roi_psnr": float(psnr_from_mse(roi_mse)),
        "whole_psnr": float(psnr_from_mse(whole_mse)),
        "temporal_delta_error": float(torch.sqrt(temporal)),
        "roi_hf_rmse": float(torch.sqrt(hf)),
    }
    if confidence is not None:
        error = (prediction - target).abs().mean(dim=2, keepdim=True)
        result["confidence_mean"] = float(
            (confidence * mask).sum() / mask.sum().clamp_min(1.0)
        )
        result["confidence_error_corr"] = float(
            confidence_error_correlation(confidence, error, mask)
        )
    return result


def add_metrics(total: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + value


def improvement_percentages(
    baseline: dict[str, float], candidate: dict[str, float]
) -> dict[str, float]:
    """Express quality changes as intuitive positive-is-better percentages."""

    roi_ratio = 10 ** (
        -(candidate["roi_psnr"] - baseline["roi_psnr"]) / 10
    )
    whole_ratio = 10 ** (
        -(candidate["whole_psnr"] - baseline["whole_psnr"]) / 10
    )
    return {
        "roi_squared_error_reduction_percent": (1 - roi_ratio) * 100,
        "whole_squared_error_reduction_percent": (1 - whole_ratio) * 100,
        "temporal_error_reduction_percent": (
            1
            - candidate["temporal_delta_error"]
            / baseline["temporal_delta_error"]
        )
        * 100,
        "high_frequency_error_reduction_percent": (
            1 - candidate["roi_hf_rmse"] / baseline["roi_hf_rmse"]
        )
        * 100,
    }


def to_image(value: torch.Tensor) -> np.ndarray:
    rgb = value.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(np.round(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def save_preview(
    output: Path,
    input_frame: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    items = [("mosaic input", input_frame), ("clean target", target)]
    items.extend((name, value) for name, value in predictions.items())
    rendered = []
    for label, value in items:
        image = to_image(value)
        cv2.rectangle(image, (0, 0), (190, 30), (0, 0, 0), -1)
        cv2.putText(image, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        rendered.append(image)
    cv2.imwrite(str(output), np.concatenate(rendered, axis=1))


def main() -> int:
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("batches must be positive")
    if args.preview_count <= 0:
        raise ValueError("preview-count must be positive")
    if args.mosaic_size_multiplier <= 0:
        raise ValueError("mosaic-size-multiplier must be positive")
    device = device_for(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    image_size = int(config.get("image_size", 384))
    keys = {
        "raw": "state_dict",
        "ema": "ema_state_dict",
    }
    selected = ("raw", "ema") if args.weights == "both" else (args.weights,)
    models = {
        name: model_from_payload(payload, keys[name], device) for name in selected
    }
    dataset = MiohRestorationDataset(
        args.metadata_root,
        sequence_frames=13,
        image_size=image_size,
        degrade=True,
        horizontal_flip=False,
        deterministic=True,
        mosaic_size_multiplier=args.mosaic_size_multiplier,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    totals: dict[str, dict[str, float]] = {"mosaic_input": {}}
    totals.update({name: {} for name in selected})
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["inputs"][:, :9].to(device)
            targets = batch["targets"][:, 2:7].to(device)
            masks = batch["masks"][:, :9].to(device)
            output_masks = masks[:, 2:7]
            values = torch.cat((inputs, masks), dim=2)
            mosaic = inputs[:, 2:7]
            add_metrics(totals["mosaic_input"], metrics(mosaic, targets, output_masks))
            for name, model in models.items():
                restored, confidence = model(values)
                add_metrics(
                    totals[name], metrics(restored, targets, output_masks, confidence)
                )
            count += 1
            if count >= args.batches:
                break
    averaged = {
        name: {key: value / count for key, value in values.items()}
        for name, values in totals.items()
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": int(payload.get("step", 0)),
        "batches": count,
        "mosaic_size_multiplier": args.mosaic_size_multiplier,
        "metrics": averaged,
        "improvement_vs_mosaic_input": {
            name: improvement_percentages(averaged["mosaic_input"], averaged[name])
            for name in selected
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evaluation-v4.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    preview_count = min(args.preview_count, len(dataset))
    preview_indices = np.linspace(
        0, len(dataset) - 1, num=preview_count, dtype=np.int64
    ).tolist()
    with torch.inference_mode():
        for preview_number, dataset_index in enumerate(preview_indices, start=1):
            sample = dataset[int(dataset_index)]
            inputs = sample["inputs"][:9].unsqueeze(0).to(device)
            targets = sample["targets"][2:7].unsqueeze(0).to(device)
            masks = sample["masks"][:9].unsqueeze(0).to(device)
            values = torch.cat((inputs, masks), dim=2)
            predictions = {
                name: model(values)[0][0, 2] for name, model in models.items()
            }
            filename = (
                "evaluation-v4-preview.png"
                if preview_count == 1
                else f"evaluation-v4-preview-{preview_number:02d}.png"
            )
            save_preview(
                args.output_dir / filename,
                inputs[0, 4],
                targets[0, 2],
                predictions,
            )
    report["preview_dataset_indices"] = preview_indices
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
