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
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        help="optional parent checkpoint for direct V4.1-vs-V4 comparisons",
    )
    parser.add_argument(
        "--baseline-weights", choices=("ema", "raw"), default="ema"
    )
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
    parser.add_argument(
        "--motion-thresholds",
        type=float,
        nargs=3,
        metavar=("STATIC", "LOW", "MEDIUM"),
        default=(1.0, 3.0, 5.0),
        help="ROI p75 optical-flow thresholds in 384px input pixels/frame",
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
        high_resolution_detail=bool(config.get("high_resolution_detail", False)),
        detail_full_channels=int(config.get("detail_full_channels", 32)),
        detail_half_channels=int(config.get("detail_half_channels", 48)),
        detail_fusion_channels=int(config.get("detail_fusion_channels", 64)),
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


MOTION_BUCKETS = ("static", "low", "medium", "high")


def roi_motion_pixels(
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Return the p75 dense-flow magnitude inside the moving ROI.

    Target frames are used rather than mosaic inputs so block edges do not
    inflate the motion estimate.  The metric is diagnostic only and never
    enters training.
    """

    if target.shape[:3] != (1, 5, 3) or mask.shape[:3] != (1, 5, 1):
        raise ValueError("motion bucketing expects one five-frame RGB sample")
    frames = (
        target[0]
        .detach()
        .float()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    masks = mask[0, :, 0].detach().float().cpu().numpy() > 0.5
    magnitudes: list[np.ndarray] = []
    for index in range(4):
        first = cv2.cvtColor(
            np.round(frames[index] * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        second = cv2.cvtColor(
            np.round(frames[index + 1] * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude = np.linalg.norm(flow, axis=2)
        roi = np.logical_or(masks[index], masks[index + 1])
        if roi.any():
            magnitudes.append(magnitude[roi])
    if not magnitudes:
        return 0.0
    return float(np.percentile(np.concatenate(magnitudes), 75))


def motion_bucket(value: float, thresholds: tuple[float, float, float]) -> str:
    if value < thresholds[0]:
        return "static"
    if value < thresholds[1]:
        return "low"
    if value < thresholds[2]:
        return "medium"
    return "high"


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
    thresholds = tuple(float(value) for value in args.motion_thresholds)
    if thresholds[0] < 0 or not thresholds[0] < thresholds[1] < thresholds[2]:
        raise ValueError("motion-thresholds must be non-negative and increasing")
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
    baseline_name = None
    if args.baseline_checkpoint is not None:
        baseline_payload = torch.load(
            args.baseline_checkpoint, map_location="cpu", weights_only=False
        )
        baseline_name = f"baseline_{args.baseline_weights}"
        models[baseline_name] = model_from_payload(
            baseline_payload,
            keys[args.baseline_weights],
            device,
        )
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
    totals.update({name: {} for name in models})
    bucket_counts = {name: 0 for name in MOTION_BUCKETS}
    bucket_motion = {name: 0.0 for name in MOTION_BUCKETS}
    bucket_totals = {
        bucket: {name: {} for name in models}
        for bucket in MOTION_BUCKETS
    }
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
            motion = roi_motion_pixels(targets, output_masks)
            bucket = motion_bucket(motion, thresholds)
            bucket_counts[bucket] += 1
            bucket_motion[bucket] += motion
            for name, model in models.items():
                restored, confidence = model(values)
                sample_metrics = metrics(
                    restored, targets, output_masks, confidence
                )
                add_metrics(totals[name], sample_metrics)
                add_metrics(bucket_totals[bucket][name], sample_metrics)
            count += 1
            if count >= args.batches:
                break
    averaged = {
        name: {key: value / count for key, value in values.items()}
        for name, values in totals.items()
    }
    bucket_averaged = {
        bucket: {
            "count": bucket_counts[bucket],
            "mean_roi_flow_p75_pixels": (
                bucket_motion[bucket] / bucket_counts[bucket]
                if bucket_counts[bucket]
                else None
            ),
            "metrics": {
                name: {
                    key: value / bucket_counts[bucket]
                    for key, value in values.items()
                }
                for name, values in bucket_totals[bucket].items()
            },
        }
        for bucket in MOTION_BUCKETS
    }
    static_low_count = bucket_counts["static"] + bucket_counts["low"]
    static_low_totals = {
        name: {
            key: bucket_totals["static"][name].get(key, 0.0)
            + bucket_totals["low"][name].get(key, 0.0)
            for key in set(bucket_totals["static"][name])
            | set(bucket_totals["low"][name])
        }
        for name in models
    }
    static_low_metrics = {
        name: {
            key: value / static_low_count
            for key, value in values.items()
        }
        for name, values in static_low_totals.items()
    } if static_low_count else {name: {} for name in models}
    if baseline_name is not None:
        for bucket, values in bucket_averaged.items():
            if not values["count"]:
                values["improvement_vs_baseline"] = {}
                continue
            bucket_metrics = values["metrics"]
            values["improvement_vs_baseline"] = {
                name: improvement_percentages(
                    bucket_metrics[baseline_name], bucket_metrics[name]
                )
                for name in selected
            }
        static_low_improvement = (
            {
                name: improvement_percentages(
                    static_low_metrics[baseline_name], static_low_metrics[name]
                )
                for name in selected
            }
            if static_low_count
            else {}
        )
        bootstrap_gate = {}
        for name in selected:
            if not static_low_count:
                bootstrap_gate[name] = {
                    "status": "insufficient_static_low_samples"
                }
                continue
            hf_reduction = static_low_improvement[name][
                "high_frequency_error_reduction_percent"
            ]
            roi_psnr_delta = (
                static_low_metrics[name]["roi_psnr"]
                - static_low_metrics[baseline_name]["roi_psnr"]
            )
            populated_hf_changes = [
                values["improvement_vs_baseline"][name][
                    "high_frequency_error_reduction_percent"
                ]
                for values in bucket_averaged.values()
                if values["count"]
            ]
            if roi_psnr_delta < -0.3:
                status = "fail_fidelity_guard"
            elif hf_reduction >= 5.0:
                status = "pass_strong_low_motion_signal"
            elif populated_hf_changes and min(populated_hf_changes) > 0:
                status = "pass_small_all_motion_signal"
            else:
                status = "diagnose_detail_path_or_loss_before_rejecting_structure"
            bootstrap_gate[name] = {
                "status": status,
                "static_low_hf_error_reduction_percent": hf_reduction,
                "static_low_roi_psnr_delta_db": roi_psnr_delta,
                "required_strong_hf_reduction_percent": 5.0,
                "minimum_roi_psnr_delta_db": -0.3,
            }
    else:
        static_low_improvement = {}
        bootstrap_gate = {}
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "step": int(payload.get("step", 0)),
        "batches": count,
        "mosaic_size_multiplier": args.mosaic_size_multiplier,
        "motion_bucketing": {
            "metric": "ROI p75 Farneback flow magnitude on clean target",
            "units": "input pixels per frame",
            "thresholds": {
                "static_below": thresholds[0],
                "low_below": thresholds[1],
                "medium_below": thresholds[2],
                "high_at_or_above": thresholds[2],
            },
            "buckets": bucket_averaged,
            "static_low_combined": {
                "count": static_low_count,
                "metrics": static_low_metrics,
                "improvement_vs_baseline": static_low_improvement,
            },
            "bootstrap_gate": bootstrap_gate,
        },
        "metrics": averaged,
        "improvement_vs_mosaic_input": {
            name: improvement_percentages(averaged["mosaic_input"], averaged[name])
            for name in selected
        },
    }
    if baseline_name is not None:
        report["baseline_checkpoint"] = str(
            args.baseline_checkpoint.resolve()
        )
        report["improvement_vs_baseline"] = {
            name: improvement_percentages(averaged[baseline_name], averaged[name])
            for name in selected
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
