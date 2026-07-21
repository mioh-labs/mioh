# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Evaluate one deterministic held-out clip from a MiohRestorer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lada.models.mioh_restorer import MiohRestorerV2, MiohRestorerV3, masked_psnr
from lada.models.mioh_restorer.training_dataset import MiohRestorationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--weights", choices=("ema", "raw"), default="ema"
    )
    parser.add_argument(
        "--degrade", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def build_model(
    payload: dict, weights: str
) -> MiohRestorerV2 | MiohRestorerV3:
    config = payload["config"]
    version = int(config.get("version", 0))
    if version == 2:
        model = MiohRestorerV2(
            window_frames=int(config["window_frames"]),
            chunk_frames=int(config["chunk_frames"]),
            channels=int(config["channels"]),
            num_blocks=int(config["num_blocks"]),
            fusion_full_channels=int(config["fusion_full_channels"]),
            fusion_half_channels=int(config["fusion_half_channels"]),
            fusion_quarter_channels=int(config["fusion_quarter_channels"]),
            detail_scale=float(config["detail_scale"]),
        )
    elif version == 3:
        model = MiohRestorerV3(
            window_frames=int(config["window_frames"]),
            channels=int(config["channels"]),
            num_blocks=int(config["num_blocks"]),
            encoder_blocks=int(config["encoder_blocks"]),
            reconstruction_blocks=int(config["reconstruction_blocks"]),
            alignment_radius=int(config["alignment_radius"]),
            first_order_dilation=int(config["first_order_dilation"]),
            second_order_dilation=int(config["second_order_dilation"]),
            alignment_key_channels=int(config["alignment_key_channels"]),
            alignment_groups=int(config.get("alignment_groups", 1)),
            hierarchical_alignment_dilations=tuple(
                int(item)
                for item in config.get(
                    "hierarchical_alignment_dilations", []
                )
            ),
            alignment_temperature=float(
                config.get("alignment_temperature", 1.0)
            ),
            detail_scale=float(config["detail_scale"]),
        )
    else:
        raise ValueError("checkpoint is not MiohRestorerV2/V3")
    state_key = "ema_state_dict" if weights == "ema" else "state_dict"
    model.load_state_dict(payload[state_key], strict=True)
    return model.eval()


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    weights = masks.expand_as(prediction)
    return float(((prediction - target).abs() * weights).sum() / weights.sum().clamp_min(1))


def temporal_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    pair_masks = torch.maximum(masks[:, 1:], masks[:, :-1]).expand_as(prediction_delta)
    squared = (prediction_delta - target_delta).square() * pair_masks
    return float((squared.sum() / pair_masks.sum().clamp_min(1)).sqrt())


def tensor_frame(values: torch.Tensor, frame_index: int) -> np.ndarray:
    frame = values[0, frame_index].detach().float().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def label_image(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def make_comparison(
    inputs: torch.Tensor,
    restored: torch.Tensor,
    targets: torch.Tensor,
) -> np.ndarray:
    frame_count = inputs.shape[1]
    indices = sorted({0, frame_count // 3, 2 * frame_count // 3, frame_count - 1})
    rows = []
    for name, values in (
        ("Mosaic input", inputs),
        ("Restored", restored),
        ("Ground truth", targets),
    ):
        row = [
            label_image(tensor_frame(values, index), f"{name}  frame {index}")
            for index in indices
        ]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("sample-index cannot be negative")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = build_model(payload, args.weights)
    device = torch.device(args.device)
    model.to(device)
    dataset = MiohRestorationDataset(
        [args.metadata_root],
        sequence_frames=int(config["window_frames"]),
        image_size=int(config["image_size"]),
        degrade=args.degrade,
        horizontal_flip=False,
        deterministic=True,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(
            f"sample-index {args.sample_index} is outside dataset of {len(dataset)}"
        )
    sample = dataset[args.sample_index]
    inputs = sample["inputs"].unsqueeze(0).to(device)
    targets = sample["targets"].unsqueeze(0).to(device)
    masks = sample["masks"].unsqueeze(0).to(device)
    with torch.inference_mode():
        restored = model(inputs, masks)

    # Compute metrics from stable CPU snapshots. Repeated small MPS reductions
    # can occasionally observe mismatched asynchronous temporary buffers while
    # training is using the GPU concurrently.
    metric_inputs = inputs.detach().float().cpu()
    metric_targets = targets.detach().float().cpu()
    metric_masks = masks.detach().float().cpu()
    metric_restored = restored.detach().float().cpu()
    input_roi_psnr = float(
        masked_psnr(metric_inputs, metric_targets, metric_masks)
    )
    restored_roi_psnr = float(
        masked_psnr(metric_restored, metric_targets, metric_masks)
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "step": int(payload.get("step", 0)),
        "weights": args.weights,
        "sample_index": args.sample_index,
        "sample_name": sample["name"],
        "frames": int(inputs.shape[1]),
        "image_size": int(inputs.shape[-1]),
        "metrics": {
            "input_roi_psnr": input_roi_psnr,
            "restored_roi_psnr": restored_roi_psnr,
            "roi_psnr_gain": restored_roi_psnr - input_roi_psnr,
            "input_roi_mae": masked_mae(
                metric_inputs, metric_targets, metric_masks
            ),
            "restored_roi_mae": masked_mae(
                metric_restored, metric_targets, metric_masks
            ),
            "input_temporal_rmse": temporal_rmse(
                metric_inputs, metric_targets, metric_masks
            ),
            "restored_temporal_rmse": temporal_rmse(
                metric_restored, metric_targets, metric_masks
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    comparison_path = args.output_dir / "comparison.png"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    comparison = make_comparison(metric_inputs, metric_restored, metric_targets)
    cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
