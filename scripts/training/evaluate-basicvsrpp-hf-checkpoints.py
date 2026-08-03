#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Compare faithful BasicVSR++ HF checkpoints sample by sample.

The aggregate MMEngine validation metrics are useful promotion guards, but a
small average can hide a few regressions.  This tool evaluates every manifest
entry deterministically, records per-sample ROI fidelity/HF/temporal metrics,
and renders synchronized full-crop and ROI-detail comparison videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config

from lada.models.basicvsrpp import register_all_modules
from lada.models.basicvsrpp.mmagic.registry import DATASETS, MODELS
from lada.models.basicvsrpp.mmagic.roi_laplacian_error import (
    roi_laplacian_error,
)
from lada.models.basicvsrpp.mmagic.roi_psnr import roi_psnr


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "checkpoint must be LABEL=/absolute/path/to/checkpoint.pth"
        )
    return label.strip(), Path(path).expanduser().resolve()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def load_generator(config: Config, checkpoint: Path, device: torch.device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint has no state_dict mapping: {checkpoint}")

    prefixes = ("generator_ema.", "generator.")
    selected = None
    selected_prefix = None
    for prefix in prefixes:
        candidate = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if candidate:
            selected = candidate
            selected_prefix = prefix
            break
    if selected is None:
        raise ValueError(
            f"checkpoint has neither generator_ema nor generator weights: {checkpoint}"
        )

    generator = MODELS.build(config.model.generator)
    generator.load_state_dict(selected, strict=True)
    generator.eval().requires_grad_(False).to(device)
    return generator, selected_prefix


def sequence_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    roi_psnr_values = []
    laplacian_values = []
    for frame in range(target.shape[0]):
        roi_psnr_values.append(
            roi_psnr(target[frame], prediction[frame], mask[frame])
        )
        laplacian_values.append(
            roi_laplacian_error(target[frame], prediction[frame], mask[frame])
        )

    temporal_values = []
    for frame in range(1, target.shape[0]):
        roi = torch.maximum(mask[frame - 1], mask[frame]) > 0
        roi = roi.expand(target.shape[1], -1, -1)
        if not bool(roi.any()):
            continue
        target_delta = target[frame] - target[frame - 1]
        prediction_delta = prediction[frame] - prediction[frame - 1]
        temporal_values.append(
            float(torch.abs(prediction_delta - target_delta)[roi].mean())
        )

    return {
        "roi_psnr": float(np.mean(roi_psnr_values)),
        "roi_laplacian_error": float(np.mean(laplacian_values)),
        "roi_temporal_error": (
            float(np.mean(temporal_values)) if temporal_values else 0.0
        ),
    }


def to_uint8_rgb(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .clamp(0, 255)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def union_roi_box(mask: np.ndarray, *, margin: int = 16) -> tuple[int, int, int, int]:
    union = np.max(mask, axis=0)
    if union.ndim == 3:
        union = np.max(union, axis=-1)
    ys, xs = np.where(union > 0)
    height, width = union.shape
    if not len(xs):
        return 0, 0, width, height

    left = max(0, int(xs.min()) - margin)
    right = min(width, int(xs.max()) + 1 + margin)
    top = max(0, int(ys.min()) - margin)
    bottom = min(height, int(ys.max()) + 1 + margin)
    size = max(96, right - left, bottom - top)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = max(0, min(width - size, center_x - size // 2))
    top = max(0, min(height - size, center_y - size // 2))
    right = min(width, left + size)
    bottom = min(height, top + size)
    return left, top, right, bottom


def labelled_tile(image_rgb: np.ndarray, label: str) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    tile = np.zeros((height + 28, width, 3), dtype=np.uint8)
    tile[28:] = image_rgb
    cv2.putText(
        tile,
        label,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def render_panels(
    samples: list[dict],
    predictions: dict[str, list[np.ndarray]],
    output_dir: Path,
    *,
    fps: int,
) -> dict[str, str]:
    full_frames = output_dir / "full-frames"
    roi_frames = output_dir / "roi-frames"
    center_frames = output_dir / "center-panels"
    full_frames.mkdir(parents=True)
    roi_frames.mkdir(parents=True)
    center_frames.mkdir(parents=True)

    frame_number = 0
    checkpoint_labels = list(predictions)
    for sample_index, sample in enumerate(samples):
        inputs = sample["input_uint8"]
        targets = sample["target_uint8"]
        mask = sample["mask_uint8"]
        left, top, right, bottom = union_roi_box(mask)
        sequence_frames = []
        for temporal_index in range(inputs.shape[0]):
            columns = [("Input", inputs[temporal_index])]
            columns.extend(
                (
                    label,
                    predictions[label][sample_index][temporal_index],
                )
                for label in checkpoint_labels
            )
            columns.append(("GT", targets[temporal_index]))

            full_tiles = []
            roi_tiles = []
            for label, image in columns:
                annotated = image.copy()
                cv2.rectangle(
                    annotated,
                    (left, top),
                    (max(left, right - 1), max(top, bottom - 1)),
                    (255, 255, 0),
                    1,
                )
                full_tiles.append(labelled_tile(annotated, label))

                roi = image[top:bottom, left:right]
                roi = cv2.resize(
                    roi,
                    (256, 256),
                    interpolation=cv2.INTER_NEAREST,
                )
                roi_tiles.append(labelled_tile(roi, label))

            full_panel = np.concatenate(full_tiles, axis=1)
            roi_panel = np.concatenate(roi_tiles, axis=1)
            name = f"{frame_number:06d}.png"
            cv2.imwrite(
                str(full_frames / name),
                cv2.cvtColor(full_panel, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(roi_frames / name),
                cv2.cvtColor(roi_panel, cv2.COLOR_RGB2BGR),
            )
            sequence_frames.append((full_panel, roi_panel))
            frame_number += 1

        center = inputs.shape[0] // 2
        full_center, roi_center = sequence_frames[center]
        cv2.imwrite(
            str(center_frames / f"sample-{sample_index:02d}-full.png"),
            cv2.cvtColor(full_center, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(center_frames / f"sample-{sample_index:02d}-roi.png"),
            cv2.cvtColor(roi_center, cv2.COLOR_RGB2BGR),
        )

        # Hold the final image briefly so adjacent independent clips are not
        # mistaken for temporal flicker.
        hold_count = max(1, fps // 2)
        for _ in range(hold_count):
            full_panel, roi_panel = sequence_frames[-1]
            name = f"{frame_number:06d}.png"
            cv2.imwrite(
                str(full_frames / name),
                cv2.cvtColor(full_panel, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(roi_frames / name),
                cv2.cvtColor(roi_panel, cv2.COLOR_RGB2BGR),
            )
            frame_number += 1

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to encode comparison videos")

    outputs = {}
    for label, frames in (("full", full_frames), ("roi", roi_frames)):
        video = output_dir / f"comparison-{label}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames / "%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        outputs[label] = str(video)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=parse_checkpoint,
        help="repeatable LABEL=/absolute/checkpoint.pth",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()

    if not args.trust_checkpoint:
        parser.error("trusted project checkpoints require --trust-checkpoint")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            parser.error(f"output directory exists: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    checkpoints = OrderedDict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        parser.error("checkpoint labels must be unique")

    register_all_modules()
    config = Config.fromfile(str(args.config.expanduser().resolve()))
    dataset_config = dict(config.val_dataloader.dataset)
    dataset_config["manifest"] = str(args.manifest.expanduser().resolve())
    dataset = DATASETS.build(dataset_config)
    device = resolve_device(args.device)

    samples = []
    for index in range(len(dataset)):
        item = dataset[index]
        data_sample = item["data_samples"]
        mask = data_sample.mask.detach().cpu().float()
        samples.append(
            {
                "index": index,
                "name": data_sample.metainfo["name"],
                "source_video_id": data_sample.metainfo["source_video_id"],
                "inputs": item["inputs"].detach().cpu(),
                "target": data_sample.gt_img.detach().cpu().float(),
                "mask": mask,
                "input_uint8": to_uint8_rgb(item["inputs"]),
                "target_uint8": to_uint8_rgb(data_sample.gt_img),
                "mask_uint8": (
                    mask.mul(255)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .numpy()
                ),
            }
        )

    predictions: dict[str, list[np.ndarray]] = {}
    report = {
        "format_version": 1,
        "config": str(args.config.expanduser().resolve()),
        "manifest": str(args.manifest.expanduser().resolve()),
        "device": str(device),
        "sample_count": len(samples),
        "checkpoints": {},
    }
    for label, checkpoint in checkpoints.items():
        generator, prefix = load_generator(config, checkpoint, device)
        per_sample = []
        rendered = []
        for sample in samples:
            inputs = sample["inputs"].unsqueeze(0).float().div_(255).to(device)
            with torch.inference_mode():
                prediction = generator(inputs)[0].clamp(0, 1).mul(255).cpu()
            metrics = sequence_metrics(
                prediction,
                sample["target"],
                sample["mask"],
            )
            per_sample.append(
                {
                    "index": sample["index"],
                    "name": sample["name"],
                    "source_video_id": sample["source_video_id"],
                    **metrics,
                }
            )
            rendered.append(to_uint8_rgb(prediction))
        predictions[label] = rendered
        aggregate = {
            key: float(np.mean([item[key] for item in per_sample]))
            for key in (
                "roi_psnr",
                "roi_laplacian_error",
                "roi_temporal_error",
            )
        }
        report["checkpoints"][label] = {
            "path": str(checkpoint),
            "state_prefix": prefix,
            "aggregate": aggregate,
            "samples": per_sample,
        }
        del generator
        if device.type == "mps":
            torch.mps.empty_cache()

    baseline_label = next(iter(checkpoints))
    baseline_samples = report["checkpoints"][baseline_label]["samples"]
    for label in list(checkpoints)[1:]:
        candidate_samples = report["checkpoints"][label]["samples"]
        for baseline, candidate in zip(
            baseline_samples, candidate_samples, strict=True
        ):
            candidate["delta_vs_baseline"] = {
                "roi_psnr": candidate["roi_psnr"] - baseline["roi_psnr"],
                "roi_laplacian_error": (
                    candidate["roi_laplacian_error"]
                    - baseline["roi_laplacian_error"]
                ),
                "roi_temporal_error": (
                    candidate["roi_temporal_error"]
                    - baseline["roi_temporal_error"]
                ),
            }

    videos = render_panels(samples, predictions, output_dir, fps=args.fps)
    report["videos"] = videos
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "per-sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "checkpoint",
                "index",
                "source_video_id",
                "name",
                "roi_psnr",
                "roi_laplacian_error",
                "roi_temporal_error",
            )
        )
        for label, checkpoint_report in report["checkpoints"].items():
            for sample in checkpoint_report["samples"]:
                writer.writerow(
                    (
                        label,
                        sample["index"],
                        sample["source_video_id"],
                        sample["name"],
                        sample["roi_psnr"],
                        sample["roi_laplacian_error"],
                        sample["roi_temporal_error"],
                    )
                )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"metrics: {metrics_path}")
    print(f"per-sample CSV: {csv_path}")
    print(f"full comparison: {videos['full']}")
    print(f"ROI comparison: {videos['roi']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
