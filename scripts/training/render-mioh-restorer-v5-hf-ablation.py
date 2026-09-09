#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Render an honest visual comparison for the bounded V5-HQ HF ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lada.models.mioh_restorer.losses_v5 import high_frequency, masked_correlation, masked_mean
from lada.models.mioh_restorer.model_v5_hq import MiohRestorerV5HQ
from lada.models.mioh_restorer.native_dataset_v5 import MiohRestorerV5NativeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def tensor_image(values: torch.Tensor) -> Image.Image:
    array = (
        values.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, "RGB")


def roi_bounds(mask: torch.Tensor, *, padding: int = 16) -> tuple[int, int, int, int]:
    array = mask.detach().float().squeeze().cpu().numpy() > 0.05
    ys, xs = np.nonzero(array)
    height, width = array.shape
    if not len(xs):
        return 0, 0, width, height
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(width, int(xs.max()) + padding + 1)
    bottom = min(height, int(ys.max()) + padding + 1)
    side = max(right - left, bottom - top)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    left = max(0, min(width - side, center_x - side // 2))
    top = max(0, min(height - side, center_y - side // 2))
    return left, top, min(width, left + side), min(height, top + side)


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def compose_sheet(
    panels: list[Image.Image],
    *,
    bounds: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> Image.Image:
    labels = ("Mosaic input", "Current dedicated", "HF ablation raw", "Ground truth")
    panel_size = 384
    gap = 12
    label_height = 36
    title_height = 86
    width = gap + 4 * (panel_size + gap)
    height = title_height + 2 * (label_height + panel_size + gap)
    sheet = Image.new("RGB", (width, height), "#17191c")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), title, fill="white", font=font(24))
    draw.text((gap, 45), subtitle, fill="#c7ccd3", font=font(16))
    for column, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = gap + column * (panel_size + gap)
        draw.text((x, title_height), label, fill="white", font=font(17))
        overview = panel.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        sheet.paste(overview, (x, title_height + label_height))
        draw.text(
            (x, title_height + label_height + panel_size + gap),
            "ROI zoom",
            fill="white",
            font=font(17),
        )
        cropped = panel.crop(bounds).resize(
            (panel_size, panel_size), Image.Resampling.LANCZOS
        )
        sheet.paste(
            cropped,
            (x, title_height + 2 * label_height + panel_size + gap),
        )
    return sheet


@torch.no_grad()
def main() -> int:
    args = parse_args()
    for path in (args.manifest, args.backbone, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = MiohRestorerV5NativeDataset(
        args.manifest,
        output_indices=(2, 3, 4, 5, 6),
        degrade=False,
        horizontal_flip=False,
        time_reverse=False,
        deterministic=True,
        mosaic_block_size_range=(6.0, 12.0),
    )
    model = MiohRestorerV5HQ()
    model.load_basicvsrpp_checkpoint(args.backbone)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(payload.get("step", -1))
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise TypeError("ablation checkpoint has no raw state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    records: list[dict[str, object]] = []
    rendered: dict[int, tuple[list[Image.Image], tuple[int, int, int, int]]] = {}
    output_indices = list(model.config.output_indices)
    center = len(output_indices) // 2
    for index in range(min(args.samples, len(dataset))):
        sample = dataset[index]
        inputs = sample["inputs"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        targets = sample["targets"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        masks = sample["masks"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        loss_masks = sample["loss_masks"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        restored, _confidence = model(inputs)
        source = inputs[:, output_indices, :3]
        backbone_raw = model.backbone(inputs[:, :, :3])[:, output_indices]
        compositor_masks = inputs[:, output_indices, 3:4].clamp(0, 1)
        backbone = source + compositor_masks * (backbone_raw - source)
        target_hf = high_frequency(targets.float())
        raw_hf = high_frequency(restored.float())
        backbone_hf = high_frequency(backbone.float())
        raw_mse = masked_mean((restored.float() - targets).square(), loss_masks)
        backbone_mse = masked_mean((backbone.float() - targets).square(), loss_masks)
        target_amplitude = torch.sqrt(
            masked_mean(target_hf.square(), loss_masks).clamp_min(1e-12)
        )
        raw_amplitude = torch.sqrt(
            masked_mean(raw_hf.square(), loss_masks).clamp_min(1e-12)
        )
        backbone_amplitude = torch.sqrt(
            masked_mean(backbone_hf.square(), loss_masks).clamp_min(1e-12)
        )
        record = {
            "index": index,
            "name": sample["name"],
            "source_video_id": sample["source_video_id"],
            "raw_psnr": float(-10 * torch.log10(raw_mse.clamp_min(1e-12))),
            "backbone_psnr": float(-10 * torch.log10(backbone_mse.clamp_min(1e-12))),
            "raw_hf_ratio": float(raw_amplitude / target_amplitude.clamp_min(1e-6)),
            "backbone_hf_ratio": float(
                backbone_amplitude / target_amplitude.clamp_min(1e-6)
            ),
            "raw_hf_correlation": float(masked_correlation(raw_hf, target_hf, loss_masks)),
            "backbone_hf_correlation": float(
                masked_correlation(backbone_hf, target_hf, loss_masks)
            ),
        }
        record["psnr_delta"] = float(record["raw_psnr"]) - float(record["backbone_psnr"])
        record["correlation_delta"] = float(record["raw_hf_correlation"]) - float(
            record["backbone_hf_correlation"]
        )
        record["amplitude_delta"] = float(record["raw_hf_ratio"]) - float(
            record["backbone_hf_ratio"]
        )
        records.append(record)
        center_input_index = output_indices[center]
        panels = [
            tensor_image(inputs[0, center_input_index, :3]),
            tensor_image(backbone[0, center]),
            tensor_image(restored[0, center]),
            tensor_image(targets[0, center]),
        ]
        rendered[index] = (panels, roi_bounds(loss_masks[0, center]))

    selected = {
        "worst-psnr": min(records, key=lambda item: float(item["psnr_delta"])),
        "worst-correlation": min(
            records, key=lambda item: float(item["correlation_delta"])
        ),
        "largest-hf-gain": max(
            records, key=lambda item: float(item["amplitude_delta"])
        ),
    }
    for label, record in selected.items():
        index = int(record["index"])
        panels, bounds = rendered[index]
        subtitle = (
            f"sample {index} | PSNR {float(record['backbone_psnr']):.2f} -> "
            f"{float(record['raw_psnr']):.2f} dB | HF ratio "
            f"{float(record['backbone_hf_ratio']):.3f} -> {float(record['raw_hf_ratio']):.3f} | "
            f"corr {float(record['backbone_hf_correlation']):.3f} -> "
            f"{float(record['raw_hf_correlation']):.3f}"
        )
        sheet = compose_sheet(
            panels,
            bounds=bounds,
            title=f"V5-HQ HF ablation checkpoint step {checkpoint_step}: {label}",
            subtitle=subtitle,
        )
        sheet.save(args.output_dir / f"{label}.png", compress_level=3)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"records": records, "selected": selected}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value["index"] for key, value in selected.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
