#!/usr/bin/env python3
"""Pilot multi-frame 2K->4K reconstruction against a paired 4K video.

This is an evaluation experiment, not an app pipeline. It starts from the
original single-frame PiperSR result, aligns neighboring 2K frames with optical
flow, then uses their observations to refine the 4K image. Only the center
frame's matching 4K video frame is used as ground truth for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from finetune_pipersr_detail import high_frequency, load_coreml_weights


def video_info(path: Path) -> tuple[int, int, float]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return stream["width"], stream["height"], float(Fraction(stream["r_frame_rate"]))


def extract_window(video: Path, start: float, count: int, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"frame-{index:03d}.png" for index in range(count)]
    if not all(path.is_file() for path in paths):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", str(video),
             "-frames:v", str(count), "-start_number", "0", "-y",
             str(directory / "frame-%03d.png")],
            check=True,
        )
    if not all(path.is_file() for path in paths):
        raise ValueError(f"incomplete frame window from {video} at {start:.3f}s")
    return paths


def image(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def similarity(lr: np.ndarray, hr: np.ndarray) -> float:
    a = cv2.resize(lr, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(hr, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def candidate_patches(center: np.ndarray, truth: np.ndarray, other: np.ndarray,
                      size: int, count: int) -> list[tuple[int, int, float, float]]:
    candidates = []
    for y in range(80, center.shape[0] - size - 79, size):
        for x in range(80, center.shape[1] - size - 79, size):
            crop = truth[y * 2:(y + size) * 2, x * 2:(x + size) * 2]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
            texture = float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1)).mean() / 255)
            change = float(np.abs(center[y:y + size, x:x + size].astype(np.float32)
                                  - other[y:y + size, x:x + size]).mean() / 255)
            # Prefer detail with some motion, but reject large changes likely
            # caused by a cut or occlusion.
            score = texture * (0.25 + min(change / 0.02, 1.0))
            if change < 0.12:
                candidates.append((x, y, score, change))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:count]


def to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frame.copy()).permute(2, 0, 1)[None].float().to(device) / 255


def quality(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    # Exclude the 32px LR crop border where optical flow has incomplete context.
    margin = 64
    prediction = prediction[..., margin:-margin, margin:-margin]
    truth = truth[..., margin:-margin, margin:-margin]
    mse = F.mse_loss(prediction, truth).item()
    pred_hf = high_frequency(prediction)
    true_hf = high_frequency(truth)
    ratio = pred_hf.abs().mean().item() / max(true_hf.abs().mean().item(), 1e-9)
    cosine = F.cosine_similarity(pred_hf.flatten(), true_hf.flatten(), dim=0).item()
    return {
        "psnr": -10 * math.log10(max(mse, 1e-12)),
        "hf_mae": F.l1_loss(pred_hf, true_hf).item(),
        "hf_ratio": ratio, "hf_cosine": cosine,
    }


def observation_grid(flow: np.ndarray, hr_size: int, device: torch.device) -> torch.Tensor:
    enlarged = cv2.resize(flow, (hr_size, hr_size), interpolation=cv2.INTER_LINEAR) * 2
    coordinates = np.arange(hr_size, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    grid = np.stack((xx + enlarged[..., 0], yy + enlarged[..., 1]), axis=-1)
    grid[..., 0] = grid[..., 0] * (2 / (hr_size - 1)) - 1
    grid[..., 1] = grid[..., 1] * (2 / (hr_size - 1)) - 1
    return torch.from_numpy(grid[None].copy()).to(device)


def reconstruct(
    baseline: torch.Tensor, observations: list[torch.Tensor],
    flows: list[np.ndarray], confidences: list[torch.Tensor],
    iterations: int, anchor_weight: float,
) -> torch.Tensor:
    device = baseline.device
    hr_size = baseline.shape[-1]
    grids = [observation_grid(flow, hr_size, device) for flow in flows]
    estimate = baseline.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=0.005)
    for _ in range(iterations):
        losses = []
        for observed, grid, confidence in zip(observations, grids, confidences):
            predicted = F.avg_pool2d(
                F.grid_sample(estimate, grid, mode="bilinear", padding_mode="border", align_corners=True),
                kernel_size=2,
            )
            losses.append(((predicted - observed).abs() * confidence).mean())
        # Fidelity to all aligned observations, with the center image retaining
        # weight 1. Other frames are downweighted for optical-flow uncertainty.
        data_loss = losses[len(losses) // 2] + sum(
            loss for index, loss in enumerate(losses) if index != len(losses) // 2
        ) / max(len(losses) - 1, 1) * 0.75
        loss = data_loss + anchor_weight * F.l1_loss(estimate, baseline)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            estimate.clamp_(0, 1)
    return estimate.detach()


def tensor_image(value: torch.Tensor) -> Image.Image:
    pixels = (value[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-video", type=Path, required=True)
    parser.add_argument("--hr-video", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--patches-per-time", type=int, default=2)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--oracle-flow", action="store_true",
                        help="Use clean 4K neighboring frames for flow only; evaluation upper bound, not deployable")
    parser.add_argument("--center-only", action="store_true",
                        help="Control: fit only the center LR observation with the same optimizer")
    args = parser.parse_args()

    lr_w, lr_h, lr_fps = video_info(args.lr_video)
    hr_w, hr_h, hr_fps = video_info(args.hr_video)
    if (hr_w, hr_h) != (lr_w * 2, lr_h * 2) or abs(lr_fps - hr_fps) > 0.01:
        raise ValueError("video resolutions or frame rates are not paired 2K/4K")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # grid_sampler_2d_backward is not implemented by PyTorch MPS here. The
    # small inverse-problem pilot runs on CPU; PiperSR inference still uses MPS.
    reconstruction_device = torch.device("cpu")
    model = load_coreml_weights(args.package).eval().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    sample_indices = list(range(0, args.radius * args.stride * 2 + 1, args.stride))
    center_index = args.radius * args.stride
    records = []
    comparisons = []

    for timestamp in args.times:
        start = timestamp - center_index / lr_fps
        if start < 0:
            raise ValueError("timestamp too close to video start")
        count = center_index * 2 + 1
        directory = args.output / f"time-{timestamp:.3f}"
        lr_paths = extract_window(args.lr_video, start, count, directory / "lr")
        hr_paths = extract_window(args.hr_video, start, count, directory / "hr")
        center = image(lr_paths[center_index])
        truth = image(hr_paths[center_index])
        alignment = similarity(center, truth)
        if alignment < 0.98:
            raise ValueError(f"2K/4K center frames not aligned at {timestamp}: {alignment:.4f}")
        other = image(lr_paths[sample_indices[0]])
        locations = candidate_patches(center, truth, other, args.patch, args.patches_per_time)
        for x, y, _, motion in locations:
            center_crop = center[y:y + args.patch, x:x + args.patch]
            truth_crop = truth[y * 2:(y + args.patch) * 2, x * 2:(x + args.patch) * 2]
            observations = []
            flows = []
            confidences = []
            gray_center = cv2.cvtColor(center_crop, cv2.COLOR_RGB2GRAY)
            for index in sample_indices:
                frame = image(lr_paths[index])[y:y + args.patch, x:x + args.patch]
                observations.append(to_tensor(frame, device))
                if index == center_index:
                    flows.append(np.zeros((args.patch, args.patch, 2), np.float32))
                    confidences.append(torch.ones((1, 1, args.patch, args.patch), device=device))
                    continue
                if args.oracle_flow:
                    hr_neighbor = image(hr_paths[index])[
                        y * 2:(y + args.patch) * 2, x * 2:(x + args.patch) * 2
                    ]
                    flow_before = cv2.cvtColor(hr_neighbor, cv2.COLOR_RGB2GRAY)
                    flow_after = cv2.cvtColor(truth_crop, cv2.COLOR_RGB2GRAY)
                else:
                    flow_before = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    flow_after = gray_center
                flow = cv2.calcOpticalFlowFarneback(
                    flow_before, flow_after, None,
                    pyr_scale=0.5, levels=4, winsize=21, iterations=4,
                    poly_n=7, poly_sigma=1.5, flags=0,
                )
                if args.oracle_flow:
                    flow = cv2.resize(flow, (args.patch, args.patch), interpolation=cv2.INTER_AREA) / 2
                flows.append(flow)
                xx, yy = np.meshgrid(np.arange(args.patch, dtype=np.float32),
                                     np.arange(args.patch, dtype=np.float32))
                aligned_center = cv2.remap(
                    center_crop, xx + flow[..., 0], yy + flow[..., 1],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
                error = np.abs(aligned_center.astype(np.float32) - frame).mean(axis=2) / 255
                confidence = np.exp(-error * 40).astype(np.float32)
                confidences.append(torch.from_numpy(confidence[None, None]).to(device))
            with torch.inference_mode():
                baseline = model(observations[args.radius]).float()
            if args.center_only:
                observations = [observations[args.radius]]
                flows = [flows[args.radius]]
                confidences = [confidences[args.radius]]
            refined = reconstruct(
                baseline.to(reconstruction_device),
                [item.to(reconstruction_device) for item in observations],
                flows,
                [item.to(reconstruction_device) for item in confidences],
                args.iterations, args.anchor_weight,
            ).to(device)
            truth_tensor = to_tensor(truth_crop, device)
            result = {
                "time": timestamp, "x": x, "y": y,
                "frame_alignment": alignment, "motion_mae": motion,
                "baseline": quality(baseline, truth_tensor),
                "multiframe": quality(refined, truth_tensor),
            }
            print(json.dumps(result), flush=True)
            records.append(result)
            if len(comparisons) < 4:
                comparisons.append((tensor_image(baseline), tensor_image(refined), Image.fromarray(truth_crop)))

    if not records:
        raise ValueError("no suitable textured patches were found")
    summary = {
        "sample_count": len(records),
        "baseline": {key: float(np.mean([record["baseline"][key] for record in records]))
                     for key in ("psnr", "hf_mae", "hf_ratio", "hf_cosine")},
        "multiframe": {key: float(np.mean([record["multiframe"][key] for record in records]))
                       for key in ("psnr", "hf_mae", "hf_ratio", "hf_cosine")},
        "psnr_wins": sum(record["multiframe"]["psnr"] > record["baseline"]["psnr"]
                         for record in records),
        "hf_cosine_wins": sum(record["multiframe"]["hf_cosine"] > record["baseline"]["hf_cosine"]
                              for record in records),
    }
    report = {"lr_video": str(args.lr_video), "hr_video": str(args.hr_video),
              "parameters": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
              "samples": records, "summary": summary}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary: {json.dumps(summary)}", flush=True)
    size = comparisons[0][0].width
    canvas = Image.new("RGB", (size * 3, size * len(comparisons) + 22), "#202020")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("PiperSR", "7-frame reconstruction", "4K reference")):
        draw.text((column * size + 4, 4), label, fill="white")
    for row, triplet in enumerate(comparisons):
        for column, sample in enumerate(triplet):
            canvas.paste(sample, (column * size, 22 + row * size))
    canvas.save(args.output / "comparison.png")


if __name__ == "__main__":
    main()
