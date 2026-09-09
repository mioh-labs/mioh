#!/usr/bin/env python3
"""Render a no-resize BasicVSR++ ROI checkpoint comparison."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--reference-label", default="Reference")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--crop-x", required=True, type=int)
    parser.add_argument("--crop-y", required=True, type=int)
    parser.add_argument("--crop-size", default=512, type=int)
    parser.add_argument("--frames", default=18, type=int)
    parser.add_argument("--center-frame", default=9, type=int)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    return parser.parse_args()


def read_crop_sequence(args: argparse.Namespace) -> tuple[list[np.ndarray], torch.Tensor]:
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {args.source}")
    crops: list[np.ndarray] = []
    for frame_index in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {frame_index}: {args.source}")
        x0, y0, size = args.crop_x, args.crop_y, args.crop_size
        crop = frame[y0 : y0 + size, x0 : x0 + size]
        if crop.shape[:2] != (size, size):
            raise RuntimeError(
                f"crop is outside frame {frame.shape[:2]}: "
                f"x={x0}, y={y0}, size={size}"
            )
        crops.append(crop.copy())
    capture.release()

    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in crops])
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0)
    return crops, tensor


def checkpoint_generator_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unexpected checkpoint payload: {path}")
    state = checkpoint.get("state_dict", checkpoint)
    for prefix in ("generator_ema.", "generator."):
        selected = {
            str(key)[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        if selected:
            return selected
    raise ValueError(f"generator weights not found: {path}")


def restore_center_frame(
    checkpoint: Path,
    sequence: torch.Tensor,
    center_frame: int,
    device: torch.device,
) -> np.ndarray:
    model = BasicVSRPlusPlusGanNet(
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
    )
    model.load_state_dict(checkpoint_generator_state(checkpoint), strict=True)
    model.requires_grad_(False).eval().to(device)
    with torch.inference_mode():
        prediction = model(sequence.to(device))[0, center_frame]
    rgb = (
        prediction.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    del prediction, model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def labeled_panel(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    header_height = 36
    panels = []
    for image, label in zip(images, labels, strict=True):
        panel = np.zeros((image.shape[0] + header_height, image.shape[1], 3), np.uint8)
        panel[header_height:] = image
        cv2.putText(
            panel,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (238, 238, 238),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write image: {path}")


def main() -> None:
    args = parse_args()
    if not 0 <= args.center_frame < args.frames:
        raise ValueError("--center-frame must be inside --frames")
    for path in (args.source, args.reference_checkpoint, args.candidate_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    source_frames, sequence = read_crop_sequence(args)
    device = torch.device(args.device)
    reference = restore_center_frame(
        args.reference_checkpoint, sequence, args.center_frame, device
    )
    candidate = restore_center_frame(
        args.candidate_checkpoint, sequence, args.center_frame, device
    )
    source = source_frames[args.center_frame]

    source_path = args.output_directory / f"source-native{args.crop_size}.png"
    reference_path = args.output_directory / f"reference-native{args.crop_size}.png"
    candidate_path = args.output_directory / f"candidate-native{args.crop_size}.png"
    panel_path = args.output_directory / f"comparison-native{args.crop_size}.png"
    write_png(source_path, source)
    write_png(reference_path, reference)
    write_png(candidate_path, candidate)
    write_png(
        panel_path,
        labeled_panel(
            [source, reference, candidate],
            [
                f"Source mosaic native{args.crop_size}",
                f"{args.reference_label} native{args.crop_size}",
                f"{args.candidate_label} native{args.crop_size}",
            ],
        ),
    )
    print(panel_path)


if __name__ == "__main__":
    main()
