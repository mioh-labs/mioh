#!/usr/bin/env python3
"""Compare visible detail and motion-compensated flicker on short 2K/4K clips.

The clean 4K frames are used only for scoring and for the evaluation flow;
inference uses the 2K frames. This is a diagnostic pilot, not an app model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from finetune_pipersr_detail import PiperSR, load_coreml_weights
from test_pipersr_multiframe import (
    candidate_patches, extract_window, image, quality, reconstruct,
    similarity, to_tensor, video_info,
)


def frame_tensors(
    lr_paths: list[Path], center_index: int, offsets: list[int],
    x: int, y: int, patch: int, device: torch.device,
) -> tuple[list[torch.Tensor], list[np.ndarray], list[torch.Tensor]]:
    center = image(lr_paths[center_index])[y:y + patch, x:x + patch]
    gray_center = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    observations = []
    flows = []
    confidences = []
    xx, yy = np.meshgrid(np.arange(patch, dtype=np.float32),
                         np.arange(patch, dtype=np.float32))
    for offset in offsets:
        frame = image(lr_paths[center_index + offset])[y:y + patch, x:x + patch]
        observations.append(to_tensor(frame, device))
        if offset == 0:
            flows.append(np.zeros((patch, patch, 2), np.float32))
            confidences.append(torch.ones((1, 1, patch, patch), device=device))
            continue
        gray_neighbor = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_neighbor, gray_center, None,
            pyr_scale=0.5, levels=4, winsize=21, iterations=4,
            poly_n=7, poly_sigma=1.5, flags=0,
        )
        flows.append(flow)
        aligned = cv2.remap(center, xx + flow[..., 0], yy + flow[..., 1],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        error = np.abs(aligned.astype(np.float32) - frame).mean(axis=2) / 255
        confidences.append(torch.from_numpy(np.exp(-error * 40)[None, None].astype(np.float32)).to(device))
    return observations, flows, confidences


def rgb_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)


def temporal_scores(previous: np.ndarray, current: np.ndarray,
                    previous_truth: np.ndarray, current_truth: np.ndarray) -> dict[str, float]:
    # Motion estimated from the real 4K pair for scoring only. Warp previous
    # output into the current frame before comparing changes.
    gray_current = cv2.cvtColor(current_truth, cv2.COLOR_RGB2GRAY)
    gray_previous = cv2.cvtColor(previous_truth, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_current, gray_previous, None,
        pyr_scale=0.5, levels=4, winsize=21, iterations=4,
        poly_n=7, poly_sigma=1.5, flags=0,
    )
    height, width = gray_current.shape
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    def warp(frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, xx + flow[..., 0], yy + flow[..., 1],
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    margin = 64
    inner = np.s_[margin:-margin, margin:-margin]
    output_delta = (current - warp(previous))[inner]
    truth_delta = (current_truth - warp(previous_truth))[inner]
    highpass = lambda value: value - cv2.GaussianBlur(value, (0, 0), 1)
    output_detail_delta = highpass(current)[inner] - highpass(warp(previous))[inner]
    truth_detail_delta = highpass(current_truth)[inner] - highpass(warp(previous_truth))[inner]
    return {
        "motion_compensated_flicker": float(np.abs(output_delta).mean()),
        "detail_flicker": float(np.abs(output_detail_delta).mean()),
        "temporal_error_to_4k": float(np.abs(output_delta - truth_delta).mean()),
        "detail_temporal_error_to_4k": float(np.abs(output_detail_delta - truth_detail_delta).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-video", type=Path, required=True)
    parser.add_argument("--hr-video", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint", type=Path,
                        help="Compare a fine-tuned single-frame model instead of multi-frame reconstruction")
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    lr_w, lr_h, lr_fps = video_info(args.lr_video)
    hr_w, hr_h, hr_fps = video_info(args.hr_video)
    if (hr_w, hr_h) != (lr_w * 2, lr_h * 2) or abs(lr_fps - hr_fps) > 0.01:
        raise ValueError("source video dimensions or frame rates do not match")
    model_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu = torch.device("cpu")
    model = load_coreml_weights(args.package).eval().to(model_device)
    candidate_model = None
    if args.checkpoint is not None:
        candidate_model = PiperSR().eval()
        candidate_model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
        candidate_model.to(model_device)
    args.output.mkdir(parents=True, exist_ok=True)
    offsets = [offset * args.stride for offset in range(-args.radius, args.radius + 1)]
    first_center = args.radius * args.stride
    count = first_center * 2 + args.frames
    scenes = []
    for timestamp in args.times:
        start = timestamp - first_center / lr_fps
        scene_dir = args.output / f"time-{timestamp:.3f}"
        lr_paths = extract_window(args.lr_video, start, count, scene_dir / "lr")
        hr_paths = extract_window(args.hr_video, start, count, scene_dir / "hr")
        center = image(lr_paths[first_center])
        truth = image(hr_paths[first_center])
        if similarity(center, truth) < 0.98:
            raise ValueError(f"unaligned center frame at {timestamp}")
        other = image(lr_paths[0])
        locations = candidate_patches(center, truth, other, args.patch, 1)
        if not locations:
            raise ValueError(f"no suitable patch at {timestamp}")
        x, y, _, _ = locations[0]
        baseline_frames = []
        refined_frames = []
        truth_frames = []
        frame_records = []
        for frame_offset in range(args.frames):
            index = first_center + frame_offset
            if candidate_model is None:
                observations, flows, confidences = frame_tensors(
                    lr_paths, index, offsets, x, y, args.patch, model_device,
                )
                center_observation = observations[args.radius]
            else:
                center_frame = image(lr_paths[index])[y:y + args.patch, x:x + args.patch]
                center_observation = to_tensor(center_frame, model_device)
            with torch.inference_mode():
                baseline = model(center_observation).float()
                if candidate_model is not None:
                    refined = candidate_model(center_observation).float()
            if candidate_model is None:
                refined = reconstruct(
                    baseline.to(cpu), [item.to(cpu) for item in observations],
                    flows, [item.to(cpu) for item in confidences],
                    args.iterations, args.anchor_weight,
                ).to(model_device)
            frame_truth = image(hr_paths[index])[
                y * 2:(y + args.patch) * 2, x * 2:(x + args.patch) * 2
            ]
            truth_tensor = to_tensor(frame_truth, model_device)
            frame_records.append({
                "frame": frame_offset,
                "baseline": quality(baseline, truth_tensor),
                "multiframe": quality(refined, truth_tensor),
            })
            baseline_frames.append(rgb_array(baseline))
            refined_frames.append(rgb_array(refined))
            truth_frames.append(rgb_array(truth_tensor))
        pair_records = []
        for index in range(1, args.frames):
            pair_records.append({
                "pair": [index - 1, index],
                "baseline": temporal_scores(baseline_frames[index - 1], baseline_frames[index],
                                            truth_frames[index - 1], truth_frames[index]),
                "multiframe": temporal_scores(refined_frames[index - 1], refined_frames[index],
                                              truth_frames[index - 1], truth_frames[index]),
                "truth": temporal_scores(truth_frames[index - 1], truth_frames[index],
                                         truth_frames[index - 1], truth_frames[index]),
            })
        scene = {"time": timestamp, "x": x, "y": y,
                 "frame_scores": frame_records, "temporal_scores": pair_records}
        scenes.append(scene)
        print(f"scene {timestamp}: {len(frame_records)} frames, {len(pair_records)} transitions", flush=True)
    def mean_values(section: str, method: str, metric: str) -> float:
        return float(np.mean([
            item[method][metric] for scene in scenes for item in scene[section]
        ]))
    summary = {
        method: {
            "psnr": mean_values("frame_scores", method, "psnr"),
            "hf_cosine": mean_values("frame_scores", method, "hf_cosine"),
            "flicker": mean_values("temporal_scores", method, "motion_compensated_flicker"),
            "detail_flicker": mean_values("temporal_scores", method, "detail_flicker"),
            "temporal_error_to_4k": mean_values("temporal_scores", method, "temporal_error_to_4k"),
            "detail_temporal_error_to_4k": mean_values("temporal_scores", method, "detail_temporal_error_to_4k"),
        } for method in ("baseline", "multiframe")
    }
    summary["truth"] = {
        "flicker": mean_values("temporal_scores", "truth", "motion_compensated_flicker"),
        "detail_flicker": mean_values("temporal_scores", "truth", "detail_flicker"),
    }
    report = {"parameters": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
              "scenes": scenes, "summary": summary}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
