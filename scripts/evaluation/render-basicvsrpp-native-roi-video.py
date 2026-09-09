#!/usr/bin/env python3
"""Render a fixed, unscaled ROI video with a BasicVSR++ checkpoint."""

from __future__ import annotations

import argparse
import gc
import subprocess
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crop-x", required=True, type=int)
    parser.add_argument("--crop-y", required=True, type=int)
    parser.add_argument("--crop-size", default=512, type=int)
    parser.add_argument("--chunk-frames", default=18, type=int)
    parser.add_argument("--overlap-frames", default=6, type=int)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--video-bitrate", default="12M")
    return parser.parse_args()


def generator_state(path: Path) -> dict[str, torch.Tensor]:
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


def load_model(path: Path, device: torch.device) -> BasicVSRPlusPlusGanNet:
    model = BasicVSRPlusPlusGanNet(
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=None,
    )
    model.load_state_dict(generator_state(path), strict=True)
    return model.requires_grad_(False).eval().to(device)


def decode_crops(args: argparse.Namespace) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {args.source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    crops: list[np.ndarray] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        x0, y0, size = args.crop_x, args.crop_y, args.crop_size
        crop = frame[y0 : y0 + size, x0 : x0 + size]
        if crop.shape[:2] != (size, size):
            raise RuntimeError(
                f"crop is outside frame {frame.shape[:2]} at frame {frame_index}: "
                f"x={x0}, y={y0}, size={size}"
            )
        crops.append(np.ascontiguousarray(crop))
        frame_index += 1
    capture.release()
    if not crops:
        raise RuntimeError(f"source contains no decodable frames: {args.source}")
    if fps <= 0:
        raise RuntimeError(f"source has invalid frame rate: {fps}")
    return crops, fps


def chunk_starts(frame_count: int, chunk: int, overlap: int) -> list[int]:
    if chunk <= 0:
        raise ValueError("--chunk-frames must be positive")
    if overlap < 0 or overlap >= chunk:
        raise ValueError("--overlap-frames must be in [0, chunk-frames)")
    if frame_count <= chunk:
        return [0]
    stride = chunk - overlap
    starts = list(range(0, frame_count - chunk + 1, stride))
    final_start = frame_count - chunk
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_chunk(
    model: BasicVSRPlusPlusGanNet,
    frames: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    inputs = (
        torch.from_numpy(rgb)
        .permute(0, 3, 1, 2)
        .float()
        .div_(255.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        prediction = model(inputs)[0].clamp(0.0, 1.0).mul(255.0).round()
    result = prediction.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    del inputs, prediction
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in result]


def restore_video(
    model: BasicVSRPlusPlusGanNet,
    crops: list[np.ndarray],
    *,
    chunk: int,
    overlap: int,
    device: torch.device,
) -> list[np.ndarray]:
    starts = chunk_starts(len(crops), chunk, overlap)
    restored: list[np.ndarray] = []
    previous_end = 0
    for chunk_index, start in enumerate(starts, start=1):
        end = min(start + chunk, len(crops))
        predicted = predict_chunk(model, crops[start:end], device)
        shared = max(0, previous_end - start)
        if shared:
            if shared >= len(predicted):
                raise RuntimeError("chunk overlap covers the complete prediction")
            for offset in range(shared):
                alpha = float(offset + 1) / float(shared + 1)
                prior_index = start + offset
                blended = cv2.addWeighted(
                    restored[prior_index], 1.0 - alpha, predicted[offset], alpha, 0.0
                )
                restored[prior_index] = blended
            restored.extend(predicted[shared:])
        else:
            restored.extend(predicted)
        previous_end = end
        print(
            f"chunk {chunk_index}/{len(starts)} frames {start}:{end} "
            f"assembled={len(restored)}",
            flush=True,
        )
    if len(restored) != len(crops):
        raise RuntimeError(
            f"restored frame count mismatch: {len(restored)} != {len(crops)}"
        )
    return restored


def encode_video(
    frames: list[np.ndarray],
    *,
    source: Path,
    output: Path,
    fps: float,
    bitrate: str,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.12f}",
        "-i",
        "pipe:0",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "h264_videotoolbox",
        "-b:v",
        bitrate,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        output.unlink(missing_ok=True)
        raise
    if return_code:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited with code {return_code}")


def main() -> None:
    args = parse_args()
    for path in (args.source, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    crops, fps = decode_crops(args)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    restored = restore_video(
        model,
        crops,
        chunk=args.chunk_frames,
        overlap=args.overlap_frames,
        device=device,
    )
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    encode_video(
        restored,
        source=args.source,
        output=args.output,
        fps=fps,
        bitrate=args.video_bitrate,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
