#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Profile the real BasicVSR++ teacher on representative clips."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lada.models.basicvsrpp.activation_analysis import (  # noqa: E402
    AlignmentCapturePolicy,
    BasicVSRPPActivationAnalyzer,
)
from lada.models.basicvsrpp.basicvsrpp_gan import (  # noqa: E402
    BasicVSRPlusPlusGanNet,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"
DEFAULT_OUTPUT = Path(
    "/Volumes/Project_HD/lada_finetune_aozora_hikari/analysis/basicvsrpp-internals"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture SPyNet flow, deformable offsets/masks, per-group fixed-shift "
            "candidates, and aligned branch outputs without modifying inference."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--clip", type=Path, action="append", default=[])
    parser.add_argument(
        "--metadata-root",
        type=Path,
        action="append",
        default=[],
        help="crop_unscaled_meta directory; may be specified more than once",
    )
    parser.add_argument("--input-kind", choices=("auto", "clean", "mosaic"), default="auto")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--activation-samples-per-branch", type=int, default=2)
    parser.add_argument(
        "--capture-branch",
        action="append",
        choices=("backward_1", "forward_1", "backward_2", "forward_2"),
        default=[],
        help="retain NPZ tensors only for this branch; repeat as needed",
    )
    parser.add_argument("--capture-stride", type=int, default=1)
    parser.add_argument("--sample-channels", type=int, default=8)
    parser.add_argument("--sample-spatial-size", type=int, default=16)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_teacher(path: Path, device: torch.device, fp16: bool) -> BasicVSRPlusPlusGanNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"unexpected BasicVSR++ checkpoint: {path}")
    prefixes = ("generator_ema.", "generator.")
    prefix = next(
        (candidate for candidate in prefixes if any(str(key).startswith(candidate) for key in checkpoint)),
        None,
    )
    state = (
        {
            str(key)[len(prefix) :]: value
            for key, value in checkpoint.items()
            if str(key).startswith(prefix)
        }
        if prefix
        else checkpoint.get("state_dict", checkpoint)
    )
    model = BasicVSRPlusPlusGanNet(
        mid_channels=64, num_blocks=15, spynet_pretrained=None
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    if fp16:
        if device.type == "cpu":
            raise ValueError("--fp16 requires MPS or CUDA")
        model.half()
    return model


def metadata_clips(
    roots: list[Path], *, input_kind: str
) -> list[tuple[Path, Path | None]]:
    result: list[tuple[Path, Path | None]] = []
    for root in roots:
        for metadata_path in sorted(root.glob("*.json")):
            if metadata_path.name.startswith("._"):
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                clean = payload.get("relative_nsfw_video_path")
                mosaic = payload.get("relative_mosaic_nsfw_video_path")
                relative = (
                    mosaic
                    if input_kind == "mosaic"
                    else clean
                    if input_kind == "clean"
                    else mosaic or clean
                )
                if not relative:
                    continue
                clip = (metadata_path.parent / relative).resolve()
                if clip.exists():
                    result.append((clip, metadata_path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return result


def choose_clips(
    clips: list[tuple[Path, Path | None]], *, limit: int, seed: int
) -> list[tuple[Path, Path | None]]:
    # Deterministic random selection prevents a lexically early source video
    # from dominating offset statistics.
    clips = list(dict.fromkeys(clips))
    random.Random(seed).shuffle(clips)
    return clips[:limit] if limit > 0 else clips


def read_clip(path: Path, *, frames: int, size: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("could not open video")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, (frame_count - frames) // 2) if frame_count > 0 else 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    images: list[np.ndarray] = []
    try:
        while len(images) < frames:
            ok, image = capture.read()
            if not ok:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            scale = max(size / height, size / width)
            if scale != 1.0:
                image = cv2.resize(
                    image,
                    (max(size, round(width * scale)), max(size, round(height * scale))),
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
                )
            height, width = image.shape[:2]
            top = (height - size) // 2
            left = (width - size) // 2
            images.append(image[top : top + size, left : left + size])
    finally:
        capture.release()
    if len(images) < 2:
        raise RuntimeError(f"only {len(images)} frame(s) decoded")
    array = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0)


def main() -> int:
    args = parse_args()
    if args.frames < 2:
        raise SystemExit("--frames must be at least 2")
    if args.size < 256 or args.size % 4:
        raise SystemExit("--size must be a multiple of 4 and at least 256")

    candidates = [(path.resolve(), None) for path in args.clip]
    candidates.extend(metadata_clips(args.metadata_root, input_kind=args.input_kind))
    selected = choose_clips(candidates, limit=args.limit, seed=args.seed)
    if not selected:
        raise SystemExit("no readable candidates found; pass --clip or --metadata-root")

    device = resolve_device(args.device)
    model = load_teacher(args.checkpoint, device, args.fp16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    dtype = torch.float16 if args.fp16 else torch.float32

    capture_policy = AlignmentCapturePolicy(
        branches=frozenset(args.capture_branch) if args.capture_branch else None,
        call_stride=args.capture_stride,
        max_calls_per_branch=args.activation_samples_per_branch,
        channels=args.sample_channels,
        spatial_size=args.sample_spatial_size,
    )
    with BasicVSRPPActivationAnalyzer(model, capture_policy=capture_policy) as analyzer:
        for index, (clip, metadata) in enumerate(selected, start=1):
            begun_frames = 0
            try:
                inputs = read_clip(clip, frames=args.frames, size=args.size).to(
                    device=device, dtype=dtype
                )
                analyzer.begin_clip(inputs.shape[1])
                begun_frames = inputs.shape[1]
                with torch.inference_mode():
                    output = model(inputs)
                if device.type == "mps":
                    torch.mps.synchronize()
                processed.append(
                    {
                        "clip": str(clip),
                        "metadata": str(metadata) if metadata else None,
                        "frames": inputs.shape[1],
                        "input_shape": list(inputs.shape),
                        "output_shape": list(output.shape),
                    }
                )
                print(f"[{index}/{len(selected)}] analyzed {clip.name}", flush=True)
            except Exception as error:  # keep a long representative run useful
                if begun_frames:
                    analyzer.abort_clip(begun_frames)
                failures.append({"clip": str(clip), "error": str(error)})
                print(f"[{index}/{len(selected)}] skipped {clip}: {error}", file=sys.stderr)
        report = analyzer.report()
        sample_path = args.output_dir / "activation-samples.npz"
        analyzer.save_samples(sample_path)

    report.update(
        {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "fp16": args.fp16,
            "requested_frames": args.frames,
            "input_size": args.size,
            "seed": args.seed,
            "processed": processed,
            "failures": failures,
            "activation_samples": str(sample_path),
        }
    )
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {report_path}")
    print(f"activation samples: {sample_path}")
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
