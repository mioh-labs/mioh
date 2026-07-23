#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Restore a short real video with an in-training MiohRestorer V5-HQ checkpoint.

This evaluator deliberately reuses Lada's production mosaic detector, tracker,
ROI cropper and compositor.  Only the restoration adapter is experimental.  A
nine-frame V5-HQ window emits five frames; windows advance by four frames and
the single overlapping output is averaged.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from lada.cli.main import process_video_file
from lada.models.mioh_restorer.model_v5_hq import MiohRestorerV5HQ
from lada.models.yolo.yolo11_coreml_segmentation_model import (
    Yolo11CoreMLSegmentationModel,
)
from lada.restorationpipeline.mioh_restorer import MiohMosaicRestorer
from lada.utils import ImageTensor


class V5HQCheckpointMosaicRestorer(MiohMosaicRestorer):
    """Clip adapter for a five-output V5-HQ training checkpoint."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("variant") != "hq":
            raise ValueError(f"not a V5-HQ checkpoint: {checkpoint}")
        self.model = MiohRestorerV5HQ()
        self.model.load_state_dict(payload["ema_state_dict"], strict=True)
        self.model.eval().to(device)
        self.device = device
        self.dtype = torch.float32
        # FrameRestorer reads this contract to make native 256x256 ROI crops.
        self.runtime = SimpleNamespace(image_size=256)

    @staticmethod
    def _prepare_video(video: list[ImageTensor]) -> torch.Tensor:
        if not video:
            raise ValueError("video must contain at least one frame")
        return (
            torch.stack(
                [torch.as_tensor(frame).permute(2, 0, 1) for frame in video], dim=0
            )
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )

    @staticmethod
    def _prepare_masks(
        masks: list[ImageTensor] | None, frames: torch.Tensor
    ) -> torch.Tensor:
        if masks is None:
            return torch.ones(
                frames.shape[0], frames.shape[1], 1, *frames.shape[-2:]
            )
        if len(masks) != frames.shape[1]:
            raise ValueError("frame and mask counts differ")
        prepared = []
        for mask in masks:
            value = torch.as_tensor(mask)
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            value = value[..., :1].permute(2, 0, 1).float()
            if value.numel() and value.max() > 1:
                value.div_(255.0)
            prepared.append(value)
        return torch.stack(prepared, dim=0).unsqueeze(0).clamp_(0, 1)

    @torch.inference_mode()
    def _restore_forward(
        self, frames: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        frame_count = frames.shape[1]
        accumulated = torch.zeros_like(frames, device=self.device)
        weights = torch.zeros(
            1,
            frame_count,
            1,
            *frames.shape[-2:],
            device=self.device,
            dtype=torch.float32,
        )
        frames = frames.to(self.device)
        masks = masks.to(self.device)
        reliability = torch.ones_like(masks)
        values = torch.cat((frames, masks, reliability), dim=2)

        for output_start in range(0, frame_count, 4):
            indices = [
                min(max(output_start - 2 + offset, 0), frame_count - 1)
                for offset in range(9)
            ]
            restored, _confidence = self.model(values[:, indices])
            valid = min(5, frame_count - output_start)
            accumulated[:, output_start : output_start + valid].add_(
                restored[:, :valid]
            )
            weights[:, output_start : output_start + valid].add_(1.0)
        return accumulated.div_(weights.clamp_min_(1.0)).cpu()

    def restore(
        self,
        video: list[ImageTensor],
        masks: list[ImageTensor] | None = None,
        *,
        bidirectional: bool = False,
    ) -> list[ImageTensor]:
        frames = self._prepare_video(video)
        prepared_masks = self._prepare_masks(masks, frames)
        forward = self._restore_forward(frames, prepared_masks)
        if bidirectional:
            backward = self._restore_forward(
                torch.flip(frames, dims=(1,)),
                torch.flip(prepared_masks, dims=(1,)),
            )
            forward = (forward + torch.flip(backward, dims=(1,))) * 0.5
        output = (
            forward.squeeze(0)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
        )
        return list(output.unbind(0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("model_weights/lada_mosaic_detection_model_v4_accurate.mlpackage"),
    )
    parser.add_argument("--max-clip-length", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.input, args.checkpoint, args.detector):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    detector = Yolo11CoreMLSegmentationModel(
        str(args.detector), device, classes=None, conf=0.15
    )
    restorer = V5HQCheckpointMosaicRestorer(args.checkpoint, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mioh-v5-hq-") as temporary:
        process_video_file(
            input_path=str(args.input),
            output_path=str(args.output),
            temp_dir_path=temporary,
            device=device,
            mosaic_restoration_model=restorer,
            mosaic_detection_model=detector,
            mosaic_restoration_model_name="mioh-restorer-v5-hq-experimental",
            preferred_pad_mode="zero",
            max_clip_length=args.max_clip_length,
            encoder="libx264",
            encoder_options="-crf 15 -preset medium -pix_fmt yuv420p",
            mp4_fast_start=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
