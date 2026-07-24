# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export Real-ESRGAN RRDBNet x2plus/x4plus to fixed-shape FP16 Core AI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL = Path("model_weights/RealESRGAN_x4plus.pth")
DEFAULT_OUTPUT = Path("model_weights/RealESRGAN_x4plus-256-fp16.aimodel")
X2_MODEL = Path("model_weights/RealESRGAN_x2plus.pth")
X2_OUTPUT = Path("model_weights/RealESRGAN_x2plus-256-fp16.aimodel")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Real-ESRGAN RRDBNet to fixed-shape FP16 Core AI"
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4, choices=(2, 4))
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = X2_MODEL if args.scale == 2 else DEFAULT_MODEL
    if args.output is None:
        args.output = X2_OUTPUT if args.scale == 2 else DEFAULT_OUTPUT
    return args


class CoreAIImageWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).clamp(0.0, 1.0)


def build_rrdbnet(scale: int) -> torch.nn.Module:
    class ResidualDenseBlock(torch.nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = torch.nn.Conv2d(
                num_feat + num_grow_ch,
                num_grow_ch,
                3,
                1,
                1,
            )
            self.conv3 = torch.nn.Conv2d(
                num_feat + 2 * num_grow_ch,
                num_grow_ch,
                3,
                1,
                1,
            )
            self.conv4 = torch.nn.Conv2d(
                num_feat + 3 * num_grow_ch,
                num_grow_ch,
                3,
                1,
                1,
            )
            self.conv5 = torch.nn.Conv2d(
                num_feat + 4 * num_grow_ch,
                num_feat,
                3,
                1,
                1,
            )
            self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            x1 = self.lrelu(self.conv1(image))
            x2 = self.lrelu(self.conv2(torch.cat((image, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((image, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((image, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((image, x1, x2, x3, x4), 1))
            return x5 * 0.2 + image

    class RRDB(torch.nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            output = self.rdb1(image)
            output = self.rdb2(output)
            output = self.rdb3(output)
            return output * 0.2 + image

    class RRDBNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            input_channels = 12 if scale == 2 else 3
            self.conv_first = torch.nn.Conv2d(input_channels, 64, 3, 1, 1)
            self.body = torch.nn.Sequential(*(RRDB() for _ in range(23)))
            self.conv_body = torch.nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = torch.nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = torch.nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = torch.nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = torch.nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            if self.scale == 2:
                image = torch.nn.functional.pixel_unshuffle(image, 2)
            feature = self.conv_first(image)
            body_feature = self.conv_body(self.body(feature))
            feature = feature + body_feature
            feature = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(
                        feature,
                        scale_factor=2,
                        mode="nearest",
                    )
                )
            )
            feature = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(
                        feature,
                        scale_factor=2,
                        mode="nearest",
                    )
                )
            )
            return self.conv_last(self.lrelu(self.conv_hr(feature)))

    return RRDBNet()


def load_model(model_path: Path, scale: int) -> CoreAIImageWrapper:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    net = build_rrdbnet(scale)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    net.load_state_dict(
        state.get("params_ema") or state.get("params") or state,
        strict=True,
    )
    return CoreAIImageWrapper(net.half()).eval()


def convert_exported_program(
    exported: torch.export.ExportedProgram,
    coreai_torch: Any,
):
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["image"],
        output_names=["enhanced"],
    )
    return converter.to_coreai()


def export_model(args: argparse.Namespace) -> Path:
    if args.imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{args.output} exists; pass --allow-overwrite")

    import coreai_torch

    model = load_model(args.model, args.scale)
    example = torch.zeros(
        (1, 3, args.imgsz, args.imgsz),
        dtype=torch.float16,
    )
    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = convert_exported_program(exported, coreai_torch)
    program.optimize()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(args.output)
    if args.verbose:
        print(f"Core AI Real-ESRGAN asset: {args.output}")
    return args.output


def main(argv: list[str] | None = None) -> int:
    export_model(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
