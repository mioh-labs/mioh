# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export Real-ESRGAN SRVGGNetCompact to fixed-shape FP16 Core AI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .export_srvgg_coreml import build_srvgg
else:
    from export_srvgg_coreml import build_srvgg  # type: ignore[import-not-found]

DEFAULT_MODEL = Path("model_weights/realesr-general-x4v3.pth")
DEFAULT_OUTPUT = Path("model_weights/realesr-general-x4v3-256-fp16.aimodel")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SRVGGNetCompact to fixed-shape FP16 Core AI"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num-conv", type=int, default=32)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


class CoreAIImageWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).clamp(0.0, 1.0)


class CoreAIPReLU(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        slope = self.weight.view(1, -1, 1, 1)
        return torch.clamp_min(image, 0.0) + slope * torch.clamp_max(image, 0.0)


def replace_prelu_for_coreai(net: torch.nn.Module) -> None:
    for name, child in list(net.named_children()):
        if isinstance(child, torch.nn.PReLU):
            setattr(net, name, CoreAIPReLU(child.weight.detach().clone()))
        else:
            replace_prelu_for_coreai(child)


def load_model(
    model_path: Path,
    scale: int,
    num_conv: int,
) -> CoreAIImageWrapper:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    net, _ = build_srvgg(scale=scale, num_conv=num_conv)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    net.load_state_dict(
        state.get("params_ema") or state.get("params") or state,
        strict=True,
    )
    replace_prelu_for_coreai(net)
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
    if args.scale != 4:
        raise ValueError("SRVGG Core AI export currently requires scale=4")
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{args.output} exists; pass --allow-overwrite")

    import coreai_torch

    model = load_model(args.model, args.scale, args.num_conv)
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
        print(f"Core AI SRVGG asset: {args.output}")
    return args.output


def main(argv: list[str] | None = None) -> int:
    export_model(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
