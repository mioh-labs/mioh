#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export trained MiohRestorer V5 weights to fixed-shape Core ML packages."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer.model_v5 import (
    FRAME_CHANNELS,
    NUM_INPUT_FRAMES,
    MiohRestorerV5,
    MiohRestorerV5DecoderExportWrapper,
    MiohRestorerV5EncoderExportWrapper,
    MiohRestorerV5ExportWrapper,
    MiohRestorerV5StatefulExportWrapper,
    V5_BUCKETS,
    flatten_encoded_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("q", "s"), required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--sizes", default="128,192,256,384,512")
    parser.add_argument("--context-frames", type=int, choices=(5, 7, 9))
    parser.add_argument(
        "--execution",
        choices=("monolithic", "split", "stateful"),
        default="monolithic",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> MiohRestorerV5:
    model = (
        MiohRestorerV5.quality(context_frames=args.context_frames or 7)
        if args.variant == "q"
        else MiohRestorerV5.shipping(context_frames=args.context_frames or 9)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    key = "ema_model" if args.weights == "ema" else "model"
    state = checkpoint.get(key)
    if state is None and args.weights == "ema":
        state = checkpoint.get("ema")
    if state is None:
        raise KeyError(f"checkpoint has no {args.weights} V5 weights")
    model.load_state_dict(state, strict=True)
    return model.eval()


def remove_existing(path: Path, allow: bool) -> None:
    if not path.exists():
        return
    if not allow:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def convert(wrapper, example, input_names, output_names, output, args):
    import coremltools as ct

    remove_existing(output, args.allow_overwrite)
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float16)
            for name, value in zip(input_names, example, strict=True)
        ],
        outputs=[ct.TensorType(name=name, dtype=np.float16) for name in output_names],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    model.user_defined_metadata["mioh.restorer"] = "v5"
    model.user_defined_metadata["mioh.variant"] = args.variant
    model.user_defined_metadata["mioh.execution"] = args.execution
    model.user_defined_metadata["mioh.imgsz"] = str(example[0].shape[-1])
    model.save(str(output))


def main() -> int:
    args = parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    if not sizes or any(value not in V5_BUCKETS for value in sizes):
        raise ValueError(f"sizes must be selected from {V5_BUCKETS}")
    if args.variant == "q" and args.execution != "monolithic":
        raise ValueError("split/stateful exports are V5-S only")
    model = load_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        values = torch.zeros(1, NUM_INPUT_FRAMES, FRAME_CHANNELS, size, size)
        values[:, :, 4] = 1
        root = args.output_dir / str(size)
        root.mkdir(parents=True, exist_ok=True)
        if args.execution == "monolithic":
            wrapper = MiohRestorerV5ExportWrapper(model).eval()
            convert(
                wrapper,
                (values.flatten(1, 2),),
                ("frames",),
                ("rgb", "confidence"),
                root / "mioh-restorer-v5.mlpackage",
                args,
            )
        elif args.execution == "split":
            encoder = MiohRestorerV5EncoderExportWrapper(model.encoder).eval()
            decoder = MiohRestorerV5DecoderExportWrapper(model.decoder).eval()
            encoded = model.encode_window(values)
            convert(
                encoder,
                (values[:, -1],),
                ("frame",),
                ("packed", "half", "quarter", "eighth", "sixteenth"),
                root / "mioh-restorer-v5-encoder.mlpackage",
                args,
            )
            convert(
                decoder,
                flatten_encoded_window(encoded),
                ("packed", "half", "quarter", "eighth", "sixteenth"),
                ("rgb", "confidence"),
                root / "mioh-restorer-v5-decoder.mlpackage",
                args,
            )
        else:
            wrapper = MiohRestorerV5StatefulExportWrapper(model).eval()
            encoded = model.encode_window(values)
            states = tuple(
                torch.cat([frame[level] for frame in encoded[:-1]], dim=1)
                for level in range(5)
            )
            convert(
                wrapper,
                (values[:, -1], *states),
                (
                    "current_frame",
                    "packed_state",
                    "half_state",
                    "quarter_state",
                    "eighth_state",
                    "sixteenth_state",
                ),
                (
                    "rgb",
                    "confidence",
                    "next_packed_state",
                    "next_half_state",
                    "next_quarter_state",
                    "next_eighth_state",
                    "next_sixteenth_state",
                ),
                root / "mioh-restorer-v5-stateful.mlpackage",
                args,
            )
        print(f"exported V5-{args.variant.upper()} {size} {args.execution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
