# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export the untrained MiohRestorerV1 deployment prototype."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer import MiohRestorerV1


DEFAULT_OUTPUT_DIR = Path("build/mioh-restorer-prototype")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MiohRestorerV1 to Core ML and Core AI"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument(
        "--chunk-frames", type=int, default=MiohRestorerV1.DEFAULT_CHUNK_FRAMES
    )
    parser.add_argument(
        "--channels", type=int, default=MiohRestorerV1.DEFAULT_CHANNELS
    )
    parser.add_argument("--blocks", type=int, default=MiohRestorerV1.DEFAULT_BLOCKS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--skip-coreai", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_model(args: argparse.Namespace) -> tuple[MiohRestorerV1, dict]:
    torch.manual_seed(args.seed)
    payload: dict = {}
    config = {
        "chunk_frames": args.chunk_frames,
        "channels": args.channels,
        "num_blocks": args.blocks,
    }
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint_config = payload.get("config", {})
        config = {
            "chunk_frames": int(
                checkpoint_config.get("chunk_frames", args.chunk_frames)
            ),
            "channels": int(checkpoint_config.get("channels", args.channels)),
            "num_blocks": int(checkpoint_config.get("num_blocks", args.blocks)),
        }
    model = MiohRestorerV1(
        chunk_frames=config["chunk_frames"],
        channels=config["channels"],
        num_blocks=config["num_blocks"],
    )
    if args.checkpoint is not None:
        model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return model.eval(), payload


def example_inputs(
    model: MiohRestorerV1,
    image_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = torch.zeros(
        (1, model.chunk_frames, 3, image_size, image_size), dtype=dtype
    )
    masks = torch.ones(
        (1, model.chunk_frames, 1, image_size, image_size), dtype=dtype
    )
    state = torch.zeros(
        model.state_shape(image_height=image_size, image_width=image_size),
        dtype=dtype,
    )
    return frames, masks, state


def remove_existing(path: Path, allow_overwrite: bool) -> None:
    if not path.exists():
        return
    if not allow_overwrite:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def export_coreai(
    model: MiohRestorerV1,
    output_path: Path,
    image_size: int,
    allow_overwrite: bool,
) -> None:
    import coreai_torch

    remove_existing(output_path, allow_overwrite)
    coreai_model = copy.deepcopy(model).half().eval()
    examples = example_inputs(coreai_model, image_size, torch.float16)
    exported = torch.export.export(coreai_model, examples)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["frames", "masks", "history"],
        output_names=["restored", "next_state"],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output_path)


def export_coreml(
    model: MiohRestorerV1,
    output_path: Path,
    image_size: int,
    allow_overwrite: bool,
) -> None:
    import coremltools as ct

    remove_existing(output_path, allow_overwrite)
    # coremltools 9 imports TorchScript graphs in FP32 even when their example
    # tensors are FP16. Trace FP32 weights to keep the frontend type-consistent;
    # the ML Program still receives FP16 tensors and uses FP16 compute below.
    coreml_model = copy.deepcopy(model).float().eval()
    frames, masks, state = example_inputs(coreml_model, image_size, torch.float32)
    traced = torch.jit.trace(coreml_model, (frames, masks, state))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="frames", shape=frames.shape, dtype=np.float16),
            ct.TensorType(name="masks", shape=masks.shape, dtype=np.float16),
            ct.TensorType(name="history", shape=state.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="restored", dtype=np.float16),
            ct.TensorType(name="next_state", dtype=np.float16),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.user_defined_metadata["mioh.restorer"] = "v1"
    mlmodel.user_defined_metadata["mioh.chunk_frames"] = str(model.chunk_frames)
    mlmodel.user_defined_metadata["mioh.channels"] = str(model.channels)
    mlmodel.user_defined_metadata["mioh.imgsz"] = str(image_size)
    mlmodel.save(str(output_path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, source_payload = build_model(args)
    if args.imgsz is None:
        args.imgsz = int(source_payload.get("config", {}).get("image_size", 256))
    if args.imgsz <= 0 or args.imgsz % 4:
        raise ValueError("imgsz must be positive and divisible by 4")
    if args.skip_coreml and args.skip_coreai:
        raise ValueError("at least one deployment target must remain enabled")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trained = bool(source_payload.get("trained", False)) or int(
        source_payload.get("step", 0)
    ) > 0
    variant = "" if trained else "-prototype"
    asset_stem = (
        f"mioh-restorer-v1{variant}-t{model.chunk_frames}-c{model.channels}"
        f"-s{args.imgsz}-fp16"
    )
    checkpoint_path = args.output_dir / f"mioh-restorer-v1{variant}.pth"
    coreml_path = args.output_dir / f"{asset_stem}.mlpackage"
    coreai_path = args.output_dir / f"{asset_stem}.aimodel"
    report_path = args.output_dir / "export-report.json"
    remove_existing(checkpoint_path, args.allow_overwrite)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "chunk_frames": model.chunk_frames,
                "channels": model.channels,
                "num_blocks": model.num_blocks,
                "image_size": args.imgsz,
            },
            "trained": trained,
            "prototype": not trained,
        },
        checkpoint_path,
    )
    report = {
        "prototype": not trained,
        "trained": trained,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "contract": {
            "frames": [1, model.chunk_frames, 3, args.imgsz, args.imgsz],
            "masks": [1, model.chunk_frames, 1, args.imgsz, args.imgsz],
            "history": list(model.state_shape(image_height=args.imgsz, image_width=args.imgsz)),
        },
        "checkpoint": str(checkpoint_path),
        "coreml": None,
        "coreai": None,
        "stage_seconds": {},
    }
    if not args.skip_coreml:
        started = time.perf_counter()
        export_coreml(model, coreml_path, args.imgsz, args.allow_overwrite)
        report["stage_seconds"]["coreml"] = round(time.perf_counter() - started, 6)
        report["coreml"] = str(coreml_path)
    if not args.skip_coreai:
        started = time.perf_counter()
        export_coreai(model, coreai_path, args.imgsz, args.allow_overwrite)
        report["stage_seconds"]["coreai"] = round(time.perf_counter() - started, 6)
        report["coreai"] = str(coreai_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
