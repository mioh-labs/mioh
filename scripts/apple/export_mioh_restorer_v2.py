# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a trained MiohRestorerV2/V3 checkpoint to Apple runtimes."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer import MiohRestorerV2, MiohRestorerV3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="export online weights instead of the default EMA weights",
    )
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--skip-coreai", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_model(
    checkpoint: Path, raw_weights: bool
) -> tuple[MiohRestorerV2 | MiohRestorerV3, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    version = int(config.get("version", 0))
    if version == 2:
        model = MiohRestorerV2(
            window_frames=int(config["window_frames"]),
            chunk_frames=int(config["chunk_frames"]),
            channels=int(config["channels"]),
            num_blocks=int(config["num_blocks"]),
            fusion_full_channels=int(config["fusion_full_channels"]),
            fusion_half_channels=int(config["fusion_half_channels"]),
            fusion_quarter_channels=int(config["fusion_quarter_channels"]),
            detail_scale=float(config["detail_scale"]),
        )
    elif version == 3:
        if int(config.get("architecture_revision", 0)) not in (
            MiohRestorerV3.SUPPORTED_ARCHITECTURE_REVISIONS
        ):
            raise ValueError("unsupported MiohRestorerV3 architecture revision")
        model = MiohRestorerV3(
            window_frames=int(config["window_frames"]),
            channels=int(config["channels"]),
            num_blocks=int(config["num_blocks"]),
            encoder_blocks=int(config["encoder_blocks"]),
            reconstruction_blocks=int(config["reconstruction_blocks"]),
            alignment_radius=int(config["alignment_radius"]),
            first_order_dilation=int(config["first_order_dilation"]),
            second_order_dilation=int(config["second_order_dilation"]),
            alignment_key_channels=int(config["alignment_key_channels"]),
            alignment_groups=int(config.get("alignment_groups", 1)),
            hierarchical_alignment_dilations=tuple(
                int(item)
                for item in config.get(
                    "hierarchical_alignment_dilations", []
                )
            ),
            alignment_temperature=float(
                config.get("alignment_temperature", 1.0)
            ),
            detail_scale=float(config["detail_scale"]),
        )
    else:
        raise ValueError("checkpoint is not MiohRestorerV2/V3")
    state_key = "state_dict" if raw_weights else "ema_state_dict"
    if state_key not in payload:
        raise ValueError(f"checkpoint is missing {state_key}")
    model.load_state_dict(payload[state_key], strict=True)
    return model.eval(), payload


def remove_existing(path: Path, allow_overwrite: bool) -> None:
    if not path.exists():
        return
    if not allow_overwrite:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def example_inputs(
    model: MiohRestorerV2 | MiohRestorerV3,
    image_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(
            (1, model.window_frames, 3, image_size, image_size), dtype=dtype
        ),
        torch.ones(
            (1, model.window_frames, 1, image_size, image_size), dtype=dtype
        ),
    )


def export_coreml(
    model: MiohRestorerV2 | MiohRestorerV3,
    output: Path,
    image_size: int,
    allow_overwrite: bool,
) -> None:
    import coremltools as ct

    remove_existing(output, allow_overwrite)
    coreml_model = copy.deepcopy(model).float().eval()
    frames, masks = example_inputs(coreml_model, image_size, torch.float32)
    traced = torch.jit.trace(coreml_model, (frames, masks), check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="frames", shape=frames.shape, dtype=np.float16),
            ct.TensorType(name="masks", shape=masks.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="restored", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    restorer_label = "2"
    if isinstance(model, MiohRestorerV3):
        restorer_label = (
            "3.1" if model.hierarchical_alignment_dilations else "3"
        )
    converted.user_defined_metadata["mioh.restorer"] = f"v{restorer_label}"
    converted.user_defined_metadata["mioh.window_frames"] = str(
        model.window_frames
    )
    converted.user_defined_metadata["mioh.channels"] = str(model.channels)
    converted.user_defined_metadata["mioh.imgsz"] = str(image_size)
    converted.save(str(output))


def export_coreai(
    model: MiohRestorerV2 | MiohRestorerV3,
    output: Path,
    image_size: int,
    allow_overwrite: bool,
) -> None:
    import coreai_torch

    remove_existing(output, allow_overwrite)
    coreai_model = copy.deepcopy(model).half().eval()
    examples = example_inputs(coreai_model, image_size, torch.float16)
    exported = torch.export.export(coreai_model, examples)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["frames", "masks"],
        output_names=["restored"],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_coreml and args.skip_coreai:
        raise ValueError("at least one deployment target must remain enabled")
    model, payload = build_model(args.checkpoint, args.raw_weights)
    config = payload["config"]
    version = int(config["version"])
    version_label = (
        "31"
        if version == 3
        and config.get("hierarchical_alignment_dilations")
        else str(version)
    )
    image_size = int(config["image_size"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"mioh-restorer-v{version_label}-t{model.window_frames}-c{model.channels}"
        f"-s{image_size}-fp16"
    )
    checkpoint_output = args.output_dir / f"mioh-restorer-v{version_label}.pth"
    coreml_output = args.output_dir / f"{stem}.mlpackage"
    coreai_output = args.output_dir / f"{stem}.aimodel"
    report_output = args.output_dir / "export-report.json"
    remove_existing(checkpoint_output, args.allow_overwrite)
    selected_state = model.state_dict()
    torch.save(
        {
            "state_dict": selected_state,
            "config": config,
            "step": int(payload.get("step", 0)),
            "trained": bool(payload.get("trained", False)),
            "weights": "raw" if args.raw_weights else "ema",
        },
        checkpoint_output,
    )
    report = {
        "step": int(payload.get("step", 0)),
        "weights": "raw" if args.raw_weights else "ema",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "contract": {
            "frames": [1, model.window_frames, 3, image_size, image_size],
            "masks": [1, model.window_frames, 1, image_size, image_size],
            "restored": [1, model.window_frames, 3, image_size, image_size],
        },
        "checkpoint": str(checkpoint_output),
        "coreml": None,
        "coreai": None,
        "stage_seconds": {},
    }
    if not args.skip_coreml:
        started = time.perf_counter()
        export_coreml(model, coreml_output, image_size, args.allow_overwrite)
        report["coreml"] = str(coreml_output)
        report["stage_seconds"]["coreml"] = round(time.perf_counter() - started, 6)
    if not args.skip_coreai:
        started = time.perf_counter()
        export_coreai(model, coreai_output, image_size, args.allow_overwrite)
        report["coreai"] = str(coreai_output)
        report["stage_seconds"]["coreai"] = round(time.perf_counter() - started, 6)
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
