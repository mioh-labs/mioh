# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a fixed-shape RF-DETR Seg checkpoint to a Core AI asset.

Validated configurations:

* RF-DETR 1.8.3 Seg Small at 384x384
* Jasna rfdetr-v6 (RF-DETR Seg Medium) at 576x576
* Jasna rfdetr-v6-large (RF-DETR Seg Large) at 768x768

Both use one deformable-attention feature level, 16 heads and two sampling
points.  The upstream six-dimensional ``[B,Q,H,L,P,2]`` sampling tensor is
folded to ``[B,Q,H,L*P,2]``.  Since ``L=1``, the rewrite is numerically exact.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .rfdetr_coreai_kernels import (
        ATTENTION_HEADS,
        FEATURE_SIDE,
        POINTS_PER_HEAD,
        build_ms_deform_attn_kernel,
        run_ms_deform_attn_kernel,
    )
else:
    from rfdetr_coreai_kernels import (  # type: ignore[import-not-found]
        ATTENTION_HEADS,
        FEATURE_SIDE,
        POINTS_PER_HEAD,
        build_ms_deform_attn_kernel,
        run_ms_deform_attn_kernel,
    )

DEFAULT_WEIGHTS = Path("model_weights/3rd_party/rf-detr-seg-small.pt")
DEFAULT_OUTPUT = Path("model_weights/rf-detr-seg-small-384-fp32.aimodel")
DEFAULT_RESOLUTION = 384
MODEL_CLASSES = {
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape RF-DETR Seg checkpoint to Core AI"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=sorted(MODEL_CLASSES),
        default="small",
        help="RF-DETR Seg architecture used by the checkpoint",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="fixed square model resolution (Jasna v6 uses 576)",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help=(
            "experimental half-precision export; FP32 is the validated default "
            "because RF-DETR's transformer is numerically unstable in FP16"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


class RFDETRExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, logits, masks = self.model(image)
        return boxes, logits, masks


def import_coreai() -> tuple[Any, Any]:
    try:
        import coreai
        import coreai_torch
    except ImportError as exc:
        raise RuntimeError(
            "Run this exporter with the .venv-coreai Python environment"
        ) from exc
    return coreai, coreai_torch


def checkpoint_num_classes(weights: Path) -> int | None:
    if not weights.is_file():
        return None
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    weight = state.get("class_embed.weight")
    if weight is None:
        return None
    # RF-DETR stores an additional background/no-object row.
    return int(weight.shape[0]) - 1


def download_or_load_model(
    weights: Path,
    *,
    fp16: bool,
    variant: str = "small",
    resolution: int = DEFAULT_RESOLUTION,
) -> RFDETRExportWrapper:
    try:
        import rfdetr
        from rfdetr.models.heads.segmentation import DepthwiseConvBlock
    except ImportError as exc:
        raise RuntimeError("rfdetr==1.8.3 is required") from exc

    weights.parent.mkdir(parents=True, exist_ok=True)

    # The custom autograd function exists only to disable cuDNN during
    # training.  Inference is exactly the underlying depthwise convolution and
    # this form is exportable.
    DepthwiseConvBlock._depthwise_conv = lambda self, x: self.dwconv(x)

    model_class = getattr(rfdetr, MODEL_CLASSES[variant])
    kwargs: dict[str, Any] = {
        "pretrain_weights": str(weights),
        "device": "cpu",
        "resolution": int(resolution),
    }
    if (num_classes := checkpoint_num_classes(weights)) is not None:
        kwargs["num_classes"] = num_classes
    instance = model_class(**kwargs)
    model = instance.model.model.eval()
    model.export()
    if fp16:
        model = model.half()
    return RFDETRExportWrapper(model).eval()


def make_example(
    seed: int,
    *,
    fp16: bool,
    resolution: int = DEFAULT_RESOLUTION,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (1, 3, resolution, resolution),
        generator=generator,
        dtype=torch.float16 if fp16 else torch.float32,
    )


@contextmanager
def fixed_shape_as_tensor():
    """Make RF-DETR's fixed spatial shape explicit to torch.export."""

    original = torch._shape_as_tensor

    def shape_as_tensor(value: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            tuple(value.shape),
            dtype=torch.long,
            device=value.device,
        )

    torch._shape_as_tensor = shape_as_tensor
    try:
        yield
    finally:
        torch._shape_as_tensor = original


@contextmanager
def use_ms_deform_attn_metal_kernel(
    kernel: Any,
    *,
    feature_side: int = FEATURE_SIDE,
):
    module = importlib.import_module(
        "rfdetr.models.ops.modules.ms_deform_attn"
    )
    cls = module.MSDeformAttn
    original = cls.forward

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        input_spatial_shapes_hw=None,
    ):
        del input_level_start_index, input_spatial_shapes_hw
        batch, query_count, _ = query.shape
        input_length = input_flatten.shape[1]
        if (
            self.n_levels != 1
            or self.n_heads != ATTENTION_HEADS
            or self.n_points != POINTS_PER_HEAD
            or input_length != feature_side * feature_side
        ):
            raise ValueError(
                "Core AI RF-DETR kernel does not match the model structure"
            )

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)

        sampling_offsets = self.sampling_offsets(query).reshape(
            batch,
            query_count,
            self.n_heads,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).reshape(
            batch,
            query_count,
            self.n_heads,
            self.n_points,
        )
        attention_weights = attention_weights.softmax(-1)

        # Fold level and point.  Seg Small has one level, therefore no
        # information or ordering changes.
        reference = reference_points.repeat_interleave(
            self.n_points,
            dim=2,
        )
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack(
                [
                    input_spatial_shapes[..., 1],
                    input_spatial_shapes[..., 0],
                ],
                -1,
            ).repeat_interleave(self.n_points, dim=0)
            sampling_locations = (
                reference[:, :, None, :, :]
                + sampling_offsets / normalizer[None, None, None, :, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference[:, :, None, :, :2]
                + sampling_offsets
                / self.n_points
                * reference[:, :, None, :, 2:]
                * 0.5
            )
        else:
            raise ValueError("reference point width must be 2 or 4")

        # Keep the native contiguous projection layout.  Feeding a transposed
        # view into a Core AI Metal kernel can preserve non-unit strides even
        # when PyTorch observed a contiguous tensor during export.
        value = value.reshape(
            batch,
            input_length,
            self.n_heads,
            self.d_model // self.n_heads,
        )
        output = run_ms_deform_attn_kernel(
            kernel,
            value,
            sampling_locations,
            attention_weights,
            feature_side=feature_side,
        )
        return self.output_proj(output)

    cls.forward = forward
    try:
        yield
    finally:
        cls.forward = original


def summarize_operators(exported: torch.export.ExportedProgram) -> dict[str, int]:
    counts = Counter(
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    )
    return dict(sorted(counts.items()))


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resolution <= 0 or args.resolution % 12:
        raise ValueError("resolution must be a positive multiple of 12")
    feature_side = args.resolution // 12
    report_path = args.report or args.output.with_suffix(".report.json")
    report: dict[str, Any] = {
        "success": False,
        "architecture": f"rfdetr-seg-{args.variant}",
        "resolution": args.resolution,
        "feature_side": feature_side,
        "dtype": "float16" if args.fp16 else "float32",
        "weights": str(args.weights),
        "output": str(args.output),
        "stages": {},
    }
    stage = "preflight"

    try:
        started = time.perf_counter()
        _, coreai_torch = import_coreai()
        if args.output.exists() and not args.allow_overwrite:
            raise FileExistsError(
                f"{args.output} exists; pass --allow-overwrite"
            )
        report["stages"][stage] = time.perf_counter() - started

        stage = "build_kernel"
        started = time.perf_counter()
        kernel = build_ms_deform_attn_kernel(
            coreai_torch,
            feature_side=feature_side,
        )
        report["stages"][stage] = time.perf_counter() - started

        stage = "load_model"
        started = time.perf_counter()
        wrapper = download_or_load_model(
            args.weights,
            fp16=args.fp16,
            variant=args.variant,
            resolution=args.resolution,
        )
        example = make_example(
            args.seed,
            fp16=args.fp16,
            resolution=args.resolution,
        )
        report["stages"][stage] = time.perf_counter() - started

        stage = "torch_export"
        started = time.perf_counter()
        with (
            fixed_shape_as_tensor(),
            use_ms_deform_attn_metal_kernel(
                kernel,
                feature_side=feature_side,
            ),
        ):
            exported = torch.export.export(
                wrapper,
                (example,),
                strict=False,
            )
        report["stages"][stage] = time.perf_counter() - started

        stage = "decompose"
        started = time.perf_counter()
        exported = exported.run_decompositions(
            coreai_torch.get_decomp_table()
        )
        report["operators"] = summarize_operators(exported)
        report["stages"][stage] = time.perf_counter() - started

        stage = "convert"
        started = time.perf_counter()
        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels([kernel])
        converter.add_exported_program(
            exported,
            input_names=["image"],
            output_names=["boxes", "logits", "masks"],
        )
        program = converter.to_coreai()
        report["stages"][stage] = time.perf_counter() - started

        stage = "optimize"
        started = time.perf_counter()
        program.optimize()
        report["stages"][stage] = time.perf_counter() - started

        stage = "save"
        started = time.perf_counter()
        remove_existing(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        program.save_asset(args.output)
        report["stages"][stage] = time.perf_counter() - started

        report["success"] = True
        report["custom_kernel"] = str(kernel.name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Core AI asset: {args.output}")
        print(f"Report: {report_path}")
        return 0
    except Exception as exc:
        report["failed_stage"] = stage
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.verbose:
            traceback.print_exc()
        print(f"RF-DETR Core AI export failed at {stage}: {exc}")
        print(f"Report: {report_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
