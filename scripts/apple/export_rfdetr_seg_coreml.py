# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export the fixed-shape Jasna RF-DETR v6 detector to Core ML.

RF-DETR's single-level deformable attention normally materializes a rank-six
``[B, Q, H, L, P, 2]`` sampling tensor. Core ML tensors are limited to rank
five. Jasna v6 uses exactly one feature level, so this exporter removes that
unit level dimension and evaluates the same bilinear samples through
``torch.nn.functional.grid_sample``. coremltools lowers the latter to the
native ML Program ``resample`` operation.

The exported model consumes an ImageNet-normalized FP32 NCHW tensor and emits
the unmodified ``boxes``, ``logits`` and ``masks`` tensors used by the existing
Core AI RF-DETR postprocessor.
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
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from .export_rfdetr_seg_coreai import (
        MODEL_CLASSES,
        RFDETRExportWrapper,
        download_or_load_model,
        fixed_shape_as_tensor,
        make_example,
    )
else:
    from export_rfdetr_seg_coreai import (  # type: ignore[import-not-found]
        MODEL_CLASSES,
        RFDETRExportWrapper,
        download_or_load_model,
        fixed_shape_as_tensor,
        make_example,
    )


DEFAULT_WEIGHTS = Path("model_weights/rfdetr-v6.pt")
DEFAULT_OUTPUT = Path("model_weights/rfdetr-v6-576-fp32.mlpackage")
DEFAULT_RESOLUTION = 576


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape Jasna RF-DETR Seg model to Core ML"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=sorted(MODEL_CLASSES),
        default="medium",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="experimental FP16 conversion; validated Jasna v6 uses FP32",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="save without running the resulting ML Program once",
    )
    return parser.parse_args(argv)


def _fixed_encoder_output_proposals(
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor | None = None,
    spatial_shapes: Any = None,
    unsigmoid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-shape equivalent of RF-DETR's meshgrid proposal helper.

    coremltools 9 rejects the traced ``meshgrid`` because its lengths arrive
    through traced scalar tensors. The model resolution is fixed, so explicit
    expand operations produce the same grid without that ambiguous operator.
    """

    proposals: list[torch.Tensor] = []
    current = 0
    batch_size = memory.shape[0]
    for level, (height_value, width_value) in enumerate(spatial_shapes):
        height = int(height_value)
        width = int(width_value)
        if memory_padding_mask is not None:
            mask = memory_padding_mask[
                :, current : current + height * width
            ].reshape(batch_size, height, width, 1)
            valid_height = torch.sum(~mask[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask[:, 0, :, 0], 1)
        else:
            valid_height = (
                torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + height
            )
            valid_width = (
                torch.zeros_like(memory[:, 0, 0], dtype=torch.long) + width
            )

        grid_y = torch.arange(
            height,
            dtype=torch.float32,
            device=memory.device,
        ).reshape(height, 1).expand(height, width)
        grid_x = torch.arange(
            width,
            dtype=torch.float32,
            device=memory.device,
        ).reshape(1, width).expand(height, width)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        scale = torch.stack((valid_width, valid_height), dim=1).reshape(
            -1, 1, 1, 2
        )
        proposals_grid = (grid.unsqueeze(0) + 0.5) / scale.float()
        width_height = torch.ones_like(proposals_grid) * (
            0.05 * (2.0**level)
        )
        proposals.append(
            torch.cat((proposals_grid, width_height), -1).reshape(
                batch_size, -1, 4
            )
        )
        current += height * width

    output_proposals = torch.cat(proposals, 1)
    valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(-1, keepdim=True)
    if unsigmoid:
        output_proposals = torch.log(
            output_proposals / (1 - output_proposals)
        )
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), float("inf")
            )
        output_proposals = output_proposals.masked_fill(
            ~valid, float("inf")
        )
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_padding_mask.unsqueeze(-1), 0.0
            )
        output_proposals = output_proposals.masked_fill(~valid, 0.0)

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), 0.0
        )
    output_memory = output_memory.masked_fill(~valid, 0.0)
    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


@contextmanager
def coreml_export_rewrites(*, feature_side: int) -> Iterator[None]:
    """Temporarily keep RF-DETR's export graph within Core ML rank limits."""

    transformer = importlib.import_module("rfdetr.models.transformer")
    deform_module = importlib.import_module(
        "rfdetr.models.ops.modules.ms_deform_attn"
    )
    cls = deform_module.MSDeformAttn
    original_proposals = transformer.gen_encoder_output_proposals
    original_forward = cls.forward

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
        if self.n_levels != 1 or input_length != feature_side * feature_side:
            raise ValueError(
                "Core ML RF-DETR rewrite requires one fixed feature level"
            )

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        head_dim = self.d_model // self.n_heads
        value = value.reshape(
            batch, input_length, self.n_heads, head_dim
        ).permute(0, 2, 3, 1)
        value = value.reshape(
            batch * self.n_heads, head_dim, feature_side, feature_side
        )

        offsets = self.sampling_offsets(query).reshape(
            batch, query_count, self.n_heads, self.n_points, 2
        )
        weights = self.attention_weights(query).reshape(
            batch, query_count, self.n_heads, self.n_points
        ).softmax(-1)
        reference = reference_points[:, :, 0, :]
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack(
                (
                    input_spatial_shapes[0, 1],
                    input_spatial_shapes[0, 0],
                )
            )
            locations = (
                reference[:, :, None, None, :]
                + offsets / normalizer[None, None, None, None, :]
            )
        elif reference_points.shape[-1] == 4:
            locations = (
                reference[:, :, None, None, :2]
                + offsets
                / self.n_points
                * reference[:, :, None, None, 2:]
                * 0.5
            )
        else:
            raise ValueError("reference point width must be 2 or 4")

        grid = (locations * 2 - 1).permute(0, 2, 1, 3, 4)
        grid = grid.reshape(
            batch * self.n_heads, query_count, self.n_points, 2
        )
        sampled = F.grid_sample(
            value,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        weights = weights.permute(0, 2, 1, 3).reshape(
            batch * self.n_heads, 1, query_count, self.n_points
        )
        output = (sampled * weights).sum(-1).reshape(
            batch, self.n_heads * head_dim, query_count
        )
        return self.output_proj(output.transpose(1, 2).contiguous())

    transformer.gen_encoder_output_proposals = _fixed_encoder_output_proposals
    cls.forward = forward
    try:
        yield
    finally:
        cls.forward = original_forward
        transformer.gen_encoder_output_proposals = original_proposals


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = np.abs(
        actual.astype(np.float32) - expected.astype(np.float32)
    )
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference))))
        if difference.size
        else 0.0,
    }


def _operator_counts(traced: torch.jit.ScriptModule) -> dict[str, int]:
    return dict(
        sorted(Counter(node.kind() for node in traced.inlined_graph.nodes()).items())
    )


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def export_model(
    *,
    weights: Path,
    output: Path,
    variant: str,
    resolution: int,
    seed: int,
    fp16: bool,
    allow_overwrite: bool,
    skip_runtime_validation: bool,
) -> dict[str, Any]:
    import coremltools as ct

    if resolution <= 0 or resolution % 12:
        raise ValueError("resolution must be a positive multiple of 12")
    if output.exists() and not allow_overwrite:
        raise FileExistsError(f"{output} exists; pass --allow-overwrite")
    dtype = np.float16 if fp16 else np.float32
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    feature_side = resolution // 12
    stages: dict[str, float] = {}

    started = time.perf_counter()
    wrapper: RFDETRExportWrapper = download_or_load_model(
        weights,
        fp16=fp16,
        variant=variant,
        resolution=resolution,
    )
    example = make_example(seed, fp16=fp16, resolution=resolution)
    with torch.inference_mode():
        reference = tuple(value.detach().cpu().numpy() for value in wrapper(example))
    stages["load_and_reference"] = time.perf_counter() - started

    started = time.perf_counter()
    with fixed_shape_as_tensor(), coreml_export_rewrites(
        feature_side=feature_side
    ), torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            (example,),
            strict=False,
            check_trace=False,
        )
        rewritten = tuple(
            value.detach().cpu().numpy() for value in traced(example)
        )
    stages["trace"] = time.perf_counter() - started
    rewrite_errors = {
        name: _error_metrics(actual, expected)
        for name, actual, expected in zip(
            ("boxes", "logits", "masks"),
            rewritten,
            reference,
            strict=True,
        )
    }
    if any(value["max_abs"] > 2e-4 for value in rewrite_errors.values()):
        raise ValueError(f"Core ML graph rewrite changed outputs: {rewrite_errors}")

    started = time.perf_counter()
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 3, resolution, resolution),
                dtype=dtype,
            )
        ],
        outputs=[
            ct.TensorType(name="boxes", dtype=dtype),
            ct.TensorType(name="logits", dtype=dtype),
            ct.TensorType(name="masks", dtype=dtype),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=precision,
    )
    stages["convert"] = time.perf_counter() - started
    model.author = "Lada Authors"
    model.license = "AGPL-3.0"
    model.short_description = "Jasna RF-DETR v6 mosaic segmentation"
    model.user_defined_metadata.update(
        {
            "model_id": "jasna-v6-coreml"
            if variant == "medium"
            else f"jasna-v6-{variant}-coreml",
            "architecture": f"rfdetr-seg-{variant}",
            "resolution": str(resolution),
            "preprocessing": "RGB NCHW ImageNet mean/std",
            "precision": "float16" if fp16 else "float32",
        }
    )

    _remove_existing(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.save(str(output))
    stages["save"] = time.perf_counter() - started

    runtime_errors: dict[str, dict[str, float]] | None = None
    if not skip_runtime_validation:
        started = time.perf_counter()
        prediction = model.predict(
            {"image": np.ascontiguousarray(example.numpy())}
        )
        runtime_errors = {
            name: _error_metrics(np.asarray(prediction[name]), expected)
            for name, expected in zip(
                ("boxes", "logits", "masks"), reference, strict=True
            )
        }
        stages["runtime_validation"] = time.perf_counter() - started

    operation_counts: dict[str, int] = {}
    for function in model._mil_program.functions.values():
        for operation in function.operations:
            operation_counts[operation.op_type] = (
                operation_counts.get(operation.op_type, 0) + 1
            )
    return {
        "success": True,
        "weights": str(weights),
        "output": str(output),
        "variant": variant,
        "resolution": resolution,
        "dtype": "float16" if fp16 else "float32",
        "outputs": {
            name: list(value.shape)
            for name, value in zip(
                ("boxes", "logits", "masks"), reference, strict=True
            )
        },
        "rewrite_errors": rewrite_errors,
        "runtime_errors": runtime_errors,
        "torch_operators": _operator_counts(traced),
        "coreml_operators": dict(sorted(operation_counts.items())),
        "stages": stages,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report or args.output.with_suffix(".report.json")
    try:
        report = export_model(
            weights=args.weights,
            output=args.output,
            variant=args.variant,
            resolution=args.resolution,
            seed=args.seed,
            fp16=args.fp16,
            allow_overwrite=args.allow_overwrite,
            skip_runtime_validation=args.skip_runtime_validation,
        )
    except Exception as exc:
        report = {
            "success": False,
            "failed": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Core ML package: {args.output}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
