#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Stage 0 Core ML conversion and execution benchmark for MiohRestorer V5."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer.model_v5 import (
    FOLDED_FRAME_CHANNELS,
    FRAME_CHANNELS,
    NUM_INPUT_FRAMES,
    MiohRestorerV5,
    MiohRestorerV5Config,
    MiohRestorerV5DecoderExportWrapper,
    MiohRestorerV5EncoderExportWrapper,
    MiohRestorerV5ExportWrapper,
    MiohRestorerV5StatefulExportWrapper,
    V5_BUCKETS,
    flatten_encoded_window,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v5-stage0"))
    parser.add_argument("--variant", choices=("q", "s"), default="s")
    parser.add_argument("--context-frames", type=int, choices=(5, 7, 9))
    parser.add_argument(
        "--factorized-coarse",
        action="store_true",
        help="experimental V5-S 18-candidate coarse search; not quality-equivalent",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(str(value) for value in V5_BUCKETS),
        help="comma-separated fixed Core ML input sizes",
    )
    parser.add_argument(
        "--execution",
        choices=("monolithic", "split", "stateful", "all"),
        default="all",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--skip-compute-plan", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size not in V5_BUCKETS for size in sizes):
        raise ValueError(f"sizes must be selected from {V5_BUCKETS}")
    return sizes


def build_model(
    variant: str,
    context_frames: int | None,
    *,
    factorized_coarse: bool,
) -> MiohRestorerV5:
    torch.manual_seed(0)
    if variant == "q":
        model = MiohRestorerV5.quality(context_frames=context_frames or 7)
    else:
        model = MiohRestorerV5.shipping(
            context_frames=context_frames or 9,
            factorized_coarse=factorized_coarse,
        )
    with torch.no_grad():
        # Exercise the heads instead of allowing the converter to remove the
        # zero-initialized identity path.
        model.decoder.base_head[-1].weight.normal_(0.0, 1e-4)
        model.decoder.texture_head[-1].weight.normal_(0.0, 1e-4)
        model.decoder.confidence_head[-1].weight.normal_(0.0, 1e-4)
    return model.eval()


def sample_window(size: int) -> torch.Tensor:
    values = torch.rand(1, NUM_INPUT_FRAMES, FRAME_CHANNELS, size, size)
    values[:, :, 3:4] = (values[:, :, 3:4] > 0.65).float()
    values[:, :, 4:5] = 1.0
    return values


def remove_existing(path: Path, allow: bool) -> None:
    if not path.exists():
        return
    if not allow:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def convert(
    wrapper: torch.nn.Module,
    example: tuple[torch.Tensor, ...],
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    output: Path,
    allow_overwrite: bool,
    metadata: dict[str, str],
):
    import coremltools as ct

    remove_existing(output, allow_overwrite)
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    converted = ct.convert(
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
    for key, value in metadata.items():
        converted.user_defined_metadata[key] = value
    converted.save(str(output))
    return converted


def all_operations(block):
    for operation in block.operations:
        yield operation
        for child in getattr(operation, "blocks", ()):
            yield from all_operations(child)


def compute_plan(path: Path) -> dict[str, object]:
    import coremltools as ct

    compiled = path.with_suffix(".mlmodelc")
    if compiled.exists():
        shutil.rmtree(compiled)
    ct.models.utils.compile_model(str(path), destination_path=str(compiled))
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.ALL
    )
    function = plan.model_structure.program.functions["main"]
    devices: dict[str, int] = {}
    operation_count = 0
    for operation in all_operations(function.block):
        operation_count += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        name = type(usage.preferred_compute_device).__name__
        devices[name] = devices.get(name, 0) + 1
    return {"operation_count": operation_count, "preferred_devices": devices}


def benchmark_predict(model, inputs: dict[str, np.ndarray], runs: int):
    for _ in range(3):
        model.predict(inputs)
    durations = []
    result = None
    for _ in range(runs):
        started = time.perf_counter()
        result = model.predict(inputs)
        durations.append((time.perf_counter() - started) * 1_000)
    assert result is not None
    values = np.asarray(durations)
    return result, {
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def numeric_report(result: dict[str, np.ndarray], reference: tuple[np.ndarray, ...], names: tuple[str, ...]):
    report: dict[str, dict[str, float]] = {}
    for name, expected in zip(names, reference, strict=True):
        difference = np.abs(np.asarray(result[name], dtype=np.float32) - expected)
        report[name] = {
            "maximum_absolute": float(difference.max()),
            "mean_absolute": float(difference.mean()),
            "psnr": float(-10 * np.log10(max(float(np.mean(difference**2)), 1e-12))),
        }
    return report


def tensor_tuple(values: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, ...]:
    return tuple(value.detach().cpu().numpy() for value in values)


def flat_window(values: torch.Tensor) -> torch.Tensor:
    return values.flatten(1, 2)


def split_examples(model: MiohRestorerV5, values: torch.Tensor):
    with torch.no_grad():
        encoded = model.encode_window(values)
    return flatten_encoded_window(encoded)


def state_examples(model: MiohRestorerV5, values: torch.Tensor):
    with torch.no_grad():
        encoded = model.encode_window(values)
        states = tuple(
            torch.cat([frame[index] for frame in encoded[:-1]], dim=1)
            for index in range(5)
        )
    return (values[:, -1], *states)


def benchmark_mps(wrapper: torch.nn.Module, example: tuple[torch.Tensor, ...], runs: int):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    wrapper = wrapper.to("mps").eval()
    deployed = tuple(value.to("mps") for value in example)
    with torch.inference_mode():
        for _ in range(3):
            wrapper(*deployed)
            torch.mps.synchronize()
        durations = []
        for _ in range(runs):
            started = time.perf_counter()
            result = wrapper(*deployed)
            torch.mps.synchronize()
            durations.append((time.perf_counter() - started) * 1_000)
    values = np.asarray(durations)
    return {
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def run_monolithic(model, values, root: Path, args) -> dict[str, object]:
    wrapper = MiohRestorerV5ExportWrapper(model).eval()
    example = (flat_window(values),)
    with torch.no_grad():
        reference = tensor_tuple(wrapper(*example))
    report: dict[str, object] = {}
    if args.mps:
        report["mps"] = benchmark_mps(wrapper, example, args.runs)
    if args.skip_coreml:
        return report
    path = root / "monolithic.mlpackage"
    coreml = convert(
        wrapper,
        example,
        input_names=("frames",),
        output_names=("rgb", "confidence"),
        output=path,
        allow_overwrite=args.allow_overwrite,
        metadata={"mioh.restorer": "v5", "mioh.execution": "monolithic"},
    )
    result, latency = benchmark_predict(
        coreml, {"frames": example[0].numpy()}, args.runs
    )
    report.update(
        {
            "packages": [str(path)],
            "latency": latency,
            "numeric": numeric_report(result, reference, ("rgb", "confidence")),
        }
    )
    if not args.skip_compute_plan:
        report["compute_plan"] = compute_plan(path)
    return report


def run_split(model, values, root: Path, args) -> dict[str, object]:
    encoder_wrapper = MiohRestorerV5EncoderExportWrapper(model.encoder).eval()
    decoder_wrapper = MiohRestorerV5DecoderExportWrapper(model.decoder).eval()
    encoder_example = (values[:, -1],)
    decoder_example = split_examples(model, values)
    with torch.no_grad():
        decoder_reference = tensor_tuple(decoder_wrapper(*decoder_example))
    report: dict[str, object] = {}
    if args.skip_coreml:
        return report
    encoder_path = root / "encoder.mlpackage"
    decoder_path = root / "decoder.mlpackage"
    encoder_model = convert(
        encoder_wrapper,
        encoder_example,
        input_names=("frame",),
        output_names=("packed", "half", "quarter", "eighth", "sixteenth"),
        output=encoder_path,
        allow_overwrite=args.allow_overwrite,
        metadata={"mioh.restorer": "v5", "mioh.execution": "split-encoder"},
    )
    decoder_model = convert(
        decoder_wrapper,
        decoder_example,
        input_names=("packed", "half", "quarter", "eighth", "sixteenth"),
        output_names=("rgb", "confidence"),
        output=decoder_path,
        allow_overwrite=args.allow_overwrite,
        metadata={"mioh.restorer": "v5", "mioh.execution": "split-decoder"},
    )
    encoded_input = {"frame": encoder_example[0].numpy()}
    decoder_input = {
        name: value.numpy()
        for name, value in zip(
            ("packed", "half", "quarter", "eighth", "sixteenth"),
            decoder_example,
            strict=True,
        )
    }
    _encoded, encoder_latency = benchmark_predict(encoder_model, encoded_input, args.runs)
    result, decoder_latency = benchmark_predict(decoder_model, decoder_input, args.runs)
    encoded_frames = [
        encoder_model.predict({"frame": values[:, index].numpy()})
        for index in range(NUM_INPUT_FRAMES)
    ]
    end_to_end_decoder_input = {
        name: np.concatenate([frame[name] for frame in encoded_frames], axis=1)
        for name in ("packed", "half", "quarter", "eighth", "sixteenth")
    }
    end_to_end_result = decoder_model.predict(end_to_end_decoder_input)
    report.update(
        {
            "packages": [str(encoder_path), str(decoder_path)],
            "encoder_latency": encoder_latency,
            "decoder_latency": decoder_latency,
            "steady_state_median_ms": encoder_latency["median_ms"] + decoder_latency["median_ms"],
            "numeric": numeric_report(result, decoder_reference, ("rgb", "confidence")),
            "end_to_end_numeric": numeric_report(
                end_to_end_result,
                decoder_reference,
                ("rgb", "confidence"),
            ),
        }
    )
    if not args.skip_compute_plan:
        report["compute_plan"] = {
            "encoder": compute_plan(encoder_path),
            "decoder": compute_plan(decoder_path),
        }
    return report


def run_stateful(model, values, root: Path, args) -> dict[str, object]:
    wrapper = MiohRestorerV5StatefulExportWrapper(model).eval()
    example = state_examples(model, values)
    names_in = (
        "current_frame",
        "packed_state",
        "half_state",
        "quarter_state",
        "eighth_state",
        "sixteenth_state",
    )
    names_out = (
        "rgb",
        "confidence",
        "next_packed_state",
        "next_half_state",
        "next_quarter_state",
        "next_eighth_state",
        "next_sixteenth_state",
    )
    with torch.no_grad():
        reference = tensor_tuple(wrapper(*example))
    report: dict[str, object] = {}
    if args.mps:
        report["mps"] = benchmark_mps(wrapper, example, args.runs)
    if args.skip_coreml:
        return report
    path = root / "stateful.mlpackage"
    coreml = convert(
        wrapper,
        example,
        input_names=names_in,
        output_names=names_out,
        output=path,
        allow_overwrite=args.allow_overwrite,
        metadata={"mioh.restorer": "v5", "mioh.execution": "stateful"},
    )
    inputs = {
        name: value.numpy() for name, value in zip(names_in, example, strict=True)
    }
    result, latency = benchmark_predict(coreml, inputs, args.runs)
    report.update(
        {
            "packages": [str(path)],
            "latency": latency,
            "numeric": numeric_report(result, reference, names_out),
        }
    )
    if not args.skip_compute_plan:
        report["compute_plan"] = compute_plan(path)
    return report


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    sizes = parse_sizes(args.sizes)
    executions = (
        ("monolithic", "split", "stateful")
        if args.execution == "all"
        else (args.execution,)
    )
    if args.variant == "q" and any(value != "monolithic" for value in executions):
        raise ValueError("split and stateful Stage 0 contracts are V5-S only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final: dict[str, object] = {"variant": args.variant, "sizes": {}}
    for size in sizes:
        model = build_model(
            args.variant,
            args.context_frames,
            factorized_coarse=args.factorized_coarse,
        )
        values = sample_window(size)
        size_root = args.output_dir / f"{args.variant}-{size}"
        size_root.mkdir(parents=True, exist_ok=True)
        size_report: dict[str, object] = {
            "parameters": parameter_count(model),
            "context_frames": model.config.context_frames,
            "outputs": len(model.config.output_indices),
            "coarse_mode": model.config.coarse_mode,
        }
        for execution in executions:
            print(f"V5-{args.variant.upper()} {size} {execution}", flush=True)
            if execution == "monolithic":
                result = run_monolithic(model, values, size_root, args)
            elif execution == "split":
                result = run_split(model, values, size_root, args)
            else:
                result = run_stateful(model, values, size_root, args)
            size_report[execution] = result
            print(json.dumps(result, indent=2), flush=True)
        final["sizes"][str(size)] = size_report
    report_path = args.output_dir / "stage0-report.json"
    report_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
