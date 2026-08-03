#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Convert and benchmark the fixed-shape MIOH Native-HF 512 prototype.

The exported contract is one continuous temporal tensor::

    frames [1, input_frames * 8, 512, 512]
        -> rgb [1, outputs * 3, 512, 512]
        -> confidence [1, outputs, 512, 512]

An untrained model is useful for the architecture smoke test.  Its zero output
head is perturbed deterministically so Core ML cannot remove the HF branch as
an identity.  A checkpoint is never perturbed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer.model_native_hf import (
    NATIVE_HF_FRAME_CHANNELS,
    MiohNativeHF512,
    MiohNativeHF512ExportWrapper,
    NativeHF512Config,
    build_mioh_native_hf512,
    native_hf_parameter_count,
)


NATIVE_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/native-hf-512-smoke")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional mioh-native-hf-512-v5 training checkpoint",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="load state_dict instead of ema_state_dict from a checkpoint",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="device for the PyTorch latency reference",
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        help="trace and benchmark PyTorch without converting or running Core ML",
    )
    parser.add_argument("--skip-compute-plan", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def _config_from_payload(payload: object) -> NativeHF512Config:
    if not isinstance(payload, dict):
        return NativeHF512Config()
    raw = payload.get("config")
    if not isinstance(raw, dict):
        return NativeHF512Config()
    names = {field.name for field in fields(NativeHF512Config)}
    values = {name: raw[name] for name in names if name in raw}
    if "output_indices" in values:
        values["output_indices"] = tuple(int(value) for value in values["output_indices"])
    return NativeHF512Config(**values)


def load_model(
    checkpoint: Path | None, *, use_ema: bool, seed: int
) -> tuple[MiohNativeHF512, dict[str, object]]:
    torch.manual_seed(seed)
    if checkpoint is None:
        model = build_mioh_native_hf512()
        with torch.no_grad():
            # Preserve the complete architecture in the converted graph.  The
            # real model begins at exact global-base identity, but a fully zero
            # residual head is legal for the converter to constant-fold away.
            model.decoder.residual_head[-1].weight.normal_(0.0, 1e-4)
            model.decoder.detail_skip.weight.normal_(0.0, 1e-4)
            model.decoder.confidence_head[-1].weight.normal_(0.0, 1e-4)
        return model.eval(), {"checkpoint": None, "weights": "deterministic-smoke"}

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Training checkpoints contain optimizer metadata and pathlib values, so
    # this local, explicitly supplied checkpoint uses the legacy loader.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Native-HF checkpoint must be a dictionary")
    config = _config_from_payload(payload)
    model = build_mioh_native_hf512(config)
    state_name = "ema_state_dict" if use_ema and "ema_state_dict" in payload else "state_dict"
    state = payload.get(state_name)
    if not isinstance(state, dict):
        # Permit a raw state dictionary for architecture experiments.
        state = payload
        state_name = "raw"
    model.load_state_dict(state, strict=True)
    return model.eval(), {
        "checkpoint": str(checkpoint.resolve()),
        "weights": state_name,
        "step": int(payload.get("step", 0)) if state_name != "raw" else None,
    }


def sample_input(model: MiohNativeHF512, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frames = model.config.input_frames
    source = torch.rand(1, frames, 3, NATIVE_SIZE, NATIVE_SIZE, generator=generator)
    mask = torch.zeros(1, frames, 1, NATIVE_SIZE, NATIVE_SIZE)
    # A stable ROI is more representative than independent per-pixel masks and
    # exercises both exact source preservation and the restoration branch.
    for frame in range(frames):
        left = 144 + frame * 2
        top = 152 + frame
        mask[:, frame, :, top : top + 208, left : left + 224] = 1.0
    reliability = torch.full_like(mask, 0.95)
    noise = torch.randn(source.shape, generator=generator) * 0.015
    base = (source + mask * noise).clamp(0, 1)
    values = torch.cat((source, mask, reliability, base), dim=2)
    return values.reshape(1, frames * NATIVE_HF_FRAME_CHANNELS, NATIVE_SIZE, NATIVE_SIZE)


def latency_statistics(durations: list[float]) -> dict[str, float]:
    values = np.asarray(durations, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def resolve_torch_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return torch.device(requested)


def benchmark_torch(
    wrapper: torch.nn.Module,
    example: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    runs: int,
) -> dict[str, object]:
    # The model is constructed on CPU.  Avoid a redundant Module.to("cpu")
    # traversal so --skip-coreml remains a useful diagnostic even when a new
    # module accidentally shadows nn.Module._apply.
    if device.type != "cpu":
        wrapper = wrapper.to(device)
    wrapper = wrapper.eval()
    deployed = example.to(device)
    durations: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            wrapper(deployed)
            if device.type == "mps":
                torch.mps.synchronize()
        for _ in range(runs):
            started = time.perf_counter()
            wrapper(deployed)
            if device.type == "mps":
                torch.mps.synchronize()
            durations.append((time.perf_counter() - started) * 1_000)
    return {"device": str(device), **latency_statistics(durations)}


def remove_existing(path: Path, allow_overwrite: bool) -> None:
    if not path.exists():
        return
    if not allow_overwrite:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def convert_coreml(
    traced: torch.jit.ScriptModule,
    input_shape: tuple[int, ...],
    output: Path,
    *,
    allow_overwrite: bool,
    model: MiohNativeHF512,
):
    import coremltools as ct

    remove_existing(output, allow_overwrite)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="frames", shape=input_shape, dtype=np.float16)
        ],
        outputs=[
            ct.TensorType(name="rgb", dtype=np.float16),
            ct.TensorType(name="confidence", dtype=np.float16),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.user_defined_metadata["mioh.restorer"] = "native-hf-512"
    converted.user_defined_metadata["mioh.execution"] = "fixed-monolithic"
    converted.user_defined_metadata["mioh.input_frames"] = str(model.config.input_frames)
    converted.user_defined_metadata["mioh.native_size"] = str(NATIVE_SIZE)
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
        device = type(usage.preferred_compute_device).__name__
        devices[device] = devices.get(device, 0) + 1
    assigned = sum(devices.values())
    neural = sum(count for name, count in devices.items() if "Neural" in name)
    result: dict[str, object] = {
        "operation_count": operation_count,
        "operations_with_preferred_device": assigned,
        "operations_without_reported_preference": operation_count - assigned,
        "preferred_devices": devices,
        "ane_preferred_operations": neural,
    }
    if neural == 0:
        result["warning"] = "no operation is reported as preferring the Neural Engine"
    return result


def benchmark_coreml(model, values: np.ndarray, *, warmup: int, runs: int):
    inputs = {"frames": values}
    for _ in range(warmup):
        model.predict(inputs)
    durations: list[float] = []
    result = None
    for _ in range(runs):
        started = time.perf_counter()
        result = model.predict(inputs)
        durations.append((time.perf_counter() - started) * 1_000)
    if result is None:
        raise RuntimeError("Core ML benchmark produced no output")
    return result, latency_statistics(durations)


def numeric_report(
    result: dict[str, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for name, expected in zip(("rgb", "confidence"), reference, strict=True):
        got = np.asarray(result[name], dtype=np.float32)
        difference = got - expected.astype(np.float32, copy=False)
        mse = float(np.mean(difference**2))
        report[name] = {
            "maximum_absolute": float(np.max(np.abs(difference))),
            "mean_absolute": float(np.mean(np.abs(difference))),
            "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
        }
    return report


def main() -> int:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise ValueError("runs must be positive and warmup must be non-negative")
    model, weight_info = load_model(
        args.checkpoint, use_ema=not args.no_ema, seed=args.seed
    )
    wrapper = MiohNativeHF512ExportWrapper(model, clamp=True).eval()
    example = sample_input(model, args.seed + 1)
    with torch.inference_mode():
        reference_tensors = wrapper(example)
    reference = tuple(value.detach().cpu().numpy() for value in reference_tensors)

    trace_started = time.perf_counter()
    traced = torch.jit.trace(wrapper, example, check_trace=False, strict=False)
    trace_seconds = time.perf_counter() - trace_started

    report: dict[str, object] = {
        "format": "mioh-native-hf-512-smoke-v1",
        "native_size": NATIVE_SIZE,
        "input_shape": list(example.shape),
        "output_shapes": [list(value.shape) for value in reference_tensors],
        "parameters": native_hf_parameter_count(model),
        "config": {
            field.name: getattr(model.config, field.name)
            for field in fields(NativeHF512Config)
        },
        "weights": weight_info,
        "trace_seconds": trace_seconds,
    }

    device = resolve_torch_device(args.device)
    report["pytorch"] = benchmark_torch(
        wrapper,
        example,
        device=device,
        warmup=args.warmup,
        runs=args.runs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_coreml:
        package = args.output_dir / "mioh-native-hf-512-fp16.mlpackage"
        conversion_started = time.perf_counter()
        coreml = convert_coreml(
            traced,
            tuple(example.shape),
            package,
            allow_overwrite=args.allow_overwrite,
            model=model,
        )
        result, latency = benchmark_coreml(
            coreml,
            example.numpy().astype(np.float16),
            warmup=args.warmup,
            runs=args.runs,
        )
        coreml_report: dict[str, object] = {
            "package": str(package.resolve()),
            "conversion_and_first_load_seconds": time.perf_counter() - conversion_started,
            "latency": latency,
            "numeric": numeric_report(result, reference),
        }
        if not args.skip_compute_plan:
            try:
                coreml_report["compute_plan"] = compute_plan(package)
            except Exception as error:  # availability varies across beta SDKs
                coreml_report["compute_plan"] = {
                    "error": f"{type(error).__name__}: {error}"
                }
        report["coreml"] = coreml_report

    report_path = args.output_dir / "native-hf-512-smoke-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {report_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
