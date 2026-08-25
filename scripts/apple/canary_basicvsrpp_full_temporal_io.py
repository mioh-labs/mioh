#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Full BasicVSR++ packed/separate temporal-I/O regression canary.

Unlike the reduced single-branch canary, this runs the spatial encoder,
bidirectional flow, all four propagation sweeps, continuation boundaries, and
the reconstruction head on decoded MIDV-670 frames.  The two arms differ only
in the propagation input contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec
from scripts.apple.basicvsrpp_coreai_kernels import (
    build_deform_conv_kernel,
    build_flow_warp_kernel,
    build_grid_sample_kernel,
)
from scripts.apple.benchmark_basicvsrpp_variable_coreai import (
    BRANCHES,
    FEATURE_SIZE,
    IMAGE_SIZE,
    MID_CHANNELS,
    Reconstruction,
    SpatialEncoder,
)
from scripts.apple.export_basicvsrpp_coreai import (
    import_coreai,
    load_generator,
    save_program_asset,
    use_deform_conv_metal_kernel,
    use_flow_warp_metal_kernel,
    use_grid_sample_metal_kernel,
)
from scripts.apple.export_basicvsrpp_variable_chunk6 import (
    BidirectionalFlow6,
    PropagationContinue6,
    PropagationStart6,
)


CHUNK_SIZE = 6
FUSED_CHANNELS = MID_CHANNELS * 5


class Spatial6(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.encoder = SpatialEncoder(generator)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)


class Reconstruction6(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.reconstruction = Reconstruction(generator)

    def forward(self, frames: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.reconstruction(frames, features)


class SeparatePropagationStart6(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module, branch: str, components: int):
        super().__init__()
        self.components = components
        packed = PropagationStart6(generator, branch)
        self.initial = packed.initial
        self.first = packed.first
        self.later = packed.later

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        feature_count = CHUNK_SIZE * self.components
        features = values[:feature_count]
        flows = values[feature_count:]
        contexts = [
            torch.cat(
                features[index * self.components : (index + 1) * self.components],
                dim=1,
            )
            for index in range(CHUNK_SIZE)
        ]
        outputs = [self.initial(contexts[0])]
        outputs.append(self.first(contexts[1], outputs[-1], flows[0]))
        for index in range(2, CHUNK_SIZE):
            outputs.append(
                self.later(
                    contexts[index],
                    outputs[-1],
                    outputs[-2],
                    flows[index - 1],
                    flows[index - 2],
                )
            )
        return torch.cat(outputs, dim=0)


class SeparatePropagationContinue6(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module, branch: str, components: int):
        super().__init__()
        self.components = components
        self.later = copy.deepcopy(PropagationContinue6(generator, branch).later)

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        feature_count = CHUNK_SIZE * self.components
        features = values[:feature_count]
        state_n1 = values[feature_count]
        state_n2 = values[feature_count + 1]
        flows = values[feature_count + 2 : feature_count + 2 + CHUNK_SIZE]
        flow_previous = values[-1]
        contexts = [
            torch.cat(
                features[index * self.components : (index + 1) * self.components],
                dim=1,
            )
            for index in range(CHUNK_SIZE)
        ]
        outputs = []
        previous = state_n1
        older = state_n2
        previous_flow = flow_previous
        for index in range(CHUNK_SIZE):
            flow = flows[index]
            result = self.later(
                contexts[index], previous, older, flow, previous_flow
            )
            outputs.append(result)
            older, previous, previous_flow = previous, result, flow
        return torch.cat(outputs, dim=0)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def compile_asset(source: Path, output_dir: Path, architecture: str) -> Path:
    destination = output_dir / f"{source.stem}.{architecture}.aimodelc"
    remove_existing(destination)
    subprocess.run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(source),
            "--output",
            str(output_dir),
            "--platform",
            "macOS",
            "--min-deployment-version",
            "27.0",
            "--preferred-compute",
            "gpu",
            "--architecture",
            architecture,
        ],
        check=True,
    )
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    return destination


def export_asset(
    *,
    module: torch.nn.Module,
    examples: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    destination: Path,
    coreai_torch,
    kernels: tuple[Any, Any, Any],
) -> None:
    grid_kernel, deform_kernel, flow_kernel = kernels
    module = module.half().eval()
    with torch.no_grad(), ExitStack() as stack:
        stack.enter_context(use_grid_sample_metal_kernel(grid_kernel))
        stack.enter_context(use_deform_conv_metal_kernel(deform_kernel))
        stack.enter_context(use_flow_warp_metal_kernel(flow_kernel))
        exported = torch.export.export(module, examples)
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels([grid_kernel, deform_kernel, flow_kernel])
    converter.add_exported_program(
        exported, input_names=list(input_names), output_names=list(output_names)
    )
    remove_existing(destination)
    save_program_asset(converter.to_coreai(), destination)


def feature_names(components: int) -> tuple[str, ...]:
    return tuple(
        f"frame{frame}_feature{component}"
        for frame in range(CHUNK_SIZE)
        for component in range(components)
    )


def feature_examples(components: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.rand(
            1,
            MID_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
            generator=generator,
            dtype=torch.float16,
        )
        for _ in range(CHUNK_SIZE * components)
    )


def export_all(
    generator: torch.nn.Module,
    output_dir: Path,
    architecture: str,
) -> dict[str, dict[str, Path]]:
    _coreai, coreai_torch = import_coreai()
    kernels = (
        build_grid_sample_kernel(coreai_torch),
        build_deform_conv_kernel(coreai_torch),
        build_flow_warp_kernel(coreai_torch),
    )
    source_root = output_dir / "source"
    compiled_root = output_dir / "compiled"
    source_root.mkdir(parents=True, exist_ok=True)
    compiled_root.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, Path]] = {"shared": {}, "packed": {}, "separate": {}}
    frames6 = torch.rand(6, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16)
    frames7 = torch.rand(7, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float16)
    features6 = torch.rand(
        6, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE, dtype=torch.float16
    )
    shared_specs = (
        ("spatial6", Spatial6(generator), (frames6,), ("frames",), ("features",)),
        (
            "flow6",
            BidirectionalFlow6(generator),
            (frames7,),
            ("frames",),
            ("backward", "forward"),
        ),
        (
            "reconstruction6",
            Reconstruction6(generator),
            (frames6, features6),
            ("frames", "features"),
            ("restored",),
        ),
    )
    for name, module, examples, inputs, outputs in shared_specs:
        source = source_root / f"{name}.aimodel"
        export_asset(
            module=module,
            examples=examples,
            input_names=inputs,
            output_names=outputs,
            destination=source,
            coreai_torch=coreai_torch,
            kernels=kernels,
        )
        assets["shared"][name] = compile_asset(source, compiled_root, architecture)

    feature = torch.rand(
        1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE, dtype=torch.float16
    )
    flow = torch.rand(1, 2, FEATURE_SIZE, FEATURE_SIZE, dtype=torch.float16) - 0.5
    for branch_index, branch in enumerate(BRANCHES):
        components = branch_index + 1
        context = torch.rand(
            CHUNK_SIZE,
            components * MID_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
            dtype=torch.float16,
        )
        flows5 = flow.repeat(CHUNK_SIZE - 1, 1, 1, 1)
        flows6 = flow.repeat(CHUNK_SIZE, 1, 1, 1)
        packed_specs = (
            (
                f"{branch}_start6",
                PropagationStart6(generator, branch),
                (context, flows5),
                ("contexts", "flows"),
            ),
            (
                f"{branch}_continue6",
                PropagationContinue6(generator, branch),
                (context, feature, feature.clone(), flows6, flow),
                ("contexts", "state_n1", "state_n2", "flows", "flow_previous"),
            ),
        )
        for name, module, examples, inputs in packed_specs:
            source = source_root / f"packed-{name}.aimodel"
            export_asset(
                module=module,
                examples=examples,
                input_names=inputs,
                output_names=("features",),
                destination=source,
                coreai_torch=coreai_torch,
                kernels=kernels,
            )
            assets["packed"][name] = compile_asset(source, compiled_root, architecture)

        components_examples = feature_examples(components, 100 + branch_index)
        separate_start_examples = components_examples + tuple(
            flow.clone() for _ in range(CHUNK_SIZE - 1)
        )
        separate_start_inputs = feature_names(components) + tuple(
            f"flow{index}" for index in range(CHUNK_SIZE - 1)
        )
        start_name = f"{branch}_start6"
        start_source = source_root / f"separate-{start_name}.aimodel"
        export_asset(
            module=SeparatePropagationStart6(generator, branch, components),
            examples=separate_start_examples,
            input_names=separate_start_inputs,
            output_names=("features",),
            destination=start_source,
            coreai_torch=coreai_torch,
            kernels=kernels,
        )
        assets["separate"][start_name] = compile_asset(
            start_source, compiled_root, architecture
        )

        separate_continue_examples = (
            components_examples
            + (feature, feature.clone())
            + tuple(flow.clone() for _ in range(CHUNK_SIZE))
            + (flow.clone(),)
        )
        separate_continue_inputs = (
            feature_names(components)
            + ("state_n1", "state_n2")
            + tuple(f"flow{index}" for index in range(CHUNK_SIZE))
            + ("flow_previous",)
        )
        continue_name = f"{branch}_continue6"
        continue_source = source_root / f"separate-{continue_name}.aimodel"
        export_asset(
            module=SeparatePropagationContinue6(generator, branch, components),
            examples=separate_continue_examples,
            input_names=separate_continue_inputs,
            output_names=("features",),
            destination=continue_source,
            coreai_torch=coreai_torch,
            kernels=kernels,
        )
        assets["separate"][continue_name] = compile_asset(
            continue_source, compiled_root, architecture
        )
    return assets


class Backend:
    def __init__(self, assets: dict[str, Path], mode: str, runner: Path):
        self.mode = mode
        self.runner = runner
        self.assets = assets
        self.runtimes: dict[str, CompiledCoreAIRuntime] = {}

    def call(self, name: str, values: dict[str, np.ndarray], outputs: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
        runtime = self.runtimes.get(name)
        if runtime is None:
            runtime = CompiledCoreAIRuntime(
                self.assets[name],
                tuple(TensorSpec(key, tuple(value.shape)) for key, value in values.items()),
                tuple(TensorSpec(key, shape) for key, shape in outputs.items()),
                runner_path=str(self.runner),
            )
            self.runtimes[name] = runtime
        return runtime.infer(values)

    def close(self) -> None:
        for runtime in self.runtimes.values():
            runtime.close()
        self.runtimes.clear()


def run_full(
    *, mode: str, frames: np.ndarray, shared: dict[str, Path], propagation: dict[str, Path], runner: Path
) -> np.ndarray:
    assets = {**shared, **propagation}
    backend = Backend(assets, mode, runner)
    frame_count = frames.shape[0]
    if frame_count < 12 or frame_count % CHUNK_SIZE:
        raise ValueError("full canary requires a multiple of six frames, at least 12")
    features: dict[str, list[np.ndarray]] = {"spatial": [None] * frame_count}  # type: ignore[list-item]
    try:
        for start in range(0, frame_count, CHUNK_SIZE):
            chunk = np.ascontiguousarray(frames[start : start + CHUNK_SIZE])
            result = backend.call(
                "spatial6", {"frames": chunk}, {"features": (6, 64, 64, 64)}
            )["features"]
            for offset in range(CHUNK_SIZE):
                features["spatial"][start + offset] = result[offset : offset + 1]

        backward = np.empty((frame_count - 1, 2, 64, 64), dtype=np.float16)
        forward = np.empty_like(backward)
        for start in range(0, frame_count - 1, CHUNK_SIZE):
            indices = [min(start + offset, frame_count - 1) for offset in range(7)]
            chunk = np.ascontiguousarray(frames[indices])
            result = backend.call(
                "flow6",
                {"frames": chunk},
                {"backward": (6, 2, 64, 64), "forward": (6, 2, 64, 64)},
            )
            valid = min(CHUNK_SIZE, frame_count - 1 - start)
            backward[start : start + valid] = result["backward"][:valid]
            forward[start : start + valid] = result["forward"][:valid]

        for branch_index, branch in enumerate(BRANCHES):
            is_backward = branch.startswith("backward")
            indices = list(reversed(range(frame_count))) if is_backward else list(range(frame_count))
            directional = backward if is_backward else forward
            components = branch_index + 1
            produced: dict[int, np.ndarray] = {}
            for chunk_start in range(0, frame_count, CHUNK_SIZE):
                chunk_indices = indices[chunk_start : chunk_start + CHUNK_SIZE]
                context_parts = [
                    [features["spatial"][frame_index]]
                    + [features[name][frame_index] for name in BRANCHES[:branch_index]]
                    for frame_index in chunk_indices
                ]
                values: dict[str, np.ndarray] = {}
                name = f"{branch}_{'start6' if chunk_start == 0 else 'continue6'}"
                if mode == "packed":
                    values["contexts"] = np.ascontiguousarray(
                        np.concatenate(
                            [np.concatenate(parts, axis=1) for parts in context_parts],
                            axis=0,
                        )
                    )
                else:
                    for frame_offset, parts in enumerate(context_parts):
                        for component, value in enumerate(parts):
                            values[f"frame{frame_offset}_feature{component}"] = np.ascontiguousarray(value)

                if chunk_start == 0:
                    flow_values = []
                    for position in range(1, CHUNK_SIZE):
                        frame_index = indices[position]
                        flow_index = frame_index if is_backward else frame_index - 1
                        flow_values.append(directional[flow_index : flow_index + 1])
                    if mode == "packed":
                        values["flows"] = np.ascontiguousarray(np.concatenate(flow_values))
                    else:
                        for index, value in enumerate(flow_values):
                            values[f"flow{index}"] = np.ascontiguousarray(value)
                else:
                    previous_frame = indices[chunk_start - 1]
                    older_frame = indices[chunk_start - 2]
                    values["state_n1"] = np.ascontiguousarray(produced[previous_frame])
                    values["state_n2"] = np.ascontiguousarray(produced[older_frame])
                    flow_values = []
                    for offset in range(CHUNK_SIZE):
                        frame_index = indices[chunk_start + offset]
                        flow_index = frame_index if is_backward else frame_index - 1
                        flow_values.append(directional[flow_index : flow_index + 1])
                    if mode == "packed":
                        values["flows"] = np.ascontiguousarray(np.concatenate(flow_values))
                    else:
                        for index, value in enumerate(flow_values):
                            values[f"flow{index}"] = np.ascontiguousarray(value)
                    previous_flow_index = previous_frame if is_backward else previous_frame - 1
                    values["flow_previous"] = np.ascontiguousarray(
                        directional[previous_flow_index : previous_flow_index + 1]
                    )
                output = backend.call(
                    name, values, {"features": (6, 64, 64, 64)}
                )["features"]
                for offset, frame_index in enumerate(chunk_indices):
                    produced[frame_index] = output[offset : offset + 1]
            features[branch] = [produced[index] for index in range(frame_count)]

        restored = []
        for start in range(0, frame_count, CHUNK_SIZE):
            fusion = np.ascontiguousarray(
                np.concatenate(
                    [
                        np.concatenate(
                            [features["spatial"][index]]
                            + [features[name][index] for name in BRANCHES],
                            axis=1,
                        )
                        for index in range(start, start + CHUNK_SIZE)
                    ],
                    axis=0,
                )
            )
            result = backend.call(
                "reconstruction6",
                {
                    "frames": np.ascontiguousarray(frames[start : start + CHUNK_SIZE]),
                    "features": fusion,
                },
                {"restored": (6, 3, 256, 256)},
            )["restored"]
            restored.append(result)
        return np.concatenate(restored, axis=0)
    finally:
        backend.close()


def decode_frames(path: Path, frame_count: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    frames = []
    try:
        while len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)
    finally:
        capture.release()
    if len(frames) != frame_count:
        raise RuntimeError(f"expected {frame_count} decoded frames, got {len(frames)}")
    return np.ascontiguousarray(np.stack(frames).astype(np.float16))


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = reference.astype(np.float32) - actual.astype(np.float32)
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    peak = max(1.0, float(np.max(np.abs(reference))))
    return {
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": rmse,
        "relative_psnr_db": float("inf") if rmse == 0 else 20 * math.log10(peak / rmse),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "output/evaluations/basicvsrpp-large-roi-step21000-vs-step27000-midv670-20260818/"
            "MIDV-670-2h18m00s-source-10s.mp4"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_weights/basicvsrpp-v1.2-detail-recovery-30000-ema.pth"),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("build/macos-standalone/mioh.app/Contents/Resources/bin/lada-coreai-runner"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/mioh-basicvsrpp-full-temporal-io")
    )
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument("--frames", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.source, args.checkpoint, args.runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    remove_existing(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if args.frames < 12 or args.frames % CHUNK_SIZE:
        raise ValueError("--frames must be a multiple of six and at least 12")
    frames = decode_frames(args.source, args.frames)
    wrapper = load_generator(args.checkpoint)
    generator = wrapper.generator
    assets = export_all(generator, args.output_dir, args.architecture)

    started = time.perf_counter()
    packed = run_full(
        mode="packed",
        frames=frames,
        shared=assets["shared"],
        propagation=assets["packed"],
        runner=args.runner,
    )
    packed_seconds = time.perf_counter() - started
    started = time.perf_counter()
    separate = run_full(
        mode="separate",
        frames=frames,
        shared=assets["shared"],
        propagation=assets["separate"],
        runner=args.runner,
    )
    separate_seconds = time.perf_counter() - started
    comparison = metrics(packed, separate)
    generator = generator.eval().to("mps")
    reference_input = torch.from_numpy(frames).unsqueeze(0).to(
        "mps", dtype=torch.float16
    )
    with torch.inference_mode():
        reference = generator(reference_input).squeeze(0).float().cpu().numpy()
    packed_reference = metrics(reference, packed)
    separate_reference = metrics(reference, separate)
    regression = (
        comparison["relative_psnr_db"] < 45.0
        or packed_reference["relative_psnr_db"] < 45.0
        or separate_reference["relative_psnr_db"] < 45.0
    )
    report: dict[str, object] = {
        "success": not regression,
        "separateInputRegressionObserved": regression,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "coreai_torch": package_version("coreai-torch"),
        "coreai_core": package_version("coreai-core"),
        "architecture": args.architecture,
        "source": str(args.source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpointSHA256": sha256(args.checkpoint),
        "decodedFrameCount": int(frames.shape[0]),
        "packedSeconds": packed_seconds,
        "separateSeconds": separate_seconds,
        "packedVsSeparate": comparison,
        "packedVsPyTorch": packed_reference,
        "separateVsPyTorch": separate_reference,
        "packedOutputMean": float(packed.astype(np.float32).mean()),
        "separateOutputMean": float(separate.astype(np.float32).mean()),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path.resolve()}")
    return 0 if not regression else 2


if __name__ == "__main__":
    raise SystemExit(main())
