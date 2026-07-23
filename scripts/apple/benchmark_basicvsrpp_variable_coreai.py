#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export and benchmark an exact, host-scheduled variable-length BasicVSR++.

This probe keeps the trained BasicVSR++ operators and weights intact, but splits
the temporal Python loops into fifteen fixed-shape Core AI assets:

* one per-frame spatial encoder;
* one adjacent-frame SPyNet flow estimator;
* first-position, second-position and later-position recurrent steps for each
  of the four propagation branches; and
* one per-frame reconstruction head.

The host may then execute any temporal length without padding to T18/T36/T90.
Intermediate tensors deliberately round-trip through the current Core AI Python
runtime.  The benchmark therefore measures the implementation that can be built
today, including host dispatch and transfer overhead; a future fused Swift
runner may retain the same tensors in accelerator-visible storage.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import mmap
import shutil
import struct
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .basicvsrpp_coreai_kernels import (
        build_deform_conv_kernel,
        build_grid_sample_kernel,
    )
    from .export_basicvsrpp_coreai import (
        import_coreai,
        load_generator,
        save_program_asset,
        use_deform_conv_metal_kernel,
        use_grid_sample_metal_kernel,
    )
except ImportError:
    from basicvsrpp_coreai_kernels import (  # type: ignore[import-not-found]
        build_deform_conv_kernel,
        build_grid_sample_kernel,
    )
    from export_basicvsrpp_coreai import (  # type: ignore[import-not-found]
        import_coreai,
        load_generator,
        save_program_asset,
        use_deform_conv_metal_kernel,
        use_grid_sample_metal_kernel,
    )

from lada.models.basicvsrpp.mmagic.flow_warp import flow_warp


BRANCHES = ("backward_1", "forward_1", "backward_2", "forward_2")
MID_CHANNELS = 64
IMAGE_SIZE = 256
FEATURE_SIZE = 64


class SpatialEncoder(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.feat_extract = copy.deepcopy(generator.feat_extract)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.feat_extract(frame)


class PairFlow(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.spynet = copy.deepcopy(generator.spynet)

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        ref = F.interpolate(ref, scale_factor=0.25, mode="bicubic")
        supp = F.interpolate(supp, scale_factor=0.25, mode="bicubic")
        return self.spynet(ref, supp)


class _PropagationBase(torch.nn.Module):
    """Shared weights and helpers for one BasicVSR++ propagation branch."""

    def __init__(self, generator: torch.nn.Module, branch: str):
        super().__init__()
        self.deform_align = copy.deepcopy(generator.deform_align[branch])
        self.backbone = copy.deepcopy(generator.backbone[branch])

    def finish(self, context: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
        return aligned + self.backbone(torch.cat((context, aligned), dim=1))


class PropagationInit(_PropagationBase):
    """First sequence position: no temporal state and no alignment."""

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        aligned = torch.zeros_like(context[:, :MID_CHANNELS])
        return self.finish(context, aligned)


class PropagationFirst(_PropagationBase):
    """Second sequence position: first-order state only."""

    def forward(
        self,
        context: torch.Tensor,
        state_n1: torch.Tensor,
        flow_n1: torch.Tensor,
    ) -> torch.Tensor:
        state_n2 = torch.zeros_like(state_n1)
        flow_n2 = torch.zeros_like(flow_n1)
        cond_n1 = flow_warp(state_n1, flow_n1.permute(0, 2, 3, 1))
        cond_n2 = torch.zeros_like(cond_n1)
        condition = torch.cat((cond_n1, context[:, :MID_CHANNELS], cond_n2), dim=1)
        aligned = self.deform_align(
            torch.cat((state_n1, state_n2), dim=1),
            condition,
            flow_n1,
            flow_n2,
        )
        return self.finish(context, aligned)


class PropagationLater(_PropagationBase):
    """Third and later positions: full second-order propagation."""

    def forward(
        self,
        context: torch.Tensor,
        state_n1: torch.Tensor,
        state_n2: torch.Tensor,
        flow_n1: torch.Tensor,
        flow_previous: torch.Tensor,
    ) -> torch.Tensor:
        flow_n2 = flow_n1 + flow_warp(
            flow_previous, flow_n1.permute(0, 2, 3, 1)
        )
        cond_n1 = flow_warp(state_n1, flow_n1.permute(0, 2, 3, 1))
        cond_n2 = flow_warp(state_n2, flow_n2.permute(0, 2, 3, 1))
        condition = torch.cat((cond_n1, context[:, :MID_CHANNELS], cond_n2), dim=1)
        aligned = self.deform_align(
            torch.cat((state_n1, state_n2), dim=1),
            condition,
            flow_n1,
            flow_n2,
        )
        return self.finish(context, aligned)


class Reconstruction(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.reconstruction = copy.deepcopy(generator.reconstruction)
        self.upsample1 = copy.deepcopy(generator.upsample1)
        self.upsample2 = copy.deepcopy(generator.upsample2)
        self.conv_hr = copy.deepcopy(generator.conv_hr)
        self.conv_last = copy.deepcopy(generator.conv_last)
        self.lrelu = copy.deepcopy(generator.lrelu)

    def forward(self, frame: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        result = self.reconstruction(features)
        result = self.lrelu(self.upsample1(result))
        result = self.lrelu(self.upsample2(result))
        result = self.lrelu(self.conv_hr(result))
        return frame + self.conv_last(result)


@dataclass(frozen=True)
class AssetSpec:
    name: str
    module: torch.nn.Module
    input_names: tuple[str, ...]
    output_name: str
    examples: tuple[torch.Tensor, ...]


def _rand(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(shape, generator=generator, dtype=torch.float16)


def build_specs(generator: torch.nn.Module) -> list[AssetSpec]:
    frame = _rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), 1)
    feature = _rand((1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), 2)
    flow = _rand((1, 2, FEATURE_SIZE, FEATURE_SIZE), 3) - 0.5
    specs = [
        AssetSpec(
            "spatial",
            SpatialEncoder(generator),
            ("frame",),
            "feature",
            (frame,),
        ),
        AssetSpec(
            "flow",
            PairFlow(generator),
            ("ref", "supp"),
            "flow",
            (frame, frame.clone()),
        ),
    ]
    for index, branch in enumerate(BRANCHES):
        context = _rand(
            (1, MID_CHANNELS * (index + 1), FEATURE_SIZE, FEATURE_SIZE),
            10 + index,
        )
        specs.extend(
            (
                AssetSpec(
                    f"{branch}_init",
                    PropagationInit(generator, branch),
                    ("context",),
                    "feature",
                    (context,),
                ),
                AssetSpec(
                    f"{branch}_first",
                    PropagationFirst(generator, branch),
                    ("context", "state_n1", "flow_n1"),
                    "feature",
                    (context, feature, flow),
                ),
                AssetSpec(
                    f"{branch}_later",
                    PropagationLater(generator, branch),
                    (
                        "context",
                        "state_n1",
                        "state_n2",
                        "flow_n1",
                        "flow_previous",
                    ),
                    "feature",
                    (context, feature, feature.clone(), flow, flow.clone()),
                ),
            )
        )
    specs.append(
        AssetSpec(
            "reconstruction",
            Reconstruction(generator),
            ("frame", "features"),
            "restored",
            (
                frame,
                _rand(
                    (1, MID_CHANNELS * 5, FEATURE_SIZE, FEATURE_SIZE), 30
                ),
            ),
        )
    )
    return specs


def export_assets(
    checkpoint: Path, output_dir: Path, *, overwrite: bool
) -> dict[str, Path]:
    _coreai, coreai_torch = import_coreai()
    generator = load_generator(checkpoint).generator
    specs = build_specs(generator)
    grid_kernel = build_grid_sample_kernel(coreai_torch)
    deform_kernel = build_deform_conv_kernel(coreai_torch)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for spec in specs:
        destination = output_dir / f"basicvsrpp-variable-{spec.name}.aimodel"
        if destination.exists():
            if not overwrite:
                outputs[spec.name] = destination
                continue
            shutil.rmtree(destination)
        module = spec.module.half().eval()
        started = time.perf_counter()
        with torch.no_grad(), ExitStack() as stack:
            stack.enter_context(use_grid_sample_metal_kernel(grid_kernel))
            stack.enter_context(use_deform_conv_metal_kernel(deform_kernel))
            exported = torch.export.export(module, spec.examples)
            exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels([grid_kernel, deform_kernel])
        converter.add_exported_program(
            exported,
            input_names=list(spec.input_names),
            output_names=[spec.output_name],
        )
        program = converter.to_coreai()
        save_program_asset(program, destination)
        print(
            f"exported {spec.name}: {time.perf_counter() - started:.2f}s -> {destination}",
            flush=True,
        )
        outputs[spec.name] = destination
    return outputs


class TorchBackend:
    def __init__(self, specs: list[AssetSpec], device: torch.device):
        self.modules = {
            spec.name: spec.module.to(device=device, dtype=torch.float32).eval()
            for spec in specs
        }
        self.device = device

    def call(self, name: str, **values: np.ndarray) -> np.ndarray:
        module = self.modules[name]
        inputs = [
            torch.from_numpy(value).to(self.device, dtype=torch.float32)
            for value in values.values()
        ]
        with torch.inference_mode():
            output = module(*inputs)
        return output.detach().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        return None


class CoreAIBackend:
    def __init__(self, assets: dict[str, Path], specs: list[AssetSpec]):
        from lada.coreai.source_runtime import load_source_model

        self.runner = asyncio.Runner()
        self.functions: dict[str, Any] = {}
        self.models: list[Any] = []
        for spec in specs:
            model = load_source_model(
                self.runner,
                assets[spec.name],
                purpose=f"variable BasicVSR++ {spec.name}",
            )
            self.models.append(model)
            self.functions[spec.name] = model.load_function("main")

    def call(self, name: str, **values: np.ndarray) -> np.ndarray:
        from coreai.runtime import NDArray

        function = self.functions[name]

        async def invoke() -> np.ndarray:
            result = await function(
                {key: NDArray(np.ascontiguousarray(value)) for key, value in values.items()}
            )
            return result[next(iter(result))].numpy().copy()

        return self.runner.run(invoke())

    def close(self) -> None:
        self.functions.clear()
        self.models.clear()
        self.runner.close()


class SwiftVariableRuntime:
    """Persistent whole-sequence runner with device-resident intermediates."""

    def __init__(
        self,
        runner_path: Path,
        compiled_dir: Path,
        maximum_frames: int,
    ):
        if not runner_path.is_file():
            raise FileNotFoundError(runner_path)
        if not compiled_dir.is_dir():
            raise FileNotFoundError(compiled_dir)
        if maximum_frames <= 0 or maximum_frames >= 65535:
            raise ValueError("maximum_frames must be 1..65534")
        self.maximum_frames = maximum_frames
        self.frame_elements = 3 * IMAGE_SIZE * IMAGE_SIZE
        self.sequence_bytes = (
            maximum_frames * self.frame_elements * np.dtype(np.float16).itemsize
        )
        self.descriptor_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="lada-variable-basicvsrpp-", suffix=".json", delete=False
        )
        self.shared_file = tempfile.NamedTemporaryFile(
            prefix="lada-variable-basicvsrpp-", suffix=".bin", delete=False
        )
        self.descriptor_path = Path(self.descriptor_file.name)
        self.shared_path = Path(self.shared_file.name)
        descriptor = {
            "maximumFrames": maximum_frames,
            "inputOffset": 0,
            "outputOffset": self.sequence_bytes,
            "byteCount": self.sequence_bytes * 2,
        }
        json.dump(descriptor, self.descriptor_file)
        self.descriptor_file.flush()
        self.descriptor_file.close()
        self.shared_file.truncate(self.sequence_bytes * 2)
        self.shared_file.flush()
        self.mapping = mmap.mmap(self.shared_file.fileno(), self.sequence_bytes * 2)
        self.process = subprocess.Popen(
            [
                str(runner_path),
                str(compiled_dir),
                str(self.descriptor_path),
                str(self.shared_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if frames.dtype != np.float16 or not frames.flags.c_contiguous:
            raise ValueError("Swift variable runner requires contiguous FP16 input")
        if frames.ndim != 5 or tuple(frames.shape[:1] + frames.shape[2:]) != (
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(f"unexpected frames shape: {frames.shape}")
        frame_count = int(frames.shape[1])
        if not 1 <= frame_count <= self.maximum_frames:
            raise ValueError(f"frame count must be 1..{self.maximum_frames}")
        assert self.process.stdin is not None and self.process.stdout is not None
        byte_count = frames.nbytes
        self.mapping.seek(0)
        self.mapping.write(frames.tobytes(order="C"))
        self.process.stdin.write(struct.pack("<H", frame_count))
        self.process.stdin.flush()
        response = self.process.stdout.read(1)
        if response != b"\x00":
            stderr = b""
            if self.process.poll() is not None and self.process.stderr is not None:
                stderr = self.process.stderr.read()
            raise RuntimeError(
                "Swift variable runner failed: "
                + stderr.decode("utf-8", errors="replace").strip()
            )
        self.mapping.seek(self.sequence_bytes)
        payload = self.mapping.read(byte_count)
        return np.frombuffer(payload, dtype=np.float16).reshape(frames.shape).copy()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            assert process.stdin is not None
            try:
                process.stdin.write(struct.pack("<H", 65535))
                process.stdin.flush()
                process.wait(timeout=5)
            except Exception:
                process.terminate()
                process.wait(timeout=5)
        mapping = getattr(self, "mapping", None)
        if mapping is not None:
            mapping.close()
        shared_file = getattr(self, "shared_file", None)
        if shared_file is not None:
            shared_file.close()
        for path in (
            getattr(self, "descriptor_path", None),
            getattr(self, "shared_path", None),
        ):
            if path is not None:
                path.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def run_variable(backend: Any, frames: np.ndarray) -> np.ndarray:
    frame_count = frames.shape[1]
    zero_flow = np.zeros((1, 2, FEATURE_SIZE, FEATURE_SIZE), dtype=np.float16)

    features: dict[str, list[np.ndarray]] = {
        "spatial": [
            backend.call("spatial", frame=frames[:, index])
            for index in range(frame_count)
        ]
    }
    flows_backward = [
        backend.call("flow", ref=frames[:, index], supp=frames[:, index + 1])
        for index in range(frame_count - 1)
    ]
    flows_forward = [
        backend.call("flow", ref=frames[:, index + 1], supp=frames[:, index])
        for index in range(frame_count - 1)
    ]

    for branch_index, branch in enumerate(BRANCHES):
        backward = branch.startswith("backward")
        indices = list(range(frame_count - 1, -1, -1)) if backward else list(range(frame_count))
        directional_flows = flows_backward if backward else flows_forward
        produced: list[tuple[int, np.ndarray]] = []
        for order, frame_index in enumerate(indices):
            context_parts = [features["spatial"][frame_index]]
            context_parts.extend(features[name][frame_index] for name in BRANCHES[:branch_index])
            if order == 0:
                flow_n1 = zero_flow
                flow_previous = zero_flow
            elif backward:
                flow_n1 = directional_flows[frame_index]
                flow_previous = (
                    directional_flows[indices[order - 1]] if order > 1 else zero_flow
                )
            else:
                flow_n1 = directional_flows[frame_index - 1]
                flow_previous = directional_flows[frame_index - 2] if order > 1 else zero_flow
            context = np.ascontiguousarray(np.concatenate(context_parts, axis=1))
            if order == 0:
                result = backend.call(f"{branch}_init", context=context)
            elif order == 1:
                result = backend.call(
                    f"{branch}_first",
                    context=context,
                    state_n1=produced[-1][1],
                    flow_n1=flow_n1,
                )
            else:
                result = backend.call(
                    f"{branch}_later",
                    context=context,
                    state_n1=produced[-1][1],
                    state_n2=produced[-2][1],
                    flow_n1=flow_n1,
                    flow_previous=flow_previous,
                )
            produced.append((frame_index, result))
        ordered: list[np.ndarray | None] = [None] * frame_count
        for frame_index, result in produced:
            ordered[frame_index] = result
        features[branch] = [item for item in ordered if item is not None]

    outputs = []
    for index in range(frame_count):
        fusion = np.ascontiguousarray(
            np.concatenate(
                [features["spatial"][index]]
                + [features[name][index] for name in BRANCHES],
                axis=1,
            )
        )
        outputs.append(
            backend.call("reconstruction", frame=frames[:, index], features=fusion)
        )
    return np.stack(outputs, axis=1)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.swift_runner is not None:
        return benchmark_swift(args)
    wrapper = load_generator(args.checkpoint)
    generator = wrapper.generator
    specs = build_specs(generator)
    assets = {
        spec.name: args.output_dir / f"basicvsrpp-variable-{spec.name}.aimodel"
        for spec in specs
    }
    missing = [path for path in assets.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing exported assets: {missing}")
    backend = CoreAIBackend(assets, specs)
    results: dict[str, Any] = {"backend": "coreai-source-host-scheduled", "lengths": {}}
    try:
        rng = np.random.default_rng(args.seed)
        maximum = max(args.lengths)
        all_frames = rng.random(
            (1, maximum, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        ).astype(np.float16)
        warm_frames = all_frames[:, : max(3, min(args.warmup_frames, maximum))]
        warm_started = time.perf_counter()
        run_variable(backend, warm_frames)
        results["first_warmup_seconds"] = time.perf_counter() - warm_started
        for length in args.lengths:
            samples = []
            output = None
            for _ in range(args.runs):
                started = time.perf_counter()
                output = run_variable(backend, all_frames[:, :length])
                samples.append(time.perf_counter() - started)
            median = float(np.median(samples))
            assert output is not None
            results["lengths"][str(length)] = {
                "median_seconds": median,
                "seconds_per_frame": median / length,
                "fps": length / median,
                "runs_seconds": samples,
                "output_mean": float(output.astype(np.float32).mean()),
                "finite": bool(np.isfinite(output).all()),
            }
            print(
                f"T{length}: {median:.3f}s, {median / length:.3f}s/frame, "
                f"{length / median:.2f}fps",
                flush=True,
            )
    finally:
        backend.close()
    return results


def benchmark_swift(args: argparse.Namespace) -> dict[str, Any]:
    if args.compiled_dir is None:
        raise ValueError("--compiled-dir is required with --swift-runner")
    maximum = max(args.lengths)
    runtime = SwiftVariableRuntime(args.swift_runner, args.compiled_dir, maximum)
    results: dict[str, Any] = {
        "backend": "coreai-swift-device-resident",
        "lengths": {},
    }
    try:
        rng = np.random.default_rng(args.seed)
        all_frames = np.ascontiguousarray(
            rng.random(
                (1, maximum, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
            ).astype(np.float16)
        )
        warm_frames = np.ascontiguousarray(
            all_frames[:, : max(3, min(args.warmup_frames, maximum))]
        )
        started = time.perf_counter()
        runtime(warm_frames)
        results["first_warmup_seconds"] = time.perf_counter() - started
        for length in args.lengths:
            frames = np.ascontiguousarray(all_frames[:, :length])
            samples = []
            output = None
            for _ in range(args.runs):
                started = time.perf_counter()
                output = runtime(frames)
                samples.append(time.perf_counter() - started)
            median = float(np.median(samples))
            assert output is not None
            results["lengths"][str(length)] = {
                "median_seconds": median,
                "seconds_per_frame": median / length,
                "fps": length / median,
                "runs_seconds": samples,
                "output_mean": float(output.astype(np.float32).mean()),
                "finite": bool(np.isfinite(output).all()),
            }
            print(
                f"Swift T{length}: {median:.3f}s, {median / length:.3f}s/frame, "
                f"{length / median:.2f}fps",
                flush=True,
            )
    finally:
        runtime.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/basicvsrpp-variable-coreai")
    )
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--swift-runner", type=Path)
    parser.add_argument("--compiled-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.export:
        export_assets(args.checkpoint, args.output_dir, overwrite=args.overwrite)
    if args.export_only:
        if not args.export:
            raise ValueError("--export-only requires --export")
        return 0
    result = benchmark(args)
    report = args.report or args.output_dir / "benchmark.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
