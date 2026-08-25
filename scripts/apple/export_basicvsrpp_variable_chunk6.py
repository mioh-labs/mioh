#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export the validated six-frame variable BasicVSR++ Core AI assets.

The recurrent propagation is unrolled over one contiguous six-frame tensor.
Keeping the temporal axis in a single input is intentional: equivalent models
with six separately named context inputs trigger a Core AI numeric regression.
"""

from __future__ import annotations

import argparse
import shutil
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .basicvsrpp_coreai_kernels import (
        build_deform_conv_kernel,
        build_flow_warp_kernel,
        build_grid_sample_kernel,
    )
    from .benchmark_basicvsrpp_variable_coreai import (
        BRANCHES,
        FEATURE_SIZE,
        IMAGE_SIZE,
        MID_CHANNELS,
        PairFlow,
        PropagationFirst,
        PropagationInit,
        PropagationLater,
        Reconstruction,
        SpatialEncoder,
    )
    from .export_basicvsrpp_coreai import (
        import_coreai,
        load_generator,
        save_program_asset,
        use_deform_conv_metal_kernel,
        use_flow_warp_metal_kernel,
        use_grid_sample_metal_kernel,
    )
except ImportError:
    from basicvsrpp_coreai_kernels import (  # type: ignore[import-not-found]
        build_deform_conv_kernel,
        build_flow_warp_kernel,
        build_grid_sample_kernel,
    )
    from benchmark_basicvsrpp_variable_coreai import (  # type: ignore[import-not-found]
        BRANCHES,
        FEATURE_SIZE,
        IMAGE_SIZE,
        MID_CHANNELS,
        PairFlow,
        PropagationFirst,
        PropagationInit,
        PropagationLater,
        Reconstruction,
        SpatialEncoder,
    )
    from export_basicvsrpp_coreai import (  # type: ignore[import-not-found]
        import_coreai,
        load_generator,
        save_program_asset,
        use_deform_conv_metal_kernel,
        use_flow_warp_metal_kernel,
        use_grid_sample_metal_kernel,
    )


CHUNK_SIZE = 6


class BidirectionalFlow6(PairFlow):
    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        references = frames[:CHUNK_SIZE]
        supports = frames[1 : CHUNK_SIZE + 1]
        return super().forward(references, supports), super().forward(
            supports, references
        )


class PropagationStart6(torch.nn.Module):
    """First chunk: internal zero state followed by five recurrent steps."""

    def __init__(self, generator: torch.nn.Module, branch: str):
        super().__init__()
        self.initial = PropagationInit(generator, branch)
        self.first = PropagationFirst(generator, branch)
        self.later = PropagationLater(generator, branch)

    def forward(self, contexts: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        outputs = [self.initial(contexts[0:1])]
        outputs.append(self.first(contexts[1:2], outputs[-1], flows[0:1]))
        for index in range(2, CHUNK_SIZE):
            outputs.append(
                self.later(
                    contexts[index : index + 1],
                    outputs[-1],
                    outputs[-2],
                    flows[index - 1 : index],
                    flows[index - 2 : index - 1],
                )
            )
        return torch.cat(outputs, dim=0)


class PropagationContinue6(torch.nn.Module):
    """Continuation chunk with the two recurrent boundary features."""

    def __init__(self, generator: torch.nn.Module, branch: str):
        super().__init__()
        self.later = PropagationLater(generator, branch)

    def forward(
        self,
        contexts: torch.Tensor,
        state_n1: torch.Tensor,
        state_n2: torch.Tensor,
        flows: torch.Tensor,
        flow_previous: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        previous = state_n1
        older = state_n2
        previous_flow = flow_previous
        for index in range(CHUNK_SIZE):
            flow = flows[index : index + 1]
            result = self.later(
                contexts[index : index + 1],
                previous,
                older,
                flow,
                previous_flow,
            )
            outputs.append(result)
            older, previous, previous_flow = previous, result, flow
        return torch.cat(outputs, dim=0)


class StatefulPropagationContinue6(torch.nn.Module):
    """Continuation chunk whose recurrent boundary is Core AI native state."""

    def __init__(self, generator: torch.nn.Module, branch: str):
        super().__init__()
        self.later = PropagationLater(generator, branch)
        self.register_buffer(
            "state_n1",
            torch.zeros(1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        )
        self.register_buffer(
            "state_n2",
            torch.zeros(1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        )
        self.register_buffer(
            "flow_previous",
            torch.zeros(1, 2, FEATURE_SIZE, FEATURE_SIZE),
        )

    def forward(
        self, contexts: torch.Tensor, flows: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        previous = self.state_n1
        older = self.state_n2
        previous_flow = self.flow_previous
        for index in range(CHUNK_SIZE):
            flow = flows[index : index + 1]
            result = self.later(
                contexts[index : index + 1],
                previous,
                older,
                flow,
                previous_flow,
            )
            outputs.append(result)
            older, previous, previous_flow = previous, result, flow
        self.state_n1.copy_(previous)
        self.state_n2.copy_(older)
        self.flow_previous.copy_(previous_flow)
        return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class ChunkAssetSpec:
    name: str
    module: torch.nn.Module
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    examples: tuple[torch.Tensor, ...]


def _random(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    value = rng.random(shape, dtype=np.float32).astype(np.float16)
    return torch.from_numpy(value)


def build_chunk_specs(generator: torch.nn.Module) -> list[ChunkAssetSpec]:
    frames6 = _random((CHUNK_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), 1)
    frames7 = _random((CHUNK_SIZE + 1, 3, IMAGE_SIZE, IMAGE_SIZE), 2)
    feature = _random((1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), 3)
    flow = _random((1, 2, FEATURE_SIZE, FEATURE_SIZE), 4) - 0.5
    specs = [
        ChunkAssetSpec(
            "spatial6",
            SpatialEncoder(generator),
            ("frames",),
            ("features",),
            (frames6,),
        ),
        ChunkAssetSpec(
            "flow6",
            BidirectionalFlow6(generator),
            ("frames",),
            ("backward", "forward"),
            (frames7,),
        ),
    ]
    for index, branch in enumerate(BRANCHES):
        context = _random(
            (
                CHUNK_SIZE,
                MID_CHANNELS * (index + 1),
                FEATURE_SIZE,
                FEATURE_SIZE,
            ),
            10 + index,
        )
        flows = flow.repeat(CHUNK_SIZE, 1, 1, 1)
        specs.extend(
            (
                ChunkAssetSpec(
                    f"{branch}_start6",
                    PropagationStart6(generator, branch),
                    ("contexts", "flows"),
                    ("features",),
                    (context, flows[: CHUNK_SIZE - 1]),
                ),
                ChunkAssetSpec(
                    f"{branch}_continue6",
                    PropagationContinue6(generator, branch),
                    (
                        "contexts",
                        "state_n1",
                        "state_n2",
                        "flows",
                        "flow_previous",
                    ),
                    ("features",),
                    (context, feature, feature.clone(), flows, flow),
                ),
            )
        )
    specs.append(
        ChunkAssetSpec(
            "reconstruction6",
            Reconstruction(generator),
            ("frames", "features"),
            ("restored",),
            (
                frames6,
                _random(
                    (
                        CHUNK_SIZE,
                        MID_CHANNELS * 5,
                        FEATURE_SIZE,
                        FEATURE_SIZE,
                    ),
                    30,
                ),
            ),
        )
    )
    return specs


def export_assets(
    checkpoint: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    fuse_flow_warp: bool = False,
    optimize: bool = False,
    native_state_continuations: bool = False,
) -> None:
    _coreai, coreai_torch = import_coreai()
    generator = load_generator(checkpoint).generator
    sampling_kernel = (
        build_flow_warp_kernel(coreai_torch)
        if fuse_flow_warp
        else build_grid_sample_kernel(coreai_torch)
    )
    deform_kernel = build_deform_conv_kernel(coreai_torch)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in build_chunk_specs(generator):
        destination = output_dir / f"basicvsrpp-variable-{spec.name}.aimodel"
        if destination.exists():
            if not overwrite:
                continue
            shutil.rmtree(destination)
        native_state = native_state_continuations and spec.name.endswith(
            "_continue6"
        )
        if native_state:
            branch = spec.name.removesuffix("_continue6")
            module = StatefulPropagationContinue6(generator, branch).half().eval()
            examples = (spec.examples[0], spec.examples[3])
            input_names = ["contexts", "flows"]
        else:
            module = spec.module.half().eval()
            examples = spec.examples
            input_names = list(spec.input_names)
        started = time.perf_counter()
        with torch.no_grad(), ExitStack() as stack:
            stack.enter_context(
                use_flow_warp_metal_kernel(sampling_kernel)
                if fuse_flow_warp
                else use_grid_sample_metal_kernel(sampling_kernel)
            )
            stack.enter_context(use_deform_conv_metal_kernel(deform_kernel))
            exported = torch.export.export(module, examples)
            exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels([sampling_kernel, deform_kernel])
        if native_state:
            converter.add_exported_program(
                exported,
                state_names=["state_n1", "state_n2", "flow_previous"],
                input_names=input_names,
                output_names=list(spec.output_names),
            )
        else:
            converter.add_exported_program(
                exported,
                input_names=input_names,
                output_names=list(spec.output_names),
            )
        program = converter.to_coreai()
        if optimize or native_state:
            program.optimize()
        save_program_asset(program, destination)
        print(
            f"exported {spec.name}: {time.perf_counter() - started:.2f}s -> "
            f"{destination}",
            flush=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fuse-flow-warp",
        action="store_true",
        help=(
            "EXPERIMENTAL: fuse BasicVSR++ flow-grid construction and bilinear "
            "sampling into one Metal kernel. Recurrent real-video validation "
            "currently rejects this path; do not use for production exports."
        ),
    )
    parser.add_argument(
        "--native-state-continuations",
        action="store_true",
        help=(
            "Export the four continuation chunks with Core AI mutable state. "
            "The Swift runner remains compatible with explicit-I/O assets."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    export_assets(
        args.checkpoint,
        args.output_dir,
        overwrite=args.overwrite,
        fuse_flow_warp=args.fuse_flow_warp,
        native_state_continuations=args.native_state_continuations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
