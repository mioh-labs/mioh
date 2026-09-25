#!/usr/bin/env python3
"""Export consecutive SwiftVR DiT layers as one Core ML or Core AI program.

Separate per-layer packages must be loaded one by one for every chunk, and
each loaded model keeps its own GPU working buffers, so a resident 30-layer
stack does not fit. A group shares one program and can stay loaded.

The group keeps the per-layer package contract: inputs hidden, context,
modulation, cosine, sine; output ``output`` (the hidden state after the last
layer of the group). Verification compares the fused graph with the upstream
blocks run one after another, then the Core ML result with PyTorch.

Several ``--latent-frames`` values produce one multifunction package with a
function per shape (``t7``, ``t6``). The shapes share every weight, which
Core ML stores once, so a 6- and 7-latent stack cost the disk of one.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_swiftvr_dit_block_coreml import BlockWrapper, portable_rope  # noqa: E402

INPUT_NAMES = ("hidden", "context", "modulation", "cosine", "sine")


class GroupWrapper(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = torch.nn.ModuleList(BlockWrapper(block) for block in blocks)

    def forward(self, hidden, context, modulation, cosine, sine):
        for layer in self.layers:
            hidden = layer(hidden, context, modulation, cosine, sine)
        return hidden


def export_coreai(wrapper, inputs, destination: Path) -> None:
    """FP16 Core AI asset with the same input and output names."""
    import coreai_torch

    half = wrapper.half().eval()
    half_inputs = tuple(value.half() for value in inputs)
    started = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(half, half_inputs)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported, input_names=list(INPUT_NAMES), output_names=["output"],
        entrypoint_name="main",
    )
    program = converter.to_coreai()
    program.optimize()
    destination.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(destination)
    print(f"Core AI asset written in {time.perf_counter() - started:.1f}s: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument(
        "--latent-frames", type=int, choices=(6, 7), nargs="+", required=True
    )
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    parser.add_argument("--backend", choices=("coreml", "coreai"), default="coreml")
    parser.add_argument(
        "--fixture-directory", type=Path,
        help="Write the inputs and the FP32 PyTorch output as little-endian f32 files.",
    )
    args = parser.parse_args()
    if not 0 <= args.first_layer < args.end_layer <= 30:
        parser.error("layer range must be within 0..<30 and nonempty")
    shapes = list(dict.fromkeys(args.latent_frames))
    if len(shapes) == 1:
        export_group(args, shapes[0], args.output)
        return
    if args.backend != "coreml" or args.fixture_directory:
        parser.error("several latent shapes need the coreml backend and no fixture")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        descriptor = ct.utils.MultiFunctionDescriptor()
        for latent_frames in shapes:
            single = Path(directory) / f"t{latent_frames}.mlpackage"
            export_group(args, latent_frames, single)
            descriptor.add_function(
                str(single), src_function_name="main",
                target_function_name=f"t{latent_frames}",
            )
        descriptor.default_function_name = f"t{shapes[0]}"
        if args.output.exists():
            shutil.rmtree(args.output)
        ct.utils.save_multifunction(descriptor, str(args.output))
    print(f"multifunction package written: {args.output}")


def export_group(args, latent_frames: int, output: Path) -> None:
    sys.path.insert(0, str(args.source))
    import swiftvr.models.transformer as transformer_module  # noqa: E402
    from swiftvr.models.transformer import (  # noqa: E402
        WanShiftWindow2DInferProcessor,
        WanTransformerBlock,
        set_attention_backend,
    )

    torch.manual_seed(1)
    set_attention_backend("sdpa")
    dim, ffn_dim, heads, window = 3072, 14336, 24, 16
    side = 16 if args.scale == 2 else 32
    grid = (latent_frames, side, side)
    tokens = latent_frames * side * side

    blocks = []
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as source:
        for index in range(args.first_layer, args.end_layer):
            with torch.device("meta"):
                block = WanTransformerBlock(
                    dim=dim, ffn_dim=ffn_dim, num_heads=heads, cross_attn_norm=True
                )
            prefix = f"blocks.{index}."
            weights = {
                key.removeprefix(prefix): source.get_tensor(key)
                for key in source.keys() if key.startswith(prefix)
            }
            block.load_state_dict(weights, strict=True, assign=True)
            block.eval()
            block.attn1.set_processor(WanShiftWindow2DInferProcessor((window, window)))
            block.attn1._thw = grid
            block.attn1._do_shift = index % 2 == 1
            blocks.append(block)

    hidden = torch.randn(1, tokens, dim)
    context = torch.randn(1, 512, dim)
    modulation = torch.randn(1, 6, dim)
    angles = torch.linspace(0.0, 1.5, tokens * (dim // heads // 2)).reshape(
        1, tokens, 1, dim // heads // 2
    )
    cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1)
    sine = torch.repeat_interleave(angles.sin(), 2, dim=-1)
    inputs = (hidden, context, modulation, cosine, sine)

    wrapper = GroupWrapper(blocks).eval()
    with torch.no_grad():
        official = hidden.clone()
        for block in blocks:
            official = block(official, context, modulation, (cosine, sine))
        official = official.numpy()
        transformer_module._apply_rotary_emb_inplace = portable_rope
        expected = wrapper(*inputs).numpy()
        difference = np.abs(expected - official)
        print(
            f"fused vs sequential upstream: max {difference.max():.6g}, "
            f"mean {difference.mean():.6g}",
            flush=True,
        )
        if difference.mean() > 1e-4:
            raise RuntimeError("Fused group differs from the upstream layers")
        if args.backend == "coreml":
            traced = torch.jit.trace(wrapper, inputs)

    if args.fixture_directory:
        fixture = args.fixture_directory
        fixture.mkdir(parents=True, exist_ok=True)
        for name, value in zip(INPUT_NAMES, inputs):
            value.numpy().astype("<f4").tofile(fixture / f"{name}.f32")
        expected.astype("<f4").tofile(fixture / "expected.f32")

    if args.backend == "coreai":
        export_coreai(wrapper, inputs, output)
        return

    started = time.perf_counter()
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=name, shape=value.shape)
            for name, value in zip(INPUT_NAMES, inputs)
        ],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=(
            ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
        ),
    )
    print(f"converted in {time.perf_counter() - started:.1f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))
    native = converted.predict(
        {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs)}
    )["output"]
    difference = np.abs(native - expected)
    reference = np.abs(expected).mean()
    print(
        f"Core ML vs PyTorch: max {difference.max():.6g}, mean {difference.mean():.6g} "
        f"({difference.mean() / reference * 100:.3f}% of mean |signal|)"
    )


if __name__ == "__main__":
    main()
