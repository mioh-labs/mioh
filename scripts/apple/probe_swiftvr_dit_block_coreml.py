#!/usr/bin/env python3
"""Probe Core ML operator coverage for one SwiftVR DiT block.

Without --checkpoint this uses random reduced-width weights. With a checkpoint
it loads one trained, full-width layer. Either way, one block alone cannot
restore video; the remaining layers and stream plumbing require validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors import safe_open


class BlockWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, context, modulation, cosine, sine):
        block = self.block
        mods = (block.scale_shift_table + modulation.float()).to(hidden.dtype)
        shift, scale, gate, cross_shift, cross_scale, cross_gate = mods.chunk(6, dim=1)
        self_attention = block.attn1(
            block.norm1(hidden) * (1 + scale) + shift,
            None,
            None,
            (cosine, sine),
        )
        hidden = hidden + self_attention * gate
        hidden = hidden + block.attn2(block.norm2(hidden), context, None, None)
        feed_forward = block.ffn(
            block.norm3(hidden) * (1 + cross_scale) + cross_shift
        )
        return hidden + feed_forward * cross_gate


def portable_rope(x, freqs_cos, freqs_sin):
    """Out-of-place equivalent of upstream's in-place RoPE."""
    first, second = x.unflatten(-1, (-1, 2)).unbind(-1)
    cosine = freqs_cos[..., 0::2].to(x.dtype)
    sine = freqs_sin[..., 1::2].to(x.dtype)
    return torch.stack(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    ).flatten(-2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--grid-time", type=int, default=1)
    parser.add_argument("--grid-height", type=int, default=4)
    parser.add_argument("--grid-width", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--fixture-directory", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source))
    import swiftvr.models.transformer as transformer_module  # noqa: E402
    from swiftvr.models.transformer import (  # noqa: E402
        WanShiftWindow2DInferProcessor,
        WanTransformerBlock,
        set_attention_backend,
    )

    torch.manual_seed(1)
    set_attention_backend("sdpa")
    dim, ffn_dim, heads = (3072, 14336, 24) if args.checkpoint else (128, 256, 4)
    if args.checkpoint:
        with torch.device("meta"):
            block = WanTransformerBlock(
                dim=dim, ffn_dim=ffn_dim, num_heads=heads,
                cross_attn_norm=True,
            )
        with safe_open(str(args.checkpoint), framework="pt", device="cpu") as source:
            prefix = f"blocks.{args.layer_index}."
            weights = {
                key.removeprefix(prefix): source.get_tensor(key)
                for key in source.keys() if key.startswith(prefix)
            }
        block.load_state_dict(weights, strict=True, assign=True)
        del weights
    else:
        block = WanTransformerBlock(
            dim=dim, ffn_dim=ffn_dim, num_heads=heads,
            cross_attn_norm=True,
        )
    block.eval()
    block.attn1.set_processor(
        WanShiftWindow2DInferProcessor((args.window_size, args.window_size))
    )
    grid = (args.grid_time, args.grid_height, args.grid_width)
    tokens = args.grid_time * args.grid_height * args.grid_width
    block.attn1._thw = grid
    block.attn1._do_shift = args.layer_index % 2 == 1
    wrapper = BlockWrapper(block).eval()
    hidden = torch.randn(1, tokens, dim)
    context = torch.randn(1, args.context_tokens, dim)
    modulation = torch.randn(1, 6, dim)
    angles = torch.linspace(0.0, 1.5, tokens * (dim // heads // 2)).reshape(
        1, tokens, 1, dim // heads // 2
    )
    cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1)
    sine = torch.repeat_interleave(angles.sin(), 2, dim=-1)
    inputs = (hidden, context, modulation, cosine, sine)
    with torch.no_grad():
        official = block(hidden.clone(), context, modulation, (cosine, sine)).numpy()
        before_rewrite = wrapper(*inputs).numpy()
        residual_difference = np.abs(before_rewrite - official)
        if residual_difference.max() > 1e-5:
            raise RuntimeError(
                "Portable residual block differs before RoPE rewrite: "
                f"max abs error={residual_difference.max():.6g}"
            )
        transformer_module._apply_rotary_emb_inplace = portable_rope
        expected = wrapper(*inputs).numpy()
        portable_difference = np.abs(expected - official)
        if portable_difference.max() > 1e-3 or portable_difference.mean() > 1e-5:
            raise RuntimeError(
                "Portable block differs from upstream: "
                f"max abs error={portable_difference.max():.6g}"
            )
        traced = torch.jit.trace(wrapper, inputs)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=name, shape=value.shape)
            for name, value in zip(
                ("hidden", "context", "modulation", "cosine", "sine"), inputs
            )
        ],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=(
            ct.precision.FLOAT16
            if args.precision == "float16"
            else ct.precision.FLOAT32
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    prediction_inputs = {
        name: value.numpy()
        for name, value in zip(
            ("hidden", "context", "modulation", "cosine", "sine"), inputs
        )
    }
    started = time.perf_counter()
    native = converted.predict(prediction_inputs)["output"]
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    converted.predict(prediction_inputs)
    warm_seconds = time.perf_counter() - started
    difference = np.abs(native - expected)
    if args.fixture_directory:
        fixture = args.fixture_directory.resolve()
        fixture.mkdir(parents=True, exist_ok=True)
        shapes = {}
        for name, value in zip(
            ("hidden", "context", "modulation", "cosine", "sine"), inputs
        ):
            data = value.detach().numpy().astype("<f4")
            data.tofile(fixture / f"{name}.f32")
            shapes[name] = list(data.shape)
        expected.astype("<f4").tofile(fixture / "expected.f32")
        shapes["expected"] = list(expected.shape)
        (fixture / "shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
    kind = "Trained full-width" if args.checkpoint else "Reduced random-weight"
    print(
        f"{kind} SwiftVR DiT block: max abs error={difference.max():.6g}, "
        f"mean abs error={difference.mean():.6g}, "
        f"cold={cold_seconds:.3f}s, warm={warm_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
