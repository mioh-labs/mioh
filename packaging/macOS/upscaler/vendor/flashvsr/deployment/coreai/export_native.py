"""Export the complete FlashVSR native-runtime component set to Core AI."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .full_model import (
    COMPACT_CHECKPOINT,
    CoreAIFirstChunkDiTBlock,
    CoreAIFirstTCDecoder,
    CoreAIHead,
    CoreAILQNext,
    CoreAILQWarmup,
    CoreAINextChunkDiTBlock,
    CoreAINextTCDecoder,
    CoreAIPatchEmbedding,
    load_compact_block_pair,
    load_lq_projection,
    load_patch_and_heads,
    build_tcdecoder_pair,
    build_tcdecoder_step,
)
from .model import build_rope_cos_sin


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export native FlashVSR Core AI components."
    )
    parser.add_argument(
        "--component",
        choices=("dit", "lq", "front", "decoder", "all"),
        default="all",
    )
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("build/coreai-native/grid16"))
    parser.add_argument("--checkpoint", type=Path, default=COMPACT_CHECKPOINT)
    parser.add_argument("--grid-height", type=int, default=16)
    parser.add_argument("--grid-width", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _prepare_output(path: Path, force: bool) -> None:
    if path.suffix != ".aimodel":
        raise ValueError(f"Core AI output must end in .aimodel: {path}")
    if path.exists():
        if not force:
            raise FileExistsError(f"Refusing to replace {path}; pass --force")
        if path.parent.name not in {"grid8", "grid16", "grid32", "coreai-native"}:
            raise ValueError(f"Refusing to replace output outside a known build dir: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _program(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
    import coreai_torch

    exported = torch.export.export(model, args=inputs)
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def _save_asset(
    output: Path,
    entries: Iterable[
        tuple[
            str,
            torch.nn.Module,
            tuple[torch.Tensor, ...],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ],
) -> tuple[Path, dict[str, tuple[torch.Tensor, ...]]]:
    import coreai_torch

    converter = coreai_torch.TorchConverter(
        mode=coreai_torch.TorchConverter.Mode.RELEASE
    )
    references: dict[str, tuple[torch.Tensor, ...]] = {}
    with torch.inference_mode():
        for name, model, inputs, input_names, output_names in entries:
            reference = model(*inputs)
            if isinstance(reference, torch.Tensor):
                reference = (reference,)
            references[name] = tuple(item.detach().float() for item in reference)
            converter.add_exported_program(
                _program(model, inputs),
                input_names=input_names,
                output_names=output_names,
                entrypoint_name=name,
            )
    asset = converter.to_coreai().save_asset(output)
    return Path(asset.path), references


async def _run_function(
    model: Any,
    name: str,
    inputs: tuple[torch.Tensor, ...] | dict[str, Any],
    input_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    from coreai.runtime import NDArray

    function = model.load_function(name)
    if isinstance(inputs, dict):
        values = inputs
    else:
        assert input_names is not None
        values = {
            key: NDArray(value.contiguous())
            for key, value in zip(input_names, inputs, strict=True)
        }
    return await function(values)


def _metrics(actual: Any, expected: torch.Tensor) -> dict[str, float]:
    delta = np.abs(actual.numpy().astype(np.float32) - expected.numpy())
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
    }


async def _validate_dit(
    output: Path,
    first_inputs: tuple[torch.Tensor, ...],
    next_x_lq_rope: tuple[torch.Tensor, ...],
    references: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, Any]:
    from coreai.runtime import AIModel, NDArray

    model = await AIModel.load(output)
    first = await _run_function(
        model,
        "first_chunk",
        first_inputs,
        ("x", "lq", "rope_cos", "rope_sin"),
    )
    next_values = {
        "x": NDArray(next_x_lq_rope[0].contiguous()),
        "lq": NDArray(next_x_lq_rope[1].contiguous()),
        "rope_cos": NDArray(next_x_lq_rope[2].contiguous()),
        "rope_sin": NDArray(next_x_lq_rope[3].contiguous()),
        "cache_k": first["cache_k"],
        "cache_v": first["cache_v"],
    }
    next_result = await _run_function(model, "next_chunk", next_values)
    result: dict[str, Any] = {}
    for prefix, values, expected in (
        ("first", first, references["first_chunk"]),
        ("next", next_result, references["next_chunk"]),
    ):
        for output_name, reference in zip(
            ("x", "cache_k", "cache_v"), expected, strict=True
        ):
            result[f"{prefix}_{output_name}"] = _metrics(
                values[output_name], reference
            )
    return result


def export_dit(args: argparse.Namespace, block: int) -> dict[str, Any]:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    height, width = args.grid_height, args.grid_width
    window_size = 2 * height * width // 128
    topk = max(int(window_size * window_size * 2.0) - 1, 0)
    common = dict(
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        height=height,
        width=width,
        topk=topk,
        kv_len=3,
        inject_lq=block == 0,
    )
    first = CoreAIFirstChunkDiTBlock(frames=6, **common).eval().to(dtype=dtype)
    next_chunk = CoreAINextChunkDiTBlock(frames=2, **common).eval().to(dtype=dtype)
    load_compact_block_pair(
        first, next_chunk, block=block, checkpoint=args.checkpoint
    )
    tokens_first = 6 * height * width
    tokens_next = 2 * height * width
    first_x = torch.randn(1, tokens_first, 1536, dtype=dtype)
    first_lq = torch.randn_like(first_x) if block == 0 else torch.zeros_like(first_x)
    first_cos, first_sin = build_rope_cos_sin(128, 6, height, width)
    first_cos, first_sin = first_cos.to(dtype), first_sin.to(dtype)
    with torch.inference_mode():
        first_reference = first(first_x, first_lq, first_cos, first_sin)
    next_x = torch.randn(1, tokens_next, 1536, dtype=dtype)
    next_lq = torch.randn_like(next_x) if block == 0 else torch.zeros_like(next_x)
    next_cos, next_sin = build_rope_cos_sin(
        128, 2, height, width, temporal_offset=6
    )
    next_cos, next_sin = next_cos.to(dtype), next_sin.to(dtype)
    next_inputs = (
        next_x,
        next_lq,
        next_cos,
        next_sin,
        first_reference[1],
        first_reference[2],
    )
    output = args.output_dir / f"dit_block_{block:02d}.aimodel"
    _prepare_output(output, args.force)
    output, references = _save_asset(
        output,
        (
            (
                "first_chunk",
                first,
                (first_x, first_lq, first_cos, first_sin),
                ("x", "lq", "rope_cos", "rope_sin"),
                ("x", "cache_k", "cache_v"),
            ),
            (
                "next_chunk",
                next_chunk,
                next_inputs,
                ("x", "lq", "rope_cos", "rope_sin", "cache_k", "cache_v"),
                ("x", "cache_k", "cache_v"),
            ),
        ),
    )
    result: dict[str, Any] = {
        "component": "dit",
        "block": block,
        "output": str(output),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }
    if args.validate:
        result["validation"] = asyncio.run(
            _validate_dit(
                output,
                (first_x, first_lq, first_cos, first_sin),
                (next_x, next_lq, next_cos, next_sin),
                references,
            )
        )
    return result


async def _validate_simple(
    output: Path,
    entries: list[tuple[str, tuple[torch.Tensor, ...], tuple[str, ...]]],
    references: dict[str, tuple[torch.Tensor, ...]],
    output_names: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    from coreai.runtime import AIModel

    model = await AIModel.load(output)
    metrics: dict[str, Any] = {}
    prior: dict[str, Any] = {}
    for name, inputs, names in entries:
        if name == "next" and prior:
            from coreai.runtime import NDArray

            values = {
                "frames": NDArray(inputs[0].contiguous()),
                "cache1": prior["cache1"],
                "cache2": prior["cache2"],
            }
            prediction = await _run_function(model, name, values)
        else:
            prediction = await _run_function(model, name, inputs, names)
        for output_name, expected in zip(
            output_names[name], references[name], strict=True
        ):
            metrics[f"{name}_{output_name}"] = _metrics(
                prediction[output_name], expected
            )
        prior = prediction
    return metrics


def export_lq(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output_height = args.grid_height * 16
    output_width = args.grid_width * 16
    warmup = CoreAILQWarmup(output_height, output_width).eval().to(dtype=dtype)
    next_chunk = CoreAILQNext(output_height, output_width).eval().to(dtype=dtype)
    load_lq_projection(warmup, next_chunk)
    first_frame = torch.randn(1, 3, 1, output_height, output_width, dtype=dtype)
    with torch.inference_mode():
        caches = warmup(first_frame)
    frames = torch.randn(1, 3, 4, output_height, output_width, dtype=dtype)
    next_inputs = (frames, caches[0], caches[1])
    output = args.output_dir / "lq_projection.aimodel"
    _prepare_output(output, args.force)
    output, references = _save_asset(
        output,
        (
            (
                "warmup",
                warmup,
                (first_frame,),
                ("first_frame",),
                ("cache1", "cache2"),
            ),
            (
                "next",
                next_chunk,
                next_inputs,
                ("frames", "cache1", "cache2"),
                ("lq", "cache1", "cache2"),
            ),
        ),
    )
    result: dict[str, Any] = {
        "component": "lq",
        "output": str(output),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }
    if args.validate:
        result["validation"] = asyncio.run(
            _validate_simple(
                output,
                [
                    ("warmup", (first_frame,), ("first_frame",)),
                    ("next", (frames,), ("frames",)),
                ],
                references,
                {
                    "warmup": ("cache1", "cache2"),
                    "next": ("lq", "cache1", "cache2"),
                },
            )
        )
    return result


def export_front(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    h, w = args.grid_height, args.grid_width
    patch_first = CoreAIPatchEmbedding().eval().to(dtype=dtype)
    patch_next = CoreAIPatchEmbedding().eval().to(dtype=dtype)
    head_first = CoreAIHead(6, h, w).eval().to(dtype=dtype)
    head_next = CoreAIHead(2, h, w).eval().to(dtype=dtype)
    load_patch_and_heads(patch_first, head_first, head_next, args.checkpoint)
    load_patch_and_heads(patch_next, head_first, head_next, args.checkpoint)
    latent_first = torch.randn(1, 16, 6, h * 2, w * 2, dtype=dtype)
    latent_next = torch.randn(1, 16, 2, h * 2, w * 2, dtype=dtype)
    tokens_first = torch.randn(1, 6 * h * w, 1536, dtype=dtype)
    tokens_next = torch.randn(1, 2 * h * w, 1536, dtype=dtype)
    output = args.output_dir / "patch_head.aimodel"
    _prepare_output(output, args.force)
    output, references = _save_asset(
        output,
        (
            ("patch_first", patch_first, (latent_first,), ("latent",), ("x",)),
            ("patch_next", patch_next, (latent_next,), ("latent",), ("x",)),
            ("head_first", head_first, (tokens_first,), ("x",), ("latent",)),
            ("head_next", head_next, (tokens_next,), ("x",), ("latent",)),
        ),
    )
    result: dict[str, Any] = {
        "component": "front",
        "output": str(output),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }
    if args.validate:
        result["validation"] = asyncio.run(
            _validate_simple(
                output,
                [
                    ("patch_first", (latent_first,), ("latent",)),
                    ("patch_next", (latent_next,), ("latent",)),
                    ("head_first", (tokens_first,), ("x",)),
                    ("head_next", (tokens_next,), ("x",)),
                ],
                references,
                {
                    "patch_first": ("x",),
                    "patch_next": ("x",),
                    "head_first": ("latent",),
                    "head_next": ("latent",),
                },
            )
        )
    return result


async def _validate_decoder(
    output: Path,
    first_inputs: tuple[torch.Tensor, ...],
    next_inputs: tuple[torch.Tensor, ...],
    references: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, Any]:
    from coreai.runtime import AIModel, NDArray

    model = await AIModel.load(output)
    first = await _run_function(
        model, "first", first_inputs, ("latent", "condition")
    )
    values: dict[str, Any] = {
        "latent": NDArray(next_inputs[0].contiguous()),
        "condition": NDArray(next_inputs[1].contiguous()),
    }
    for index in range(9):
        values[f"memory{index}"] = first[f"memory{index}"]
    next_result = await _run_function(model, "next", values)
    metrics: dict[str, Any] = {}
    names = ("video", *(f"memory{i}" for i in range(9)))
    for prefix, prediction, expected in (
        ("first", first, references["first"]),
        ("next", next_result, references["next"]),
    ):
        for name, reference in zip(names, expected, strict=True):
            metrics[f"{prefix}_{name}"] = _metrics(prediction[name], reference)
    return metrics


def export_decoder(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    step = build_tcdecoder_step().eval().to(dtype=dtype)
    latent_height = args.grid_height * 2
    latent_width = args.grid_width * 2
    output_height = args.grid_height * 16
    output_width = args.grid_width * 16
    latent = torch.randn(1, 16, latent_height, latent_width, dtype=dtype)
    condition = torch.randn(1, 3, 4, output_height, output_width, dtype=dtype)
    memories = (
        *(torch.randn(1, 512, latent_height, latent_width, dtype=dtype) for _ in range(3)),
        *(torch.randn(1, 256, latent_height * 2, latent_width * 2, dtype=dtype) for _ in range(3)),
        *(torch.randn(1, 128, latent_height * 4, latent_width * 4, dtype=dtype) for _ in range(3)),
    )
    inputs = (latent, condition, *memories)
    memory_names = tuple(f"memory{i}" for i in range(9))
    output_names = ("video", *memory_names)
    output = args.output_dir / "tcdecoder.aimodel"
    _prepare_output(output, args.force)
    output, references = _save_asset(
        output,
        (
            (
                "step",
                step,
                inputs,
                ("latent", "condition", *memory_names),
                output_names,
            ),
        ),
    )
    result: dict[str, Any] = {
        "component": "decoder",
        "output": str(output),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }
    if args.validate:
        result["validation"] = asyncio.run(
            _validate_simple(
                output,
                [("step", inputs, ("latent", "condition", *memory_names))],
                references,
                {"step": output_names},
            )
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    torch.manual_seed(231)
    results: list[dict[str, Any]] = []
    if args.component in {"front", "all"}:
        results.append(export_front(args))
    if args.component in {"lq", "all"}:
        results.append(export_lq(args))
    if args.component in {"decoder", "all"}:
        results.append(export_decoder(args))
    if args.component == "dit":
        if not 0 <= args.block < 30:
            raise ValueError("--block must be in 0...29")
        results.append(export_dit(args, args.block))
    elif args.component == "all":
        for block in range(30):
            results.append(export_dit(args, block))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
