#!/usr/bin/env python3
"""Export and compile the complete 10Eros-Max H3 DiT for Mioh."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--compiled-directory", type=Path, required=True)
    parser.add_argument(
        "--asset-prefix",
        default="10eros-max-h3",
        help="Unique filename prefix for this checkpoint's compiled assets.",
    )
    parser.add_argument(
        "--configuration-name",
        help="Configuration directory name (defaults to <asset-prefix>-dit-configuration).",
    )
    parser.add_argument(
        "--model-name",
        default="10Eros-Max H3 TURBO Hybrid Beta3 skip-edges INT8 ConvRot",
    )
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument("--preferred-compute", default="gpu")
    parser.add_argument("--dynamic-max-tokens", type=int, default=131_072)
    parser.add_argument(
        "--dynamic-sample-tokens",
        type=int,
        default=256,
        help=(
            "Representative token count used while exporting a dynamic DiT "
            "graph. A realistic non-trivial sample avoids poor Core AI "
            "specialization from the old 8-token trace."
        ),
    )
    parser.add_argument(
        "--fixed-block-tokens",
        type=int,
        help=(
            "Export exact-shape BF16 block assets and keep them as .aimodel "
            "sources. This avoids macOS 27 beta 6 runtime reshaping and the "
            "coreai-build fixed-BF16 type bug."
        ),
    )
    parser.add_argument("--block-group-size", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--block-scalar-type", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--skip-components",
        action="store_true",
        help="Reuse already-compiled non-block DiT assets and only export block groups.",
    )
    parser.add_argument(
        "--fragment-output",
        type=Path,
        help="Write the denoiser fragment here instead of next to the compiled directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_asset(
    source: Path,
    compiled_directory: Path,
    architecture: str,
    preferred_compute: str,
    overwrite: bool,
) -> Path:
    expected = compiled_directory / f"{source.stem}.{architecture}.aimodelc"
    if expected.is_dir() and not overwrite:
        return expected
    run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(source),
            "--output",
            str(compiled_directory),
            "--platform",
            "macOS",
            "--min-deployment-version",
            "27.0",
            "--preferred-compute",
            preferred_compute,
            "--architecture",
            architecture,
        ]
    )
    generated = compiled_directory / f"{source.stem}.aimodelc"
    if not expected.is_dir() and generated.is_dir():
        generated.rename(expected)
    if not expected.is_dir():
        raise FileNotFoundError(expected)
    return expected


def stage_manifest(
    asset: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    input_constraints: dict[str, tuple[str, list[int]]],
    output_constraints: dict[str, tuple[str, list[int]]],
    *,
    function_name: str = "main",
    compute_units: str | None = None,
    logical_layer_count: int | None = None,
) -> dict:
    def constraints(values):
        return {
            name: {"scalarType": scalar, "shape": shape}
            for name, (scalar, shape) in values.items()
        }

    result = {
        "backend": "coreai",
        "asset": asset,
        "function": function_name,
        "inputs": inputs,
        "outputs": outputs,
        "inputConstraints": constraints(input_constraints),
        "outputConstraints": constraints(output_constraints),
    }
    if compute_units is not None:
        result["computeUnits"] = compute_units
    if logical_layer_count is not None:
        result["logicalLayerCount"] = logical_layer_count
    return result


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.asset_prefix):
        raise ValueError("--asset-prefix must contain lowercase letters, numbers, and hyphens")
    configuration_name = args.configuration_name or (
        f"{args.asset_prefix}-dit-configuration"
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", configuration_name):
        raise ValueError("--configuration-name must be a safe directory name")
    args.source_directory.mkdir(parents=True, exist_ok=True)
    args.compiled_directory.mkdir(parents=True, exist_ok=True)
    if args.fixed_block_tokens is not None:
        if args.fixed_block_tokens <= 0:
            raise ValueError("--fixed-block-tokens must be positive")
        if args.block_scalar_type != "bfloat16":
            raise ValueError("fixed block assets currently require bfloat16")
        if args.source_directory.resolve() != args.compiled_directory.resolve():
            raise ValueError(
                "fixed block assets are final .aimodel files; use the same "
                "--source-directory and --compiled-directory"
            )
    elif not 1 <= args.dynamic_sample_tokens <= args.dynamic_max_tokens:
        raise ValueError(
            "--dynamic-sample-tokens must be within the dynamic token bounds"
        )
    script_directory = Path(__file__).resolve().parent
    overwrite = ["--overwrite"] if args.overwrite else []
    checkpoint_digest = file_sha256(args.checkpoint)

    components = [
        ("text-refiner", f"{args.asset_prefix}-text-refiner-dynamic-bf16", 8192),
        ("video-projection", f"{args.asset_prefix}-video-projection-dynamic", args.dynamic_max_tokens),
        ("audio-projection", f"{args.asset_prefix}-audio-projection-dynamic", args.dynamic_max_tokens),
        ("final-video", f"{args.asset_prefix}-final-video-dynamic", args.dynamic_max_tokens),
        ("final-audio", f"{args.asset_prefix}-final-audio-dynamic", args.dynamic_max_tokens),
    ]
    groups = [
        (first, min(args.block_group_size, 50 - first))
        for first in range(0, 50, args.block_group_size)
    ]
    selected_components = [] if args.skip_components else components
    total = len(selected_components) + len(groups)
    completed = 0
    for stage, stem, maximum in selected_components:
        completed += 1
        source = args.source_directory / f"{stem}.aimodel"
        print(f"[{completed}/{total}] {stage}", flush=True)
        if args.overwrite or not source.is_dir():
            run(
                [
                    sys.executable,
                    str(script_directory / "export_10eros_max_h3_dit_components.py"),
                    "--stage",
                    stage,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--tokens",
                    "8",
                    "--dynamic-max-tokens",
                    str(maximum),
                    "--output",
                    str(source),
                    "--skip-reference",
                    *overwrite,
                ]
            )
        compile_asset(
            source,
            args.compiled_directory,
            args.architecture,
            args.preferred_compute,
            args.overwrite,
        )

    scalar_suffix = "bf16" if args.block_scalar_type == "bfloat16" else "fp16"
    for layer, layer_count in groups:
        completed += 1
        last = layer + layer_count - 1
        shape_suffix = (
            f"fixed{args.fixed_block_tokens}"
            if args.fixed_block_tokens is not None
            else "dynamic"
        )
        stem = (
            f"{args.asset_prefix}-dit-blocks{layer:02d}-{last:02d}"
            f"-{shape_suffix}-{scalar_suffix}"
        )
        source = args.source_directory / f"{stem}.aimodel"
        graph_identity = (
            f"w{checkpoint_digest[:16]}_b{layer:02d}_{last:02d}"
        )
        print(f"[{completed}/{total}] DiT blocks {layer:02d}-{last:02d}", flush=True)
        if args.overwrite or not source.is_dir():
            run(
                [
                    sys.executable,
                    str(script_directory / "export_10eros_max_h3_dit_block.py"),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--layer",
                    str(layer),
                    "--layer-count",
                    str(layer_count),
                    "--scalar-type",
                    args.block_scalar_type,
                    "--tokens",
                    str(
                        args.fixed_block_tokens
                        or args.dynamic_sample_tokens
                    ),
                    "--dynamic-max-tokens",
                    str(args.fixed_block_tokens or args.dynamic_max_tokens),
                    *(["--fixed-shape"] if args.fixed_block_tokens is not None else []),
                    "--output",
                    str(source),
                    "--graph-identity",
                    graph_identity,
                    "--skip-reference",
                    *overwrite,
                ]
            )
        if args.fixed_block_tokens is None:
            compile_asset(
                source,
                args.compiled_directory,
                args.architecture,
                args.preferred_compute,
                args.overwrite,
            )

    configuration = args.compiled_directory.parent / configuration_name
    configuration.mkdir(parents=True, exist_ok=True)
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as handle:
        table = handle.get_tensor("adaln_t_table").float().numpy()
        inverse_frequency = handle.get_tensor("rope.inv_freq").float().numpy()
    np.asarray(table, dtype=np.float32).tofile(configuration / "adaln_t_table.f32")
    np.asarray(inverse_frequency, dtype=np.float32).tofile(
        configuration / "rope_inv_freq.f32"
    )
    metadata = {
        "schemaVersion": 1,
        "model": args.model_name,
        "checkpoint": args.checkpoint.name,
        "checkpointSHA256": checkpoint_digest,
        "adalnTableShape": list(table.shape),
        "ropeInverseFrequencyShape": list(inverse_frequency.shape),
        "dynamicMaximumTokens": args.dynamic_max_tokens,
        "dynamicSampleTokens": (
            None
            if args.fixed_block_tokens is not None
            else args.dynamic_sample_tokens
        ),
        "fixedBlockTokens": args.fixed_block_tokens,
        "blocks": 50,
        "blockAssets": len(groups),
        "blockGroupSize": args.block_group_size,
        "scalarType": args.block_scalar_type,
        "architecture": args.architecture,
    }
    (configuration / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    relative = lambda stem: f"coreai/{stem}.{args.architecture}.aimodelc"
    block_relative = lambda stem: (
        f"coreai/{stem}.aimodel"
        if args.fixed_block_tokens is not None
        else relative(stem)
    )
    text_refiner = stage_manifest(
        relative(f"{args.asset_prefix}-text-refiner-dynamic-bf16"),
        {"context": "context"},
        {"textHidden": "text_hidden"},
        {"context": ("bfloat16", [-1, 5120])},
        {"textHidden": ("bfloat16", [-1, 5376])},
        compute_units=args.preferred_compute,
    )
    video_projection = stage_manifest(
        relative(f"{args.asset_prefix}-video-projection-dynamic"),
        {"videoRows": "video_rows"},
        {"videoHidden": "video_hidden"},
        {"videoRows": ("float32", [-1, 96])},
        {"videoHidden": ("bfloat16", [-1, 5376])},
        compute_units=args.preferred_compute,
    )
    audio_projection = stage_manifest(
        relative(f"{args.asset_prefix}-audio-projection-dynamic"),
        {"audioRows": "audio_rows"},
        {"audioHidden": "audio_hidden"},
        {"audioRows": ("float32", [-1, 32])},
        {"audioHidden": ("bfloat16", [-1, 5376])},
        compute_units=args.preferred_compute,
    )
    blocks = []
    for index, layer_count in groups:
        last = index + layer_count - 1
        graph_identity = (
            f"w{checkpoint_digest[:16]}_b{index:02d}_{last:02d}"
        )
        entrypoint_name = f"main_{graph_identity}"
        graph_salt_name = f"graph_identity_salt_{graph_identity}"
        graph_salt_width = index + 1
        shape_suffix = (
            f"fixed{args.fixed_block_tokens}"
            if args.fixed_block_tokens is not None
            else "dynamic"
        )
        token_dimension = args.fixed_block_tokens or -1
        blocks.append(stage_manifest(
            block_relative(
                f"{args.asset_prefix}-dit-blocks{index:02d}-{last:02d}"
                f"-{shape_suffix}-{scalar_suffix}"
            ),
            {
                "hiddenStates": "hidden_states",
                "timestepCoordinates": "timestep_coordinates",
                "modulationWeights": "modulation_weights",
                "ropeCosine": "rope_cosine",
                "ropeSine": "rope_sine",
                "graphSalt": graph_salt_name,
            },
            {"hiddenStates": "hidden_states_out"},
            {
                "hiddenStates": (args.block_scalar_type, [token_dimension, 5376]),
                "timestepCoordinates": (args.block_scalar_type, [4, 8]),
                "modulationWeights": (args.block_scalar_type, [token_dimension, 12]),
                "ropeCosine": (args.block_scalar_type, [token_dimension, 48]),
                "ropeSine": (args.block_scalar_type, [token_dimension, 48]),
                "graphSalt": (args.block_scalar_type, [graph_salt_width]),
            },
            {"hiddenStates": (args.block_scalar_type, [token_dimension, 5376])},
            function_name=entrypoint_name,
            compute_units=args.preferred_compute,
            logical_layer_count=layer_count,
        ))
    final_video = stage_manifest(
        relative(f"{args.asset_prefix}-final-video-dynamic"),
        {
            "hiddenStates": "hidden_states",
            "timestepCoordinate": "timestep_coordinate",
        },
        {"videoRows": "video_rows"},
        {
            "hiddenStates": ("bfloat16", [-1, 5376]),
            "timestepCoordinate": ("bfloat16", [1, 8]),
        },
        {"videoRows": ("bfloat16", [-1, 96])},
        compute_units=args.preferred_compute,
    )
    final_audio = stage_manifest(
        relative(f"{args.asset_prefix}-final-audio-dynamic"),
        {
            "hiddenStates": "hidden_states",
            "timestepCoordinate": "timestep_coordinate",
        },
        {"audioRows": "audio_rows"},
        {
            "hiddenStates": ("bfloat16", [-1, 5376]),
            "timestepCoordinate": ("bfloat16", [1, 8]),
        },
        {"audioRows": ("bfloat16", [-1, 32])},
        compute_units=args.preferred_compute,
    )
    fragment = {
        "textRefiner": text_refiner,
        "videoProjection": video_projection,
        "audioProjection": audio_projection,
        "blocks": blocks,
        "finalVideo": final_video,
        "finalAudio": final_audio,
        "adalnTableAsset": f"{configuration_name}/adaln_t_table.f32",
        "ropeInverseFrequencyAsset": f"{configuration_name}/rope_inv_freq.f32",
        "dynamicMaximumTokens": args.dynamic_max_tokens,
    }
    fragment_path = args.fragment_output or (
        args.compiled_directory.parent
        / f"{args.asset_prefix}-denoiser-composite-manifest.json"
    )
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(
        json.dumps(fragment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(fragment_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
