#!/usr/bin/env python3
"""Reproducibly export the complete fixed-profile H3 Qwen encoder to Core AI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--compiled-directory", type=Path, required=True)
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument("--preferred-compute", default="gpu")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Reuse compiled assets and only regenerate the runtime manifest.",
    )
    parser.add_argument(
        "--repeated-stages-only",
        action="store_true",
        help="Export only the weight-distinct vision/language layers; reuse static assets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stage_manifest(
    asset: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    input_constraints: dict[str, tuple[str, list[int]]],
    output_constraints: dict[str, tuple[str, list[int]]],
    *,
    compute_units: str | None = None,
) -> dict:
    def constraints(values):
        return {
            name: {"scalarType": scalar, "shape": shape}
            for name, (scalar, shape) in values.items()
        }

    result = {
        "backend": "coreai",
        "asset": asset,
        "function": "main",
        "inputs": inputs,
        "outputs": outputs,
        "inputConstraints": constraints(input_constraints),
        "outputConstraints": constraints(output_constraints),
    }
    if compute_units is not None:
        result["computeUnits"] = compute_units
    return result


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    scripts = Path(__file__).resolve().parent
    args.source_directory.mkdir(parents=True, exist_ok=True)
    args.compiled_directory.mkdir(parents=True, exist_ok=True)
    common = ["--checkpoint", str(args.checkpoint)]
    overwrite = ["--overwrite"] if args.overwrite else []
    jobs: list[tuple[str, list[str]]] = []
    jobs.append(
        (
            "qwen-token-embedding-s4152",
            [
                sys.executable,
                str(scripts / "export_minimax_h3_qwen_embedding.py"),
                *common,
                "--sequence-length",
                "4152",
            ],
        )
    )
    jobs.append(
        (
            "qwen-vision-patch-b1",
            [
                sys.executable,
                str(scripts / "export_minimax_h3_qwen_vision.py"),
                "--stage",
                "patch",
                *common,
                "--batch",
                "1",
                "--skip-reference",
            ],
        )
    )
    for index in range(27):
        jobs.append(
            (
                f"qwen-vision-block{index:02d}-b1-u1",
                [
                    sys.executable,
                    str(scripts / "export_minimax_h3_qwen_vision.py"),
                    "--stage",
                    "block",
                    "--block",
                    str(index),
                    *common,
                    "--batch",
                    "1",
                    "--skip-reference",
                ],
            )
        )
    for index in range(3):
        jobs.append(
            (
                f"qwen-vision-deepstack{index}-b1",
                [
                    sys.executable,
                    str(scripts / "export_minimax_h3_qwen_vision.py"),
                    "--stage",
                    "deepstack",
                    "--deepstack",
                    str(index),
                    *common,
                    "--batch",
                    "1",
                    "--skip-reference",
                ],
            )
        )
    jobs.append(
        (
            "qwen-vision-merger-b1",
            [
                sys.executable,
                str(scripts / "export_minimax_h3_qwen_vision.py"),
                "--stage",
                "merger",
                *common,
                "--batch",
                "1",
                "--skip-reference",
            ],
        )
    )
    for index in range(50):
        jobs.append(
            (
                f"qwen-language-layer{index:02d}-s4152-lut4-u1",
                [
                    sys.executable,
                    str(scripts / "export_minimax_h3_qwen_language_layer.py"),
                    *common,
                    "--layer",
                    str(index),
                    "--sequence-length",
                    "4152",
                    "--skip-reference",
                ],
            )
        )

    for job_index, (stem, command) in enumerate(jobs, start=1):
        portable = args.source_directory / f"{stem}.aimodel"
        compiled = args.compiled_directory / f"{stem}.{args.architecture}.aimodelc"
        print(f"[{job_index}/{len(jobs)}] {stem}", flush=True)
        repeated = stem.startswith("qwen-vision-block") or stem.startswith(
            "qwen-language-layer"
        )
        should_build = not args.manifest_only and (
            not args.repeated_stages_only or repeated
        )
        if should_build and (args.overwrite or not portable.exists()):
            run([*command, "--output", str(portable), *overwrite])
        if should_build and (args.overwrite or not compiled.exists()):
            run(
                [
                    "xcrun",
                    "coreai-build",
                    "compile",
                    str(portable),
                    "--output",
                    str(args.compiled_directory),
                    "--platform",
                    "macOS",
                    "--min-deployment-version",
                    "27.0",
                    "--preferred-compute",
                    args.preferred_compute,
                    "--architecture",
                    args.architecture,
                ]
            )
        if not compiled.is_dir():
            generated = args.compiled_directory / f"{stem}.aimodelc"
            if generated.is_dir():
                generated.rename(compiled)
        if not compiled.is_dir():
            raise FileNotFoundError(compiled)

    relative = lambda stem: f"coreai/{stem}.{args.architecture}.aimodelc"
    embedding = stage_manifest(
        relative("qwen-token-embedding-s4152"),
        {"inputIDs": "input_ids"},
        {"tokenEmbeddings": "token_embeddings"},
        {"inputIDs": ("int32", [1, 4152])},
        {"tokenEmbeddings": ("float16", [1, 4152, 5120])},
        compute_units=args.preferred_compute,
    )
    patch = stage_manifest(
        relative("qwen-vision-patch-b1"),
        {"pixelPatches": "pixel_patches"},
        {"visionHidden": "vision_hidden"},
        {"pixelPatches": ("float16", [1, 1620, 1536])},
        {"visionHidden": ("float16", [1, 1620, 1152])},
        compute_units=args.preferred_compute,
    )
    vision_blocks = [
        stage_manifest(
            relative(f"qwen-vision-block{index:02d}-b1-u1"),
            {"visionHidden": "vision_hidden"},
            {"visionHiddenOut": "vision_hidden_out"},
            {"visionHidden": ("float16", [1, 1620, 1152])},
            {"visionHiddenOut": ("float16", [1, 1620, 1152])},
            compute_units=args.preferred_compute,
        )
        for index in range(27)
    ]
    deepstack = [
        stage_manifest(
            relative(f"qwen-vision-deepstack{index}-b1"),
            {"visionHidden": "vision_hidden"},
            {"deepstack": f"deepstack_{index}"},
            {"visionHidden": ("float16", [1, 1620, 1152])},
            {"deepstack": ("float16", [1, 405, 5120])},
            compute_units=args.preferred_compute,
        )
        for index in range(3)
    ]
    merger = stage_manifest(
        relative("qwen-vision-merger-b1"),
        {"visionHidden": "vision_hidden"},
        {"visionMerged": "vision_merged"},
        {"visionHidden": ("float16", [1, 1620, 1152])},
        {"visionMerged": ("float16", [1, 405, 5120])},
        compute_units=args.preferred_compute,
    )
    language_layers = [
        stage_manifest(
            relative(f"qwen-language-layer{index:02d}-s4152-lut4-u1"),
            {
                "hiddenStates": "hidden_states",
                "ropeCosine": "rope_cosine",
                "ropeSine": "rope_sine",
            },
            {"hiddenStatesOut": "hidden_states_out"},
            {
                "hiddenStates": ("float16", [1, 4152, 5120]),
                "ropeCosine": ("float16", [1, 1, 4152, 128]),
                "ropeSine": ("float16", [1, 1, 4152, 128]),
            },
            {"hiddenStatesOut": ("float16", [1, 4152, 5120])},
            compute_units=args.preferred_compute,
        )
        for index in range(50)
    ]
    fragment = {
        "sequenceLength": 4152,
        "visionBlockBatch": 10,
        "visionPatchesPerBlock": 1620,
        "visualTokensPerBlock": 405,
        "tokenEmbedding": embedding,
        "visionPatch": patch,
        "visionBlocks": vision_blocks,
        "visionDeepstackMergers": deepstack,
        "visionFinalMerger": merger,
        "languageLayers": language_layers,
        "deepstackVisionBlockIndices": [8, 16, 24],
        "deepstackLanguageLayerIndices": [0, 1, 2],
    }
    fragment_path = args.compiled_directory.parent / "qwen-composite-manifest.json"
    fragment_path.write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")
    print(fragment_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
