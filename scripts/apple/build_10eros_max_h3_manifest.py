#!/usr/bin/env python3
"""Assemble Mioh's reproducible native 10Eros-Max H3 runtime manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--denoiser-manifest", type=Path)
    parser.add_argument(
        "--model-identifier",
        default="10eros-max-h3-turbo-hybrid-beta3-native-v1",
    )
    parser.add_argument(
        "--conditioning-mode",
        choices=("ref2va", "fl2va"),
        default="ref2va",
    )
    return parser.parse_args()


def stage(asset, inputs, outputs, input_constraints, output_constraints):
    def constraints(values):
        return {
            name: {"scalarType": scalar, "shape": shape}
            for name, (scalar, shape) in values.items()
        }

    return {
        "backend": "coreai",
        "asset": asset,
        "function": "main",
        "inputs": inputs,
        "outputs": outputs,
        "inputConstraints": constraints(input_constraints),
        "outputConstraints": constraints(output_constraints),
    }


def simple_flow_sigmas(steps: int, shift: float) -> list[float]:
    """Match ComfyUI's simple scheduler for ModelSamplingAV/DiscreteFlow."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    result = []
    # ComfyUI samples from ModelSamplingDiscreteFlow.sigmas, a 1000-entry
    # table built from timesteps 1...1000.  Its Simple scheduler uses
    # int(index * 1000 / steps), so calculating 1-indexed table positions is
    # required for exact parity (using index / steps is only approximate).
    timestep_count = 1000
    for index in range(steps):
        table_index = timestep_count - 1 - int(index * timestep_count / steps)
        timestep = (table_index + 1) / timestep_count
        result.append(shift * timestep / (1.0 + (shift - 1.0) * timestep))
    result.append(0.0)
    return result


def main() -> int:
    args = parse_args()
    root = args.model_directory.resolve()
    qwen_path = root / "qwen-composite-manifest.json"
    denoiser_path = (
        args.denoiser_manifest.resolve()
        if args.denoiser_manifest
        else root / "10eros-max-h3-denoiser-composite-manifest.json"
    )
    for path in (qwen_path, denoiser_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stages = {
        "videoEncoder": stage(
            "coreai/video-encoder-tile256.aimodelc",
            {"videoTile": "video_tile"},
            {"videoLatentTile": "video_latent_tile"},
            {"videoTile": ("float16", [1, 3, 17, 256, 256])},
            {"videoLatentTile": ("float16", [1, 24, 5, 16, 16])},
        ),
        "audioEncoder": stage(
            "coreai/audio-encoder.aimodelc",
            {"audio": "audio"},
            {"referenceAudioLatent": "reference_audio_latent"},
            {"audio": ("float32", [1, 2, 320000])},
            {"referenceAudioLatent": ("float32", [1, 32, 2, 400])},
        ),
        "videoDecoder": stage(
            "coreai/video-decoder-raw-tile7x16.aimodelc",
            {"videoLatentTile": "video_latent_tile"},
            {"videoRawTile": "video_raw_tile"},
            {"videoLatentTile": ("float16", [1, 24, 7, 16, 16])},
            {"videoRawTile": ("float32", [1, 3, 28, 256, 256])},
        ),
        "audioDecoder": stage(
            "coreai/audio-decoder.aimodelc",
            {"audioLatent": "audio_latent"},
            {"audio": "audio"},
            {"audioLatent": ("float32", [1, 32, 2, 405])},
            {"audio": ("float32", [1, 2, 324000])},
        ),
    }
    sigmas = (
        simple_flow_sigmas(steps=20, shift=12.0)
        if args.conditioning_mode == "fl2va"
        # The current 10Eros-Max TURBO recipe specifies ER-SDE with the
        # standard Simple scheduler at six steps.  Keep this derived from the
        # flow shift instead of freezing the older hand-tuned seven-step list.
        else simple_flow_sigmas(steps=6, shift=12.0)
    )
    manifest = {
        "schemaVersion": 1,
        "modelIdentifier": args.model_identifier,
        "conditioningMode": args.conditioning_mode,
        "tokenizerDirectory": "tokenizer/qwen25",
        "qwenComposite": json.loads(qwen_path.read_text(encoding="utf-8")),
        "denoiserComposite": json.loads(
            denoiser_path.read_text(encoding="utf-8")
        ),
        "stages": stages,
        "sampler": (
            "res_multistep"
            if args.conditioning_mode == "fl2va"
            else "er_sde"
        ),
        "samplerNoise": 1.0,
        "samplerMaxStage": 3,
        "sigmas": sigmas,
        "videoShift": 12.0,
        "audioShift": 3.0,
        "visualConditionNoiseAug": 0.999,
        "audioConditionNoiseAug": 1.0,
    }
    output = (args.output or (root / "manifest.json")).resolve()
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
