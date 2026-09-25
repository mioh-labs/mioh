#!/usr/bin/env python3
"""Export selected SwiftVR DiT layers as shape-specific Core ML probes.

Each layer runs in a fresh conversion process to bound peak memory. This is
not a complete SwiftVR model pack: ReAE and the DiT input/output components
are separate release gates documented in docs/swiftvr-native-roi-integration.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), required=True)
    parser.add_argument("--latent-frames", type=int, choices=(6, 7), required=True)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if not 0 <= args.first_layer < args.end_layer <= 30:
        parser.error("layer range must be within 0..<30 and nonempty")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not (args.source / "swiftvr" / "models" / "transformer.py").is_file():
        parser.error(f"SwiftVR source not found: {args.source}")

    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    side = 16 if args.scale == 2 else 32
    probe = Path(__file__).with_name("probe_swiftvr_dit_block_coreml.py")
    layers = []
    for index in range(args.first_layer, args.end_layer):
        name = f"dit-block-{index:02d}-t{args.latent_frames}-{args.scale}x-{args.precision}.mlpackage"
        output = root / name
        model_file = output / "Data/com.apple.CoreML/model.mlmodel"
        weight_file = output / "Data/com.apple.CoreML/weights/weight.bin"
        if output.exists() and not (model_file.is_file() and weight_file.is_file()):
            raise RuntimeError(
                f"Incomplete model package at {output}; inspect it before retrying"
            )
        if output.is_dir():
            print(f"[{index + 1}/30] reuse {name}", flush=True)
        else:
            print(f"[{index + 1}/30] export {name}", flush=True)
            command = [
                sys.executable,
                str(probe),
                "--source", str(args.source),
                "--checkpoint", str(args.checkpoint),
                "--output", str(output),
                "--layer-index", str(index),
                "--window-size", "16",
                "--grid-time", str(args.latent_frames),
                "--grid-height", str(side),
                "--grid-width", str(side),
                "--context-tokens", "512",
                "--precision", args.precision,
            ]
            subprocess.run(command, check=True)
            if not (model_file.is_file() and weight_file.is_file()):
                raise RuntimeError(f"Exporter produced an incomplete package: {output}")
        layers.append({"index": index, "asset": name})

    manifest = {
        "format": "swiftvr-dit-coreml-probe-v1",
        "created_at_unix": time.time(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "scale": args.scale,
        "latent_frames": args.latent_frames,
        "precision": args.precision,
        "attention_window": [16, 16],
        "context_tokens": 512,
        "layers": layers,
        "complete": args.first_layer == 0 and args.end_layer == 30,
    }
    manifest_path = root / f"manifest-t{args.latent_frames}-{args.scale}x-{args.precision}.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_path.replace(manifest_path)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
