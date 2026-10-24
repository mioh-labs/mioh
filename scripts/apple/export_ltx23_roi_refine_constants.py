#!/usr/bin/env python3
"""Precompute everything outside the DiT blocks that the Core AI ROI refine needs, for a fixed window and sigma
schedule, so the Swift side only does matrix products and elementwise math.

Writes one .safetensors with
  sigmas                      (S,)          the schedule, ending at 0
  modulation_{i}              (2, 9, 4096)  AdaLN rows for [sigma_i, 0]   (blocks add their own table)
  prompt_modulation_{i}       (2, 4096)     prompt AdaLN for sigma_i
  embedding_{i}               (2, 4096)     output-head timestep rows for [sigma_i, 0]
  rope_cosine, rope_sine      (32, T, 64)   target tokens then the appended reference tokens
  context                     (1024, 4096)  prompt context after the transformer's own preparation
  patchify_weight/bias        (4096, 128) / (4096,)
  proj_out_weight/bias        (128, 4096) / (128,)
  scale_shift_table           (2, 4096)
metadata: fps, frames, width, height, prompt.

usage: export_ltx23_roi_refine_constants.py --checkpoint CKPT --prompt-embedding EMB --output CONST.safetensors
           [--fps 29.97002997] [--sigmas 1,0.99375,...,0.421875,0]

The default exports every distilled sigma so the server can run any step count (1-8; ROIRefineConstants.scheduled)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt-embedding", type=Path)
    parser.add_argument(
        "--base-constants", type=Path,
        help="Derive a smaller rectangular geometry from existing square constants without loading the checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30000 / 1001)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--size", type=int, default=512, help="Square size (legacy shorthand)")
    parser.add_argument("--width", type=int, help="Pixel width; overrides --size")
    parser.add_argument("--height", type=int, help="Pixel height; overrides --size")
    parser.add_argument("--sigmas", default="1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    width = args.width or args.size
    height = args.height or args.size
    if width % 32 or height % 32:
        raise SystemExit("--width and --height must be divisible by 32")
    if args.base_constants:
        return derive_from_square(args, width, height)
    if not args.checkpoint or not args.prompt_embedding:
        raise SystemExit("--checkpoint and --prompt-embedding are required without --base-constants")
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from ltx_core.model.transformer.modality import Modality, TokenTimesteps
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape

    from run_ltx23_roi_refine_coreai import load_head

    head = load_head(args.checkpoint, "cpu").float()
    latent_frames = (args.frames - 1) // 8 + 1
    source = torch.zeros(1, 128, latent_frames, height // 32, width // 32)
    shape = VideoLatentShape.from_torch_shape(source.shape)
    tools = VideoLatentTools(patchifier=VideoLatentPatchifier(1), target_shape=shape, fps=args.fps)
    state = VideoConditionByReferenceLatent(source).apply_to(
        tools.create_initial_state("cpu", torch.float32, initial_latent=source), tools)
    tokens = state.latent.shape[1]
    saved = load_file(str(args.prompt_embedding))
    context = saved["embeddings"].float()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    indices = torch.zeros(1, tokens, dtype=torch.int32)
    out = {"sigmas": torch.tensor(sigmas)}
    with torch.no_grad():
        for i, sigma in enumerate(sigmas[:-1]):
            prepared = head.video_args_preprocessor.prepare(Modality(
                latent=state.latent, sigma=torch.tensor([sigma]),
                timesteps=TokenTimesteps(torch.tensor([[sigma, 0.0]]), indices, zero_index=1),
                positions=state.positions, context=context, enabled=True))
            out[f"modulation_{i}"] = prepared.timestep_embedding.modulation.reshape(2, 9, -1)
            out[f"prompt_modulation_{i}"] = prepared.prompt_timestep.reshape(2, -1)
            out[f"embedding_{i}"] = prepared.timestep_embedding.embedding.reshape(2, -1)
            if i == 0:
                cosine, sine = prepared.positional_embeddings
                out["rope_cosine"], out["rope_sine"] = cosine[0], sine[0]
                out["context"] = prepared.context[0]
    out["patchify_weight"], out["patchify_bias"] = head.patchify_proj.weight, head.patchify_proj.bias
    out["proj_out_weight"], out["proj_out_bias"] = head.proj_out.weight, head.proj_out.bias
    out["scale_shift_table"] = head.scale_shift_table
    out = {k: v.float().contiguous() for k, v in out.items()}
    save_file(out, str(args.output), metadata={
        "fps": repr(args.fps), "frames": str(args.frames), "width": str(width), "height": str(height),
        "size": str(args.size) if width == height else "rectangular", "tokens": str(tokens),
        "prompt": args.prompt_embedding.name})
    print(args.output, {k: tuple(v.shape) for k, v in out.items()})
    return 0


def derive_from_square(args: argparse.Namespace, width: int, height: int) -> int:
    """Crop deterministic RoPE rows from square constants; every other tensor is geometry-independent."""
    with safe_open(args.base_constants, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    source = load_file(str(args.base_constants))
    cosine = source["rope_cosine"]
    sine = source["rope_sine"]
    latent_frames = (args.frames - 1) // 8 + 1
    source_target_tokens = cosine.shape[1] // 2
    source_area, remainder = divmod(source_target_tokens, latent_frames)
    source_side = int(source_area**0.5)
    if remainder or source_side * source_side != source_area:
        raise SystemExit(f"base constants are not square: rope tokens={cosine.shape[1]}")
    target_height, target_width = height // 32, width // 32
    if target_height > source_side or target_width > source_side:
        raise SystemExit(f"requested {width}x{height} exceeds {source_side * 32}px base constants")
    indices = []
    for half in range(2):
        half_start = half * source_target_tokens
        for frame in range(latent_frames):
            frame_start = half_start + frame * source_area
            for y in range(target_height):
                for x in range(target_width):
                    indices.append(frame_start + y * source_side + x)
    out = dict(source)
    out["rope_cosine"] = cosine[:, indices].contiguous()
    out["rope_sine"] = sine[:, indices].contiguous()
    fps = float(metadata.get("fps", args.fps))
    tokens = len(indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.output), metadata={
        "fps": repr(fps), "frames": str(args.frames), "width": str(width), "height": str(height),
        "size": "rectangular", "tokens": str(tokens), "derived_from": args.base_constants.name,
        "prompt": metadata.get("prompt", "unknown"),
    })
    print(args.output, {"rope": tuple(out["rope_cosine"].shape), "tokens": tokens})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
