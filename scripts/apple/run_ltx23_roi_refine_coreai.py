#!/usr/bin/env python3
"""Prototype mioh ROI refine with the LTX-2.3 DiT blocks running as Core AI assets.

Everything except the 48 blocks comes from ltx-core (PyTorch, CPU/MPS): VAE encode/decode, patchify projection,
AdaLN timestep tables, RoPE, output head. The blocks run as the int8 Core AI groups written by
export_ltx23_dit_block_coreai.py. Same recipe as the Swift `roi-refine` command: the source clip is the appended
reference and the clean context, mask tokens start from the source re-noised to sigmas[0].

usage: run_ltx23_roi_refine_coreai.py --checkpoint CKPT --groups DIR --prompt-embedding EMB.safetensors
           --source reference.mp4 --token-mask mask.f32 --output out.mp4 [--sigmas 0.909375,0.421875,0]"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

FPS = 30000 / 1001
SALT_FROM_LAYER = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True, help="directory of groupNN.aimodel")
    parser.add_argument("--prompt-embedding", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--token-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigmas", default="0.909375,0.421875,0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lora-arg", default="build/ltx2-mac/lora/lora_weights_step_01750.safetensors",
                        help="the --lora string the groups were exported with (salt names of older groups)")
    return parser.parse_args()


def read_video(path: Path, frames: int = 33) -> torch.Tensor:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(frames), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    video = np.frombuffer(raw, np.uint8).reshape(frames, 512, 512, 3)
    return torch.from_numpy(video.copy()).permute(3, 0, 1, 2)[None].float() / 127.5 - 1      # (1, 3, F, H, W)


def write_video(path: Path, video: torch.Tensor) -> None:
    frames = ((video[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512",
                                "-r", "30000/1001", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                                str(path)], stdin=subprocess.PIPE)
    process.stdin.write(frames.tobytes())
    process.stdin.close()
    process.wait()


def load_head(checkpoint: Path, device: str):
    """LTXModel with no transformer blocks: patchify projection, AdaLN tables, RoPE settings, output head."""
    from ltx_core.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator

    with safe_open(str(checkpoint), framework="pt") as handle:
        metadata = handle.metadata()
        config = json.loads(metadata["config"])
        config["transformer"]["num_layers"] = 0
        head = LTXVideoOnlyModelConfigurator.from_metadata({"config": config})
        prefix = "model.diffusion_model."
        wanted = {name for name, _ in head.named_parameters()} | {name for name, _ in head.named_buffers()}
        state = {key[len(prefix):]: handle.get_tensor(key) for key in handle.keys()
                 if key.startswith(prefix) and key[len(prefix):] in wanted}
    missing = wanted - set(state)
    if missing:
        raise KeyError(f"head tensors missing from checkpoint: {sorted(missing)[:5]}")
    head.load_state_dict(state, strict=True)
    return head.to(device=device, dtype=torch.bfloat16).eval()


def load_vae(checkpoint: Path, device: str):
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator
    from ltx_core.model.video_vae.model_configurator import (
        VideoDecoderConfigurator,
        video_decoder_sd_ops_for_checkpoint,
    )

    encoder = SingleGPUModelBuilder(model_path=str(checkpoint), model_class_configurator=VideoEncoderConfigurator,
                                    model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER).build(
        device=torch.device(device), dtype=torch.bfloat16)
    decoder = SingleGPUModelBuilder(model_path=str(checkpoint), model_class_configurator=VideoDecoderConfigurator,
                                    model_sd_ops=video_decoder_sd_ops_for_checkpoint(str(checkpoint),
                                                                                     diffusion_vae=False)).build(
        device=torch.device(device), dtype=torch.bfloat16)
    return encoder.eval(), decoder.eval()


class CoreAIBlocks:
    def __init__(self, directory: Path, checkpoint_name: str, lora: str):
        from lada.coreai.source_runtime import load_source_model

        self.checkpoint_name, self.lora = checkpoint_name, lora

        self.runner = asyncio.Runner()
        self.functions = []
        for asset in sorted(directory.glob("group*.aimodel")):
            report = json.loads(Path(f"{asset}.json").read_text())
            model = load_source_model(self.runner, asset, purpose=f"ltx23 {asset.name}")
            # groups exported before the report carried the salt name: rebuild it the way the exporter did
            salt_name = report.get("saltInput") or "salt_" + hashlib.sha256(
                f"{self.checkpoint_name}:{self.lora}:{report['layers'][0]}:{len(report['layers'])}".encode()
            ).hexdigest()[:16]
            self.functions.append((model, model.load_function("main"), salt_name, report["layers"][0] + 1))
        if len(self.functions) * 4 != 48:
            raise ValueError(f"expected 12 four-block groups in {directory}, found {len(self.functions)}")

    def __call__(self, x, modulation, row_select, prompt_modulation, context, cosine, sine) -> np.ndarray:
        from coreai.runtime import NDArray

        fixed = {"modulation": modulation, "row_select": row_select, "prompt_modulation": prompt_modulation,
                 "context": context, "rope_cosine": cosine, "rope_sine": sine}
        fixed = {k: np.ascontiguousarray(v, np.float32) for k, v in fixed.items()}
        for _, function, salt_name, salt_width in self.functions:
            async def call(x=x, function=function, salt_name=salt_name, salt_width=salt_width):
                # fresh NDArrays per call: reusing one across calls fed later groups garbage
                feeds = {k: NDArray(v) for k, v in fixed.items()}
                feeds["x"] = NDArray(np.ascontiguousarray(x, np.float32))
                feeds[salt_name] = NDArray(np.zeros(salt_width, np.float32))
                result = await function(feeds)
                return result[next(iter(result))].numpy()

            x = self.runner.run(call())
        return x


def main() -> int:
    args = parse_args()
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from ltx_core.model.transformer.modality import Modality, TokenTimesteps
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape

    timing = {}
    tic = time.perf_counter()
    head = load_head(args.checkpoint, args.device)
    encoder, decoder = load_vae(args.checkpoint, args.device)
    blocks = CoreAIBlocks(args.groups, args.checkpoint.name, args.lora_arg)
    saved = load_file(str(args.prompt_embedding))
    context = saved["embeddings"].to(args.device, torch.bfloat16)                   # (1, 1024, 4096)
    timing["load"] = time.perf_counter() - tic

    tic = time.perf_counter()
    with torch.no_grad():
        source = encoder(read_video(args.source).to(args.device, torch.bfloat16))     # (1, 128, 5, 16, 16)
    shape = VideoLatentShape.from_torch_shape(source.shape)
    tools = VideoLatentTools(patchifier=VideoLatentPatchifier(1), target_shape=shape, fps=FPS)
    state = tools.create_initial_state(args.device, torch.bfloat16, initial_latent=source)
    state = VideoConditionByReferenceLatent(source).apply_to(state, tools)
    target = shape.frames * shape.height * shape.width
    # the appended reference tokens are zeros in state.latent (the noisy side); their values live in clean_latent
    clean = state.clean_latent.float()                                                # (1, 2 * target, 128)
    draw = torch.from_numpy(np.fromfile(args.token_mask, np.float32) > 0.5).to(args.device)
    draw_all = torch.cat([draw, torch.zeros(clean.shape[1] - target, dtype=torch.bool, device=args.device)])
    sigmas = [float(s) for s in args.sigmas.split(",")]
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(args.seed)).to(args.device)
    latent = torch.where(draw_all[None, :, None], sigmas[0] * noise + (1 - sigmas[0]) * clean, clean)
    indices = torch.where(draw_all, 0, 1).to(torch.int32)[None]                       # row 0 = sigma, row 1 = zero
    row_select = torch.nn.functional.one_hot(indices[0].long(), 2).float().cpu().numpy()
    timing["encode"] = time.perf_counter() - tic

    tic = time.perf_counter()
    for sigma, following in zip(sigmas[:-1], sigmas[1:]):
        modality = Modality(
            latent=latent.to(torch.bfloat16), sigma=torch.tensor([sigma], device=args.device),
            timesteps=TokenTimesteps(torch.tensor([[sigma, 0.0]], device=args.device), indices, zero_index=1),
            positions=state.positions, context=context, enabled=True)
        with torch.no_grad():
            prepared = head.video_args_preprocessor.prepare(modality)
            modulation = prepared.timestep_embedding.modulation.reshape(2, 9, -1)
            cosine, sine = prepared.positional_embeddings
            x = blocks(prepared.x[0].float().cpu().numpy(), modulation.float().cpu().numpy(), row_select,
                       prepared.prompt_timestep.reshape(2, -1).float().cpu().numpy(),
                       prepared.context[0].float().cpu().numpy(),
                       cosine[0].float().cpu().numpy(), sine[0].float().cpu().numpy())
            velocity = head._process_output(head.scale_shift_table, head.norm_out, head.proj_out,
                                            torch.from_numpy(x)[None].to(args.device, torch.bfloat16),
                                            prepared.timestep_embedding).float()
        x0 = (latent - sigma * velocity)[0, :target][draw]
        reference = clean[0, :target][draw]
        print(f"sigma {sigma:.3f}: velocity std {velocity[0, :target][draw].std():.3f}  "
              f"corr(predicted x0, source) {torch.corrcoef(torch.stack([x0.flatten(), reference.flatten()]))[0, 1]:.3f}",
              flush=True)
        latent = torch.where(draw_all[None, :, None], latent + (following - sigma) * velocity, clean)
    timing["denoise"] = time.perf_counter() - tic

    tic = time.perf_counter()
    with torch.no_grad():
        out_latent = tools.patchifier.unpatchify(latent[:, :target].to(torch.bfloat16), shape)
        video = decoder(out_latent)
    write_video(args.output, video)
    timing["decode"] = time.perf_counter() - tic
    print(json.dumps({k: round(v, 2) for k, v in timing.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
