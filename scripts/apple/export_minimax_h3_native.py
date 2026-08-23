#!/usr/bin/env python3
"""Export MiniMax H3 stages for Mioh's native Swift runtime.

The shipped runtime never imports Python. This build-time exporter reads the
official ComfyUI model definition and checkpoint, then writes a Core ML model
package or a portable Core AI program asset.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "audio-encoder",
            "audio-decoder",
            "video-encoder-tile",
            "video-decoder-tile",
        ),
        required=True,
    )
    parser.add_argument("--backend", choices=("coreml", "coreai"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-samples", type=int, default=320_000)
    parser.add_argument("--audio-latent-frames", type=int, default=400)
    parser.add_argument("--video-tile-size", type=int, default=256)
    parser.add_argument("--video-latent-frames", type=int, default=7)
    parser.add_argument("--trace-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--coreml-float32-boundary", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class _PlainOperations:
    Linear = torch.nn.Linear
    LayerNorm = torch.nn.LayerNorm
    Conv1d = torch.nn.Conv1d
    ConvTranspose1d = torch.nn.ConvTranspose1d
    GroupNorm = torch.nn.GroupNorm
    RMSNorm = torch.nn.RMSNorm


class _PlainConv3d(torch.nn.Conv3d):
    def forward(self, value: torch.Tensor, autopad: str | None = None) -> torch.Tensor:
        if autopad != "causal_zero":
            return super().forward(value)
        # A one-frame causal convolution sees only the final temporal tap; all
        # earlier taps multiply the implicit zero history.
        weight = self.weight[:, :, -value.shape[2] :, :, :]
        return torch.nn.functional.conv3d(
            value,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


_PlainOperations.Conv3d = _PlainConv3d


def _install_comfy_ops_shim() -> None:
    """Provide only the inference ops used by the H3 audio VAE definition."""

    ops = types.ModuleType("comfy.ops")
    ops.disable_weight_init = _PlainOperations
    ops.cast_to_input = lambda value, reference: value.to(
        device=reference.device, dtype=reference.dtype
    )

    def cast_bias_weight(module, reference, offloadable=False):
        del offloadable
        weight = ops.cast_to_input(module.weight, reference)
        bias = None if module.bias is None else ops.cast_to_input(module.bias, reference)
        return weight, bias, None

    ops.cast_bias_weight = cast_bias_weight
    ops.uncast_bias_weight = lambda *args, **kwargs: None
    ops.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.ops = ops
    sys.modules["comfy"] = comfy
    sys.modules["comfy.ops"] = ops


def load_audio_vae(comfy_root: Path, checkpoint: Path) -> torch.nn.Module:
    _install_comfy_ops_shim()
    source = comfy_root / "comfy/ldm/minimax/audio_vae.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("mioh_minimax_h3_audio_vae", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MiniMaxH3AudioVAE()
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval()


def _install_video_vae_shims() -> None:
    _install_comfy_ops_shim()
    comfy = sys.modules["comfy"]
    ops = sys.modules["comfy.ops"]

    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
    comfy.model_management = model_management
    sys.modules["comfy.model_management"] = model_management

    rmsnorm = types.ModuleType("comfy.rmsnorm")
    rmsnorm.rms_norm = lambda value, weight, eps: torch.nn.functional.rms_norm(
        value, (value.shape[-1],), weight, eps
    )
    comfy.rmsnorm = rmsnorm
    sys.modules["comfy.rmsnorm"] = rmsnorm

    quant_ops = types.ModuleType("comfy.quant_ops")

    def apply_rope_split_half(query, key, rotation):
        pairs = rotation.shape[-3]
        cosine = rotation[..., 0, 0].squeeze(2).unsqueeze(2)
        sine = rotation[..., 1, 0].squeeze(2).unsqueeze(2)

        def rotate(value):
            first = value[..., :pairs]
            second = value[..., pairs : 2 * pairs]
            return torch.cat(
                (first * cosine - second * sine, first * sine + second * cosine),
                dim=-1,
            )

        return rotate(query), rotate(key)

    quant_ops.ck = types.SimpleNamespace(apply_rope_split_half=apply_rope_split_half)
    comfy.quant_ops = quant_ops
    sys.modules["comfy.quant_ops"] = quant_ops

    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    modules = types.ModuleType("comfy.ldm.modules")
    modules.__path__ = []
    attention = types.ModuleType("comfy.ldm.modules.attention")

    def optimized_attention(query, key, value, heads, skip_reshape=False):
        del heads
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        if skip_reshape:
            return output.transpose(1, 2).flatten(2)
        return output

    attention.optimized_attention = optimized_attention
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.modules"] = modules
    sys.modules["comfy.ldm.modules.attention"] = attention


def load_video_vae(comfy_root: Path, checkpoint: Path) -> torch.nn.Module:
    _install_video_vae_shims()
    source = comfy_root / "comfy/ldm/minimax/vae.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location("mioh_minimax_h3_video_vae", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def flat_temporal_group_norm(layer, value):
        if value.dim() != 5:
            return torch.nn.functional.group_norm(
                value, layer.num_groups, layer.weight, layer.bias, layer.eps
            )
        batch, channels, frames, height, width = value.shape
        flattened = value.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        normalized = torch.nn.functional.group_norm(
            flattened, layer.num_groups, layer.weight, layer.bias, layer.eps
        )
        return normalized.reshape(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )

    # The reference's singleton-depth rank-5 GroupNorm is mathematically the
    # same operation, but Core ML lowers it through a rank-6 intermediate.
    # Flattening time into batch keeps the exact statistics and rank <= 5.
    module.TemporalIsolatedGroupNorm.forward = flat_temporal_group_norm

    def exportable_attention(layer, value, rotary_pos_emb=None):
        batch, sequence, _ = value.shape
        qkv = layer.to_qkv(value).view(batch, sequence, -1, 3 * layer.dim_head)
        query, key, val = torch.chunk(qkv, 3, dim=-1)
        query = torch.nn.functional.rms_norm(
            query, (query.shape[-1],), layer.norm_q.weight, layer.norm_q.eps
        )
        key = torch.nn.functional.rms_norm(
            key, (key.shape[-1],), layer.norm_k.weight, layer.norm_k.eps
        )
        if rotary_pos_emb is not None:
            rotary_dimensions = rotary_pos_emb.shape[-3] * 2
            rotated_query, rotated_key = sys.modules["comfy.quant_ops"].ck.apply_rope_split_half(
                query[..., :rotary_dimensions],
                key[..., :rotary_dimensions],
                rotary_pos_emb,
            )
            query = torch.cat((rotated_query, query[..., rotary_dimensions:]), dim=-1)
            key = torch.cat((rotated_key, key[..., rotary_dimensions:]), dim=-1)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), val.transpose(1, 2)
        )
        # nan_to_num lowers through aten.isnan, which Core AI 0.4.1 cannot
        # represent. Valid finite VAE inputs do not exercise that emergency
        # guard, so omit it at the export boundary.
        return layer.to_out(attended.transpose(1, 2).flatten(2))

    module.Attention.forward = exportable_attention

    def exportable_transformer_block(layer, value, rotary_pos_emb=None):
        # Core ML's PyTorch frontend has no lowering for the in-place
        # addcmul_ used by the reference. These two expressions are exactly
        # x + branch * learned_scale and preserve the reference arithmetic.
        attention_value = torch.nn.functional.rms_norm(
            value, (value.shape[-1],), layer.norm1.weight, layer.norm1.eps
        )
        value = value + layer.attn(attention_value, rotary_pos_emb) * layer.scale1.to(
            value
        )
        feed_forward_value = torch.nn.functional.rms_norm(
            value, (value.shape[-1],), layer.norm2.weight, layer.norm2.eps
        )
        return value + layer.ff(feed_forward_value) * layer.scale2.to(value)

    module.TransformerBlock.forward = exportable_transformer_block
    model = module.MiniMaxH3VideoVAE(tiling=False)
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


class AudioEncoder(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(audio)


class AudioDecoder(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, audio_latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(audio_latent)


class VideoEncoderTile(torch.nn.Module):
    """One 17-frame, 256px spatial tile; Swift owns temporal/spatial stitching."""

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.register_buffer("pixel_mean", vae.pixel_mean)
        self.register_buffer("pixel_std", vae.pixel_std)
        self.register_buffer("latents_mean", vae.latents_mean)
        self.register_buffer("latents_std", vae.latents_std)

    def forward(self, video_tile: torch.Tensor) -> torch.Tensor:
        # AVFoundation supplies RGB [0, 1]. The Comfy VAE wrapper would first
        # map that to [-1, 1], after which the model's normalization maps it
        # back to ImageNet-normalized [0, 1]. Combine those two affine steps.
        normalized = (video_tile - self.pixel_mean.to(video_tile)) / self.pixel_std.to(
            video_tile
        )
        moments = self.quant_conv(self.encoder(normalized))
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return ((mean - latent_mean) / latent_std).to(video_tile.dtype)


class VideoDecoderTile(torch.nn.Module):
    """One seven-token, 16x16 latent tile; Swift owns blending and final RGB."""

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.register_buffer("latents_mean", vae.latents_mean)
        self.register_buffer("latents_std", vae.latents_std)

    def forward(self, video_latent_tile: torch.Tensor) -> torch.Tensor:
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(video_latent_tile)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(video_latent_tile)
        latent = video_latent_tile * latent_std + latent_mean
        # Keep the raw ImageNet-normalized decoder output here. The reference
        # implementation blends spatial/temporal overlaps first and performs
        # the affine RGB conversion and clamp exactly once on finalized frames.
        return self.decoder(self.post_quant_conv(latent)).float()


def model_and_example(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.Tensor, str, str]:
    if args.stage == "video-encoder-tile":
        if args.video_tile_size <= 0 or args.video_tile_size % 16:
            raise ValueError("--video-tile-size must be a positive multiple of 16")
        vae = load_video_vae(args.comfy_root, args.checkpoint)
        return (
            VideoEncoderTile(vae).eval(),
            torch.zeros(
                (1, 3, 17, args.video_tile_size, args.video_tile_size),
                dtype=torch.float16,
            ),
            "video_tile",
            "video_latent_tile",
        )
    if args.stage == "video-decoder-tile":
        if args.video_latent_frames <= 0:
            raise ValueError("--video-latent-frames must be positive")
        vae = load_video_vae(args.comfy_root, args.checkpoint)
        return (
            VideoDecoderTile(vae).eval(),
            torch.zeros(
                (1, 24, args.video_latent_frames, 16, 16),
                dtype=torch.float16,
            ),
            "video_latent_tile",
            "video_raw_tile",
        )
    vae = load_audio_vae(args.comfy_root, args.checkpoint)
    if args.stage == "audio-encoder":
        if args.audio_samples <= 0:
            raise ValueError("--audio-samples must be positive")
        return (
            AudioEncoder(vae).eval(),
            torch.zeros((1, 2, args.audio_samples), dtype=torch.float32),
            "audio",
            "reference_audio_latent",
        )
    if args.audio_latent_frames <= 0:
        raise ValueError("--audio-latent-frames must be positive")
    return (
        AudioDecoder(vae).eval(),
        torch.zeros((1, 32, 2, args.audio_latent_frames), dtype=torch.float32),
        "audio_latent",
        "audio",
    )


def remove_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def export_coreml(
    model: torch.nn.Module,
    example: torch.Tensor,
    input_name: str,
    output_name: str,
    destination: Path,
    trace_device: str,
    float32_boundary: bool,
) -> None:
    import coremltools as ct
    import numpy as np

    device = torch.device(trace_device)
    if float32_boundary:
        model = model.float()
        example = example.float()
    model = model.to(device)
    example = example.to(device)
    traced = torch.jit.trace(model, (example,), strict=True).cpu()
    example = example.cpu()
    numpy_dtype = np.float16 if example.dtype == torch.float16 else np.float32
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=example.shape, dtype=numpy_dtype)],
        outputs=[ct.TensorType(name=output_name, dtype=numpy_dtype)],
        convert_to="mlprogram",
        compute_precision=(
            ct.precision.FLOAT32 if float32_boundary else ct.precision.FLOAT16
        ),
        minimum_deployment_target=ct.target.iOS18,
    )
    converted.user_defined_metadata["mioh.model"] = "minimax-h3"
    converted.user_defined_metadata["mioh.stage"] = destination.stem
    converted.save(str(destination))


def export_coreai(
    model: torch.nn.Module,
    example: torch.Tensor,
    input_name: str,
    output_name: str,
    destination: Path,
) -> None:
    import coreai_torch

    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=[input_name],
        output_names=[output_name],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    remove_existing(args.output, args.overwrite)
    model, example, input_name, output_name = model_and_example(args)
    if not args.skip_reference:
        device = torch.device(args.trace_device)
        with torch.no_grad():
            reference = model.to(device)(example.to(device)).cpu()
        model = model.cpu()
        print(
            f"reference {args.stage}: input={tuple(example.shape)} "
            f"output={tuple(reference.shape)} dtype={reference.dtype}",
            flush=True,
        )
    if args.backend == "coreml":
        export_coreml(
            model,
            example,
            input_name,
            output_name,
            args.output,
            args.trace_device,
            args.coreml_float32_boundary,
        )
    else:
        export_coreai(model, example, input_name, output_name, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
