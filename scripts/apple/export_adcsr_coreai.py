# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "coreai-core==1.0.0b2",
#   "coreai-torch>=0.4.1",
#   "diffusers==0.37.1",
#   "transformers",
#   "huggingface_hub",
#   "pytorch-lightning",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

"""Export the official AdcSR x4 checkpoint to a fixed-shape Core AI asset.

The FP16 mode is intentionally marked experimental.  AdcSR's SD 2.1-derived
attention and normalization path is known to overflow on low-variance tiles;
the generated asset must pass the runtime finite-output gate before use.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_weights/adcsr_x4_float16-experimental.aimodel"),
    )
    parser.add_argument(
        "--precision",
        choices=("float16", "mixed16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--torch-probes",
        action="store_true",
        help="run slow CPU finite-output probes before export",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_model() -> nn.Module:
    from diffusers import UNet2DConditionModel
    from diffusers.models.autoencoders.vae import Decoder

    source = snapshot_download(
        "Guaishou74851/AdcSR",
        allow_patterns=["*.py", "bsr/**"],
    )
    sys.path.insert(0, source)
    from model import Net  # type: ignore[import-not-found]  # noqa: PLC0415

    config = UNet2DConditionModel.load_config(
        "flax/stable-diffusion-2-1-base", subfolder="unet"
    )
    unet = UNet2DConditionModel.from_config(config)
    decoder = Decoder(
        in_channels=4,
        out_channels=3,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[64, 128, 256, 256],
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",
        mid_block_add_attention=True,
    )
    half_decoder = torch.load(
        hf_hub_download(
            "Guaishou74851/AdcSR", "weight/pretrained/halfDecoder.ckpt"
        ),
        map_location="cpu",
        weights_only=False,
    )
    decoder.load_state_dict(
        {
            key.replace("decoder.", ""): value
            for key, value in half_decoder["state_dict"].items()
            if "decoder" in key
        },
        strict=True,
    )
    model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))
    model.load_state_dict(
        torch.load(
            hf_hub_download("Guaishou74851/AdcSR", "weight/net_params_200.pkl"),
            map_location="cpu",
            weights_only=False,
        )
    )
    return torch.nn.Sequential(
        model.module,
        *decoder.up_blocks,
        decoder.conv_norm_out,
        decoder.conv_act,
        decoder.conv_out,
    ).eval()


class AdcSR(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model(lr)


def _upcast_group_norm(module: nn.GroupNorm, value: torch.Tensor) -> torch.Tensor:
    output = F.group_norm(
        value.float(),
        module.num_groups,
        module.weight.float() if module.weight is not None else None,
        module.bias.float() if module.bias is not None else None,
        module.eps,
    )
    return output.to(value.dtype)


def _upcast_layer_norm(module: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
    output = F.layer_norm(
        value.float(),
        module.normalized_shape,
        module.weight.float() if module.weight is not None else None,
        module.bias.float() if module.bias is not None else None,
        module.eps,
    )
    return output.to(value.dtype)


def _upcast_self_attention(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    query = module.to_q(hidden_states)
    key = module.to_k(hidden_states)
    value = module.to_v(hidden_states)
    batch, tokens, inner = query.shape
    heads = module.heads
    head_dim = inner // heads
    query = query.view(batch, tokens, heads, head_dim).transpose(1, 2).float()
    key = key.view(batch, tokens, heads, head_dim).transpose(1, 2).float()
    value = value.view(batch, tokens, heads, head_dim).transpose(1, 2).float()
    weights = torch.matmul(query, key.transpose(-1, -2)) * module.scale
    weights = torch.softmax(weights, dim=-1)
    output = torch.matmul(weights, value)
    output = output.transpose(1, 2).reshape(batch, tokens, inner).to(hidden_states.dtype)
    output = module.to_out[0](output)
    output = module.to_out[1](output)
    return output


def apply_mixed16_stability(module: nn.Module) -> None:
    from diffusers.models.attention import BasicTransformerBlock

    for child in module.modules():
        if isinstance(child, nn.GroupNorm):
            child.forward = types.MethodType(_upcast_group_norm, child)
        elif isinstance(child, nn.LayerNorm):
            child.forward = types.MethodType(_upcast_layer_norm, child)
        if isinstance(child, BasicTransformerBlock):
            child.attn1.forward = types.MethodType(_upcast_self_attention, child.attn1)


def finite_probe(module: nn.Module, dtype: torch.dtype) -> None:
    probes = {
        "flat-zero": torch.zeros(1, 3, 128, 128, dtype=dtype),
        "flat-gray": torch.full((1, 3, 128, 128), 0.1, dtype=dtype),
        "low-variance": torch.full((1, 3, 128, 128), -0.2, dtype=dtype),
        "structured": torch.zeros(1, 3, 128, 128, dtype=dtype),
    }
    probes["low-variance"][:, :, 64, 64] += 1 / 255
    probes["structured"][:, :, ::4, ::4] = 0.5
    with torch.inference_mode():
        for name, probe in probes.items():
            output = module(probe)
            finite = bool(torch.isfinite(output).all())
            print(
                f"[TORCH] {name}: finite={finite} "
                f"min={output.float().amin().item():.6f} "
                f"max={output.float().amax().item():.6f}"
            )
            if not finite:
                raise RuntimeError(f"PyTorch {name} probe produced non-finite output")


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    import coreai_torch
    from coreai.runtime import AIModelAssetMetadata

    dtype = torch.float32 if args.precision == "float32" else torch.float16
    module = AdcSR(build_model()).to(dtype=dtype).eval()
    if args.precision == "mixed16":
        apply_mixed16_stability(module)
    print(f"[PARAM] {sum(p.numel() for p in module.parameters()) / 1e6:.2f}M")
    if args.torch_probes:
        finite_probe(module, dtype)

    example = torch.zeros(1, 3, 128, 128, dtype=dtype)
    example[:, :, ::4, ::4] = 0.5
    exported = torch.export.export(module, args=(example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported_program=exported,
        input_names=["lr"],
        output_names=["sr"],
    )
    program = converter.to_coreai()
    program.optimize()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = AIModelAssetMetadata()
    metadata.author = "Bingchen Li et al. (AdcSR, CVPR 2025)"
    metadata.license = (
        "Apache-2.0 (code); weights derived from Stable Diffusion 2.1 "
        "(CreativeML OpenRAIL++-M)"
    )
    metadata.model_description = (
        f"Experimental AdcSR x4 {args.precision} export. "
        "lr [1,3,128,128] -> sr [1,3,512,512]; host color matching."
    )
    metadata.creation_date = int(time.time())
    program.save_asset(args.output, metadata)
    print(f"[OK] saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
