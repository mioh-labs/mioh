"""Build the inference-only FlashVSR-v1.1 model bundle used by mioh.

The normal checkpoint is a general Wan DiT checkpoint stored in FP32.  The
FlashVSR tiny/tiny-long upscaler always uses one fixed timestep and one fixed
positive prompt, then converts every weight to BF16 before inference.  This
builder performs that conversion once, bakes the fixed cross-attention KV
caches and timestep embeddings, and omits the VAE used only by ``full`` mode.

The source model is never modified or deleted.  A separate, auditable bundle
is written so the reference checkpoint remains available for comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_SOURCE = Path(
    "models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors"
)
DEFAULT_PROMPT = Path("models/posi_prompt.pth")
DEFAULT_LQ = Path("models/FlashVSR-v1.1/LQ_proj_in.ckpt")
DEFAULT_DECODER = Path("models/FlashVSR-v1.1/TCDecoder.ckpt")
DEFAULT_OUTPUT = Path("models/FlashVSR-v1.1-upscale")

COMPACT_DIT_NAME = (
    "diffusion_pytorch_model_streaming_dmd.compact-bf16.safetensors"
)
COMPACT_DECODER_NAME = "TCDecoder.compact-bf16.ckpt"
FORMAT_VERSION = 1
FIXED_TIMESTEP = 1000.0
DEFAULT_EPS = 1e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a pruned BF16 FlashVSR upscaling-only bundle."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--lq-projection", type=Path, default=DEFAULT_LQ)
    parser.add_argument("--tcdecoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--freq-dim", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _linear(
    value: torch.Tensor,
    source: Any,
    weight_key: str,
    bias_key: str,
    device: torch.device,
) -> torch.Tensor:
    weight = source.get_tensor(weight_key).to(device=device, dtype=torch.bfloat16)
    bias = source.get_tensor(bias_key).to(device=device, dtype=torch.bfloat16)
    result = F.linear(value, weight, bias)
    del weight, bias
    return result


def _sinusoidal_embedding(
    dim: int, position: torch.Tensor
) -> torch.Tensor:
    position_f32 = position.to(dtype=torch.float32)
    exponent = torch.arange(
        dim // 2, dtype=torch.float32, device=position.device
    ).div(dim // 2)
    sinusoid = torch.outer(position_f32, torch.pow(10000.0, -exponent))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(
        position.dtype
    )


def _rms_norm(
    value: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(
        value.float().pow(2).mean(dim=-1, keepdim=True) + eps
    )
    return normalized.to(value.dtype) * weight


def _precompute_constants(
    source: Any,
    prompt_path: Path,
    *,
    block_count: int,
    dim: int,
    freq_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
    if not isinstance(prompt, torch.Tensor) or prompt.ndim != 3:
        raise ValueError(f"Expected a [B,S,C] prompt tensor: {prompt_path}")
    context = prompt.to(device=device, dtype=torch.bfloat16)
    context = _linear(
        context,
        source,
        "text_embedding.0.weight",
        "text_embedding.0.bias",
        device,
    )
    context = F.gelu(context, approximate="tanh")
    context = _linear(
        context,
        source,
        "text_embedding.2.weight",
        "text_embedding.2.bias",
        device,
    )

    timestep = torch.tensor(
        [FIXED_TIMESTEP], device=device, dtype=torch.bfloat16
    )
    fixed_t = _sinusoidal_embedding(freq_dim, timestep)
    fixed_t = _linear(
        fixed_t,
        source,
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        device,
    )
    fixed_t = F.silu(fixed_t)
    fixed_t = _linear(
        fixed_t,
        source,
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        device,
    )
    fixed_t_mod = _linear(
        F.silu(fixed_t),
        source,
        "time_projection.1.weight",
        "time_projection.1.bias",
        device,
    ).unflatten(1, (6, dim))

    result = {
        "fixed_t": fixed_t.detach().to("cpu").contiguous(),
        "fixed_t_mod": fixed_t_mod.detach().to("cpu").contiguous(),
    }
    for block in range(block_count):
        prefix = f"blocks.{block}.cross_attn"
        cache_k = _linear(
            context,
            source,
            f"{prefix}.k.weight",
            f"{prefix}.k.bias",
            device,
        )
        norm_weight = source.get_tensor(f"{prefix}.norm_k.weight").to(
            device=device, dtype=torch.bfloat16
        )
        cache_k = _rms_norm(cache_k, norm_weight, DEFAULT_EPS)
        cache_v = _linear(
            context,
            source,
            f"{prefix}.v.weight",
            f"{prefix}.v.bias",
            device,
        )
        result[f"{prefix}.cache_k"] = (
            cache_k.detach().to("cpu").contiguous()
        )
        result[f"{prefix}.cache_v"] = (
            cache_v.detach().to("cpu").contiguous()
        )
        del cache_k, cache_v, norm_weight
    del context, prompt, fixed_t, fixed_t_mod
    return result


def _is_baked_key(key: str) -> bool:
    if key.startswith("text_embedding."):
        return True
    if key.startswith("time_embedding.") or key.startswith("time_projection."):
        return True
    return key.endswith(
        (
            ".cross_attn.k.weight",
            ".cross_attn.k.bias",
            ".cross_attn.v.weight",
            ".cross_attn.v.bias",
            ".cross_attn.norm_k.weight",
        )
    )


def _tensor_tree_to_bfloat16(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.bfloat16)
    if isinstance(value, dict):
        return {key: _tensor_tree_to_bfloat16(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_bfloat16(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_bfloat16(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_outputs(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace existing output: {names}")


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.checkpoint,
        args.prompt,
        args.lq_projection,
        args.tcdecoder,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = _resolve_device(args.device)
    output_dir = args.output_dir
    dit_output = output_dir / COMPACT_DIT_NAME
    decoder_output = output_dir / COMPACT_DECODER_NAME
    lq_output = output_dir / args.lq_projection.name
    manifest_output = output_dir / "upscale-bundle.json"
    outputs = [dit_output, decoder_output, lq_output, manifest_output]
    _check_outputs(outputs, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        block_indices = {
            int(key.split(".")[1])
            for key in keys
            if key.startswith("blocks.")
        }
        if block_indices != set(range(max(block_indices) + 1)):
            raise ValueError("DiT block indices are not contiguous")
        block_count = len(block_indices)
        patch_shape = source.get_slice("patch_embedding.weight").get_shape()
        dim = patch_shape[0]
        in_dim = patch_shape[1]
        patch_size = patch_shape[2:]
        ffn_dim = source.get_slice("blocks.0.ffn.0.weight").get_shape()[0]
        out_features = source.get_slice("head.head.weight").get_shape()[0]
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        if out_features % patch_volume:
            raise ValueError("Output channels do not match the patch volume")
        out_dim = out_features // patch_volume

        compact = _precompute_constants(
            source,
            args.prompt,
            block_count=block_count,
            dim=dim,
            freq_dim=args.freq_dim,
            device=device,
        )
        omitted_bytes = 0
        source_bytes = 0
        for key in keys:
            tensor = source.get_tensor(key)
            source_bytes += tensor.numel() * tensor.element_size()
            if _is_baked_key(key):
                omitted_bytes += tensor.numel() * tensor.element_size()
                continue
            compact[key] = tensor.to(dtype=torch.bfloat16).contiguous()

    context_length = compact["blocks.0.cross_attn.cache_k"].shape[1]
    compact.update(
        {
            "compact_format_version": torch.tensor(FORMAT_VERSION, dtype=torch.int32),
            "compact_num_heads": torch.tensor(args.num_heads, dtype=torch.int32),
            "compact_freq_dim": torch.tensor(args.freq_dim, dtype=torch.int32),
            "compact_context_length": torch.tensor(
                context_length, dtype=torch.int32
            ),
        }
    )
    metadata = {
        "format": "flashvsr-upscale-only",
        "format_version": str(FORMAT_VERSION),
        "dtype": "bfloat16",
        "fixed_prompt": args.prompt.name,
        "fixed_timestep": str(FIXED_TIMESTEP),
        "num_layers": str(block_count),
        "num_heads": str(args.num_heads),
        "dim": str(dim),
        "ffn_dim": str(ffn_dim),
        "in_dim": str(in_dim),
        "out_dim": str(out_dim),
        "context_length": str(context_length),
    }
    save_file(compact, dit_output, metadata=metadata)
    del compact

    decoder = torch.load(args.tcdecoder, map_location="cpu", weights_only=True)
    torch.save(_tensor_tree_to_bfloat16(decoder), decoder_output)
    del decoder
    shutil.copy2(args.lq_projection, lq_output)

    manifest = {
        "format_version": FORMAT_VERSION,
        "purpose": "FlashVSR-v1.1 tiny/tiny-long video upscaling only",
        "dtype": "bfloat16",
        "fixed_timestep": FIXED_TIMESTEP,
        "fixed_prompt_source": args.prompt.name,
        "architecture": {
            "dim": dim,
            "ffn_dim": ffn_dim,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "patch_size": patch_size,
            "num_heads": args.num_heads,
            "num_layers": block_count,
            "context_length": context_length,
        },
        "omitted": [
            "Wan2.1_VAE.pth (full mode only)",
            "text_embedding.* (fixed prompt baked into cross-attention caches)",
            "time_embedding.* (fixed one-step timestep baked)",
            "time_projection.* (fixed one-step modulation baked)",
            "blocks.*.cross_attn.k.* (fixed prompt baked)",
            "blocks.*.cross_attn.v.* (fixed prompt baked)",
            "blocks.*.cross_attn.norm_k.* (fixed prompt baked)",
            "models/posi_prompt.pth (fixed prompt baked)",
        ],
        "source_dit_bytes": source_bytes,
        "source_baked_parameter_bytes": omitted_bytes,
        "files": {},
    }
    for path in (dit_output, decoder_output, lq_output):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest["bundle_bytes"] = sum(
        entry["bytes"] for entry in manifest["files"].values()
    )
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    manifest = build_bundle(args)
    before = (
        args.checkpoint.stat().st_size
        + args.lq_projection.stat().st_size
        + args.tcdecoder.stat().st_size
        + Path("models/FlashVSR-v1.1/Wan2.1_VAE.pth").stat().st_size
        + args.prompt.stat().st_size
    )
    after = manifest["bundle_bytes"]
    print(f"FlashVSR upscaling bundle: {args.output_dir}")
    print(f"Before: {before / 2**30:.2f} GiB")
    print(f"After:  {after / 2**30:.2f} GiB")
    print(f"Saved:  {(before - after) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
