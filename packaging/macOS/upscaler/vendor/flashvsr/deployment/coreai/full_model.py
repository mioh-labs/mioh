"""Export-friendly pieces of the complete FlashVSR-v1.1 native pipeline.

The production graph is intentionally split at streaming-state boundaries:

* one causal LQ projection asset;
* thirty DiT block assets (two entry points each: first/next chunk);
* one shared patch/head asset;
* one causal TCDecoder asset.

All classes are inference-only and use real-valued RoPE.  The split lets the
Swift host retain each block's KV cache without Python or PyTorch at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .model import (
    CoreAIFirstChunkAttention,
    CoreAINextChunkAttention,
    RMSNorm,
)


COMPACT_CHECKPOINT = Path(
    "models/FlashVSR-v1.1-upscale/"
    "diffusion_pytorch_model_streaming_dmd.compact-bf16.safetensors"
)
LQ_CHECKPOINT = Path("models/FlashVSR-v1.1-upscale/LQ_proj_in.ckpt")
DECODER_CHECKPOINT = Path(
    "models/FlashVSR-v1.1-upscale/TCDecoder.compact-bf16.ckpt"
)


def _copy_parameter(target: torch.Tensor, value: torch.Tensor) -> None:
    with torch.no_grad():
        target.copy_(value.to(dtype=target.dtype))


def _rms_channel(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1) * (x.shape[1] ** 0.5) * gamma


def _replicate_pad_spatial_3d(x: torch.Tensor) -> torch.Tensor:
    """One-pixel H/W replicate pad without Core ML's rank-5 pad operator."""

    horizontal = torch.cat((x[..., :1], x, x[..., -1:]), dim=4)
    return torch.cat(
        (horizontal[:, :, :, :1], horizontal, horizontal[:, :, :, -1:]), dim=3
    )


def _unshuffle_weight(channels: int, factor: int, *, temporal: bool) -> torch.Tensor:
    """Fixed grouped-convolution weight equivalent to pixel unshuffle."""

    spatial = torch.eye(factor * factor).reshape(factor * factor, 1, factor, factor)
    weight = spatial.repeat(channels, 1, 1, 1)
    return weight.unsqueeze(2) if temporal else weight


def _pixel_unshuffle_16(video: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[B,3,F,H,W] -> [B,768,F,H/16,W/16]."""

    # A grouped 3-D convolution with a one-hot spatial kernel is exactly the
    # same rearrangement, but remains a native rank-5 ANE operation.  Calling
    # pixel_unshuffle causes MPSGraph to synthesize a rank-6 reshape.
    return F.conv3d(video, weight, stride=(1, 16, 16), groups=video.shape[1])


class CoreAILQWarmup(nn.Module):
    """Initialize causal LQ convolution caches from the first video frame."""

    def __init__(self, height: int = 256, width: int = 256) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.conv1 = nn.Conv3d(768, 2048, (4, 3, 3), stride=(2, 1, 1))
        self.register_buffer("norm1_gamma", torch.ones(2048, 1, 1, 1))
        self.register_buffer(
            "unshuffle_weight", _unshuffle_weight(3, 16, temporal=True), persistent=False
        )

    def forward(self, first_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shuffled_frame = _pixel_unshuffle_16(first_frame, self.unshuffle_weight)
        shuffled = torch.cat((shuffled_frame, shuffled_frame, shuffled_frame, shuffled_frame), dim=2)
        cache1 = shuffled[:, :, -2:]
        temporal = torch.cat((shuffled[:, :, :1], shuffled[:, :, :1], shuffled), dim=2)
        padded = _replicate_pad_spatial_3d(temporal)
        hidden = F.silu(_rms_channel(self.conv1(padded), self.norm1_gamma))
        return cache1, hidden[:, :, -2:]


class CoreAILQNext(nn.Module):
    """Process four high-resolution LQ frames and update causal caches."""

    def __init__(self, height: int = 256, width: int = 256) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.conv1 = nn.Conv3d(768, 2048, (4, 3, 3), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(2048, 3072, (4, 3, 3), stride=(2, 1, 1))
        self.linear = nn.Linear(3072, 1536)
        self.register_buffer("norm1_gamma", torch.ones(2048, 1, 1, 1))
        self.register_buffer("norm2_gamma", torch.ones(3072, 1, 1, 1))
        self.register_buffer(
            "unshuffle_weight", _unshuffle_weight(3, 16, temporal=True), persistent=False
        )

    def forward(
        self,
        frames: torch.Tensor,
        cache1: torch.Tensor,
        cache2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shuffled = _pixel_unshuffle_16(frames, self.unshuffle_weight)
        next_cache1 = shuffled[:, :, -2:]
        conv1_input = torch.cat((cache1, shuffled), dim=2)
        conv1_input = _replicate_pad_spatial_3d(conv1_input)
        hidden = F.silu(_rms_channel(self.conv1(conv1_input), self.norm1_gamma))
        next_cache2 = hidden[:, :, -2:]
        conv2_input = torch.cat((cache2, hidden), dim=2)
        conv2_input = _replicate_pad_spatial_3d(conv2_input)
        feature = F.silu(_rms_channel(self.conv2(conv2_input), self.norm2_gamma))
        tokens = feature.permute(0, 2, 3, 4, 1).reshape(1, -1, 3072)
        return self.linear(tokens), next_cache1, next_cache2


def load_lq_projection(
    warmup: CoreAILQWarmup,
    next_chunk: CoreAILQNext,
    checkpoint: str | Path = LQ_CHECKPOINT,
) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for model in (warmup, next_chunk):
        _copy_parameter(model.conv1.weight, state["conv1.weight"])
        _copy_parameter(model.conv1.bias, state["conv1.bias"])
        _copy_parameter(model.norm1_gamma, state["norm1.gamma"])
    _copy_parameter(next_chunk.conv2.weight, state["conv2.weight"])
    _copy_parameter(next_chunk.conv2.bias, state["conv2.bias"])
    _copy_parameter(next_chunk.norm2_gamma, state["norm2.gamma"])
    _copy_parameter(next_chunk.linear.weight, state["linear_layers.0.weight"])
    _copy_parameter(next_chunk.linear.bias, state["linear_layers.0.bias"])


class _CoreAIDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        *,
        frames: int,
        height: int,
        width: int,
        topk: int,
        kv_len: int,
        first_chunk: bool,
        inject_lq: bool,
        local_range: int = 9,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        attention_type = (
            CoreAIFirstChunkAttention if first_chunk else CoreAINextChunkAttention
        )
        self.self_attn = attention_type(
            dim,
            num_heads,
            frames=frames,
            height=height,
            width=width,
            topk=topk,
            kv_len=kv_len,
            local_range=local_range,
            eps=eps,
        )
        self.cross_q = nn.Linear(dim, dim)
        self.cross_o = nn.Linear(dim, dim)
        self.cross_norm_q = RMSNorm(dim, eps=eps)
        self.register_buffer("cross_k", torch.empty(1, 512, dim))
        self.register_buffer("cross_v", torch.empty(1, 512, dim))
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn1 = nn.Linear(dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, dim)
        self.inject_lq = inject_lq
        self.num_heads = num_heads
        self.eps = eps
        self.register_buffer("shift_msa", torch.empty(1, 1, dim))
        self.register_buffer("scale_msa", torch.empty(1, 1, dim))
        self.register_buffer("gate_msa", torch.empty(1, 1, dim))
        self.register_buffer("shift_mlp", torch.empty(1, 1, dim))
        self.register_buffer("scale_mlp", torch.empty(1, 1, dim))
        self.register_buffer("gate_mlp", torch.empty(1, 1, dim))

    def _layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), eps=self.eps)

    def _cross_attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self.cross_norm_q(self.cross_q(x))
        q = q.reshape(1, q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.cross_k.reshape(1, 512, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.cross_v.reshape(1, 512, self.num_heads, -1).permute(0, 2, 1, 3)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.permute(0, 2, 1, 3).reshape(1, x.shape[1], -1)
        return self.cross_o(attended)

    def _before_attention(
        self, x: torch.Tensor, lq: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.inject_lq:
            if lq is None:
                raise ValueError("block 0 requires LQ features")
            x = x + lq
        normalized = self._layer_norm(x)
        attention_input = normalized * (1.0 + self.scale_msa) + self.shift_msa
        return x, attention_input

    def _after_attention(
        self, x: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.gate_msa * attention_output
        x = x + self._cross_attention(self.norm3(x))
        normalized = self._layer_norm(x)
        mlp_input = normalized * (1.0 + self.scale_mlp) + self.shift_mlp
        mlp = self.ffn2(F.gelu(self.ffn1(mlp_input), approximate="tanh"))
        return x + self.gate_mlp * mlp

    def load_block_state(
        self,
        state: Mapping[str, torch.Tensor],
        fixed_t_mod: torch.Tensor,
    ) -> None:
        self.self_attn.load_attention_state(
            {
                name.removeprefix("self_attn."): value
                for name, value in state.items()
                if name.startswith("self_attn.")
            }
        )
        for target_name, source_name in (
            ("weight", "cross_attn.q.weight"),
            ("bias", "cross_attn.q.bias"),
        ):
            _copy_parameter(getattr(self.cross_q, target_name), state[source_name])
        for target_name, source_name in (
            ("weight", "cross_attn.o.weight"),
            ("bias", "cross_attn.o.bias"),
        ):
            _copy_parameter(getattr(self.cross_o, target_name), state[source_name])
        _copy_parameter(self.cross_norm_q.weight, state["cross_attn.norm_q.weight"])
        _copy_parameter(self.cross_k, state["cross_attn.cache_k"])
        _copy_parameter(self.cross_v, state["cross_attn.cache_v"])
        _copy_parameter(self.norm3.weight, state["norm3.weight"])
        _copy_parameter(self.norm3.bias, state["norm3.bias"])
        _copy_parameter(self.ffn1.weight, state["ffn.0.weight"])
        _copy_parameter(self.ffn1.bias, state["ffn.0.bias"])
        _copy_parameter(self.ffn2.weight, state["ffn.2.weight"])
        _copy_parameter(self.ffn2.bias, state["ffn.2.bias"])

        modulation = state["modulation"].to(torch.float32) + fixed_t_mod.float()
        chunks = modulation.chunk(6, dim=1)
        for name, value in zip(
            (
                "shift_msa",
                "scale_msa",
                "gate_msa",
                "shift_mlp",
                "scale_mlp",
                "gate_mlp",
            ),
            chunks,
            strict=True,
        ):
            _copy_parameter(getattr(self, name), value)


class CoreAIFirstChunkDiTBlock(_CoreAIDiTBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, first_chunk=True, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        lq: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, attention_input = self._before_attention(
            x, lq if self.inject_lq else None
        )
        attended, cache_k, cache_v = self.self_attn(
            attention_input, rope_cos, rope_sin
        )
        return self._after_attention(x, attended), cache_k, cache_v


class CoreAINextChunkDiTBlock(_CoreAIDiTBlock):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, first_chunk=False, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        lq: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, attention_input = self._before_attention(
            x, lq if self.inject_lq else None
        )
        attended, next_k, next_v = self.self_attn(
            attention_input, rope_cos, rope_sin, cache_k, cache_v
        )
        return self._after_attention(x, attended), next_k, next_v


def load_compact_block_pair(
    first: CoreAIFirstChunkDiTBlock,
    next_chunk: CoreAINextChunkDiTBlock,
    *,
    block: int,
    checkpoint: str | Path = COMPACT_CHECKPOINT,
) -> None:
    prefix = f"blocks.{block}."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        state = {
            key.removeprefix(prefix): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(prefix)
        }
        fixed_t_mod = handle.get_tensor("fixed_t_mod")
    first.load_block_state(state, fixed_t_mod)
    next_chunk.load_block_state(state, fixed_t_mod)


class CoreAIPatchEmbedding(nn.Module):
    def __init__(self, dim: int = 1536) -> None:
        super().__init__()
        self.projection = nn.Conv3d(16, dim, (1, 2, 2), stride=(1, 2, 2))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = self.projection(latent)
        return value.permute(0, 2, 3, 4, 1).reshape(1, -1, value.shape[1])


class CoreAIHead(nn.Module):
    def __init__(self, frames: int, height: int, width: int, dim: int = 1536) -> None:
        super().__init__()
        self.frames = frames
        self.height = height
        self.width = width
        self.eps = 1e-6
        self.linear = nn.Linear(dim, 64)
        self.register_buffer("shift", torch.empty(1, 1, dim))
        self.register_buffer("scale", torch.empty(1, 1, dim))
        # Fixed transposed-convolution weight for the checkpoint's
        # [sub_h, sub_w, C] channel convention.  This is pixel shuffle without
        # the rank-6 reshape synthesized by MPSGraph.
        shuffle_weight = torch.zeros(64, 16, 2, 2)
        for subpixel in range(4):
            for channel in range(16):
                shuffle_weight[subpixel * 16 + channel, channel, subpixel // 2, subpixel % 2] = 1
        self.register_buffer(
            "pixel_shuffle_weight",
            shuffle_weight,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(x, (x.shape[-1],), eps=self.eps)
        value = self.linear(normalized * (1.0 + self.scale) + self.shift)
        value = value.reshape(1, self.frames, self.height, self.width, 64)
        value = value.permute(0, 1, 4, 2, 3).reshape(
            self.frames, 64, self.height, self.width
        )
        value = F.conv_transpose2d(value, self.pixel_shuffle_weight, stride=2)
        return value.reshape(
            1, self.frames, 16, self.height * 2, self.width * 2
        ).permute(0, 2, 1, 3, 4)


def load_patch_and_heads(
    patch: CoreAIPatchEmbedding,
    first_head: CoreAIHead,
    next_head: CoreAIHead,
    checkpoint: str | Path = COMPACT_CHECKPOINT,
) -> None:
    names = {
        "patch_embedding.weight",
        "patch_embedding.bias",
        "head.head.weight",
        "head.head.bias",
        "head.modulation",
        "fixed_t",
    }
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(name) for name in names}
    _copy_parameter(patch.projection.weight, state["patch_embedding.weight"])
    _copy_parameter(patch.projection.bias, state["patch_embedding.bias"])
    modulation = state["head.modulation"].float() + state["fixed_t"].float().unsqueeze(1)
    shift, scale = modulation.chunk(2, dim=1)
    for head in (first_head, next_head):
        _copy_parameter(head.linear.weight, state["head.head.weight"])
        _copy_parameter(head.linear.bias, state["head.head.bias"])
        _copy_parameter(head.shift, shift)
        _copy_parameter(head.scale, scale)


def _condition_pixel_shuffle(
    condition: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Match TCDecoder.PixelShuffle3d(4, 8, 8) in NTCHW layout."""

    batch, channels, frames, height, width = condition.shape
    if frames != 4:
        raise ValueError(f"Core AI decoder step requires four condition frames, got {frames}")

    # Merging C and F preserves the decoder's [C, frame, spatial] ordering.
    # Grouped convolution applies the one-hot 8x8 spatial rearrangement to
    # every C/F plane without any tensor rank above five.
    planes = condition.reshape(batch, channels * frames, height, width)
    return F.conv2d(
        planes, weight, stride=8, groups=channels * frames
    ).unsqueeze(1)


class _CoreAITCDecoder(nn.Module):
    def __init__(self, decoder: nn.Sequential, *, trim_first: bool) -> None:
        super().__init__()
        self.decoder = decoder
        self.trim_first = trim_first
        self.register_buffer(
            "condition_unshuffle_weight",
            _unshuffle_weight(3 * 4, 8, temporal=False),
            persistent=False,
        )

    def _run(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        memories: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, ...]:
        frames = list(
            torch.cat(
                (_condition_pixel_shuffle(condition, self.condition_unshuffle_weight), latent),
                dim=2,
            ).unbind(1)
        )
        memory_inputs = list(memories) if memories is not None else []
        memory_index = 0
        next_memories: list[torch.Tensor] = []
        for layer in self.decoder:
            layer_name = type(layer).__name__
            if layer_name == "MemBlock":
                previous = (
                    memory_inputs[memory_index]
                    if memory_index < len(memory_inputs)
                    else torch.zeros_like(frames[0])
                )
                processed: list[torch.Tensor] = []
                for frame in frames:
                    processed.append(layer(frame, previous))
                    previous = frame
                frames = processed
                next_memories.append(previous)
                memory_index += 1
            elif layer_name == "TGrow":
                grown: list[torch.Tensor] = []
                stride = int(layer.stride)
                for frame in frames:
                    value = layer(frame)
                    grown.extend(value.chunk(stride, dim=0))
                frames = grown
            else:
                frames = [layer(frame) for frame in frames]
        video = torch.stack(frames, dim=1)
        if self.trim_first:
            video = video[:, 3:]
        return (video, *next_memories)


class CoreAIFirstTCDecoder(_CoreAITCDecoder):
    def __init__(self, decoder: nn.Sequential) -> None:
        super().__init__(decoder, trim_first=True)

    def forward(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return self._run(latent, condition, None)


class CoreAINextTCDecoder(_CoreAITCDecoder):
    def __init__(self, decoder: nn.Sequential) -> None:
        super().__init__(decoder, trim_first=False)

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        memory0: torch.Tensor,
        memory1: torch.Tensor,
        memory2: torch.Tensor,
        memory3: torch.Tensor,
        memory4: torch.Tensor,
        memory5: torch.Tensor,
        memory6: torch.Tensor,
        memory7: torch.Tensor,
        memory8: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self._run(
            latent,
            condition,
            (
                memory0,
                memory1,
                memory2,
                memory3,
                memory4,
                memory5,
                memory6,
                memory7,
                memory8,
            ),
        )


class CoreAITCDecoderStep(_CoreAITCDecoder):
    """Decode one latent frame to four video frames with explicit memory."""

    def __init__(self, decoder: nn.Sequential) -> None:
        super().__init__(decoder, trim_first=False)

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        memory0: torch.Tensor,
        memory1: torch.Tensor,
        memory2: torch.Tensor,
        memory3: torch.Tensor,
        memory4: torch.Tensor,
        memory5: torch.Tensor,
        memory6: torch.Tensor,
        memory7: torch.Tensor,
        memory8: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self._run(
            latent.unsqueeze(1),
            condition,
            (
                memory0,
                memory1,
                memory2,
                memory3,
                memory4,
                memory5,
                memory6,
                memory7,
                memory8,
            ),
        )


def build_tcdecoder_pair(
    checkpoint: str | Path = DECODER_CHECKPOINT,
) -> tuple[CoreAIFirstTCDecoder, CoreAINextTCDecoder]:
    # Load the decoder module directly. Importing ``src.models`` executes the
    # application's pipeline package and unnecessarily requires video-I/O
    # dependencies in the isolated conversion environment.
    decoder_path = Path(__file__).resolve().parents[2] / "src/models/TCDecoder.py"
    spec = importlib.util.spec_from_file_location(
        "_flashvsr_coreai_tcdecoder", decoder_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TCDecoder module: {decoder_path}")
    decoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder_module)
    build_tcdecoder = decoder_module.build_tcdecoder

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    def one() -> nn.Sequential:
        model = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            device="cpu",
            dtype=torch.float32,
            new_latent_channels=16 + 768,
        )
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"Unexpected compact TCDecoder keys: {result.unexpected_keys}"
            )
        return model.decoder.eval()

    return CoreAIFirstTCDecoder(one()), CoreAINextTCDecoder(one())


def build_tcdecoder_step(
    checkpoint: str | Path = DECODER_CHECKPOINT,
) -> CoreAITCDecoderStep:
    # Reuse the audited pair builder and retain only one copy of the weights.
    first, _ = build_tcdecoder_pair(checkpoint)
    return CoreAITCDecoderStep(first.decoder)
