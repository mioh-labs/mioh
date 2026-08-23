"""Export-friendly FlashVSR streaming self-attention for Apple runtimes.

The normal PyTorch model uses complex tensors for RoPE and a device-specific
sparse attention kernel.  Current Core AI conversion does not lower that
complex RoPE graph.  This module keeps the same weights, window ordering,
dynamic top-k block mask, and cache policy while expressing RoPE with real
cosine/sine tensors and using dense SDPA as the portable correctness backend.

The dense backend is deliberately a first-stage implementation.  Production
resolution needs a Core AI custom Metal implementation of the existing
128-by-64 block-sparse kernel to avoid quadratic attention work.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


_WINDOW = (2, 8, 8)
_QUERY_BLOCK = 128
_KEY_BLOCK = 64


class RMSNorm(nn.Module):
    """Numerically equivalent to the RMSNorm used by the PyTorch model."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight


def _axis_angles(
    dim: int,
    length: int,
    *,
    offset: int = 0,
    theta: float = 10000.0,
) -> torch.Tensor:
    exponent = torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim
    inv_freq = 1.0 / torch.pow(theta, exponent)
    positions = torch.arange(offset, offset + length, dtype=torch.float32)
    return torch.outer(positions, inv_freq)


def build_rope_cos_sin(
    head_dim: int,
    frames: int,
    height: int,
    width: int,
    *,
    temporal_offset: int = 0,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build real-valued 3-D RoPE inputs shaped ``[tokens, 1, head_dim/2]``."""

    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    spatial_dim = head_dim // 3
    temporal_dim = head_dim - 2 * spatial_dim
    pair_count = temporal_dim // 2 + 2 * (spatial_dim // 2)
    if pair_count != head_dim // 2:
        raise ValueError(
            "head_dim is incompatible with FlashVSR's 3-D RoPE split; "
            f"got {head_dim}, which produces {pair_count * 2} rotated channels"
        )
    f_angles = _axis_angles(
        temporal_dim,
        frames,
        offset=temporal_offset,
        theta=theta,
    )
    h_angles = _axis_angles(spatial_dim, height, theta=theta)
    w_angles = _axis_angles(spatial_dim, width, theta=theta)

    angles = torch.cat(
        [
            f_angles[:, None, None, :].expand(frames, height, width, -1),
            h_angles[None, :, None, :].expand(frames, height, width, -1),
            w_angles[None, None, :, :].expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(frames * height * width, 1, pair_count)
    return torch.cos(angles), torch.sin(angles)


def apply_real_rope(
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Apply RoPE without complex dtypes so Core AI can lower the graph."""

    batch, tokens, channels = x.shape
    head_dim = channels // num_heads
    pairs = x.float().reshape(batch, tokens, num_heads, head_dim // 2, 2)
    real = pairs[..., 0]
    imag = pairs[..., 1]
    cos = rope_cos.to(dtype=torch.float32)
    sin = rope_sin.to(dtype=torch.float32)
    rotated = torch.stack(
        (real * cos - imag * sin, real * sin + imag * cos),
        dim=-1,
    )
    return rotated.flatten(start_dim=2).to(x.dtype)


def _partition_windows(x: torch.Tensor) -> torch.Tensor:
    batch, frames, height, width, channels = x.shape
    wf, wh, ww = _WINDOW
    x = x.reshape(
        batch,
        frames // wf,
        wf,
        height // wh,
        wh,
        width // ww,
        ww,
        channels,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.reshape(-1, wf * wh * ww, channels)


def _reverse_windows(
    windows: torch.Tensor,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    wf, wh, ww = _WINDOW
    nf, nh, nw = frames // wf, height // wh, width // ww
    batch = windows.shape[0] // (nf * nh * nw)
    x = windows.reshape(batch, nf, nh, nw, wf, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.reshape(batch, frames, height, width, -1)


def _build_local_mask(
    block_height: int,
    block_width: int,
    local_range: int,
) -> torch.Tensor:
    rows = torch.arange(block_height)
    columns = torch.arange(block_width)
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    all_rows = yy.reshape(-1)
    all_columns = xx.reshape(-1)
    row_start = all_rows - local_range // 2
    row_end = row_start + local_range - 1
    column_start = all_columns - local_range // 2
    column_end = column_start + local_range - 1
    in_row = (all_rows[None, :] >= row_start[:, None]) & (
        all_rows[None, :] <= row_end[:, None]
    )
    in_column = (all_columns[None, :] >= column_start[:, None]) & (
        all_columns[None, :] <= column_end[:, None]
    )
    return in_row & in_column


def _dynamic_block_mask(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    local_mask: torch.Tensor,
    *,
    num_heads: int,
    query_time_windows: int,
    topk: int,
) -> torch.Tensor:
    """Reproduce ``generate_draft_block_mask_sage`` without einops."""

    # Selection is deliberately FP32 even when the projections/attention run
    # in FP16.  Near-equal block scores otherwise cross the top-k threshold on
    # Apple accelerators and change the sparse pattern rather than merely
    # introducing normal FP16 rounding error.
    q_mean = q_windows.float().mean(dim=1)
    q_heads = q_mean.reshape(q_mean.shape[0], num_heads, -1).permute(1, 0, 2)

    k_split = k_windows.reshape(k_windows.shape[0], 2, 64, k_windows.shape[2])
    k_mean = k_split.float().mean(dim=2).reshape(-1, k_windows.shape[2])
    k_heads = k_mean.reshape(k_mean.shape[0], num_heads, -1).permute(1, 0, 2)
    k_first, k_second = torch.chunk(k_heads, 2, dim=1)

    scale = math.sqrt(q_heads.shape[-1])
    scores_first = torch.einsum("hld,hmd->hlm", q_heads, k_first) / scale
    scores_second = torch.einsum("hld,hmd->hlm", q_heads, k_second) / scale
    scores = torch.cat((scores_first, scores_second), dim=-1)

    repeat_length = scores.shape[1] // local_mask.shape[0]
    repeat_count = (scores.shape[2] // 2) // local_mask.shape[1]
    allowed = local_mask.unsqueeze(1).unsqueeze(0)
    allowed = allowed.repeat(repeat_length, 1, repeat_count, 1)
    allowed = allowed.reshape(scores.shape[1], scores.shape[2] // 2)
    allowed = allowed.repeat_interleave(2, dim=1)
    allowed = allowed.unsqueeze(0).repeat(num_heads, 1, 1)
    scores = scores.masked_fill(~allowed, -float("inf"))

    attention_map = torch.softmax(scores, dim=-1)
    head_time = num_heads * query_time_windows
    rows_per_time = attention_map.shape[1] // query_time_windows
    attention_map = attention_map.reshape(
        num_heads,
        query_time_windows,
        rows_per_time,
        attention_map.shape[2],
    ).reshape(head_time, rows_per_time, attention_map.shape[2])
    flat = attention_map.reshape(head_time, -1)
    apply_topk = min(flat.shape[1] - 1, topk)
    if apply_topk <= 0:
        selected = torch.zeros_like(flat, dtype=torch.bool)
    else:
        threshold = torch.topk(
            flat,
            k=apply_topk + 1,
            dim=1,
            largest=True,
        ).values[:, -1:]
        selected = flat > threshold
    selected = selected.reshape(head_time, rows_per_time, attention_map.shape[2])
    return selected.reshape(
        num_heads,
        query_time_windows * rows_per_time,
        attention_map.shape[2],
    ).unsqueeze(0)


class _CoreAIStreamingSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        frames: int,
        height: int,
        width: int,
        topk: int,
        kv_len: int,
        local_range: int = 9,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if frames % 2 or height % 8 or width % 8:
            raise ValueError("frames/height/width must divide the (2, 8, 8) window")
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("frames, height, and width must be positive")
        if topk < 0 or kv_len <= 0:
            raise ValueError("topk must be non-negative and kv_len must be positive")

        self.dim = dim
        self.num_heads = num_heads
        self.frames = frames
        self.height = height
        self.width = width
        self.topk = topk
        self.kv_len = kv_len

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.register_buffer(
            "local_mask",
            _build_local_mask(height // 8, width // 8, local_range),
            persistent=False,
        )

    @property
    def token_count(self) -> int:
        return self.frames * self.height * self.width

    @property
    def spatial_window_count(self) -> int:
        return (self.height // 8) * (self.width // 8)

    def load_attention_state(self, state: Mapping[str, torch.Tensor]) -> None:
        result = self.load_state_dict(dict(state), strict=False)
        allowed_missing = {"local_mask"}
        missing = set(result.missing_keys) - allowed_missing
        if missing or result.unexpected_keys:
            raise ValueError(
                f"Invalid self-attention state: missing={sorted(missing)}, "
                f"unexpected={sorted(result.unexpected_keys)}"
            )

    def _run(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        pre_cache_k: torch.Tensor | None,
        pre_cache_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, channels = x.shape
        if batch != 1:
            raise ValueError("Core AI prototype currently supports batch size 1")

        q = apply_real_rope(
            self.norm_q(self.q(x)), rope_cos, rope_sin, self.num_heads
        )
        k = apply_real_rope(
            self.norm_k(self.k(x)), rope_cos, rope_sin, self.num_heads
        )
        v = self.v(x)

        q_windows = _partition_windows(
            q.reshape(batch, self.frames, self.height, self.width, channels)
        )
        k_windows = _partition_windows(
            k.reshape(batch, self.frames, self.height, self.width, channels)
        )
        v_windows = _partition_windows(
            v.reshape(batch, self.frames, self.height, self.width, channels)
        )
        if pre_cache_k is not None and pre_cache_v is not None:
            k_windows = torch.cat((pre_cache_k, k_windows), dim=0)
            v_windows = torch.cat((pre_cache_v, v_windows), dim=0)

        query_window_count = q_windows.shape[0]
        key_window_count = k_windows.shape[0]
        q_tokens = q_windows.reshape(1, query_window_count * _QUERY_BLOCK, channels)
        k_tokens = k_windows.reshape(1, key_window_count * _QUERY_BLOCK, channels)
        v_tokens = v_windows.reshape(1, key_window_count * _QUERY_BLOCK, channels)

        block_mask = _dynamic_block_mask(
            q_windows,
            k_windows,
            self.local_mask,
            num_heads=self.num_heads,
            query_time_windows=self.frames // 2,
            topk=self.topk,
        )
        dense_mask = block_mask.repeat_interleave(_QUERY_BLOCK, dim=-2)
        dense_mask = dense_mask.repeat_interleave(_KEY_BLOCK, dim=-1)

        q_heads = q_tokens.reshape(
            1, q_tokens.shape[1], self.num_heads, -1
        ).permute(0, 2, 1, 3)
        k_heads = k_tokens.reshape(
            1, k_tokens.shape[1], self.num_heads, -1
        ).permute(0, 2, 1, 3)
        v_heads = v_tokens.reshape(
            1, v_tokens.shape[1], self.num_heads, -1
        ).permute(0, 2, 1, 3)
        attended = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=dense_mask,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(
            1, q_tokens.shape[1], channels
        )

        attended_windows = attended.reshape(
            query_window_count, _QUERY_BLOCK, channels
        )
        restored = _reverse_windows(
            attended_windows,
            self.frames,
            self.height,
            self.width,
        ).reshape(1, tokens, channels)

        max_cache_windows = self.kv_len * self.spatial_window_count
        if k_windows.shape[0] > max_cache_windows:
            cache_k = k_windows[self.spatial_window_count :]
            cache_v = v_windows[self.spatial_window_count :]
        else:
            cache_k = k_windows
            cache_v = v_windows
        return self.o(restored), cache_k, cache_v


class CoreAIFirstChunkAttention(_CoreAIStreamingSelfAttention):
    """Core AI ``first_chunk`` entry point (creates the initial KV cache)."""

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._run(x, rope_cos, rope_sin, None, None)


class CoreAINextChunkAttention(_CoreAIStreamingSelfAttention):
    """Core AI ``next_chunk`` entry point (consumes and updates the KV cache)."""

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._run(x, rope_cos, rope_sin, cache_k, cache_v)


def load_flashvsr_attention_weights(
    checkpoint: str | Path,
    *,
    block: int = 0,
) -> tuple[int, dict[str, torch.Tensor]]:
    """Load one DiT self-attention layer without materializing the 5.3 GiB file."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    prefix = f"blocks.{block}.self_attn."
    wanted = {
        "q.weight",
        "q.bias",
        "k.weight",
        "k.bias",
        "v.weight",
        "v.bias",
        "o.weight",
        "o.bias",
        "norm_q.weight",
        "norm_k.weight",
    }
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for name in wanted:
            key = prefix + name
            if key not in available:
                raise KeyError(f"Checkpoint is missing {key!r}")
            state[name] = handle.get_tensor(key)
    dim = state["q.weight"].shape[0]
    return dim, state
