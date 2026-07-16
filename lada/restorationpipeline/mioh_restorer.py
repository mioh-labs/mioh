# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Runtime adapters for the stateful MiohRestorerV1 prototype."""

from __future__ import annotations

import asyncio
import re
import threading
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch

from lada.coreai import CompiledCoreAIRuntime, TensorSpec
from lada.coreai.source_runtime import load_source_model
from lada.models.mioh_restorer import MiohRestorerV1
from lada.utils import ImageTensor


_CONTRACT_PATTERN = re.compile(r"-t(?P<frames>\d+)-c(?P<channels>\d+)-s(?P<size>\d+)(?:-|\.)")


def infer_mioh_contract(model_path: Path) -> tuple[int, int, int]:
    match = _CONTRACT_PATTERN.search(Path(model_path).name.lower())
    if match is None:
        return (
            MiohRestorerV1.DEFAULT_CHUNK_FRAMES,
            MiohRestorerV1.DEFAULT_CHANNELS,
            256,
        )
    return (
        int(match.group("frames")),
        int(match.group("channels")),
        int(match.group("size")),
    )


class ChunkRuntime(Protocol):
    chunk_frames: int
    channels: int
    image_size: int
    dtype: torch.dtype

    def __call__(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class TorchMiohRestorerRuntime:
    def __init__(
        self,
        model: MiohRestorerV1,
        device: torch.device | str = "cpu",
        *,
        image_size: int = 256,
    ) -> None:
        if image_size <= 0 or image_size % MiohRestorerV1.DOWNSCALE:
            raise ValueError("image_size must be positive and divisible by 4")
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.chunk_frames = model.chunk_frames
        self.channels = model.channels
        self.image_size = image_size
        self.dtype = next(model.parameters()).dtype

    def __call__(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            return self.model(
                frames.to(device=self.device, dtype=self.dtype),
                masks.to(device=self.device, dtype=self.dtype),
                state.to(device=self.device, dtype=self.dtype),
            )


def load_torch_mioh_restorer_runtime(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
    fp16: bool,
) -> TorchMiohRestorerRuntime:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    model = MiohRestorerV1(
        chunk_frames=int(config.get("chunk_frames", MiohRestorerV1.DEFAULT_CHUNK_FRAMES)),
        channels=int(config.get("channels", MiohRestorerV1.DEFAULT_CHANNELS)),
        num_blocks=int(config.get("num_blocks", MiohRestorerV1.DEFAULT_BLOCKS)),
    )
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    if fp16:
        model.half()
    return TorchMiohRestorerRuntime(
        model,
        device,
        image_size=int(config.get("image_size", 256)),
    )


class CoreAIMiohRestorerRuntime:
    def __init__(
        self,
        model_path: Path,
        *,
        chunk_frames: int | None = None,
        channels: int | None = None,
        image_size: int | None = None,
        runner_path: str | None = None,
        runtime_factory: Callable[..., CompiledCoreAIRuntime] = CompiledCoreAIRuntime,
    ) -> None:
        inferred_frames, inferred_channels, inferred_size = infer_mioh_contract(
            model_path
        )
        self.chunk_frames = chunk_frames or inferred_frames
        self.channels = channels or inferred_channels
        self.image_size = image_size or inferred_size
        self.dtype = torch.float16
        self.model_path = Path(model_path)
        self._compiled = self.model_path.suffix == ".aimodelc"
        self._lock = threading.Lock()
        self._runner: asyncio.Runner | None = None
        self._source_model = None
        self._source_function = None
        state_shape = (
            1,
            self.channels,
            self.image_size // 4,
            self.image_size // 4,
        )
        frame_shape = (
            1,
            self.chunk_frames,
            3,
            self.image_size,
            self.image_size,
        )
        mask_shape = (
            1,
            self.chunk_frames,
            1,
            self.image_size,
            self.image_size,
        )
        self._frame_shape = frame_shape
        self._state_shape = state_shape
        self.runtime = (
            runtime_factory(
                self.model_path,
                inputs=(
                    TensorSpec("frames", frame_shape),
                    TensorSpec("masks", mask_shape),
                    TensorSpec("history", state_shape),
                ),
                outputs=(
                    TensorSpec("restored", frame_shape),
                    TensorSpec("next_state", state_shape),
                ),
                runner_path=runner_path,
            )
            if self._compiled
            else None
        )

    def _ensure_source_loaded(self) -> None:
        if self._source_function is not None:
            return
        self._runner = asyncio.Runner()
        self._source_model = load_source_model(
            self._runner,
            self.model_path,
            purpose="MiohRestorer",
        )
        self._source_function = self._source_model.load_function("main")

    def __call__(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = {
            "frames": np.ascontiguousarray(
                frames.detach().cpu().numpy(), dtype=np.float16
            ),
            "masks": np.ascontiguousarray(
                masks.detach().cpu().numpy(), dtype=np.float16
            ),
            "history": np.ascontiguousarray(
                state.detach().cpu().numpy(), dtype=np.float16
            ),
        }
        with self._lock:
            if self.runtime is not None:
                result = self.runtime.infer(values)
            else:
                self._ensure_source_loaded()
                assert self._runner is not None and self._source_function is not None
                from coreai.runtime import NDArray

                async def infer_source() -> dict[str, np.ndarray]:
                    outputs = await self._source_function(
                        {name: NDArray(value) for name, value in values.items()}
                    )
                    return {
                        name: output.numpy().copy() for name, output in outputs.items()
                    }

                result = self._runner.run(infer_source())
        if tuple(result["restored"].shape) != self._frame_shape:
            raise ValueError(
                f"unexpected restored shape {result['restored'].shape}; "
                f"expected {self._frame_shape}"
            )
        if tuple(result["next_state"].shape) != self._state_shape:
            raise ValueError(
                f"unexpected next_state shape {result['next_state'].shape}; "
                f"expected {self._state_shape}"
            )
        return torch.from_numpy(result["restored"]), torch.from_numpy(result["next_state"])

    def close(self) -> None:
        with self._lock:
            if self.runtime is not None:
                self.runtime.close()
            if self._runner is not None:
                self._runner.close()
            self._runner = None
            self._source_model = None
            self._source_function = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CoreMLMiohRestorerRuntime:
    def __init__(
        self,
        model_path: Path,
        *,
        chunk_frames: int | None = None,
        channels: int | None = None,
        image_size: int | None = None,
        compute_units=None,
    ) -> None:
        import coremltools as ct

        inferred_frames, inferred_channels, inferred_size = infer_mioh_contract(
            model_path
        )
        self.chunk_frames = chunk_frames or inferred_frames
        self.channels = channels or inferred_channels
        self.image_size = image_size or inferred_size
        self.dtype = torch.float16
        unit = compute_units or ct.ComputeUnit.ALL
        path = Path(model_path)
        if path.suffix == ".mlmodelc":
            self.model = ct.models.CompiledMLModel(str(path), compute_units=unit)
        else:
            self.model = ct.models.MLModel(str(path), compute_units=unit)

    def __call__(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.model.predict(
            {
                "frames": np.ascontiguousarray(frames.detach().cpu().numpy(), dtype=np.float16),
                "masks": np.ascontiguousarray(masks.detach().cpu().numpy(), dtype=np.float16),
                "history": np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.float16),
            }
        )
        return (
            torch.from_numpy(np.asarray(result["restored"])),
            torch.from_numpy(np.asarray(result["next_state"])),
        )


class MiohMosaicRestorer:
    """Clip adapter with streaming and bidirectional offline modes."""

    def __init__(self, runtime: ChunkRuntime) -> None:
        self.runtime = runtime
        self.dtype = runtime.dtype

    @staticmethod
    def _prepare_frames(video: list[ImageTensor]) -> torch.Tensor:
        if not video:
            raise ValueError("video must contain at least one frame")
        frames = torch.stack(
            [torch.as_tensor(frame).permute(2, 0, 1) for frame in video], dim=0
        ).unsqueeze(0)
        return frames.to(dtype=torch.float32).div_(255.0)

    @staticmethod
    def _prepare_masks(
        masks: list[ImageTensor] | None,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        if masks is None:
            return torch.ones(
                (frames.shape[0], frames.shape[1], 1, frames.shape[-2], frames.shape[-1]),
                dtype=frames.dtype,
            )
        if len(masks) != frames.shape[1]:
            raise ValueError(
                f"mask count ({len(masks)}) must match frame count ({frames.shape[1]})"
            )
        prepared = []
        for mask in masks:
            item = torch.as_tensor(mask)
            if item.ndim == 3:
                item = item[..., :1]
            elif item.ndim == 2:
                item = item.unsqueeze(-1)
            else:
                raise ValueError("masks must be HxW or HxWxC")
            item = item.permute(2, 0, 1).to(dtype=torch.float32)
            if item.numel() and item.max().item() > 1.0:
                item.div_(255.0)
            if item.shape[-2:] != frames.shape[-2:]:
                raise ValueError(
                    "mask dimensions must match the corresponding frame dimensions"
                )
            prepared.append(item)
        return torch.stack(prepared, dim=0).unsqueeze(0)

    def _run_direction(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if frames.shape[-2:] != (self.runtime.image_size, self.runtime.image_size):
            raise ValueError(f"MiohRestorerV1 requires {self.runtime.image_size}x{self.runtime.image_size} ROI frames")
        state = torch.zeros(
            (
                1,
                self.runtime.channels,
                self.runtime.image_size // 4,
                self.runtime.image_size // 4,
            ),
            dtype=self.runtime.dtype,
        )
        outputs = []
        for start in range(0, frames.shape[1], self.runtime.chunk_frames):
            end = min(frames.shape[1], start + self.runtime.chunk_frames)
            chunk = frames[:, start:end]
            mask_chunk = masks[:, start:end]
            valid_frames = end - start
            if valid_frames < self.runtime.chunk_frames:
                padding = self.runtime.chunk_frames - valid_frames
                chunk = torch.cat(
                    (chunk, torch.zeros_like(chunk[:, :1]).expand(-1, padding, -1, -1, -1)),
                    dim=1,
                )
                mask_chunk = torch.cat(
                    (
                        mask_chunk,
                        torch.zeros_like(mask_chunk[:, :1]).expand(-1, padding, -1, -1, -1),
                    ),
                    dim=1,
                )
            restored, state = self.runtime(chunk, mask_chunk, state)
            outputs.append(restored[:, :valid_frames].cpu())
            state = state.detach().cpu()
        return torch.cat(outputs, dim=1)

    def restore(
        self,
        video: list[ImageTensor],
        masks: list[ImageTensor] | None = None,
        *,
        bidirectional: bool = False,
    ) -> list[ImageTensor]:
        frames = self._prepare_frames(video)
        prepared_masks = self._prepare_masks(masks, frames)
        forward = self._run_direction(frames, prepared_masks)
        if bidirectional:
            backward = self._run_direction(
                torch.flip(frames, dims=(1,)),
                torch.flip(prepared_masks, dims=(1,)),
            )
            forward = (forward + torch.flip(backward, dims=(1,))) * 0.5
        output = (
            forward.squeeze(0)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
        )
        return list(output.unbind(0))
