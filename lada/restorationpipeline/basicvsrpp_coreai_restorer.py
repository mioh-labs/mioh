# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Fixed-shape Core AI runtime adapter for BasicVSR++ restoration."""

from __future__ import annotations

import asyncio
import mmap
import os
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import psutil
import torch

from lada.restorationpipeline.basicvsrpp_mosaic_restorer import (
    BasicvsrppMosaicRestorer,
)

DEFAULT_COREAI_FRAME_COUNT = 18
SUPPORTED_COREAI_FRAME_COUNTS = (18, 36, 90)
DEFAULT_COREAI_MAX_INFLIGHT = 2


class CoreAIModelRuntime:
    def __init__(
        self,
        model_path: Path,
        frame_count: int = DEFAULT_COREAI_FRAME_COUNT,
        max_inflight: int = DEFAULT_COREAI_MAX_INFLIGHT,
    ):
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        if frame_count not in SUPPORTED_COREAI_FRAME_COUNTS:
            raise ValueError(f"unsupported Core AI frame count: {frame_count}")
        if max_inflight <= 0:
            raise ValueError("Core AI max_inflight must be positive")
        self.model_path = model_path
        self.frame_count = frame_count
        self._backend = "swift" if model_path.suffix == ".aimodelc" else "python"
        self.max_inflight = (
            1
            if self._backend == "swift" or frame_count >= 90
            else max_inflight
        )
        self._transport = (
            "shared-memory" if self._backend == "swift" else "in-process"
        )
        self._runner: asyncio.Runner | None = None
        self._model = None
        self._function = None
        self._process: subprocess.Popen | None = None
        self._shared_memory_file = None
        self._shared_memory: mmap.mmap | None = None
        self._shared_memory_path: Path | None = None
        self._slot_size = frame_count * 3 * 256 * 256 * 2
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._backend == "swift":
            self._ensure_swift_runner()
            return
        if self._function is not None:
            return
        try:
            from coreai.runtime import AIModel
        except ImportError as exc:
            raise RuntimeError(
                "Core AI BasicVSR++ requires the isolated coreai-torch environment"
            ) from exc

        self._runner = asyncio.Runner()
        self._model = self._runner.run(AIModel.load(self.model_path))
        self._function = self._model.load_function("main")

    def _ensure_swift_runner(self) -> None:
        if self._process is not None:
            return
        runner_path = os.environ.get("LADA_COREAI_SWIFT_RUNNER") or shutil.which(
            "lada-coreai-runner"
        )
        if not runner_path:
            raise RuntimeError(
                "compiled Core AI models require the lada-coreai-runner executable"
            )
        shared_file = tempfile.NamedTemporaryFile(
            prefix="lada-coreai-", suffix=".bin", delete=False
        )
        shared_file.truncate(self._slot_size * self.max_inflight)
        shared_file.flush()
        shared_memory = mmap.mmap(
            shared_file.fileno(), self._slot_size * self.max_inflight
        )
        shared_path = Path(shared_file.name)
        try:
            process = subprocess.Popen(
                [
                    runner_path,
                    str(self.model_path.resolve()),
                    str(self.frame_count),
                    str(shared_path),
                    str(self.max_inflight),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            shared_memory.close()
            shared_file.close()
            shared_path.unlink(missing_ok=True)
            raise
        self._shared_memory_file = shared_file
        self._shared_memory = shared_memory
        self._shared_memory_path = shared_path
        self._process = process

    @staticmethod
    def _read_exact(stream, byte_count: int) -> bytes:
        chunks = []
        remaining = byte_count
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError("Core AI Swift runner closed its output")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _infer_swift_many(
        self, inputs: Iterable[torch.Tensor]
    ) -> list[torch.Tensor]:
        assert self._process is not None
        assert self._process.stdin is not None and self._process.stdout is not None
        assert self._shared_memory is not None
        available_slots = deque(range(self.max_inflight))
        pending: dict[int, tuple[int, tuple[int, ...], int]] = {}
        output_tensors: list[torch.Tensor | None] = []

        def submit(input_tensor: torch.Tensor, index: int) -> None:
            self._validate_input(input_tensor)
            input_array = input_tensor.detach().cpu().contiguous().numpy()
            payload = input_array.tobytes()
            if len(payload) > self._slot_size:
                raise ValueError(
                    f"Core AI input requires {len(payload)} bytes; "
                    f"shared-memory slot has {self._slot_size}"
                )
            slot = available_slots.popleft()
            offset = slot * self._slot_size
            self._shared_memory.seek(offset)
            self._shared_memory.write(payload)
            self._process.stdin.write(bytes((slot,)))
            pending[slot] = (index, tuple(input_array.shape), len(payload))

        def receive() -> None:
            try:
                slot = self._read_exact(self._process.stdout, 1)[0]
            except (EOFError, OSError) as exc:
                error = ""
                if (
                    self._process.stderr is not None
                    and self._process.poll() is not None
                ):
                    error = self._process.stderr.read().decode(
                        "utf-8", errors="replace"
                    )
                raise RuntimeError(error.strip() or str(exc)) from exc
            if slot == 254:
                self._process.terminate()
                self._process.wait(timeout=5)
                error = "Core AI Swift runner failed"
                if self._process.stderr is not None:
                    detail = self._process.stderr.read().decode(
                        "utf-8", errors="replace"
                    ).strip()
                    if detail:
                        error = detail
                raise RuntimeError(error)
            if slot not in pending:
                raise RuntimeError(f"unexpected Core AI completion slot: {slot}")
            index, shape, byte_count = pending.pop(slot)
            self._shared_memory.seek(slot * self._slot_size)
            output = np.frombuffer(
                self._shared_memory.read(byte_count), dtype=np.float16
            ).reshape(shape)
            output_tensors[index] = torch.from_numpy(output.copy())
            available_slots.append(slot)

        for input_tensor in inputs:
            if not available_slots:
                self._process.stdin.flush()
                receive()
            index = len(output_tensors)
            output_tensors.append(None)
            submit(input_tensor, index)
        self._process.stdin.flush()
        while pending:
            receive()

        if any(output is None for output in output_tensors):
            raise RuntimeError("Core AI Swift runner did not return every output")
        return [output for output in output_tensors if output is not None]

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if inputs.shape[1] != self.frame_count:
            raise ValueError(
                f"Core AI BasicVSR++ requires exactly {self.frame_count} frames"
            )
        if inputs.dtype != torch.float16:
            raise ValueError("Core AI BasicVSR++ requires FP16 input")

    def _effective_max_inflight(self) -> int:
        memory = psutil.virtual_memory()
        if memory.total and memory.available / memory.total < 0.25:
            return 1
        return self.max_inflight

    def infer_many(self, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        with self._lock:
            self._ensure_loaded()
            if self._backend == "swift":
                return self._infer_swift_many(inputs)
            assert self._runner is not None and self._function is not None
            from coreai.runtime import NDArray
            effective_max_inflight = self._effective_max_inflight()

            async def infer_one(input_tensor: torch.Tensor) -> np.ndarray:
                input_array = input_tensor.detach().cpu().contiguous().numpy()
                outputs = await self._function({"frames": NDArray(input_array)})
                output = outputs["restored"].numpy().copy()
                if output.shape != tuple(input_tensor.shape):
                    raise ValueError(
                        f"unexpected Core AI output shape {output.shape}; "
                        f"expected {tuple(input_tensor.shape)}"
                    )
                return output

            async def infer_stream() -> list[np.ndarray]:
                pending: deque[asyncio.Task[np.ndarray]] = deque()
                completed: list[np.ndarray] = []
                try:
                    for input_tensor in inputs:
                        self._validate_input(input_tensor)
                        pending.append(asyncio.create_task(infer_one(input_tensor)))
                        if len(pending) >= effective_max_inflight:
                            completed.append(await pending.popleft())
                    while pending:
                        completed.append(await pending.popleft())
                finally:
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                return completed

            output_arrays = self._runner.run(infer_stream())

        return [torch.from_numpy(output) for output in output_arrays]

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.infer_many((inputs,))[0]

    def close(self) -> None:
        with self._lock:
            runner = self._runner
            process = self._process
            shared_memory = self._shared_memory
            shared_file = self._shared_memory_file
            shared_path = self._shared_memory_path
            self._runner = None
            self._model = None
            self._function = None
            self._process = None
            self._shared_memory = None
            self._shared_memory_file = None
            if runner is not None:
                runner.close()
            if process is not None:
                try:
                    if process.stdin is not None:
                        process.stdin.write(b"\xff")
                        process.stdin.flush()
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
            if shared_memory is not None:
                shared_memory.close()
            if shared_file is not None:
                shared_file.close()
            if shared_path is not None:
                shared_path.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FixedCoreAIModelAdapter:
    def __init__(
        self,
        runtime: Callable[[torch.Tensor], torch.Tensor],
        frame_count: int = DEFAULT_COREAI_FRAME_COUNT,
    ):
        if frame_count not in SUPPORTED_COREAI_FRAME_COUNTS:
            raise ValueError(f"unsupported Core AI frame count: {frame_count}")
        self.runtime = runtime
        self.frame_count = frame_count

    def _pad_input(self, inputs: torch.Tensor) -> tuple[torch.Tensor, int]:
        frame_count = inputs.shape[1]
        if frame_count <= 0 or frame_count > self.frame_count:
            raise ValueError(
                f"Core AI BasicVSR++ chunk must contain 1..{self.frame_count} frames"
            )

        if frame_count < self.frame_count:
            padding = inputs[:, -1:].expand(
                -1,
                self.frame_count - frame_count,
                -1,
                -1,
                -1,
            )
            inputs = torch.cat((inputs, padding), dim=1)

        return inputs, frame_count

    def infer_many(self, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        padded_inputs: list[torch.Tensor] = []
        frame_counts: list[int] = []
        for item in inputs:
            padded, frame_count = self._pad_input(item)
            padded_inputs.append(padded)
            frame_counts.append(frame_count)

        runtime_infer_many = getattr(self.runtime, "infer_many", None)
        if callable(runtime_infer_many):
            outputs = runtime_infer_many(padded_inputs)
        else:
            outputs = [self.runtime(item) for item in padded_inputs]

        if len(outputs) != len(padded_inputs):
            raise ValueError(
                f"Core AI returned {len(outputs)} outputs for {len(padded_inputs)} inputs"
            )
        result = []
        for output, padded, frame_count in zip(
            outputs, padded_inputs, frame_counts, strict=True
        ):
            if output.shape != padded.shape:
                raise ValueError(
                    f"unexpected Core AI output shape {tuple(output.shape)}; "
                    f"expected {tuple(padded.shape)}"
                )
            result.append(output[:, :frame_count])
        return result

    def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return self.infer_many((inputs,))[0]


class CoreAIBasicvsrppMosaicRestorer(BasicvsrppMosaicRestorer):
    def __init__(
        self,
        model_path: Path,
        frame_count: int = DEFAULT_COREAI_FRAME_COUNT,
        runtime: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        fixed_model = FixedCoreAIModelAdapter(
            runtime or CoreAIModelRuntime(model_path, frame_count),
            frame_count,
        )
        super().__init__(fixed_model, torch.device("cpu"), fp16=True)
        self.model_path = model_path
        self.frame_count = frame_count

    def restore(self, video, max_frames=-1):
        del max_frames
        return self._restore_unlocked(video, max_frames=self.frame_count)

    def _run_model_chunks(self, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.model.infer_many(chunks)

    def close(self) -> None:
        close_runtime = getattr(self.model.runtime, "close", None)
        if callable(close_runtime):
            close_runtime()


FixedT18ModelAdapter = FixedCoreAIModelAdapter
