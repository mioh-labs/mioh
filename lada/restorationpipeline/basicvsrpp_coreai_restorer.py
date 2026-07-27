# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Fixed-shape Core AI runtime adapter for BasicVSR++ restoration."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import psutil
import torch

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec
from lada.coreai.source_runtime import load_source_model
from lada.coreai.variable_basicvsrpp_runtime import VariableBasicVSRPPRuntime
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
        self._compiled_runtime: CompiledCoreAIRuntime | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._backend == "swift":
            if self._compiled_runtime is None:
                shape = (1, self.frame_count, 3, 256, 256)
                self._compiled_runtime = CompiledCoreAIRuntime(
                    self.model_path,
                    inputs=(TensorSpec("frames", shape),),
                    outputs=(TensorSpec("restored", shape),),
                )
            return
        if self._function is not None:
            return
        self._runner = asyncio.Runner()
        self._model = load_source_model(
            self._runner,
            self.model_path,
            purpose="BasicVSR++",
        )
        self._function = self._model.load_function("main")

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
                assert self._compiled_runtime is not None
                result = []
                for input_tensor in inputs:
                    self._validate_input(input_tensor)
                    input_array = (
                        input_tensor.detach().cpu().contiguous().numpy()
                    )
                    output = self._compiled_runtime.infer(
                        {"frames": input_array}
                    )["restored"]
                    if output.shape != input_array.shape:
                        raise ValueError(
                            f"unexpected Core AI output shape {output.shape}; "
                            f"expected {input_array.shape}"
                        )
                    result.append(torch.from_numpy(output.copy()))
                return result
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
            compiled_runtime = self._compiled_runtime
            self._runner = None
            self._model = None
            self._function = None
            self._compiled_runtime = None
            if runner is not None:
                runner.close()
            if compiled_runtime is not None:
                compiled_runtime.close()

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

    def restore(
        self,
        video,
        max_frames=-1,
        temporal_overlap: int = 8,
        enable_crossfade: bool = True,
    ):
        del max_frames
        return self._restore_unlocked(
            video,
            max_frames=self.frame_count,
            temporal_overlap=temporal_overlap,
            enable_crossfade=enable_crossfade,
        )

    def _run_model_chunks(self, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.model.infer_many(chunks)

    def close(self) -> None:
        close_runtime = getattr(self.model.runtime, "close", None)
        if callable(close_runtime):
            close_runtime()


FixedT18ModelAdapter = FixedCoreAIModelAdapter


class VariableCoreAIModelRuntime:
    """Torch-facing adapter for the six-frame-chunk Swift variable runner."""

    def __init__(
        self,
        models_path: Path,
        runner_path: str | None = None,
        runtime_factory: Callable[..., VariableBasicVSRPPRuntime] = (
            VariableBasicVSRPPRuntime
        ),
    ):
        self.models_path = Path(models_path)
        if runner_path is None:
            self._runtime = runtime_factory(self.models_path)
        else:
            self._runtime = runtime_factory(
                self.models_path,
                runner_path=runner_path,
            )
        self._lock = threading.Lock()

    @staticmethod
    def _validate_input(inputs: torch.Tensor) -> None:
        if inputs.ndim != 5 or tuple(inputs.shape[:1] + inputs.shape[2:]) != (
            1,
            3,
            256,
            256,
        ):
            raise ValueError(
                "variable Core AI BasicVSR++ requires shape (1,T,3,256,256)"
            )
        if inputs.dtype != torch.float16:
            raise ValueError("variable Core AI BasicVSR++ requires FP16 input")

    def infer_many(self, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        with self._lock:
            outputs: list[torch.Tensor] = []
            for input_tensor in inputs:
                self._validate_input(input_tensor)
                input_array = input_tensor.detach().cpu().contiguous().numpy()
                output = self._runtime.infer(input_array)
                if output.shape != input_array.shape:
                    raise ValueError(
                        f"unexpected variable Core AI output shape {output.shape}; "
                        f"expected {input_array.shape}"
                    )
                outputs.append(torch.from_numpy(output))
            return outputs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.infer_many((inputs,))[0]

    def close(self) -> None:
        with self._lock:
            self._runtime.close()


class VariableCoreAIModelAdapter:
    def __init__(self, runtime: VariableCoreAIModelRuntime):
        self.runtime = runtime

    def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
        return self.runtime(inputs)

    def infer_many(self, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        return self.runtime.infer_many(inputs)


class CoreAIVariableBasicvsrppMosaicRestorer(BasicvsrppMosaicRestorer):
    """Variable-length BasicVSR++ without fixed-T padding."""

    def __init__(
        self,
        models_path: Path,
        runner_path: str | None = None,
        runtime: VariableCoreAIModelRuntime | None = None,
    ):
        adapter = VariableCoreAIModelAdapter(
            runtime or VariableCoreAIModelRuntime(models_path, runner_path)
        )
        super().__init__(adapter, torch.device("cpu"), fp16=True)
        self.models_path = Path(models_path)

    def _run_model_chunks(self, chunks: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.model.infer_many(chunks)

    def close(self) -> None:
        self.model.runtime.close()
