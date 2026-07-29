# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Persistent shared-memory transport for compiled Core AI models."""

from __future__ import annotations

import json
import math
import mmap
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Core AI tensor name must not be empty")
        if not self.shape or any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError(f"invalid Core AI tensor shape for {self.name}: {self.shape}")

    @property
    def byte_count(self) -> int:
        count = math.prod(self.shape) * np.dtype(np.float16).itemsize
        if count <= 0 or count > (1 << 63) - 1:
            raise ValueError(f"Core AI tensor is too large: {self.name}")
        return count


class CompiledCoreAIRuntime:
    """Runs one fixed-shape compiled model through the Swift Core AI helper."""

    def __init__(
        self,
        model_path: Path,
        inputs: tuple[TensorSpec, ...],
        outputs: tuple[TensorSpec, ...],
        runner_path: str | None = None,
        process_factory: Callable[..., subprocess.Popen] | None = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        if not inputs or not outputs:
            raise ValueError("compiled Core AI models require inputs and outputs")
        names = [item.name for item in (*inputs, *outputs)]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tensor name in compiled Core AI contract")
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.runner_path = runner_path
        self._process_factory = process_factory or subprocess.Popen
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._mapping: mmap.mmap | None = None
        self._shared_file: BinaryIO | None = None
        self._shared_path: Path | None = None
        self._descriptor_path: Path | None = None
        self._layouts, self._slot_stride = self._build_layouts()

    def _build_layouts(self) -> tuple[dict[str, dict[str, object]], int]:
        layouts: dict[str, dict[str, object]] = {}
        offset = 0
        for spec in (*self.inputs, *self.outputs):
            byte_count = spec.byte_count
            layouts[spec.name] = {
                "name": spec.name,
                "shape": list(spec.shape),
                "offset": offset,
                "byteCount": byte_count,
            }
            offset += byte_count
            if offset > (1 << 63) - 1:
                raise ValueError("compiled Core AI shared-memory slot is too large")
        return layouts, offset

    @property
    def descriptor(self) -> dict[str, object]:
        return {
            "function": "main",
            "slotCount": 1,
            "slotStride": self._slot_stride,
            "inputs": [dict(self._layouts[item.name]) for item in self.inputs],
            "outputs": [dict(self._layouts[item.name]) for item in self.outputs],
        }

    @property
    def shared_path(self) -> Path:
        if self._shared_path is None:
            raise RuntimeError("compiled Core AI runtime has not started")
        return self._shared_path

    @property
    def descriptor_path(self) -> Path:
        if self._descriptor_path is None:
            raise RuntimeError("compiled Core AI runtime has not started")
        return self._descriptor_path

    def _resolve_runner(self) -> str:
        runner = (
            self.runner_path
            or os.environ.get("LADA_COREAI_SWIFT_RUNNER")
            or shutil.which("lada-coreai-runner")
        )
        if not runner:
            raise RuntimeError(
                "compiled Core AI models require the lada-coreai-runner executable"
            )
        return runner

    def _ensure_started(self) -> None:
        if self._process is not None:
            return
        runner = self._resolve_runner()
        descriptor_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="lada-coreai-", suffix=".json", delete=False
        )
        descriptor_path = Path(descriptor_file.name)
        shared_file = tempfile.NamedTemporaryFile(
            prefix="lada-coreai-", suffix=".bin", delete=False
        )
        shared_path = Path(shared_file.name)
        mapping: mmap.mmap | None = None
        try:
            json.dump(self.descriptor, descriptor_file, separators=(",", ":"))
            descriptor_file.flush()
            descriptor_file.close()
            shared_file.truncate(self._slot_stride)
            shared_file.flush()
            mapping = mmap.mmap(shared_file.fileno(), self._slot_stride)
            process = self._process_factory(
                [
                    runner,
                    str(self.model_path.resolve()),
                    str(descriptor_path),
                    str(shared_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            if mapping is not None:
                mapping.close()
            if not descriptor_file.closed:
                descriptor_file.close()
            shared_file.close()
            descriptor_path.unlink(missing_ok=True)
            shared_path.unlink(missing_ok=True)
            raise
        self._descriptor_path = descriptor_path
        self._shared_file = shared_file
        self._shared_path = shared_path
        self._mapping = mapping
        self._process = process

    @staticmethod
    def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = byte_count
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError("Core AI Swift runner closed its output")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _validate_inputs(self, values: Mapping[str, np.ndarray]) -> None:
        expected = {item.name for item in self.inputs}
        received = set(values)
        if received != expected:
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(
                f"Core AI inputs do not match contract; missing={missing}, extra={extra}"
            )
        for spec in self.inputs:
            value = values[spec.name]
            if value.dtype != np.float16:
                raise ValueError(f"Core AI tensor {spec.name} requires FP16 input")
            if tuple(value.shape) != spec.shape:
                raise ValueError(
                    f"Core AI tensor {spec.name} requires shape {spec.shape}; "
                    f"got {tuple(value.shape)}"
                )
            if not value.flags.c_contiguous:
                raise ValueError(f"Core AI tensor {spec.name} must be contiguous")

    def _write_tensor(self, name: str, value: np.ndarray) -> None:
        if self._mapping is None:
            raise RuntimeError("compiled Core AI runtime has not started")
        layout = self._layouts[name]
        payload = memoryview(value).cast("B")
        if payload.nbytes != layout["byteCount"]:
            raise ValueError(f"Core AI tensor {name} byte count does not match contract")
        self._mapping.seek(int(layout["offset"]))
        self._mapping.write(payload)

    def _read_outputs(self) -> dict[str, np.ndarray]:
        if self._mapping is None:
            raise RuntimeError("compiled Core AI runtime has not started")
        result: dict[str, np.ndarray] = {}
        for spec in self.outputs:
            layout = self._layouts[spec.name]
            # Read directly from the persistent mmap and make only the one
            # owning copy returned to the caller. mmap.read() would first
            # allocate an equally large bytes object, doubling every Core AI
            # output at the host boundary.
            result[spec.name] = np.ndarray(
                shape=spec.shape,
                dtype=np.float16,
                buffer=self._mapping,
                offset=int(layout["offset"]),
            ).copy()
        return result

    def infer(self, values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._lock:
            self._ensure_started()
            self._validate_inputs(values)
            for spec in self.inputs:
                self._write_tensor(spec.name, values[spec.name])
            assert self._process is not None
            assert self._process.stdin is not None and self._process.stdout is not None
            self._process.stdin.write(b"\x00")
            self._process.stdin.flush()
            try:
                response = self._read_exact(self._process.stdout, 1)[0]
            except (EOFError, OSError) as exc:
                raise RuntimeError(self._runner_error(str(exc))) from exc
            if response == 254:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                raise RuntimeError(self._runner_error("Core AI Swift runner failed"))
            if response != 0:
                raise RuntimeError(f"unexpected Core AI completion slot: {response}")
            return self._read_outputs()

    def _runner_error(self, fallback: str) -> str:
        if self._process is None or self._process.stderr is None:
            return fallback
        if self._process.poll() is None:
            return fallback
        detail = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        return detail or fallback

    def close(self) -> None:
        with self._lock:
            process = self._process
            mapping = self._mapping
            shared_file = self._shared_file
            shared_path = self._shared_path
            descriptor_path = self._descriptor_path
            self._process = None
            self._mapping = None
            self._shared_file = None
            self._shared_path = None
            self._descriptor_path = None
            if process is not None:
                try:
                    if process.stdin is not None and process.poll() is None:
                        process.stdin.write(b"\xff")
                        process.stdin.flush()
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
            if mapping is not None:
                mapping.close()
            if shared_file is not None:
                shared_file.close()
            if shared_path is not None:
                shared_path.unlink(missing_ok=True)
            if descriptor_path is not None:
                descriptor_path.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
