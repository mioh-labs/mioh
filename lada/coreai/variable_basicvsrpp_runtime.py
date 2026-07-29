# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Persistent transport for the variable-length Swift BasicVSR++ runner."""

from __future__ import annotations

import json
import mmap
import os
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np


IMAGE_SIZE = 256
CHANNELS = 3
STOP_COMMAND = 0xFFFF


class VariableBasicVSRPPRuntime:
    """Runs the eleven six-frame BasicVSR++ assets in one Swift process.

    The Swift side owns reusable Metal buffers for frames, flows and propagated
    features. Only the input and final restored frames cross the mmap boundary;
    intermediate tensors stay in shared GPU/Core AI memory. Recurrent steps are
    dispatched in contiguous six-frame chunks to avoid per-frame call overhead.
    """

    def __init__(
        self,
        models_path: Path,
        runner_path: str | None = None,
        process_factory: Callable[..., subprocess.Popen] | None = None,
    ):
        self.models_path = Path(models_path)
        if not self.models_path.is_dir():
            raise FileNotFoundError(self.models_path)
        self.runner_path = runner_path
        self._process_factory = process_factory or subprocess.Popen
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._mapping: mmap.mmap | None = None
        self._shared_file: BinaryIO | None = None
        self._shared_path: Path | None = None
        self._descriptor_path: Path | None = None
        self._maximum_frames = 0
        self._sequence_bytes = 0

    def _resolve_runner(self) -> str:
        runner = (
            self.runner_path
            or os.environ.get("LADA_VARIABLE_COREAI_SWIFT_RUNNER")
            or shutil.which("lada-basicvsrpp-variable-runner")
        )
        if not runner:
            raise RuntimeError(
                "variable Core AI BasicVSR++ requires "
                "lada-basicvsrpp-variable-runner"
            )
        return runner

    @staticmethod
    def _validate_frames(frames: np.ndarray) -> int:
        if frames.dtype != np.float16:
            raise ValueError("variable Core AI BasicVSR++ requires FP16 input")
        if frames.ndim != 5 or tuple(frames.shape[:1] + frames.shape[2:]) != (
            1,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                "variable Core AI BasicVSR++ requires shape "
                f"(1,T,3,256,256); got {tuple(frames.shape)}"
            )
        frame_count = int(frames.shape[1])
        if not 1 <= frame_count < STOP_COMMAND:
            raise ValueError("variable Core AI frame count must be 1..65534")
        if not frames.flags.c_contiguous:
            raise ValueError("variable Core AI input must be contiguous")
        return frame_count

    @staticmethod
    def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = byte_count
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError("variable Core AI Swift runner closed its output")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _start(self, maximum_frames: int) -> None:
        runner = self._resolve_runner()
        frame_bytes = CHANNELS * IMAGE_SIZE * IMAGE_SIZE * np.dtype(np.float16).itemsize
        sequence_bytes = maximum_frames * frame_bytes
        descriptor_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="lada-variable-basicvsrpp-", suffix=".json", delete=False
        )
        shared_file = tempfile.NamedTemporaryFile(
            prefix="lada-variable-basicvsrpp-", suffix=".bin", delete=False
        )
        descriptor_path = Path(descriptor_file.name)
        shared_path = Path(shared_file.name)
        mapping: mmap.mmap | None = None
        try:
            json.dump(
                {
                    "maximumFrames": maximum_frames,
                    "inputOffset": 0,
                    "outputOffset": sequence_bytes,
                    "byteCount": sequence_bytes * 2,
                },
                descriptor_file,
                separators=(",", ":"),
            )
            descriptor_file.flush()
            descriptor_file.close()
            shared_file.truncate(sequence_bytes * 2)
            shared_file.flush()
            mapping = mmap.mmap(shared_file.fileno(), sequence_bytes * 2)
            process = self._process_factory(
                [
                    runner,
                    str(self.models_path.resolve()),
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
        self._process = process
        self._mapping = mapping
        self._shared_file = shared_file
        self._shared_path = shared_path
        self._descriptor_path = descriptor_path
        self._maximum_frames = maximum_frames
        self._sequence_bytes = sequence_bytes

    def _stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(struct.pack("<H", STOP_COMMAND))
                process.stdin.flush()
                process.wait(timeout=5)
            except Exception:
                process.terminate()
                process.wait(timeout=5)
        if self._mapping is not None:
            self._mapping.close()
        if self._shared_file is not None:
            self._shared_file.close()
        for path in (self._descriptor_path, self._shared_path):
            if path is not None:
                path.unlink(missing_ok=True)
        self._process = None
        self._mapping = None
        self._shared_file = None
        self._shared_path = None
        self._descriptor_path = None
        self._maximum_frames = 0
        self._sequence_bytes = 0

    def _ensure_started(self, frame_count: int) -> None:
        if self._process is not None and frame_count <= self._maximum_frames:
            return
        self._stop()
        self._start(frame_count)

    def infer(self, frames: np.ndarray) -> np.ndarray:
        with self._lock:
            frame_count = self._validate_frames(frames)
            self._ensure_started(frame_count)
            assert self._mapping is not None and self._process is not None
            assert self._process.stdin is not None and self._process.stdout is not None
            self._mapping.seek(0)
            # ``frames`` is already validated as contiguous. Passing its
            # buffer directly avoids a second whole-sequence bytes allocation
            # for every temporal chunk.
            self._mapping.write(memoryview(frames).cast("B"))
            self._process.stdin.write(struct.pack("<H", frame_count))
            self._process.stdin.flush()
            response = self._read_exact(self._process.stdout, 1)
            if response != b"\x00":
                raise RuntimeError("variable Core AI runner returned an invalid response")
            # Copy once into the returned array instead of mmap.read() bytes
            # followed by a second NumPy copy.
            return np.ndarray(
                shape=frames.shape,
                dtype=np.float16,
                buffer=self._mapping,
                offset=self._sequence_bytes,
            ).copy()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
