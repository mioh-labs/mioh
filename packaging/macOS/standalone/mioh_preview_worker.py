#!/usr/bin/env python3
"""Persistent restored-preview worker used by the standalone mioh app."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, TextIO

import av
import numpy as np


PROTOCOL_STREAM = sys.stdout


def isolate_process_group() -> None:
    """Make the preview worker the owner of its Core AI child process tree."""
    try:
        if os.getpgrp() != os.getpid():
            os.setpgid(0, 0)
    except OSError:
        # Sandboxed/test runners may not permit changing the process group.
        pass


def emit_event(
    kind: str,
    *,
    generation: int,
    stream: TextIO | None = None,
    **payload,
) -> None:
    destination = stream or PROTOCOL_STREAM
    destination.write(
        json.dumps(
            {"kind": kind, "generation": generation, **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    destination.flush()


class PreviewCommand:
    def __init__(
        self,
        command: str,
        seconds: float | None = None,
        sequence: int | None = None,
    ) -> None:
        self.command = command
        self.seconds = seconds
        self.sequence = sequence

    @classmethod
    def parse(cls, line: str) -> "PreviewCommand":
        payload = json.loads(line)
        command = str(payload["command"])
        if command not in {"release_through", "set_buffer_limit", "stop"}:
            raise ValueError(f"Unsupported preview command: {command}")
        seconds = payload.get("seconds")
        sequence = payload.get("sequence")
        if command == "release_through" and sequence is None:
            raise ValueError("release_through requires sequence")
        return cls(
            command,
            float(seconds) if seconds is not None else None,
            int(sequence) if sequence is not None else None,
        )


class SegmentEncoder:
    def __init__(
        self,
        *,
        output_dir: Path,
        width: int,
        height: int,
        fps: Fraction,
        generation: int,
        preferred_codec: str = "h264_videotoolbox",
        segment_seconds: float = 2.0,
        stream_factory: Callable | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = Fraction(fps)
        self.generation = generation
        self.preferred_codec = preferred_codec
        self.segment_ns = int(segment_seconds * 1_000_000_000)
        self.stream_factory = stream_factory or (
            lambda container, codec, rate: container.add_stream(codec, rate=rate)
        )
        self.sequence = 0
        self.container = None
        self.video_stream = None
        self.segment_path: Path | None = None
        self.segment_final_path: Path | None = None
        self.segment_start_ns: int | None = None
        self.last_pts_ns: int | None = None
        self.frames_in_segment = 0
        self.frame_duration_ns = int(
            Fraction(1_000_000_000 * self.fps.denominator, self.fps.numerator)
        )
        self.active_codec = preferred_codec
        self.active_options: dict[str, str] = {}

    def _new_path(self) -> Path:
        return self.output_dir / (
            f"preview-g{self.generation}-{self.sequence:06d}.mp4"
        )

    def _open_with_codec(self, codec: str, options: dict[str, str]) -> None:
        final_path = self._new_path()
        working_path = Path(f"{final_path}.part")
        working_path.unlink(missing_ok=True)
        container = av.open(str(working_path), mode="w", format="mp4")
        try:
            stream = self.stream_factory(container, codec, self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(self.fps.denominator, self.fps.numerator)
            stream.options = options
        except Exception:
            container.close()
            working_path.unlink(missing_ok=True)
            raise
        self.container = container
        self.video_stream = stream
        self.segment_path = working_path
        self.segment_final_path = final_path
        self.active_codec = codec
        self.active_options = options

    def _open_segment(self, start_ns: int) -> None:
        self.segment_start_ns = start_ns
        self.frames_in_segment = 0
        try:
            options = {"realtime": "1"} if self.preferred_codec == "h264_videotoolbox" else {}
            if self.preferred_codec == "libx264":
                options = {"preset": "ultrafast", "tune": "zerolatency"}
            self._open_with_codec(self.preferred_codec, options)
        except Exception as exc:
            if self.preferred_codec == "libx264":
                raise
            print(
                f"VideoToolbox preview encoder unavailable, using libx264: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._open_with_codec(
                "libx264", {"preset": "ultrafast", "tune": "zerolatency"}
            )

    def add_frame(self, frame: np.ndarray, pts_ns: int) -> list[dict]:
        events = []
        if self.segment_start_ns is not None and pts_ns >= self.segment_start_ns + self.segment_ns:
            events.append(self._close_segment(self.segment_start_ns + self.segment_ns))
        if self.container is None:
            self._open_segment(pts_ns)

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self.frames_in_segment
        video_frame.time_base = Fraction(self.fps.denominator, self.fps.numerator)
        for packet in self.video_stream.encode(video_frame):
            self.container.mux(packet)
        self.frames_in_segment += 1
        self.last_pts_ns = pts_ns
        return events

    def _close_segment(self, end_ns: int) -> dict:
        if self.segment_path is None or self.segment_final_path is None:
            raise RuntimeError("cannot close a segment that is not open")
        working_path = self.segment_path
        final_path = self.segment_final_path
        for packet in self.video_stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        # Only completed, playable files use the .mp4 suffix. The rename is
        # atomic on the same volume, so capacity accounting and the Swift
        # player can never observe a half-written segment.
        working_path.replace(final_path)
        event = {
            "kind": "segment",
            "generation": self.generation,
            "sequence": self.sequence,
            "start_ns": self.segment_start_ns,
            "end_ns": end_ns,
            "path": str(final_path),
            "codec": self.active_codec,
        }
        self.container = None
        self.video_stream = None
        self.segment_path = None
        self.segment_final_path = None
        self.segment_start_ns = None
        self.frames_in_segment = 0
        self.sequence += 1
        return event

    def finish(self) -> list[dict]:
        if self.container is None or self.last_pts_ns is None:
            return []
        return [self._close_segment(self.last_pts_ns + self.frame_duration_ns)]

    def discard(self) -> None:
        if self.container is not None:
            self.container.close()
        if self.segment_path is not None:
            self.segment_path.unlink(missing_ok=True)
        self.container = None
        self.video_stream = None
        self.segment_path = None
        self.segment_final_path = None


class SwiftVideoToolboxSegmentEncoder:
    """Shared-memory bridge to the CVPixelBufferPool-backed Swift encoder."""

    def __init__(
        self,
        *,
        output_dir: Path,
        width: int,
        height: int,
        fps: Fraction,
        generation: int,
        preferred_codec: str = "h264_videotoolbox",
        segment_seconds: float = 2.0,
        runner_path: str | None = None,
        process_factory: Callable = subprocess.Popen,
    ) -> None:
        if preferred_codec != "h264_videotoolbox":
            raise ValueError("Swift preview encoder supports h264_videotoolbox only")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = Fraction(fps)
        self.generation = int(generation)
        self.sequence = 0
        self.segment_path: Path | None = None
        self.active_codec = "h264_videotoolbox"
        self.active_options = {"pixel_buffer_pool": "1"}
        self._frame_bytes = self.width * self.height * 3
        self._mapping_bytes = self._frame_bytes + 8
        self._shared_path = (
            self.output_dir / f".preview-frame-g{self.generation}.bin"
        )
        self._shared_file = self._shared_path.open("w+b")
        self._shared_file.truncate(self._mapping_bytes)
        self._shared_file.flush()
        self._mapping = mmap.mmap(self._shared_file.fileno(), self._mapping_bytes)
        runner = (
            runner_path
            or os.environ.get("LADA_PREVIEW_VIDEOTOOLBOX_RUNNER")
            or shutil.which("mioh-preview-videotoolbox-encoder")
        )
        if not runner:
            self._cleanup_transport()
            raise RuntimeError("Swift VideoToolbox preview encoder is unavailable")
        try:
            self._process = process_factory(
                [
                    runner,
                    str(self._shared_path),
                    str(self.output_dir),
                    str(self.width),
                    str(self.height),
                    str(self.fps.numerator),
                    str(self.fps.denominator),
                    str(self.generation),
                    str(float(segment_seconds)),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            self._cleanup_transport()
            raise

    def _runner_error(self, fallback: str) -> str:
        process = getattr(self, "_process", None)
        if process is not None and process.stderr is not None and process.poll() is not None:
            try:
                detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                if detail:
                    return detail
            except Exception:
                pass
        return fallback

    def _request(self, command: int) -> dict:
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(
                self._runner_error("Swift VideoToolbox preview encoder exited")
            )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(bytes((command,)))
        process.stdin.flush()
        response = process.stdout.readline()
        if not response:
            raise RuntimeError(
                self._runner_error("Swift VideoToolbox preview encoder disconnected")
            )
        payload = json.loads(response)
        if payload.get("status") != "ok":
            raise RuntimeError(f"Unexpected preview encoder response: {payload}")
        active_path = payload.get("active_path")
        self.segment_path = Path(active_path) if active_path else None
        return payload

    def _segment_event(self, payload: dict) -> list[dict]:
        value = payload.get("segment")
        if value is None:
            return []
        self.sequence = int(value["sequence"]) + 1
        return [{
            "kind": "segment",
            "generation": self.generation,
            "sequence": int(value["sequence"]),
            "start_ns": int(value["start_ns"]),
            "end_ns": int(value["end_ns"]),
            "path": str(value["path"]),
            "codec": str(value["codec"]),
        }]

    def add_frame(self, frame: np.ndarray, pts_ns: int) -> list[dict]:
        if frame.dtype != np.uint8 or frame.shape != (
            self.height,
            self.width,
            3,
        ):
            raise ValueError(
                "preview frame must be uint8 BGR with shape "
                f"({self.height}, {self.width}, 3)"
            )
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        self._mapping.seek(0)
        self._mapping.write(memoryview(frame).cast("B"))
        struct.pack_into("<q", self._mapping, self._frame_bytes, int(pts_ns))
        return self._segment_event(self._request(0))

    def finish(self) -> list[dict]:
        if self._process.poll() is not None:
            self._cleanup_transport()
            return []
        try:
            events = self._segment_event(self._request(1))
            self._process.wait(timeout=10)
            return events
        finally:
            self._cleanup_transport()

    def discard(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                self._request(2)
                process.wait(timeout=2)
            except Exception:
                process.terminate()
        self._cleanup_transport()

    def _cleanup_transport(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            try:
                mapping.close()
            except Exception:
                pass
            self._mapping = None
        shared_file = getattr(self, "_shared_file", None)
        if shared_file is not None:
            try:
                shared_file.close()
            except Exception:
                pass
            self._shared_file = None
        shared_path = getattr(self, "_shared_path", None)
        if shared_path is not None:
            Path(shared_path).unlink(missing_ok=True)


_ASYNC_ENCODER_STOP = object()


class AsyncSegmentEncoder:
    """Overlap Swift color conversion/VideoToolbox append with restoration.

    Only two frames may wait outside the underlying encoder. This is enough to
    overlap the stages without turning an encoder slowdown into an unbounded
    full-resolution frame queue.
    """

    def __init__(
        self,
        encoder,
        *,
        max_pending_frames: int = 2,
        startup_fallback_factory: Callable | None = None,
    ) -> None:
        self.encoder = encoder
        self.active_codec = encoder.active_codec
        self.active_options = dict(encoder.active_options)
        self._startup_fallback_factory = startup_fallback_factory
        self._completed_frames = 0
        self._tasks: queue.Queue = queue.Queue(
            maxsize=max(1, int(max_pending_frames))
        )
        self._events: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="mioh-preview-encoder",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            try:
                if task is _ASYNC_ENCODER_STOP:
                    return
                frame, pts_ns = task
                try:
                    events = self.encoder.add_frame(frame, pts_ns)
                except BaseException as primary_error:
                    if (
                        self._completed_frames != 0
                        or self._startup_fallback_factory is None
                    ):
                        raise
                    # VideoToolbox reports encoder-session exhaustion on the
                    # first append, not during AVAssetWriter construction.
                    # At this point no frame has been committed, so retrying
                    # the same frame through libx264 loses no timeline data.
                    self.encoder.discard()
                    self.encoder = self._startup_fallback_factory()
                    self.active_codec = self.encoder.active_codec
                    self.active_options = dict(self.encoder.active_options)
                    self._startup_fallback_factory = None
                    print(
                        "Swift VideoToolbox preview encoder unavailable; "
                        f"using libx264: {primary_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    events = self.encoder.add_frame(frame, pts_ns)
                self._completed_frames += 1
                for event in events:
                    self._events.put(event)
            except BaseException as exc:
                self._error = exc
                return
            finally:
                self._tasks.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"asynchronous preview encoder failed: {self._error}"
            ) from self._error

    def poll_events(self) -> list[dict]:
        self._raise_if_failed()
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def add_frame(self, frame: np.ndarray, pts_ns: int) -> list[dict]:
        if self._closed:
            raise RuntimeError("asynchronous preview encoder is closed")
        events = self.poll_events()
        while True:
            self._raise_if_failed()
            try:
                self._tasks.put((frame, int(pts_ns)), timeout=0.05)
                break
            except queue.Full:
                continue
        return events + self.poll_events()

    def finish(self) -> list[dict]:
        if self._closed:
            return []
        self._tasks.put(_ASYNC_ENCODER_STOP)
        self._thread.join()
        self._raise_if_failed()
        events = self.poll_events()
        events.extend(self.encoder.finish())
        self._closed = True
        return events

    def discard(self) -> None:
        if self._closed:
            return
        while True:
            try:
                self._tasks.get_nowait()
                self._tasks.task_done()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._tasks.put(_ASYNC_ENCODER_STOP)
            self._thread.join()
        self.encoder.discard()
        self._closed = True


def _resolve_model(value: str, lookup: Callable, kind: str) -> tuple[str, str]:
    model = lookup(value)
    if model is not None:
        return model.name, model.path
    path = Path(value)
    if path.exists():
        return value, str(path)
    raise ValueError(f"Unknown {kind} model: {value}")


def load_preview_models(config):
    import torch

    from lada import ModelFiles
    from lada.restorationpipeline import load_models
    from lada.utils import video_utils

    restoration_name, restoration_path = _resolve_model(
        config.restoration_model,
        ModelFiles.get_restoration_model_by_name,
        "restoration",
    )
    _detection_name, detection_path = _resolve_model(
        config.detection_model,
        ModelFiles.get_detection_model_by_name,
        "detection",
    )
    detection, restoration, pad_mode = load_models(
        torch.device(config.device),
        restoration_name,
        restoration_path,
        None,
        detection_path,
        fp16=config.fp16,
        detect_face_mosaics=config.detect_face_mosaics,
    )
    return {
        "detection": detection,
        "restoration": restoration,
        "pad_mode": pad_mode,
        "restoration_name": restoration_name,
        "metadata": video_utils.get_video_meta_data(config.input),
    }


def create_preview_restorer(config, models):
    from lada import ModelFiles
    from lada.restorationpipeline.frame_restorer import FrameRestorer

    realtime_optimize = bool(getattr(config, "realtime_optimize", False))
    enhancer_name = "none" if realtime_optimize else config.roi_enhancer
    enhancer_path = None if realtime_optimize else (config.roi_enhancer_model or None)
    if enhancer_path:
        enhancer_model = ModelFiles.get_enhancer_model_by_name(enhancer_path)
        if enhancer_model is not None:
            enhancer_path = enhancer_model.path
    return FrameRestorer(
        config.device,
        config.input,
        config.max_clip_length,
        models["restoration_name"],
        models["detection"],
        models["restoration"],
        models["pad_mode"],
        False,
        restore_sharpen_strength=config.sharpen_strength,
        restore_detail_boost=config.detail_boost,
        restore_blend_feather=config.blend_feather,
        restore_texture_mix=config.texture_mix,
        restore_smooth_strength=config.smooth_strength,
        restore_roi_enhancer=enhancer_name,
        restore_roi_enhancer_model_path=enhancer_path,
        restore_roi_enhancer_scale=config.roi_enhancer_scale,
        restore_roi_enhancer_strength=(
            0.0 if realtime_optimize else config.roi_enhancer_strength
        ),
        restore_roi_enhancer_tile=config.roi_enhancer_tile,
        restore_effect_upscale=1 if realtime_optimize else config.effect_upscale,
        fp16_enabled=config.fp16,
        mosaic_detection_empty_lookahead=config.detection_empty_lookahead,
        restore_max_frames=config.restore_max_frames,
        restore_temporal_overlap=config.restore_temporal_overlap,
        restore_crossfade=config.restore_crossfade,
    )


def _native_swift_preview_compatibility(config) -> tuple[bool, str]:
    """Return whether the no-Python-frame native preview path is applicable."""
    if os.environ.get("LADA_NATIVE_SWIFT_PREVIEW") != "1":
        return False, "native Swift preview is disabled"
    runner = os.environ.get("LADA_NATIVE_COREAI_PREVIEW_RUNNER")
    if not runner or not Path(runner).is_file():
        return False, "native Swift preview runner is unavailable"
    if config.restoration_model not in {
        "basicvsrpp-v1.2-coreai-variable",
        "basicvsrpp-v1.2-coreai-variable-hq",
    }:
        return False, "restoration model is not the variable Core AI model"
    if "coreai" not in config.detection_model or "jasna" in config.detection_model:
        return False, "detector is not a supported YOLO Core AI model"
    # These pixel-domain controls currently remain on the mature Python path.
    # Keeping the gate strict prevents the fast path from silently changing a
    # saved user's appearance settings.
    if any(
        abs(float(value)) > 1e-9
        for value in (
            config.sharpen_strength,
            config.detail_boost,
            config.texture_mix,
            config.smooth_strength,
            config.roi_enhancer_strength,
        )
    ):
        return False, "pixel-domain postprocessing is enabled"
    if config.effect_upscale != 1:
        return False, "effect upscaling is enabled"
    return True, ""


def run_native_swift_preview(config, output_dir: Path) -> int | None:
    """Run the CVPixelBuffer-resident Swift/Core AI preview pipeline.

    Python resolves user-facing model names and relays the existing JSON
    control protocol only. Decoded/restored frames never enter this process.
    """
    compatible, reason = _native_swift_preview_compatibility(config)
    if not compatible:
        print(
            f"Swift native preview bypassed: {reason}",
            file=sys.stderr,
            flush=True,
        )
        return None

    from lada import ModelFiles

    restoration = ModelFiles.get_restoration_model_by_name(
        config.restoration_model
    )
    detection = ModelFiles.get_detection_model_by_name(config.detection_model)
    if restoration is None or detection is None:
        return None
    restoration_path = Path(restoration.path)
    detection_path = Path(detection.path)
    if not restoration_path.is_dir() or not detection_path.is_dir():
        return None

    if config.restoration_model.endswith("-variable-hq"):
        restoration_runner = os.environ.get(
            "LADA_VARIABLE_COREAI_HQ_SWIFT_RUNNER"
        )
    else:
        restoration_runner = os.environ.get(
            "LADA_VARIABLE_COREAI_SWIFT_RUNNER"
        )
    native_runner = os.environ.get("LADA_NATIVE_COREAI_PREVIEW_RUNNER")
    if (
        not restoration_runner
        or not Path(restoration_runner).is_file()
        or not native_runner
    ):
        return None

    candidate_channels = 37 if detection_path.name.startswith(
        "lada_mosaic_detection_model_v2-"
    ) else 38
    temporal_frames = min(
        18,
        max(
            2,
            int(config.restore_max_frames or config.max_clip_length or 18),
        ),
    )
    payload = {
        "input": str(Path(config.input).resolve()),
        "outputDirectory": str(Path(output_dir).resolve()),
        "detectionModel": str(detection_path.resolve()),
        "detectionCandidateChannels": candidate_channels,
        "restorationModels": str(restoration_path.resolve()),
        "restorationRunner": str(Path(restoration_runner).resolve()),
        "startNanoseconds": int(config.start_ns),
        "generation": int(config.generation),
        "segmentSeconds": float(config.segment_seconds),
        "bufferLimitSeconds": float(config.buffer_limit),
        "temporalBatchFrames": temporal_frames,
        "ringCapacity": max(temporal_frames * 2, 24),
        "confidenceThreshold": 0.25,
        "iouThreshold": 0.7,
        "contextFraction": 0.30,
        "blendFeather": float(config.blend_feather),
    }
    config_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="mioh-native-preview-",
            suffix=".json",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            config_path = Path(stream.name)
        print(
            "Realtime preview: Swift CVPixelBuffer ring + Core AI "
            f"({temporal_frames} frames)",
            file=sys.stderr,
            flush=True,
        )
        completed = subprocess.run(
            [native_runner, str(config_path)],
            stdin=sys.stdin,
            stdout=PROTOCOL_STREAM,
            stderr=sys.stderr,
            check=False,
        )
        return int(completed.returncode)
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


class PreviewSession:
    def __init__(
        self,
        config,
        output_dir: Path,
        *,
        initial_generation: int = 0,
        model_loader: Callable = load_preview_models,
        restorer_factory: Callable = create_preview_restorer,
        encoder_factory: Callable | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_loader = model_loader
        self.restorer_factory = restorer_factory
        self.encoder_factory = encoder_factory
        self.models = None
        self.restorer = None
        self.encoder = None
        self.generation = initial_generation
        self.stopped = False
        self.commands: queue.Queue[PreviewCommand] = queue.Queue()
        self.control_thread: threading.Thread | None = None
        self._buffer_full_reported = False
        self._released_through = -1

    def _load_models_once(self):
        if self.models is None:
            self.models = self.model_loader(self.config)
        return self.models

    def _stop_restorer(self) -> None:
        if self.restorer is not None:
            self.restorer.stop()
            self.restorer = None
        if self.encoder is not None:
            self.encoder.discard()
            self.encoder = None

    def _delete_segments(self) -> None:
        for path in self.output_dir.glob("preview-g*-*.mp4*"):
            path.unlink(missing_ok=True)

    def start_generation(self, start_ns: int) -> None:
        models = self._load_models_once()
        self._stop_restorer()
        self._delete_segments()
        self._released_through = -1
        self.restorer = self.restorer_factory(self.config, models)
        self.restorer.start(start_ns=start_ns)
        self._buffer_full_reported = False

    def _release_segments_through(self, sequence: int) -> None:
        """Release completed player segments so the rolling buffer can refill."""
        if sequence <= self._released_through:
            return
        for released_sequence in range(self._released_through + 1, sequence + 1):
            path = self.output_dir / (
                f"preview-g{self.generation}-{released_sequence:06d}.mp4"
            )
            path.unlink(missing_ok=True)
        self._released_through = sequence
        self._buffer_full_reported = False

    def has_buffer_capacity(self) -> bool:
        limit = max(
            1,
            int(math.ceil(self.config.buffer_limit / self.config.segment_seconds)),
        )
        count = sum(
            1
            for _ in self.output_dir.glob(
                f"preview-g{self.generation}-*.mp4"
            )
        )
        return count < limit

    def _read_commands(self, stream: TextIO) -> None:
        try:
            for line in stream:
                try:
                    self.commands.put(PreviewCommand.parse(line))
                except Exception as exc:
                    print(f"Ignoring invalid preview command: {exc}", file=sys.stderr)
        finally:
            # The GUI owns this worker through stdin. When the app is replaced,
            # crashes, or quits, EOF must stop the preview and its Core AI child
            # processes instead of leaving a detached worker behind.
            self.commands.put(PreviewCommand("stop"))

    def _next_command(self) -> PreviewCommand | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def _apply_command(self, command: PreviewCommand) -> bool:
        if command.command == "stop":
            self.stopped = True
            return True
        if command.command == "release_through":
            if command.sequence is None or command.sequence < 0:
                raise ValueError("release_through requires non-negative sequence")
            self._release_segments_through(command.sequence)
            emit_event(
                "released",
                generation=self.generation,
                sequence=self._released_through,
            )
            return False
        if command.command == "set_buffer_limit":
            if command.seconds is None or command.seconds <= 0:
                raise ValueError("set_buffer_limit requires positive seconds")
            self.config.buffer_limit = command.seconds
            emit_event(
                "buffer_limit",
                generation=self.generation,
                seconds=self.config.buffer_limit,
            )
        return False

    def _new_encoder(self):
        metadata = self.models["metadata"]
        encoder_factory = self.encoder_factory
        if encoder_factory is None:
            runner = os.environ.get("LADA_PREVIEW_VIDEOTOOLBOX_RUNNER")
            encoder_factory = (
                SwiftVideoToolboxSegmentEncoder
                if runner and Path(runner).is_file()
                else SegmentEncoder
            )
        encoder_kwargs = dict(
            output_dir=self.output_dir,
            width=metadata.video_width,
            height=metadata.video_height,
            fps=metadata.video_fps_exact,
            generation=self.generation,
            preferred_codec="h264_videotoolbox",
            segment_seconds=self.config.segment_seconds,
        )
        encoder = encoder_factory(**encoder_kwargs)
        if encoder_factory is SwiftVideoToolboxSegmentEncoder:
            fallback_kwargs = dict(encoder_kwargs)
            fallback_kwargs["preferred_codec"] = "libx264"
            return AsyncSegmentEncoder(
                encoder,
                max_pending_frames=2,
                startup_fallback_factory=lambda: SegmentEncoder(
                    **fallback_kwargs
                ),
            )
        return encoder

    def _emit_encoder_events(self, events: list[dict]) -> None:
        for event in events:
            emit_event(**event)
            emit_event(
                "progress",
                generation=self.generation,
                position_ns=event["end_ns"],
            )

    def run(self, start_ns: int = 0, control_stream: TextIO | None = None) -> None:
        from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, ErrorMarker

        self._load_models_once()
        metadata = self.models["metadata"]
        emit_event(
            "ready",
            generation=self.generation,
            duration=metadata.duration,
            fps=float(metadata.video_fps_exact),
            width=metadata.video_width,
            height=metadata.video_height,
            segment_seconds=self.config.segment_seconds,
        )
        if control_stream is not None:
            self.control_thread = threading.Thread(
                target=self._read_commands,
                args=(control_stream,),
                name="mioh-preview-control",
                daemon=True,
            )
            self.control_thread.start()
        self.start_generation(start_ns)
        self.encoder = self._new_encoder()

        while not self.stopped:
            poll_events = getattr(self.encoder, "poll_events", None)
            if poll_events is not None:
                self._emit_encoder_events(poll_events())
            command = self._next_command()
            if command is not None:
                generation_changed = self._apply_command(command)
                if self.stopped:
                    break
                continue

            if not self.has_buffer_capacity():
                if not self._buffer_full_reported:
                    emit_event(
                        "buffer_full",
                        generation=self.generation,
                        seconds=self.config.buffer_limit,
                    )
                    self._buffer_full_reported = True
                time.sleep(0.05)
                continue
            self._buffer_full_reported = False

            try:
                result = self.restorer.get_frame_restoration_queue().get(timeout=0.1)
            except queue.Empty:
                continue
            if result is STOP_MARKER:
                continue
            if result is EOF_MARKER:
                self._emit_encoder_events(self.encoder.finish())
                emit_event("ended", generation=self.generation)
                self._stop_restorer()
                self.stopped = True
                break
            if isinstance(result, ErrorMarker):
                raise RuntimeError(str(result))

            frame, pts = result
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            pts_ns = int(
                Fraction(pts) * metadata.time_base * 1_000_000_000
            )
            self._emit_encoder_events(self.encoder.add_frame(frame, pts_ns))

    def stop(self, *, remove_output: bool) -> None:
        self.stopped = True
        self._stop_restorer()
        if remove_output:
            shutil.rmtree(self.output_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mioh restored preview worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-ns", type=int, default=0)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restoration-model", required=True)
    parser.add_argument("--detection-model", required=True)
    parser.add_argument("--max-clip-length", type=int, default=180)
    parser.add_argument("--restore-max-frames", type=int)
    parser.add_argument("--restore-temporal-overlap", type=int, default=8)
    parser.add_argument("--enable-crossfade", dest="restore_crossfade", action="store_true", default=True)
    parser.add_argument("--disable-crossfade", dest="restore_crossfade", action="store_false")
    parser.add_argument("--detect-face-mosaics", action="store_true")
    parser.add_argument("--detection-empty-lookahead", type=int, default=10)
    parser.add_argument("--sharpen-strength", type=float, default=0.0)
    parser.add_argument("--detail-boost", type=float, default=0.0)
    parser.add_argument("--blend-feather", type=float, default=1.0)
    parser.add_argument("--texture-mix", type=float, default=0.0)
    parser.add_argument("--smooth-strength", type=float, default=0.0)
    parser.add_argument("--roi-enhancer", default="none")
    parser.add_argument("--roi-enhancer-model", default="")
    parser.add_argument("--roi-enhancer-scale", type=int, default=4)
    parser.add_argument("--roi-enhancer-strength", type=float, default=0.0)
    parser.add_argument("--roi-enhancer-tile", type=int, default=0)
    parser.add_argument("--effect-upscale", type=int, default=1)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--buffer-limit", type=float, default=8.0)
    parser.add_argument(
        "--realtime-optimize",
        action="store_true",
        help="Keep restoration enabled while skipping secondary enhancement work.",
    )
    return parser


def main() -> int:
    isolate_process_group()
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global PROTOCOL_STREAM
    PROTOCOL_STREAM = sys.stdout
    sys.stdout = sys.stderr
    native_result = run_native_swift_preview(args, output_dir)
    if native_result is not None:
        return native_result
    session = PreviewSession(
        args,
        output_dir,
        initial_generation=args.generation,
    )
    try:
        session.run(args.start_ns, control_stream=sys.stdin)
        return 0
    except Exception as exc:
        emit_event(
            "error",
            generation=session.generation,
            message="リアルタイムプレビューを開始できませんでした",
            detail=str(exc),
        )
        return 1
    finally:
        session.stop(remove_output=False)


if __name__ == "__main__":
    raise SystemExit(main())
