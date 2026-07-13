#!/usr/bin/env python3
"""Persistent restored-preview worker used by the standalone mioh app."""

from __future__ import annotations

import argparse
import json
import math
import queue
import shutil
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, TextIO

import av
import numpy as np


PROTOCOL_STREAM = sys.stdout


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
        position_ns: int | None = None,
        seconds: float | None = None,
    ) -> None:
        self.command = command
        self.position_ns = position_ns
        self.seconds = seconds

    @classmethod
    def parse(cls, line: str) -> "PreviewCommand":
        payload = json.loads(line)
        command = str(payload["command"])
        if command not in {"seek", "set_buffer_limit", "stop"}:
            raise ValueError(f"Unsupported preview command: {command}")
        position_ns = payload.get("position_ns")
        seconds = payload.get("seconds")
        return cls(
            command,
            int(position_ns) if position_ns is not None else None,
            float(seconds) if seconds is not None else None,
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
        path = self._new_path()
        container = av.open(str(path), mode="w", format="mp4")
        try:
            stream = self.stream_factory(container, codec, self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(self.fps.denominator, self.fps.numerator)
            stream.options = options
        except Exception:
            container.close()
            path.unlink(missing_ok=True)
            raise
        self.container = container
        self.video_stream = stream
        self.segment_path = path
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
        for packet in self.video_stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        event = {
            "kind": "segment",
            "generation": self.generation,
            "sequence": self.sequence,
            "start_ns": self.segment_start_ns,
            "end_ns": end_ns,
            "path": str(self.segment_path),
            "codec": self.active_codec,
        }
        self.container = None
        self.video_stream = None
        self.segment_path = None
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

    enhancer_path = config.roi_enhancer_model or None
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
        restore_roi_enhancer=config.roi_enhancer,
        restore_roi_enhancer_model_path=enhancer_path,
        restore_roi_enhancer_scale=config.roi_enhancer_scale,
        restore_roi_enhancer_strength=config.roi_enhancer_strength,
        restore_roi_enhancer_tile=config.roi_enhancer_tile,
        restore_effect_upscale=config.effect_upscale,
        fp16_enabled=config.fp16,
        mosaic_detection_empty_lookahead=config.detection_empty_lookahead,
        restore_max_frames=config.restore_max_frames,
    )


class PreviewSession:
    def __init__(
        self,
        config,
        output_dir: Path,
        *,
        model_loader: Callable = load_preview_models,
        restorer_factory: Callable = create_preview_restorer,
        encoder_factory: Callable = SegmentEncoder,
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
        self.generation = 0
        self.stopped = False
        self.commands: queue.Queue[PreviewCommand] = queue.Queue()
        self.control_thread: threading.Thread | None = None
        self._buffer_full_reported = False

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
        for path in self.output_dir.glob("preview-g*-*.mp4"):
            path.unlink(missing_ok=True)

    def start_generation(self, start_ns: int) -> None:
        models = self._load_models_once()
        self._stop_restorer()
        self._delete_segments()
        self.restorer = self.restorer_factory(self.config, models)
        self.restorer.start(start_ns=start_ns)
        self._buffer_full_reported = False

    def seek(self, position_ns: int) -> None:
        self.generation += 1
        self.start_generation(position_ns)

    def has_buffer_capacity(self) -> bool:
        limit = max(
            1,
            int(math.ceil(self.config.buffer_limit / self.config.segment_seconds)),
        )
        active_path = getattr(self.encoder, "segment_path", None)
        count = sum(
            1
            for path in self.output_dir.glob(
                f"preview-g{self.generation}-*.mp4"
            )
            if path != active_path
        )
        return count < limit

    def _read_commands(self, stream: TextIO) -> None:
        for line in stream:
            try:
                self.commands.put(PreviewCommand.parse(line))
            except Exception as exc:
                print(f"Ignoring invalid preview command: {exc}", file=sys.stderr)

    def _next_command(self) -> PreviewCommand | None:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def _apply_command(self, command: PreviewCommand) -> bool:
        if command.command == "stop":
            self.stopped = True
            return True
        if command.command == "seek":
            if command.position_ns is None:
                raise ValueError("seek requires position_ns")
            self.seek(command.position_ns)
            return True
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
        return self.encoder_factory(
            output_dir=self.output_dir,
            width=metadata.video_width,
            height=metadata.video_height,
            fps=metadata.video_fps_exact,
            generation=self.generation,
            preferred_codec="h264_videotoolbox",
            segment_seconds=self.config.segment_seconds,
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
            command = self._next_command()
            if command is not None:
                generation_changed = self._apply_command(command)
                if self.stopped:
                    break
                if generation_changed:
                    self.encoder = self._new_encoder()
                    emit_event("progress", generation=self.generation, position_ns=command.position_ns or 0)
                continue

            if not self.has_buffer_capacity():
                if not self._buffer_full_reported:
                    emit_event("buffer_full", generation=self.generation)
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
                for event in self.encoder.finish():
                    emit_event(**event)
                emit_event("ended", generation=self.generation)
                break
            if isinstance(result, ErrorMarker):
                raise RuntimeError(str(result))

            frame, pts = result
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            pts_ns = int(
                Fraction(pts) * metadata.time_base * 1_000_000_000
            )
            for event in self.encoder.add_frame(frame, pts_ns):
                emit_event(**event)
                emit_event("progress", generation=self.generation, position_ns=event["end_ns"])

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
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restoration-model", required=True)
    parser.add_argument("--detection-model", required=True)
    parser.add_argument("--max-clip-length", type=int, default=180)
    parser.add_argument("--restore-max-frames", type=int)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global PROTOCOL_STREAM
    PROTOCOL_STREAM = sys.stdout
    sys.stdout = sys.stderr
    session = PreviewSession(args, output_dir)
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
