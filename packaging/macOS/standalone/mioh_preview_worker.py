#!/usr/bin/env python3
"""Persistent restored-preview worker used by the standalone mioh app."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, TextIO

import av
import numpy as np


def emit_event(
    kind: str,
    *,
    generation: int,
    stream: TextIO | None = None,
    **payload,
) -> None:
    destination = stream or sys.stdout
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mioh restored preview worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-ns", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emit_event(
        "error",
        generation=0,
        message="Preview session is not initialized",
        detail="Persistent restoration session is unavailable",
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
