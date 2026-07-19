# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Select and describe local videos with a multimodal llama.cpp server.

The local Gemma server accepts images, not video containers.  This module
therefore extracts ordered representative frames with ffmpeg, combines them
into a labelled contact sheet and asks the model for a strict JSON decision.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from lada.utils import video_utils


DEFAULT_API_URL = "http://127.0.0.1:18080/v1"


@dataclass(frozen=True)
class AnalysisWindow:
    start: float
    end: float


def collect_video_files(inputs: Iterable[Path]) -> list[Path]:
    files: dict[str, Path] = {}
    for input_path in inputs:
        candidates = input_path.rglob("*") if input_path.is_dir() else [input_path]
        for candidate in candidates:
            if candidate.is_file() and video_utils.is_video_file(candidate):
                resolved = candidate.expanduser().resolve()
                files[str(resolved)] = resolved
    return [files[key] for key in sorted(files)]


def probe_video(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe failed: {path}")
    payload = json.loads(completed.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"video stream not found: {path}")
    stream = payload["streams"][0]
    duration = float(payload.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise RuntimeError(f"video duration is unavailable: {path}")
    return {
        "duration": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "codec": stream.get("codec_name"),
        "fps": stream.get("avg_frame_rate"),
    }


def build_analysis_windows(
    duration: float,
    window_seconds: float,
    scan_interval: float,
) -> list[AnalysisWindow]:
    """Build windows to inspect.

    A non-positive window length means one representative pass over the whole
    file.  Otherwise windows start at scan_interval boundaries.  The last
    partial window is retained so the end of a video is not silently omitted.
    """
    if duration <= 0:
        return []
    if window_seconds <= 0:
        return [AnalysisWindow(0.0, duration)]
    if scan_interval <= 0:
        scan_interval = window_seconds
    starts: list[float] = []
    start = 0.0
    while start < duration:
        starts.append(start)
        start += scan_interval
    return [AnalysisWindow(start, min(start + window_seconds, duration)) for start in starts]


def sample_timestamps(window: AnalysisWindow, count: int) -> list[float]:
    if count <= 0 or window.end <= window.start:
        return []
    duration = window.end - window.start
    # Sample at cell centres. This avoids title/transition frames at boundaries.
    return [window.start + duration * ((index + 0.5) / count) for index in range(count)]


def extract_frame(
    video: Path,
    timestamp: float,
    ffmpeg: str = "ffmpeg",
    max_side: int = 640,
) -> np.ndarray:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0 or not completed.stdout:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(error or f"could not extract frame at {timestamp:.3f}s")
    encoded = np.frombuffer(completed.stdout, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"ffmpeg returned an invalid image at {timestamp:.3f}s")
    return frame


def format_timestamp(seconds: float) -> str:
    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def make_contact_sheet(
    frames: list[np.ndarray],
    timestamps: list[float],
    columns: int = 4,
    tile_width: int = 384,
    tile_height: int = 240,
) -> bytes:
    if not frames or len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must be non-empty and have equal length")
    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)
    label_height = 30
    sheet = np.full((rows * (tile_height + label_height), columns * tile_width, 3), 20, np.uint8)
    for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        row, column = divmod(index, columns)
        available_height = tile_height
        scale = min(tile_width / frame.shape[1], available_height / frame.shape[0])
        width = max(1, int(round(frame.shape[1] * scale)))
        height = max(1, int(round(frame.shape[0] * scale)))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        x = column * tile_width + (tile_width - width) // 2
        y = row * (tile_height + label_height) + (available_height - height) // 2
        sheet[y:y + height, x:x + width] = resized
        label_y = row * (tile_height + label_height) + tile_height + 21
        cv2.putText(
            sheet,
            f"#{index + 1}  {format_timestamp(timestamp)}",
            (column * tile_width + 8, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("failed to encode contact sheet")
    return encoded.tobytes()


def build_prompt(criteria: str, window: AnalysisWindow) -> str:
    return f"""You are classifying an ordered contact sheet made from one video interval.
The tiles are chronological and labelled #1, #2, ... with timestamps.

Selection criteria supplied by the user:
{criteria}

Judge only visible evidence. Do not guess identities or facts outside the frames.
Describe the concrete activity at a non-graphic category level. Also classify body
orientation and composition. If evidence is ambiguous, lower confidence instead of
inventing details. Return exactly one JSON object and no markdown.

Interval: {window.start:.3f}-{window.end:.3f} seconds

Required JSON shape:
{{
  "matches_criteria": true,
  "confidence": 0.0,
  "summary_ja": "brief Japanese description",
  "activities_ja": ["category"],
  "body_orientations": ["front|back|profile|three_quarter|mixed|unclear"],
  "poses": ["standing|sitting|lying|kneeling|crouching|mixed|unclear"],
  "framings": ["extreme_close_up|close_up|medium|full_body|wide|mixed|unclear"],
  "camera_angles": ["eye_level|high_angle|low_angle|overhead|mixed|unclear"],
  "visible_people": 0,
  "sharpness": 0.0,
  "occlusion": 0.0,
  "evidence_frames": [1],
  "reasons_ja": ["reason"]
}}

All numeric scores are between 0 and 1. occlusion=1 means heavily obstructed.
"""


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Gemma did not return JSON: {text[:300]}")
        payload = json.loads(stripped[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Gemma response must be a JSON object")
    return payload


class GemmaClient:
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: str = "auto",
        timeout: float = 180.0,
        retries: int = 2,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemma API HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Gemma API unavailable at {self.api_url}: {error.reason}") from error

    def resolve_model(self) -> str:
        if self.model != "auto":
            return self.model
        payload = self._request("GET", "/models")
        models = payload.get("data") or payload.get("models") or []
        if not models:
            raise RuntimeError("Gemma server reported no loaded model")
        first = models[0]
        self.model = first.get("id") or first.get("model") or first.get("name")
        if not self.model:
            raise RuntimeError("could not determine the loaded Gemma model name")
        return self.model

    def classify(self, jpeg: bytes, prompt: str) -> dict[str, Any]:
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        payload = {
            "model": self.resolve_model(),
            "temperature": 0,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._request("POST", "/chat/completions", payload)
                content = response["choices"][0]["message"]["content"]
                return parse_json_object(content)
            except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        assert last_error is not None
        raise last_error


def criteria_digest(criteria: str) -> str:
    return hashlib.sha256(criteria.strip().encode("utf-8")).hexdigest()[:16]


def record_key(record: dict[str, Any]) -> tuple[str, float, float, str]:
    return (
        str(record.get("video", "")),
        round(float(record.get("window_start", 0)), 3),
        round(float(record.get("window_end", 0)), 3),
        str(record.get("criteria_id", "")),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if isinstance(value, dict):
                records.append(value)
    return records


def write_selected_list(records: Iterable[dict[str, Any]], output: Path) -> list[Path]:
    selected = sorted(
        {
            Path(record["video"])
            for record in records
            if record.get("status") == "ok" and record.get("selected") is True
        },
        key=str,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{path}\n" for path in selected), encoding="utf-8")
    return selected


def materialize_selected_files(selected: Iterable[Path], output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    for source in selected:
        name = source.name
        if name in used_names or (output_dir / name).exists():
            suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
            name = f"{source.stem}-{suffix}{source.suffix}"
        used_names.add(name)
        destination = output_dir / name
        if destination.exists() or destination.is_symlink():
            continue
        if mode == "symlink":
            destination.symlink_to(source)
        elif mode == "hardlink":
            destination.hardlink_to(source)
        elif mode == "copy":
            shutil.copy2(source, destination)
        else:
            raise ValueError(f"unsupported materialization mode: {mode}")
