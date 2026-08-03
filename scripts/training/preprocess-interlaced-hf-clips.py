#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Audit and losslessly prepare selected HF-training source clips.

The command has two deliberately separate phases:

``scan``
    Probe complete source files cheaply at several timestamps with FFmpeg's
    ``idet`` filter.  No source is rewritten.  The resulting JSON records the
    observed cadence, field order, colour metadata and a sampled source-file
    fingerprint.

``prepare``
    Read a JSONL scene list and transcode only those selected frame ranges.
    Interlaced sources are converted with ``bwdif=send_frame`` so 29.97 fps
    input produces exactly the requested number of 29.97 fps progressive
    frames.  Output is lossless FFV1/yuv420p, tagged BT.709 limited-range, and
    verified before an atomic rename.  A provenance manifest contains every
    command, input fingerprint and output SHA-256.

``prepare-progressive``
    Read selected ranges from progressive 29.97/30/59.94/60 fps sources.
    Frames are selected deterministically with FFmpeg's ``fps`` filter at the
    30000/1001 training rate.  There is no spatial resampling.  The output
    contract and provenance guarantees are otherwise the same as ``prepare``.

Scene JSONL example::

    {"name":"stars152-a","source":"/path/STARS-152.mp4",
     "start_frame":107892,"frame_count":240}

Existing short scene clips can be processed without another temporal cut by
omitting ``start_frame`` and ``frame_count``.  This keeps whole-film
deinterlacing out of the dataset preparation path.

Progressive scene JSONL uses an input-rate start position and an output-rate
frame count::

    {"name":"fc2-train-a","source":"/path/source.mp4",
     "start_frame":17982,"output_frame_count":3600}

``start_seconds`` may replace ``start_frame``.  ``frame_count`` is accepted as
an alias for ``output_frame_count`` so existing selection manifests remain
easy to adapt.  In the progressive command it always means *output* frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


WORKING_RATE = Fraction(30_000, 1_001)
SUPPORTED_PROGRESSIVE_RATES = frozenset(
    {
        WORKING_RATE,
        Fraction(30, 1),
        Fraction(60_000, 1_001),
        Fraction(60, 1),
    }
)
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".ts", ".m2ts"}
SCAN_SCHEMA = "mioh.hf-interlace-scan.v1"
PROVENANCE_SCHEMA = "mioh.hf-deinterlace-provenance.v1"
PROGRESSIVE_PROVENANCE_SCHEMA = "mioh.hf-progressive-provenance.v1"
IDET_RE = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_program(value: str) -> str:
    resolved = shutil.which(value) if os.sep not in value else value
    if not resolved or not Path(resolved).is_file():
        raise FileNotFoundError(f"required executable not found: {value}")
    return str(Path(resolved).resolve())


def _run(
    command: Sequence[str], *, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _tool_version(program: str) -> str:
    result = _run([program, "-version"])
    return result.stdout.splitlines()[0].strip()


def sampled_file_fingerprint(path: Path, chunk_size: int = 1 << 20) -> dict[str, Any]:
    """Hash first/middle/last chunks plus size without reading a whole film."""

    path = path.resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(b"mioh-sampled-file-v1\0")
    digest.update(str(stat.st_size).encode("ascii"))
    offsets = sorted(
        {
            0,
            max(0, stat.st_size // 2 - chunk_size // 2),
            max(0, stat.st_size - chunk_size),
        }
    )
    with path.open("rb") as source:
        for offset in offsets:
            source.seek(offset)
            payload = source.read(chunk_size)
            digest.update(offset.to_bytes(8, "little"))
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return {
        "algorithm": "sha256-size-first-middle-last-1MiB-v1",
        "digest": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(ffprobe: str, path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command += [
        "-show_entries",
        (
            "stream=index,codec_name,width,height,pix_fmt,color_range,color_space,"
            "color_transfer,color_primaries,field_order,r_frame_rate,avg_frame_rate,"
            "duration,nb_frames,nb_read_frames:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(_run(command).stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected one primary video stream in {path}")
    stream = dict(streams[0])
    format_value = payload.get("format") or {}
    duration = stream.get("duration") or format_value.get("duration")
    if duration is None:
        raise ValueError(f"video duration is unavailable: {path}")
    stream["duration_seconds"] = float(duration)
    return stream


def parse_idet(stderr: str) -> dict[str, int]:
    matches = IDET_RE.findall(stderr)
    if not matches:
        raise RuntimeError("FFmpeg idet did not emit a multi-frame summary")
    tff, bff, progressive, undetermined = (int(value) for value in matches[-1])
    return {
        "tff": tff,
        "bff": bff,
        "progressive": progressive,
        "undetermined": undetermined,
    }


def idet_sample(
    ffmpeg: str,
    path: Path,
    *,
    start_seconds: float,
    frame_count: int,
) -> dict[str, Any]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-vf",
        "idet",
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    result = _run(command)
    return {
        "start_seconds": round(start_seconds, 6),
        "requested_frames": frame_count,
        "counts": parse_idet(result.stderr),
    }


def sample_positions(duration: float, sample_count: int, sample_frames: int) -> list[float]:
    if sample_count <= 0 or sample_frames <= 0:
        raise ValueError("sample count and frame count must be positive")
    sample_duration = float(Fraction(sample_frames, 1) / WORKING_RATE)
    maximum = max(0.0, duration - sample_duration)
    if sample_count == 1:
        return [maximum / 2.0]
    return [maximum * (0.15 + 0.70 * index / (sample_count - 1)) for index in range(sample_count)]


def aggregate_idet(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in ("tff", "bff", "progressive", "undetermined")}
    for sample in samples:
        for key in counts:
            counts[key] += int(sample["counts"][key])
    total = sum(counts.values())
    decided = counts["tff"] + counts["bff"] + counts["progressive"]
    interlaced = counts["tff"] + counts["bff"]
    ratios = {
        "interlaced": interlaced / decided if decided else 0.0,
        "progressive": counts["progressive"] / decided if decided else 0.0,
        "tff_within_interlaced": counts["tff"] / interlaced if interlaced else 0.0,
        "bff_within_interlaced": counts["bff"] / interlaced if interlaced else 0.0,
        "undetermined": counts["undetermined"] / total if total else 1.0,
    }
    if ratios["interlaced"] >= 0.80:
        classification = "interlaced"
        if ratios["tff_within_interlaced"] >= 0.80:
            parity = "tff"
        elif ratios["bff_within_interlaced"] >= 0.80:
            parity = "bff"
        else:
            parity = None
    elif ratios["progressive"] >= 0.80:
        classification = "progressive"
        parity = None
    else:
        classification = "mixed_or_unknown"
        parity = None
    return {
        "counts": counts,
        "ratios": ratios,
        "classification": classification,
        "parity": parity,
    }


def scan_source(
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    *,
    sample_count: int,
    sample_frames: int,
) -> dict[str, Any]:
    path = path.resolve()
    probe = probe_video(ffprobe, path)
    samples = [
        idet_sample(ffmpeg, path, start_seconds=start, frame_count=sample_frames)
        for start in sample_positions(probe["duration_seconds"], sample_count, sample_frames)
    ]
    return {
        "source": str(path),
        "source_fingerprint": sampled_file_fingerprint(path),
        "probe": probe,
        "idet_samples": samples,
        "idet_aggregate": aggregate_idet(samples),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, indent=2, ensure_ascii=False, sort_keys=True)
        temporary.write("\n")
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def expand_inputs(values: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        value = value.expanduser()
        if value.is_dir():
            result.extend(
                path for path in sorted(value.iterdir()) if path.suffix.lower() in VIDEO_SUFFIXES
            )
        elif value.is_file():
            result.append(value)
        else:
            raise FileNotFoundError(value)
    unique = {path.resolve(): None for path in result}
    if not unique:
        raise ValueError("no video inputs found")
    return list(unique)


def build_scan_report(
    ffmpeg: str,
    ffprobe: str,
    inputs: Sequence[Path],
    *,
    sample_count: int,
    sample_frames: int,
) -> dict[str, Any]:
    return {
        "schema": SCAN_SCHEMA,
        "created_at": _utc_now(),
        "tools": {
            "ffmpeg": _tool_version(ffmpeg),
            "ffprobe": _tool_version(ffprobe),
        },
        "working_rate": str(WORKING_RATE),
        "sample_count": sample_count,
        "sample_frames": sample_frames,
        "sources": [
            scan_source(
                ffmpeg,
                ffprobe,
                path,
                sample_count=sample_count,
                sample_frames=sample_frames,
            )
            for path in inputs
        ],
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"scene list is empty: {path}")
    return records


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not value:
        raise ValueError("scene name is empty after sanitization")
    return value[:120]


def _rate_is_2997(probe: dict[str, Any]) -> bool:
    try:
        return Fraction(probe["avg_frame_rate"]) == WORKING_RATE
    except (KeyError, ValueError, ZeroDivisionError):
        return False


def _progressive_source_rate(probe: dict[str, Any]) -> Fraction:
    """Return an exact supported CFR rate or reject ambiguous/VFR input."""

    value = probe.get("avg_frame_rate")
    if value in (None, "", "0/0", "N/A"):
        value = probe.get("r_frame_rate")
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid progressive source frame rate: {value!r}") from error
    if rate in SUPPORTED_PROGRESSIVE_RATES:
        return rate
    # Some containers derive avg_frame_rate from a millisecond-rounded
    # duration, yielding e.g. 19001/317 for an actual 60000/1001 CFR stream.
    # Canonicalize only sub-ppm drift; this does not admit approximate/VFR
    # material such as 59.6 fps.
    for candidate in SUPPORTED_PROGRESSIVE_RATES:
        if abs(rate - candidate) / candidate <= Fraction(1, 1_000_000):
            return candidate
    allowed = ", ".join(str(item) for item in sorted(SUPPORTED_PROGRESSIVE_RATES))
    raise ValueError(
        f"unsupported progressive source frame rate {rate}; expected one of {allowed}"
    )


def _fraction_from_record(value: object, *, field: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} is not a finite rational value: {value!r}") from error
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _progressive_scene_values(
    record: dict[str, Any],
    *,
    root: Path,
    scan: dict[str, Any],
) -> tuple[Path, Fraction, Fraction, int, str]:
    """Resolve one progressive scene without probing or decoding the film."""

    if "source" not in record:
        raise ValueError("scene is missing source")
    source = Path(str(record["source"])).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    source_rate = _progressive_source_rate(scan.get("probe") or {})
    has_start_frame = "start_frame" in record
    has_start_seconds = "start_seconds" in record
    if has_start_frame and has_start_seconds:
        raise ValueError("scene cannot specify both start_frame and start_seconds")
    if has_start_seconds:
        start_seconds = _fraction_from_record(
            record["start_seconds"], field="start_seconds"
        )
    else:
        start_frame = int(record.get("start_frame", 0))
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        start_seconds = Fraction(start_frame, 1) / source_rate

    has_output_count = "output_frame_count" in record
    has_legacy_count = "frame_count" in record
    if has_output_count and has_legacy_count:
        if int(record["output_frame_count"]) != int(record["frame_count"]):
            raise ValueError(
                "output_frame_count and frame_count aliases disagree"
            )
    if not has_output_count and not has_legacy_count:
        raise ValueError(
            "progressive scene requires output_frame_count (or frame_count alias)"
        )
    output_count = int(
        record["output_frame_count"] if has_output_count else record["frame_count"]
    )
    if output_count <= 0:
        raise ValueError("output_frame_count must be positive")

    probe = scan.get("probe") or {}
    duration = _fraction_from_record(
        probe.get("duration_seconds"), field="source duration"
    )
    # N output frames sample timestamps 0 through (N-1)/WORKING_RATE.  Permit
    # one native-frame tolerance for conservative container duration metadata;
    # exact frame-count verification still rejects a genuinely short source.
    last_sample = start_seconds + Fraction(output_count - 1, 1) / WORKING_RATE
    if last_sample > duration + Fraction(1, 1) / source_rate:
        raise ValueError(
            "progressive scene exceeds source duration: "
            f"last sample {float(last_sample):.6f}s, duration {float(duration):.6f}s"
        )

    default_start = int(start_seconds * source_rate)
    default_name = f"{source.stem}-f{default_start:09d}-n{output_count:06d}"
    name = _safe_name(str(record.get("name", default_name)))
    return source, source_rate, start_seconds, output_count, name


def _scene_values(
    record: dict[str, Any], *, root: Path, ffprobe: str
) -> tuple[Path, int, int, str]:
    if "source" not in record:
        raise ValueError("scene is missing source")
    source = Path(str(record["source"])).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # Do not use ffprobe -count_frames on a multi-hour source: it decodes the
    # entire video merely to validate a short selected range.  Container frame
    # counts are used when available.  Duration is a cheap conservative
    # fallback for an explicitly bounded scene.  Exact counting is reserved
    # for the documented "process this whole short clip" form.
    probe = probe_video(ffprobe, source, count_frames=False)
    if not _rate_is_2997(probe):
        raise ValueError(f"scene source is not 30000/1001 fps: {source}")
    available = probe.get("nb_frames")
    if available in (None, "N/A") and "frame_count" not in record:
        probe = probe_video(ffprobe, source, count_frames=True)
        available = probe.get("nb_read_frames") or probe.get("nb_frames")
    if available in (None, "N/A"):
        available = int(probe["duration_seconds"] * float(WORKING_RATE))
    available = int(available)
    start = int(record.get("start_frame", 0))
    count = int(record.get("frame_count", available - start))
    if start < 0 or count <= 0 or start + count > available:
        raise ValueError(
            f"scene range [{start}, {start + count}) exceeds {available} frames in {source}"
        )
    default_name = f"{source.stem}-f{start:09d}-n{count:06d}"
    return source, start, count, _safe_name(str(record.get("name", default_name)))


def build_deinterlace_command(
    ffmpeg: str,
    *,
    source: Path,
    output: Path,
    start_frame: int,
    frame_count: int,
    parity: str,
) -> list[str]:
    if parity not in {"tff", "bff"}:
        raise ValueError(f"unsupported field parity: {parity}")
    seek = float(Fraction(start_frame, 1) / WORKING_RATE)
    filter_graph = (
        f"bwdif=mode=send_frame:parity={parity}:deint=all,"
        "setfield=prog,setpts=N*1001/(30000*TB),format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:"
        "color_trc=bt709:colorspace=bt709"
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "warning",
        "-y",
        "-ss",
        f"{seek:.9f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-vf",
        filter_graph,
        "-fps_mode",
        "passthrough",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-metadata",
        f"comment=mioh HF preprocessing: bwdif send_frame {parity}",
        str(output),
    ]


def build_progressive_command(
    ffmpeg: str,
    *,
    source: Path,
    output: Path,
    start_seconds: Fraction,
    output_frame_count: int,
    source_rate: Fraction,
) -> list[str]:
    """Build a CFR-normalizing command with no spatial filtering."""

    if source_rate not in SUPPORTED_PROGRESSIVE_RATES:
        raise ValueError(f"unsupported progressive source rate: {source_rate}")
    if start_seconds < 0:
        raise ValueError("start_seconds must be non-negative")
    if output_frame_count <= 0:
        raise ValueError("output_frame_count must be positive")
    seek = f"{float(start_seconds):.12f}"
    filter_graph = (
        "setpts=PTS-STARTPTS,"
        "fps=fps=30000/1001:start_time=0:round=near:eof_action=pass,"
        "setfield=prog,format=yuv420p,"
        "setparams=range=limited:color_primaries=bt709:"
        "color_trc=bt709:colorspace=bt709"
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "warning",
        "-y",
        "-ss",
        seek,
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(output_frame_count),
        "-vf",
        filter_graph,
        "-fps_mode",
        "passthrough",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-metadata",
        (
            "comment=mioh HF preprocessing: progressive fps selection "
            f"{source_rate} to {WORKING_RATE}"
        ),
        str(output),
    ]


def _verify_prepared_progressive_clip(
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    *,
    expected_frames: int,
    expected_width: int | None,
    expected_height: int | None,
    operation: str,
    require_idet_progressive: bool,
) -> dict[str, Any]:
    probe = probe_video(ffprobe, path, count_frames=True)
    actual_frames = int(probe.get("nb_read_frames") or 0)
    errors: list[str] = []
    if actual_frames != expected_frames:
        errors.append(f"frame count {actual_frames} != {expected_frames}")
    if not _rate_is_2997(probe):
        errors.append(f"average frame rate is {probe.get('avg_frame_rate')}")
    if expected_width is not None and int(probe.get("width") or 0) != expected_width:
        errors.append(f"width {probe.get('width')} != {expected_width}")
    if expected_height is not None and int(probe.get("height") or 0) != expected_height:
        errors.append(f"height {probe.get('height')} != {expected_height}")
    expected_tags = {
        "codec_name": "ffv1",
        "pix_fmt": "yuv420p",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "field_order": "progressive",
    }
    for key, expected in expected_tags.items():
        if probe.get(key) != expected:
            errors.append(f"{key}={probe.get(key)!r}, expected {expected!r}")
    idet_frames = min(expected_frames, 300)
    post_idet = idet_sample(
        ffmpeg,
        path,
        start_seconds=0.0,
        frame_count=idet_frames,
    )
    aggregate = aggregate_idet([post_idet])
    # idet is a classifier rather than a ground-truth comb detector.  Enforce
    # its 80% consensus after bwdif, where removal of combing is the operation
    # under test.  For an already-audited progressive source, record idet but
    # rely on the exact frame/rate/field-order contract: high-frequency test
    # patterns can otherwise produce false interlace positives.
    if require_idet_progressive and aggregate["ratios"]["progressive"] < 0.80:
        errors.append(
            f"post-{operation} idet progressive ratio is "
            f"{aggregate['ratios']['progressive']:.4f}"
        )
    verification = {
        "passed": not errors,
        "errors": errors,
        "probe": probe,
        "idet": aggregate,
        "idet_progressive_required": require_idet_progressive,
    }
    if errors:
        raise RuntimeError(
            f"{operation} clip verification failed for {path}: "
            + "; ".join(errors)
        )
    return verification


def verify_deinterlaced_clip(
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    return _verify_prepared_progressive_clip(
        ffmpeg,
        ffprobe,
        path,
        expected_frames=expected_frames,
        expected_width=None,
        expected_height=None,
        operation="bwdif",
        require_idet_progressive=True,
    )


def verify_progressive_clip(
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, Any]:
    return _verify_prepared_progressive_clip(
        ffmpeg,
        ffprobe,
        path,
        expected_frames=expected_frames,
        expected_width=expected_width,
        expected_height=expected_height,
        operation="progressive-fps",
        require_idet_progressive=False,
    )


def prepare_scenes(
    ffmpeg: str,
    ffprobe: str,
    *,
    scan_report_path: Path,
    scenes_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    report = _read_json(scan_report_path)
    if report.get("schema") != SCAN_SCHEMA:
        raise ValueError(f"unsupported scan report schema: {report.get('schema')}")
    scans = {
        Path(record["source"]).resolve(): record for record in report.get("sources", [])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for record in _read_jsonl(scenes_path):
        source, start, count, name = _scene_values(
            record, root=scenes_path.parent, ffprobe=ffprobe
        )
        scan = scans.get(source)
        if scan is None:
            raise ValueError(f"source is absent from scan report: {source}")
        current_fingerprint = sampled_file_fingerprint(source)
        if current_fingerprint != scan.get("source_fingerprint"):
            raise RuntimeError(f"source changed after interlace scan: {source}")
        aggregate = scan["idet_aggregate"]
        if aggregate.get("classification") != "interlaced":
            raise ValueError(f"source is not classified as interlaced: {source}")
        parity = aggregate.get("parity")
        if parity not in {"tff", "bff"}:
            raise ValueError(f"field parity is not reliable for {source}")
        source_probe = scan.get("probe") or {}
        for key in ("color_space", "color_transfer", "color_primaries"):
            if source_probe.get(key) != "bt709":
                raise ValueError(f"{source} lacks required BT.709 {key} metadata")
        if source_probe.get("color_range") != "tv":
            raise ValueError(f"{source} is not tagged limited-range")

        output = output_dir / f"{name}--bwdif-{parity}.mkv"
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        temporary = output_dir / f".{output.stem}.{uuid.uuid4().hex}.part.mkv"
        command = build_deinterlace_command(
            ffmpeg,
            source=source,
            output=temporary,
            start_frame=start,
            frame_count=count,
            parity=parity,
        )
        try:
            _run(command)
            verification = verify_deinterlaced_clip(
                ffmpeg,
                ffprobe,
                temporary,
                expected_frames=count,
            )
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        results.append(
            {
                "name": name,
                "source": str(source),
                "source_fingerprint": current_fingerprint,
                "source_scan_classification": aggregate,
                "start_frame": start,
                "frame_count": count,
                "working_rate": str(WORKING_RATE),
                "operation": {
                    "filter": "bwdif send_frame, known parity, all frames",
                    "parity": parity,
                    "codec": "ffv1 level 3 slicecrc",
                    "pixel_format": "yuv420p",
                    "colour": "BT.709 limited-range",
                },
                "command": command[:-1] + [str(output)],
                "output": str(output.resolve()),
                "output_sha256": sha256_file(output),
                "verification": verification,
            }
        )
    return {
        "schema": PROVENANCE_SCHEMA,
        "created_at": _utc_now(),
        "tools": {
            "ffmpeg": _tool_version(ffmpeg),
            "ffprobe": _tool_version(ffprobe),
        },
        "scan_report": str(scan_report_path.resolve()),
        "scan_report_sha256": sha256_file(scan_report_path),
        "scene_list": str(scenes_path.resolve()),
        "scene_list_sha256": sha256_file(scenes_path),
        "clips": results,
    }


def _validate_progressive_scan(scan: dict[str, Any], source: Path) -> Fraction:
    aggregate = scan.get("idet_aggregate") or {}
    if aggregate.get("classification") != "progressive":
        raise ValueError(f"source is not classified as progressive: {source}")
    probe = scan.get("probe") or {}
    rate = _progressive_source_rate(probe)

    # Missing colour tags are common in consumer H.264 files and are recorded
    # as an explicit metadata normalization in provenance.  Explicitly
    # contradictory tags are rejected because merely relabelling them as 709
    # limited would change the interpreted picture rather than prepare it.
    unknown = {None, "", "unknown", "unspecified", "N/A"}
    for key in ("color_space", "color_transfer", "color_primaries"):
        value = probe.get(key)
        if value not in unknown and value != "bt709":
            raise ValueError(f"{source} has incompatible {key}={value!r}")
    color_range = probe.get("color_range")
    if color_range not in unknown and color_range != "tv":
        raise ValueError(f"{source} is not limited-range: {color_range!r}")
    return rate


def prepare_progressive_scenes(
    ffmpeg: str,
    ffprobe: str,
    *,
    scan_report_path: Path,
    scenes_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    report = _read_json(scan_report_path)
    if report.get("schema") != SCAN_SCHEMA:
        raise ValueError(f"unsupported scan report schema: {report.get('schema')}")
    scans = {
        Path(record["source"]).resolve(): record for record in report.get("sources", [])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for record in _read_jsonl(scenes_path):
        if "source" not in record:
            raise ValueError("scene is missing source")
        unresolved_source = Path(str(record["source"])).expanduser()
        if not unresolved_source.is_absolute():
            unresolved_source = scenes_path.parent / unresolved_source
        source = unresolved_source.resolve()
        scan = scans.get(source)
        if scan is None:
            raise ValueError(f"source is absent from scan report: {source}")
        source_rate = _validate_progressive_scan(scan, source)
        current_fingerprint = sampled_file_fingerprint(source)
        if current_fingerprint != scan.get("source_fingerprint"):
            raise RuntimeError(f"source changed after cadence scan: {source}")

        source, parsed_rate, start_seconds, output_count, name = (
            _progressive_scene_values(
                record,
                root=scenes_path.parent,
                scan=scan,
            )
        )
        if parsed_rate != source_rate:
            raise AssertionError("progressive source rate changed while resolving scene")
        source_probe = scan.get("probe") or {}
        width = int(source_probe["width"])
        height = int(source_probe["height"])

        output = output_dir / f"{name}--progressive-2997.mkv"
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        temporary = output_dir / f".{output.stem}.{uuid.uuid4().hex}.part.mkv"
        command = build_progressive_command(
            ffmpeg,
            source=source,
            output=temporary,
            start_seconds=start_seconds,
            output_frame_count=output_count,
            source_rate=source_rate,
        )
        try:
            _run(command)
            verification = verify_progressive_clip(
                ffmpeg,
                ffprobe,
                temporary,
                expected_frames=output_count,
                expected_width=width,
                expected_height=height,
            )
            os.replace(temporary, output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        source_frame_position = start_seconds * source_rate
        results.append(
            {
                "name": name,
                "source": str(source),
                "source_fingerprint": current_fingerprint,
                "source_scan_classification": scan["idet_aggregate"],
                "source_rate": str(source_rate),
                "start_seconds": str(start_seconds),
                "start_source_frame": (
                    int(source_frame_position)
                    if source_frame_position.denominator == 1
                    else None
                ),
                "output_frame_count": output_count,
                "working_rate": str(WORKING_RATE),
                "operation": {
                    "filter": (
                        "setpts, fps=30000/1001 start_time=0 round=near, "
                        "setfield progressive"
                    ),
                    "codec": "ffv1 level 3 slicecrc",
                    "pixel_format": "yuv420p",
                    "spatial_resize": False,
                    "colour": "BT.709 limited-range metadata normalization",
                },
                "command": command[:-1] + [str(output)],
                "output": str(output.resolve()),
                "output_sha256": sha256_file(output),
                "verification": verification,
            }
        )
    return {
        "schema": PROGRESSIVE_PROVENANCE_SCHEMA,
        "created_at": _utc_now(),
        "tools": {
            "ffmpeg": _tool_version(ffmpeg),
            "ffprobe": _tool_version(ffprobe),
        },
        "scan_report": str(scan_report_path.resolve()),
        "scan_report_sha256": sha256_file(scan_report_path),
        "scene_list": str(scenes_path.resolve()),
        "scene_list_sha256": sha256_file(scenes_path),
        "clips": results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="classify source cadence without rewriting it")
    scan.add_argument("inputs", type=Path, nargs="+")
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--sample-count", type=int, default=3)
    scan.add_argument("--sample-frames", type=int, default=300)

    prepare = subparsers.add_parser("prepare", help="deinterlace only selected scene ranges")
    prepare.add_argument("--scan-report", type=Path, required=True)
    prepare.add_argument("--scenes", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    progressive = subparsers.add_parser(
        "prepare-progressive",
        help="select progressive source ranges at an exact 30000/1001 output rate",
    )
    progressive.add_argument("--scan-report", type=Path, required=True)
    progressive.add_argument("--scenes", type=Path, required=True)
    progressive.add_argument("--output-dir", type=Path, required=True)
    progressive.add_argument("--manifest", type=Path, required=True)
    progressive.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ffmpeg = _require_program(args.ffmpeg)
    ffprobe = _require_program(args.ffprobe)
    if args.command == "scan":
        inputs = expand_inputs(args.inputs)
        report = build_scan_report(
            ffmpeg,
            ffprobe,
            inputs,
            sample_count=args.sample_count,
            sample_frames=args.sample_frames,
        )
        _atomic_json(args.output, report)
        for source in report["sources"]:
            aggregate = source["idet_aggregate"]
            print(
                f"{aggregate['classification']:>16s} "
                f"{str(aggregate['parity'] or '-'):>3s}  {source['source']}"
            )
        print(f"scan report: {args.output.resolve()}")
        return 0

    if args.command == "prepare-progressive":
        provenance = prepare_progressive_scenes(
            ffmpeg,
            ffprobe,
            scan_report_path=args.scan_report,
            scenes_path=args.scenes,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    else:
        provenance = prepare_scenes(
            ffmpeg,
            ffprobe,
            scan_report_path=args.scan_report,
            scenes_path=args.scenes,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    _atomic_json(args.manifest, provenance)
    print(f"prepared clips: {len(provenance['clips'])}")
    print(f"provenance manifest: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        raise
