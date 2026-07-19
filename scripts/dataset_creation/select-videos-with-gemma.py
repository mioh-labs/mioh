# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lada.datasetcreation.gemma_video_selector import (
    DEFAULT_API_URL,
    GemmaClient,
    build_analysis_windows,
    build_prompt,
    collect_video_files,
    criteria_digest,
    extract_frame,
    load_jsonl,
    make_contact_sheet,
    materialize_selected_files,
    probe_video,
    record_key,
    sample_timestamps,
    write_selected_list,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select local videos by showing representative contact sheets to local Gemma 4"
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="one or more video files/directories (searched recursively)")
    criteria = parser.add_mutually_exclusive_group(required=True)
    criteria.add_argument("--criteria", help="natural-language selection criteria")
    criteria.add_argument("--criteria-file", type=Path, help="UTF-8 text file containing selection criteria")
    parser.add_argument("--output", type=Path, required=True, help="append-only JSONL result manifest")
    parser.add_argument("--selected-list", type=Path,
                        help="selected video paths (default: <output stem>-selected.txt)")
    parser.add_argument("--selected-dir", type=Path,
                        help="optionally materialize selected videos in this directory")
    parser.add_argument("--selected-mode", choices=("symlink", "hardlink", "copy"), default="symlink")

    sampling = parser.add_argument_group("Sampling")
    sampling.add_argument("--window-seconds", type=float, default=0,
                          help="seconds per classified window; 0 samples across the whole file")
    sampling.add_argument("--scan-interval", type=float, default=0,
                          help="seconds between window starts; 0 uses window-seconds")
    sampling.add_argument("--frames-per-window", type=int, default=8)
    sampling.add_argument("--frame-max-side", type=int, default=640)
    sampling.add_argument("--contact-sheet-columns", type=int, default=4)
    sampling.add_argument("--start-index", type=int, default=0)
    sampling.add_argument("--max-videos", type=int, default=0, help="0 means no limit")

    model = parser.add_argument_group("Gemma / llama.cpp")
    model.add_argument("--api-url", default=DEFAULT_API_URL)
    model.add_argument("--model", default="auto", help="auto uses the model loaded by llama-server")
    model.add_argument("--timeout", type=float, default=180)
    model.add_argument("--retries", type=int, default=2)
    model.add_argument("--min-confidence", type=float, default=0.6,
                       help="matches_criteria also needs at least this confidence")
    model.add_argument("--ffmpeg", default="ffmpeg")
    model.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--no-resume", action="store_true",
                        help="do not skip successful windows already present in the JSONL")
    args = parser.parse_args(argv)
    if args.window_seconds < 0 or args.scan_interval < 0:
        parser.error("window and scan durations cannot be negative")
    if args.frames_per_window < 1 or args.frames_per_window > 16:
        parser.error("--frames-per-window must be between 1 and 16")
    if args.frame_max_side < 128:
        parser.error("--frame-max-side must be at least 128")
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    if args.start_index < 0 or args.max_videos < 0:
        parser.error("indices and limits cannot be negative")
    return args


def read_criteria(args) -> str:
    value = args.criteria if args.criteria is not None else args.criteria_file.read_text(encoding="utf-8")
    value = value.strip()
    if not value:
        raise ValueError("selection criteria cannot be empty")
    return value


def main(argv=None) -> int:
    args = parse_args(argv)
    criteria = read_criteria(args)
    criteria_id = criteria_digest(criteria)
    videos = collect_video_files(args.input)[args.start_index:]
    if args.max_videos:
        videos = videos[:args.max_videos]
    if not videos:
        raise FileNotFoundError("no supported videos found")

    previous = load_jsonl(args.output)
    completed = {
        record_key(record)
        for record in previous
        if record.get("status") == "ok"
    } if not args.no_resume else set()
    client = GemmaClient(args.api_url, args.model, args.timeout, args.retries)
    print(f"Gemma model: {client.resolve_model()}")
    print(f"Videos: {len(videos)} / criteria: {criteria_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with args.output.open("a", encoding="utf-8") as manifest:
        for video_index, video in enumerate(videos, start=args.start_index):
            print(f"[{video_index}] {video}")
            try:
                metadata = probe_video(video, args.ffprobe)
                windows = build_analysis_windows(
                    metadata["duration"], args.window_seconds, args.scan_interval
                )
            except Exception as error:
                record = {
                    "status": "error", "video": str(video), "criteria_id": criteria_id,
                    "error": str(error),
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                print(f"  ERROR: {error}", file=sys.stderr)
                continue

            for window in windows:
                key = (str(video), round(window.start, 3), round(window.end, 3), criteria_id)
                if key in completed:
                    print(f"  skip {window.start:.1f}-{window.end:.1f}s (already completed)")
                    continue
                try:
                    timestamps = sample_timestamps(window, args.frames_per_window)
                    frames = [
                        extract_frame(video, timestamp, args.ffmpeg, args.frame_max_side)
                        for timestamp in timestamps
                    ]
                    sheet = make_contact_sheet(frames, timestamps, args.contact_sheet_columns)
                    decision = client.classify(sheet, build_prompt(criteria, window))
                    confidence = max(0.0, min(1.0, float(decision.get("confidence", 0))))
                    model_match = decision.get("matches_criteria") is True
                    selected = model_match and confidence >= args.min_confidence
                    record = {
                        "status": "ok",
                        "video": str(video),
                        "video_index": video_index,
                        "criteria_id": criteria_id,
                        "window_start": round(window.start, 3),
                        "window_end": round(window.end, 3),
                        "sample_timestamps": [round(value, 3) for value in timestamps],
                        "metadata": metadata,
                        "model_match": model_match,
                        "selected": selected,
                        "decision": decision,
                    }
                    completed.add(key)
                    processed += 1
                    mark = "SELECT" if selected else "reject"
                    print(f"  {mark} {window.start:.1f}-{window.end:.1f}s confidence={confidence:.2f}")
                except Exception as error:
                    record = {
                        "status": "error", "video": str(video), "video_index": video_index,
                        "criteria_id": criteria_id, "window_start": round(window.start, 3),
                        "window_end": round(window.end, 3), "error": str(error),
                    }
                    print(f"  ERROR {window.start:.1f}-{window.end:.1f}s: {error}", file=sys.stderr)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()

    records = load_jsonl(args.output)
    selected_list = args.selected_list or args.output.with_name(
        f"{args.output.stem}-selected.txt"
    )
    selected = write_selected_list(
        (record for record in records if record.get("criteria_id") == criteria_id),
        selected_list,
    )
    if args.selected_dir:
        materialize_selected_files(selected, args.selected_dir, args.selected_mode)
    print(f"Completed windows: {processed}; selected videos: {len(selected)}")
    print(f"Manifest: {args.output}")
    print(f"Selected list: {selected_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
