# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lada.datasetcreation.gemma_representative_selector import (
    CLASSIFIER_VERSION,
    build_representative_prompt,
    collect_dataset_clips,
    materialize_representative_dataset,
    remove_legacy_flat_representative_links,
    select_representatives,
    split_representatives_by_source,
)
from lada.datasetcreation.gemma_video_selector import (
    DEFAULT_API_URL,
    GemmaClient,
    extract_frame,
    load_jsonl,
    make_contact_sheet,
    sample_timestamps,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Choose one strong representative from consecutive similar LADA dataset clips"
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="dataset containing crop_unscaled_img/mask/meta")
    parser.add_argument("--output", type=Path, required=True,
                        help="append-only per-clip Gemma classification JSONL")
    parser.add_argument("--summary", type=Path,
                        help="representative-group JSON (default: <output stem>-groups.json)")
    parser.add_argument("--selected-list", type=Path,
                        help="representative video list (default: <output stem>-selected.txt)")
    parser.add_argument("--selected-root", type=Path,
                        help="optional output dataset containing selected video/mask/metadata triplets")
    parser.add_argument("--selected-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--frames-per-clip", type=int, default=6)
    parser.add_argument("--frame-max-side", type=int, default=640)
    parser.add_argument("--contact-sheet-columns", type=int, default=3)
    parser.add_argument("--max-duration", type=float, default=8.25)
    parser.add_argument("--min-file-age", type=float, default=15,
                        help="skip files modified this recently so a running collector is safe")
    parser.add_argument("--max-index-gap", type=int, default=3,
                        help="largest clip-number gap still considered consecutive")
    parser.add_argument("--min-similarity", type=float, default=0.6,
                        help="weighted activity/orientation/pose/framing similarity")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--max-per-source", type=int, default=40,
                        help="maximum representatives from one original source video; 0 disables")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=0, help="0 means no limit")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.frames_per_clip <= 16:
        parser.error("--frames-per-clip must be between 1 and 16")
    if args.max_duration <= 0 or args.min_file_age < 0:
        parser.error("durations must be positive and file age cannot be negative")
    if args.max_index_gap < 1:
        parser.error("--max-index-gap must be at least 1")
    if args.max_per_source < 0:
        parser.error("--max-per-source cannot be negative")
    if not 0 <= args.min_similarity <= 1 or not 0 <= args.min_confidence <= 1:
        parser.error("similarity and confidence must be between 0 and 1")
    if args.train_ratio <= 0 or args.validation_ratio < 0 or args.train_ratio + args.validation_ratio > 1:
        parser.error("invalid train/validation/test ratios")
    if args.start_index < 0 or args.max_clips < 0:
        parser.error("indices and limits cannot be negative")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    clips, skipped = collect_dataset_clips(
        args.dataset_root, max_duration=args.max_duration, min_file_age=args.min_file_age
    )
    clips = clips[args.start_index:]
    if args.max_clips:
        clips = clips[:args.max_clips]
    if not clips:
        raise RuntimeError("no complete dataset clips are ready")

    previous = load_jsonl(args.output)
    previous_by_video = {
        record["video"]: record for record in previous
        if record.get("status") == "ok"
        and record.get("classifier_version") == CLASSIFIER_VERSION
    } if not args.no_resume else {}
    client = GemmaClient(args.api_url, args.model, args.timeout, args.retries)
    print(f"Gemma model: {client.resolve_model()}")
    print(f"Ready clips: {len(clips)}; incomplete/rejected metadata: {len(skipped)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    classifications: dict[str, dict] = {
        video: record["decision"] for video, record in previous_by_video.items()
    }

    with args.output.open("a", encoding="utf-8") as manifest:
        for position, clip in enumerate(clips, start=args.start_index):
            video_key = str(clip.video_path)
            if video_key in previous_by_video:
                print(f"[{position}] skip {clip.video_path.name}")
                continue
            print(f"[{position}] classify {clip.source_name} / #{clip.clip_index:06d}")
            try:
                timestamps = sample_timestamps(clip.analysis_window, args.frames_per_clip)
                frames = [
                    extract_frame(clip.video_path, timestamp, args.ffmpeg, args.frame_max_side)
                    for timestamp in timestamps
                ]
                sheet = make_contact_sheet(frames, timestamps, args.contact_sheet_columns)
                decision = client.classify(
                    sheet, build_representative_prompt(clip, len(timestamps))
                )
                classifications[video_key] = decision
                record = {
                    "status": "ok",
                    "classifier_version": CLASSIFIER_VERSION,
                    "video": video_key,
                    "metadata": str(clip.metadata_path),
                    "source_name": clip.source_name,
                    "clip_index": clip.clip_index,
                    "duration": round(clip.duration, 6),
                    "scene_shape": [clip.scene_height, clip.scene_width],
                    "sample_timestamps": [round(value, 6) for value in timestamps],
                    "decision": decision,
                }
                print(
                    f"  usable={decision.get('usable')} confidence={decision.get('confidence')} "
                    f"{decision.get('activity_key')} / {decision.get('body_orientation')} / "
                    f"{decision.get('pose')} / {decision.get('framing')}"
                )
            except Exception as error:
                record = {
                    "status": "error", "classifier_version": CLASSIFIER_VERSION,
                    "video": video_key, "metadata": str(clip.metadata_path),
                    "source_name": clip.source_name, "clip_index": clip.clip_index,
                    "error": str(error),
                }
                print(f"  ERROR: {error}", file=sys.stderr)
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

    classified = [
        (clip, classifications[str(clip.video_path)])
        for clip in clips if str(clip.video_path) in classifications
    ]
    groups, selected = select_representatives(
        classified,
        max_index_gap=args.max_index_gap,
        min_similarity=args.min_similarity,
        min_confidence=args.min_confidence,
        max_per_source=args.max_per_source,
    )
    splits = split_representatives_by_source(
        selected,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.split_seed,
    )
    summary_path = args.summary or args.output.with_name(f"{args.output.stem}-groups.json")
    selected_list = args.selected_list or args.output.with_name(f"{args.output.stem}-selected.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({
            "classifier_version": CLASSIFIER_VERSION,
            "dataset_root": str(args.dataset_root.resolve()),
            "ready_clips": len(clips),
            "classified_clips": len(classified),
            "selected_clips": len(selected),
            "max_per_source": args.max_per_source,
            "split_ratios": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": 1.0 - args.train_ratio - args.validation_ratio,
            },
            "splits": {
                name: {
                    "clips": len(values),
                    "sources": len({clip.source_name for clip in values}),
                }
                for name, values in splits.items()
            },
            "skipped_inputs": skipped,
            "groups": groups,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_list.parent.mkdir(parents=True, exist_ok=True)
    selected_list.write_text(
        "".join(f"{clip.video_path}\n" for clip in selected), encoding="utf-8"
    )
    for split_name, split_clips in splits.items():
        split_list = selected_list.with_name(
            f"{selected_list.stem}-{split_name}{selected_list.suffix or '.txt'}"
        )
        split_list.write_text(
            "".join(f"{clip.video_path}\n" for clip in split_clips), encoding="utf-8"
        )
    if args.selected_root:
        remove_legacy_flat_representative_links(args.selected_root)
        for split_name, split_clips in splits.items():
            materialize_representative_dataset(
                split_clips, args.selected_root / split_name, args.selected_mode
            )
    print(f"Classified: {len(classified)}; groups: {len(groups)}; representatives: {len(selected)}")
    print(f"Summary: {summary_path}")
    print(f"Selected list: {selected_list}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
